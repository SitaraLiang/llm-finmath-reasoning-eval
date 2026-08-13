import argparse
import hashlib
import json
import re
import socket
import statistics
import sys
import time
from pathlib import Path
from string import Template
from urllib import error, request

import yaml

from eval_embeddings import (
    DEFAULT_MATRIX_THRESHOLD,
    DEFAULT_MODEL,
    EmbeddingModel,
    build_matrix_rows,
    flatten_atoms,
    field_coverage_score,
    get_subquestions,
    load_yaml,
    write_csv,
)


DEFAULT_PREDICTIONS = Path("outputs/parsed_results")
DEFAULT_GROUND_TRUTH = Path("data/ground_truth")
DEFAULT_OUTPUT = Path("outputs/evaluation")
DEFAULT_CONFIG = Path("config/evaluation/example.yaml")
ATOM_DIMENSIONS = {
    "D1": "outcomes",
    "D3": "arguments",
}
D4_SOURCES = (
    "global_assumption",
    "local_assumption",
    "gt_precondition",
    "previous_outcome",
)
D4_SOURCE_METRICS = (
    "mean_best_score",
    "recall",
    "precision",
    "f1",
)
JUDGE_LABEL_SCORES = {
    "equivalent": 1.0,
    "partially_correct": 0.5,
    "incorrect": 0.0,
    "unrelated": 0.0,
}
DEFAULT_JUDGE_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_JUDGE_PROMPT = """You are judging whether a model answer covers a ground-truth mathematical statement.

Dimension:
{dimension}

Field:
{field}

Ground-truth statement:
{gt_text}

Model statement:
{model_text}

Choose exactly one label:
- equivalent: the model statement fully expresses the same mathematical content.
- partially_correct: the model statement captures part of the content but misses or weakens something important.
- incorrect: the model statement is mathematically wrong or contradicts the ground truth.
- unrelated: the model statement is about a different mathematical fact.

Return only YAML, with exactly these keys:
label: equivalent|partially_correct|incorrect|unrelated
confidence: 0.0-1.0
reason: one short sentence
"""


def progress(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def dump_yaml(data) -> str:
    return yaml.dump(
        data,
        Dumper=yaml.Dumper,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        indent=4,
    )


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_config(config_path: Path) -> dict:
    """Load an optional YAML/JSON evaluation configuration."""
    if not config_path.exists():
        raise SystemExit(f"Error: Config file '{config_path}' does not exist.")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        loaded = json.loads(text)
    else:
        loaded = yaml.safe_load(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"Error: Config file '{config_path}' must contain a mapping.")
    return loaded


def get_nested(config: dict, keys: list[str], default=None):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def parse_prediction_path(path: Path, prediction_root: Path) -> dict:
    """Parse outputs/parsed_results/{model}/{lang}/{variation}/pcN_qM_type.yaml."""
    relative = path.relative_to(prediction_root)
    if len(relative.parts) < 4:
        raise ValueError(
            f"Expected path like {{call1_model}}/{{lang}}/{{variation}}/pcN_qM_type.yaml, got {relative}."
        )

    call1_model, language, variation = relative.parts[:3]
    stem = path.stem
    match = re.fullmatch(r"(pc\d+_q\d+)(?:_(.+))?", stem)
    if not match:
        raise ValueError(f"Could not parse exercise and strategy from filename '{path.name}'.")

    exercise = match.group(1)
    strategy = match.group(2) or ""
    return {
        "call1_model": call1_model,
        "language": language,
        "variation": variation,
        "exercise": exercise,
        "strategy": strategy,
    }


def find_prediction_files(prediction_root: Path) -> list[Path]:
    if not prediction_root.exists():
        raise SystemExit(f"Error: prediction directory does not exist: {prediction_root}")
    files = sorted(
        path
        for path in prediction_root.rglob("*.yaml")
        if path.is_file() and not path.name.startswith(".")
    )
    files.extend(
        sorted(
            path
            for path in prediction_root.rglob("*.yml")
            if path.is_file() and not path.name.startswith(".")
        )
    )
    if not files:
        raise SystemExit(f"Error: no YAML prediction files found under {prediction_root}")
    return files


def ground_truth_path_for(metadata: dict, ground_truth_root: Path) -> Path:
    return ground_truth_root / metadata["language"] / f"{metadata['exercise']}.yaml"


def matrix_output_dir(metadata: dict, output_root: Path) -> Path:
    return (
        output_root
        / "matrices"
        / metadata["call1_model"]
        / metadata["language"]
        / metadata["variation"]
        / f"{metadata['exercise']}_{metadata['strategy']}".rstrip("_")
    )


def subquestion_atom_records(data: dict, subquestion_index: int) -> list[dict]:
    subquestions = get_subquestions(data)
    if subquestion_index > len(subquestions):
        return []
    subquestion = subquestions[subquestion_index - 1]
    if not isinstance(subquestion, dict):
        return []
    return flatten_atoms(
        subquestion.get("atoms", []),
        f"subquestions[{subquestion_index}].atoms",
    )


def is_proof_atom(value) -> bool:
    return (
        isinstance(value, dict)
        and all(key in value for key in ("preconditions", "arguments", "outcomes"))
    )


def collect_ordered_atom_records(value, path: str, next_index: list[int]) -> tuple[list[dict], list[dict]]:
    """Collect atoms and order constraints induced by LIST/SET structure.

    YAML lists are ordered: every atom in an earlier child must appear before
    every atom in a later child. YAML tuples represent unordered mathematical
    sets, so no cross-child order constraints are added inside tuples.
    """
    if is_proof_atom(value):
        atom_index = next_index[0]
        next_index[0] += 1
        return [{"gt_atom_index": atom_index, "path": path, "atom": value}], []

    if not isinstance(value, (list, tuple)):
        return [], []

    kind = "set" if isinstance(value, tuple) else "list"
    records = []
    constraints = []
    child_groups = []
    for child_index, child in enumerate(value, start=1):
        child_records, child_constraints = collect_ordered_atom_records(
            child,
            f"{path}/{kind}[{child_index}]",
            next_index,
        )
        records.extend(child_records)
        constraints.extend(child_constraints)
        child_groups.append([record["gt_atom_index"] for record in child_records])

    if isinstance(value, list):
        for left_index, left_group in enumerate(child_groups):
            for right_group in child_groups[left_index + 1:]:
                for before_atom in left_group:
                    for after_atom in right_group:
                        constraints.append(
                            {
                                "before_gt_atom_index": before_atom,
                                "after_gt_atom_index": after_atom,
                            }
                        )

    return records, constraints


def subquestion_order_constraints(data: dict, subquestion_index: int) -> tuple[list[dict], list[dict]]:
    subquestions = get_subquestions(data)
    if subquestion_index > len(subquestions):
        return [], []
    subquestion = subquestions[subquestion_index - 1]
    if not isinstance(subquestion, dict):
        return [], []
    return collect_ordered_atom_records(
        subquestion.get("atoms", []),
        f"subquestions[{subquestion_index}].atoms",
        [1],
    )


def normalize_statement_list(values) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]
    return [str(value).strip() for value in values if str(value).strip()]


def append_unique_statement(targets: list[dict], seen: set[str], source: str, path: str, text: str) -> None:
    normalized = text.strip()
    if not normalized or normalized in seen:
        return
    seen.add(normalized)
    targets.append(
        {
            "source": source,
            "path": path,
            "text": normalized,
        }
    )


class LLMJudge:
    """Optional second-stage verifier for ambiguous embedding matches.

    The judge is intended for trusted local Ollama models. It is only called
    when an embedding score falls between low/high thresholds, and all decisions
    are cached so repeated evaluations remain cheap and reproducible.
    """

    def __init__(
        self,
        model: str,
        endpoint: str,
        timeout: int,
        cache_path: Path,
        temperature: float,
        prompt_template: str,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.cache_path = cache_path
        self.temperature = temperature
        self.prompt_template = prompt_template
        self.cache = self._load_cache(cache_path)
        self.calls = 0
        self.cache_hits = 0
        self.failures = 0

    @staticmethod
    def _load_cache(path: Path) -> dict:
        if not path.exists():
            return {}
        loaded = load_yaml(path)
        if isinstance(loaded, dict):
            return loaded
        return {}

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(dump_yaml(self.cache), encoding="utf-8")

    def key(self, dimension: str, field: str, gt_text: str, model_text: str) -> str:
        payload = {
            "judge_model": self.model,
            "dimension": dimension,
            "field": field,
            "gt_text": gt_text,
            "model_text": model_text,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def prompt(self, dimension: str, field: str, gt_text: str, model_text: str) -> str:
        values = {
            "dimension": dimension,
            "field": field,
            "gt_text": gt_text,
            "model_text": model_text,
        }
        return Template(self.prompt_template).safe_substitute(values).strip()

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 180,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama judge timed out for model '{self.model}' after {self.timeout}s."
            ) from exc
        except socket.timeout as exc:
            raise RuntimeError(
                f"Ollama judge timed out for model '{self.model}' after {self.timeout}s."
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"Ollama judge request failed for model '{self.model}': {exc}") from exc
        return result.get("response", "")

    @staticmethod
    def parse_response(raw_response: str) -> dict:
        text = raw_response.strip()
        fenced = re.search(r"(?ms)```(?:yaml|yml)?\s*(.*?)\s*```", text)
        if fenced:
            text = fenced.group(1).strip()
        parsed = yaml.safe_load(text)
        if not isinstance(parsed, dict):
            raise ValueError("judge response is not a YAML mapping")
        label = str(parsed.get("label", "")).strip()
        if label not in JUDGE_LABEL_SCORES:
            raise ValueError(f"judge returned invalid label '{label}'")
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("judge confidence must be numeric") from exc
        confidence = max(0.0, min(1.0, confidence))
        reason = str(parsed.get("reason", "")).strip()
        return {
            "label": label,
            "confidence": confidence,
            "reason": reason,
            "raw_response": raw_response,
        }

    def judge(self, dimension: str, field: str, gt_text: str, model_text: str) -> dict:
        key = self.key(dimension, field, gt_text, model_text)
        cached = self.cache.get(key)
        if isinstance(cached, dict):
            self.cache_hits += 1
            return {**cached, "cache_key": key, "cache_hit": True}

        self.calls += 1
        try:
            raw_response = self.generate(self.prompt(dimension, field, gt_text, model_text))
            decision = self.parse_response(raw_response)
        except Exception as exc:
            self.failures += 1
            decision = {
                "label": "judge_error",
                "confidence": 0.0,
                "reason": str(exc),
                "raw_response": "",
            }
        self.cache[key] = decision
        return {**decision, "cache_key": key, "cache_hit": False}


def apply_judge_to_rows(
    rows: list[dict],
    judge: LLMJudge | None,
    low_threshold: float,
    high_threshold: float,
) -> None:
    """Apply optional judge decisions to ambiguous row matches in-place."""
    for row in rows:
        score = float(row["score"])
        row["embedding_score"] = f"{score:.6f}"
        row["judge_used"] = False
        row["judge_label"] = ""
        row["judge_confidence"] = ""
        row["judge_reason"] = ""
        row["judge_cache_key"] = ""
        row["judge_cache_hit"] = ""
        row["judge_error"] = ""

        if not judge or not row.get("model_atom_index"):
            continue
        if score < low_threshold or score > high_threshold:
            continue

        decision = judge.judge(
            row["dimension"],
            row["field"],
            row["gt_text"],
            row["model_text"],
        )
        row["judge_used"] = True
        row["judge_label"] = decision["label"]
        row["judge_confidence"] = f"{float(decision.get('confidence', 0.0)):.6f}"
        row["judge_reason"] = decision.get("reason", "")
        row["judge_cache_key"] = decision.get("cache_key", "")
        row["judge_cache_hit"] = decision.get("cache_hit", "")

        if decision["label"] == "judge_error":
            row["judge_error"] = decision.get("reason", "")
            continue

        judged_score = JUDGE_LABEL_SCORES[decision["label"]]
        row["score"] = f"{judged_score:.6f}"
        row["matched"] = decision["label"] in {"equivalent", "partially_correct"}


def collect_gt_outcomes_before_subquestion(gt_data: dict, subquestion_index: int) -> list[dict]:
    outcomes = []
    seen = set()
    for previous_index in range(1, subquestion_index):
        for atom_index, atom_record in enumerate(subquestion_atom_records(gt_data, previous_index), start=1):
            for outcome_index, outcome in enumerate(
                normalize_statement_list(atom_record["atom"].get("outcomes", [])),
                start=1,
            ):
                append_unique_statement(
                    outcomes,
                    seen,
                    "previous_outcome",
                    f"{atom_record['path']}.outcomes[{outcome_index}]",
                    outcome,
                )
    return outcomes


def collect_d4_targets(gt_data: dict, subquestion_index: int) -> list[dict]:
    """Collect the GT statements that prediction preconditions should cover.

    D4 is defined as coverage of required hypotheses: global assumptions,
    current subquestion assumptions, current GT atom preconditions, and outcomes
    established by preceding subquestions.
    """
    targets = []
    seen = set()

    for index, assumption in enumerate(
        normalize_statement_list(gt_data.get("assumption_global", [])),
        start=1,
    ):
        append_unique_statement(
            targets,
            seen,
            "global_assumption",
            f"assumption_global[{index}]",
            assumption,
        )

    subquestions = get_subquestions(gt_data)
    if subquestion_index <= len(subquestions) and isinstance(subquestions[subquestion_index - 1], dict):
        subquestion = subquestions[subquestion_index - 1]
        for index, assumption in enumerate(
            normalize_statement_list(subquestion.get("assumptions", [])),
            start=1,
        ):
            append_unique_statement(
                targets,
                seen,
                "local_assumption",
                f"subquestions[{subquestion_index}].assumptions[{index}]",
                assumption,
            )

    for previous_outcome in collect_gt_outcomes_before_subquestion(gt_data, subquestion_index):
        append_unique_statement(
            targets,
            seen,
            previous_outcome["source"],
            previous_outcome["path"],
            previous_outcome["text"],
        )

    for atom_record in subquestion_atom_records(gt_data, subquestion_index):
        for index, precondition in enumerate(
            normalize_statement_list(atom_record["atom"].get("preconditions", [])),
            start=1,
        ):
            append_unique_statement(
                targets,
                seen,
                "gt_precondition",
                f"{atom_record['path']}.preconditions[{index}]",
                precondition,
            )

    return targets


def build_d4_rows(
    targets: list[dict],
    pred_atoms: list[dict],
    embedder: EmbeddingModel,
    threshold: float,
) -> list[dict]:
    rows = []
    for target_index, target in enumerate(targets, start=1):
        if not pred_atoms:
            rows.append(
                {
                    "dimension": "D4",
                    "field": "required_precondition_coverage",
                    "gt_atom_index": target_index,
                    "model_atom_index": "",
                    "gt_atom_path": target["path"],
                    "model_atom_path": "",
                    "gt_source": target["source"],
                    "score": "0.000000",
                    "matched": False,
                    "gt_text": target["text"],
                    "model_text": "",
                }
            )
            continue
        for pred_index, pred_record in enumerate(pred_atoms, start=1):
            pred_values = normalize_statement_list(pred_record["atom"].get("preconditions", []))
            score = field_coverage_score([target["text"]], pred_values, embedder)
            rows.append(
                {
                    "dimension": "D4",
                    "field": "required_precondition_coverage",
                    "gt_atom_index": target_index,
                    "model_atom_index": pred_index,
                    "gt_atom_path": target["path"],
                    "model_atom_path": pred_record["path"],
                    "gt_source": target["source"],
                    "score": f"{score:.6f}",
                    "matched": score >= threshold,
                    "gt_text": target["text"],
                    "model_text": " | ".join(pred_values),
                }
            )
    return rows


def summarize_dimension(rows: list[dict]) -> dict:
    if not rows:
        return {
            "mean_best_score": 0.0,
            "coverage": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "gt_items": 0,
            "matched_gt_items": 0,
            "prediction_items": 0,
            "matched_prediction_items": 0,
        }

    by_gt_atom = {}
    by_prediction_atom = {}
    for row in rows:
        gt_key = (row["subquestion_index"], row["gt_atom_index"])
        by_gt_atom.setdefault(gt_key, []).append(row)
        if row.get("model_atom_index"):
            prediction_key = (row["subquestion_index"], row["model_atom_index"])
            by_prediction_atom.setdefault(prediction_key, []).append(row)

    best_scores = []
    matched_gt_items = 0
    for candidates in by_gt_atom.values():
        best = max(candidates, key=lambda item: float(item["score"]))
        score = float(best["score"])
        best_scores.append(score)
        if best["matched"]:
            matched_gt_items += 1

    matched_prediction_items = 0
    for candidates in by_prediction_atom.values():
        if any(row["matched"] for row in candidates):
            matched_prediction_items += 1

    gt_items = len(best_scores)
    prediction_items = len(by_prediction_atom)
    recall = matched_gt_items / gt_items if gt_items else 0.0
    precision = matched_prediction_items / prediction_items if prediction_items else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "mean_best_score": statistics.fmean(best_scores) if best_scores else 0.0,
        "coverage": recall,
        "precision": precision,
        "f1": f1,
        "gt_items": gt_items,
        "matched_gt_items": matched_gt_items,
        "prediction_items": prediction_items,
        "matched_prediction_items": matched_prediction_items,
    }


def summarize_d4_sources(rows: list[dict]) -> dict[str, dict]:
    summaries = {}
    for row in rows:
        source = row.get("gt_source") or "unknown"
        summaries.setdefault(source, []).append(row)
    return {
        source: summarize_dimension(source_rows)
        for source, source_rows in sorted(summaries.items())
    }


def d4_source_summary_rows(base_metadata: dict, d4_source_summaries: dict[str, dict]) -> list[dict]:
    rows = []
    for source, summary in d4_source_summaries.items():
        rows.append(
            {
                **base_metadata,
                "gt_source": source,
                "D4_source_mean_best_score": f"{summary['mean_best_score']:.6f}",
                "D4_source_recall": f"{summary['coverage']:.6f}",
                "D4_source_precision": f"{summary['precision']:.6f}",
                "D4_source_f1": f"{summary['f1']:.6f}",
                "D4_source_gt_items": summary["gt_items"],
                "D4_source_matched_gt_items": summary["matched_gt_items"],
                "D4_source_prediction_items": summary["prediction_items"],
                "D4_source_matched_prediction_items": summary["matched_prediction_items"],
            }
        )
    return rows


def atom_pair_scores(rows: list[dict], subquestion_index: int) -> dict[tuple[int, int], dict[str, float]]:
    pair_scores = {}
    for row in rows:
        if row["dimension"] not in {"D1", "D3"}:
            continue
        if row["subquestion_index"] != subquestion_index:
            continue
        if not row.get("model_atom_index"):
            continue
        key = (int(row["gt_atom_index"]), int(row["model_atom_index"]))
        pair_scores.setdefault(key, {})[row["dimension"]] = float(row["score"])
    return pair_scores


def best_atom_matches_for_d2(
    rows: list[dict],
    subquestion_index: int,
    threshold: float,
) -> dict[int, dict]:
    pair_scores = atom_pair_scores(rows, subquestion_index)
    by_gt_atom = {}
    for (gt_atom_index, model_atom_index), scores in pair_scores.items():
        score_values = [scores[dimension] for dimension in ("D1", "D3") if dimension in scores]
        if not score_values:
            continue
        combined_score = statistics.fmean(score_values)
        candidate = {
            "gt_atom_index": gt_atom_index,
            "model_atom_index": model_atom_index,
            "combined_score": combined_score,
            "matched": combined_score >= threshold,
        }
        current = by_gt_atom.get(gt_atom_index)
        if current is None or candidate["combined_score"] > current["combined_score"]:
            by_gt_atom[gt_atom_index] = candidate
    return by_gt_atom


def build_d2_rows(
    gt_data: dict,
    all_rows: list[dict],
    subquestion_count: int,
    threshold: float,
) -> tuple[list[dict], dict]:
    d2_rows = []
    for sub_index in range(1, subquestion_count + 1):
        gt_atoms, constraints = subquestion_order_constraints(gt_data, sub_index)
        if not gt_atoms:
            continue
        best_matches = best_atom_matches_for_d2(all_rows, sub_index, threshold)
        path_by_atom = {
            record["gt_atom_index"]: record["path"]
            for record in gt_atoms
        }
        for constraint_index, constraint in enumerate(constraints, start=1):
            before_index = constraint["before_gt_atom_index"]
            after_index = constraint["after_gt_atom_index"]
            before_match = best_matches.get(before_index)
            after_match = best_matches.get(after_index)
            before_model_index = (
                before_match["model_atom_index"]
                if before_match and before_match["matched"]
                else ""
            )
            after_model_index = (
                after_match["model_atom_index"]
                if after_match and after_match["matched"]
                else ""
            )
            if before_model_index == "" or after_model_index == "":
                status = "unmatched"
                respected = False
            elif before_model_index <= after_model_index:
                status = "respected"
                respected = True
            else:
                status = "violated"
                respected = False
            d2_rows.append(
                {
                    "subquestion_index": sub_index,
                    "constraint_index": constraint_index,
                    "before_gt_atom_index": before_index,
                    "after_gt_atom_index": after_index,
                    "before_gt_atom_path": path_by_atom.get(before_index, ""),
                    "after_gt_atom_path": path_by_atom.get(after_index, ""),
                    "before_model_atom_index": before_model_index,
                    "after_model_atom_index": after_model_index,
                    "before_match_score": (
                        f"{before_match['combined_score']:.6f}" if before_match else ""
                    ),
                    "after_match_score": (
                        f"{after_match['combined_score']:.6f}" if after_match else ""
                    ),
                    "status": status,
                    "respected": respected,
                }
            )

    total_constraints = len(d2_rows)
    respected_constraints = sum(row["status"] == "respected" for row in d2_rows)
    violated_constraints = sum(row["status"] == "violated" for row in d2_rows)
    unmatched_constraints = sum(row["status"] == "unmatched" for row in d2_rows)
    matched_constraints = respected_constraints + violated_constraints
    if total_constraints:
        order_score = (
            respected_constraints / matched_constraints
            if matched_constraints
            else 0.0
        )
        coverage = matched_constraints / total_constraints
        strict_score = respected_constraints / total_constraints
    else:
        order_score = 1.0
        coverage = 1.0
        strict_score = 1.0
    summary = {
        "D2_order_score": order_score,
        "D2_order_coverage": coverage,
        "D2_strict_order_score": strict_score,
        "D2_total_constraints": total_constraints,
        "D2_respected_constraints": respected_constraints,
        "D2_violated_constraints": violated_constraints,
        "D2_unmatched_constraints": unmatched_constraints,
    }
    return d2_rows, summary


def evaluate_one(
    prediction_path: Path,
    prediction_root: Path,
    ground_truth_root: Path,
    output_root: Path,
    embedder: EmbeddingModel,
    embedding_model_name: str,
    threshold: float,
    judge: LLMJudge | None,
    judge_low_threshold: float,
    judge_high_threshold: float,
) -> dict:
    metadata = parse_prediction_path(prediction_path, prediction_root)
    gt_path = ground_truth_path_for(metadata, ground_truth_root)
    out_dir = matrix_output_dir(metadata, output_root)

    base_row = {
        **metadata,
        "prediction_path": str(prediction_path),
        "ground_truth_path": str(gt_path),
        "output_dir": str(out_dir),
        "threshold": f"{threshold:.6f}",
        "embedding_model": embedding_model_name,
        "judge_model": judge.model if judge else "",
        "judge_low_threshold": f"{judge_low_threshold:.6f}" if judge else "",
        "judge_high_threshold": f"{judge_high_threshold:.6f}" if judge else "",
    }

    if not gt_path.exists():
        return {**base_row, "status": "failed", "error": f"Missing ground truth file: {gt_path}"}

    try:
        gt_data = load_yaml(gt_path)
        pred_data = load_yaml(prediction_path)
        gt_subquestions = get_subquestions(gt_data)
        pred_subquestions = get_subquestions(pred_data)
    except Exception as exc:
        return {**base_row, "status": "failed", "error": str(exc)}

    all_rows = []
    best_rows = []
    d4_source_rows = []
    subquestion_count = max(len(gt_subquestions), len(pred_subquestions))

    for sub_index in range(1, subquestion_count + 1):
        gt_atoms = subquestion_atom_records(gt_data, sub_index)
        pred_atoms = subquestion_atom_records(pred_data, sub_index)

        for dimension, field in ATOM_DIMENSIONS.items():
            rows = build_matrix_rows(gt_atoms, pred_atoms, dimension, field, embedder, threshold)
            for row in rows:
                row["subquestion_index"] = sub_index
                row["gt_source"] = "gt_atom"
            apply_judge_to_rows(rows, judge, judge_low_threshold, judge_high_threshold)
            all_rows.extend(rows)

            for gt_atom_index in range(1, len(gt_atoms) + 1):
                candidates = [row for row in rows if row["gt_atom_index"] == gt_atom_index]
                if candidates:
                    best = max(candidates, key=lambda row: float(row["score"]))
                    best_rows.append(
                        {
                            "subquestion_index": sub_index,
                            "dimension": dimension,
                            "field": field,
                            "gt_atom_index": gt_atom_index,
                            "best_model_atom_index": best["model_atom_index"],
                            "best_score": best["score"],
                            "matched": best["matched"],
                        }
                    )
                else:
                    best_rows.append(
                        {
                            "subquestion_index": sub_index,
                            "dimension": dimension,
                            "field": field,
                            "gt_atom_index": gt_atom_index,
                            "best_model_atom_index": "",
                            "best_score": "0.000000",
                            "matched": False,
                        }
                    )

        d4_targets = collect_d4_targets(gt_data, sub_index)
        d4_rows = build_d4_rows(d4_targets, pred_atoms, embedder, threshold)
        for row in d4_rows:
            row["subquestion_index"] = sub_index
        apply_judge_to_rows(d4_rows, judge, judge_low_threshold, judge_high_threshold)
        all_rows.extend(d4_rows)

        for target_index in range(1, len(d4_targets) + 1):
            candidates = [row for row in d4_rows if row["gt_atom_index"] == target_index]
            if candidates:
                best = max(candidates, key=lambda row: float(row["score"]))
                best_rows.append(
                    {
                        "subquestion_index": sub_index,
                        "dimension": "D4",
                        "field": "required_precondition_coverage",
                        "gt_atom_index": target_index,
                        "gt_source": best["gt_source"],
                        "best_model_atom_index": best["model_atom_index"],
                        "best_score": best["score"],
                        "matched": best["matched"],
                    }
                )
            else:
                target = d4_targets[target_index - 1]
                best_rows.append(
                    {
                        "subquestion_index": sub_index,
                        "dimension": "D4",
                        "field": "required_precondition_coverage",
                        "gt_atom_index": target_index,
                        "gt_source": target["source"],
                        "best_model_atom_index": "",
                        "best_score": "0.000000",
                        "matched": False,
                    }
                )

    write_csv(
        out_dir / "atom_alignment_matrix.csv",
        all_rows,
        [
            "subquestion_index",
            "dimension",
            "field",
            "gt_atom_index",
            "model_atom_index",
            "gt_atom_path",
            "model_atom_path",
            "gt_source",
            "embedding_score",
            "score",
            "matched",
            "judge_used",
            "judge_label",
            "judge_confidence",
            "judge_reason",
            "judge_cache_key",
            "judge_cache_hit",
            "judge_error",
            "gt_text",
            "model_text",
        ],
    )
    write_csv(
        out_dir / "atom_alignment_best_matches.csv",
        best_rows,
        [
            "subquestion_index",
            "dimension",
            "field",
            "gt_atom_index",
            "gt_source",
            "best_model_atom_index",
            "best_score",
            "matched",
        ],
    )
    d2_rows, d2_summary = build_d2_rows(gt_data, all_rows, subquestion_count, threshold)
    write_csv(
        out_dir / "d2_order_constraints.csv",
        d2_rows,
        [
            "subquestion_index",
            "constraint_index",
            "before_gt_atom_index",
            "after_gt_atom_index",
            "before_gt_atom_path",
            "after_gt_atom_path",
            "before_model_atom_index",
            "after_model_atom_index",
            "before_match_score",
            "after_match_score",
            "status",
            "respected",
        ],
    )

    dimension_summaries = {
        dimension: summarize_dimension([row for row in all_rows if row["dimension"] == dimension])
        for dimension in ("D1", "D3", "D4")
    }
    d4_source_summaries = summarize_d4_sources([row for row in all_rows if row["dimension"] == "D4"])
    d4_source_rows.extend(
        d4_source_summary_rows(
            {
                **metadata,
                "prediction_path": str(prediction_path),
                "ground_truth_path": str(gt_path),
                "threshold": f"{threshold:.6f}",
            },
            d4_source_summaries,
        )
    )
    d1 = dimension_summaries["D1"]["mean_best_score"]
    d3 = dimension_summaries["D3"]["mean_best_score"]
    d4 = dimension_summaries["D4"]["mean_best_score"]
    overall = statistics.fmean([d1, d3, d4])
    overall_with_d2 = statistics.fmean([d1, d3, d4, d2_summary["D2_strict_order_score"]])

    return {
        **base_row,
        "status": "ok",
        "error": "",
        "ground_truth_subquestions": len(gt_subquestions),
        "prediction_subquestions": len(pred_subquestions),
        "D1_mean_best_score": f"{d1:.6f}",
        "D1_coverage": f"{dimension_summaries['D1']['coverage']:.6f}",
        "D1_precision": f"{dimension_summaries['D1']['precision']:.6f}",
        "D1_f1": f"{dimension_summaries['D1']['f1']:.6f}",
        "D3_mean_best_score": f"{d3:.6f}",
        "D3_coverage": f"{dimension_summaries['D3']['coverage']:.6f}",
        "D3_precision": f"{dimension_summaries['D3']['precision']:.6f}",
        "D3_f1": f"{dimension_summaries['D3']['f1']:.6f}",
        "D4_mean_best_score": f"{d4:.6f}",
        "D4_coverage": f"{dimension_summaries['D4']['coverage']:.6f}",
        "D4_precision": f"{dimension_summaries['D4']['precision']:.6f}",
        "D4_f1": f"{dimension_summaries['D4']['f1']:.6f}",
        "D2_order_score": f"{d2_summary['D2_order_score']:.6f}",
        "D2_order_coverage": f"{d2_summary['D2_order_coverage']:.6f}",
        "D2_strict_order_score": f"{d2_summary['D2_strict_order_score']:.6f}",
        "D2_total_constraints": d2_summary["D2_total_constraints"],
        "D2_respected_constraints": d2_summary["D2_respected_constraints"],
        "D2_violated_constraints": d2_summary["D2_violated_constraints"],
        "D2_unmatched_constraints": d2_summary["D2_unmatched_constraints"],
        "overall_mean_score": f"{overall:.6f}",
        "overall_with_D2_score": f"{overall_with_d2:.6f}",
        "_d4_source_rows": d4_source_rows,
    }


def aggregate_by(rows: list[dict], keys: list[str]) -> list[dict]:
    groups = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = tuple(row[item] for item in keys)
        groups.setdefault(key, []).append(row)

    aggregate_rows = []
    metrics = [
        "D1_mean_best_score",
        "D1_coverage",
        "D1_precision",
        "D1_f1",
        "D3_mean_best_score",
        "D3_coverage",
        "D3_precision",
        "D3_f1",
        "D4_mean_best_score",
        "D4_coverage",
        "D4_precision",
        "D4_f1",
        "D2_order_score",
        "D2_order_coverage",
        "D2_strict_order_score",
        "overall_mean_score",
        "overall_with_D2_score",
    ]
    for key, group_rows in sorted(groups.items()):
        row = {name: value for name, value in zip(keys, key)}
        row["cases"] = len(group_rows)
        for metric in metrics:
            row[metric] = f"{statistics.fmean(float(item[metric]) for item in group_rows):.6f}"
        aggregate_rows.append(row)
    return aggregate_rows


def d4_source_column(source: str, metric: str) -> str:
    return f"D4_{source}_{metric}"


def add_d4_source_columns(
    aggregate_rows: list[dict],
    d4_source_rows: list[dict],
    keys: list[str],
) -> None:
    grouped = {}
    for row in d4_source_rows:
        key = tuple(row[item] for item in keys)
        source = row["gt_source"]
        grouped.setdefault((key, source), []).append(row)

    for aggregate_row in aggregate_rows:
        key = tuple(aggregate_row[item] for item in keys)
        for source in D4_SOURCES:
            source_rows = grouped.get((key, source), [])
            for metric in D4_SOURCE_METRICS:
                column = d4_source_column(source, metric)
                source_metric = f"D4_source_{metric}"
                if source_rows:
                    aggregate_row[column] = (
                        f"{statistics.fmean(float(row[source_metric]) for row in source_rows):.6f}"
                    )
                else:
                    aggregate_row[column] = ""


def d4_source_columns() -> list[str]:
    return [
        d4_source_column(source, metric)
        for source in D4_SOURCES
        for metric in D4_SOURCE_METRICS
    ]


def cli_or_config(cli_value, config: dict, keys: list[str], default):
    if cli_value is not None:
        return cli_value
    return get_nested(config, keys, default)


def config_path_value(value) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate final parsed Call 1 responses against ground-truth proof YAML.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Optional Ollama model used as a second-stage judge for ambiguous matches.",
    )
    parser.add_argument(
        "--judge-endpoint",
        default=None,
        help=f"Ollama generate endpoint for the judge. Default: {DEFAULT_JUDGE_ENDPOINT}",
    )
    parser.add_argument(
        "--judge-timeout",
        type=int,
        default=None,
        help="Timeout in seconds for each LLM judge request.",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=None,
        help="Temperature for LLM judge calls.",
    )
    parser.add_argument(
        "--judge-low-threshold",
        type=float,
        default=None,
        help="Embedding scores below this value are rejected without LLM judge.",
    )
    parser.add_argument(
        "--judge-high-threshold",
        type=float,
        default=None,
        help="Embedding scores above this value are accepted without LLM judge.",
    )
    parser.add_argument(
        "--judge-cache",
        type=Path,
        default=None,
        help="YAML cache path for LLM judge decisions. Default: <output>/judge_cache.yaml.",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    project_root = args.project_root.resolve()

    config = load_config(resolve_path(args.config, project_root)) if args.config else {}
    prediction_path = config_path_value(
        cli_or_config(args.predictions, config, ["input", "predictions"], DEFAULT_PREDICTIONS)
    )
    ground_truth_path = config_path_value(
        cli_or_config(args.ground_truth, config, ["input", "ground_truth"], DEFAULT_GROUND_TRUTH)
    )
    output_path = config_path_value(
        cli_or_config(args.output, config, ["output", "root_directory"], DEFAULT_OUTPUT)
    )
    prediction_root = resolve_path(prediction_path, project_root)
    ground_truth_root = resolve_path(ground_truth_path, project_root)
    output_root = resolve_path(output_path, project_root)

    embedding_model = cli_or_config(args.model, config, ["embedding", "model"], DEFAULT_MODEL)
    threshold = float(cli_or_config(args.threshold, config, ["embedding", "threshold"], 0.415))

    judge_model = cli_or_config(args.judge_model, config, ["judge", "model"], "")
    judge_enabled = bool(get_nested(config, ["judge", "enabled"], False))
    if args.judge_model is not None:
        judge_enabled = True
    judge_endpoint = cli_or_config(
        args.judge_endpoint,
        config,
        ["judge", "endpoint"],
        DEFAULT_JUDGE_ENDPOINT,
    )
    judge_timeout = int(cli_or_config(args.judge_timeout, config, ["judge", "timeout_seconds"], 120))
    judge_temperature = float(
        cli_or_config(args.judge_temperature, config, ["judge", "temperature"], 0.0)
    )
    judge_low_threshold = float(
        cli_or_config(args.judge_low_threshold, config, ["judge", "low_threshold"], 0.375)
    )
    judge_high_threshold = float(
        cli_or_config(args.judge_high_threshold, config, ["judge", "high_threshold"], 0.415)
    )
    judge_prompt = get_nested(config, ["prompts", "judge"], DEFAULT_JUDGE_PROMPT)
    judge_cache_path = (
        resolve_path(args.judge_cache, project_root)
        if args.judge_cache
        else output_root / "judge_cache.yaml"
    )

    judge = None
    if judge_enabled:
        if not judge_model:
            raise SystemExit("Error: judge.enabled is true but no judge.model is configured.")
        if judge_low_threshold > judge_high_threshold:
            raise SystemExit("Error: --judge-low-threshold must be <= --judge-high-threshold.")
        judge = LLMJudge(
            judge_model,
            judge_endpoint,
            judge_timeout,
            judge_cache_path,
            judge_temperature,
            judge_prompt,
        )

    prediction_files = find_prediction_files(prediction_root)
    progress(
        "Evaluation starting: "
        f"{len(prediction_files)} prediction file(s), model={embedding_model}, "
        f"threshold={threshold:.6f}."
    )
    if judge:
        progress(
            "LLM judge enabled: "
            f"model={judge_model}, ambiguous band="
            f"[{judge_low_threshold:.3f}, {judge_high_threshold:.3f}], "
            f"cache={judge_cache_path}."
        )

    embedder = EmbeddingModel(embedding_model)
    summary_rows = []
    d4_source_rows = []
    for index, prediction_path in enumerate(prediction_files, start=1):
        try:
            metadata = parse_prediction_path(prediction_path, prediction_root)
            label = (
                f"{metadata['exercise']}_{metadata['strategy']} | "
                f"model={metadata['call1_model']} | lang={metadata['language']} | "
                f"variation={metadata['variation']}"
            )
        except ValueError:
            label = str(prediction_path)
        progress(f"[{index}/{len(prediction_files)}] {label}: started")
        row = evaluate_one(
            prediction_path,
            prediction_root,
            ground_truth_root,
            output_root,
            embedder,
            embedding_model,
            threshold,
            judge,
            judge_low_threshold,
            judge_high_threshold,
        )
        d4_source_rows.extend(row.pop("_d4_source_rows", []))
        summary_rows.append(row)
        progress(f"[{index}/{len(prediction_files)}] {label}: {row['status']}")

    summary_fields = [
        "status",
        "error",
        "exercise",
        "strategy",
        "call1_model",
        "language",
        "variation",
        "embedding_model",
        "threshold",
        "judge_model",
        "judge_low_threshold",
        "judge_high_threshold",
        "ground_truth_subquestions",
        "prediction_subquestions",
        "D1_mean_best_score",
        "D1_coverage",
        "D1_precision",
        "D1_f1",
        "D3_mean_best_score",
        "D3_coverage",
        "D3_precision",
        "D3_f1",
        "D4_mean_best_score",
        "D4_coverage",
        "D4_precision",
        "D4_f1",
        "D2_order_score",
        "D2_order_coverage",
        "D2_strict_order_score",
        "D2_total_constraints",
        "D2_respected_constraints",
        "D2_violated_constraints",
        "D2_unmatched_constraints",
        "overall_mean_score",
        "overall_with_D2_score",
        "prediction_path",
        "ground_truth_path",
        "output_dir",
    ]
    for row in summary_rows:
        for field in summary_fields:
            row.setdefault(field, "")
    write_csv(output_root / "evaluation_summary.csv", summary_rows, summary_fields)

    by_model = aggregate_by(summary_rows, ["call1_model"])
    add_d4_source_columns(by_model, d4_source_rows, ["call1_model"])
    write_csv(
        output_root / "evaluation_by_model.csv",
        by_model,
        [
            "call1_model",
            "cases",
            "D1_mean_best_score",
            "D1_coverage",
            "D1_precision",
            "D1_f1",
            "D3_mean_best_score",
            "D3_coverage",
            "D3_precision",
            "D3_f1",
            "D4_mean_best_score",
            "D4_coverage",
            "D4_precision",
            "D4_f1",
            "D2_order_score",
            "D2_order_coverage",
            "D2_strict_order_score",
            "overall_mean_score",
            "overall_with_D2_score",
            *d4_source_columns(),
        ],
    )

    by_model_strategy = aggregate_by(summary_rows, ["call1_model", "strategy"])
    add_d4_source_columns(by_model_strategy, d4_source_rows, ["call1_model", "strategy"])
    write_csv(
        output_root / "evaluation_by_model_strategy.csv",
        by_model_strategy,
        [
            "call1_model",
            "strategy",
            "cases",
            "D1_mean_best_score",
            "D1_coverage",
            "D1_precision",
            "D1_f1",
            "D3_mean_best_score",
            "D3_coverage",
            "D3_precision",
            "D3_f1",
            "D4_mean_best_score",
            "D4_coverage",
            "D4_precision",
            "D4_f1",
            "D2_order_score",
            "D2_order_coverage",
            "D2_strict_order_score",
            "overall_mean_score",
            "overall_with_D2_score",
            *d4_source_columns(),
        ],
    )

    ok_count = sum(row["status"] == "ok" for row in summary_rows)
    failed_count = len(summary_rows) - ok_count
    if judge:
        judge.save()
        progress(
            "LLM judge complete: "
            f"{judge.calls} new call(s), {judge.cache_hits} cache hit(s), "
            f"{judge.failures} failure(s). Cache: {judge.cache_path}"
        )
    progress(f"Evaluation complete: {ok_count} ok, {failed_count} failed.")
    print(f"Wrote summary: {output_root / 'evaluation_summary.csv'}")
    print(f"Wrote model aggregate: {output_root / 'evaluation_by_model.csv'}")
    print(f"Wrote model/strategy aggregate: {output_root / 'evaluation_by_model_strategy.csv'}")

    if failed_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
