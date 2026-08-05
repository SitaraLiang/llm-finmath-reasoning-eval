import argparse
import re
import statistics
import sys
import time
from pathlib import Path

from eval_embeddings import (
    DEFAULT_MATRIX_THRESHOLD,
    DEFAULT_SANITY_MODEL,
    EmbeddingModel,
    build_matrix_rows,
    flatten_atoms,
    field_coverage_score,
    get_subquestions,
    load_yaml,
    write_csv,
)


DEFAULT_PREDICTIONS = Path("outputs/parsed_responses")
DEFAULT_GROUND_TRUTH = Path("data/ground_truth")
DEFAULT_OUTPUT = Path("outputs/evaluation")
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


def progress(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def parse_prediction_path(path: Path, prediction_root: Path) -> dict:
    """Parse outputs/parsed_responses/{model}/{lang}/{variation}/pcN_qM_type.yaml."""
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
    threshold: float,
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
            "score",
            "matched",
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


def aggregate_by_d4_source(rows: list[dict], keys: list[str]) -> list[dict]:
    groups = {}
    for row in rows:
        key = tuple(row[item] for item in keys)
        groups.setdefault(key, []).append(row)

    aggregate_rows = []
    metrics = [
        "D4_source_mean_best_score",
        "D4_source_recall",
        "D4_source_precision",
        "D4_source_f1",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate final parsed Call 1 responses against ground-truth proof YAML.",
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_SANITY_MODEL)
    parser.add_argument("--threshold", type=float, default=0.415)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    prediction_root = resolve_path(args.predictions, project_root)
    ground_truth_root = resolve_path(args.ground_truth, project_root)
    output_root = resolve_path(args.output, project_root)

    prediction_files = find_prediction_files(prediction_root)
    progress(
        "Evaluation starting: "
        f"{len(prediction_files)} prediction file(s), model={args.model}, "
        f"threshold={args.threshold:.6f}."
    )

    embedder = EmbeddingModel(args.model)
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
            args.threshold,
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
        "threshold",
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

    d4_source_fields = [
        "exercise",
        "strategy",
        "call1_model",
        "language",
        "variation",
        "gt_source",
        "threshold",
        "D4_source_mean_best_score",
        "D4_source_recall",
        "D4_source_precision",
        "D4_source_f1",
        "D4_source_gt_items",
        "D4_source_matched_gt_items",
        "D4_source_prediction_items",
        "D4_source_matched_prediction_items",
        "prediction_path",
        "ground_truth_path",
    ]
    for row in d4_source_rows:
        for field in d4_source_fields:
            row.setdefault(field, "")
    write_csv(output_root / "evaluation_d4_by_source.csv", d4_source_rows, d4_source_fields)

    d4_by_model_source = aggregate_by_d4_source(d4_source_rows, ["call1_model", "gt_source"])
    write_csv(
        output_root / "evaluation_d4_by_model_source.csv",
        d4_by_model_source,
        [
            "call1_model",
            "gt_source",
            "cases",
            "D4_source_mean_best_score",
            "D4_source_recall",
            "D4_source_precision",
            "D4_source_f1",
        ],
    )

    d4_by_model_strategy_source = aggregate_by_d4_source(
        d4_source_rows,
        ["call1_model", "strategy", "gt_source"],
    )
    write_csv(
        output_root / "evaluation_d4_by_model_strategy_source.csv",
        d4_by_model_strategy_source,
        [
            "call1_model",
            "strategy",
            "gt_source",
            "cases",
            "D4_source_mean_best_score",
            "D4_source_recall",
            "D4_source_precision",
            "D4_source_f1",
        ],
    )

    ok_count = sum(row["status"] == "ok" for row in summary_rows)
    failed_count = len(summary_rows) - ok_count
    progress(f"Evaluation complete: {ok_count} ok, {failed_count} failed.")
    print(f"Wrote summary: {output_root / 'evaluation_summary.csv'}")
    print(f"Wrote model aggregate: {output_root / 'evaluation_by_model.csv'}")
    print(f"Wrote model/strategy aggregate: {output_root / 'evaluation_by_model_strategy.csv'}")
    print(f"Wrote D4 source breakdown: {output_root / 'evaluation_d4_by_source.csv'}")

    if failed_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
