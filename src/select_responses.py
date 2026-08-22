"""Select the most faithful valid Call 2 conversion for each Call 1 response.

Selection never reads ground truth. Candidates first pass the same structural
validator used by Call 2, then receive deterministic source-fidelity scores.
The selected YAML is copied unchanged; provenance and diagnostics are written
to a separate selection report.
"""

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import time

from conversion_validator import expected_subquestion_count, validate_converted_exercise


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "selection" / "experiment_v1.yaml"
PROMPT_ABBREVIATIONS = {
    "seq": "strictly_sequential",
    "acc": "prompt_accumulation",
    "gtf": "ground_truth_forcing",
    "self": "self_history",
}
DEFAULT_WEIGHTS = {
    "formula_fidelity": 0.30,
    "source_coverage": 0.30,
    "non_hallucination": 0.20,
    "atom_completeness": 0.10,
    "non_duplication": 0.10,
}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "if", "in", "is", "it", "of", "on", "or", "that", "the", "then", "this",
    "to", "we", "with", "un", "une", "et", "est", "sont", "au", "aux", "avec",
    "ce", "ces", "cette", "dans", "de", "des", "du", "en", "la", "le", "les",
    "par", "pour", "que", "qui", "sur",
}


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Error: Config file '{path}' does not exist.")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit("Error: selection requires PyYAML.") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise SystemExit(f"Error: Config file '{path}' must contain a mapping.")
    return data


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_nested(config: dict, keys: list[str], default=None):
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def parse_exercise_filename(path: Path) -> dict | None:
    match = re.fullmatch(r"pc(\d+)_q(\d+)_([A-Za-z0-9-]+)", path.stem)
    if not match:
        return None
    abbreviation = match.group(3)
    return {
        "pc": match.group(1),
        "exercise": match.group(2),
        "strategy_abbreviation": abbreviation,
        "strategy": PROMPT_ABBREVIATIONS.get(abbreviation, abbreviation),
    }


def parse_call1_path(path: Path, root: Path) -> dict | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if len(relative.parts) != 4:
        return None
    model, language, variation, _ = relative.parts
    exercise = parse_exercise_filename(path)
    if exercise is None:
        return None
    return {
        **exercise,
        "call1_model": model,
        "language": language,
        "variation": variation,
        "path": path,
    }


def parse_call2_path(path: Path, root: Path) -> dict | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if len(relative.parts) != 5:
        return None
    call1_model, call2_model, language, variation, _ = relative.parts
    exercise = parse_exercise_filename(path)
    if exercise is None:
        return None
    return {
        **exercise,
        "call1_model": call1_model,
        "call2_model": call2_model,
        "language": language,
        "variation": variation,
        "path": path,
    }


def response_key(metadata: dict) -> tuple[str, ...]:
    return (
        metadata["call1_model"],
        metadata["language"],
        metadata["variation"],
        metadata["pc"],
        metadata["exercise"],
        metadata["strategy_abbreviation"],
    )


def selected_output_path(root: Path, metadata: dict) -> Path:
    filename = (
        f"pc{metadata['pc']}_q{metadata['exercise']}_"
        f"{metadata['strategy_abbreviation']}.yaml"
    )
    return (
        root
        / metadata["call1_model"]
        / metadata["language"]
        / metadata["variation"]
        / filename
    )


def configured_filter(config: dict, name: str) -> set[str]:
    values = get_nested(config, ["filters", name], []) or []
    if not isinstance(values, list):
        raise SystemExit(f"Error: filters.{name} must be a list.")
    return {str(value) for value in values}


def discover_call1_responses(root: Path, config: dict) -> list[dict]:
    if not root.is_dir():
        raise SystemExit(f"Error: Call 1 input directory does not exist: {root}")
    filters = {
        "call1_model": configured_filter(config, "call1_models"),
        "language": configured_filter(config, "languages"),
        "variation": configured_filter(config, "variations"),
        "strategy": configured_filter(config, "strategies"),
    }
    responses = []
    for path in sorted(root.rglob("pc*_q*_*.txt")):
        metadata = parse_call1_path(path, root)
        if metadata is None:
            continue
        if any(values and metadata[field] not in values for field, values in filters.items()):
            continue
        responses.append(metadata)
    if not responses:
        raise SystemExit(f"Error: No Call 1 responses found under {root}")
    return responses


def discover_call2_candidates(root: Path, config: dict) -> dict[tuple[str, ...], list[dict]]:
    if not root.is_dir():
        raise SystemExit(f"Error: Call 2 candidate directory does not exist: {root}")
    model_filter = configured_filter(config, "call2_models")
    grouped = {}
    for suffix in ("*.yaml", "*.yml"):
        for path in sorted(root.rglob(suffix)):
            metadata = parse_call2_path(path, root)
            if metadata is None:
                continue
            if model_filter and metadata["call2_model"] not in model_filter:
                continue
            grouped.setdefault(response_key(metadata), []).append(metadata)
    return grouped


def load_candidate(path: Path):
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Error: selection requires PyYAML.") from exc
    try:
        return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.FullLoader), None
    except (OSError, yaml.YAMLError) as exc:
        return None, str(exc)


def collect_atoms(value) -> list[dict]:
    if isinstance(value, dict):
        if all(key in value for key in ("preconditions", "arguments", "outcomes")):
            return [value]
        atoms = []
        for child in value.values():
            atoms.extend(collect_atoms(child))
        return atoms
    if isinstance(value, (list, tuple)):
        atoms = []
        for child in value:
            atoms.extend(collect_atoms(child))
        return atoms
    return []


def candidate_strings(data: dict) -> list[str]:
    strings = []
    for atom in collect_atoms(data.get("subquestions", [])):
        for field in ("preconditions", "arguments", "outcomes"):
            strings.extend(item for item in atom[field] if isinstance(item, str))
    return strings


def content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
    return {token for token in tokens if len(token) > 1 and token not in STOPWORDS}


def normalize_formula(text: str) -> str:
    text = text.replace("$", "")
    text = re.sub(r"\\(?:!|,|;|:|quad|qquad)", "", text)
    return re.sub(r"\s+", "", text).casefold()


def source_formulas(text: str) -> list[str]:
    patterns = (
        r"\$\$(.+?)\$\$",
        r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)",
        r"\\\[(.+?)\\\]",
        r"\\\((.+?)\\\)",
    )
    formulas = []
    for pattern in patterns:
        formulas.extend(re.findall(pattern, text, flags=re.DOTALL))
    return [formula for formula in formulas if normalize_formula(formula)]


def formula_like_statements(strings: list[str]) -> list[str]:
    return [
        text
        for text in strings
        if "=" in text or "\\" in text or re.search(r"[∫∑√≤≥→∞]", text)
    ]


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def formula_fidelity(source: str, strings: list[str]) -> float:
    source_items = [normalize_formula(item) for item in source_formulas(source)]
    candidate_items = [normalize_formula(item) for item in formula_like_statements(strings)]
    source_items = [item for item in source_items if item]
    candidate_items = [item for item in candidate_items if item]
    if not source_items and not candidate_items:
        return 1.0
    if not source_items or not candidate_items:
        return 0.0
    source_blob = normalize_formula(source)
    candidate_blob = "".join(candidate_items)
    precision = sum(item in source_blob for item in candidate_items) / len(candidate_items)
    recall = sum(item in candidate_blob for item in source_items) / len(source_items)
    return f1(precision, recall)


def atom_completeness(atoms: list[dict]) -> float:
    if not atoms:
        return 0.0
    scores = []
    for atom in atoms:
        argument_score = 1.0 if atom.get("arguments") else 0.0
        outcome_score = 1.0 if atom.get("outcomes") else 0.0
        scores.append((argument_score + outcome_score) / 2)
    return sum(scores) / len(scores)


def non_duplication(atoms: list[dict]) -> float:
    if not atoms:
        return 0.0
    canonical = [
        repr(
            tuple(
                (field, tuple(str(item).strip() for item in atom.get(field, [])))
                for field in ("preconditions", "arguments", "outcomes")
            )
        )
        for atom in atoms
    ]
    return len(set(canonical)) / len(canonical)


def score_candidate(data: dict, source: str, weights: dict[str, float]) -> dict:
    strings = candidate_strings(data)
    atoms = collect_atoms(data.get("subquestions", []))
    source_tokens = content_tokens(source)
    converted_tokens = content_tokens("\n".join(strings))
    overlap = source_tokens & converted_tokens
    components = {
        "formula_fidelity": formula_fidelity(source, strings),
        "source_coverage": len(overlap) / len(source_tokens) if source_tokens else 1.0,
        "non_hallucination": len(overlap) / len(converted_tokens) if converted_tokens else 0.0,
        "atom_completeness": atom_completeness(atoms),
        "non_duplication": non_duplication(atoms),
    }
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise SystemExit("Error: ranking weights must have a positive sum.")
    score = sum(components[name] * weights[name] for name in DEFAULT_WEIGHTS) / total_weight
    return {"score": score, "atom_count": len(atoms), **components}


def configured_weights(config: dict) -> dict[str, float]:
    configured = get_nested(config, ["ranking", "weights"], {}) or {}
    unknown = set(configured) - set(DEFAULT_WEIGHTS)
    if unknown:
        raise SystemExit(f"Error: Unknown ranking weight(s): {', '.join(sorted(unknown))}")
    weights = {**DEFAULT_WEIGHTS, **{key: float(value) for key, value in configured.items()}}
    if any(value < 0 for value in weights.values()):
        raise SystemExit("Error: ranking weights cannot be negative.")
    return weights


def evaluate_candidate(candidate: dict, source: str, weights: dict, expected: int | None) -> dict:
    data, error = load_candidate(candidate["path"])
    if error is None and not isinstance(data, dict):
        error = "Top-level YAML value must be a mapping."
    if error is None:
        data, error = validate_converted_exercise(data, expected)
    result = {
        "call2_model": candidate["call2_model"],
        "path": str(candidate["path"]),
        "valid": error is None,
        "error": error or "",
    }
    if error is None:
        result.update(score_candidate(data, source, weights))
    return result


def dump_yaml(data: dict) -> str:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Error: selection requires PyYAML.") from exc
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, indent=4)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def run_selection(config: dict) -> dict:
    call1_root = project_path(get_nested(config, ["input", "call1_root"], "outputs/call1/plain_text"))
    call2_root = project_path(
        get_nested(config, ["input", "call2_root"], "outputs/call2_zeroshot_per_question")
    )
    output_root = project_path(
        get_nested(config, ["output", "root_directory"], "outputs/selected_responses")
    )
    report_name = str(get_nested(config, ["output", "report_file"], "selection_report.yaml"))
    synchronize = bool(get_nested(config, ["output", "synchronize_outputs"], True))
    tie_margin = float(get_nested(config, ["ranking", "tie_margin"], 0.03))
    weights = configured_weights(config)

    responses = discover_call1_responses(call1_root, config)
    candidates_by_key = discover_call2_candidates(call2_root, config)
    selections = []
    selected_count = 0
    all_failed_count = 0
    removed_stale_count = 0

    progress(f"Selection starting: {len(responses)} Call 1 response(s).")
    for index, response in enumerate(responses, start=1):
        source = response["path"].read_text(encoding="utf-8")
        candidates = candidates_by_key.get(response_key(response), [])
        expected = expected_subquestion_count(source)
        evaluated = [evaluate_candidate(item, source, weights, expected) for item in candidates]
        valid = sorted(
            (item for item in evaluated if item["valid"]),
            key=lambda item: (-item["score"], item["call2_model"]),
        )
        destination = selected_output_path(output_root, response)
        record = {
            "status": "selected" if valid else "all_candidates_failed",
            "call1_model": response["call1_model"],
            "language": response["language"],
            "variation": response["variation"],
            "exercise": f"pc{response['pc']}_q{response['exercise']}",
            "strategy": response["strategy"],
            "strategy_abbreviation": response["strategy_abbreviation"],
            "call1_path": str(response["path"]),
            "output_path": str(destination),
            "selected_call2_model": None,
            "selection_score": None,
            "selection_margin": None,
            "needs_review": False,
            "candidates": evaluated,
        }
        if valid:
            winner = valid[0]
            margin = winner["score"] - valid[1]["score"] if len(valid) > 1 else None
            source_candidate = Path(winner["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_candidate, destination)
            record.update(
                {
                    "selected_call2_model": winner["call2_model"],
                    "selection_score": round(winner["score"], 6),
                    "selection_margin": round(margin, 6) if margin is not None else None,
                    "needs_review": margin is not None and margin <= tie_margin,
                }
            )
            selected_count += 1
        else:
            all_failed_count += 1
            if synchronize and destination.exists():
                destination.unlink()
                removed_stale_count += 1
        selections.append(record)
        progress(
            f"[{index}/{len(responses)}] {record['exercise']}_{record['strategy_abbreviation']} | "
            f"model={record['call1_model']} | lang={record['language']}: {record['status']}"
        )

    report = {
        "summary": {
            "call1_responses": len(responses),
            "responses_selected": selected_count,
            "all_candidates_failed": all_failed_count,
            "selection_coverage": round(selected_count / len(responses), 6),
            "stale_outputs_removed": removed_stale_count,
            "call1_root": str(call1_root),
            "call2_root": str(call2_root),
            "selected_root": str(output_root),
        },
        "ranking": {"weights": weights, "tie_margin": tie_margin},
        "selections": selections,
    }
    report_path = output_root / report_name
    atomic_write(report_path, dump_yaml(report))
    progress(f"Selection complete. Wrote report: {report_path}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_selection(load_config(args.config.resolve()))


if __name__ == "__main__":
    main()
