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
        return {"mean_best_score": 0.0, "coverage": 0.0, "gt_atoms": 0, "matched_gt_atoms": 0}

    by_gt_atom = {}
    for row in rows:
        key = (row["subquestion_index"], row["gt_atom_index"])
        by_gt_atom.setdefault(key, []).append(row)

    best_scores = []
    matched = 0
    for candidates in by_gt_atom.values():
        best = max(candidates, key=lambda item: float(item["score"]))
        score = float(best["score"])
        best_scores.append(score)
        if best["matched"]:
            matched += 1

    gt_atoms = len(best_scores)
    return {
        "mean_best_score": statistics.fmean(best_scores) if best_scores else 0.0,
        "coverage": matched / gt_atoms if gt_atoms else 0.0,
        "gt_atoms": gt_atoms,
        "matched_gt_atoms": matched,
    }


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

    dimension_summaries = {
        dimension: summarize_dimension([row for row in all_rows if row["dimension"] == dimension])
        for dimension in ("D1", "D3", "D4")
    }
    d1 = dimension_summaries["D1"]["mean_best_score"]
    d3 = dimension_summaries["D3"]["mean_best_score"]
    d4 = dimension_summaries["D4"]["mean_best_score"]
    overall = statistics.fmean([d1, d3, d4])

    return {
        **base_row,
        "status": "ok",
        "error": "",
        "ground_truth_subquestions": len(gt_subquestions),
        "prediction_subquestions": len(pred_subquestions),
        "D1_mean_best_score": f"{d1:.6f}",
        "D1_coverage": f"{dimension_summaries['D1']['coverage']:.6f}",
        "D3_mean_best_score": f"{d3:.6f}",
        "D3_coverage": f"{dimension_summaries['D3']['coverage']:.6f}",
        "D4_mean_best_score": f"{d4:.6f}",
        "D4_coverage": f"{dimension_summaries['D4']['coverage']:.6f}",
        "overall_mean_score": f"{overall:.6f}",
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
        "D3_mean_best_score",
        "D3_coverage",
        "D4_mean_best_score",
        "D4_coverage",
        "overall_mean_score",
    ]
    for key, group_rows in sorted(groups.items()):
        row = {name: value for name, value in zip(keys, key)}
        row["cases"] = len(group_rows)
        for metric in metrics:
            row[metric] = f"{statistics.fmean(float(item[metric]) for item in group_rows):.6f}"
        aggregate_rows.append(row)
    return aggregate_rows


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
        "D3_mean_best_score",
        "D3_coverage",
        "D4_mean_best_score",
        "D4_coverage",
        "overall_mean_score",
        "prediction_path",
        "ground_truth_path",
        "output_dir",
    ]
    for row in summary_rows:
        for field in summary_fields:
            row.setdefault(field, "")
    write_csv(output_root / "evaluation_summary.csv", summary_rows, summary_fields)

    by_model = aggregate_by(summary_rows, ["call1_model"])
    write_csv(
        output_root / "evaluation_by_model.csv",
        by_model,
        [
            "call1_model",
            "cases",
            "D1_mean_best_score",
            "D1_coverage",
            "D3_mean_best_score",
            "D3_coverage",
            "D4_mean_best_score",
            "D4_coverage",
            "overall_mean_score",
        ],
    )

    by_model_strategy = aggregate_by(summary_rows, ["call1_model", "strategy"])
    write_csv(
        output_root / "evaluation_by_model_strategy.csv",
        by_model_strategy,
        [
            "call1_model",
            "strategy",
            "cases",
            "D1_mean_best_score",
            "D1_coverage",
            "D3_mean_best_score",
            "D3_coverage",
            "D4_mean_best_score",
            "D4_coverage",
            "overall_mean_score",
        ],
    )

    ok_count = sum(row["status"] == "ok" for row in summary_rows)
    failed_count = len(summary_rows) - ok_count
    progress(f"Evaluation complete: {ok_count} ok, {failed_count} failed.")
    print(f"Wrote summary: {output_root / 'evaluation_summary.csv'}")
    print(f"Wrote model aggregate: {output_root / 'evaluation_by_model.csv'}")
    print(f"Wrote model/strategy aggregate: {output_root / 'evaluation_by_model_strategy.csv'}")

    if failed_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
