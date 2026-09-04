import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

import yaml


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SANITY_MODEL = "answerdotai/ModernBERT-base"
DEFAULT_PAIRS = Path("data/evaluation")
DEFAULT_OUTPUT_DIR = Path("outputs/evaluation")
DEFAULT_THRESHOLD_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "threshold"
DEFAULT_CALIBRATION_FILENAME = "calibration.yaml"
DEFAULT_MATRIX_THRESHOLD = 0.75
DIMENSIONS = {
    "D3": "outcomes",
    "D2": "arguments",
    "D1": "preconditions",
}


class EmbeddingModel:
    """Small wrapper around sentence-transformers cosine embeddings."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                "Error: eval_embeddings.py requires sentence-transformers. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.model = SentenceTransformer(model_name)
        self.cache: dict[str, object] = {}

    def encode(self, text: str):
        if text not in self.cache:
            self.cache[text] = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        return self.cache[text]

    def cosine(self, text_a: str, text_b: str) -> float:
        import numpy as np

        return float(np.dot(self.encode(text_a), self.encode(text_b)))


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def model_slug(model_name: str) -> str:
    return (
        model_name.lower()
        .replace("/", "-")
        .replace(":", "-")
        .replace(".", "-")
        .replace("_", "-")
    )


def threshold_model_slug(model_name: str) -> str:
    """Short folder name for threshold-calibration outputs."""
    normalized = model_name.lower()
    if normalized == "sentence-transformers/all-minilm-l6-v2":
        return "all-minilm-l6-v2"
    if normalized == "answerdotai/modernbert-base":
        return "modernBert"
    return model_slug(model_name.split("/")[-1])


def unique_model_names(model_names: list[str]) -> list[str]:
    seen = set()
    unique = []
    for model_name in model_names:
        normalized = model_name.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def formulation_pairs_language(path: Path, data: dict) -> str:
    """Infer calibration language from metadata, source paths, or parent folder."""
    metadata_language = data.get("metadata", {}).get("language")
    if metadata_language:
        return str(metadata_language)
    languages = set()
    for statement in data.get("selected_statements", []):
        source_file = str(statement.get("source_file", ""))
        match = re.search(r"(?:^|/)ground_truth/([^/]+)/", source_file)
        if match:
            languages.add(match.group(1))
    if len(languages) == 1:
        return languages.pop()
    if path.parent.name not in {"evaluation", "data", ""}:
        return path.parent.name
    raise SystemExit(
        f"Error: Could not infer one language for formulation-pair file '{path}'. "
        "Add metadata.language."
    )


def discover_formulation_pair_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise SystemExit(f"Error: formulation-pair input does not exist: {input_path}")
    files = sorted(input_path.glob("*/formulation_pairs.y*ml"))
    if not files:
        raise SystemExit(
            f"Error: no {{lang}}/formulation_pairs.yaml files found under {input_path}."
        )
    return files


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_calibration_registry(path: Path, entries: list[dict], policy: dict) -> None:
    """Merge generated model/language thresholds into a portable YAML registry."""
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise SystemExit(f"Error: calibration registry '{path}' must be a mapping.")
    else:
        loaded = {}

    registry = {
        "version": 1,
        "policy": policy,
        "models": loaded.get("models", {}),
    }
    if not isinstance(registry["models"], dict):
        raise SystemExit(f"Error: calibration registry '{path}' has invalid models data.")

    for entry in entries:
        model_data = registry["models"].setdefault(entry["model"], {})
        languages = model_data.setdefault("languages", {})
        languages[entry["language"]] = {
            "embedding_threshold": entry["embedding_threshold"],
            "selected_embedding_threshold": entry["selected_embedding_threshold"],
            "judge_low_threshold": entry["judge_low_threshold"],
            "judge_high_threshold": entry["judge_high_threshold"],
            "pair_count": entry["pair_count"],
            "balanced_accuracy": entry["balanced_accuracy"],
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        yaml.safe_dump(
            registry,
            allow_unicode=True,
            sort_keys=True,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def threshold_metrics(rows: list[dict], threshold: float) -> dict[str, float]:
    correct = 0
    equivalent_rows = [row for row in rows if row["expected_match"]]
    unrelated_rows = [row for row in rows if not row["expected_match"]]
    for row in rows:
        predicted_match = row["cosine_value"] >= threshold
        correct += predicted_match == row["expected_match"]
    equivalent_recall = (
        sum(row["cosine_value"] >= threshold for row in equivalent_rows)
        / len(equivalent_rows)
        if equivalent_rows
        else 0.0
    )
    unrelated_recall = (
        sum(row["cosine_value"] < threshold for row in unrelated_rows)
        / len(unrelated_rows)
        if unrelated_rows
        else 0.0
    )
    return {
        "accuracy": correct / len(rows),
        "equivalent_recall": equivalent_recall,
        "unrelated_recall": unrelated_recall,
        "balanced_accuracy": (equivalent_recall + unrelated_recall) / 2,
    }


def sweep_thresholds(rows: list[dict], step: float) -> tuple[float, list[dict]]:
    """Choose a cosine threshold from labeled equivalent/unrelated pairs."""
    if not rows:
        raise ValueError("Cannot choose a threshold without scored pairs.")
    if step <= 0 or step > 1:
        raise ValueError("--threshold-step must be in the interval (0, 1].")

    thresholds = []
    current = 0.0
    while current <= 1.0 + 1e-12:
        thresholds.append(round(current, 6))
        current += step

    sweep_rows = []
    best_balanced_accuracy = -1.0
    best_thresholds = []
    for threshold in thresholds:
        metrics = threshold_metrics(rows, threshold)
        sweep_rows.append(
            {
                "threshold": f"{threshold:.6f}",
                "accuracy": f"{metrics['accuracy']:.6f}",
                "balanced_accuracy": f"{metrics['balanced_accuracy']:.6f}",
                "equivalent_recall": f"{metrics['equivalent_recall']:.6f}",
                "unrelated_recall": f"{metrics['unrelated_recall']:.6f}",
            }
        )
        if metrics["balanced_accuracy"] > best_balanced_accuracy:
            best_balanced_accuracy = metrics["balanced_accuracy"]
            best_thresholds = [threshold]
        elif metrics["balanced_accuracy"] == best_balanced_accuracy:
            best_thresholds.append(threshold)

    # If a whole interval performs equally well, choose its midpoint rather than
    # an arbitrary edge. This keeps the suggested threshold stable and readable.
    recommended_threshold = (min(best_thresholds) + max(best_thresholds)) / 2
    recommended_threshold = round(recommended_threshold, 6)
    return recommended_threshold, sweep_rows


def score_statement_pairs(
    pairs_path: Path,
    output_dir: Path,
    model_name: str,
    threshold: float | None,
    threshold_step: float,
    language: str,
    judge_low_margin: float,
    judge_high_threshold: float,
) -> dict:
    data = load_yaml(pairs_path)
    embedder = EmbeddingModel(model_name)
    rows = []

    for template in data.get("pair_templates", []):
        for variant_index, text_b in enumerate(template.get("text_b_variants", []), start=1):
            text_a = template["text_a"]
            cosine = embedder.cosine(text_a, text_b)
            expected_match = template["label"] == "equivalent"
            rows.append(
                {
                    "language": language,
                    "pair_id": template["id"],
                    "source_statement_id": template["source_statement_id"],
                    "concept": template["concept"],
                    "label": template["label"],
                    "variant_index": variant_index,
                    "cosine_value": cosine,
                    "expected_match": expected_match,
                    "text_a": text_a,
                    "text_b": text_b,
                }
            )

    if not rows:
        raise SystemExit(f"Error: no pair variants found in {pairs_path}.")

    recommended_threshold, sweep_rows = sweep_thresholds(rows, threshold_step)
    selected_threshold = recommended_threshold if threshold is None else threshold
    selected_metrics = threshold_metrics(rows, selected_threshold)
    recommended_metrics = threshold_metrics(rows, recommended_threshold)
    generated_judge_low = max(0.0, recommended_threshold - judge_low_margin)
    if generated_judge_low > judge_high_threshold:
        raise SystemExit(
            "Error: generated judge low threshold exceeds --judge-high-threshold."
        )

    output_rows = []
    for row in rows:
        predicted_match = row["cosine_value"] >= selected_threshold
        output_rows.append(
                {
                    "language": language,
                    "pair_id": row["pair_id"],
                    "source_statement_id": row["source_statement_id"],
                    "concept": row["concept"],
                    "label": row["label"],
                    "variant_index": row["variant_index"],
                    "cosine": f"{row['cosine_value']:.6f}",
                    "threshold": f"{selected_threshold:.6f}",
                    "recommended_threshold": f"{recommended_threshold:.6f}",
                    "predicted_match": predicted_match,
                    "expected_match": row["expected_match"],
                    "correct_at_threshold": predicted_match == row["expected_match"],
                    "text_a": row["text_a"],
                    "text_b": row["text_b"],
                }
        )

    scores_path = output_dir / "formulation_pair_scores.csv"
    write_csv(
        scores_path,
        output_rows,
        [
            "language",
            "pair_id",
            "source_statement_id",
            "concept",
            "label",
            "variant_index",
            "cosine",
            "threshold",
            "recommended_threshold",
            "predicted_match",
            "expected_match",
            "correct_at_threshold",
            "text_a",
            "text_b",
        ],
    )

    summary_rows = []
    for label in sorted({row["label"] for row in output_rows}):
        label_scores = [float(row["cosine"]) for row in output_rows if row["label"] == label]
        label_correct = [row["correct_at_threshold"] for row in output_rows if row["label"] == label]
        summary_rows.append(
            {
                "language": language,
                "label": label,
                "count": len(label_scores),
                "mean": f"{statistics.fmean(label_scores):.6f}",
                "median": f"{statistics.median(label_scores):.6f}",
                "min": f"{min(label_scores):.6f}",
                "max": f"{max(label_scores):.6f}",
                "selected_threshold": f"{selected_threshold:.6f}",
                "recommended_threshold": f"{recommended_threshold:.6f}",
                "selected_balanced_accuracy": f"{selected_metrics['balanced_accuracy']:.6f}",
                "accuracy_at_threshold": f"{sum(label_correct) / len(label_correct):.6f}",
            }
        )
    summary_rows.append(
        {
            "language": language,
            "label": "overall",
            "count": len(output_rows),
            "mean": "",
            "median": "",
            "min": "",
            "max": "",
            "selected_threshold": f"{selected_threshold:.6f}",
            "recommended_threshold": f"{recommended_threshold:.6f}",
            "selected_balanced_accuracy": f"{selected_metrics['balanced_accuracy']:.6f}",
            "accuracy_at_threshold": f"{sum(row['correct_at_threshold'] for row in output_rows) / len(output_rows):.6f}",
        }
    )

    summary_path = output_dir / "formulation_pair_summary.csv"
    write_csv(
        summary_path,
        summary_rows,
        [
            "language",
            "label",
            "count",
            "mean",
            "median",
            "min",
            "max",
            "selected_threshold",
            "recommended_threshold",
            "selected_balanced_accuracy",
            "accuracy_at_threshold",
        ],
    )

    sweep_path = output_dir / "formulation_threshold_sweep.csv"
    write_csv(
        sweep_path,
        sweep_rows,
        [
            "threshold",
            "accuracy",
            "balanced_accuracy",
            "equivalent_recall",
            "unrelated_recall",
        ],
    )

    print(f"Wrote pair scores: {scores_path}")
    print(f"Wrote pair summary: {summary_path}")
    print(f"Wrote threshold sweep: {sweep_path}")
    print(f"Recommended threshold: {recommended_threshold:.6f}")
    return {
        "model": model_name,
        "language": language,
        "embedding_threshold": recommended_threshold,
        "selected_embedding_threshold": round(float(selected_threshold), 6),
        "judge_low_threshold": round(generated_judge_low, 6),
        "judge_high_threshold": round(float(judge_high_threshold), 6),
        "pair_count": len(output_rows),
        "balanced_accuracy": round(float(recommended_metrics["balanced_accuracy"]), 6),
    }


def is_atom(value) -> bool:
    return (
        isinstance(value, dict)
        and all(key in value for key in ("preconditions", "arguments", "outcomes"))
    )


def flatten_atoms(value, path: str = "atoms") -> list[dict]:
    """Flatten nested LIST/SET proof structures into atom records.

    YAML lists are ordered proof lists. YAML !!python/tuple values are treated by
    the benchmark as unordered mathematical sets, but we preserve their serialized
    order here only to keep deterministic row/column ids.
    """
    if is_atom(value):
        return [{"path": path, "atom": value}]
    if isinstance(value, (list, tuple)):
        kind = "set" if isinstance(value, tuple) else "list"
        atoms = []
        for index, child in enumerate(value, start=1):
            atoms.extend(flatten_atoms(child, f"{path}/{kind}[{index}]"))
        return atoms
    return []


def get_subquestions(data: dict) -> list[dict]:
    subquestions = data.get("subquestions", [])
    if not isinstance(subquestions, list):
        raise ValueError("Top-level 'subquestions' must be a list.")
    return subquestions


def normalize_strings(values) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        return [str(values)]
    return [str(value) for value in values if str(value).strip()]


def field_coverage_score(gt_values: list[str], pred_values: list[str], embedder: EmbeddingModel) -> float:
    """Score how well model-side statements cover GT-side statements."""
    if not gt_values:
        return 1.0
    if not pred_values:
        return 0.0
    best_scores = []
    for gt_value in gt_values:
        best_scores.append(max(embedder.cosine(gt_value, pred_value) for pred_value in pred_values))
    return statistics.fmean(best_scores)


def build_matrix_rows(
    gt_atoms: list[dict],
    pred_atoms: list[dict],
    dimension: str,
    field: str,
    embedder: EmbeddingModel,
    threshold: float,
) -> list[dict]:
    rows = []
    for gt_index, gt_record in enumerate(gt_atoms, start=1):
        gt_values = normalize_strings(gt_record["atom"].get(field, []))
        for pred_index, pred_record in enumerate(pred_atoms, start=1):
            pred_values = normalize_strings(pred_record["atom"].get(field, []))
            score = field_coverage_score(gt_values, pred_values, embedder)
            rows.append(
                {
                    "dimension": dimension,
                    "field": field,
                    "gt_atom_index": gt_index,
                    "model_atom_index": pred_index,
                    "gt_atom_path": gt_record["path"],
                    "model_atom_path": pred_record["path"],
                    "score": f"{score:.6f}",
                    "matched": score >= threshold,
                    "gt_text": " | ".join(gt_values),
                    "model_text": " | ".join(pred_values),
                }
            )
    return rows


def build_atom_matrices(
    ground_truth_path: Path,
    prediction_path: Path,
    output_dir: Path,
    model_name: str,
    threshold: float,
) -> None:
    gt_data = load_yaml(ground_truth_path)
    pred_data = load_yaml(prediction_path)
    gt_subquestions = get_subquestions(gt_data)
    pred_subquestions = get_subquestions(pred_data)
    embedder = EmbeddingModel(model_name)

    all_rows = []
    best_rows = []
    subquestion_count = max(len(gt_subquestions), len(pred_subquestions))
    for sub_index in range(1, subquestion_count + 1):
        gt_sub = gt_subquestions[sub_index - 1] if sub_index <= len(gt_subquestions) else {}
        pred_sub = pred_subquestions[sub_index - 1] if sub_index <= len(pred_subquestions) else {}
        gt_atoms = flatten_atoms(gt_sub.get("atoms", []), f"subquestions[{sub_index}].atoms")
        pred_atoms = flatten_atoms(pred_sub.get("atoms", []), f"subquestions[{sub_index}].atoms")

        for dimension, field in DIMENSIONS.items():
            rows = build_matrix_rows(gt_atoms, pred_atoms, dimension, field, embedder, threshold)
            for row in rows:
                row["subquestion_index"] = sub_index
            all_rows.extend(rows)

            for gt_atom_index in range(1, len(gt_atoms) + 1):
                candidates = [
                    row for row in rows if row["gt_atom_index"] == gt_atom_index
                ]
                if not candidates:
                    best_score = 0.0
                    best_model_atom_index = ""
                    matched = False
                else:
                    best = max(candidates, key=lambda row: float(row["score"]))
                    best_score = float(best["score"])
                    best_model_atom_index = best["model_atom_index"]
                    matched = best["matched"]
                best_rows.append(
                    {
                        "subquestion_index": sub_index,
                        "dimension": dimension,
                        "field": field,
                        "gt_atom_index": gt_atom_index,
                        "best_model_atom_index": best_model_atom_index,
                        "best_score": f"{best_score:.6f}",
                        "matched": matched,
                    }
                )

    matrix_path = output_dir / "atom_alignment_matrix.csv"
    write_csv(
        matrix_path,
        all_rows,
        [
            "subquestion_index",
            "dimension",
            "field",
            "gt_atom_index",
            "model_atom_index",
            "gt_atom_path",
            "model_atom_path",
            "score",
            "matched",
            "gt_text",
            "model_text",
        ],
    )

    best_path = output_dir / "atom_alignment_best_matches.csv"
    write_csv(
        best_path,
        best_rows,
        [
            "subquestion_index",
            "dimension",
            "field",
            "gt_atom_index",
            "best_model_atom_index",
            "best_score",
            "matched",
        ],
    )

    print(f"Wrote atom alignment matrix: {matrix_path}")
    print(f"Wrote best-match summary: {best_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute ModernBERT-style cosine similarities for evaluation calibration and atom matrices.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--sanity-model",
        default=None,
        help=(
            "Optional extra embedding model to run in pairs mode. Kept for "
            "backward compatibility; prefer pairs --models for several models."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Cosine threshold for match labels. In pairs mode, omit this to use "
            "the automatically recommended threshold. In matrix mode, omit this "
            f"to use {DEFAULT_MATRIX_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Step size for automatic threshold search in pairs mode. Default: 0.01",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root used to resolve relative paths. Default: current working directory.",
    )

    subparsers = parser.add_subparsers(dest="command")

    pairs = subparsers.add_parser(
        "pairs",
        help="Score language-specific formulation-pair files for threshold calibration.",
    )
    pairs.add_argument("--input", type=Path, default=DEFAULT_PAIRS)
    pairs.add_argument(
        "--models",
        action="append",
        default=[],
        help=(
            "Additional SentenceTransformer model to test in pairs mode. "
            "Can be repeated. Each model writes to its own output subfolder."
        ),
    )
    pairs.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_THRESHOLD_OUTPUT_DIR,
        help=(
            "Root directory for threshold-calibration outputs. Each embedding "
            "language/model is written to its own subdirectory. Default: "
            "outputs/evaluation/threshold."
        ),
    )
    pairs.add_argument(
        "--calibration-file",
        type=Path,
        default=None,
        help=(
            "Generated threshold registry. Relative paths use --project-root. "
            "Default: <output-dir>/calibration.yaml."
        ),
    )
    pairs.add_argument(
        "--judge-low-margin",
        type=float,
        default=0.10,
        help="Set judge low threshold to recommended embedding threshold minus this margin.",
    )
    pairs.add_argument(
        "--judge-high-threshold",
        type=float,
        default=1.0,
        help="Generated conservative judge upper threshold. Default: 1.0.",
    )

    matrix = subparsers.add_parser(
        "matrix",
        help="Build D1/D2/D3 atom alignment matrices for one GT YAML and one converted response YAML.",
    )
    matrix.add_argument("--ground-truth", type=Path, required=True)
    matrix.add_argument("--prediction", type=Path, required=True)
    matrix.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "matrices")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    project_root = args.project_root.resolve()
    if args.command == "pairs":
        if args.judge_low_margin < 0 or args.judge_low_margin > 1:
            raise SystemExit("Error: --judge-low-margin must be in [0, 1].")
        if args.judge_high_threshold < 0 or args.judge_high_threshold > 1:
            raise SystemExit("Error: --judge-high-threshold must be in [0, 1].")
        output_root = resolve_path(args.output_dir, project_root)
        calibration_path = (
            resolve_path(args.calibration_file, project_root)
            if args.calibration_file
            else output_root / DEFAULT_CALIBRATION_FILENAME
        )
        pairs_input = resolve_path(args.input, project_root)
        pair_files = discover_formulation_pair_files(pairs_input)
        model_names = unique_model_names([args.model, *args.models])
        if args.sanity_model:
            model_names = unique_model_names([*model_names, args.sanity_model])

        seen_languages = set()
        calibration_entries = []
        for pairs_path in pair_files:
            pair_data = load_yaml(pairs_path)
            language = formulation_pairs_language(pairs_path, pair_data)
            if language in seen_languages:
                raise SystemExit(f"Error: multiple formulation-pair files for language '{language}'.")
            seen_languages.add(language)
            for model_name in model_names:
                output_dir = output_root / language / threshold_model_slug(model_name)
                print(f"Running embedding model: {model_name} (language={language})")
                calibration_entries.append(score_statement_pairs(
                    pairs_path,
                    output_dir,
                    model_name,
                    args.threshold,
                    args.threshold_step,
                    language,
                    args.judge_low_margin,
                    args.judge_high_threshold,
                ))
        update_calibration_registry(
            calibration_path,
            calibration_entries,
            {
                "judge_low_margin": round(float(args.judge_low_margin), 6),
                "judge_high_threshold": round(float(args.judge_high_threshold), 6),
            },
        )
        print(f"Wrote calibration registry: {calibration_path}")
        return

    if args.command == "matrix":
        output_dir = resolve_path(args.output_dir, project_root)
        ground_truth_path = resolve_path(args.ground_truth, project_root)
        prediction_path = resolve_path(args.prediction, project_root)
        threshold = args.threshold
        if threshold is None:
            threshold = DEFAULT_MATRIX_THRESHOLD
            print(
                f"No --threshold supplied for matrix mode; using {threshold:.2f}. "
                "Use the recommended threshold from formulation_pair_summary.csv "
                "after running pairs mode."
            )
        build_atom_matrices(
            ground_truth_path,
            prediction_path,
            output_dir,
            args.model,
            threshold,
        )


if __name__ == "__main__":
    main()
