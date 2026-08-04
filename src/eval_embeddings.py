import argparse
import csv
import statistics
import sys
from pathlib import Path

import yaml


DEFAULT_MODEL = "answerdotai/ModernBERT-base"
DEFAULT_PAIRS = Path("data/evaluation/formulation_pairs.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs/evaluation")
DEFAULT_MATRIX_THRESHOLD = 0.75
DIMENSIONS = {
    "D1": "outcomes",
    "D3": "arguments",
    "D4": "preconditions",
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


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def threshold_accuracy(rows: list[dict], threshold: float) -> float:
    correct = 0
    for row in rows:
        predicted_match = row["cosine_value"] >= threshold
        correct += predicted_match == row["expected_match"]
    return correct / len(rows)


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
    best_accuracy = -1.0
    best_thresholds = []
    for threshold in thresholds:
        accuracy = threshold_accuracy(rows, threshold)
        equivalent_rows = [row for row in rows if row["expected_match"]]
        unrelated_rows = [row for row in rows if not row["expected_match"]]
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
        sweep_rows.append(
            {
                "threshold": f"{threshold:.6f}",
                "accuracy": f"{accuracy:.6f}",
                "equivalent_recall": f"{equivalent_recall:.6f}",
                "unrelated_recall": f"{unrelated_recall:.6f}",
            }
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_thresholds = [threshold]
        elif accuracy == best_accuracy:
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
) -> None:
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

    output_rows = []
    for row in rows:
        predicted_match = row["cosine_value"] >= selected_threshold
        output_rows.append(
                {
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
                "label": label,
                "count": len(label_scores),
                "mean": f"{statistics.fmean(label_scores):.6f}",
                "median": f"{statistics.median(label_scores):.6f}",
                "min": f"{min(label_scores):.6f}",
                "max": f"{max(label_scores):.6f}",
                "selected_threshold": f"{selected_threshold:.6f}",
                "recommended_threshold": f"{recommended_threshold:.6f}",
                "accuracy_at_threshold": f"{sum(label_correct) / len(label_correct):.6f}",
            }
        )
    summary_rows.append(
        {
            "label": "overall",
            "count": len(output_rows),
            "mean": "",
            "median": "",
            "min": "",
            "max": "",
            "selected_threshold": f"{selected_threshold:.6f}",
            "recommended_threshold": f"{recommended_threshold:.6f}",
            "accuracy_at_threshold": f"{sum(row['correct_at_threshold'] for row in output_rows) / len(output_rows):.6f}",
        }
    )

    summary_path = output_dir / "formulation_pair_summary.csv"
    write_csv(
        summary_path,
        summary_rows,
        [
            "label",
            "count",
            "mean",
            "median",
            "min",
            "max",
            "selected_threshold",
            "recommended_threshold",
            "accuracy_at_threshold",
        ],
    )

    sweep_path = output_dir / "formulation_threshold_sweep.csv"
    write_csv(
        sweep_path,
        sweep_rows,
        ["threshold", "accuracy", "equivalent_recall", "unrelated_recall"],
    )

    print(f"Wrote pair scores: {scores_path}")
    print(f"Wrote pair summary: {summary_path}")
    print(f"Wrote threshold sweep: {sweep_path}")
    print(f"Recommended threshold: {recommended_threshold:.6f}")


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
        help="Score formulation_pairs.yaml for threshold calibration.",
    )
    pairs.add_argument("--input", type=Path, default=DEFAULT_PAIRS)
    pairs.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    matrix = subparsers.add_parser(
        "matrix",
        help="Build D1/D3/D4 atom alignment matrices for one GT YAML and one converted response YAML.",
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
    output_dir = resolve_path(args.output_dir, project_root)

    if args.command == "pairs":
        pairs_path = resolve_path(args.input, project_root)
        score_statement_pairs(
            pairs_path,
            output_dir,
            args.model,
            args.threshold,
            args.threshold_step,
        )
        return

    if args.command == "matrix":
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
