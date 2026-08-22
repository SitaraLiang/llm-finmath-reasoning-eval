"""Create language-specific formulation-pair calibration seeds.

The script reads ``data/evaluation/{lang}/ground_truth_statements.csv`` and
writes ``data/evaluation/{lang}/formulation_pairs.yaml``. It is intentionally
language-agnostic: source statements are selected from the CSV inventory and
human-authored variants live only in YAML, never in Python.

Existing outputs are preserved unless ``--overwrite`` is supplied. During an
overwrite, completed variants are retained and only new entries receive TODO
placeholders. Use ``--reselect`` with ``--overwrite`` to discard the previous
selection and draw a new balanced sample from the current statement inventory.
"""

import argparse
import csv
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = ("equivalent", "related_but_not_equivalent", "unrelated")
FIELDS = ("assumption_global", "assumptions", "preconditions", "arguments", "outcomes")
VARIANTS_PER_LABEL = 2


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_statements(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def normalize_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def exercise_name(source_file: str) -> str:
    return Path(source_file).stem


def statement_key(statement: dict) -> tuple[str, str, str, str, str]:
    """Identify one statement within one language's ground-truth inventory."""
    return (
        exercise_name(str(statement.get("source_file", ""))),
        str(statement.get("subquestion_index", "")),
        str(statement.get("field", "")),
        str(statement.get("atom_path", "")),
        normalize_text(str(statement.get("text", ""))),
    )


def unique_statements(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    result = []
    for row in rows:
        key = statement_key(row)
        if not row.get("text", "").strip() or key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def balanced_selection(
    rows: list[dict[str, str]],
    count: int,
    retained: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Select statements deterministically while distributing source fields."""
    rows = unique_statements(rows)
    by_key = {statement_key(row): row for row in rows}
    selected = []
    selected_keys = set()

    for statement in retained or []:
        row = by_key.get(statement_key(statement))
        if row is not None and statement_key(row) not in selected_keys:
            selected.append(row)
            selected_keys.add(statement_key(row))
            if len(selected) == count:
                return selected

    queues = {
        field: [row for row in rows if row.get("field") == field]
        for field in FIELDS
    }
    positions = {field: 0 for field in FIELDS}
    while len(selected) < count:
        added = False
        for field in FIELDS:
            queue = queues[field]
            while positions[field] < len(queue):
                row = queue[positions[field]]
                positions[field] += 1
                key = statement_key(row)
                if key in selected_keys:
                    continue
                selected.append(row)
                selected_keys.add(key)
                added = True
                break
            if len(selected) == count:
                break
        if not added:
            break

    if len(selected) < count:
        raise ValueError(
            f"Requested {count} unique statements, but only {len(selected)} are available."
        )
    return selected


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Error: seed_pairs.py requires PyYAML.") from exc
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return data


def existing_annotations(data: dict) -> tuple[dict, dict]:
    statements = {
        statement_key(statement): statement
        for statement in data.get("selected_statements", [])
        if isinstance(statement, dict)
    }
    templates = {}
    statement_by_id = {
        statement.get("id"): statement
        for statement in data.get("selected_statements", [])
        if isinstance(statement, dict)
    }
    for template in data.get("pair_templates", []):
        if not isinstance(template, dict):
            continue
        source = statement_by_id.get(template.get("source_statement_id"))
        label = template.get("label")
        if source and label in LABELS:
            templates[(statement_key(source), label)] = template
    return statements, templates


def concept_name(statement: dict, statement_id: str, previous: dict | None) -> str:
    """Keep reviewed concept names; otherwise create a neutral stable label."""
    if previous and previous.get("concept"):
        return str(previous["concept"])
    exercise = re.sub(r"[^a-zA-Z0-9]+", "_", exercise_name(statement["source_file"]))
    field = re.sub(r"[^a-zA-Z0-9]+", "_", statement["field"])
    return f"{exercise}_{field}_{statement_id}"


def build_pairs(
    language: str,
    rows: list[dict[str, str]],
    concept_count: int,
    existing: dict | None = None,
    reselect: bool = False,
) -> dict:
    previous_statements, previous_templates = existing_annotations(existing or {})
    retained = [] if reselect else list(previous_statements.values())
    selected_rows = balanced_selection(rows, concept_count, retained)

    selected_statements = []
    pair_templates = []
    used_ids = {
        str(previous_statements[statement_key(row)].get("id"))
        for row in selected_rows
        if statement_key(row) in previous_statements
        and previous_statements[statement_key(row)].get("id")
    }
    next_id = 1
    for row in selected_rows:
        key = statement_key(row)
        previous = previous_statements.get(key)
        if previous and previous.get("id"):
            statement_id = str(previous["id"])
        else:
            while f"sem_{next_id:04d}" in used_ids:
                next_id += 1
            statement_id = f"sem_{next_id:04d}"
            used_ids.add(statement_id)
            next_id += 1
        concept = concept_name(row, statement_id, previous)
        statement = {
            "id": statement_id,
            "type": "semantic",
            "concept": concept,
            "field": row["field"],
            "source_file": row["source_file"],
            "subquestion_index": row["subquestion_index"],
            "atom_path": row["atom_path"],
            "text": row["text"],
        }
        selected_statements.append(statement)

        for label in LABELS:
            previous_template = previous_templates.get((key, label), {})
            variants = previous_template.get("text_b_variants")
            if not isinstance(variants, list) or not variants:
                variants = ["TODO"] * VARIANTS_PER_LABEL
            pair_templates.append(
                {
                    "id": f"{statement_id}_{label}",
                    "type": "semantic",
                    "label": label,
                    "concept": concept,
                    "source_statement_id": statement_id,
                    "text_a": row["text"],
                    "text_b_variants": variants,
                }
            )

    return {
        "metadata": {
            "language": language,
            "description": "Language-specific embedding threshold calibration set.",
            "generated_from": f"data/evaluation/{language}/ground_truth_statements.csv",
            "labels": list(LABELS),
            "concept_count": concept_count,
            "variants_per_label": VARIANTS_PER_LABEL,
        },
        "selected_statements": selected_statements,
        "pair_templates": pair_templates,
        "pairs": (existing or {}).get("pairs", []),
    }


def dump_yaml(path: Path, data: dict) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Error: seed_pairs.py requires PyYAML.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="data/evaluation")
    parser.add_argument("--output-root", default="data/evaluation")
    parser.add_argument("--language", action="append", default=[])
    parser.add_argument("--concept-count", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reselect",
        action="store_true",
        help="Select a new balanced sample instead of retaining prior selections.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.concept_count < 1:
        raise SystemExit("Error: --concept-count must be positive.")
    if args.reselect and not args.overwrite:
        raise SystemExit("Error: --reselect requires --overwrite.")

    input_root = project_path(args.input_root)
    output_root = project_path(args.output_root)
    files = sorted(input_root.glob("*/ground_truth_statements.csv"))
    if args.language:
        languages = set(args.language)
        files = [path for path in files if path.parent.name in languages]
    if not files:
        raise SystemExit(f"Error: No language statement CSV files found under {input_root}.")

    written = skipped = failed = 0
    for input_path in files:
        language = input_path.parent.name
        output_path = output_root / language / "formulation_pairs.yaml"
        if output_path.exists() and not args.overwrite:
            print(f"Skipped existing: {output_path}")
            skipped += 1
            continue
        try:
            existing = load_yaml(output_path) if output_path.exists() else None
            data = build_pairs(
                language,
                read_statements(input_path),
                args.concept_count,
                existing=existing,
                reselect=args.reselect,
            )
            dump_yaml(output_path, data)
            print(f"Written: {output_path}")
            written += 1
        except (OSError, ValueError) as exc:
            print(f"Error for language '{language}': {exc}", file=sys.stderr)
            failed += 1

    print(f"Pair files written: {written}")
    print(f"Pair files skipped: {skipped}")
    print(f"Pair files failed: {failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
