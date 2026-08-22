import argparse
import csv
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
def project_path(path_value: str | Path) -> Path:
    """Resolve relative paths against the project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml_file(path: Path):
    """Load benchmark-generated YAML, preserving !!python/tuple values.

    Ground-truth YAML files are local trusted benchmark files. PyYAML FullLoader
    is used here because parser.py serializes mathematical sets as
    !!python/tuple. Do not use this loader for arbitrary untrusted YAML.
    """
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "Error: PyYAML is required. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    try:
        return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    except Exception as exc:
        raise SystemExit(f"Error: Could not read YAML file '{path}': {exc}") from exc


def parse_exercise_id(path: Path) -> tuple[str, str]:
    """Extract pc and exercise ids from pc{n}_q{m}.yaml."""
    match = re.fullmatch(r"pc(\d+)_q(\d+)", path.stem)
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def language_from_path(path: Path, input_root: Path) -> str:
    """Infer language from data/ground_truth/{lang}/... when possible."""
    try:
        relative = path.relative_to(input_root)
    except ValueError:
        return ""
    if len(relative.parts) >= 2:
        return relative.parts[0]
    return ""


def normalize_statement(text: str) -> str:
    """Keep formulas intact while removing accidental leading/trailing space."""
    return " ".join(text.strip().split())


def collect_field_values(value, field: str) -> list[str]:
    """Collect string statements from one statement field value."""
    if isinstance(value, str):
        return [normalize_statement(value)]
    if isinstance(value, (list, tuple)):
        statements = []
        for item in value:
            statements.extend(collect_field_values(item, field))
        return statements
    return []


def walk_atoms(value, atom_path: str, rows: list[dict], metadata: dict) -> None:
    """Recursively collect atom preconditions, arguments, and outcomes."""
    if isinstance(value, dict):
        for field in ("preconditions", "arguments", "outcomes"):
            if field not in value:
                continue
            for statement in collect_field_values(value[field], field):
                if not statement:
                    continue
                rows.append(
                    {
                        **metadata,
                        "field": field,
                        "atom_path": atom_path,
                        "text": statement,
                    }
                )
        return

    if isinstance(value, (list, tuple)):
        container = "tuple" if isinstance(value, tuple) else "list"
        for index, child in enumerate(value, start=1):
            walk_atoms(child, f"{atom_path}/{container}[{index}]", rows, metadata)


def collect_from_file(path: Path, input_root: Path) -> list[dict]:
    data = load_yaml_file(path)
    if not isinstance(data, dict):
        return []

    pc, exercise = parse_exercise_id(path)
    base_metadata = {
        "source_file": str(path.relative_to(PROJECT_ROOT))
        if path.is_relative_to(PROJECT_ROOT)
        else str(path),
        "language": language_from_path(path, input_root),
        "pc": pc,
        "exercise": exercise,
        "subquestion_index": "",
        "field": "",
        "atom_path": "",
        "text": "",
    }
    rows = []

    for statement in collect_field_values(data.get("assumption_global", []), "assumption_global"):
        if statement:
            rows.append(
                {
                    **base_metadata,
                    "field": "assumption_global",
                    "atom_path": "global",
                    "text": statement,
                }
            )

    subquestions = data.get("subquestions", [])
    if not isinstance(subquestions, list):
        return rows

    for sub_index, subquestion in enumerate(subquestions, start=1):
        if not isinstance(subquestion, dict):
            continue
        metadata = {**base_metadata, "subquestion_index": str(sub_index)}
        for statement in collect_field_values(subquestion.get("assumptions", []), "assumptions"):
            if statement:
                rows.append(
                    {
                        **metadata,
                        "field": "assumptions",
                        "atom_path": f"subquestions[{sub_index}].assumptions",
                        "text": statement,
                    }
                )
        if "atoms" in subquestion:
            walk_atoms(
                subquestion["atoms"],
                f"subquestions[{sub_index}].atoms",
                rows,
                metadata,
            )

    return rows


def discover_yaml_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise SystemExit(f"Error: Input directory '{input_dir}' does not exist.")
    if not input_dir.is_dir():
        raise SystemExit(f"Error: Input path '{input_dir}' is not a directory.")

    files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    )
    if not files:
        raise SystemExit(f"Error: No YAML files found under '{input_dir}'.")
    return files


def write_csv(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "language",
        "pc",
        "exercise",
        "subquestion_index",
        "field",
        "atom_path",
        "text",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_language_csvs(output_root: Path, rows: list[dict]) -> list[Path]:
    """Write one statement inventory per language."""
    grouped = {}
    for row in rows:
        language = row.get("language", "")
        if not language:
            raise SystemExit(
                f"Error: Could not infer language for statement from {row['source_file']}."
            )
        grouped.setdefault(language, []).append(row)

    paths = []
    for language, language_rows in sorted(grouped.items()):
        path = output_root / language / "ground_truth_statements.csv"
        write_csv(path, language_rows)
        paths.append(path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract candidate mathematical statements from ground-truth YAML "
            "for embedding-similarity calibration."
        )
    )
    parser.add_argument(
        "--input",
        default="data/ground_truth",
        help="Ground-truth YAML directory. Defaults to data/ground_truth.",
    )
    parser.add_argument(
        "--output",
        default="data/evaluation",
        help=(
            "Output root. One CSV is written to {output}/{lang}/"
            "ground_truth_statements.csv. A path ending in .csv keeps the old "
            "combined-file behavior."
        ),
    )
    parser.add_argument(
        "--language",
        action="append",
        default=[],
        help="Only extract this language. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = project_path(args.input)
    output_path = project_path(args.output)

    yaml_files = discover_yaml_files(input_dir)
    rows = []
    for yaml_file in yaml_files:
        language = language_from_path(yaml_file, input_dir)
        if args.language and language not in set(args.language):
            continue
        rows.extend(collect_from_file(yaml_file, input_dir))

    if not rows:
        print(
            f"Error: No statements were extracted from '{input_dir}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    if output_path.suffix.lower() == ".csv":
        write_csv(output_path, rows)
        output_paths = [output_path]
    else:
        output_paths = write_language_csvs(output_path, rows)
    print("Ground-truth statement extraction complete.")
    print(f"Input YAML files: {len(yaml_files)}")
    print(f"Statements extracted: {len(rows)}")
    for path in output_paths:
        print(f"Output CSV: {path}")


if __name__ == "__main__":
    main()
