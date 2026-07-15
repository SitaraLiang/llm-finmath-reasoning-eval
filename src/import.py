import argparse
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = PROJECT_ROOT / "data" / "raw_tex"


def resolve_source_path(source_path: Path) -> Path:
    """Resolve source paths relative to the directory where the command is run."""
    if source_path.is_absolute():
        return source_path
    return (Path.cwd() / source_path).resolve()


def resolve_destination_path(destination_path: Path) -> Path:
    """Resolve destination paths relative to the project root."""
    if destination_path.is_absolute():
        return destination_path
    return (PROJECT_ROOT / destination_path).resolve()


def resolve_exercise_dir(source_dir: Path) -> Path:
    """Return the directory that should be searched for exercise .tex files."""
    if not source_dir.exists():
        raise SystemExit(f"Error: The source directory '{source_dir}' does not exist.")
    if not source_dir.is_dir():
        raise SystemExit(f"Error: The source path '{source_dir}' is not a directory.")

    if any(is_exercise_tex_file(path) for path in source_dir.rglob("pc*_q*_*.tex")):
        return source_dir

    raise SystemExit(
        "Error: The source directory does not contain matching exercise files "
    )


def is_exercise_tex_file(path: Path) -> bool:
    """Return True for names like pc2_q1_en.tex or pc2_q1_fr.tex."""
    if path.suffix != ".tex":
        return False

    stem_parts = path.stem.split("_")
    if len(stem_parts) != 3:
        return False

    pc_part, q_part, lang_part = stem_parts
    return (
        pc_part.startswith("pc")
        and pc_part[2:].isdigit()
        and q_part.startswith("q")
        and q_part[1:].isdigit()
        and bool(lang_part)
        and lang_part.isalpha()
    )


def get_language(path: Path) -> str:
    """Extract the language suffix from a pc{n}_q{m}_{lang}.tex filename."""
    return path.stem.split("_")[-1]


def find_tex_files(source_dir: Path) -> list[Path]:
    """Return exercise .tex files under source_dir, searched recursively."""
    exercise_dir = resolve_exercise_dir(source_dir)
    tex_files = sorted(
        path for path in exercise_dir.rglob("pc*_q*_*.tex") if is_exercise_tex_file(path)
    )
    # filter old versions of the files (e.g., pc2_q1_en_old.tex)
    tex_files = [path for path in tex_files if not path.stem.endswith("_old")]
    if not tex_files:
        raise SystemExit(
            "Error: No exercise .tex files matching 'pc{n}_q{m}_{lang}.tex' were "
            f"found in '{exercise_dir}'."
        )

    return tex_files


def copy_tex_files(tex_files: list[Path], destination_dir: Path) -> int:
    """Copy .tex files into destination_dir/{lang} while preserving file names."""
    destination_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for source_file in tex_files:
        language_dir = destination_dir / get_language(source_file)
        language_dir.mkdir(parents=True, exist_ok=True)
        destination_file = language_dir / source_file.name
        shutil.copy2(source_file, destination_file)
        copied += 1

    return copied


def import_tex_files(source_dir: Path, destination_dir: Path) -> None:
    """Import matching exercise .tex files from source_dir into destination_dir."""
    tex_files = find_tex_files(source_dir)
    copied = copy_tex_files(tex_files, destination_dir)

    print("Import complete.")
    print(f".tex files found: {len(tex_files)}")
    print(f".tex files copied: {copied}")
    print(f"Destination: {destination_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import pc{n}_q{m}_{lang}.tex exercise files from a downloaded Overleaf "
            "project into the framework raw_tex directory."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help=(
            "Path to the directory containing exercise .tex files."
        ),
    )
    parser.add_argument(
        "--destination",
        default=DEFAULT_DESTINATION,
        type=Path,
        help="Destination directory for imported .tex files. Defaults to data/raw_tex/.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_dir = resolve_source_path(args.source)
    destination_dir = resolve_destination_path(args.destination)
    import_tex_files(source_dir, destination_dir)


if __name__ == "__main__":
    main()
