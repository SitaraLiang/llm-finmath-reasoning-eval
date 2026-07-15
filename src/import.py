import argparse
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = PROJECT_ROOT / "data" / "raw_tex"


def find_tex_files(source_dir: Path) -> list[Path]:
    """Return all .tex files under source_dir, searched recursively."""
    if not source_dir.exists():
        raise SystemExit(f"Error: The source directory '{source_dir}' does not exist.")
    if not source_dir.is_dir():
        raise SystemExit(f"Error: The source path '{source_dir}' is not a directory.")

    tex_files = sorted(source_dir.rglob("*.tex"))
    if not tex_files:
        raise SystemExit(f"Error: No .tex files were found in '{source_dir}'.")

    return tex_files


def copy_tex_files(tex_files: list[Path], destination_dir: Path) -> int:
    """Copy .tex files into destination_dir while preserving file names."""
    destination_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for source_file in tex_files:
        destination_file = destination_dir / source_file.name
        shutil.copy2(source_file, destination_file)
        copied += 1

    return copied


def import_tex_files(source_dir: Path, destination_dir: Path) -> None:
    """Import all .tex files from source_dir into destination_dir."""
    tex_files = find_tex_files(source_dir)
    copied = copy_tex_files(tex_files, destination_dir)

    print("Import complete.")
    print(f".tex files found: {len(tex_files)}")
    print(f".tex files copied: {copied}")
    print(f"Destination: {destination_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import .tex exercise files from a downloaded Overleaf project into "
            "the framework raw_tex directory."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to the downloaded Overleaf project directory.",
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
    import_tex_files(args.source, args.destination)


if __name__ == "__main__":
    main()
