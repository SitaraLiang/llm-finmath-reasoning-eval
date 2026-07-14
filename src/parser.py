import argparse
import json
from pathlib import Path
import re
import sys


TAG_RE = r"%@(?:CONTEXT|ASSUMPTION_GLOBAL|ASSUMPTION(?:_END)?|QUESTION(?:_END)?|ATOM(?:_END)?|PRECOND(?:_END)?|ARGUMENT(?:_END|:CALCUL)?|OUTCOME(?:_END)?|LIST_START|LIST_END|SET_START|SET_END)\b"


def normalize_tags(latex_content: str) -> str:
    return re.sub(r"%\s+@", "%@", latex_content)


def clean_text(text: str) -> str:
    text = re.sub(r"\\titledquestion\{(.*?)\}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\\item\[[^\]]*\]", "", text)
    text = re.sub(r"\\begin\{(?:itemize|enumerate)\}", "", text)
    text = re.sub(r"\\end\{(?:itemize|enumerate)\}", "", text)
    text = re.sub(r"^[\s%]+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = re.sub(r"\s*\(Step\s+\d+\)\.?$", "", text)
    text = text.rstrip(" :;")
    return text


def strip_trailing_sentence_punctuation(text: str) -> str:
    return re.sub(r"[.,](\$)?$", r"\1", text)


def strip_trailing_conjunction(text: str) -> str:
    return re.sub(r"\s+(?:and|et)$", "", text).strip()


def strip_leading_conjunction(text: str) -> str:
    return re.sub(r"^(?:and|et)\s+", "", text).strip()


def clean_question(text: str, assumptions: list[str]) -> str:
    list_match = re.search(r"\\begin\{(?:itemize|enumerate)\}.*?\\end\{(?:itemize|enumerate)\}", text, re.DOTALL)
    if list_match and assumptions:
        prefix = clean_text(text[:list_match.start()])
        inline_items = []
        for item in re.finditer(r"%@ASSUMPTION\b\s*(.*?)(?=%@ASSUMPTION|\\end\{(?:itemize|enumerate)\})", list_match.group(0), re.DOTALL):
            value = strip_trailing_sentence_punctuation(clean_text(item.group(1)))
            if value:
                inline_items.append(value)
        if inline_items:
            joined_items = ", ".join(inline_items[:-1])
            if joined_items:
                joined_items = f"{joined_items}, and {inline_items[-1]}"
            else:
                joined_items = inline_items[-1]
            return strip_trailing_sentence_punctuation(f"{prefix} {joined_items}")

    text = re.sub(r"\\begin\{(?:itemize|enumerate)\}.*?\\end\{(?:itemize|enumerate)\}", "", text, flags=re.DOTALL)
    return clean_text(text)


def split_tagged_items(text: str, tag: str, keep_percent_comments: bool = False) -> list[str]:
    items = []
    end_tag = rf"%@{tag}_END\b"
    pattern = re.compile(rf"%@{tag}\b\s*(.*?)(?={end_tag}|{TAG_RE}|\\begin\{{xsolution\}}|\\end\{{xsolution\}}|\\part\b|\Z)", re.DOTALL)
    for match in pattern.finditer(text):
        raw_value = match.group(1)
        value = clean_text(raw_value)
        if tag == "ASSUMPTION":
            value = strip_trailing_sentence_punctuation(value)
        if value:
            items.append(value)
    return items


def append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def has_tag(line: str, tag: str) -> bool:
    return re.search(rf"%@{tag}\b", line) is not None


def remove_tag_prefix(line: str, tag: str) -> str:
    return re.sub(rf"^.*?%@{tag}\b", "", line, count=1).strip()


def split_before_tag(line: str, tag: str) -> str:
    return re.split(rf"%@{tag}\b", line, maxsplit=1)[0].strip()


def extract_question_block(question_zone: str) -> tuple[str, list[str]]:
    assumptions = split_tagged_items(question_zone, "ASSUMPTION")

    question_text = question_zone
    question_match = re.search(r"%@QUESTION\b(.*)", question_text, re.DOTALL)
    if question_match:
        question_text = question_match.group(1)

    question_text = re.split(r"%@QUESTION_END\b", question_text, maxsplit=1)[0]
    question_text = re.sub(r"%@ASSUMPTION(?:_END)?\b", " ", question_text)
    question_text = re.sub(TAG_RE, " ", question_text)

    return clean_text(question_text), assumptions


def parse_latex_exercise(latex_content: str) -> dict:
    # Initialisation de l'exercice
    exercise = {"context": "", "assumption_global": [], "subquestions": []}

    # Nettoyage minimal et normalisation des espaces pour les TAGs
    content = normalize_tags(latex_content)

    # 1. Extraction du CONTEXT global
    context_match = re.search(
        r"%@CONTEXT\s*(.*?)(?=\\titledquestion|\\part|%@)", content, re.DOTALL
    )
    if context_match:
        context_text = clean_text(context_match.group(1))
        if not context_text:
            title_match = re.search(r"\\titledquestion\{(.*?)\}", content, re.DOTALL)
            if title_match:
                context_text = clean_text(title_match.group(1))
        exercise["context"] = context_text

    # 2. Extraction des ASSUMPTION_GLOBAL (si présentes avant les parts)
    exercise["assumption_global"] = split_tagged_items(content.split(r"\begin{parts}", 1)[0], "ASSUMPTION_GLOBAL")

    # 3. Découpage par sous-questions (\part)
    parts = re.split(r"\\part\b", content)

    for part_content in parts[1:]:
        subquestion = {"question": "", "assumptions": [], "atoms": []}

        # Localiser la zone de l'énoncé de la question (avant xsolution)
        solution_start_idx = part_content.find(r"\begin{xsolution}")
        question_zone = (
            part_content[:solution_start_idx]
            if solution_start_idx != -1
            else part_content
        )

        # Extraction de la QUESTION et des ASSUMPTION locales.
        # Les hypothèses locales sont désormais balisées à l'intérieur du bloc question.
        subquestion["question"], subquestion["assumptions"] = extract_question_block(question_zone)

        # 4. Traitement de la solution et de ses ATOMs
        if solution_start_idx != -1:
            solution_zone_match = re.search(
                r"\\begin{xsolution}(.*?)\\end{xsolution}",
                part_content,
                re.DOTALL,
            )
            if solution_zone_match:
                solution_text = solution_zone_match.group(1)

                # Découpage par blocs d'atomes
                atom_blocks = re.split(r"%@ATOM\b", solution_text)

                for block in atom_blocks[1:]:
                    atom = {"preconditions": [], "arguments": [], "outcome": ""}

                    lines = block.split("\n")
                    current_tag = None
                    current_buffer = []

                    def flush_current():
                        nonlocal current_tag, current_buffer
                        value = clean_text(" ".join(current_buffer))
                        if current_tag == "PRECOND":
                            stripped = strip_trailing_conjunction(value)
                            if stripped != value:
                                value = strip_trailing_sentence_punctuation(stripped)
                            elif (
                                "standard Brownian motion" in value
                                or value.startswith("$X_0")
                                or value.startswith("Assume ")
                            ):
                                value = strip_trailing_sentence_punctuation(value)
                            append_unique(atom["preconditions"], value)
                        elif current_tag == "ARGUMENT":
                            value = strip_leading_conjunction(value)
                            value = strip_trailing_sentence_punctuation(strip_trailing_conjunction(value))
                            append_unique(atom["arguments"], value)
                        elif current_tag == "ARGUMENT:CALCUL":
                            append_unique(atom["arguments"], "calcul")
                        elif current_tag == "OUTCOME":
                            value = re.sub(r"^which directly gives\s+", "", value)
                            if value.startswith("\\("):
                                value = strip_trailing_sentence_punctuation(value)
                            if value.startswith("So ") or value.startswith("The process is therefore") or value.startswith("This always gives"):
                                value = strip_trailing_sentence_punctuation(value)
                            atom["outcome"] = value
                        current_buffer = []

                    for line in lines:
                        line_stripped = line.strip()

                        if has_tag(line_stripped, "PRECOND"):
                            flush_current()
                            current_tag = "PRECOND"
                            content_line = remove_tag_prefix(line, "PRECOND")
                            if content_line:
                                current_buffer.append(content_line)

                        elif has_tag(line_stripped, "ARGUMENT:CALCUL"):
                            flush_current()
                            current_tag = "ARGUMENT:CALCUL"
                            content_line = remove_tag_prefix(line, "ARGUMENT:CALCUL")
                            if content_line:
                                current_buffer.append(content_line)

                        elif has_tag(line_stripped, "ARGUMENT"):
                            flush_current()
                            current_tag = "ARGUMENT"
                            content_line = remove_tag_prefix(line, "ARGUMENT")
                            if content_line:
                                current_buffer.append(content_line)

                        elif has_tag(line_stripped, "OUTCOME"):
                            flush_current()
                            current_tag = "OUTCOME"
                            content_line = remove_tag_prefix(line, "OUTCOME")
                            if content_line:
                                current_buffer.append(content_line)

                        elif (
                            has_tag(line_stripped, "PRECOND_END")
                            or has_tag(line_stripped, "ARGUMENT_END")
                            or has_tag(line_stripped, "OUTCOME_END")
                            or has_tag(line_stripped, "ATOM_END")
                        ):
                            for end_tag in ["PRECOND_END", "ARGUMENT_END", "OUTCOME_END", "ATOM_END"]:
                                if has_tag(line_stripped, end_tag) and current_tag:
                                    before_end = split_before_tag(line, end_tag)
                                    break
                            else:
                                before_end = ""
                            if before_end:
                                current_buffer.append(before_end)
                            flush_current()
                            current_tag = None

                            if has_tag(line_stripped, "ATOM_END"):
                                break

                        elif (
                            has_tag(line_stripped, "LIST_END")
                            or has_tag(line_stripped, "SET_END")
                            or has_tag(line_stripped, "LIST_START")
                            or has_tag(line_stripped, "SET_START")
                        ):
                            if current_tag:
                                before_end = re.split(r"%@(?:LIST_END|SET_END|LIST_START|SET_START)\b", line, maxsplit=1)[0].strip()
                                if before_end:
                                    current_buffer.append(before_end)
                            flush_current()
                            current_tag = None

                        elif current_tag and line_stripped:
                            if not any(
                                t in line_stripped
                                for t in [
                                    "%@LIST_START",
                                    "%@LIST_END",
                                    "%@SET_START",
                                    "%@SET_END",
                                ]
                            ):
                                current_buffer.append(line_stripped)

                    flush_current()

                    atom["preconditions"] = [p for p in atom["preconditions"] if p]
                    atom["arguments"] = [a for a in atom["arguments"] if a]

                    if atom["arguments"]:
                        subquestion["atoms"].append(atom)

        exercise["subquestions"].append(subquestion)

    return exercise


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw_tex" / "en"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "ground_truth" / "en"


def read_latex_file(filepath: Path) -> str:
    try:
        return filepath.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}", file=sys.stderr)
        sys.exit(1)


def write_json_file(output_path: Path, parsed_json: dict) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(parsed_json, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Successfully saved parsed JSON to {output_path}")
    except Exception as e:
        print(f"Error saving JSON to file '{output_path}': {e}", file=sys.stderr)
        sys.exit(1)


def parse_file_to_json(input_path: Path, output_path: Path) -> None:
    latex_content = read_latex_file(input_path)
    parsed_json = parse_latex_exercise(latex_content)
    write_json_file(output_path, parsed_json)


def parse_directory_to_json(input_dir: Path, output_dir: Path) -> None:
    if not input_dir.exists():
        print(f"Error: The input directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"Error: The input path '{input_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    tex_files = sorted(input_dir.glob("*.tex"))
    if not tex_files:
        print(
            f"Error: The input directory '{input_dir}' does not contain any .tex files.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in tex_files:
        output_path = output_dir / f"{input_path.stem}.json"
        parse_file_to_json(input_path, output_path)

    print(f"Parsed {len(tex_files)} file(s) from {input_dir} to {output_dir}")


if __name__ == "__main__":
    # Configuration d'argparse
    parser = argparse.ArgumentParser(
        description="Parse annotated LaTeX math exercises into structural JSON."
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        type=Path,
        help=(
            "Path to an input .tex file or to an input directory containing one "
            ".tex file per exercise."
        ),
        default=DEFAULT_INPUT_DIR,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        type=Path,
        help=(
            "Path to an output .json file when parsing one input file, or to an "
            "output directory when parsing an input directory."
        ),
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    if args.input_path.is_dir():
        if args.output_path.suffix == ".json":
            print(
                "Error: When input is a directory, output must also be a directory.",
                file=sys.stderr,
            )
            sys.exit(1)
        parse_directory_to_json(args.input_path, args.output_path)
    elif args.input_path.is_file():
        output_path = args.output_path
        if output_path.suffix != ".json":
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / f"{args.input_path.stem}.json"
        parse_file_to_json(args.input_path, output_path)
    else:
        print(f"Error: The input path '{args.input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
