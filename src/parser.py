import argparse
from pathlib import Path
import re
import sys

import yaml


TAG_RE = r"%@(?:CONTEXT|ASSUMPTION_GLOBAL|ASSUMPTION(?:_END)?|QUESTION(?:_END)?|ATOM(?:_END)?|PRECOND(?:_END)?|ARGUMENT(?:_END|:CALCUL)?|OUTCOME(?:_END)?|STRENGTH|LIST_START|LIST_END|SET_START|SET_END)\b"


class AnnotationParseError(ValueError):
    """Raised when annotated LaTeX tags are malformed or unclosed."""

    def __init__(self, message: str, source_name: str = "<latex>", line_number: int | None = None):
        location = source_name
        if line_number is not None:
            location = f"{location}:{line_number}"
        super().__init__(f"{location}: {message}")


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


def strip_latex_comment(line: str) -> str:
    return re.split(r"(?<!\\)%", line, maxsplit=1)[0].strip()


def is_non_tag_comment(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("%") and not re.search(TAG_RE, stripped)


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


def clean_argument_text(value: str) -> str:
    value = strip_leading_conjunction(value)
    value = strip_trailing_sentence_punctuation(strip_trailing_conjunction(value))
    value = value.replace("(or by taking", "or by taking")
    return value


def clean_outcome_text(value: str) -> str:
    value = re.sub(r"^which directly gives\s+", "", value)
    if value.startswith("\\("):
        value = strip_trailing_sentence_punctuation(value)
    if value.startswith("So ") or value.startswith("The process is therefore") or value.startswith("This always gives"):
        value = strip_trailing_sentence_punctuation(value)
    return value


def parse_strength(value: str, source_name: str, line_number: int) -> float:
    numeric_match = re.match(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value.strip())
    if numeric_match:
        return float(numeric_match.group(0))
    try:
        return float(value)
    except ValueError as exc:
        raise AnnotationParseError(
            f"Invalid @STRENGTH value {value!r}; expected a numeric value.",
            source_name,
            line_number,
        ) from exc


def parse_atom_block(block_lines: list[tuple[int, str]], source_name: str = "<latex>", atom_line: int | None = None) -> dict:
    atom = {"preconditions": [], "arguments": [], "outcomes": []}
    current_tag = None
    current_buffer = []

    def flush_current() -> None:
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
            append_unique(atom["arguments"], clean_argument_text(value))
        elif current_tag == "ARGUMENT:CALCUL":
            append_unique(atom["arguments"], "Calculation")
        elif current_tag == "OUTCOME":
            value = clean_outcome_text(value)
            append_unique(atom["outcomes"], value)
        current_buffer = []

    for line_number, line in block_lines:
        line_stripped = line.strip()

        if has_tag(line_stripped, "STRENGTH"):
            flush_current()
            current_tag = None
            strength_value = remove_tag_prefix(line, "STRENGTH")
            strength_value = strength_value[1:].strip() if strength_value.startswith(":") else strength_value
            atom["strength"] = parse_strength(clean_text(strength_value), source_name, line_number)

        elif has_tag(line_stripped, "PRECOND"):
            flush_current()
            current_tag = "PRECOND"

        elif has_tag(line_stripped, "ARGUMENT:CALCUL"):
            flush_current()
            current_tag = "ARGUMENT:CALCUL"

        elif has_tag(line_stripped, "ARGUMENT"):
            flush_current()
            current_tag = "ARGUMENT"

        elif has_tag(line_stripped, "OUTCOME"):
            flush_current()
            current_tag = "OUTCOME"

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
            before_end = strip_latex_comment(before_end)
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
                before_end = strip_latex_comment(before_end)
                if before_end:
                    current_buffer.append(before_end)
            flush_current()
            current_tag = None

        elif current_tag and line_stripped:
            content_line = strip_latex_comment(line_stripped)
            if content_line and not is_non_tag_comment(line_stripped):
                current_buffer.append(content_line)

    flush_current()

    atom["preconditions"] = [p for p in atom["preconditions"] if p]
    atom["arguments"] = [a for a in atom["arguments"] if a]
    atom["outcomes"] = [o for o in atom["outcomes"] if o]
    if not atom["arguments"]:
        raise AnnotationParseError(
            "Atom has no @ARGUMENT or @ARGUMENT:CALCUL content.",
            source_name,
            atom_line,
        )
    return atom


def stable_yaml_key(value) -> str:
    return yaml.dump(
        value,
        Dumper=yaml.Dumper,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )


def dedupe_set_items(items: list):
    seen = set()
    unique = []
    for item in items:
        key = stable_yaml_key(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def materialize_container(container: dict):
    values = [materialize_node(item) for item in container["items"]]
    if container["kind"] == "list":
        return values

    # Tuples are used by this benchmark as the YAML-serializable representation
    # of unordered mathematical sets. We preserve input order only to keep the
    # serialized YAML deterministic; tuple position is not semantic.
    return tuple(dedupe_set_items(values))


def materialize_node(node):
    if isinstance(node, dict) and "kind" in node and "items" in node:
        return materialize_container(node)
    return node


def add_child(stack: list[dict], roots: list, child) -> None:
    if stack:
        stack[-1]["items"].append(child)
    else:
        roots.append(child)


def starts_structure_or_atom(line: str) -> bool:
    return (
        has_tag(line, "ATOM")
        or has_tag(line, "LIST_START")
        or has_tag(line, "SET_START")
        or has_tag(line, "LIST_END")
        or has_tag(line, "SET_END")
    )


def collect_atom_lines(lines: list[tuple[int, str]], start_index: int) -> tuple[list[tuple[int, str]], int]:
    atom_line_number, atom_line = lines[start_index]
    block_lines = []
    initial_content = remove_tag_prefix(atom_line, "ATOM")
    if initial_content and not atom_line.strip().startswith("%"):
        block_lines.append((atom_line_number, initial_content))

    index = start_index + 1
    while index < len(lines):
        line_number, line = lines[index]
        line_stripped = line.strip()
        if has_tag(line_stripped, "ATOM_END"):
            block_lines.append((line_number, line))
            return block_lines, index + 1
        if starts_structure_or_atom(line_stripped):
            return block_lines, index
        block_lines.append((line_number, line))
        index += 1
    return block_lines, index


def parse_solution_structure(solution_text: str, source_name: str = "<latex>") -> dict:
    numbered_lines = list(enumerate(solution_text.split("\n"), start=1))
    stack = []
    roots = []
    saw_container = False
    index = 0

    while index < len(numbered_lines):
        line_number, line = numbered_lines[index]
        line_stripped = line.strip()

        if has_tag(line_stripped, "SET_START"):
            saw_container = True
            container = {"kind": "set", "items": [], "line": line_number}
            add_child(stack, roots, container)
            stack.append(container)
            index += 1
            continue

        if has_tag(line_stripped, "LIST_START"):
            saw_container = True
            container = {"kind": "list", "items": [], "line": line_number}
            add_child(stack, roots, container)
            stack.append(container)
            index += 1
            continue

        if has_tag(line_stripped, "LIST_END"):
            if not stack or stack[-1]["kind"] != "list":
                raise AnnotationParseError("Encountered @LIST_END without matching @LIST_START.", source_name, line_number)
            stack.pop()
            index += 1
            continue

        if has_tag(line_stripped, "SET_END"):
            if not stack or stack[-1]["kind"] != "set":
                raise AnnotationParseError("Encountered @SET_END without matching @SET_START.", source_name, line_number)
            stack.pop()
            index += 1
            continue

        if has_tag(line_stripped, "ATOM"):
            block_lines, next_index = collect_atom_lines(numbered_lines, index)
            atom = parse_atom_block(block_lines, source_name, line_number)
            add_child(stack, roots, atom)
            index = next_index
            continue

        index += 1

    if stack:
        container = stack[-1]
        raise AnnotationParseError(
            f"Unclosed @{container['kind'].upper()}_START annotation.",
            source_name,
            container["line"],
        )

    materialized_roots = [materialize_node(root) for root in roots]
    if not materialized_roots:
        return {"atoms": []}
    if saw_container and len(materialized_roots) == 1:
        return {"atoms": materialized_roots[0]}
    return {"atoms": materialized_roots}


def parse_latex_exercise(latex_content: str, source_name: str = "<latex>") -> dict:
    exercise = {"context": "", "assumption_global": [], "subquestions": []}

    content = normalize_tags(latex_content)

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

    exercise["assumption_global"] = split_tagged_items(content.split(r"\begin{parts}", 1)[0], "ASSUMPTION_GLOBAL")

    parts = re.split(r"\\part\b", content)

    for part_content in parts[1:]:
        subquestion = {"question": "", "assumptions": [], "atoms": []}

        solution_start_idx = part_content.find(r"\begin{xsolution}")
        question_zone = (
            part_content[:solution_start_idx]
            if solution_start_idx != -1
            else part_content
        )

        subquestion["question"], subquestion["assumptions"] = extract_question_block(question_zone)

        if solution_start_idx != -1:
            solution_zone_match = re.search(
                r"\\begin{xsolution}(.*?)\\end{xsolution}",
                part_content,
                re.DOTALL,
            )
            if solution_zone_match:
                solution_text = solution_zone_match.group(1)
                subquestion.update(parse_solution_structure(solution_text, source_name))

        exercise["subquestions"].append(subquestion)

    return exercise


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw_tex" / "en"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "ground_truth" / "en"
YAML_SUFFIXES = {".yaml", ".yml"}


def read_latex_file(filepath: Path) -> str:
    try:
        return filepath.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}", file=sys.stderr)
        sys.exit(1)


def write_yaml_file(output_path: Path, parsed_data: dict) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            yaml.dump(
                parsed_data,
                file,
                Dumper=yaml.Dumper,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
                indent=4,
            )
    except Exception as e:
        print(f"Error saving YAML to file '{output_path}': {e}", file=sys.stderr)
        sys.exit(1)


def load_yaml_file(input_path: Path):
    """Load trusted benchmark YAML, reconstructing tuples used as math sets.

    This loader is appropriate only for YAML generated by this benchmark or
    another trusted source. Do not use Python-specific YAML loaders on arbitrary
    untrusted files.
    """
    try:
        with input_path.open("r", encoding="utf-8") as file:
            return yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(f"Error loading YAML file '{input_path}': {e}", file=sys.stderr)
        sys.exit(1)


def parse_file_to_yaml(input_path: Path, output_path: Path) -> None:
    latex_content = read_latex_file(input_path)
    try:
        parsed_data = parse_latex_exercise(latex_content, str(input_path))
    except AnnotationParseError as exc:
        print(f"Error parsing '{input_path}': {exc}", file=sys.stderr)
        sys.exit(1)
    write_yaml_file(output_path, parsed_data)


def parse_directory_to_yaml(input_dir: Path, output_dir: Path) -> None:
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
        output_path = output_dir / f"{input_path.stem}.yaml"
        parse_file_to_yaml(input_path, output_path)

    print("Parsing complete.")
    print(f".tex files parsed: {len(tex_files)}")


def normalize_output_file_path(input_path: Path, output_path: Path) -> Path:
    if output_path.suffix in YAML_SUFFIXES:
        return output_path
    if output_path.suffix == ".json":
        return output_path.with_suffix(".yaml")
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / f"{input_path.stem}.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse annotated LaTeX math exercises into structural YAML."
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
            "Path to an output .yaml/.yml file when parsing one input file, or to an "
            "output directory when parsing an input directory. A legacy .json suffix "
            "is rewritten to .yaml."
        ),
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    if args.input_path.is_dir():
        if args.output_path.suffix in YAML_SUFFIXES or args.output_path.suffix == ".json":
            print(
                "Error: When input is a directory, output must also be a directory.",
                file=sys.stderr,
            )
            sys.exit(1)
        parse_directory_to_yaml(args.input_path, args.output_path)
    elif args.input_path.is_file():
        parse_file_to_yaml(args.input_path, normalize_output_file_path(args.input_path, args.output_path))
    else:
        print(f"Error: The input path '{args.input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
