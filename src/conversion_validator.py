import re


ATOM_REQUIRED_KEYS = ("preconditions", "arguments", "outcomes")
ATOM_ALLOWED_KEYS = set(ATOM_REQUIRED_KEYS) | {"strength"}


def strip_yaml_fence(raw_response: str) -> str:
    """Extract the YAML payload from a fenced model response when available."""
    text = strip_thinking_blocks(raw_response).strip()
    fenced = re.search(r"```(?:yaml|yml)\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    fenced = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    opening_fence = re.search(r"```(?:yaml|yml)?\s*", text, re.IGNORECASE)
    if opening_fence:
        return text[opening_fence.end():].strip()
    return text


def strip_thinking_blocks(raw_response: str) -> str:
    """Remove model reasoning side-channels such as Qwen <think> blocks."""
    return re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        raw_response,
        flags=re.DOTALL | re.IGNORECASE,
    )


def quote_problematic_list_scalars(yaml_text: str) -> str:
    """Quote model-produced YAML list scalars that YAML may misread."""
    fixed_lines = []
    pattern = re.compile(r"^(\s*-\s+)(\{.*|\*\*.*)$")
    for line in yaml_text.splitlines():
        match = pattern.match(line)
        if not match:
            fixed_lines.append(line)
            continue
        prefix, value = match.groups()
        if re.match(r"^\{\s*['\"]?[\w-]+['\"]?\s*:", value):
            fixed_lines.append(line)
            continue
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        fixed_lines.append(f'{prefix}"{escaped}"')
    return "\n".join(fixed_lines)


def single_quote_double_quoted_latex_scalars(yaml_text: str) -> str:
    """Avoid YAML interpreting LaTeX backslashes as double-quote escapes."""
    fixed_lines = []
    pattern = re.compile(r'^(\s*-\s+)"(.*)"(\s*)$')
    for line in yaml_text.splitlines():
        match = pattern.match(line)
        if not match:
            fixed_lines.append(line)
            continue
        prefix, value, suffix = match.groups()
        if "\\" not in value:
            fixed_lines.append(line)
            continue
        escaped = value.replace("'", "''")
        fixed_lines.append(f"{prefix}'{escaped}'{suffix}")
    return "\n".join(fixed_lines)


def normalize_python_tuple_tags(yaml_text: str) -> str:
    """Accept common model typos for !!python/tuple."""
    return re.sub(r"!+python/tuple", "!!python/tuple", yaml_text)


def normalize_yaml_response_text(yaml_text: str) -> str:
    yaml_text = normalize_python_tuple_tags(yaml_text)
    yaml_text = single_quote_double_quoted_latex_scalars(yaml_text)
    return quote_problematic_list_scalars(yaml_text)


def parse_yaml_response(raw_response: str) -> tuple[dict | None, str | None]:
    yaml_text = normalize_yaml_response_text(strip_yaml_fence(raw_response))
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "Error: YAML parsing requires PyYAML. Install it with "
            "`pip install -r requirements.txt`."
        ) from exc

    try:
        documents = list(yaml.load_all(yaml_text, Loader=yaml.FullLoader))
    except yaml.YAMLError as exc:
        fixed_yaml_text = normalize_yaml_response_text(yaml_text)
        if fixed_yaml_text == yaml_text:
            return None, str(exc)
        try:
            documents = list(yaml.load_all(fixed_yaml_text, Loader=yaml.FullLoader))
        except yaml.YAMLError:
            return None, str(exc)
    non_empty_documents = [document for document in documents if document is not None]
    if not non_empty_documents:
        return None, "YAML response did not contain any non-empty document."
    data = non_empty_documents[0]
    if not isinstance(data, dict):
        return None, "Top-level YAML value must be a mapping."
    return data, None


def expected_subquestion_count(source_answer) -> int | None:
    """Infer how many top-level subquestions the conversion should return.

    Call 1 files are organized by script-generated "Question N:" blocks. Those
    are more reliable than model-generated "Answer:" labels, because a model can
    accidentally produce extra "Answer:" text inside one subquestion.
    """
    if isinstance(source_answer, str):
        question_labels = re.findall(r"(?m)^Question\s+\d+:\s*$", source_answer)
        if question_labels:
            return len(question_labels)
        answer_labels = re.findall(r"(?m)^Answer:\s*$", source_answer)
        return len(answer_labels) or None
    if isinstance(source_answer, dict):
        subquestions = source_answer.get("subquestions")
        if isinstance(subquestions, list):
            return len(subquestions)
    return None


def is_partial_atom(value) -> bool:
    """Return True for model-split atom fragments such as {'arguments': [...]}."""
    if not isinstance(value, dict):
        return False
    keys = set(value)
    return (
        bool(keys & set(ATOM_REQUIRED_KEYS))
        and keys.issubset(ATOM_ALLOWED_KEYS)
        and not set(ATOM_REQUIRED_KEYS).issubset(keys)
    )


def repair_split_atom_list(items: list) -> list:
    """Merge consecutive partial atom dicts into one complete atom dict."""
    repaired = []
    current = {}

    for item in items:
        item = repair_converted_exercise(item)

        if is_partial_atom(item):
            if set(current) & set(item):
                if current:
                    repaired.append(current)
                current = dict(item)
            else:
                current.update(item)

            if set(ATOM_REQUIRED_KEYS).issubset(current):
                repaired.append(current)
                current = {}
            continue

        if current:
            repaired.append(current)
            current = {}
        repaired.append(item)

    if current:
        repaired.append(current)
    return repaired


def repair_converted_exercise(value):
    """Repair common proof-YAML shape mistakes without changing valid atoms."""
    if isinstance(value, dict):
        return {key: repair_converted_exercise(item) for key, item in value.items()}
    if isinstance(value, list):
        return repair_split_atom_list(value)
    if isinstance(value, tuple):
        return tuple(repair_split_atom_list(list(value)))
    return value


def validate_converted_exercise(
    data: dict,
    expected_subquestions: int | None = None,
) -> tuple[dict | None, str | None]:
    forbidden_keys = {"name", "description", "version", "parameters", "scenarios"}
    present_forbidden_keys = sorted(forbidden_keys & set(data))
    if present_forbidden_keys:
        return (
            None,
            "Converted YAML looks like a generic benchmark configuration; "
            f"forbidden top-level key(s): {', '.join(present_forbidden_keys)}.",
        )
    if "subquestions" not in data:
        return None, "Converted YAML must contain a top-level 'subquestions' key."
    if not isinstance(data["subquestions"], list):
        return None, "Top-level 'subquestions' value must be a list."
    if expected_subquestions is not None and len(data["subquestions"]) != expected_subquestions:
        return (
            None,
            f"Expected exactly {expected_subquestions} top-level subquestion item(s), "
            f"because the Call 1 source contains {expected_subquestions} Question "
            f"block(s), but converted YAML contains {len(data['subquestions'])}. "
            "Do not split proof steps into extra top-level subquestions; keep them "
            "inside the atoms structure for the corresponding Question block.",
        )
    for index, subquestion in enumerate(data["subquestions"], start=1):
        if not isinstance(subquestion, dict):
            return None, f"subquestions[{index}] must be a mapping."
        if "atoms" not in subquestion:
            return None, f"subquestions[{index}] must contain an 'atoms' key."
        error_message = validate_atom_container(
            subquestion["atoms"],
            f"subquestions[{index}].atoms",
        )
        if error_message:
            return None, error_message
    return data, None


def validate_single_question_conversion(data: dict) -> tuple[object | None, str | None]:
    """Validate a single-question conversion and return its atoms structure.

    Per-question Call 2 can ask the model for a smaller payload:

        atoms: ...

    For compatibility with existing prompts/models, this also accepts:

        subquestions:
        - atoms: ...
    """
    if "atoms" in data:
        atoms = data["atoms"]
        error_message = validate_atom_container(atoms, "atoms")
        if error_message:
            return None, error_message
        return atoms, None

    if "subquestions" in data:
        if not isinstance(data["subquestions"], list):
            return None, "Top-level 'subquestions' value must be a list."
        if len(data["subquestions"]) != 1:
            return (
                None,
                "Single-question conversion must contain exactly one "
                "subquestions item when using the subquestions wrapper.",
            )
        subquestion = data["subquestions"][0]
        if not isinstance(subquestion, dict):
            return None, "subquestions[1] must be a mapping."
        if "atoms" not in subquestion:
            return None, "subquestions[1] must contain an 'atoms' key."
        atoms = subquestion["atoms"]
        error_message = validate_atom_container(atoms, "subquestions[1].atoms")
        if error_message:
            return None, error_message
        return atoms, None

    return None, "Single-question conversion must contain an 'atoms' key."


def validate_string_list(value, location: str) -> str | None:
    if not isinstance(value, list):
        return f"{location} must be a list of strings."
    for index, item in enumerate(value, start=1):
        if not isinstance(item, str):
            return f"{location}[{index}] must be a string."
    return None


def validate_atom(value, location: str) -> str | None:
    if not isinstance(value, dict):
        return f"{location} must be an atom mapping, list, or !!python/tuple."

    missing_keys = [key for key in ATOM_REQUIRED_KEYS if key not in value]
    if missing_keys:
        return f"{location} is missing atom key(s): {', '.join(missing_keys)}."

    unexpected_keys = sorted(set(value) - ATOM_ALLOWED_KEYS)
    if unexpected_keys:
        return f"{location} has unexpected key(s): {', '.join(unexpected_keys)}."

    for key in ATOM_REQUIRED_KEYS:
        error_message = validate_string_list(value[key], f"{location}.{key}")
        if error_message:
            return error_message
    if not value["arguments"]:
        return f"{location}.arguments must contain at least one argument."
    if "strength" in value and not isinstance(value["strength"], (int, float)):
        return f"{location}.strength must be numeric."
    return None


def validate_atom_container(value, location: str) -> str | None:
    """Validate nested proof structures.

    Lists are ordered proof lists. Tuples are serialized as !!python/tuple and
    are interpreted by the benchmark as unordered mathematical sets.
    """
    if isinstance(value, dict):
        return validate_atom(value, location)
    if not isinstance(value, (list, tuple)):
        return f"{location} must be an atom mapping, list, or !!python/tuple."
    for index, item in enumerate(value, start=1):
        error_message = validate_atom_container(item, f"{location}[{index}]")
        if error_message:
            return error_message
    return None
