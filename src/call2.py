import argparse
import json
from pathlib import Path
import re
import socket
import sys
import time
from string import Template
from urllib import error, request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "call2" / "example.yaml"
PROMPT_TYPE_ABBREVIATIONS = {
    "strictly_sequential": "seq",
    "prompt_accumulation": "acc",
    "ground_truth_forcing": "gtf",
    "self_history": "self",
}
ABBREVIATION_TO_PROMPT_TYPE = {
    abbreviation: prompt_type
    for prompt_type, abbreviation in PROMPT_TYPE_ABBREVIATIONS.items()
}


def load_config(config_path: Path) -> dict:
    """Load a JSON or YAML configuration file."""
    if not config_path.exists():
        raise SystemExit(f"Error: Config file '{config_path}' does not exist.")

    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "Error: YAML config files require PyYAML. Install it with "
            "`pip install -r requirements.txt`."
        ) from exc

    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Error: Config file '{config_path}' must contain a mapping.")
    return loaded


def project_path(path_value: str | Path) -> Path:
    """Resolve relative paths against the project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_nested(config: dict, keys: list[str], default=None):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def progress(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def sanitize_path_component(value: str) -> str:
    """Make model identifiers safe for directory names."""
    return re.sub(r"[\/\\:\s]+", "-", value).strip("-")


def infer_call1_output_mode(input_root: Path, suffix: str) -> str:
    if input_root.name in {"plain_text", "native_yaml"}:
        return input_root.name
    if suffix.lower() in {".yaml", ".yml"}:
        return "native_yaml"
    return "plain_text"


def parse_call1_path(path: Path, input_root: Path) -> dict | None:
    """Parse Call 1 complete-exercise output paths.

    Supported roots:
    - outputs/call1:
      {mode}/{model1}/{language}/{variation}/pc{n}_q{m}.{txt,yaml}
    - outputs/call1/plain_text or outputs/call1/native_yaml:
      {model1}/{language}/{variation}/pc{n}_q{m}.{txt,yaml}
    """
    try:
        relative = path.relative_to(input_root)
    except ValueError:
        return None

    if len(relative.parts) < 4:
        return None

    if relative.parts[0] in {"plain_text", "native_yaml"}:
        if len(relative.parts) < 5:
            return None
        output_mode, model1, language, variation = relative.parts[:4]
    else:
        output_mode = infer_call1_output_mode(input_root, path.suffix)
        model1, language, variation = relative.parts[:3]

    match = re.fullmatch(r"pc(\d+)_q(\d+)(?:_([A-Za-z0-9-]+))?", path.stem)
    if not match:
        return None

    prompt_type_abbrev = match.group(3) or ""
    prompt_type = parse_prompt_type_from_call1_output(path)
    if prompt_type == "unknown" and prompt_type_abbrev:
        prompt_type = ABBREVIATION_TO_PROMPT_TYPE.get(prompt_type_abbrev, prompt_type_abbrev)
    if not prompt_type_abbrev and prompt_type in PROMPT_TYPE_ABBREVIATIONS:
        prompt_type_abbrev = PROMPT_TYPE_ABBREVIATIONS[prompt_type]
    return {
        "call1_output_mode": output_mode,
        "model1": model1,
        "language": language,
        "variation": variation,
        "pc": match.group(1),
        "exercise": match.group(2),
        "prompt_type": prompt_type,
        "prompt_type_abbrev": prompt_type_abbrev or "unknown",
    }


def parse_prompt_type_from_call1_output(path: Path) -> str:
    if path.suffix.lower() != ".txt":
        return "unknown"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Prompt type:"):
            return line.split(":", 1)[1].strip() or "unknown"
    return "unknown"


def discover_call1_inputs(config: dict) -> list[dict]:
    """Find complete-exercise Call 1 output files selected by config filters."""
    input_root = project_path(get_nested(config, ["input", "root_directory"], "outputs/call1"))
    if not input_root.exists():
        raise SystemExit(f"Error: Input root directory '{input_root}' does not exist.")
    if not input_root.is_dir():
        raise SystemExit(f"Error: Input root path '{input_root}' is not a directory.")

    filters = get_nested(config, ["input", "filters"], {})
    mode_filter = set(filters.get("output_modes", ["plain_text"]))
    model_filter = {sanitize_path_component(model) for model in filters.get("models", [])}
    language_filter = set(filters.get("languages", []))
    variation_filter = set(filters.get("variations", []))
    prompt_type_filter = set(filters.get("prompt_types", []))

    suffixes = {".txt", ".yaml", ".yml"}
    files = []
    for path in sorted(input_root.rglob("pc*_q*.*")):
        if path.suffix.lower() not in suffixes:
            continue
        metadata = parse_call1_path(path, input_root)
        if metadata is None:
            continue
        if mode_filter and metadata["call1_output_mode"] not in mode_filter:
            continue
        if model_filter and metadata["model1"] not in model_filter:
            continue
        if language_filter and metadata["language"] not in language_filter:
            continue
        if variation_filter and metadata["variation"] not in variation_filter:
            continue
        if prompt_type_filter and metadata["prompt_type"] not in prompt_type_filter:
            continue
        metadata["path"] = path
        files.append(metadata)

    if not files:
        raise SystemExit(f"Error: No complete-exercise Call 1 files found under '{input_root}'.")
    return files


def enabled_models(config: dict) -> list[dict]:
    """Return enabled Call 2 model configs."""
    models = config.get("models", [])
    enabled = [model for model in models if model.get("enabled", True)]
    if not enabled:
        raise SystemExit("Error: No enabled Call 2 models configured.")
    return enabled


def require_prompt_templates(config: dict) -> dict:
    prompts = config.get("prompts")
    if not isinstance(prompts, dict):
        raise SystemExit("Error: config must define a 'prompts' mapping.")
    if not prompts.get("conversion"):
        raise SystemExit("Error: config.prompts.conversion is required.")
    if not prompts.get("repair"):
        raise SystemExit("Error: config.prompts.repair is required.")
    return prompts


def read_call1_output(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() not in {".yaml", ".yml"}:
        return text

    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "Error: Reading Call 1 YAML outputs requires PyYAML. Install it with "
            "`pip install -r requirements.txt`."
        ) from exc

    # Call 1 native YAML is benchmark-generated local data. FullLoader preserves
    # Python tuple tags if they are present. Do not use it for untrusted YAML.
    return yaml.load(text, Loader=yaml.FullLoader)


def dump_yaml(data) -> str:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "Error: YAML output requires PyYAML. Install it with "
            "`pip install -r requirements.txt`."
        ) from exc
    return yaml.dump(
        data,
        Dumper=yaml.Dumper,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        indent=4,
    )


def render_source_answer(source_answer) -> str:
    if isinstance(source_answer, str):
        return source_answer
    return dump_yaml(source_answer)


def build_conversion_prompt(prompts: dict, metadata: dict, source_answer) -> str:
    values = {
        **metadata,
        "source_answer": render_source_answer(source_answer),
    }
    return Template(prompts["conversion"]).safe_substitute(values).strip()


def build_repair_prompt(prompts: dict, raw_response: str, error_message: str) -> str:
    values = {
        "raw_response": raw_response,
        "error_message": error_message,
    }
    return Template(prompts["repair"]).safe_substitute(values).strip()


def ollama_generate(model: dict, prompt: str, endpoint: str, timeout: int) -> str:
    """Call Ollama's /api/generate endpoint."""
    parameters = model.get("parameters", {})
    payload = {
        "model": model["id"],
        "prompt": prompt,
        "stream": False,
        "options": {},
    }
    if "temperature" in parameters:
        payload["options"]["temperature"] = parameters["temperature"]
    if "max_output_tokens" in parameters:
        payload["options"]["num_predict"] = parameters["max_output_tokens"]

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except TimeoutError as exc:
        raise RuntimeError(
            f"Ollama request timed out for model '{model['id']}' after {timeout}s."
        ) from exc
    except socket.timeout as exc:
        raise RuntimeError(
            f"Ollama request timed out for model '{model['id']}' after {timeout}s."
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"Ollama request failed for model '{model['id']}': {exc}") from exc

    return result.get("response", "")


def strip_yaml_fence(raw_response: str) -> str:
    """Extract the YAML payload from a fenced model response when available."""
    text = raw_response.strip()
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


def quote_problematic_list_scalars(yaml_text: str) -> str:
    """Quote model-produced YAML list scalars that YAML may misread.

    Model outputs sometimes include lines like:
        - {X_t = t W_{1/t}, t > 0}
    YAML treats the leading "{" as a flow mapping, then fails on LaTeX braces.
    They may also include Markdown headings like:
        - **Starts at Zero:**
    YAML treats the leading "*" as an alias marker. This fallback preserves the
    text as a string while leaving real mappings and tagged structures alone.
    """
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


def validate_converted_exercise(data: dict) -> tuple[dict | None, str | None]:
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

    required_keys = ("preconditions", "arguments", "outcomes")
    missing_keys = [key for key in required_keys if key not in value]
    if missing_keys:
        return f"{location} is missing atom key(s): {', '.join(missing_keys)}."

    allowed_keys = set(required_keys) | {"strength"}
    unexpected_keys = sorted(set(value) - allowed_keys)
    if unexpected_keys:
        return f"{location} has unexpected key(s): {', '.join(unexpected_keys)}."

    for key in required_keys:
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


def output_path(config: dict, metadata: dict, model2: str) -> Path:
    """Build output YAML path for a converted complete exercise."""
    output_config = config.get("output", {})
    root = project_path(output_config.get("root_directory", "outputs/call2"))
    directory_template = output_config.get(
        "directory_template", "{model1}/{model2}/{language}/{variation}"
    )
    filename_template = output_config.get(
        "filename_template", "pc{pc}_q{exercise}_{prompt_type_abbrev}.yaml"
    )
    values = {
        **metadata,
        "model2": sanitize_path_component(model2),
        "model2_raw": model2,
    }
    return root / directory_template.format(**values) / filename_template.format(**values)


def raw_output_path(parsed_output_path: Path) -> Path:
    """Return sidecar path for the raw model response."""
    return parsed_output_path.with_suffix(".raw.txt")


def prompt_output_path(parsed_output_path: Path) -> Path:
    """Return sidecar path for the prompt sent to the model."""
    return parsed_output_path.with_suffix(".prompt.txt")


def atomic_write_text(path: Path, content: str, overwrite: bool) -> bool:
    """Atomically save content. Return True if written."""
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)
    return True


def write_error_report(config: dict, summary: dict, failed_jobs: list[dict]) -> None:
    output_config = config.get("output", {})
    root = project_path(output_config.get("root_directory", "outputs/call2"))
    report_filename = output_config.get("error_report_filename", "error_files.yaml")
    report_path = root / report_filename
    if not failed_jobs:
        if report_path.exists():
            report_path.unlink()
        return
    report = {
        "summary": summary,
        "failed_jobs": failed_jobs,
    }
    atomic_write_text(report_path, dump_yaml(report), overwrite=True)
    progress(f"Wrote error report: {report_path}")


def convert_with_repairs(
    model: dict,
    prompts: dict,
    metadata: dict,
    source_answer,
    endpoint: str,
    timeout: int,
    repair_attempts: int,
) -> tuple[dict | None, str, str | None, str]:
    """Convert a complete Call 1 answer to YAML, with optional YAML repair."""
    conversion_prompt = build_conversion_prompt(prompts, metadata, source_answer)
    raw_response = ollama_generate(
        model,
        conversion_prompt,
        endpoint,
        timeout,
    )
    parsed, error_message = parse_yaml_response(raw_response)
    if parsed is not None:
        parsed, error_message = validate_converted_exercise(parsed)

    attempts = 0
    while parsed is None and attempts < repair_attempts:
        attempts += 1
        raw_response = ollama_generate(
            model,
            build_repair_prompt(
                prompts,
                f"Original task:\n{conversion_prompt}\n\nInvalid response:\n{raw_response}",
                error_message or "unknown error",
            ),
            endpoint,
            timeout,
        )
        parsed, error_message = parse_yaml_response(raw_response)
        if parsed is not None:
            parsed, error_message = validate_converted_exercise(parsed)
    return parsed, raw_response, error_message, conversion_prompt


def run_call2(config: dict) -> None:
    inputs = discover_call1_inputs(config)
    models = enabled_models(config)
    prompts = require_prompt_templates(config)
    overwrite = get_nested(config, ["output", "overwrite_existing"], False)
    save_raw = get_nested(config, ["output", "save_raw_response"], True)
    save_prompt = get_nested(config, ["output", "save_prompt"], True)
    endpoint = get_nested(config, ["ollama", "endpoint"], "http://localhost:11434/api/generate")
    timeout = get_nested(config, ["ollama", "timeout_seconds"], 600)
    max_retries = get_nested(config, ["execution", "max_retries"], 1)
    retry_delay = get_nested(config, ["execution", "retry_delay_seconds"], 2)
    repair_attempts = get_nested(config, ["conversion", "repair", "max_attempts"], 1)

    written = 0
    skipped = 0
    failed = 0
    failed_jobs = []
    total_jobs = len(inputs) * len(models)
    current_job = 0

    def current_summary() -> dict:
        return {
            "call1_input_files": len(inputs),
            "conversion_jobs": total_jobs,
            "yaml_files_written": written,
            "yaml_files_skipped": skipped,
            "conversions_failed": failed,
        }

    progress(
        "Call 2 starting: "
        f"{len(inputs)} Call 1 file(s), {len(models)} model(s), "
        f"{total_jobs} conversion job(s)."
    )

    for item in inputs:
        source_answer = read_call1_output(item["path"])
        for model in models:
            current_job += 1
            out_path = output_path(config, item, model["id"])
            job_label = (
                f"[{current_job}/{total_jobs}] pc{item['pc']}_q{item['exercise']} | "
                f"call1_model={item['model1']} | call2_model={model['id']} | "
                f"variation={item['variation']} | prompt={item['prompt_type']}"
            )
            if out_path.exists() and not overwrite:
                skipped += 1
                progress(f"{job_label}: skipped existing {out_path}")
                continue

            try:
                progress(f"{job_label}: started")
                parsed = None
                raw_response = ""
                error_message = None
                conversion_prompt = ""
                for attempt in range(1, max_retries + 1):
                    try:
                        if max_retries > 1:
                            progress(f"{job_label}: Ollama attempt {attempt}/{max_retries}")
                        parsed, raw_response, error_message, conversion_prompt = convert_with_repairs(
                            model,
                            prompts,
                            item,
                            source_answer,
                            endpoint,
                            timeout,
                            repair_attempts,
                        )
                        break
                    except RuntimeError:
                        if attempt == max_retries:
                            raise
                        progress(f"{job_label}: retrying in {retry_delay}s")
                        time.sleep(retry_delay)

                if save_raw:
                    atomic_write_text(raw_output_path(out_path), raw_response, overwrite=True)
                if save_prompt:
                    atomic_write_text(prompt_output_path(out_path), conversion_prompt, overwrite=True)

                if parsed is None:
                    failed += 1
                    failed_jobs.append(
                        {
                            "exercise": f"pc{item['pc']}_q{item['exercise']}",
                            "call1_model": item["model1"],
                            "call2_model": model["id"],
                            "language": item["language"],
                            "variation": item["variation"],
                            "prompt_type": item["prompt_type"],
                            "call1_output_mode": item["call1_output_mode"],
                            "call1_path": str(item["path"]),
                            "output_path": str(out_path),
                            "error": error_message or "unknown YAML validation error",
                        }
                    )
                    write_error_report(config, current_summary(), failed_jobs)
                    print(
                        f"Error: Could not parse/validate Call 2 YAML generated from "
                        f"{item['path']}: {error_message}",
                        file=sys.stderr,
                    )
                    progress(f"{job_label}: failed YAML validation")
                    continue

                result = {
                    "pc": item["pc"],
                    "exercise": item["exercise"],
                    "language": item["language"],
                    "variation": item["variation"],
                    "prompt_type": item["prompt_type"],
                    "call1_output_mode": item["call1_output_mode"],
                    "call1_model": item["model1"],
                    "call2_model": model["id"],
                    "subquestions": parsed["subquestions"],
                }
                if atomic_write_text(out_path, dump_yaml(result), overwrite=True):
                    written += 1
                    progress(f"{job_label}: wrote {out_path}")
            except RuntimeError as exc:
                failed += 1
                failed_jobs.append(
                    {
                        "exercise": f"pc{item['pc']}_q{item['exercise']}",
                        "call1_model": item["model1"],
                        "call2_model": model["id"],
                        "language": item["language"],
                        "variation": item["variation"],
                        "prompt_type": item["prompt_type"],
                        "call1_output_mode": item["call1_output_mode"],
                        "call1_path": str(item["path"]),
                        "output_path": str(out_path),
                        "error": str(exc),
                    }
                )
                write_error_report(config, current_summary(), failed_jobs)
                print(f"Error: {exc}", file=sys.stderr)
                progress(f"{job_label}: failed")

    progress("Call 2 complete.")
    write_error_report(config, current_summary(), failed_jobs)
    print(f"Call 1 input files: {len(inputs)}")
    print(f"Conversion jobs: {total_jobs}")
    print(f"YAML files written: {written}")
    print(f"YAML files skipped: {skipped}")
    print(f"Conversions failed: {failed}")
    if failed_jobs:
        print("Failed jobs:")
        for job in failed_jobs:
            print(
                "- "
                f"{job['exercise']} | call1_model={job['call1_model']} | "
                f"call2_model={job['call2_model']} | language={job['language']} | "
                f"variation={job['variation']} | prompt={job['prompt_type']} | "
                f"error={job['error']}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Call 2 text-to-YAML conversion.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to a YAML or JSON Call 2 config file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    run_call2(config)


if __name__ == "__main__":
    main()
