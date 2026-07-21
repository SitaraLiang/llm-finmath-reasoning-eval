import argparse
import json
from pathlib import Path
import re
import sys
import time
from string import Template
from urllib import error, request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "call1" / "example.yaml"
CANONICAL_PROMPT_TYPES = {
    "strictly_sequential",
    "prompt_accumulation",
    "ground_truth_forcing",
    "self_history",
}
PROMPT_TYPE_ABBREVIATIONS = {
    "strictly_sequential": "seq",
    "prompt_accumulation": "acc",
    "ground_truth_forcing": "gtf",
    "self_history": "self",
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


def parse_exercise_filename(path: Path) -> tuple[str, str]:
    """Extract pc and exercise ids from pc{n}_q{m}.{yaml,yml,json}."""
    match = re.fullmatch(r"pc(\d+)_q(\d+)", path.stem)
    if not match:
        raise ValueError(f"File name '{path.name}' does not match pc{{n}}_q{{m}}.yaml")
    return match.group(1), match.group(2)


def infer_language(path: Path, root: Path) -> str:
    """Infer language from the first path component under root, e.g. en/pc2_q1.yaml."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return "unknown"
    if len(relative.parts) > 1:
        return relative.parts[0]
    return "unknown"


def discover_inputs(config: dict) -> list[dict]:
    """Find parser-generated exercise YAML files selected by the config filters."""
    root = project_path(get_nested(config, ["input", "root_directory"], "data/ground_truth"))
    if not root.exists():
        raise SystemExit(f"Error: Input root directory '{root}' does not exist.")
    if not root.is_dir():
        raise SystemExit(f"Error: Input root path '{root}' is not a directory.")

    languages = set(get_nested(config, ["input", "filters", "languages"], []))
    suffixes = {".yaml", ".yml", ".json"}
    files = []
    for path in sorted(root.rglob("pc*_q*.*")):
        if path.suffix.lower() not in suffixes:
            continue
        try:
            pc, exercise = parse_exercise_filename(path)
        except ValueError:
            continue
        language = infer_language(path, root)
        if languages and language not in languages:
            continue
        files.append({"path": path, "pc": pc, "exercise": exercise, "language": language})

    if not files:
        raise SystemExit(f"Error: No exercise YAML files found under '{root}'.")
    return files


def load_exercise_file(path: Path) -> dict:
    """Load one trusted parser output file.

    Parser-generated YAML may contain PyYAML's !!python/tuple tag, which is used
    by this benchmark to represent unordered mathematical sets. FullLoader
    reconstructs those tuples and should only be used on trusted benchmark files,
    not arbitrary YAML received from an untrusted source.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "Error: Ground-truth YAML files require PyYAML. Install it with "
            "`pip install -r requirements.txt`."
        ) from exc

    loaded = yaml.load(text, Loader=yaml.FullLoader)
    if not isinstance(loaded, dict):
        raise SystemExit(f"Error: Exercise file '{path}' must contain a mapping.")
    return loaded


def enabled_models(config: dict) -> list[dict]:
    """Return enabled Call 1 model configs."""
    models = config.get("models", [])
    enabled = [model for model in models if model.get("enabled", True)]
    if not enabled:
        raise SystemExit("Error: No enabled models configured.")
    return enabled


def sanitize_path_component(value: str) -> str:
    """Make model identifiers safe for directory names."""
    return re.sub(r"[\/\\:\s]+", "-", value).strip("-")


def format_items(items: list[str]) -> str:
    if not items:
        return "none"
    return "\n".join(f"- {item}" for item in items)


def validate_prompt_types(prompt_types: list[str]) -> None:
    unsupported = sorted(set(prompt_types) - CANONICAL_PROMPT_TYPES)
    if unsupported:
        expected = ", ".join(sorted(CANONICAL_PROMPT_TYPES))
        raise SystemExit(
            f"Error: Unsupported prompt type(s): {', '.join(unsupported)}. "
            f"Supported values are: {expected}."
        )


def validate_output_modes(output_modes: list[str]) -> None:
    unsupported = sorted(set(output_modes) - {"plain_text", "native_yaml"})
    if unsupported:
        raise SystemExit(
            f"Error: Unsupported output mode(s): {', '.join(unsupported)}. "
            "Supported values are: plain_text, native_yaml."
        )


def ensure_no_output_collisions(config: dict, prompt_types: list[str]) -> None:
    if len(prompt_types) <= 1:
        return
    output_config = config.get("output", {})
    directory_template = output_config.get(
        "directory_template", "{mode}/{model}/{language}/{variation}"
    )
    filename_template = output_config.get(
        "filename_template", "pc{pc}_q{exercise}{suffix}"
    )
    prompt_tokens = {"{prompt_type}", "{prompt_type_abbrev}"}
    templates = f"{directory_template}/{filename_template}"
    if not any(token in templates for token in prompt_tokens):
        raise SystemExit(
            "Error: Multiple prompt_types are configured, but output.directory_template "
            "and output.filename_template do not include {prompt_type} or "
            "{prompt_type_abbrev}. Add one of them to a template, or run one "
            "prompt_type at a time."
        )


def require_prompt_templates(config: dict) -> dict:
    prompts = config.get("prompts")
    if not isinstance(prompts, dict):
        raise SystemExit("Error: config must define a 'prompts' mapping.")
    if not prompts.get("common_header"):
        raise SystemExit("Error: config.prompts.common_header is required.")
    strategies = prompts.get("strategies")
    if not isinstance(strategies, dict):
        raise SystemExit("Error: config.prompts.strategies must be a mapping.")
    missing = sorted(CANONICAL_PROMPT_TYPES - set(strategies))
    if missing:
        raise SystemExit(
            "Error: config.prompts.strategies is missing template(s): "
            + ", ".join(missing)
        )
    return prompts


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


def get_ground_truth_answer(
    subquestion: dict,
    exercise_path: Path,
    previous_subquestion_index: int,
    fallback_keys: list[str],
) -> str:
    for key in fallback_keys:
        value = subquestion.get(key)
        if value in (None, "", [], ()):
            continue
        if isinstance(value, str):
            return value
        return dump_yaml(value).strip()
    raise SystemExit(
        f"Error: Ground-truth forcing requires a previous answer in '{exercise_path}', "
        f"subquestion {previous_subquestion_index}. Expected one of: {', '.join(fallback_keys)}."
    )


def format_question_block(index: int, subquestion: dict) -> str:
    return f"Question {index}:\n{subquestion.get('question', '')}"


def format_accumulated_questions(subquestions: list[dict], current_index: int) -> str:
    return "\n\n".join(
        format_question_block(index, subquestions[index - 1])
        for index in range(1, current_index + 1)
    )


def format_ground_truth_history(
    subquestions: list[dict],
    current_index: int,
    exercise_path: Path,
    fallback_keys: list[str],
) -> str:
    if current_index == 1:
        return "none"
    blocks = []
    for index in range(1, current_index):
        subquestion = subquestions[index - 1]
        answer = get_ground_truth_answer(subquestion, exercise_path, index, fallback_keys)
        blocks.append(f"{format_question_block(index, subquestion)}\n\nExact solution:\n{answer}")
    return "\n\n".join(blocks)


def format_self_history(subquestions: list[dict], generated_answers: list[str]) -> str:
    if not generated_answers:
        return "none"
    blocks = []
    for index, answer in enumerate(generated_answers, start=1):
        subquestion = subquestions[index - 1]
        blocks.append(f"{format_question_block(index, subquestion)}\n\nYour answer:\n{answer}")
    return "\n\n".join(blocks)


def build_prompt(
    exercise: dict,
    exercise_path: Path,
    subquestion: dict,
    subquestions: list[dict],
    subquestion_index: int,
    prompt_type: str,
    generated_answers: list[str],
    prompts: dict,
    ground_truth_keys: list[str],
) -> str:
    """Build the user prompt from config templates and strategy variables."""
    values = {
        "context": exercise.get("context") or "None",
        "global_assumptions": format_items(exercise.get("assumption_global", [])),
        "current_assumptions": format_items(subquestion.get("assumptions", [])),
        "current_question_number": str(subquestion_index),
        "current_question": subquestion.get("question", ""),
        "accumulated_questions": "",
        "ground_truth_history": "",
        "self_history": "",
    }
    if prompt_type == "prompt_accumulation":
        values["accumulated_questions"] = format_accumulated_questions(
            subquestions, subquestion_index
        )
    elif prompt_type == "ground_truth_forcing":
        values["ground_truth_history"] = format_ground_truth_history(
            subquestions,
            subquestion_index,
            exercise_path,
            ground_truth_keys,
        )
    elif prompt_type == "self_history":
        values["self_history"] = format_self_history(subquestions, generated_answers)

    header = Template(prompts["common_header"]).safe_substitute(values).strip()
    strategy = Template(prompts["strategies"][prompt_type]).safe_substitute(values).strip()
    return f"{header}\n\n{strategy}".strip()


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
    except error.URLError as exc:
        raise RuntimeError(f"Ollama request failed for model '{model['id']}': {exc}") from exc

    answer = result.get("response", "")
    if not answer.strip():
        done_reason = result.get("done_reason", "unknown")
        raise RuntimeError(
            f"Ollama returned an empty response for model '{model['id']}' "
            f"(done_reason={done_reason})."
        )
    return answer


def output_path(
    config: dict,
    output_mode: str,
    model_id: str,
    language: str,
    variation: str,
    pc: str,
    exercise: str,
    prompt_type: str,
) -> Path:
    """Build the single output path for a complete exercise."""
    output_config = config.get("output", {})
    root = project_path(output_config.get("root_directory", "outputs/call1"))
    mode_dir = "native_yaml" if output_mode == "native_yaml" else "plain_text"
    suffix = ".yaml" if output_mode == "native_yaml" else ".txt"

    directory_template = output_config.get(
        "directory_template", "{mode}/{model}/{language}/{variation}"
    )
    filename_template = output_config.get(
        "filename_template", "pc{pc}_q{exercise}{suffix}"
    )
    values = {
        "mode": mode_dir,
        "model": sanitize_path_component(model_id),
        "model1": model_id,
        "language": language,
        "variation": variation,
        "pc": pc,
        "exercise": exercise,
        "prompt_type": prompt_type,
        "prompt_type_abbrev": PROMPT_TYPE_ABBREVIATIONS[prompt_type],
        "suffix": suffix,
    }
    return root / directory_template.format(**values) / filename_template.format(**values)


def atomic_write_text(path: Path, content: str, overwrite: bool) -> bool:
    """Atomically save content. Return True if the file was written."""
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)
    return True


def serialize_plain_text_result(result: dict) -> str:
    lines = [
        f"Exercise: pc{result['pc']}_q{result['exercise']}",
        f"Model: {result['model']}",
        f"Variation: {result['variation']}",
        f"Prompt type: {result['prompt_type']}",
        "",
    ]
    for answer in result["answers"]:
        lines.extend(
            [
                f"Question {answer['subquestion']}:",
                answer["question"],
                "",
                "Answer:",
                answer["answer"],
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def serialize_result(result: dict, output_mode: str) -> str:
    if output_mode == "native_yaml":
        return dump_yaml(result)
    return serialize_plain_text_result(result)


def generate_complete_exercise_answers(
    exercise_data: dict,
    exercise_path: Path,
    model: dict,
    prompt_type: str,
    prompts: dict,
    ground_truth_keys: list[str],
    endpoint: str,
    timeout: int,
    max_retries: int,
    retry_delay: int,
    progress_prefix: str,
) -> list[dict]:
    subquestions = exercise_data.get("subquestions", [])
    generated_answers = []
    results = []

    for sub_index, subquestion in enumerate(subquestions, start=1):
        progress(
            f"{progress_prefix} subquestion {sub_index}/{len(subquestions)}: sending prompt"
        )
        prompt = build_prompt(
            exercise_data,
            exercise_path,
            subquestion,
            subquestions,
            sub_index,
            prompt_type,
            generated_answers,
            prompts,
            ground_truth_keys,
        )

        answer = ""
        for attempt in range(1, max_retries + 1):
            try:
                if max_retries > 1:
                    progress(
                        f"{progress_prefix} subquestion {sub_index}/{len(subquestions)}: "
                        f"Ollama attempt {attempt}/{max_retries}"
                    )
                answer = ollama_generate(model, prompt, endpoint, timeout)
                progress(
                    f"{progress_prefix} subquestion {sub_index}/{len(subquestions)}: "
                    f"received {len(answer.strip())} character(s)"
                )
                break
            except RuntimeError:
                if attempt == max_retries:
                    raise
                progress(
                    f"{progress_prefix} subquestion {sub_index}/{len(subquestions)}: "
                    f"retrying in {retry_delay}s"
                )
                time.sleep(retry_delay)

        generated_answers.append(answer)
        results.append(
            {
                "subquestion": sub_index,
                "question": subquestion.get("question", ""),
                "answer": answer,
            }
        )

    return results


def run_call1(config: dict) -> None:
    inputs = discover_inputs(config)
    models = enabled_models(config)
    prompts = require_prompt_templates(config)
    variations = get_nested(config, ["input", "filters", "variations"], ["baseline"])
    prompt_types = get_nested(config, ["input", "filters", "prompt_types"], ["strictly_sequential"])
    output_modes = config.get("output_modes", ["plain_text"])
    ground_truth_keys = get_nested(
        config,
        ["input", "ground_truth_answer_keys"],
        ["solution", "ground_truth", "answer", "expected_answer", "atoms"],
    )
    validate_prompt_types(prompt_types)
    validate_output_modes(output_modes)
    ensure_no_output_collisions(config, prompt_types)

    overwrite = get_nested(config, ["output", "overwrite_existing"], False)
    endpoint = get_nested(config, ["ollama", "endpoint"], "http://localhost:11434/api/generate")
    timeout = get_nested(config, ["ollama", "timeout_seconds"], 600)
    max_retries = get_nested(config, ["execution", "max_retries"], 1)
    retry_delay = get_nested(config, ["execution", "retry_delay_seconds"], 2)

    written = 0
    skipped = 0
    failed = 0
    total_jobs = (
        len(inputs)
        * len(variations)
        * len(prompt_types)
        * len(output_modes)
        * len(models)
    )
    current_job = 0

    progress(
        "Call 1 starting: "
        f"{len(inputs)} exercise(s), {len(models)} model(s), "
        f"{len(variations)} variation(s), {len(prompt_types)} prompt type(s), "
        f"{len(output_modes)} output mode(s), {total_jobs} complete exercise job(s)."
    )

    for item in inputs:
        exercise_data = load_exercise_file(item["path"])
        for variation in variations:
            for prompt_type in prompt_types:
                for output_mode in output_modes:
                    for model in models:
                        current_job += 1
                        out_path = output_path(
                            config,
                            output_mode,
                            model["id"],
                            item["language"],
                            variation,
                            item["pc"],
                            item["exercise"],
                            prompt_type,
                        )
                        job_label = (
                            f"[{current_job}/{total_jobs}] "
                            f"pc{item['pc']}_q{item['exercise']} | "
                            f"model={model['id']} | variation={variation} | "
                            f"prompt={prompt_type} | mode={output_mode}"
                        )
                        if out_path.exists() and not overwrite:
                            skipped += 1
                            progress(f"{job_label}: skipped existing {out_path}")
                            continue

                        try:
                            progress(f"{job_label}: started")
                            answers = generate_complete_exercise_answers(
                                exercise_data,
                                item["path"],
                                model,
                                prompt_type,
                                prompts,
                                ground_truth_keys,
                                endpoint,
                                timeout,
                                max_retries,
                                retry_delay,
                                job_label,
                            )
                            result = {
                                "pc": item["pc"],
                                "exercise": item["exercise"],
                                "language": item["language"],
                                "variation": variation,
                                "prompt_type": prompt_type,
                                "model": model["id"],
                                "answers": answers,
                            }
                            if atomic_write_text(
                                out_path,
                                serialize_result(result, output_mode),
                                overwrite=True,
                            ):
                                written += 1
                                progress(f"{job_label}: wrote {out_path}")
                        except RuntimeError as exc:
                            failed += 1
                            print(f"Error: {exc}", file=sys.stderr)
                            progress(f"{job_label}: failed")

    progress("Call 1 complete.")
    print(f"Input exercise files: {len(inputs)}")
    print(f"Complete exercise jobs: {total_jobs}")
    print(f"Complete exercise files written: {written}")
    print(f"Complete exercise files skipped: {skipped}")
    print(f"Complete exercise generations failed: {failed}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Call 1 answer generation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to a YAML or JSON Call 1 config file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    run_call1(config)


if __name__ == "__main__":
    main()
