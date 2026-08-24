"""Recursive YAML/JSON configuration loading shared by pipeline stages."""

import json
from pathlib import Path


def merge_config(base: dict, override: dict) -> dict:
    """Recursively merge mappings; lists and scalar values replace their base."""
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path, loading: tuple[Path, ...] = ()) -> dict:
    """Load a config and recursively merge parents listed under ``extends``."""
    config_path = config_path.resolve()
    if not config_path.exists():
        raise SystemExit(f"Error: Config file '{config_path}' does not exist.")
    if config_path in loading:
        chain = " -> ".join(str(path) for path in (*loading, config_path))
        raise SystemExit(f"Error: Circular config inheritance: {chain}")

    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        loaded = json.loads(text)
    else:
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

    parents = loaded.get("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or not all(isinstance(path, str) for path in parents):
        raise SystemExit(f"Error: Config file '{config_path}' extends must be a string or list.")

    merged = {}
    for parent in parents:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        merged = merge_config(merged, load_config(parent_path, (*loading, config_path)))
    return merge_config(merged, loaded)
