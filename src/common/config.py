"""
YAML-based configuration loader following FAIR's config-first pattern.

All experiments are fully specified by a YAML config file. No command-line
argument overrides — this ensures perfect reproducibility.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and resolve environment variable references."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config = _resolve_env_vars(config)
    config["_config_path"] = str(config_path.resolve())
    return config


def _resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve ${ENV_VAR} references in config values."""
    if isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]
            return os.environ.get(env_var, obj)
        return obj
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    return obj


def save_config(config: dict[str, Any], output_path: str | Path) -> None:
    """Save a config snapshot for reproducibility."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove internal keys before saving
    config_copy = copy.deepcopy(config)
    config_copy.pop("_config_path", None)

    with open(output_path, "w") as f:
        yaml.dump(config_copy, f, default_flow_style=False, sort_keys=False)


def get_nested(config: dict, key_path: str, default: Any = None) -> Any:
    """Get a nested config value using dot notation: 'model.encoder.embed_dim'."""
    keys = key_path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def merge_configs(base: dict, override: dict) -> dict:
    """Deep merge override into base config."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
