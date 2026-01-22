"""Configuration loading with environment variable substitution."""

import os
import re
from pathlib import Path

import yaml

from llm_proxy.models.config import BackendConfig, ModelMapping, ProxyConfig

ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def substitute_env_vars(value: str) -> str:
    """Replace ${VAR} patterns with environment variable values."""
    if not isinstance(value, str):
        return value

    def replace_match(match: re.Match) -> str:
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            raise ValueError(f"Environment variable '{var_name}' not set")
        return env_value

    return ENV_VAR_PATTERN.sub(replace_match, value)


def substitute_env_vars_recursive(obj: dict | list | str) -> dict | list | str:
    """Recursively substitute env vars in a nested structure."""
    if isinstance(obj, dict):
        return {k: substitute_env_vars_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [substitute_env_vars_recursive(item) for item in obj]
    elif isinstance(obj, str):
        return substitute_env_vars(obj)
    return obj


def load_config(path: Path) -> ProxyConfig:
    """Load configuration from YAML file with env var substitution."""
    with open(path) as f:
        raw_config = yaml.safe_load(f)

    # Substitute environment variables
    config_data = substitute_env_vars_recursive(raw_config)

    return ProxyConfig(**config_data)
