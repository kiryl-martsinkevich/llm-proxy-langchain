"""Tests for configuration loading."""

import os
import tempfile
from pathlib import Path

import pytest
from llm_proxy.config import load_config, substitute_env_vars


def test_substitute_env_vars_simple(monkeypatch):
    """Substitute ${VAR} with environment variable."""
    monkeypatch.setenv("TEST_VAR", "hello")
    result = substitute_env_vars("prefix_${TEST_VAR}_suffix")
    assert result == "prefix_hello_suffix"


def test_substitute_env_vars_multiple(monkeypatch):
    """Substitute multiple variables."""
    monkeypatch.setenv("HOST", "localhost")
    monkeypatch.setenv("PORT", "8080")
    result = substitute_env_vars("http://${HOST}:${PORT}")
    assert result == "http://localhost:8080"


def test_substitute_env_vars_missing_raises():
    """Missing env var raises error."""
    with pytest.raises(ValueError, match="Environment variable 'MISSING_VAR' not set"):
        substitute_env_vars("${MISSING_VAR}")


def test_substitute_env_vars_no_vars():
    """String without vars returned unchanged."""
    result = substitute_env_vars("plain string")
    assert result == "plain string"


def test_load_config_from_yaml(monkeypatch):
    """Load config from YAML file with env substitution."""
    monkeypatch.setenv("TEST_API_KEY", "secret123")

    yaml_content = """
models:
  claude-3-haiku-20240307:
    backend: ollama
    model: llama3.2

backends:
  ollama:
    base_url: "http://localhost:11434/v1"
  databricks:
    base_url: "https://example.databricks.com"
    api_key: "${TEST_API_KEY}"

server:
  host: "127.0.0.1"
  port: 9000
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()

        config = load_config(Path(f.name))

        assert config.models["claude-3-haiku-20240307"].backend == "ollama"
        assert config.backends["databricks"].api_key == "secret123"
        assert config.server["port"] == 9000

    os.unlink(f.name)
