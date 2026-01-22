"""Tests for configuration models."""

import pytest
from llm_proxy.models.config import BackendConfig, ModelMapping, ProxyConfig


def test_backend_config_minimal():
    """Backend config with just base_url."""
    config = BackendConfig(base_url="http://localhost:11434/v1")
    assert config.base_url == "http://localhost:11434/v1"
    assert config.api_key is None


def test_backend_config_with_api_key():
    """Backend config with API key."""
    config = BackendConfig(
        base_url="https://workspace.databricks.com/serving-endpoints",
        api_key="dapi-xxxxx",
    )
    assert config.api_key == "dapi-xxxxx"


def test_model_mapping():
    """Model mapping links Anthropic model to backend."""
    mapping = ModelMapping(backend="ollama", model="llama3.2")
    assert mapping.backend == "ollama"
    assert mapping.model == "llama3.2"


def test_proxy_config_complete():
    """Full proxy configuration."""
    config = ProxyConfig(
        models={
            "claude-3-haiku-20240307": ModelMapping(backend="ollama", model="llama3.2"),
        },
        backends={
            "ollama": BackendConfig(base_url="http://localhost:11434/v1"),
        },
        server={"host": "0.0.0.0", "port": 8080},
    )
    assert "claude-3-haiku-20240307" in config.models
    assert "ollama" in config.backends
    assert config.server["port"] == 8080


def test_proxy_config_default_server():
    """Server config has sensible defaults."""
    config = ProxyConfig(
        models={},
        backends={},
    )
    assert config.server["host"] == "0.0.0.0"
    assert config.server["port"] == 8080
