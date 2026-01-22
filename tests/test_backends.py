"""Tests for backend routing and factory."""

import pytest
from langchain_openai import ChatOpenAI

from llm_proxy.backends.router import resolve_backend, BackendNotFoundError
from llm_proxy.backends.factory import create_chat_model
from llm_proxy.models.config import BackendConfig, ModelMapping, ProxyConfig


@pytest.fixture
def sample_config() -> ProxyConfig:
    """Sample proxy configuration."""
    return ProxyConfig(
        models={
            "claude-3-haiku-20240307": ModelMapping(backend="ollama", model="llama3.2"),
            "claude-3-opus-20240229": ModelMapping(
                backend="databricks", model="llama-70b"
            ),
        },
        backends={
            "ollama": BackendConfig(base_url="http://localhost:11434/v1"),
            "databricks": BackendConfig(
                base_url="https://example.databricks.com/serving-endpoints",
                api_key="dapi-test",
            ),
        },
    )


def test_resolve_backend_found(sample_config):
    """Resolve known model to backend."""
    backend_config, model_name = resolve_backend(
        "claude-3-haiku-20240307", sample_config
    )

    assert backend_config.base_url == "http://localhost:11434/v1"
    assert model_name == "llama3.2"


def test_resolve_backend_not_found(sample_config):
    """Unknown model raises error."""
    with pytest.raises(BackendNotFoundError) as exc_info:
        resolve_backend("claude-unknown", sample_config)

    assert "claude-unknown" in str(exc_info.value)


def test_create_chat_model_ollama():
    """Create ChatOpenAI for Ollama backend."""
    backend_config = BackendConfig(base_url="http://localhost:11434/v1")

    model = create_chat_model(backend_config, "llama3.2")

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "llama3.2"
    assert str(model.openai_api_base) == "http://localhost:11434/v1"


def test_create_chat_model_with_api_key():
    """Create ChatOpenAI with API key."""
    backend_config = BackendConfig(
        base_url="https://example.databricks.com",
        api_key="dapi-secret",
    )

    model = create_chat_model(backend_config, "llama-70b")

    assert isinstance(model, ChatOpenAI)
    assert model.openai_api_key.get_secret_value() == "dapi-secret"
