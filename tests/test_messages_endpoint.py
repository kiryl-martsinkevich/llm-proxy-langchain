"""Tests for /v1/messages endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from llm_proxy.models.config import BackendConfig, ModelMapping, ProxyConfig
from llm_proxy.routes.messages import create_messages_router


@pytest.fixture
def sample_config() -> ProxyConfig:
    """Sample proxy configuration."""
    return ProxyConfig(
        models={
            "claude-3-haiku-20240307": ModelMapping(backend="ollama", model="llama3.2"),
        },
        backends={
            "ollama": BackendConfig(base_url="http://localhost:11434/v1"),
        },
    )


@pytest.fixture
def mock_ai_response():
    """Mock LangChain AI response."""
    return AIMessage(
        content="Hello! How can I help you today?",
        response_metadata={"finish_reason": "stop"},
        usage_metadata={"input_tokens": 10, "output_tokens": 8, "total_tokens": 18},
    )


@pytest.fixture
def app(sample_config, mock_ai_response):
    """FastAPI app with messages router."""
    app = FastAPI()

    with patch("llm_proxy.routes.messages.create_chat_model") as mock_factory:
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_ai_response)
        mock_factory.return_value = mock_model

        router = create_messages_router(sample_config)
        app.include_router(router)

    return app


def test_messages_simple_request(app, mock_ai_response):
    """Simple messages request returns Anthropic format."""
    with patch("llm_proxy.routes.messages.create_chat_model") as mock_factory:
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_ai_response)
        mock_factory.return_value = mock_model

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-haiku-20240307",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["model"] == "claude-3-haiku-20240307"
        assert len(data["content"]) == 1
        assert data["content"][0]["type"] == "text"


def test_messages_unknown_model(sample_config):
    """Unknown model returns 400 error."""
    app = FastAPI()
    router = create_messages_router(sample_config)
    app.include_router(router)

    client = TestClient(app)
    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-unknown",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert data["type"] == "error"
    assert data["error"]["type"] == "invalid_request_error"


def test_messages_invalid_request(sample_config):
    """Invalid request body returns 400."""
    app = FastAPI()
    router = create_messages_router(sample_config)
    app.include_router(router)

    client = TestClient(app)
    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-3-haiku-20240307",
            # missing max_tokens and messages
        },
    )

    assert response.status_code == 422  # Pydantic validation error
