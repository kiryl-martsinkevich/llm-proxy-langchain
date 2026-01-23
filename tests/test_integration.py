"""Integration tests for full request/response cycle with mocked backend."""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage


@pytest.fixture
def config_file():
    """Create temporary config file."""
    yaml_content = """
models:
  claude-3-haiku-20240307:
    backend: ollama
    model: llama3.2
  claude-3-sonnet-20240229:
    backend: openai
    model: gpt-4o

backends:
  ollama:
    base_url: "http://localhost:11434/v1"
  openai:
    base_url: "http://localhost:8080/v1"
    api_key: "test-key"
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(yaml_content)
        f.flush()
        yield f.name

    # Cleanup
    os.unlink(f.name)


@pytest.fixture
def mock_response():
    """Standard mock AI response."""
    return AIMessage(
        content="I'm a helpful assistant!",
        response_metadata={"finish_reason": "stop"},
        usage_metadata={"input_tokens": 15, "output_tokens": 6, "total_tokens": 21},
    )


@pytest.fixture
def app_client(config_file, mock_response, monkeypatch):
    """Create test client with mocked backend."""
    # Set env vars
    monkeypatch.setenv("LLM_PROXY_CONFIG", config_file)

    # Create mock model
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=mock_response)
    mock_model.bind_tools = MagicMock(return_value=mock_model)

    with patch("llm_proxy.routes.messages.create_chat_model") as mock_factory:
        mock_factory.return_value = mock_model

        # Import and create app after patching
        from llm_proxy.main import create_app

        app = create_app()
        client = TestClient(app)

        yield client, mock_model


def test_full_request_response_cycle(app_client):
    """Full request/response cycle with mocked backend."""
    client, mock_model = app_client

    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-3-haiku-20240307",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello, who are you?"}],
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify Anthropic response format
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["model"] == "claude-3-haiku-20240307"
    assert len(data["content"]) == 1
    assert data["content"][0]["type"] == "text"
    assert data["content"][0]["text"] == "I'm a helpful assistant!"
    assert data["stop_reason"] == "end_turn"
    assert data["usage"]["input_tokens"] == 15
    assert data["usage"]["output_tokens"] == 6

    # Verify mock was called
    mock_model.ainvoke.assert_called_once()


def test_conversation_history(app_client):
    """Multi-turn conversation passes correctly."""
    client, mock_model = app_client

    messages = [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": "What is my name?"},
    ]

    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-3-haiku-20240307",
            "max_tokens": 1024,
            "messages": messages,
        },
    )

    assert response.status_code == 200

    # Verify all messages were passed to mock
    mock_model.ainvoke.assert_called_once()
    call_args = mock_model.ainvoke.call_args

    # First positional argument should be list of LangChain messages
    langchain_messages = call_args[0][0]
    assert len(langchain_messages) == 3

    # Verify message types and content
    assert langchain_messages[0].content == "My name is Alice."
    assert langchain_messages[1].content == "Nice to meet you, Alice!"
    assert langchain_messages[2].content == "What is my name?"


def test_tool_calling(app_client, monkeypatch):
    """Tool definition passes to backend."""
    client, mock_model = app_client

    # Create tool response mock with tool_calls
    tool_response = AIMessage(
        content="",
        response_metadata={"finish_reason": "tool_calls"},
        usage_metadata={"input_tokens": 20, "output_tokens": 15, "total_tokens": 35},
        tool_calls=[
            {
                "id": "call_123",
                "name": "get_weather",
                "args": {"location": "San Francisco"},
            }
        ],
    )
    mock_model.ainvoke = AsyncMock(return_value=tool_response)

    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name",
                    }
                },
                "required": ["location"],
            },
        }
    ]

    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-3-haiku-20240307",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "What's the weather in SF?"}],
            "tools": tools,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify tool_use response format
    assert data["type"] == "message"
    assert data["stop_reason"] == "tool_use"

    # Find tool_use block in content
    tool_use_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
    assert len(tool_use_blocks) == 1

    tool_use = tool_use_blocks[0]
    assert tool_use["id"] == "call_123"
    assert tool_use["name"] == "get_weather"
    assert tool_use["input"] == {"location": "San Francisco"}

    # Verify bind_tools was called
    mock_model.bind_tools.assert_called_once()
    bind_tools_args = mock_model.bind_tools.call_args[0][0]

    # Verify tool was translated to OpenAI function format
    assert len(bind_tools_args) == 1
    assert bind_tools_args[0]["type"] == "function"
    assert bind_tools_args[0]["function"]["name"] == "get_weather"
