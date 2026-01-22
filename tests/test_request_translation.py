"""Tests for Anthropic to LangChain request translation."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llm_proxy.models.anthropic import Message, MessagesRequest, Tool
from llm_proxy.translation.request import (
    translate_messages,
    translate_tools,
    build_langchain_kwargs,
)


def test_translate_simple_user_message():
    """Translate simple user message."""
    messages = [Message(role="user", content="Hello")]
    result = translate_messages(messages, system=None)

    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "Hello"


def test_translate_with_system():
    """System prompt becomes SystemMessage."""
    messages = [Message(role="user", content="Hi")]
    result = translate_messages(messages, system="Be helpful")

    assert len(result) == 2
    assert isinstance(result[0], SystemMessage)
    assert result[0].content == "Be helpful"
    assert isinstance(result[1], HumanMessage)


def test_translate_conversation():
    """Multi-turn conversation."""
    messages = [
        Message(role="user", content="Hi"),
        Message(role="assistant", content="Hello!"),
        Message(role="user", content="How are you?"),
    ]
    result = translate_messages(messages, system=None)

    assert len(result) == 3
    assert isinstance(result[0], HumanMessage)
    assert isinstance(result[1], AIMessage)
    assert isinstance(result[2], HumanMessage)


def test_translate_multimodal_message():
    """Message with text and image."""
    messages = [
        Message(
            role="user",
            content=[
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgo=",
                    },
                },
            ],
        )
    ]
    result = translate_messages(messages, system=None)

    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)
    # LangChain uses list of content blocks for multimodal
    assert isinstance(result[0].content, list)
    assert result[0].content[0]["type"] == "text"
    assert result[0].content[1]["type"] == "image_url"


def test_translate_tools():
    """Convert Anthropic tools to LangChain format."""
    tools = [
        Tool(
            name="get_weather",
            description="Get the weather",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
    ]
    result = translate_tools(tools)

    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "get_weather"
    assert result[0]["function"]["parameters"]["properties"]["city"]["type"] == "string"


def test_build_langchain_kwargs():
    """Build kwargs for ChatOpenAI invocation."""
    request = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        messages=[Message(role="user", content="Hi")],
        temperature=0.7,
        top_p=0.9,
        stop_sequences=["END"],
    )
    kwargs = build_langchain_kwargs(request)

    assert kwargs["max_tokens"] == 1024
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9
    assert kwargs["stop"] == ["END"]


def test_build_langchain_kwargs_minimal():
    """Minimal kwargs only includes max_tokens."""
    request = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=512,
        messages=[Message(role="user", content="Hi")],
    )
    kwargs = build_langchain_kwargs(request)

    assert kwargs["max_tokens"] == 512
    assert "temperature" not in kwargs
    assert "stop" not in kwargs
