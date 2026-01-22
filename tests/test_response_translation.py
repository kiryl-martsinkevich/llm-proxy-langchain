"""Tests for LangChain to Anthropic response translation."""

import pytest
from langchain_core.messages import AIMessage

from llm_proxy.models.anthropic import MessagesResponse
from llm_proxy.translation.response import translate_response, map_stop_reason


def test_translate_simple_response():
    """Translate simple text response."""
    ai_message = AIMessage(
        content="Hello, how can I help?",
        response_metadata={"finish_reason": "stop"},
        usage_metadata={"input_tokens": 10, "output_tokens": 8, "total_tokens": 18},
    )

    result = translate_response(ai_message, model="claude-3-haiku-20240307")

    assert result.role == "assistant"
    assert result.model == "claude-3-haiku-20240307"
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert result.content[0].text == "Hello, how can I help?"
    assert result.stop_reason == "end_turn"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 8


def test_translate_response_with_tool_calls():
    """Translate response with tool calls."""
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_123",
                "name": "get_weather",
                "args": {"city": "London"},
            }
        ],
        response_metadata={"finish_reason": "tool_calls"},
        usage_metadata={"input_tokens": 15, "output_tokens": 20, "total_tokens": 35},
    )

    result = translate_response(ai_message, model="claude-3-haiku-20240307")

    assert len(result.content) == 1
    assert result.content[0].type == "tool_use"
    assert result.content[0].id == "call_123"
    assert result.content[0].name == "get_weather"
    assert result.content[0].input == {"city": "London"}
    assert result.stop_reason == "tool_use"


def test_translate_response_text_and_tool():
    """Response with both text and tool call."""
    ai_message = AIMessage(
        content="Let me check the weather for you.",
        tool_calls=[
            {
                "id": "call_456",
                "name": "get_weather",
                "args": {"city": "Paris"},
            }
        ],
        response_metadata={"finish_reason": "tool_calls"},
        usage_metadata={"input_tokens": 20, "output_tokens": 25, "total_tokens": 45},
    )

    result = translate_response(ai_message, model="claude-3-haiku-20240307")

    assert len(result.content) == 2
    assert result.content[0].type == "text"
    assert result.content[0].text == "Let me check the weather for you."
    assert result.content[1].type == "tool_use"


def test_map_stop_reason():
    """Map OpenAI finish reasons to Anthropic stop reasons."""
    assert map_stop_reason("stop") == "end_turn"
    assert map_stop_reason("length") == "max_tokens"
    assert map_stop_reason("tool_calls") == "tool_use"
    assert map_stop_reason("content_filter") == "end_turn"
    assert map_stop_reason(None) == "end_turn"


def test_translate_response_generates_id():
    """Response ID is generated."""
    ai_message = AIMessage(
        content="Hi",
        response_metadata={"finish_reason": "stop"},
        usage_metadata={"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
    )

    result = translate_response(ai_message, model="claude-3-haiku-20240307")

    assert result.id.startswith("msg_proxy_")


def test_translate_response_missing_usage():
    """Handle missing usage metadata gracefully."""
    ai_message = AIMessage(
        content="Hi",
        response_metadata={"finish_reason": "stop"},
    )

    result = translate_response(ai_message, model="claude-3-haiku-20240307")

    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0
