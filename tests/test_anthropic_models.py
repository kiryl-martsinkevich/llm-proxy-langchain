"""Tests for Anthropic API Pydantic models."""

import pytest
from llm_proxy.models.anthropic import (
    ContentBlock,
    ImageSource,
    Message,
    MessagesRequest,
    MessagesResponse,
    TextBlock,
    ImageBlock,
    ToolUseBlock,
    ToolResultBlock,
    Tool,
    Usage,
)


def test_text_block():
    """Simple text content block."""
    block = TextBlock(type="text", text="Hello world")
    assert block.type == "text"
    assert block.text == "Hello world"


def test_image_block_base64():
    """Image block with base64 data."""
    source = ImageSource(type="base64", media_type="image/png", data="abc123")
    block = ImageBlock(type="image", source=source)
    assert block.source.type == "base64"
    assert block.source.data == "abc123"


def test_tool_use_block():
    """Tool use block from assistant."""
    block = ToolUseBlock(
        type="tool_use",
        id="tool_123",
        name="get_weather",
        input={"city": "London"},
    )
    assert block.name == "get_weather"
    assert block.input["city"] == "London"


def test_tool_result_block():
    """Tool result block from user."""
    block = ToolResultBlock(
        type="tool_result",
        tool_use_id="tool_123",
        content="The weather is sunny",
    )
    assert block.tool_use_id == "tool_123"


def test_message_simple_string():
    """Message with simple string content."""
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_message_content_blocks():
    """Message with content blocks."""
    msg = Message(
        role="user",
        content=[
            {"type": "text", "text": "What is this?"},
        ],
    )
    assert isinstance(msg.content, list)


def test_messages_request_minimal():
    """Minimal valid messages request."""
    req = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        messages=[Message(role="user", content="Hi")],
    )
    assert req.model == "claude-3-haiku-20240307"
    assert req.stream is False


def test_messages_request_full():
    """Full messages request with all options."""
    req = MessagesRequest(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        messages=[Message(role="user", content="Hi")],
        system="You are helpful",
        temperature=0.7,
        top_p=0.9,
        stop_sequences=["END"],
        stream=True,
        tools=[
            Tool(
                name="get_weather",
                description="Get weather for a city",
                input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
            )
        ],
    )
    assert req.system == "You are helpful"
    assert req.stream is True
    assert len(req.tools) == 1


def test_messages_response():
    """Messages response structure."""
    resp = MessagesResponse(
        id="msg_123",
        type="message",
        role="assistant",
        model="claude-3-haiku-20240307",
        content=[TextBlock(type="text", text="Hello!")],
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    assert resp.id == "msg_123"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 10
