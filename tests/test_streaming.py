"""Tests for streaming translation functions."""

import json

import pytest
from langchain_core.messages import AIMessageChunk

from llm_proxy.translation.streaming import (
    StreamingState,
    translate_stream_start,
    translate_stream_delta,
    translate_stream_end,
)


@pytest.fixture
def streaming_state() -> StreamingState:
    """Create a fresh streaming state for tests."""
    return StreamingState(model="claude-3-haiku-20240307")


def test_stream_start_event(streaming_state):
    """Verify message_start and content_block_start events are generated."""
    events = list(translate_stream_start(streaming_state))

    assert len(events) == 2

    # First event: message_start
    msg_start = events[0]
    assert msg_start["event"] == "message_start"
    data = json.loads(msg_start["data"])
    assert data["type"] == "message_start"
    assert data["message"]["type"] == "message"
    assert data["message"]["role"] == "assistant"
    assert data["message"]["model"] == "claude-3-haiku-20240307"
    assert data["message"]["content"] == []
    assert data["message"]["id"] == streaming_state.message_id

    # Second event: content_block_start
    block_start = events[1]
    assert block_start["event"] == "content_block_start"
    data = json.loads(block_start["data"])
    assert data["type"] == "content_block_start"
    assert data["index"] == 0
    assert data["content_block"]["type"] == "text"
    assert data["content_block"]["text"] == ""


def test_stream_delta_text(streaming_state):
    """Verify text delta translation generates content_block_delta event."""
    # Simulate stream start
    list(translate_stream_start(streaming_state))

    # Create a text chunk
    chunk = AIMessageChunk(content="Hello")
    events = list(translate_stream_delta(chunk, streaming_state))

    assert len(events) == 1
    delta_event = events[0]
    assert delta_event["event"] == "content_block_delta"

    data = json.loads(delta_event["data"])
    assert data["type"] == "content_block_delta"
    assert data["index"] == 0
    assert data["delta"]["type"] == "text_delta"
    assert data["delta"]["text"] == "Hello"

    # State should track that we have text
    assert streaming_state.has_text is True


def test_stream_delta_tool_call(streaming_state):
    """Verify tool call handling generates appropriate events."""
    # Simulate stream start
    list(translate_stream_start(streaming_state))

    # Create a tool call chunk
    chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "id": "call_123",
                "name": "get_weather",
                "args": '{"city": "London"}',
                "index": 0,
            }
        ],
    )
    events = list(translate_stream_delta(chunk, streaming_state))

    # Should have content_block_start for tool + content_block_delta for args
    assert len(events) >= 1

    # Find the tool-related events
    tool_start_events = [
        e for e in events if e["event"] == "content_block_start"
    ]
    tool_delta_events = [
        e for e in events if e["event"] == "content_block_delta"
    ]

    # Should have a tool block start
    assert len(tool_start_events) == 1
    tool_start_data = json.loads(tool_start_events[0]["data"])
    assert tool_start_data["content_block"]["type"] == "tool_use"
    assert tool_start_data["content_block"]["id"] == "call_123"
    assert tool_start_data["content_block"]["name"] == "get_weather"

    # Should have an input delta
    assert len(tool_delta_events) == 1
    tool_delta_data = json.loads(tool_delta_events[0]["data"])
    assert tool_delta_data["delta"]["type"] == "input_json_delta"


def test_stream_end(streaming_state):
    """Verify final events: content_block_stop, message_delta, message_stop."""
    # Simulate stream start and some text content
    list(translate_stream_start(streaming_state))
    streaming_state.has_text = True
    streaming_state.output_tokens = 10

    events = list(translate_stream_end(streaming_state, stop_reason="end_turn"))

    assert len(events) == 3

    # content_block_stop
    block_stop = events[0]
    assert block_stop["event"] == "content_block_stop"
    data = json.loads(block_stop["data"])
    assert data["type"] == "content_block_stop"
    assert data["index"] == 0

    # message_delta with stop_reason and usage
    msg_delta = events[1]
    assert msg_delta["event"] == "message_delta"
    data = json.loads(msg_delta["data"])
    assert data["type"] == "message_delta"
    assert data["delta"]["stop_reason"] == "end_turn"
    assert data["usage"]["output_tokens"] == 10

    # message_stop
    msg_stop = events[2]
    assert msg_stop["event"] == "message_stop"
    data = json.loads(msg_stop["data"])
    assert data["type"] == "message_stop"
