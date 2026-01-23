"""Streaming translation for Anthropic SSE format."""

import json
import uuid
from dataclasses import dataclass, field
from typing import Generator

from langchain_core.messages import AIMessageChunk


def _generate_message_id() -> str:
    """Generate a unique message ID for streaming."""
    return f"msg_proxy_{uuid.uuid4().hex[:24]}"


@dataclass
class StreamingState:
    """Track state during streaming response."""

    model: str
    message_id: str = field(default_factory=_generate_message_id)
    current_block_index: int = 0
    has_text: bool = False
    tool_call_ids: dict[int, str] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0


def _make_event(event: str, data: dict) -> dict:
    """Create an SSE event dict for sse-starlette."""
    return {"event": event, "data": json.dumps(data)}


def translate_stream_start(state: StreamingState) -> Generator[dict, None, None]:
    """Generate message_start and content_block_start events.

    Args:
        state: The streaming state to use.

    Yields:
        SSE event dicts for message_start and content_block_start.
    """
    # message_start event
    yield _make_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": state.message_id,
                "type": "message",
                "role": "assistant",
                "model": state.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": state.input_tokens,
                    "output_tokens": state.output_tokens,
                },
            },
        },
    )

    # Initial content_block_start for text
    yield _make_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )


def translate_stream_delta(
    chunk: AIMessageChunk, state: StreamingState
) -> Generator[dict, None, None]:
    """Translate AIMessageChunk to content_block_delta events.

    Args:
        chunk: The LangChain AIMessageChunk from streaming.
        state: The streaming state to update.

    Yields:
        SSE event dicts for content_block_delta events.
    """
    # Handle text content
    if chunk.content:
        text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
        if text:
            state.has_text = True
            yield _make_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text},
                },
            )

    # Handle tool call chunks
    if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
        for tool_chunk in chunk.tool_call_chunks:
            tool_index = tool_chunk.get("index", 0)
            tool_id = tool_chunk.get("id")
            tool_name = tool_chunk.get("name")
            tool_args = tool_chunk.get("args", "")

            # Check if this is a new tool call
            if tool_index not in state.tool_call_ids and tool_id:
                state.tool_call_ids[tool_index] = tool_id

                # Need to close the text block and start a tool block
                # The block index for tools starts after the text block
                tool_block_index = state.current_block_index + 1 + tool_index

                # Emit content_block_start for the tool
                yield _make_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": tool_block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": tool_name or "",
                            "input": {},
                        },
                    },
                )

            # Emit input_json_delta for the arguments
            if tool_args:
                tool_block_index = state.current_block_index + 1 + tool_index
                yield _make_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": tool_block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": tool_args,
                        },
                    },
                )

    # Update token counts from usage metadata if available
    if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
        if "input_tokens" in chunk.usage_metadata:
            state.input_tokens = chunk.usage_metadata["input_tokens"]
        if "output_tokens" in chunk.usage_metadata:
            state.output_tokens = chunk.usage_metadata["output_tokens"]


def translate_stream_end(
    state: StreamingState, stop_reason: str
) -> Generator[dict, None, None]:
    """Generate final streaming events.

    Args:
        state: The streaming state.
        stop_reason: The reason for stopping (end_turn, max_tokens, tool_use).

    Yields:
        SSE event dicts for content_block_stop, message_delta, and message_stop.
    """
    # Close the text content block (index 0)
    yield _make_event(
        "content_block_stop",
        {"type": "content_block_stop", "index": 0},
    )

    # Close any tool blocks
    for tool_index in sorted(state.tool_call_ids.keys()):
        tool_block_index = state.current_block_index + 1 + tool_index
        yield _make_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": tool_block_index},
        )

    # message_delta with stop_reason and usage
    yield _make_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": state.output_tokens},
        },
    )

    # message_stop
    yield _make_event(
        "message_stop",
        {"type": "message_stop"},
    )
