"""Streaming translation for Anthropic SSE format."""

import json
from dataclasses import dataclass, field
from typing import Generator

from langchain_core.messages import AIMessageChunk

from llm_proxy.translation.response import generate_message_id


@dataclass
class StreamingState:
    """Track state during streaming response."""

    model: str
    message_id: str = field(default_factory=generate_message_id)
    current_block_index: int = 0
    has_text: bool = False
    tool_call_ids: dict[int, str] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0


def _event(event_type: str, data: dict) -> dict:
    """Create an SSE event dict."""
    return {"event": event_type, "data": json.dumps(data)}


def translate_stream_start(state: StreamingState) -> Generator[dict, None, None]:
    """Generate message_start and initial content_block_start events."""
    yield _event("message_start", {
        "type": "message_start",
        "message": {
            "id": state.message_id,
            "type": "message",
            "role": "assistant",
            "model": state.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": state.input_tokens, "output_tokens": state.output_tokens},
        },
    })
    yield _event("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })


def translate_stream_delta(chunk: AIMessageChunk, state: StreamingState) -> Generator[dict, None, None]:
    """Translate AIMessageChunk to content_block_delta events."""
    # Handle text content
    if chunk.content:
        text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
        if text:
            state.has_text = True
            yield _event("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            })

    # Handle tool call chunks
    if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
        for tc in chunk.tool_call_chunks:
            tool_index = tc.get("index", 0)
            tool_id = tc.get("id")
            tool_block_index = state.current_block_index + 1 + tool_index

            # New tool call - emit content_block_start
            if tool_index not in state.tool_call_ids and tool_id:
                state.tool_call_ids[tool_index] = tool_id
                yield _event("content_block_start", {
                    "type": "content_block_start",
                    "index": tool_block_index,
                    "content_block": {"type": "tool_use", "id": tool_id, "name": tc.get("name") or "", "input": {}},
                })

            # Emit input_json_delta for arguments
            if tc.get("args"):
                yield _event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": tool_block_index,
                    "delta": {"type": "input_json_delta", "partial_json": tc["args"]},
                })

    # Update token counts
    if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
        state.input_tokens = chunk.usage_metadata.get("input_tokens", state.input_tokens)
        state.output_tokens = chunk.usage_metadata.get("output_tokens", state.output_tokens)


def translate_stream_end(state: StreamingState, stop_reason: str) -> Generator[dict, None, None]:
    """Generate final streaming events."""
    # Close text block
    yield _event("content_block_stop", {"type": "content_block_stop", "index": 0})

    # Close tool blocks
    for tool_index in sorted(state.tool_call_ids.keys()):
        yield _event("content_block_stop", {
            "type": "content_block_stop",
            "index": state.current_block_index + 1 + tool_index,
        })

    yield _event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": state.output_tokens},
    })
    yield _event("message_stop", {"type": "message_stop"})
