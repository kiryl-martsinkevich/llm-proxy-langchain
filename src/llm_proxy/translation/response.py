"""Translate LangChain responses to Anthropic format."""

import uuid
from typing import Literal

from langchain_core.messages import AIMessage

from llm_proxy.models.anthropic import MessagesResponse, TextBlock, ToolUseBlock, Usage

StopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]

STOP_REASON_MAP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "content_filter": "end_turn"}


def map_stop_reason(finish_reason: str | None) -> StopReason:
    """Map OpenAI finish reason to Anthropic stop reason."""
    return STOP_REASON_MAP.get(finish_reason or "", "end_turn")


def generate_message_id() -> str:
    """Generate a unique message ID."""
    return f"msg_proxy_{uuid.uuid4().hex[:24]}"


def translate_response(ai_message: AIMessage, model: str) -> MessagesResponse:
    """Translate LangChain AIMessage to Anthropic MessagesResponse."""
    content: list[TextBlock | ToolUseBlock] = []

    if ai_message.content:
        text = ai_message.content if isinstance(ai_message.content, str) else str(ai_message.content)
        if text:
            content.append(TextBlock(type="text", text=text))

    if ai_message.tool_calls:
        for tc in ai_message.tool_calls:
            content.append(ToolUseBlock(type="tool_use", id=tc["id"], name=tc["name"], input=tc["args"]))

    usage_meta = getattr(ai_message, "usage_metadata", None) or {}
    response_meta = getattr(ai_message, "response_metadata", None) or {}

    return MessagesResponse(
        id=generate_message_id(),
        type="message",
        role="assistant",
        model=model,
        content=content,
        stop_reason=map_stop_reason(response_meta.get("finish_reason")),
        usage=Usage(input_tokens=usage_meta.get("input_tokens", 0), output_tokens=usage_meta.get("output_tokens", 0)),
    )
