"""Pydantic models for Anthropic API request/response schemas."""

from typing import Any, Literal

from pydantic import BaseModel


# Content block types
class TextBlock(BaseModel):
    """Text content block."""

    type: Literal["text"]
    text: str


class ImageSource(BaseModel):
    """Image source for base64 or URL."""

    type: Literal["base64", "url"]
    media_type: str | None = None
    data: str | None = None
    url: str | None = None


class ImageBlock(BaseModel):
    """Image content block."""

    type: Literal["image"]
    source: ImageSource


class ToolUseBlock(BaseModel):
    """Tool use block from assistant."""

    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(BaseModel):
    """Tool result block from user."""

    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[TextBlock | ImageBlock]
    is_error: bool = False


ContentBlock = TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock


# Message types
class Message(BaseModel):
    """A message in the conversation."""

    role: Literal["user", "assistant"]
    content: str | list[dict[str, Any]]


# Tool definition
class Tool(BaseModel):
    """Tool definition for function calling."""

    name: str
    description: str
    input_schema: dict[str, Any]


# Request/Response
class MessagesRequest(BaseModel):
    """Request body for POST /v1/messages."""

    model: str
    max_tokens: int
    messages: list[Message]
    system: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    tools: list[Tool] | None = None


class Usage(BaseModel):
    """Token usage information."""

    input_tokens: int
    output_tokens: int


class MessagesResponse(BaseModel):
    """Response body for POST /v1/messages."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[TextBlock | ToolUseBlock]
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    usage: Usage
