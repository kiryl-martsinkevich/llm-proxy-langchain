"""Messages endpoint handler."""

import json
import logging
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)

from llm_proxy.backends.factory import create_chat_model
from llm_proxy.backends.router import BackendNotFoundError, resolve_backend
from llm_proxy.models.anthropic import MessagesRequest, MessagesResponse
from llm_proxy.models.config import ProxyConfig
from llm_proxy.translation.request import (
    build_langchain_kwargs,
    translate_messages,
    translate_tools,
)
from llm_proxy.translation.response import map_stop_reason, translate_response
from llm_proxy.translation.streaming import (
    StreamingState,
    translate_stream_delta,
    translate_stream_end,
    translate_stream_start,
)


def make_error_response(
    status_code: int, error_type: str, message: str
) -> JSONResponse:
    """Create Anthropic-style error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            },
        },
    )


async def _stream_response(
    chat_model, messages: list, kwargs: dict, model: str
) -> AsyncGenerator[dict, None]:
    """Stream response from LangChain model as SSE events.

    Args:
        chat_model: The LangChain chat model to use.
        messages: Translated messages for the model.
        kwargs: Additional kwargs for the model invocation.
        model: The original model name from the request.

    Yields:
        SSE event dicts in Anthropic streaming format.
    """
    state = StreamingState(model=model)
    finish_reason = None

    # Emit start events
    for event in translate_stream_start(state):
        yield event

    # Stream chunks from the model
    async for chunk in chat_model.astream(messages, **kwargs):
        # Track finish reason from response metadata
        if hasattr(chunk, "response_metadata") and chunk.response_metadata:
            finish_reason = chunk.response_metadata.get("finish_reason", finish_reason)

        # Translate and yield delta events
        for event in translate_stream_delta(chunk, state):
            yield event

    # Determine stop reason
    stop_reason = map_stop_reason(finish_reason)

    # Emit end events
    for event in translate_stream_end(state, stop_reason):
        yield event


def create_messages_router(config: ProxyConfig) -> APIRouter:
    """Create messages router with configuration."""
    router = APIRouter()

    @router.post("/v1/messages")
    async def create_message(request: MessagesRequest) -> MessagesResponse:
        """Handle POST /v1/messages."""
        # Log incoming request
        logger.info(f">>> REQUEST model={request.model} stream={request.stream} tools={len(request.tools) if request.tools else 0} messages={len(request.messages)}")

        # Log last message content (truncated) for debugging
        if request.messages:
            last_msg = request.messages[-1]
            content_preview = str(last_msg.content)[:200] if last_msg.content else ""
            logger.info(f">>> LAST_MSG role={last_msg.role} content_preview={content_preview}...")

        # Resolve backend for the requested model
        try:
            backend_config, backend_model = resolve_backend(request.model, config)
        except BackendNotFoundError as e:
            return make_error_response(400, "invalid_request_error", str(e))

        logger.info(f"Routing to backend model={backend_model}")

        # Create LangChain model
        chat_model = create_chat_model(backend_config, backend_model)

        # Translate request to LangChain format
        has_tools = bool(request.tools)
        messages = translate_messages(request.messages, request.system, has_tools)
        kwargs = build_langchain_kwargs(request)

        # Bind tools if present
        if request.tools:
            tools = translate_tools(request.tools)
            logger.debug(f"Bound {len(tools)} tools")
            chat_model = chat_model.bind_tools(tools)

        # Handle streaming separately
        if request.stream:
            logger.info("Starting streaming response")
            return EventSourceResponse(
                _stream_response(chat_model, messages, kwargs, request.model)
            )

        # Invoke the model
        try:
            ai_message = await chat_model.ainvoke(messages, **kwargs)
            logger.debug(f"Backend response: content_len={len(ai_message.content) if ai_message.content else 0} tool_calls={len(ai_message.tool_calls) if ai_message.tool_calls else 0}")
        except Exception as e:
            logger.error(f"Backend error: {e}")
            return make_error_response(502, "api_error", f"Backend error: {str(e)}")

        # Translate response to Anthropic format
        response = translate_response(ai_message, request.model)

        # Log response details
        content_types = [block.type for block in response.content]
        logger.info(f"<<< RESPONSE stop_reason={response.stop_reason} content_types={content_types}")
        for block in response.content:
            if block.type == "tool_use":
                logger.info(f"<<< TOOL_USE name={block.name} id={block.id}")
            elif block.type == "text":
                logger.info(f"<<< TEXT preview={block.text[:200]}...")

        return response

    @router.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> dict[str, Any]:
        """Handle POST /v1/messages/count_tokens.

        Returns estimated token count. Uses rough estimate since
        Claude Code handles inaccurate counts gracefully.
        """
        body = await request.body()
        # Rough estimate: ~4 characters per token
        estimate = max(1, len(body) // 4)
        logger.debug(f"Token count estimate: {estimate}")
        return {"input_tokens": estimate}

    @router.post("/api/event_logging/batch")
    async def event_logging_batch() -> dict[str, str]:
        """Handle POST /api/event_logging/batch.

        Telemetry endpoint - acknowledge and discard.
        """
        return {"status": "ok"}

    return router
