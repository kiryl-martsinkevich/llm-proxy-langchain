"""Messages endpoint handler."""

from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

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
        # Resolve backend for the requested model
        try:
            backend_config, backend_model = resolve_backend(request.model, config)
        except BackendNotFoundError as e:
            return make_error_response(400, "invalid_request_error", str(e))

        # Create LangChain model
        chat_model = create_chat_model(backend_config, backend_model)

        # Translate request to LangChain format
        messages = translate_messages(request.messages, request.system)
        kwargs = build_langchain_kwargs(request)

        # Bind tools if present
        if request.tools:
            tools = translate_tools(request.tools)
            chat_model = chat_model.bind_tools(tools)

        # Handle streaming separately
        if request.stream:
            return EventSourceResponse(
                _stream_response(chat_model, messages, kwargs, request.model)
            )

        # Invoke the model
        try:
            ai_message = await chat_model.ainvoke(messages, **kwargs)
        except Exception as e:
            return make_error_response(502, "api_error", f"Backend error: {str(e)}")

        # Translate response to Anthropic format
        response = translate_response(ai_message, request.model)
        return response

    return router
