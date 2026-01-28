"""Messages endpoint handler."""

import logging
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from llm_proxy.backends.factory import create_chat_model
from llm_proxy.backends.router import BackendNotFoundError, resolve_backend
from llm_proxy.models.anthropic import MessagesRequest, MessagesResponse
from llm_proxy.models.config import ProxyConfig
from llm_proxy.translation.request import build_langchain_kwargs, translate_messages, translate_tools
from llm_proxy.translation.response import map_stop_reason, translate_response
from llm_proxy.translation.streaming import StreamingState, translate_stream_delta, translate_stream_end, translate_stream_start

logger = logging.getLogger(__name__)


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


async def _stream_response(chat_model, messages: list, kwargs: dict, model: str) -> AsyncGenerator[dict, None]:
    """Stream response from LangChain model as Anthropic SSE events."""
    state = StreamingState(model=model)
    finish_reason = None

    for event in translate_stream_start(state):
        yield event

    async for chunk in chat_model.astream(messages, **kwargs):
        if hasattr(chunk, "response_metadata") and chunk.response_metadata:
            finish_reason = chunk.response_metadata.get("finish_reason", finish_reason)
        for event in translate_stream_delta(chunk, state):
            yield event

    for event in translate_stream_end(state, map_stop_reason(finish_reason)):
        yield event


def create_messages_router(config: ProxyConfig) -> APIRouter:
    """Create messages router with configuration."""
    router = APIRouter()

    @router.post("/v1/messages")
    async def create_message(request: MessagesRequest) -> MessagesResponse:
        """Handle POST /v1/messages."""
        tool_count = len(request.tools) if request.tools else 0
        logger.info(f"Request: model={request.model} stream={request.stream} tools={tool_count}")

        try:
            backend_config, backend_model = resolve_backend(request.model, config)
        except BackendNotFoundError as e:
            return make_error_response(400, "invalid_request_error", str(e))

        logger.debug(f"Routing {request.model} -> {backend_model}")
        chat_model = create_chat_model(backend_config, backend_model)

        messages = translate_messages(request.messages, request.system, bool(request.tools))
        kwargs = build_langchain_kwargs(request)

        if request.tools:
            chat_model = chat_model.bind_tools(translate_tools(request.tools))

        if request.stream:
            return EventSourceResponse(_stream_response(chat_model, messages, kwargs, request.model))

        try:
            ai_message = await chat_model.ainvoke(messages, **kwargs)
        except Exception as e:
            logger.error(f"Backend error: {e}")
            return make_error_response(502, "api_error", f"Backend error: {e}")

        response = translate_response(ai_message, request.model)
        logger.info(f"Response: stop_reason={response.stop_reason} blocks={[b.type for b in response.content]}")
        return response

    @router.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> dict[str, Any]:
        """Return estimated token count (~4 chars per token)."""
        body = await request.body()
        return {"input_tokens": max(1, len(body) // 4)}

    @router.post("/api/event_logging/batch")
    async def event_logging_batch() -> dict[str, str]:
        """Telemetry endpoint - acknowledge and discard."""
        return {"status": "ok"}

    return router
