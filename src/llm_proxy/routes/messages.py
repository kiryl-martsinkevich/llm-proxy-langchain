"""Messages endpoint handler."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from llm_proxy.backends.factory import create_chat_model
from llm_proxy.backends.router import BackendNotFoundError, resolve_backend
from llm_proxy.models.anthropic import MessagesRequest, MessagesResponse
from llm_proxy.models.config import ProxyConfig
from llm_proxy.translation.request import (
    build_langchain_kwargs,
    translate_messages,
    translate_tools,
)
from llm_proxy.translation.response import translate_response


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
            # TODO: Implement streaming in Task 10
            return make_error_response(
                400, "invalid_request_error", "Streaming not yet implemented"
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
