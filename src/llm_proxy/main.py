"""FastAPI application entry point."""

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from llm_proxy.auth import APIKeyAuth
from llm_proxy.config import load_config
from llm_proxy.routes.messages import create_messages_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Loads config from LLM_PROXY_CONFIG env var or "config.yaml".
    Adds APIKeyAuth middleware using PROXY_API_KEY env var.
    Registers messages router and health endpoint.

    Returns:
        Configured FastAPI application.
    """
    # Load configuration
    config_path = Path(os.environ.get("LLM_PROXY_CONFIG", "config.yaml"))
    config = load_config(config_path)

    # Create FastAPI app
    app = FastAPI(
        title="LLM Proxy",
        description="Anthropic-compatible API proxy for OpenAI-compatible backends",
        version="0.1.0",
    )

    # Add auth middleware
    api_key = os.environ.get("PROXY_API_KEY")
    auth = APIKeyAuth(api_key=api_key)
    app.middleware("http")(auth)

    # Register messages router
    messages_router = create_messages_router(config)
    app.include_router(messages_router)

    # Add health endpoint
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


# Module-level app for uvicorn (e.g., uvicorn llm_proxy.main:app)
# Only created if config file exists to allow safe import
app: FastAPI | None = None
try:
    app = create_app()
except Exception:
    # Config not available at import time - use create_app() directly
    pass


def main():
    """Run the application with uvicorn."""
    # Load .env file if it exists
    load_dotenv()

    # Load config to get server settings
    config_path = Path(os.environ.get("LLM_PROXY_CONFIG", "config.yaml"))
    config = load_config(config_path)

    # Use config values, with env var overrides
    host = os.environ.get("LLM_PROXY_HOST", config.server.get("host", "127.0.0.1"))
    port = int(os.environ.get("LLM_PROXY_PORT", config.server.get("port", 8000)))
    uvicorn.run(create_app, host=host, port=port, reload=False, factory=True)


if __name__ == "__main__":
    main()
