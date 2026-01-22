"""Factory for creating LangChain chat models."""

from langchain_openai import ChatOpenAI

from llm_proxy.models.config import BackendConfig


def create_chat_model(backend_config: BackendConfig, model_name: str) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance for the given backend.

    Args:
        backend_config: Backend configuration with URL and optional API key.
        model_name: The model name to use on the backend.

    Returns:
        Configured ChatOpenAI instance.
    """
    # Use a dummy key for backends that don't require auth (like Ollama)
    api_key = backend_config.api_key or "not-needed"

    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=backend_config.base_url,
    )
