"""Route model names to backend configurations."""

from llm_proxy.models.config import BackendConfig, ProxyConfig


class BackendNotFoundError(Exception):
    """Raised when a model cannot be mapped to a backend."""

    pass


def resolve_backend(
    anthropic_model: str, config: ProxyConfig
) -> tuple[BackendConfig, str]:
    """
    Resolve an Anthropic model name to a backend configuration.

    Returns:
        Tuple of (backend_config, actual_model_name)

    Raises:
        BackendNotFoundError: If the model is not configured.
    """
    if anthropic_model not in config.models:
        raise BackendNotFoundError(
            f"Model '{anthropic_model}' not found in configuration. "
            f"Available models: {list(config.models.keys())}"
        )

    model_mapping = config.models[anthropic_model]
    backend_name = model_mapping.backend

    if backend_name not in config.backends:
        raise BackendNotFoundError(
            f"Backend '{backend_name}' for model '{anthropic_model}' not found. "
            f"Available backends: {list(config.backends.keys())}"
        )

    backend_config = config.backends[backend_name]
    return backend_config, model_mapping.model
