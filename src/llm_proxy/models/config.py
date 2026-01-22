"""Configuration models for LLM Proxy."""

from pydantic import BaseModel


class BackendConfig(BaseModel):
    """Configuration for a single backend."""

    base_url: str
    api_key: str | None = None


class ModelMapping(BaseModel):
    """Maps an Anthropic model name to a backend model."""

    backend: str
    model: str


class ServerConfig(BaseModel):
    """Server configuration with defaults."""

    host: str = "0.0.0.0"
    port: int = 8080


class ProxyConfig(BaseModel):
    """Root configuration for the proxy."""

    models: dict[str, ModelMapping]
    backends: dict[str, BackendConfig]
    server: dict[str, str | int] = {"host": "0.0.0.0", "port": 8080}
