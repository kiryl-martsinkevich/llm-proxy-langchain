"""Configuration models for LLM Proxy."""

from pydantic import BaseModel, Field


class BackendConfig(BaseModel):
    """Configuration for a single backend."""

    base_url: str
    api_key: str | None = None
    verify_ssl: bool = True


class ModelMapping(BaseModel):
    """Maps an Anthropic model name to a backend model."""

    backend: str
    model: str


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 8080


class ProxyConfig(BaseModel):
    """Root configuration for the proxy."""

    models: dict[str, ModelMapping]
    backends: dict[str, BackendConfig]
    server: ServerConfig = Field(default_factory=ServerConfig)
