"""Tests for FastAPI application entry point."""

import os
import tempfile

from fastapi.testclient import TestClient


def test_app_starts_with_config(monkeypatch):
    """App starts with valid config file."""
    # Create temp config file
    yaml_content = """
models:
  claude-3-haiku-20240307:
    backend: ollama
    model: llama3.2

backends:
  ollama:
    base_url: "http://localhost:11434/v1"
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(yaml_content)
        f.flush()

        try:
            # Set env var to point to our config
            monkeypatch.setenv("LLM_PROXY_CONFIG", f.name)

            # Import create_app after setting env var
            from llm_proxy.main import create_app

            app = create_app()
            client = TestClient(app)

            # Verify /health returns 200 with {"status": "ok"}
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
        finally:
            os.unlink(f.name)


def test_app_with_auth(monkeypatch):
    """App enforces auth when PROXY_API_KEY is set."""
    # Create temp config file
    yaml_content = """
models:
  claude-3-haiku-20240307:
    backend: ollama
    model: llama3.2

backends:
  ollama:
    base_url: "http://localhost:11434/v1"
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(yaml_content)
        f.flush()

        try:
            # Set env vars
            monkeypatch.setenv("LLM_PROXY_CONFIG", f.name)
            monkeypatch.setenv("PROXY_API_KEY", "test-secret-key")

            # Import create_app after setting env vars
            from llm_proxy.main import create_app

            app = create_app()
            client = TestClient(app)

            # POST /v1/messages without auth -> 401
            response = client.post("/v1/messages", json={})
            assert response.status_code == 401
            assert response.json()["error"]["type"] == "authentication_error"

            # POST /v1/messages with auth but empty body -> 422 (validation error)
            response = client.post(
                "/v1/messages",
                json={},
                headers={"x-api-key": "test-secret-key"},
            )
            assert response.status_code == 422  # Validation error, not auth
        finally:
            os.unlink(f.name)
