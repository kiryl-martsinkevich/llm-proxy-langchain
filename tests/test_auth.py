"""Tests for API key authentication."""

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from llm_proxy.auth import APIKeyAuth, get_api_key


@pytest.fixture
def app_with_auth():
    """FastAPI app with auth middleware."""
    app = FastAPI()
    auth = APIKeyAuth(api_key="test-secret-key")

    @app.get("/protected")
    async def protected(request: Request):
        key = get_api_key(request)
        return {"key_preview": key[:8] + "..."}

    app.middleware("http")(auth)
    return app


def test_valid_api_key_header(app_with_auth):
    """Valid x-api-key header passes."""
    client = TestClient(app_with_auth)
    response = client.get(
        "/protected",
        headers={"x-api-key": "test-secret-key"},
    )
    assert response.status_code == 200


def test_valid_bearer_token(app_with_auth):
    """Valid Authorization Bearer token passes."""
    client = TestClient(app_with_auth)
    response = client.get(
        "/protected",
        headers={"Authorization": "Bearer test-secret-key"},
    )
    assert response.status_code == 200


def test_missing_api_key(app_with_auth):
    """Missing API key returns 401."""
    client = TestClient(app_with_auth)
    response = client.get("/protected")
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"


def test_invalid_api_key(app_with_auth):
    """Invalid API key returns 401."""
    client = TestClient(app_with_auth)
    response = client.get(
        "/protected",
        headers={"x-api-key": "wrong-key"},
    )
    assert response.status_code == 401


def test_auth_disabled():
    """No auth when api_key is None."""
    app = FastAPI()
    auth = APIKeyAuth(api_key=None)

    @app.get("/open")
    async def open_endpoint():
        return {"status": "ok"}

    app.middleware("http")(auth)
    client = TestClient(app)

    response = client.get("/open")
    assert response.status_code == 200
