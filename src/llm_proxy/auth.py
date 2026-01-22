"""API key authentication middleware."""

from fastapi import Request, Response
from starlette.responses import JSONResponse


def get_api_key(request: Request) -> str | None:
    """Extract API key from request headers."""
    # Try x-api-key header first (Anthropic style)
    api_key = request.headers.get("x-api-key")
    if api_key:
        return api_key

    # Try Authorization Bearer token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]

    return None


def make_auth_error_response() -> JSONResponse:
    """Create Anthropic-style authentication error response."""
    return JSONResponse(
        status_code=401,
        content={
            "type": "error",
            "error": {
                "type": "authentication_error",
                "message": "Invalid or missing API key",
            },
        },
    )


class APIKeyAuth:
    """API key authentication middleware."""

    def __init__(self, api_key: str | None):
        """
        Initialize auth middleware.

        Args:
            api_key: Expected API key, or None to disable auth.
        """
        self.api_key = api_key

    async def __call__(
        self, request: Request, call_next
    ) -> Response:
        """Check API key on each request."""
        # Skip auth if no key configured
        if self.api_key is None:
            return await call_next(request)

        # Extract and validate key
        provided_key = get_api_key(request)
        if provided_key != self.api_key:
            return make_auth_error_response()

        return await call_next(request)
