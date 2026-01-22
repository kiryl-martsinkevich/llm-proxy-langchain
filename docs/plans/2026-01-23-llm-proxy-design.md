# LLM Proxy Service Design

An LLM API proxy that exposes Anthropic-compatible endpoints for serving models hosted on OpenAI-compatible backends (Databricks, Ollama, etc.).

## Goals

- **Local development**: Run Anthropic SDK code against local models (Ollama) without code changes
- **Enterprise deployment**: Serve internal models (Databricks) through a standardized Anthropic-like interface
- **Full API compatibility**: Messages, streaming, tools, vision, system prompts, multi-turn

## Tech Stack

- Python with `uv` for package management
- FastAPI for the web server
- LangChain (`langchain-openai`) as the translation layer
- Pydantic for schema validation
- YAML config file + environment variables for configuration

## Architecture

```
┌─────────────────┐     ┌─────────────────────────────────────┐     ┌──────────────┐
│  Anthropic SDK  │────▶│         LLM Proxy Service           │────▶│   Ollama     │
│  (your code)    │     │                                     │     └──────────────┘
└─────────────────┘     │  ┌─────────────┐  ┌──────────────┐  │     ┌──────────────┐
                        │  │ FastAPI     │  │ LangChain    │  │────▶│  Databricks  │
                        │  │ Endpoints   │──│ Translation  │  │     └──────────────┘
                        │  └─────────────┘  └──────────────┘  │     ┌──────────────┐
                        └─────────────────────────────────────┘     └──────────────┘
```

### Request Flow

1. Client sends Anthropic-format request to `POST /v1/messages`
2. FastAPI validates the request against Anthropic's schema
3. Router looks up the requested model in config, finds the backend
4. Translation layer converts Anthropic request → LangChain `ChatOpenAI` call
5. LangChain calls the OpenAI-compatible backend
6. Response is translated back to Anthropic format and returned

### Key Components

- **FastAPI app** - Exposes `/v1/messages` endpoint (streaming and non-streaming)
- **Model router** - Maps model names to backend configurations
- **Request translator** - Anthropic schema → LangChain messages
- **Response translator** - LangChain output → Anthropic schema
- **Config loader** - Reads YAML config + env var overrides

## Project Structure

```
llm-proxy-langchain/
├── pyproject.toml              # uv project config, dependencies
├── config.yaml                 # Model mappings and backend configs
├── .env.example                # Template for secrets
│
├── src/
│   └── llm_proxy/
│       ├── __init__.py
│       ├── main.py             # FastAPI app entry point
│       ├── config.py           # Config loading (YAML + env vars)
│       ├── auth.py             # API key validation middleware
│       │
│       ├── routes/
│       │   ├── __init__.py
│       │   └── messages.py     # POST /v1/messages endpoint
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── anthropic.py    # Pydantic models for Anthropic API
│       │   └── config.py       # Config schema models
│       │
│       ├── translation/
│       │   ├── __init__.py
│       │   ├── request.py      # Anthropic → LangChain
│       │   └── response.py     # LangChain → Anthropic
│       │
│       └── backends/
│           ├── __init__.py
│           ├── router.py       # Model name → backend resolver
│           └── factory.py      # Creates LangChain ChatOpenAI instances
│
└── tests/
    ├── conftest.py
    ├── test_translation.py
    └── test_endpoints.py
```

### Dependencies

- `fastapi` + `uvicorn` - Web server
- `langchain-openai` - OpenAI-compatible backend support
- `pydantic` + `pydantic-settings` - Schema validation and config
- `pyyaml` - Config file parsing
- `sse-starlette` - Server-sent events for streaming

## Configuration

### config.yaml

```yaml
# Model mappings - map Anthropic model names to backends
models:
  claude-3-opus-20240229:
    backend: databricks
    model: databricks-meta-llama-3-1-70b-instruct

  claude-3-sonnet-20240229:
    backend: databricks
    model: databricks-meta-llama-3-1-70b-instruct

  claude-3-haiku-20240307:
    backend: ollama
    model: llama3.2

# Backend configurations
backends:
  ollama:
    base_url: "http://localhost:11434/v1"
    # api_key not required for Ollama

  databricks:
    base_url: "${DATABRICKS_HOST}/serving-endpoints"
    api_key: "${DATABRICKS_TOKEN}"  # Resolved from env var

# Server settings
server:
  host: "0.0.0.0"
  port: 8080
```

### Environment Variables

```bash
# Proxy API key - clients use this to authenticate
PROXY_API_KEY=sk-proxy-your-secret-key

# Backend credentials
DATABRICKS_HOST=https://your-workspace.databricks.com
DATABRICKS_TOKEN=dapi-xxxxx
```

### Behaviors

- `${VAR}` syntax in YAML gets resolved from environment variables
- Missing env vars for required secrets cause startup failure with clear error
- Model names are exact matches (client asks for `claude-3-haiku-20240307`, config must have that key)

## Request Translation (Anthropic → LangChain)

### Example Anthropic Request

```json
{
  "model": "claude-3-haiku-20240307",
  "max_tokens": 1024,
  "system": "You are a helpful assistant.",
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": [
      {"type": "text", "text": "What's in this image?"},
      {"type": "image", "source": {"type": "base64", "data": "..."}}
    ]}
  ],
  "tools": [{"name": "get_weather", "description": "...", "input_schema": {...}}]
}
```

### Translation Mapping

| Anthropic | LangChain |
|-----------|-----------|
| `system` | `SystemMessage(content=...)` |
| `messages[role=user]` | `HumanMessage(content=...)` |
| `messages[role=assistant]` | `AIMessage(content=...)` |
| `content` (text) | String content |
| `content` (image) | LangChain image content block |
| `tools` | `bind_tools()` with converted schema |
| `max_tokens` | `max_tokens` parameter |
| `temperature` | `temperature` parameter |
| `top_p` | `top_p` parameter |
| `stop_sequences` | `stop` parameter |

### Tool Schema Conversion

Anthropic uses `input_schema`, OpenAI uses `parameters` - the translator handles this mapping. The schemas themselves are JSON Schema, so they're compatible.

## Response Translation (LangChain → Anthropic)

### LangChain Response

```python
AIMessage(
    content="Here's what I see in the image...",
    tool_calls=[{"name": "get_weather", "args": {"city": "London"}}],
    usage_metadata={"input_tokens": 150, "output_tokens": 50}
)
```

### Anthropic Response Format

```json
{
  "id": "msg_proxy_abc123",
  "type": "message",
  "role": "assistant",
  "model": "claude-3-haiku-20240307",
  "content": [
    {"type": "text", "text": "Here's what I see..."}
  ],
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 150, "output_tokens": 50}
}
```

### Translation Mapping

| LangChain | Anthropic |
|-----------|-----------|
| `content` (string) | `content: [{"type": "text", "text": ...}]` |
| `tool_calls` | `content: [{"type": "tool_use", "id": ..., "name": ..., "input": ...}]` |
| `usage_metadata` | `usage: {input_tokens, output_tokens}` |
| `response_metadata.finish_reason` | `stop_reason` (mapped: `stop`→`end_turn`, `tool_calls`→`tool_use`) |

### Generated Fields

- `id`: Generated as `msg_proxy_{uuid}`
- `type`: Always `"message"`
- `model`: Echo back the requested model name (not the backend model)

## Streaming Support

### Anthropic Streaming Format (SSE)

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_...","model":"claude-3-haiku-20240307",...}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}
```

### Implementation

1. Client sends `"stream": true` in request
2. Endpoint returns `EventSourceResponse` (from `sse-starlette`)
3. Call LangChain with `.astream()` for async token iteration
4. Wrap each chunk in Anthropic's event structure
5. Track content block index for tool calls (each tool is a separate block)

### Stream Translation

- First chunk → emit `message_start` + `content_block_start`
- Each token → emit `content_block_delta`
- Tool call detected → emit `content_block_stop` for text, then `content_block_start` for tool
- Stream ends → emit `content_block_stop`, `message_delta` (with usage), `message_stop`

## Error Handling

### Anthropic Error Format

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "model: Unknown model 'claude-unknown'"
  }
}
```

### Error Mapping

| Scenario | Anthropic Error Type | HTTP Status |
|----------|---------------------|-------------|
| Invalid API key | `authentication_error` | 401 |
| Unknown model name | `invalid_request_error` | 400 |
| Malformed request body | `invalid_request_error` | 400 |
| Backend unreachable | `api_error` | 502 |
| Backend timeout | `api_error` | 504 |
| Backend returns error | `api_error` | 502 |
| Rate limited by backend | `rate_limit_error` | 429 |

### Implementation

- FastAPI exception handlers catch errors and format as Anthropic responses
- Pydantic validation errors → `invalid_request_error` with field details
- LangChain/httpx exceptions → wrapped as `api_error` with backend context
- Sensitive backend details (URLs, tokens) are not leaked to clients

### Logging

- Log all requests with model, backend, latency, token counts
- Log errors with full context (including backend response) for debugging
- Use structured logging (JSON) for production observability

## Testing Strategy

### Unit Tests (`tests/test_translation.py`)

- Test request translation: Anthropic messages → LangChain messages
- Test response translation: LangChain output → Anthropic format
- Test tool schema conversion both directions
- Test multimodal content (images) translation
- Test edge cases: empty messages, missing fields, max values

### Integration Tests (`tests/test_endpoints.py`)

- Use FastAPI's `TestClient` with mocked LangChain backends
- Test `/v1/messages` non-streaming happy path
- Test `/v1/messages` streaming (collect SSE events)
- Test authentication (valid key, invalid key, missing key)
- Test unknown model returns proper error
- Test backend failure returns proper error format

### Manual/E2E Testing

- Run proxy against real Ollama instance locally
- Use official Anthropic Python SDK as client
- Verify streaming works in real terminal
- Test with actual tool calling round-trip

### Test Fixtures

- Sample Anthropic requests (simple, multimodal, tools)
- Sample LangChain responses (text, tool calls, errors)
- Mock `ChatOpenAI` that returns predictable responses

## Authentication

- Simple shared key model
- Clients authenticate with `PROXY_API_KEY`
- Proxy manages separate credentials for each backend in config
- API key passed in `x-api-key` header or `Authorization: Bearer` header (matching Anthropic SDK behavior)
