# LLM Proxy

An Anthropic-compatible API proxy that enables applications built for the Anthropic Claude API to seamlessly work with OpenAI-compatible backends like Ollama, Databricks, or OpenAI itself.

## Overview

LLM Proxy acts as a translation layer between Anthropic's Messages API format and OpenAI-compatible backends. This allows you to:

- **Use local models** with Ollama during development
- **Switch backends** without changing application code
- **Deploy to enterprise services** like Databricks in production
- **Test with different models** by simply updating configuration

## Features

- **Full Anthropic API Compatibility** - Supports `/v1/messages` endpoint with streaming and non-streaming modes
- **Tool/Function Calling** - Translates Anthropic tools to OpenAI functions and back
- **Multimodal Support** - Handles text and image content blocks
- **Streaming** - Server-Sent Events (SSE) with proper Anthropic event format
- **Flexible Configuration** - YAML-based config with environment variable substitution
- **Multiple Backends** - Route different models to different backends
- **Authentication** - API key validation via `x-api-key` header or Bearer token

## Quick Start

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm-proxy-langchain

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Configuration

1. **Set up environment variables:**

```bash
cp .env.example .env
# Edit .env with your credentials
```

```bash
# Required: API key clients use to authenticate with the proxy
PROXY_API_KEY=sk-proxy-your-secret-key

# Backend credentials (as needed)
OPENAI_API_KEY=sk-...
DATABRICKS_HOST=https://your-workspace.databricks.com
DATABRICKS_TOKEN=dapi-xxxxx
```

2. **Configure model mappings** in `config.yaml`:

```yaml
# Map Anthropic model names to backend models
models:
  claude-opus-4-5-20251101:
    backend: openai
    model: gpt-5.1

  claude-sonnet-4-5-20250929:
    backend: ollama
    model: llama3.2

  claude-haiku-4-5-20251001:
    backend: openai
    model: gpt-5-mini

# Backend configurations
backends:
  ollama:
    base_url: "http://localhost:11434/v1"

  openai:
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"

  databricks:
    base_url: "${DATABRICKS_HOST}/serving-endpoints"
    api_key: "${DATABRICKS_TOKEN}"

# Server settings
server:
  host: "0.0.0.0"
  port: 8081
```

### Running the Proxy

```bash
# Using the CLI entry point
LLM_PROXY_CONFIG=config.yaml llm-proxy

# Or with Python module
LLM_PROXY_CONFIG=config.yaml python -m llm_proxy.main

# Or with uvicorn directly
LLM_PROXY_CONFIG=config.yaml uvicorn llm_proxy.main:app --host 0.0.0.0 --port 8081
```

The proxy will start on `http://localhost:8081`.

## Usage

### With the Anthropic Python SDK

```python
import anthropic

# Point the client to your proxy
client = anthropic.Anthropic(
    base_url="http://localhost:8081",
    api_key="sk-proxy-your-secret-key"  # Your PROXY_API_KEY
)

# Use exactly as you would with the Anthropic API
message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(message.content[0].text)
```

### With Streaming

```python
with client.messages.stream(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Tell me a story"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### With Tool Calling

```python
message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    tools=[{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    }],
    messages=[{"role": "user", "content": "What's the weather in Paris?"}]
)
```

### With cURL

```bash
curl -X POST http://localhost:8081/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-proxy-your-secret-key" \
  -d '{
    "model": "claude-sonnet-4-5-20250929",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Main messages endpoint (streaming and non-streaming) |
| `/v1/messages/count_tokens` | POST | Token counting (rough estimate) |
| `/api/event_logging/batch` | POST | Telemetry endpoint (accepts and discards) |
| `/health` | GET | Health check |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Application                        │
│                  (using Anthropic SDK)                       │
└─────────────────────────┬───────────────────────────────────┘
                          │ Anthropic API format
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      LLM Proxy                               │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Auth         │→ │ Translation  │→ │ Backend Router   │  │
│  │ Middleware   │  │ Layer        │  │                  │  │
│  └──────────────┘  └──────────────┘  └────────┬─────────┘  │
└──────────────────────────────────────────────────────────────┘
                          │ OpenAI API format
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────────┐
    │  Ollama  │   │  OpenAI  │   │  Databricks  │
    │ (Local)  │   │  (Cloud) │   │ (Enterprise) │
    └──────────┘   └──────────┘   └──────────────┘
```

## Project Structure

```
llm-proxy-langchain/
├── config.yaml              # Model mappings and backend configuration
├── .env.example             # Example environment variables
├── pyproject.toml           # Project metadata and dependencies
├── src/llm_proxy/
│   ├── main.py              # FastAPI application and entry point
│   ├── config.py            # Configuration loader
│   ├── auth.py              # API key authentication
│   ├── models/
│   │   ├── config.py        # Configuration Pydantic models
│   │   └── anthropic.py     # Anthropic API Pydantic models
│   ├── backends/
│   │   ├── router.py        # Model → backend routing
│   │   └── factory.py       # LangChain ChatOpenAI factory
│   ├── routes/
│   │   └── messages.py      # /v1/messages endpoint handler
│   └── translation/
│       ├── request.py       # Anthropic → LangChain translation
│       ├── response.py      # LangChain → Anthropic translation
│       └── streaming.py     # Streaming SSE translation
└── tests/                   # Test suite
```

## Development

### Running Tests

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
pytest

# Run with coverage
pytest --cov=llm_proxy
```

### Logging

Enable debug logging for troubleshooting:

```bash
LOG_LEVEL=DEBUG llm-proxy
```

Log files can be configured via:
```bash
LOG_FILE=/path/to/proxy.log llm-proxy
```

## Backend-Specific Notes

### Ollama

For local development with Ollama:

1. Install and start Ollama
2. Pull a model: `ollama pull llama3.2`
3. Configure the backend:

```yaml
backends:
  ollama:
    base_url: "http://localhost:11434/v1"
```

### Databricks

For Databricks Model Serving:

```yaml
backends:
  databricks:
    base_url: "${DATABRICKS_HOST}/serving-endpoints"
    api_key: "${DATABRICKS_TOKEN}"
```

### OpenAI

For direct OpenAI API access:

```yaml
backends:
  openai:
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `PROXY_API_KEY` | API key for client authentication | Yes |
| `LLM_PROXY_CONFIG` | Path to configuration file | Yes |
| `OPENAI_API_KEY` | OpenAI API key | If using OpenAI backend |
| `DATABRICKS_HOST` | Databricks workspace URL | If using Databricks |
| `DATABRICKS_TOKEN` | Databricks API token | If using Databricks |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, etc.) | No |
| `LOG_FILE` | Path to log file | No |

## Limitations

- Token counting is approximate (uses character-based estimation)
- Some advanced Anthropic features may not have direct equivalents in all backends
- Image support depends on backend capabilities

## License

See [LICENSE](LICENSE) for details.
