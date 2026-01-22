# LLM Proxy Service Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an Anthropic-compatible API proxy that routes requests to OpenAI-compatible backends (Ollama, Databricks) using LangChain.

**Architecture:** FastAPI server exposes `/v1/messages` endpoint. Requests are validated against Anthropic schema, translated to LangChain messages, sent to configured backend via `ChatOpenAI`, and responses translated back to Anthropic format.

**Tech Stack:** Python 3.11+, uv, FastAPI, LangChain (langchain-openai), Pydantic, PyYAML, sse-starlette

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/llm_proxy/__init__.py`
- Create: `.python-version`

**Step 1: Initialize uv project**

Run: `uv init --lib --name llm-proxy`
Expected: Creates pyproject.toml

**Step 2: Set Python version**

Create `.python-version`:
```
3.11
```

**Step 3: Update pyproject.toml with dependencies**

Replace `pyproject.toml`:
```toml
[project]
name = "llm-proxy"
version = "0.1.0"
description = "Anthropic-compatible API proxy for OpenAI-compatible backends"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "langchain-openai>=0.3.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.6.0",
    "pyyaml>=6.0.2",
    "sse-starlette>=2.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.25.0",
    "httpx>=0.28.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/llm_proxy"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Step 4: Create package init**

Create `src/llm_proxy/__init__.py`:
```python
"""LLM Proxy - Anthropic-compatible API proxy for OpenAI backends."""

__version__ = "0.1.0"
```

**Step 5: Create tests directory**

Run: `mkdir -p tests && touch tests/__init__.py`

**Step 6: Install dependencies**

Run: `uv sync --all-extras`
Expected: Creates uv.lock, installs all dependencies

**Step 7: Verify installation**

Run: `uv run python -c "import fastapi; import langchain_openai; print('OK')"`
Expected: `OK`

**Step 8: Commit**

```bash
git add pyproject.toml .python-version src/ tests/ uv.lock
git commit -m "feat: initialize project with uv and dependencies"
```

---

## Task 2: Configuration Schema

**Files:**
- Create: `src/llm_proxy/models/__init__.py`
- Create: `src/llm_proxy/models/config.py`
- Create: `tests/test_config_models.py`

**Step 1: Create models package**

Create `src/llm_proxy/models/__init__.py`:
```python
"""Pydantic models for LLM Proxy."""
```

**Step 2: Write failing test for config models**

Create `tests/test_config_models.py`:
```python
"""Tests for configuration models."""

import pytest
from llm_proxy.models.config import BackendConfig, ModelMapping, ProxyConfig


def test_backend_config_minimal():
    """Backend config with just base_url."""
    config = BackendConfig(base_url="http://localhost:11434/v1")
    assert config.base_url == "http://localhost:11434/v1"
    assert config.api_key is None


def test_backend_config_with_api_key():
    """Backend config with API key."""
    config = BackendConfig(
        base_url="https://workspace.databricks.com/serving-endpoints",
        api_key="dapi-xxxxx",
    )
    assert config.api_key == "dapi-xxxxx"


def test_model_mapping():
    """Model mapping links Anthropic model to backend."""
    mapping = ModelMapping(backend="ollama", model="llama3.2")
    assert mapping.backend == "ollama"
    assert mapping.model == "llama3.2"


def test_proxy_config_complete():
    """Full proxy configuration."""
    config = ProxyConfig(
        models={
            "claude-3-haiku-20240307": ModelMapping(backend="ollama", model="llama3.2"),
        },
        backends={
            "ollama": BackendConfig(base_url="http://localhost:11434/v1"),
        },
        server={"host": "0.0.0.0", "port": 8080},
    )
    assert "claude-3-haiku-20240307" in config.models
    assert "ollama" in config.backends
    assert config.server["port"] == 8080


def test_proxy_config_default_server():
    """Server config has sensible defaults."""
    config = ProxyConfig(
        models={},
        backends={},
    )
    assert config.server["host"] == "0.0.0.0"
    assert config.server["port"] == 8080
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_config_models.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'llm_proxy.models.config'"

**Step 4: Implement config models**

Create `src/llm_proxy/models/config.py`:
```python
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
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_config_models.py -v`
Expected: All 5 tests PASS

**Step 6: Commit**

```bash
git add src/llm_proxy/models/ tests/test_config_models.py
git commit -m "feat: add configuration Pydantic models"
```

---

## Task 3: Config Loader with Environment Variable Substitution

**Files:**
- Create: `src/llm_proxy/config.py`
- Create: `tests/test_config_loader.py`
- Create: `config.yaml`
- Create: `.env.example`

**Step 1: Write failing test for config loader**

Create `tests/test_config_loader.py`:
```python
"""Tests for configuration loading."""

import os
import tempfile
from pathlib import Path

import pytest
from llm_proxy.config import load_config, substitute_env_vars


def test_substitute_env_vars_simple(monkeypatch):
    """Substitute ${VAR} with environment variable."""
    monkeypatch.setenv("TEST_VAR", "hello")
    result = substitute_env_vars("prefix_${TEST_VAR}_suffix")
    assert result == "prefix_hello_suffix"


def test_substitute_env_vars_multiple(monkeypatch):
    """Substitute multiple variables."""
    monkeypatch.setenv("HOST", "localhost")
    monkeypatch.setenv("PORT", "8080")
    result = substitute_env_vars("http://${HOST}:${PORT}")
    assert result == "http://localhost:8080"


def test_substitute_env_vars_missing_raises():
    """Missing env var raises error."""
    with pytest.raises(ValueError, match="Environment variable 'MISSING_VAR' not set"):
        substitute_env_vars("${MISSING_VAR}")


def test_substitute_env_vars_no_vars():
    """String without vars returned unchanged."""
    result = substitute_env_vars("plain string")
    assert result == "plain string"


def test_load_config_from_yaml(monkeypatch):
    """Load config from YAML file with env substitution."""
    monkeypatch.setenv("TEST_API_KEY", "secret123")

    yaml_content = """
models:
  claude-3-haiku-20240307:
    backend: ollama
    model: llama3.2

backends:
  ollama:
    base_url: "http://localhost:11434/v1"
  databricks:
    base_url: "https://example.databricks.com"
    api_key: "${TEST_API_KEY}"

server:
  host: "127.0.0.1"
  port: 9000
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()

        config = load_config(Path(f.name))

        assert config.models["claude-3-haiku-20240307"].backend == "ollama"
        assert config.backends["databricks"].api_key == "secret123"
        assert config.server["port"] == 9000

    os.unlink(f.name)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_loader.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'llm_proxy.config'"

**Step 3: Implement config loader**

Create `src/llm_proxy/config.py`:
```python
"""Configuration loading with environment variable substitution."""

import os
import re
from pathlib import Path

import yaml

from llm_proxy.models.config import BackendConfig, ModelMapping, ProxyConfig

ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def substitute_env_vars(value: str) -> str:
    """Replace ${VAR} patterns with environment variable values."""
    if not isinstance(value, str):
        return value

    def replace_match(match: re.Match) -> str:
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            raise ValueError(f"Environment variable '{var_name}' not set")
        return env_value

    return ENV_VAR_PATTERN.sub(replace_match, value)


def substitute_env_vars_recursive(obj: dict | list | str) -> dict | list | str:
    """Recursively substitute env vars in a nested structure."""
    if isinstance(obj, dict):
        return {k: substitute_env_vars_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [substitute_env_vars_recursive(item) for item in obj]
    elif isinstance(obj, str):
        return substitute_env_vars(obj)
    return obj


def load_config(path: Path) -> ProxyConfig:
    """Load configuration from YAML file with env var substitution."""
    with open(path) as f:
        raw_config = yaml.safe_load(f)

    # Substitute environment variables
    config_data = substitute_env_vars_recursive(raw_config)

    return ProxyConfig(**config_data)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config_loader.py -v`
Expected: All 5 tests PASS

**Step 5: Create example config file**

Create `config.yaml`:
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

  databricks:
    base_url: "${DATABRICKS_HOST}/serving-endpoints"
    api_key: "${DATABRICKS_TOKEN}"

# Server settings
server:
  host: "0.0.0.0"
  port: 8080
```

**Step 6: Create .env.example**

Create `.env.example`:
```bash
# Proxy API key - clients use this to authenticate
PROXY_API_KEY=sk-proxy-your-secret-key

# Backend credentials
DATABRICKS_HOST=https://your-workspace.databricks.com
DATABRICKS_TOKEN=dapi-xxxxx
```

**Step 7: Commit**

```bash
git add src/llm_proxy/config.py tests/test_config_loader.py config.yaml .env.example
git commit -m "feat: add config loader with env var substitution"
```

---

## Task 4: Anthropic Request/Response Models

**Files:**
- Create: `src/llm_proxy/models/anthropic.py`
- Create: `tests/test_anthropic_models.py`

**Step 1: Write failing test for Anthropic models**

Create `tests/test_anthropic_models.py`:
```python
"""Tests for Anthropic API Pydantic models."""

import pytest
from llm_proxy.models.anthropic import (
    ContentBlock,
    ImageSource,
    Message,
    MessagesRequest,
    MessagesResponse,
    TextBlock,
    ImageBlock,
    ToolUseBlock,
    ToolResultBlock,
    Tool,
    Usage,
)


def test_text_block():
    """Simple text content block."""
    block = TextBlock(type="text", text="Hello world")
    assert block.type == "text"
    assert block.text == "Hello world"


def test_image_block_base64():
    """Image block with base64 data."""
    source = ImageSource(type="base64", media_type="image/png", data="abc123")
    block = ImageBlock(type="image", source=source)
    assert block.source.type == "base64"
    assert block.source.data == "abc123"


def test_tool_use_block():
    """Tool use block from assistant."""
    block = ToolUseBlock(
        type="tool_use",
        id="tool_123",
        name="get_weather",
        input={"city": "London"},
    )
    assert block.name == "get_weather"
    assert block.input["city"] == "London"


def test_tool_result_block():
    """Tool result block from user."""
    block = ToolResultBlock(
        type="tool_result",
        tool_use_id="tool_123",
        content="The weather is sunny",
    )
    assert block.tool_use_id == "tool_123"


def test_message_simple_string():
    """Message with simple string content."""
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_message_content_blocks():
    """Message with content blocks."""
    msg = Message(
        role="user",
        content=[
            {"type": "text", "text": "What is this?"},
        ],
    )
    assert isinstance(msg.content, list)


def test_messages_request_minimal():
    """Minimal valid messages request."""
    req = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        messages=[Message(role="user", content="Hi")],
    )
    assert req.model == "claude-3-haiku-20240307"
    assert req.stream is False


def test_messages_request_full():
    """Full messages request with all options."""
    req = MessagesRequest(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        messages=[Message(role="user", content="Hi")],
        system="You are helpful",
        temperature=0.7,
        top_p=0.9,
        stop_sequences=["END"],
        stream=True,
        tools=[
            Tool(
                name="get_weather",
                description="Get weather for a city",
                input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
            )
        ],
    )
    assert req.system == "You are helpful"
    assert req.stream is True
    assert len(req.tools) == 1


def test_messages_response():
    """Messages response structure."""
    resp = MessagesResponse(
        id="msg_123",
        type="message",
        role="assistant",
        model="claude-3-haiku-20240307",
        content=[TextBlock(type="text", text="Hello!")],
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    assert resp.id == "msg_123"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 10
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_anthropic_models.py -v`
Expected: FAIL with "cannot import name 'ContentBlock'"

**Step 3: Implement Anthropic models**

Create `src/llm_proxy/models/anthropic.py`:
```python
"""Pydantic models for Anthropic API request/response schemas."""

from typing import Any, Literal

from pydantic import BaseModel, Field


# Content block types
class TextBlock(BaseModel):
    """Text content block."""

    type: Literal["text"]
    text: str


class ImageSource(BaseModel):
    """Image source for base64 or URL."""

    type: Literal["base64", "url"]
    media_type: str | None = None
    data: str | None = None
    url: str | None = None


class ImageBlock(BaseModel):
    """Image content block."""

    type: Literal["image"]
    source: ImageSource


class ToolUseBlock(BaseModel):
    """Tool use block from assistant."""

    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(BaseModel):
    """Tool result block from user."""

    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[TextBlock | ImageBlock]
    is_error: bool = False


ContentBlock = TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock


# Message types
class Message(BaseModel):
    """A message in the conversation."""

    role: Literal["user", "assistant"]
    content: str | list[dict[str, Any]]


# Tool definition
class Tool(BaseModel):
    """Tool definition for function calling."""

    name: str
    description: str
    input_schema: dict[str, Any]


# Request/Response
class MessagesRequest(BaseModel):
    """Request body for POST /v1/messages."""

    model: str
    max_tokens: int
    messages: list[Message]
    system: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    tools: list[Tool] | None = None


class Usage(BaseModel):
    """Token usage information."""

    input_tokens: int
    output_tokens: int


class MessagesResponse(BaseModel):
    """Response body for POST /v1/messages."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[TextBlock | ToolUseBlock]
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    usage: Usage
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_anthropic_models.py -v`
Expected: All 11 tests PASS

**Step 5: Commit**

```bash
git add src/llm_proxy/models/anthropic.py tests/test_anthropic_models.py
git commit -m "feat: add Anthropic API Pydantic models"
```

---

## Task 5: Request Translation (Anthropic → LangChain)

**Files:**
- Create: `src/llm_proxy/translation/__init__.py`
- Create: `src/llm_proxy/translation/request.py`
- Create: `tests/test_request_translation.py`

**Step 1: Create translation package**

Create `src/llm_proxy/translation/__init__.py`:
```python
"""Translation between Anthropic and LangChain formats."""
```

**Step 2: Write failing test for request translation**

Create `tests/test_request_translation.py`:
```python
"""Tests for Anthropic to LangChain request translation."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llm_proxy.models.anthropic import Message, MessagesRequest, Tool
from llm_proxy.translation.request import (
    translate_messages,
    translate_tools,
    build_langchain_kwargs,
)


def test_translate_simple_user_message():
    """Translate simple user message."""
    messages = [Message(role="user", content="Hello")]
    result = translate_messages(messages, system=None)

    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "Hello"


def test_translate_with_system():
    """System prompt becomes SystemMessage."""
    messages = [Message(role="user", content="Hi")]
    result = translate_messages(messages, system="Be helpful")

    assert len(result) == 2
    assert isinstance(result[0], SystemMessage)
    assert result[0].content == "Be helpful"
    assert isinstance(result[1], HumanMessage)


def test_translate_conversation():
    """Multi-turn conversation."""
    messages = [
        Message(role="user", content="Hi"),
        Message(role="assistant", content="Hello!"),
        Message(role="user", content="How are you?"),
    ]
    result = translate_messages(messages, system=None)

    assert len(result) == 3
    assert isinstance(result[0], HumanMessage)
    assert isinstance(result[1], AIMessage)
    assert isinstance(result[2], HumanMessage)


def test_translate_multimodal_message():
    """Message with text and image."""
    messages = [
        Message(
            role="user",
            content=[
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgo=",
                    },
                },
            ],
        )
    ]
    result = translate_messages(messages, system=None)

    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)
    # LangChain uses list of content blocks for multimodal
    assert isinstance(result[0].content, list)
    assert result[0].content[0]["type"] == "text"
    assert result[0].content[1]["type"] == "image_url"


def test_translate_tools():
    """Convert Anthropic tools to LangChain format."""
    tools = [
        Tool(
            name="get_weather",
            description="Get the weather",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
    ]
    result = translate_tools(tools)

    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "get_weather"
    assert result[0]["function"]["parameters"]["properties"]["city"]["type"] == "string"


def test_build_langchain_kwargs():
    """Build kwargs for ChatOpenAI invocation."""
    request = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        messages=[Message(role="user", content="Hi")],
        temperature=0.7,
        top_p=0.9,
        stop_sequences=["END"],
    )
    kwargs = build_langchain_kwargs(request)

    assert kwargs["max_tokens"] == 1024
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9
    assert kwargs["stop"] == ["END"]


def test_build_langchain_kwargs_minimal():
    """Minimal kwargs only includes max_tokens."""
    request = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=512,
        messages=[Message(role="user", content="Hi")],
    )
    kwargs = build_langchain_kwargs(request)

    assert kwargs["max_tokens"] == 512
    assert "temperature" not in kwargs
    assert "stop" not in kwargs
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_request_translation.py -v`
Expected: FAIL with "cannot import name 'translate_messages'"

**Step 4: Implement request translation**

Create `src/llm_proxy/translation/request.py`:
```python
"""Translate Anthropic requests to LangChain format."""

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from llm_proxy.models.anthropic import Message, MessagesRequest, Tool


def translate_content_block(block: dict[str, Any]) -> dict[str, Any]:
    """Translate a single content block to LangChain format."""
    if block["type"] == "text":
        return {"type": "text", "text": block["text"]}
    elif block["type"] == "image":
        source = block["source"]
        if source["type"] == "base64":
            data_url = f"data:{source['media_type']};base64,{source['data']}"
            return {"type": "image_url", "image_url": {"url": data_url}}
        else:
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    elif block["type"] == "tool_result":
        # Tool results are handled specially in the message
        return {"type": "text", "text": str(block.get("content", ""))}
    return block


def translate_message(msg: Message) -> BaseMessage:
    """Translate a single Anthropic message to LangChain message."""
    # Handle string content
    if isinstance(msg.content, str):
        if msg.role == "user":
            return HumanMessage(content=msg.content)
        else:
            return AIMessage(content=msg.content)

    # Handle content blocks (multimodal)
    content_blocks = [translate_content_block(block) for block in msg.content]

    if msg.role == "user":
        return HumanMessage(content=content_blocks)
    else:
        return AIMessage(content=content_blocks)


def translate_messages(
    messages: list[Message], system: str | None
) -> list[BaseMessage]:
    """Translate Anthropic messages to LangChain messages."""
    result: list[BaseMessage] = []

    if system:
        result.append(SystemMessage(content=system))

    for msg in messages:
        result.append(translate_message(msg))

    return result


def translate_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Translate Anthropic tools to OpenAI function format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


def build_langchain_kwargs(request: MessagesRequest) -> dict[str, Any]:
    """Build kwargs for LangChain model invocation."""
    kwargs: dict[str, Any] = {
        "max_tokens": request.max_tokens,
    }

    if request.temperature is not None:
        kwargs["temperature"] = request.temperature

    if request.top_p is not None:
        kwargs["top_p"] = request.top_p

    if request.stop_sequences:
        kwargs["stop"] = request.stop_sequences

    return kwargs
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_request_translation.py -v`
Expected: All 8 tests PASS

**Step 6: Commit**

```bash
git add src/llm_proxy/translation/ tests/test_request_translation.py
git commit -m "feat: add request translation (Anthropic → LangChain)"
```

---

## Task 6: Response Translation (LangChain → Anthropic)

**Files:**
- Create: `src/llm_proxy/translation/response.py`
- Create: `tests/test_response_translation.py`

**Step 1: Write failing test for response translation**

Create `tests/test_response_translation.py`:
```python
"""Tests for LangChain to Anthropic response translation."""

import pytest
from langchain_core.messages import AIMessage

from llm_proxy.models.anthropic import MessagesResponse
from llm_proxy.translation.response import translate_response, map_stop_reason


def test_translate_simple_response():
    """Translate simple text response."""
    ai_message = AIMessage(
        content="Hello, how can I help?",
        response_metadata={"finish_reason": "stop"},
        usage_metadata={"input_tokens": 10, "output_tokens": 8},
    )

    result = translate_response(ai_message, model="claude-3-haiku-20240307")

    assert result.role == "assistant"
    assert result.model == "claude-3-haiku-20240307"
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert result.content[0].text == "Hello, how can I help?"
    assert result.stop_reason == "end_turn"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 8


def test_translate_response_with_tool_calls():
    """Translate response with tool calls."""
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_123",
                "name": "get_weather",
                "args": {"city": "London"},
            }
        ],
        response_metadata={"finish_reason": "tool_calls"},
        usage_metadata={"input_tokens": 15, "output_tokens": 20},
    )

    result = translate_response(ai_message, model="claude-3-haiku-20240307")

    assert len(result.content) == 1
    assert result.content[0].type == "tool_use"
    assert result.content[0].id == "call_123"
    assert result.content[0].name == "get_weather"
    assert result.content[0].input == {"city": "London"}
    assert result.stop_reason == "tool_use"


def test_translate_response_text_and_tool():
    """Response with both text and tool call."""
    ai_message = AIMessage(
        content="Let me check the weather for you.",
        tool_calls=[
            {
                "id": "call_456",
                "name": "get_weather",
                "args": {"city": "Paris"},
            }
        ],
        response_metadata={"finish_reason": "tool_calls"},
        usage_metadata={"input_tokens": 20, "output_tokens": 25},
    )

    result = translate_response(ai_message, model="claude-3-haiku-20240307")

    assert len(result.content) == 2
    assert result.content[0].type == "text"
    assert result.content[0].text == "Let me check the weather for you."
    assert result.content[1].type == "tool_use"


def test_map_stop_reason():
    """Map OpenAI finish reasons to Anthropic stop reasons."""
    assert map_stop_reason("stop") == "end_turn"
    assert map_stop_reason("length") == "max_tokens"
    assert map_stop_reason("tool_calls") == "tool_use"
    assert map_stop_reason("content_filter") == "end_turn"
    assert map_stop_reason(None) == "end_turn"


def test_translate_response_generates_id():
    """Response ID is generated."""
    ai_message = AIMessage(
        content="Hi",
        response_metadata={"finish_reason": "stop"},
        usage_metadata={"input_tokens": 5, "output_tokens": 1},
    )

    result = translate_response(ai_message, model="claude-3-haiku-20240307")

    assert result.id.startswith("msg_proxy_")


def test_translate_response_missing_usage():
    """Handle missing usage metadata gracefully."""
    ai_message = AIMessage(
        content="Hi",
        response_metadata={"finish_reason": "stop"},
    )

    result = translate_response(ai_message, model="claude-3-haiku-20240307")

    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_response_translation.py -v`
Expected: FAIL with "cannot import name 'translate_response'"

**Step 3: Implement response translation**

Create `src/llm_proxy/translation/response.py`:
```python
"""Translate LangChain responses to Anthropic format."""

import uuid
from typing import Literal

from langchain_core.messages import AIMessage

from llm_proxy.models.anthropic import (
    MessagesResponse,
    TextBlock,
    ToolUseBlock,
    Usage,
)


StopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]


def map_stop_reason(finish_reason: str | None) -> StopReason:
    """Map OpenAI finish reason to Anthropic stop reason."""
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    return mapping.get(finish_reason or "", "end_turn")


def generate_message_id() -> str:
    """Generate a unique message ID."""
    return f"msg_proxy_{uuid.uuid4().hex[:24]}"


def translate_response(ai_message: AIMessage, model: str) -> MessagesResponse:
    """Translate LangChain AIMessage to Anthropic MessagesResponse."""
    content: list[TextBlock | ToolUseBlock] = []

    # Add text content if present
    if ai_message.content:
        text = (
            ai_message.content
            if isinstance(ai_message.content, str)
            else str(ai_message.content)
        )
        if text:
            content.append(TextBlock(type="text", text=text))

    # Add tool calls if present
    if ai_message.tool_calls:
        for tool_call in ai_message.tool_calls:
            content.append(
                ToolUseBlock(
                    type="tool_use",
                    id=tool_call["id"],
                    name=tool_call["name"],
                    input=tool_call["args"],
                )
            )

    # Extract usage metadata
    usage_meta = getattr(ai_message, "usage_metadata", None) or {}
    usage = Usage(
        input_tokens=usage_meta.get("input_tokens", 0),
        output_tokens=usage_meta.get("output_tokens", 0),
    )

    # Extract stop reason
    response_meta = getattr(ai_message, "response_metadata", None) or {}
    finish_reason = response_meta.get("finish_reason")
    stop_reason = map_stop_reason(finish_reason)

    return MessagesResponse(
        id=generate_message_id(),
        type="message",
        role="assistant",
        model=model,
        content=content,
        stop_reason=stop_reason,
        usage=usage,
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_response_translation.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/llm_proxy/translation/response.py tests/test_response_translation.py
git commit -m "feat: add response translation (LangChain → Anthropic)"
```

---

## Task 7: Backend Router and Factory

**Files:**
- Create: `src/llm_proxy/backends/__init__.py`
- Create: `src/llm_proxy/backends/router.py`
- Create: `src/llm_proxy/backends/factory.py`
- Create: `tests/test_backends.py`

**Step 1: Create backends package**

Create `src/llm_proxy/backends/__init__.py`:
```python
"""Backend routing and LangChain client factory."""
```

**Step 2: Write failing test for backend routing**

Create `tests/test_backends.py`:
```python
"""Tests for backend routing and factory."""

import pytest
from langchain_openai import ChatOpenAI

from llm_proxy.backends.router import resolve_backend, BackendNotFoundError
from llm_proxy.backends.factory import create_chat_model
from llm_proxy.models.config import BackendConfig, ModelMapping, ProxyConfig


@pytest.fixture
def sample_config() -> ProxyConfig:
    """Sample proxy configuration."""
    return ProxyConfig(
        models={
            "claude-3-haiku-20240307": ModelMapping(backend="ollama", model="llama3.2"),
            "claude-3-opus-20240229": ModelMapping(
                backend="databricks", model="llama-70b"
            ),
        },
        backends={
            "ollama": BackendConfig(base_url="http://localhost:11434/v1"),
            "databricks": BackendConfig(
                base_url="https://example.databricks.com/serving-endpoints",
                api_key="dapi-test",
            ),
        },
    )


def test_resolve_backend_found(sample_config):
    """Resolve known model to backend."""
    backend_config, model_name = resolve_backend(
        "claude-3-haiku-20240307", sample_config
    )

    assert backend_config.base_url == "http://localhost:11434/v1"
    assert model_name == "llama3.2"


def test_resolve_backend_not_found(sample_config):
    """Unknown model raises error."""
    with pytest.raises(BackendNotFoundError) as exc_info:
        resolve_backend("claude-unknown", sample_config)

    assert "claude-unknown" in str(exc_info.value)


def test_create_chat_model_ollama():
    """Create ChatOpenAI for Ollama backend."""
    backend_config = BackendConfig(base_url="http://localhost:11434/v1")

    model = create_chat_model(backend_config, "llama3.2")

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "llama3.2"
    assert str(model.openai_api_base) == "http://localhost:11434/v1"


def test_create_chat_model_with_api_key():
    """Create ChatOpenAI with API key."""
    backend_config = BackendConfig(
        base_url="https://example.databricks.com",
        api_key="dapi-secret",
    )

    model = create_chat_model(backend_config, "llama-70b")

    assert isinstance(model, ChatOpenAI)
    assert model.openai_api_key.get_secret_value() == "dapi-secret"
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_backends.py -v`
Expected: FAIL with "cannot import name 'resolve_backend'"

**Step 4: Implement backend router**

Create `src/llm_proxy/backends/router.py`:
```python
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
```

**Step 5: Implement backend factory**

Create `src/llm_proxy/backends/factory.py`:
```python
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
```

**Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_backends.py -v`
Expected: All 4 tests PASS

**Step 7: Commit**

```bash
git add src/llm_proxy/backends/ tests/test_backends.py
git commit -m "feat: add backend router and ChatOpenAI factory"
```

---

## Task 8: Authentication Middleware

**Files:**
- Create: `src/llm_proxy/auth.py`
- Create: `tests/test_auth.py`

**Step 1: Write failing test for authentication**

Create `tests/test_auth.py`:
```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_auth.py -v`
Expected: FAIL with "cannot import name 'APIKeyAuth'"

**Step 3: Implement authentication**

Create `src/llm_proxy/auth.py`:
```python
"""API key authentication middleware."""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_auth.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/llm_proxy/auth.py tests/test_auth.py
git commit -m "feat: add API key authentication middleware"
```

---

## Task 9: Messages Endpoint (Non-Streaming)

**Files:**
- Create: `src/llm_proxy/routes/__init__.py`
- Create: `src/llm_proxy/routes/messages.py`
- Create: `tests/test_messages_endpoint.py`

**Step 1: Create routes package**

Create `src/llm_proxy/routes/__init__.py`:
```python
"""FastAPI route handlers."""
```

**Step 2: Write failing test for messages endpoint**

Create `tests/test_messages_endpoint.py`:
```python
"""Tests for /v1/messages endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from llm_proxy.models.config import BackendConfig, ModelMapping, ProxyConfig
from llm_proxy.routes.messages import create_messages_router


@pytest.fixture
def sample_config() -> ProxyConfig:
    """Sample proxy configuration."""
    return ProxyConfig(
        models={
            "claude-3-haiku-20240307": ModelMapping(backend="ollama", model="llama3.2"),
        },
        backends={
            "ollama": BackendConfig(base_url="http://localhost:11434/v1"),
        },
    )


@pytest.fixture
def mock_ai_response():
    """Mock LangChain AI response."""
    return AIMessage(
        content="Hello! How can I help you today?",
        response_metadata={"finish_reason": "stop"},
        usage_metadata={"input_tokens": 10, "output_tokens": 8},
    )


@pytest.fixture
def app(sample_config, mock_ai_response):
    """FastAPI app with messages router."""
    app = FastAPI()

    with patch("llm_proxy.routes.messages.create_chat_model") as mock_factory:
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_ai_response)
        mock_factory.return_value = mock_model

        router = create_messages_router(sample_config)
        app.include_router(router)

    return app


def test_messages_simple_request(app, mock_ai_response):
    """Simple messages request returns Anthropic format."""
    with patch("llm_proxy.routes.messages.create_chat_model") as mock_factory:
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_ai_response)
        mock_factory.return_value = mock_model

        client = TestClient(app)
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-haiku-20240307",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["model"] == "claude-3-haiku-20240307"
        assert len(data["content"]) == 1
        assert data["content"][0]["type"] == "text"


def test_messages_unknown_model(sample_config):
    """Unknown model returns 400 error."""
    app = FastAPI()
    router = create_messages_router(sample_config)
    app.include_router(router)

    client = TestClient(app)
    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-unknown",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert data["type"] == "error"
    assert data["error"]["type"] == "invalid_request_error"


def test_messages_invalid_request(sample_config):
    """Invalid request body returns 400."""
    app = FastAPI()
    router = create_messages_router(sample_config)
    app.include_router(router)

    client = TestClient(app)
    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-3-haiku-20240307",
            # missing max_tokens and messages
        },
    )

    assert response.status_code == 422  # Pydantic validation error
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_messages_endpoint.py -v`
Expected: FAIL with "cannot import name 'create_messages_router'"

**Step 4: Implement messages endpoint**

Create `src/llm_proxy/routes/messages.py`:
```python
"""Messages endpoint handler."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from llm_proxy.backends.factory import create_chat_model
from llm_proxy.backends.router import BackendNotFoundError, resolve_backend
from llm_proxy.models.anthropic import MessagesRequest, MessagesResponse
from llm_proxy.models.config import ProxyConfig
from llm_proxy.translation.request import (
    build_langchain_kwargs,
    translate_messages,
    translate_tools,
)
from llm_proxy.translation.response import translate_response


def make_error_response(
    status_code: int, error_type: str, message: str
) -> JSONResponse:
    """Create Anthropic-style error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            },
        },
    )


def create_messages_router(config: ProxyConfig) -> APIRouter:
    """Create messages router with configuration."""
    router = APIRouter()

    @router.post("/v1/messages")
    async def create_message(request: MessagesRequest) -> MessagesResponse:
        """Handle POST /v1/messages."""
        # Resolve backend for the requested model
        try:
            backend_config, backend_model = resolve_backend(request.model, config)
        except BackendNotFoundError as e:
            return make_error_response(400, "invalid_request_error", str(e))

        # Create LangChain model
        chat_model = create_chat_model(backend_config, backend_model)

        # Translate request to LangChain format
        messages = translate_messages(request.messages, request.system)
        kwargs = build_langchain_kwargs(request)

        # Bind tools if present
        if request.tools:
            tools = translate_tools(request.tools)
            chat_model = chat_model.bind_tools(tools)

        # Handle streaming separately
        if request.stream:
            # TODO: Implement streaming in Task 10
            return make_error_response(
                400, "invalid_request_error", "Streaming not yet implemented"
            )

        # Invoke the model
        try:
            ai_message = await chat_model.ainvoke(messages, **kwargs)
        except Exception as e:
            return make_error_response(502, "api_error", f"Backend error: {str(e)}")

        # Translate response to Anthropic format
        response = translate_response(ai_message, request.model)
        return response

    return router
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_messages_endpoint.py -v`
Expected: All 3 tests PASS

**Step 6: Commit**

```bash
git add src/llm_proxy/routes/ tests/test_messages_endpoint.py
git commit -m "feat: add /v1/messages endpoint (non-streaming)"
```

---

## Task 10: Streaming Support

**Files:**
- Modify: `src/llm_proxy/routes/messages.py`
- Create: `src/llm_proxy/translation/streaming.py`
- Create: `tests/test_streaming.py`

**Step 1: Write failing test for streaming**

Create `tests/test_streaming.py`:
```python
"""Tests for streaming response translation."""

import json
import pytest
from langchain_core.messages import AIMessageChunk

from llm_proxy.translation.streaming import (
    StreamingState,
    translate_stream_start,
    translate_stream_delta,
    translate_stream_end,
)


def test_stream_start_event():
    """Generate message_start and content_block_start events."""
    state = StreamingState(model="claude-3-haiku-20240307")
    events = translate_stream_start(state)

    assert len(events) == 2

    # First event: message_start
    msg_start = json.loads(events[0])
    assert msg_start["type"] == "message_start"
    assert msg_start["message"]["model"] == "claude-3-haiku-20240307"
    assert msg_start["message"]["role"] == "assistant"

    # Second event: content_block_start
    block_start = json.loads(events[1])
    assert block_start["type"] == "content_block_start"
    assert block_start["index"] == 0
    assert block_start["content_block"]["type"] == "text"


def test_stream_delta_text():
    """Generate content_block_delta for text chunk."""
    state = StreamingState(model="claude-3-haiku-20240307")
    chunk = AIMessageChunk(content="Hello")

    events = translate_stream_delta(chunk, state)

    assert len(events) == 1
    delta = json.loads(events[0])
    assert delta["type"] == "content_block_delta"
    assert delta["index"] == 0
    assert delta["delta"]["type"] == "text_delta"
    assert delta["delta"]["text"] == "Hello"


def test_stream_delta_tool_call():
    """Generate events for tool call chunk."""
    state = StreamingState(model="claude-3-haiku-20240307")
    state.current_block_index = 0  # Already started text block
    state.has_text = True

    chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "index": 0,
                "id": "call_123",
                "name": "get_weather",
                "args": '{"city": "London"}',
            }
        ],
    )

    events = translate_stream_delta(chunk, state)

    # Should have: content_block_stop (text), content_block_start (tool), content_block_delta (tool)
    assert len(events) >= 2


def test_stream_end():
    """Generate final events."""
    state = StreamingState(model="claude-3-haiku-20240307")
    state.current_block_index = 0
    state.input_tokens = 10
    state.output_tokens = 5

    events = translate_stream_end(state, stop_reason="end_turn")

    # Should have: content_block_stop, message_delta, message_stop
    assert len(events) == 3

    block_stop = json.loads(events[0])
    assert block_stop["type"] == "content_block_stop"

    msg_delta = json.loads(events[1])
    assert msg_delta["type"] == "message_delta"
    assert msg_delta["delta"]["stop_reason"] == "end_turn"
    assert msg_delta["usage"]["output_tokens"] == 5

    msg_stop = json.loads(events[2])
    assert msg_stop["type"] == "message_stop"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_streaming.py -v`
Expected: FAIL with "cannot import name 'StreamingState'"

**Step 3: Implement streaming translation**

Create `src/llm_proxy/translation/streaming.py`:
```python
"""Streaming response translation for Server-Sent Events."""

import json
import uuid
from dataclasses import dataclass, field

from langchain_core.messages import AIMessageChunk


@dataclass
class StreamingState:
    """Track state during streaming."""

    model: str
    message_id: str = field(default_factory=lambda: f"msg_proxy_{uuid.uuid4().hex[:24]}")
    current_block_index: int = -1
    has_text: bool = False
    tool_call_ids: dict[int, str] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0


def translate_stream_start(state: StreamingState) -> list[str]:
    """Generate initial stream events."""
    events = []

    # message_start event
    message_start = {
        "type": "message_start",
        "message": {
            "id": state.message_id,
            "type": "message",
            "role": "assistant",
            "model": state.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }
    events.append(json.dumps(message_start))

    # content_block_start for first text block
    state.current_block_index = 0
    block_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    events.append(json.dumps(block_start))

    return events


def translate_stream_delta(
    chunk: AIMessageChunk, state: StreamingState
) -> list[str]:
    """Translate a stream chunk to Anthropic events."""
    events = []

    # Handle text content
    if chunk.content:
        state.has_text = True
        delta = {
            "type": "content_block_delta",
            "index": state.current_block_index,
            "delta": {"type": "text_delta", "text": chunk.content},
        }
        events.append(json.dumps(delta))

    # Handle tool calls
    if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
        for tool_chunk in chunk.tool_call_chunks:
            tool_index = tool_chunk.get("index", 0)

            # Check if this is a new tool call
            if tool_index not in state.tool_call_ids:
                # Close previous block if exists
                if state.current_block_index >= 0:
                    events.append(
                        json.dumps(
                            {"type": "content_block_stop", "index": state.current_block_index}
                        )
                    )

                # Start new tool_use block
                state.current_block_index += 1
                tool_id = tool_chunk.get("id", f"call_{uuid.uuid4().hex[:8]}")
                state.tool_call_ids[tool_index] = tool_id

                block_start = {
                    "type": "content_block_start",
                    "index": state.current_block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_chunk.get("name", ""),
                        "input": {},
                    },
                }
                events.append(json.dumps(block_start))

            # Stream tool input
            if tool_chunk.get("args"):
                delta = {
                    "type": "content_block_delta",
                    "index": state.current_block_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": tool_chunk["args"],
                    },
                }
                events.append(json.dumps(delta))

    # Track token usage if available
    if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
        state.input_tokens = chunk.usage_metadata.get("input_tokens", state.input_tokens)
        state.output_tokens += chunk.usage_metadata.get("output_tokens", 0)

    return events


def translate_stream_end(state: StreamingState, stop_reason: str) -> list[str]:
    """Generate final stream events."""
    events = []

    # Close current content block
    if state.current_block_index >= 0:
        events.append(
            json.dumps({"type": "content_block_stop", "index": state.current_block_index})
        )

    # message_delta with stop reason and usage
    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": state.output_tokens},
    }
    events.append(json.dumps(message_delta))

    # message_stop
    events.append(json.dumps({"type": "message_stop"}))

    return events
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_streaming.py -v`
Expected: All 4 tests PASS

**Step 5: Update messages endpoint for streaming**

Modify `src/llm_proxy/routes/messages.py` - replace the entire file:
```python
"""Messages endpoint handler."""

from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from llm_proxy.backends.factory import create_chat_model
from llm_proxy.backends.router import BackendNotFoundError, resolve_backend
from llm_proxy.models.anthropic import MessagesRequest, MessagesResponse
from llm_proxy.models.config import ProxyConfig
from llm_proxy.translation.request import (
    build_langchain_kwargs,
    translate_messages,
    translate_tools,
)
from llm_proxy.translation.response import translate_response, map_stop_reason
from llm_proxy.translation.streaming import (
    StreamingState,
    translate_stream_start,
    translate_stream_delta,
    translate_stream_end,
)


def make_error_response(
    status_code: int, error_type: str, message: str
) -> JSONResponse:
    """Create Anthropic-style error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            },
        },
    )


def create_messages_router(config: ProxyConfig) -> APIRouter:
    """Create messages router with configuration."""
    router = APIRouter()

    @router.post("/v1/messages")
    async def create_message(request: MessagesRequest):
        """Handle POST /v1/messages."""
        # Resolve backend for the requested model
        try:
            backend_config, backend_model = resolve_backend(request.model, config)
        except BackendNotFoundError as e:
            return make_error_response(400, "invalid_request_error", str(e))

        # Create LangChain model
        chat_model = create_chat_model(backend_config, backend_model)

        # Translate request to LangChain format
        messages = translate_messages(request.messages, request.system)
        kwargs = build_langchain_kwargs(request)

        # Bind tools if present
        if request.tools:
            tools = translate_tools(request.tools)
            chat_model = chat_model.bind_tools(tools)

        # Handle streaming
        if request.stream:
            return EventSourceResponse(
                stream_response(chat_model, messages, kwargs, request.model)
            )

        # Non-streaming invocation
        try:
            ai_message = await chat_model.ainvoke(messages, **kwargs)
        except Exception as e:
            return make_error_response(502, "api_error", f"Backend error: {str(e)}")

        # Translate response to Anthropic format
        response = translate_response(ai_message, request.model)
        return response

    return router


async def stream_response(
    chat_model, messages: list, kwargs: dict, model: str
) -> AsyncGenerator[dict, None]:
    """Stream response as Server-Sent Events."""
    state = StreamingState(model=model)
    stop_reason = "end_turn"

    try:
        # Send initial events
        for event_data in translate_stream_start(state):
            yield {"event": event_data.split('"type":"')[1].split('"')[0], "data": event_data}

        # Stream chunks
        async for chunk in chat_model.astream(messages, **kwargs):
            # Track finish reason
            if hasattr(chunk, "response_metadata"):
                finish_reason = chunk.response_metadata.get("finish_reason")
                if finish_reason:
                    stop_reason = map_stop_reason(finish_reason)

            for event_data in translate_stream_delta(chunk, state):
                event_type = event_data.split('"type":"')[1].split('"')[0]
                yield {"event": event_type, "data": event_data}

        # Send final events
        for event_data in translate_stream_end(state, stop_reason):
            event_type = event_data.split('"type":"')[1].split('"')[0]
            yield {"event": event_type, "data": event_data}

    except Exception as e:
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": str(e)},
        }
        yield {"event": "error", "data": str(error_event)}
```

**Step 6: Run all tests to verify nothing broke**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add src/llm_proxy/translation/streaming.py src/llm_proxy/routes/messages.py tests/test_streaming.py
git commit -m "feat: add streaming support for /v1/messages"
```

---

## Task 11: FastAPI Application Entry Point

**Files:**
- Create: `src/llm_proxy/main.py`
- Create: `tests/test_app.py`

**Step 1: Write failing test for app setup**

Create `tests/test_app.py`:
```python
"""Tests for FastAPI application setup."""

import os
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def test_app_starts_with_config(monkeypatch):
    """App starts with valid config file."""
    config_content = """
models:
  claude-3-haiku-20240307:
    backend: ollama
    model: llama3.2

backends:
  ollama:
    base_url: "http://localhost:11434/v1"

server:
  host: "0.0.0.0"
  port: 8080
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        f.flush()

        monkeypatch.setenv("LLM_PROXY_CONFIG", f.name)

        # Import after setting env var
        from llm_proxy.main import create_app

        app = create_app()
        client = TestClient(app)

        # Health check should work
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    os.unlink(f.name)


def test_app_with_auth(monkeypatch):
    """App enforces auth when PROXY_API_KEY is set."""
    config_content = """
models: {}
backends: {}
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        f.flush()

        monkeypatch.setenv("LLM_PROXY_CONFIG", f.name)
        monkeypatch.setenv("PROXY_API_KEY", "test-key")

        from llm_proxy.main import create_app

        app = create_app()
        client = TestClient(app)

        # Request without key should fail
        response = client.post("/v1/messages", json={})
        assert response.status_code == 401

        # Request with key should get further (fail on validation, not auth)
        response = client.post(
            "/v1/messages",
            json={},
            headers={"x-api-key": "test-key"},
        )
        assert response.status_code == 422  # Validation error, not auth

    os.unlink(f.name)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_app.py -v`
Expected: FAIL with "cannot import name 'create_app'"

**Step 3: Implement main application**

Create `src/llm_proxy/main.py`:
```python
"""FastAPI application entry point."""

import os
from pathlib import Path

from fastapi import FastAPI

from llm_proxy.auth import APIKeyAuth
from llm_proxy.config import load_config
from llm_proxy.routes.messages import create_messages_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Load configuration
    config_path = Path(os.environ.get("LLM_PROXY_CONFIG", "config.yaml"))
    config = load_config(config_path)

    # Create FastAPI app
    app = FastAPI(
        title="LLM Proxy",
        description="Anthropic-compatible API proxy for OpenAI-compatible backends",
        version="0.1.0",
    )

    # Add authentication middleware
    api_key = os.environ.get("PROXY_API_KEY")
    auth = APIKeyAuth(api_key=api_key)
    app.middleware("http")(auth)

    # Register routes
    messages_router = create_messages_router(config)
    app.include_router(messages_router)

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    return app


# For running with uvicorn directly
app = create_app() if os.environ.get("LLM_PROXY_CONFIG") else None


def main():
    """Run the server with uvicorn."""
    import uvicorn

    config_path = Path(os.environ.get("LLM_PROXY_CONFIG", "config.yaml"))
    config = load_config(config_path)

    host = config.server.get("host", "0.0.0.0")
    port = config.server.get("port", 8080)

    uvicorn.run(
        "llm_proxy.main:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_app.py -v`
Expected: All 2 tests PASS

**Step 5: Add entry point to pyproject.toml**

Edit `pyproject.toml` to add scripts section after `[project.optional-dependencies]`:
```toml
[project.scripts]
llm-proxy = "llm_proxy.main:main"
```

**Step 6: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add src/llm_proxy/main.py tests/test_app.py pyproject.toml
git commit -m "feat: add FastAPI application entry point"
```

---

## Task 12: Integration Test with Mocked Backend

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

Create `tests/test_integration.py`:
```python
"""Integration tests with mocked LangChain backend."""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from llm_proxy.main import create_app


@pytest.fixture
def config_file():
    """Create temporary config file."""
    config_content = """
models:
  claude-3-haiku-20240307:
    backend: mock
    model: mock-model

backends:
  mock:
    base_url: "http://mock:11434/v1"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_response():
    """Standard mock AI response."""
    return AIMessage(
        content="I'm a helpful assistant!",
        response_metadata={"finish_reason": "stop"},
        usage_metadata={"input_tokens": 15, "output_tokens": 6},
    )


@pytest.fixture
def app_client(config_file, mock_response, monkeypatch):
    """Create test client with mocked backend."""
    monkeypatch.setenv("LLM_PROXY_CONFIG", config_file)
    monkeypatch.setenv("PROXY_API_KEY", "test-api-key")

    with patch("llm_proxy.routes.messages.create_chat_model") as mock_factory:
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        mock_factory.return_value = mock_model

        app = create_app()
        yield TestClient(app), mock_model


def test_full_request_response_cycle(app_client):
    """Full request/response cycle with mocked backend."""
    client, mock_model = app_client

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test-api-key"},
        json={
            "model": "claude-3-haiku-20240307",
            "max_tokens": 1024,
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "Hello!"},
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify Anthropic response format
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["model"] == "claude-3-haiku-20240307"
    assert data["content"][0]["type"] == "text"
    assert data["content"][0]["text"] == "I'm a helpful assistant!"
    assert data["stop_reason"] == "end_turn"
    assert data["usage"]["input_tokens"] == 15
    assert data["usage"]["output_tokens"] == 6


def test_conversation_history(app_client):
    """Multi-turn conversation passes correctly."""
    client, mock_model = app_client

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test-api-key"},
        json={
            "model": "claude-3-haiku-20240307",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
            ],
        },
    )

    assert response.status_code == 200

    # Verify LangChain was called with correct message count
    call_args = mock_model.ainvoke.call_args
    messages = call_args[0][0]
    assert len(messages) == 3  # All 3 messages passed


def test_tool_calling(app_client):
    """Tool definition passes to backend."""
    client, mock_model = app_client

    # Update mock to return tool call
    tool_response = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "get_weather", "args": {"city": "NYC"}}],
        response_metadata={"finish_reason": "tool_calls"},
        usage_metadata={"input_tokens": 20, "output_tokens": 15},
    )
    mock_model.ainvoke = AsyncMock(return_value=tool_response)

    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test-api-key"},
        json={
            "model": "claude-3-haiku-20240307",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "What's the weather in NYC?"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Verify tool use in response
    assert data["stop_reason"] == "tool_use"
    assert data["content"][0]["type"] == "tool_use"
    assert data["content"][0]["name"] == "get_weather"
    assert data["content"][0]["input"]["city"] == "NYC"

    # Verify bind_tools was called
    mock_model.bind_tools.assert_called_once()
```

**Step 2: Run integration tests**

Run: `uv run pytest tests/test_integration.py -v`
Expected: All 3 tests PASS

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests with mocked backend"
```

---

## Task 13: Final Verification and Documentation

**Files:**
- Update: `src/llm_proxy/models/__init__.py`

**Step 1: Update models package exports**

Replace `src/llm_proxy/models/__init__.py`:
```python
"""Pydantic models for LLM Proxy."""

from llm_proxy.models.anthropic import (
    ContentBlock,
    ImageBlock,
    ImageSource,
    Message,
    MessagesRequest,
    MessagesResponse,
    TextBlock,
    Tool,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from llm_proxy.models.config import BackendConfig, ModelMapping, ProxyConfig

__all__ = [
    "BackendConfig",
    "ContentBlock",
    "ImageBlock",
    "ImageSource",
    "Message",
    "MessagesRequest",
    "MessagesResponse",
    "ModelMapping",
    "ProxyConfig",
    "TextBlock",
    "Tool",
    "ToolResultBlock",
    "ToolUseBlock",
    "Usage",
]
```

**Step 2: Run full test suite with coverage**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 3: Verify the server starts**

Run: `LLM_PROXY_CONFIG=config.yaml uv run python -c "from llm_proxy.main import create_app; print('App creates successfully')" 2>&1 || echo "Expected: needs valid config"`
Expected: Either success message or expected config error (Databricks env vars not set)

**Step 4: Commit final changes**

```bash
git add src/llm_proxy/models/__init__.py
git commit -m "chore: update model exports and finalize implementation"
```

---

## Summary

Implementation complete. The proxy now supports:
- Full Anthropic Messages API (`/v1/messages`)
- Streaming via Server-Sent Events
- Tool/function calling
- Multimodal content (images)
- Model-based routing to multiple backends
- API key authentication
- Anthropic-compatible error responses
- Configurable via YAML + environment variables

To run locally:
```bash
# Set up environment
cp .env.example .env
# Edit .env with your credentials

# Start the proxy
LLM_PROXY_CONFIG=config.yaml uv run llm-proxy
```

To test with Anthropic SDK:
```python
import anthropic

client = anthropic.Anthropic(
    api_key="your-proxy-api-key",
    base_url="http://localhost:8080",
)

response = client.messages.create(
    model="claude-3-haiku-20240307",  # Routes to Ollama
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
```
