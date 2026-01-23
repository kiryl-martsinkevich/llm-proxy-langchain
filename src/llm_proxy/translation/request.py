"""Translate Anthropic requests to LangChain format."""

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from llm_proxy.models.anthropic import Message, MessagesRequest, TextBlock, Tool


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
    messages: list[Message], system: str | list[TextBlock] | None
) -> list[BaseMessage]:
    """Translate Anthropic messages to LangChain messages."""
    result: list[BaseMessage] = []

    if system:
        # Handle both string and list of TextBlock formats
        if isinstance(system, str):
            result.append(SystemMessage(content=system))
        else:
            # Concatenate text from all system blocks
            system_text = "\n".join(block.text for block in system)
            result.append(SystemMessage(content=system_text))

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
