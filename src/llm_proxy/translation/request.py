"""Translate Anthropic requests to LangChain format."""

from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

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


def translate_assistant_message(msg: Message) -> list[BaseMessage]:
    """Translate assistant message, extracting tool_use into tool_calls.

    Anthropic puts tool_use blocks in content[], but OpenAI/LangChain
    expects them in the tool_calls field of AIMessage.
    """
    # Handle string content
    if isinstance(msg.content, str):
        return [AIMessage(content=msg.content)]

    text_parts = []
    tool_calls = []

    for block in msg.content:
        if block["type"] == "text":
            text_parts.append(block["text"])
        elif block["type"] == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "name": block["name"],
                "args": block["input"],
            })

    text = "\n".join(text_parts) if text_parts else ""

    if tool_calls:
        return [AIMessage(content=text, tool_calls=tool_calls)]
    else:
        return [AIMessage(content=text)]


def translate_user_message(msg: Message) -> list[BaseMessage]:
    """Translate user message, splitting tool_results into ToolMessages.

    Anthropic puts tool_result blocks in user message content[], but
    OpenAI/LangChain expects separate messages with role="tool".
    """
    # Handle string content
    if isinstance(msg.content, str):
        return [HumanMessage(content=msg.content)]

    tool_results = []
    other_content = []

    for block in msg.content:
        if block["type"] == "tool_result":
            tool_results.append(block)
        else:
            other_content.append(translate_content_block(block))

    result: list[BaseMessage] = []

    # Tool results become ToolMessage instances (must come first to maintain order)
    for tr in tool_results:
        content = tr.get("content", "")
        # Handle content that might be a list of blocks
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            content = "\n".join(text_parts)
        result.append(ToolMessage(
            content=str(content),
            tool_call_id=tr["tool_use_id"],
        ))

    # Remaining content becomes HumanMessage
    if other_content:
        result.append(HumanMessage(content=other_content))

    return result


# Instruction to inject into system prompt for better tool usage
TOOL_USAGE_INSTRUCTION = """
IMPORTANT: You have access to tools. When asked to create, write, or save files, you MUST use the Write tool - do not output file contents as text. When asked to read files, use the Read tool. When asked to run commands, use the Bash tool. Never say "I can't save files" or "copy this code" - use your tools instead.
"""


def translate_messages(
    messages: list[Message],
    system: str | list[TextBlock] | None,
    has_tools: bool = False,
) -> list[BaseMessage]:
    """Translate Anthropic messages to LangChain messages."""
    result: list[BaseMessage] = []

    # Build system message with optional tool usage instruction
    system_text = ""
    if system:
        # Handle both string and list of TextBlock formats
        if isinstance(system, str):
            system_text = system
        else:
            # Concatenate text from all system blocks
            system_text = "\n".join(block.text for block in system)

    # Inject tool usage instruction when tools are available
    if has_tools:
        system_text = TOOL_USAGE_INSTRUCTION + "\n" + system_text

    if system_text:
        result.append(SystemMessage(content=system_text))

    for msg in messages:
        if msg.role == "assistant":
            result.extend(translate_assistant_message(msg))
        else:  # user
            result.extend(translate_user_message(msg))

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
