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
    if block["type"] == "image":
        source = block["source"]
        if source["type"] == "base64":
            return {"type": "image_url", "image_url": {"url": f"data:{source['media_type']};base64,{source['data']}"}}
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return block


def translate_assistant_message(msg: Message) -> list[BaseMessage]:
    """Translate assistant message, extracting tool_use into tool_calls."""
    if isinstance(msg.content, str):
        return [AIMessage(content=msg.content)]

    text_parts = []
    tool_calls = []
    for block in msg.content:
        if block["type"] == "text":
            text_parts.append(block["text"])
        elif block["type"] == "tool_use":
            tool_calls.append({"id": block["id"], "name": block["name"], "args": block["input"]})

    text = "\n".join(text_parts)
    return [AIMessage(content=text, tool_calls=tool_calls)] if tool_calls else [AIMessage(content=text)]


def _extract_tool_result_content(content: str | list) -> str:
    """Extract text content from a tool result."""
    if isinstance(content, str):
        return content
    text_parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text_parts.append(item.get("text", ""))
        elif isinstance(item, str):
            text_parts.append(item)
    return "\n".join(text_parts)


def translate_user_message(msg: Message) -> list[BaseMessage]:
    """Translate user message, splitting tool_results into ToolMessages."""
    if isinstance(msg.content, str):
        return [HumanMessage(content=msg.content)]

    result: list[BaseMessage] = []
    other_content = []

    for block in msg.content:
        if block["type"] == "tool_result":
            content = _extract_tool_result_content(block.get("content", ""))
            result.append(ToolMessage(content=content, tool_call_id=block["tool_use_id"]))
        else:
            other_content.append(translate_content_block(block))

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

    # Build system message
    if system:
        system_text = system if isinstance(system, str) else "\n".join(block.text for block in system)
        if has_tools:
            system_text = TOOL_USAGE_INSTRUCTION + "\n" + system_text
        result.append(SystemMessage(content=system_text))
    elif has_tools:
        result.append(SystemMessage(content=TOOL_USAGE_INSTRUCTION))

    for msg in messages:
        translator = translate_assistant_message if msg.role == "assistant" else translate_user_message
        result.extend(translator(msg))

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
    kwargs: dict[str, Any] = {"max_tokens": request.max_tokens}
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.stop_sequences:
        kwargs["stop"] = request.stop_sequences
    return kwargs
