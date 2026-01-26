# Tool Call Translation Fixes Design

## Problem

The LLM proxy fails to properly translate tool calls between Anthropic and OpenAI formats, causing:

1. **Repeated tool calls** - Model keeps calling the same tool because it doesn't see prior results
2. **Lost tool history** - Assistant tool_use and user tool_result not translated correctly
3. **404 noise** - Missing endpoints for `/v1/messages/count_tokens` and `/api/event_logging/batch`

## Root Cause

Anthropic and OpenAI have fundamentally different conversation structures for tool calls:

**Anthropic format:**
```
User: "do something"
Assistant: {content: [text_block, tool_use_block, tool_use_block]}
User: {content: [tool_result_block, tool_result_block, text_block]}
```

**OpenAI format:**
```
User: "do something"
Assistant: {content: "text", tool_calls: [{id, function: {name, arguments}}]}
Tool: {role: "tool", tool_call_id: "...", content: "result"}
Tool: {role: "tool", tool_call_id: "...", content: "result"}
User: "follow up text"
```

**Current code flaws in `request.py`:**
- `translate_message()` treats assistant tool_use as regular content
- `translate_content_block()` converts tool_result to plain text, losing tool_call_id
- No message splitting for tool results into separate ToolMessage instances

## Solution

### 1. Fix Assistant Message Translation

Extract `tool_use` blocks and convert to LangChain `AIMessage` with `tool_calls`:

```python
def translate_assistant_message(msg: Message) -> list[BaseMessage]:
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
                "args": block["input"]
            })

    text = "\n".join(text_parts) if text_parts else ""
    return [AIMessage(content=text, tool_calls=tool_calls if tool_calls else None)]
```

### 2. Fix User Message Translation

Split `tool_result` blocks into separate `ToolMessage` instances:

```python
def translate_user_message(msg: Message) -> list[BaseMessage]:
    if isinstance(msg.content, str):
        return [HumanMessage(content=msg.content)]

    tool_results = []
    other_content = []

    for block in msg.content:
        if block["type"] == "tool_result":
            tool_results.append(block)
        else:
            other_content.append(translate_content_block(block))

    result = []

    # Tool results become ToolMessage instances
    for tr in tool_results:
        content = tr.get("content", "")
        if isinstance(content, list):
            content = "\n".join(b.get("text", "") for b in content)
        result.append(ToolMessage(
            content=str(content),
            tool_call_id=tr["tool_use_id"]
        ))

    # Remaining content becomes HumanMessage
    if other_content:
        result.append(HumanMessage(content=other_content))

    return result
```

### 3. Add Missing Endpoints

**Token counting (stub):**
```python
@router.post("/v1/messages/count_tokens")
async def count_tokens(request: MessagesRequest):
    # Rough estimate - Claude Code handles inaccurate counts gracefully
    estimate = len(json.dumps(request.model_dump())) // 4
    return {"input_tokens": estimate}
```

**Event logging (acknowledge and discard):**
```python
@router.post("/api/event_logging/batch")
async def event_logging_batch():
    return {"status": "ok"}
```

## Tasks

1. Fix tool_use translation in assistant messages
2. Fix tool_result translation in user messages
3. Add /v1/messages/count_tokens endpoint
4. Add /api/event_logging/batch endpoint
5. Test tool call round-trip with Claude Code

## Success Criteria

- No repeated tool calls in Claude Code
- Tool results properly linked by tool_call_id
- No more 404 errors for count_tokens and event_logging
- Full skill/plugin workflows complete successfully
