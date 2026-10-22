"""AI reviewer for high-risk tool calls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agent.llm import complete_chat_once
from tools.request import ToolRequest


REVIEWER_PROMPT = (
    "You are a strict approval reviewer for an AI agent. "
    "Review only whether this high-risk tool call is justified by the user request. "
    "Approve only when intent is clear, scope is narrow, and no secret/destructive action is hidden. "
    "userMessage is the current turn; earlierUserMessages (oldest first) are the first and the most recent earlier "
    "requests of the conversation, so a short reply such as 'continue' can refer to them. "
    'Return only JSON: {"approved": true|false, "reason": "short reason"}.'
)


@dataclass(frozen=True)
class ReviewDecision:
    approved: bool
    reason: str


def review_tool_request(
    request: ToolRequest,
    *,
    user_message: str,
    assistant_message: dict[str, Any],
    model: str | None = None,
    earlier_user_messages: list[str] | None = None,
) -> ReviewDecision:
    payload = {
        "earlierUserMessages": list(earlier_user_messages or []),
        "userMessage": user_message,
        "assistantMessage": str(assistant_message.get("content") or ""),
        "tool": request.name,
        "target": describe_request(request),
    }
    response = complete_chat_once(
        [
            {
                "role": "system",
                "content": REVIEWER_PROMPT,
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        model=model,
        tools=None,
    )
    return parse_review_decision(str(response.get("content") or ""))


def parse_review_decision(text: str) -> ReviewDecision:
    # Models often wrap the object in a code fence or a sentence: read the outermost {...}.
    start, end = text.find("{"), text.rfind("}")
    try:
        parsed = json.loads(text[start : end + 1] if 0 <= start < end else text.strip())
    except json.JSONDecodeError:
        return ReviewDecision(False, "reviewer returned invalid JSON")
    if not isinstance(parsed, dict):
        return ReviewDecision(False, "reviewer returned non-object JSON")
    return ReviewDecision(
        approved=bool(parsed.get("approved")),
        reason=str(parsed.get("reason") or "").strip()[:500] or "no reason",
    )


def describe_request(request: ToolRequest) -> dict[str, Any]:
    if request.name == "rag" and request.rag_request:
        item = request.rag_request
        return {
            "action": item.action,
            "path": item.path,
            "name": item.name,
            "splitMode": item.split_mode,
            "chunkModel": item.chunk_model,
            "overwrite": item.overwrite,
        }
    if request.name == "fileEditor" and request.file_edit:
        edit = request.file_edit
        return {
            "action": edit.action,
            "path": edit.path,
            "oldText": edit.old_text[:1000],
            "newText": edit.new_text[:1000],
            "content": edit.content[:1000],
        }
    if request.name == "mcp" and request.mcp_request:
        item = request.mcp_request
        return {
            "action": item.action,
            "server": item.server,
            "tool": item.tool,
            "arguments": item.arguments,
        }
    if request.name == "automation" and request.automation_request:
        item = request.automation_request
        return {
            "action": item.action,
            "title": item.title,
            "prompt": item.prompt,
            "mcpServer": item.mcp_server,
            "mcpTool": item.mcp_tool,
            "mcpArguments": item.mcp_arguments,
            "mcpConfig": item.mcp_config,
            "schedule": item.schedule,
            "code": item.code[:1000],
        }
    if request.name == "settings" and request.settings_request:
        item = request.settings_request
        return {"action": item.action, "patch": item.patch, "settings": item.settings}
    if request.name == "python":
        return {"code": request.code[:1000]}
    return {"query": request.query, "url": request.url}


USER_MESSAGE_MARK = 'userMessage: "'
EARLIER_LIMIT = 600


def earlier_user_messages(messages: list[dict[str, Any]], current: str, recent: int = 3) -> list[str]:
    """The conversation's first request and the last few before the current one, as the user typed them."""

    texts = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
        content = str(content or "")
        marker = content.rfind(USER_MESSAGE_MARK)
        if marker < 0:
            continue  # tool budget notices and other system-made user messages
        text = content[marker + len(USER_MESSAGE_MARK) :].removesuffix('"')
        texts.append(text[:EARLIER_LIMIT])
    if texts and texts[-1] == current[:EARLIER_LIMIT]:
        texts.pop()  # the current turn is sent separately
    if len(texts) <= recent + 1:
        return texts
    return [texts[0], *texts[-recent:]]
