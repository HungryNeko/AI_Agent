"""AI reviewer for high-risk tool calls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agent.llm import complete_chat_once
from tools.request import ToolRequest


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
) -> ReviewDecision:
    payload = {
        "userMessage": user_message,
        "assistantMessage": str(assistant_message.get("content") or ""),
        "tool": request.name,
        "target": describe_request(request),
    }
    response = complete_chat_once(
        [
            {
                "role": "system",
                "content": (
                    "You are a strict approval reviewer for an AI agent. "
                    "Review only whether this high-risk tool call is justified by the user request. "
                    "Approve only when intent is clear, scope is narrow, and no secret/destructive action is hidden. "
                    "Return only JSON: {\"approved\": true|false, \"reason\": \"short reason\"}."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        model=model,
        tools=None,
    )
    return parse_review_decision(str(response.get("content") or ""))


def parse_review_decision(text: str) -> ReviewDecision:
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return ReviewDecision(False, "reviewer returned invalid JSON")
    if not isinstance(parsed, dict):
        return ReviewDecision(False, "reviewer returned non-object JSON")
    return ReviewDecision(
        approved=bool(parsed.get("approved")),
        reason=str(parsed.get("reason") or "").strip()[:500] or "no reason",
    )


def describe_request(request: ToolRequest) -> dict[str, Any]:
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
