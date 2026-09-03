"""Read saved conversation history for the model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent import session_store

HistoryAction = Literal["list", "read", "search"]


@dataclass(frozen=True)
class HistoryRequest:
    action: HistoryAction
    conversation_id: str = ""
    query: str = ""
    limit: int = 10


def execute(request: HistoryRequest) -> str:
    if request.action == "list":
        return format_list(session_store.list_conversations(limit=request.limit))
    if request.action == "search":
        return format_list(session_store.list_conversations(limit=request.limit, query=request.query))
    if request.action == "read":
        if not request.conversation_id:
            return 'toolError: "history read requires conversation_id."'
        return format_conversation(session_store.read_conversation(request.conversation_id))
    return f'toolError: "unknown history action: {request.action}"'


def format_list(items: list[dict[str, object]]) -> str:
    if not items:
        return 'historyResult: "no saved conversations"'
    lines = ["historyResult:"]
    for item in items:
        lines.append(
            f"- id={item.get('id')} updated={item.get('updated_at')} "
            f"title={item.get('title')} messages={item.get('message_count')}"
        )
    return "\n".join(lines)


def format_conversation(conversation: dict[str, object]) -> str:
    lines = [
        "historyResult:",
        f"id: {conversation.get('id')}",
        f"title: {conversation.get('title')}",
        f"updated: {conversation.get('updated_at')}",
    ]
    summary = str(conversation.get("summary") or "").strip()
    if summary:
        lines.append("summary:")
        lines.append(summary)
    lines.append("events:")
    for event in conversation.get("events") or []:
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        text = str(event.get("text") or "").strip()
        if text:
            lines.append(f"- {event_type}: {text[:1000]}")
    return "\n".join(lines)
