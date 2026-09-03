"""JSON conversation history storage.

This intentionally stays simple: one JSON file per conversation under
backend/runtime/conversations. It is enough for a learning project, easy to
inspect by hand, and reusable by the history tool.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent.config import PROJECT_ROOT

CONVERSATION_ROOT = PROJECT_ROOT / "backend" / "runtime" / "conversations"
SUMMARY_LIMIT = 12_000
TITLE_LIMIT = 80


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def create_conversation_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid4().hex[:8]}"


def list_conversations(*, limit: int = 50, query: str = "") -> list[dict[str, Any]]:
    CONVERSATION_ROOT.mkdir(parents=True, exist_ok=True)
    items = []
    needle = query.strip().lower()
    for path in CONVERSATION_ROOT.glob("*.json"):
        conversation = read_conversation(path.stem)
        haystack = json.dumps(conversation, ensure_ascii=False).lower()
        if needle and needle not in haystack:
            continue
        items.append(
            {
                "id": conversation.get("id") or path.stem,
                "title": conversation.get("title") or "Untitled",
                "created_at": conversation.get("created_at") or "",
                "updated_at": conversation.get("updated_at") or "",
                "summary": conversation.get("summary") or "",
                "message_count": len(conversation.get("events") or []),
            }
        )
    items.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return items[: max(1, int(limit))]


def read_conversation(conversation_id: str) -> dict[str, Any]:
    path = conversation_path(conversation_id)
    if not path.exists():
        return {
            "id": conversation_id,
            "title": "Untitled",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "summary": "",
            "events": [],
            "state": {},
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"conversation is not a JSON object: {conversation_id}")
    data.setdefault("id", conversation_id)
    data.setdefault("events", [])
    data.setdefault("state", {})
    return data


def save_turn(
    conversation_id: str,
    *,
    user_text: str,
    turn_events: list[dict[str, Any]],
    state: dict[str, Any],
    attachments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    now = utc_now()
    conversation = read_conversation(conversation_id)
    if not conversation.get("created_at"):
        conversation["created_at"] = now
    events = list(conversation.get("events") or [])
    was_empty = not events
    user_event: dict[str, Any] = {"type": "user", "text": user_text, "ts": now}
    if attachments:
        user_event["attachments"] = attachments
    events.append(user_event)
    for event in turn_events:
        public_event = {key: value for key, value in event.items() if key != "state"}
        public_event.setdefault("ts", now)
        events.append(public_event)

    conversation.update(
        {
            "id": conversation_id,
            "title": make_turn_title(user_text, turn_events) if was_empty else conversation.get("title") or make_title(user_text),
            "updated_at": now,
            "summary": state.get("conversation_summary") or conversation.get("summary") or "",
            "events": events,
            "state": state,
        }
    )
    write_conversation(conversation)
    return conversation


def rename_conversation(conversation_id: str, title: str) -> dict[str, Any]:
    conversation = read_conversation(conversation_id)
    conversation["title"] = make_title(title)
    conversation["updated_at"] = utc_now()
    write_conversation(conversation)
    return conversation


def delete_conversation(conversation_id: str) -> None:
    path = conversation_path(conversation_id)
    if path.exists():
        path.unlink()


def compress_conversation(conversation_id: str) -> dict[str, Any]:
    conversation = read_conversation(conversation_id)
    state = dict(conversation.get("state") or {})
    compacted = compact_state(state, conversation.get("events") or [])
    conversation["state"] = compacted
    conversation["summary"] = compacted.get("conversation_summary", "")
    conversation["updated_at"] = utc_now()
    write_conversation(conversation)
    return conversation


def compact_state(state: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    summary_parts = []
    existing = str(state.get("conversation_summary") or "").strip()
    if existing:
        summary_parts.append(existing)
    summary_parts.append(events_to_transcript(events))
    summary = trim_text("\n\n".join(part for part in summary_parts if part), SUMMARY_LIMIT)

    system_messages = [
        message
        for message in state.get("messages") or []
        if isinstance(message, dict) and message.get("role") == "system"
    ]
    compacted = dict(state)
    compacted["messages"] = system_messages[:1]
    compacted["conversation_summary"] = summary
    compacted["web_search_results"] = []
    compacted["rag_results"] = []
    compacted["tool_events"] = []
    compacted["response"] = ""
    return compacted


def events_to_transcript(events: list[dict[str, Any]]) -> str:
    lines = []
    for event in events:
        event_type = str(event.get("type") or "event")
        if event_type not in {"user", "assistant", "assistant_progress", "tool_call", "error", "approval_required", "ai_review"}:
            continue
        text = str(event.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"{event_type}: {text}")
    return "\n".join(lines)


def conversation_path(conversation_id: str) -> Path:
    clean = conversation_id.strip()
    if not clean or "/" in clean or "\\" in clean or ".." in clean:
        raise ValueError("invalid conversation id")
    path = (CONVERSATION_ROOT / f"{clean}.json").resolve()
    root = CONVERSATION_ROOT.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("invalid conversation id") from exc
    return path


def write_conversation(conversation: dict[str, Any]) -> None:
    CONVERSATION_ROOT.mkdir(parents=True, exist_ok=True)
    conversation_id = str(conversation.get("id") or create_conversation_id())
    path = conversation_path(conversation_id)
    path.write_text(json.dumps(conversation, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def make_title(text: str) -> str:
    clean = sanitize_title_text(text)
    return trim_text(clean, TITLE_LIMIT) or "Untitled"


def make_turn_title(user_text: str, turn_events: list[dict[str, Any]]) -> str:
    for event in turn_events:
        if event.get("type") == "assistant":
            text = str(event.get("text") or "").strip()
            if text:
                for line in text.splitlines():
                    title = make_title(line)
                    if title != "Untitled":
                        return title
    return make_title(user_text)


def sanitize_title_text(text: str) -> str:
    clean = re.sub(r"!\[[^\]]*]\([^)]*\)", " ", text)
    clean = re.sub(r"\[[^\]]+]\([^)]*\)", " ", clean)
    clean = re.sub(r"https?://\S+", " ", clean)
    clean = re.sub(r"`{1,3}[^`]*`{1,3}", " ", clean)
    clean = clean.replace("#", " ").replace("*", " ").replace("_", " ")
    clean = " ".join(clean.split()).strip(" -:|,.;")
    if not clean or clean.lower().startswith(("backend/runtime/uploads", "data:image/")):
        return ""
    return clean


def trim_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20].rstrip() + "\n...[truncated]"
