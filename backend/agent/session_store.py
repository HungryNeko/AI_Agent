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
from agent import titles
from tools import parameterSave

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
        if needle and needle not in json.dumps(conversation, ensure_ascii=False).lower():
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
            "title": make_title(titles.heuristic_title(user_text) or user_text) if was_empty else conversation.get("title") or make_title(user_text),
            "title_auto": True if was_empty else conversation.get("title_auto", False),
            "updated_at": now,
            "summary": state.get("conversation_summary") or conversation.get("summary") or "",
            "events": events,
            "state": state,
            "parameters": parameterSave.list_parameters(conversation_id),
        }
    )
    write_conversation(conversation)
    return conversation


def set_auto_title(conversation_id: str, title: str) -> None:
    """Apply a generated title unless the user renamed the conversation meanwhile."""

    if not conversation_path(conversation_id).exists():
        return
    conversation = read_conversation(conversation_id)
    if not conversation.get("title_auto"):
        return
    conversation["title"] = make_title(title)
    write_conversation(conversation)


def rename_conversation(conversation_id: str, title: str) -> dict[str, Any]:
    conversation = read_conversation(conversation_id)
    conversation["title"] = make_title(title)
    conversation["title_auto"] = False
    conversation["updated_at"] = utc_now()
    write_conversation(conversation)
    return conversation


def set_plan_decision(conversation_id: str, decision: str) -> dict[str, Any]:
    if decision not in {"approved", "rejected"}:
        raise ValueError("plan decision must be approved or rejected")
    conversation = read_conversation(conversation_id)
    state = dict(conversation.get("state") or {})
    plan = dict(state.get("plan") or {})
    if not plan.get("content"):
        raise ValueError("conversation has no plan to review")
    plan["status"] = decision
    state["plan"] = plan
    now = utc_now()
    events = list(conversation.get("events") or [])
    events.append(
        {
            "type": "plan_decision",
            "decision": decision,
            "text": "Plan approved for Agent mode." if decision == "approved" else "Plan rejected.",
            "ts": now,
        }
    )
    conversation.update({"state": state, "events": events, "updated_at": now})
    write_conversation(conversation)
    return conversation


def delete_conversation(conversation_id: str) -> None:
    path = conversation_path(conversation_id)
    if path.exists():
        path.unlink()
    parameterSave.delete_all(conversation_id)


def branch_conversation(conversation_id: str, event_index: int, *, before: bool = False) -> dict[str, Any]:
    """Copy a conversation up to (and including) an event into a brand-new conversation.

    With ``before=True`` the event itself is left out, which is how forking from a
    user message works: the new conversation ends right before that message.
    """

    source = read_conversation(conversation_id)
    events = list(source.get("events") or [])
    if event_index < 0 or event_index >= len(events):
        raise ValueError("branch event index is out of range")
    cut = event_index if before else event_index + 1
    branch_events = events[:cut]
    branch_id = create_conversation_id()
    state = rewind_state(source, branch_events)
    entries = []
    for event in branch_events:
        if event.get("type") == "parameters_changed":
            entries = event.get("parameters", [])
    parameterSave.clone_parameters(conversation_id, branch_id, entries)
    state["conversation_id"] = branch_id
    state["parameters"] = parameterSave.list_parameters(branch_id)
    now = utc_now()
    branch = {
        "id": branch_id,
        "title": make_title(f"Branch: {source.get('title') or 'Untitled'}"),
        "title_auto": False,
        "created_at": now,
        "updated_at": now,
        "summary": state.get("conversation_summary", ""),
        "events": branch_events,
        "state": state,
        "parameters": state["parameters"],
        "parent_id": conversation_id,
        "parent_event_index": event_index,
    }
    write_conversation(branch)
    return branch


def truncate_conversation(conversation_id: str, event_index: int) -> dict[str, Any]:
    """Drop a user message and everything after it so that the turn can be re-run."""

    conversation = read_conversation(conversation_id)
    events = list(conversation.get("events") or [])
    if event_index < 0 or event_index >= len(events):
        raise ValueError("event index is out of range")
    if events[event_index].get("type") != "user":
        raise ValueError("only a user message can be edited")
    kept = events[:event_index]
    state = rewind_state(conversation, kept)
    state["conversation_id"] = conversation_id
    state["parameters"] = parameterSave.list_parameters(conversation_id)
    conversation.update(
        {
            "events": kept,
            "state": state,
            "summary": state.get("conversation_summary", ""),
            "updated_at": utc_now(),
        }
    )
    write_conversation(conversation)
    return conversation


def rewind_state(source: dict[str, Any], kept_events: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild the chat state as it was after ``kept_events``.

    When the saved message list still lines up with the events, it is cut at the
    exact turn boundary. Otherwise (compressed or mid-turn cuts) the kept events are
    folded into a summary, like a manual compression.
    """

    source_events = list(source.get("events") or [])
    state = dict(source.get("state") or {})
    sliced = slice_messages(state, source_events, kept_events)
    if sliced is not None:
        state["messages"] = sliced
        state["web_search_results"] = []
        state["rag_results"] = []
        state["tool_events"] = []
        state["response"] = ""
        state["conversation_summary"] = ""
    else:
        state["conversation_summary"] = ""
        state = compact_state(state, kept_events)
    if not any(event.get("type") == "user" for event in kept_events):
        state["tools_announced"] = False
    state["question_pending"] = None
    state["tool_error"] = ""
    plan = plan_from_events(kept_events)
    if plan:
        state["plan"] = plan
    else:
        state.pop("plan", None)
    return state


def slice_messages(
    state: dict[str, Any],
    source_events: list[dict[str, Any]],
    kept_events: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    messages = list(state.get("messages") or [])
    user_positions = [i for i, message in enumerate(messages) if isinstance(message, dict) and message.get("role") == "user"]
    total_turns = sum(1 for event in source_events if event.get("type") == "user")
    if state.get("conversation_summary") or len(user_positions) != total_turns:
        return None
    # The cut must sit on a turn boundary: the next source event starts a new turn.
    next_event = source_events[len(kept_events)] if len(kept_events) < len(source_events) else None
    if next_event is not None and next_event.get("type") != "user":
        return None
    keep_turns = sum(1 for event in kept_events if event.get("type") == "user")
    if keep_turns >= total_turns:
        return messages
    return messages[: user_positions[keep_turns]]


def plan_from_events(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    plan: dict[str, Any] | None = None
    for event in events:
        kind = event.get("type")
        if kind in {"plan_updated", "plan_ready"}:
            plan = {
                "name": event.get("name") or "implementation-plan",
                "content": event.get("content") or "",
                "status": event.get("status") or ("ready" if kind == "plan_ready" else "draft"),
            }
        elif kind == "plan_decision" and plan:
            plan = {**plan, "status": event.get("decision") or plan.get("status")}
    return plan


def clear_plan(conversation_id: str) -> dict[str, Any]:
    conversation = read_conversation(conversation_id)
    state = dict(conversation.get("state") or {})
    state["plan"] = None
    conversation.update({"state": state, "updated_at": utc_now()})
    write_conversation(conversation)
    return conversation


def read_conversation_page(
    conversation_id: str,
    *,
    limit: int,
    before: int | None = None,
    since: int | None = None,
    include_state: bool = True,
) -> dict[str, Any]:
    """Return a window of events ending before ``before`` (or at the end), newest last.

    ``since`` instead returns everything from that index on, which lets a client that
    already holds the older part refresh just the tail.
    """

    conversation = read_conversation(conversation_id)
    events = list(conversation.get("events") or [])
    total = len(events)
    end = total if before is None else max(0, min(int(before), total))
    start = max(0, end - max(1, int(limit)))
    if since is not None:
        end = total
        start = max(0, min(int(since), total))
    page = dict(conversation)
    page["events"] = events[start:end]
    page["event_offset"] = start
    page["total_events"] = total
    page["has_more"] = start > 0
    if not include_state:
        page.pop("state", None)
    return page


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
        if event_type not in {
            "user",
            "assistant",
            "assistant_progress",
            "tool_call",
            "question_required",
            "error",
            "approval_required",
            "ai_review",
        }:
            continue
        text = str(event.get("text") or event.get("question") or "").strip()
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
