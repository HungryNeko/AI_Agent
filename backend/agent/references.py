"""Resolve lightweight @ references from user messages."""

from __future__ import annotations

import re
from pathlib import Path

from agent import session_store
from agent.config import PROJECT_ROOT

TOKEN_RE = re.compile(r"@(tool|file|history):([^\s]+)")
ALLOWED_FILE_ROOTS = [
    PROJECT_ROOT / "data" / "instruction.md",
    PROJECT_ROOT / "data" / "memory",
    PROJECT_ROOT / "data" / "skills",
    PROJECT_ROOT / "data" / "knowledge",
    PROJECT_ROOT / "backend" / "runtime" / "uploads",
]
MAX_REFERENCE_CHARS = 8_000


def resolve_reference_context(message: str) -> str:
    items = []
    for kind, value in TOKEN_RE.findall(message):
        if kind == "tool":
            items.append(f"requestedTool: {value}")
        elif kind == "file":
            resolved = resolve_file(value)
            if resolved and resolved.is_file():
                text = resolved.read_text(encoding="utf-8", errors="replace")
                items.append(f"referencedFile: {relative_to_project(resolved)}\n{trim(text)}")
        elif kind == "history":
            conversation = session_store.read_conversation(value)
            summary = session_store.events_to_transcript(conversation.get("events") or [])
            items.append(f"referencedConversation: {value}\n{trim(summary)}")
    if not items:
        return ""
    return "references:\n" + "\n\n".join(items)


def resolve_file(path_text: str) -> Path | None:
    raw = Path(path_text)
    path = raw if raw.is_absolute() else PROJECT_ROOT / raw
    resolved = path.resolve() if path.exists() else path.parent.resolve() / path.name
    for root in ALLOWED_FILE_ROOTS:
        root_resolved = root.resolve()
        if resolved == root_resolved or is_relative_to(resolved, root_resolved):
            return resolved
    return None


def trim(text: str) -> str:
    clean = text.strip()
    if len(clean) <= MAX_REFERENCE_CHARS:
        return clean
    return clean[: MAX_REFERENCE_CHARS - 20].rstrip() + "\n...[truncated]"


def relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
