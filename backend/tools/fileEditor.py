"""Project-scoped file editing tool for model-requested code changes."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tools.settings import FileEditorSettings


Action = Literal["list", "read", "write", "replace", "insertAfter", "insertBefore", "append"]
WRITE_ACTIONS = {"write", "replace", "insertAfter", "insertBefore", "append"}
PROTECTED_PARTS = {".git", "backend/runtime", "node_modules", "__pycache__", ".venv", ".conda"}
PROTECTED_NAMES = {".env", ".env.local", ".env.production"}
MAX_DIFF_CHARS = 20_000


@dataclass(frozen=True)
class FileEditRequest:
    action: Action
    path: str = ""
    content: str = ""
    old_text: str = ""
    new_text: str = ""
    anchor: str = ""
    pattern: str = "**/*"
    overwrite: bool = False
    replace_all: bool = False
    start_line: int | None = None
    end_line: int | None = None
    max_results: int = 80


def execute(request: FileEditRequest, settings: FileEditorSettings) -> dict[str, Any]:
    if request.action == "list":
        return list_files(request, settings)
    if request.action == "read":
        return read_file(request, settings)
    if request.action == "write":
        return write_file(request, settings)
    if request.action == "replace":
        return replace_in_file(request, settings)
    if request.action == "insertAfter":
        return insert_by_anchor(request, settings, after=True)
    if request.action == "insertBefore":
        return insert_by_anchor(request, settings, after=False)
    if request.action == "append":
        return append_file(request, settings)
    raise ValueError(f"unknown fileEditor action: {request.action}")


def list_files(request: FileEditRequest, settings: FileEditorSettings) -> dict[str, Any]:
    root = resolve_root(settings)
    base = resolve_path(request.path or ".", settings, allow_missing=False)
    if not base.is_dir():
        raise ValueError("list path must be a directory.")
    max_results = max(1, min(request.max_results, settings.max_list_results))
    files: list[str] = []
    for path in sorted(base.glob(request.pattern or "**/*")):
        if len(files) >= max_results:
            break
        if path.is_file() and is_visible_file(path, root):
            files.append(relative_path(path, root))
    return {"action": "list", "path": relative_path(base, root), "files": files, "truncated": len(files) >= max_results}


def read_file(request: FileEditRequest, settings: FileEditorSettings) -> dict[str, Any]:
    root = resolve_root(settings)
    path = resolve_path(request.path, settings, allow_missing=False)
    ensure_regular_file(path)
    if path.stat().st_size > settings.max_file_bytes:
        raise ValueError(f"file is too large to read: {path.stat().st_size} bytes")

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    start = max(1, request.start_line or 1)
    end = request.end_line or len(lines)
    if end < start:
        raise ValueError("endLine must be greater than or equal to startLine.")
    selected = "".join(lines[start - 1 : end])
    truncated = len(selected) > settings.max_read_chars
    if truncated:
        selected = selected[: settings.max_read_chars]
    return {
        "action": "read",
        "path": relative_path(path, root),
        "startLine": start,
        "endLine": min(end, len(lines)),
        "totalLines": len(lines),
        "truncated": truncated,
        "content": selected,
    }


def write_file(request: FileEditRequest, settings: FileEditorSettings) -> dict[str, Any]:
    root = resolve_root(settings)
    path = resolve_path(request.path, settings, allow_missing=True)
    exists = path.exists()
    if exists and not request.overwrite:
        raise ValueError("file already exists; set overwrite=true to replace it.")
    ensure_parent_allowed(path, settings)
    ensure_content_size(request.content, settings)
    old_text = path.read_text(encoding="utf-8") if exists else ""
    return finish_write_action(
        request,
        settings,
        path,
        old_text,
        request.content,
        {"created": not exists, "bytes": len(request.content.encode("utf-8"))},
        root=root,
    )


def replace_in_file(request: FileEditRequest, settings: FileEditorSettings) -> dict[str, Any]:
    root = resolve_root(settings)
    path = resolve_path(request.path, settings, allow_missing=False)
    ensure_regular_file(path)
    if not request.old_text:
        raise ValueError("replace requires oldText.")
    ensure_content_size(request.new_text, settings)
    text = path.read_text(encoding="utf-8")
    count = text.count(request.old_text)
    if count == 0:
        raise ValueError("oldText was not found.")
    if count > 1 and not request.replace_all:
        raise ValueError(f"oldText matched {count} times; use a more specific anchor or replaceAll=true.")
    replacements = count if request.replace_all else 1
    new_text = text.replace(request.old_text, request.new_text, -1 if request.replace_all else 1)
    return finish_write_action(
        request,
        settings,
        path,
        text,
        new_text,
        {"replacements": replacements},
        root=root,
    )


def insert_by_anchor(request: FileEditRequest, settings: FileEditorSettings, *, after: bool) -> dict[str, Any]:
    root = resolve_root(settings)
    path = resolve_path(request.path, settings, allow_missing=False)
    ensure_regular_file(path)
    if not request.anchor:
        raise ValueError(f"{request.action} requires anchor.")
    ensure_content_size(request.content, settings)
    text = path.read_text(encoding="utf-8")
    count = text.count(request.anchor)
    if count == 0:
        raise ValueError("anchor was not found.")
    if count > 1:
        raise ValueError(f"anchor matched {count} times; use a more specific anchor.")
    replacement = request.anchor + request.content if after else request.content + request.anchor
    new_text = text.replace(request.anchor, replacement, 1)
    return finish_write_action(
        request,
        settings,
        path,
        text,
        new_text,
        {"insertions": 1},
        root=root,
    )


def append_file(request: FileEditRequest, settings: FileEditorSettings) -> dict[str, Any]:
    root = resolve_root(settings)
    path = resolve_path(request.path, settings, allow_missing=False)
    ensure_regular_file(path)
    ensure_content_size(request.content, settings)
    text = path.read_text(encoding="utf-8")
    new_text = text + request.content
    return finish_write_action(
        request,
        settings,
        path,
        text,
        new_text,
        {"bytes": len(request.content.encode("utf-8"))},
        root=root,
    )


def finish_write_action(
    request: FileEditRequest,
    settings: FileEditorSettings,
    path: Path,
    old_text: str,
    new_text: str,
    details: dict[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    rel_path = relative_path(path, root)
    base: dict[str, Any] = {"action": request.action, "path": rel_path, **details}

    if settings.approval == "readOnly":
        return {
            **base,
            "applied": False,
            "approvalRequired": True,
            "reason": "fileEditor approval is readOnly; write actions are disabled.",
        }

    if settings.approval == "manual":
        return {
            **base,
            "applied": False,
            "approvalRequired": True,
            "reason": "manual approval required before applying this edit.",
            "diff": make_unified_diff(rel_path, old_text, new_text),
        }

    if settings.approval == "auto":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_text, encoding="utf-8")
        return {**base, "applied": True, "approvalRequired": False}

    raise ValueError("unknown fileEditor approval setting.")


def make_unified_diff(path: str, old_text: str, new_text: str) -> str:
    diff = "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"{path} (before)",
            tofile=f"{path} (after)",
        )
    )
    if not diff:
        return "no text changes"
    if len(diff) > MAX_DIFF_CHARS:
        return diff[:MAX_DIFF_CHARS] + "\n...diff truncated..."
    return diff


def resolve_root(settings: FileEditorSettings) -> Path:
    if settings.root:
        root = Path(settings.root)
        if not root.is_absolute():
            root = project_root() / root
        return root.resolve()
    return project_root()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_text: str, settings: FileEditorSettings, *, allow_missing: bool) -> Path:
    if not path_text.strip():
        raise ValueError("path is required.")
    root = resolve_root(settings)
    raw = Path(path_text)
    path = raw if raw.is_absolute() else root / raw
    resolved_parent = path.parent.resolve()
    resolved = path.resolve() if path.exists() else resolved_parent / path.name
    ensure_within_root(resolved, root)
    ensure_not_protected(resolved, root)
    if not allow_missing and not resolved.exists():
        raise ValueError(f"path does not exist: {path_text}")
    return resolved


def ensure_within_root(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("path must stay inside the configured file editor root.") from exc


def ensure_not_protected(path: Path, root: Path) -> None:
    rel = relative_path(path, root).replace("\\", "/")
    parts = set(rel.split("/"))
    if path.name.lower() in PROTECTED_NAMES:
        raise ValueError("protected file cannot be edited.")
    if any(part in parts for part in PROTECTED_PARTS):
        raise ValueError("protected path cannot be edited.")
    if rel.startswith("backend/runtime/"):
        raise ValueError("protected runtime path cannot be edited.")


def ensure_regular_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError("path must be a file.")


def ensure_parent_allowed(path: Path, settings: FileEditorSettings) -> None:
    root = resolve_root(settings)
    ensure_within_root(path.parent, root)
    ensure_not_protected(path, root)


def ensure_content_size(content: str, settings: FileEditorSettings) -> None:
    size = len(content.encode("utf-8"))
    if size > settings.max_write_bytes:
        raise ValueError(f"content is too large to write: {size} bytes")


def is_visible_file(path: Path, root: Path) -> bool:
    try:
        ensure_not_protected(path, root)
    except ValueError:
        return False
    return True


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)
