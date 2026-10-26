"""Project paths and small file helpers shared by the RAG modules.

Every module calls `paths.project_root()` through this module, so tests patch one place.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_project_path(path_text: str, *, allow_missing: bool = False) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = project_root() / path
    resolved = path.resolve() if path.exists() else path.parent.resolve() / path.name
    if not allow_missing and not resolved.exists():
        raise ValueError(f"path does not exist: {path_text}")
    return resolved


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root())).replace("\\", "/")
    except ValueError:
        return str(path)


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def read_text_file(path: Path, *, max_bytes: int) -> str:
    if path.stat().st_size > max_bytes:
        return path.read_bytes()[:max_bytes].decode("utf-8", errors="replace")
    return path.read_text(encoding="utf-8", errors="replace")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_atomic(path: Path, data: bytes) -> None:
    """Write a file so that a concurrent reader sees the old or the new content, never half of it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
