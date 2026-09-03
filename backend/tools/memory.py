"""File-backed long-term memory helpers."""

from __future__ import annotations

from pathlib import Path

TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def memory_root(root: str = "data/memory") -> Path:
    path = Path(root)
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def iter_memory_files(root: str = "data/memory") -> list[Path]:
    base = memory_root(root)
    if not base.exists():
        return []
    return [path for path in sorted(base.rglob("*")) if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES]


def read_text_file(path: Path, *, max_bytes: int) -> str:
    if path.stat().st_size > max_bytes:
        return path.read_bytes()[:max_bytes].decode("utf-8", errors="replace")
    return path.read_text(encoding="utf-8", errors="replace")