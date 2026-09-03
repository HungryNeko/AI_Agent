"""File-backed skill helpers.

A skill is a small folder under data/skills. The entrypoint is SKILL.md, matching
Codex's simple file-first convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml"}


@dataclass(frozen=True)
class SkillDocument:
    name: str
    path: Path
    text: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def skills_root(root: str = "data/skills") -> Path:
    path = Path(root)
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def iter_skill_files(root: str = "data/skills") -> list[Path]:
    base = skills_root(root)
    if not base.exists():
        return []
    files: list[Path] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        files.append(path)
    return files


def load_skill(name: str, root: str = "data/skills", *, max_bytes: int = 200_000) -> SkillDocument:
    clean_name = name.strip().replace("\\", "/").strip("/")
    if not clean_name or "/" in clean_name or ".." in clean_name:
        raise ValueError("skill name must be a single folder name.")
    base = skills_root(root)
    path = (base / clean_name / "SKILL.md").resolve()
    path.relative_to(base)
    if not path.is_file():
        raise ValueError(f"skill does not exist: {clean_name}")
    return SkillDocument(name=clean_name, path=path, text=read_text_file(path, max_bytes=max_bytes))


def read_text_file(path: Path, *, max_bytes: int) -> str:
    if path.stat().st_size > max_bytes:
        return path.read_bytes()[:max_bytes].decode("utf-8", errors="replace")
    return path.read_text(encoding="utf-8", errors="replace")