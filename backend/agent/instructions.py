"""Small editable instruction file support."""

from __future__ import annotations

from pathlib import Path

from agent.config import PROJECT_ROOT

INSTRUCTION_PATH = PROJECT_ROOT / "data" / "instruction.md"

DEFAULT_INSTRUCTION = """
# Instruction

- Keep answers concise and actionable.
- Use `data/instruction.md` for short always-on behavior rules that the user or AI can edit.
- Use `data/memory` for durable user/project facts when the user asks to remember something.
- Use `data/skills/<name>/SKILL.md` for longer repeatable workflows.
- Use `data/knowledge` for larger reference material; search it with `rag`.
- Use the `history` tool when exact older conversation details are needed after compression.
""".strip()


def ensure_instruction_file() -> Path:
    INSTRUCTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not INSTRUCTION_PATH.exists():
        INSTRUCTION_PATH.write_text(DEFAULT_INSTRUCTION + "\n", encoding="utf-8")
    return INSTRUCTION_PATH


def load_instruction() -> str:
    path = ensure_instruction_file()
    return path.read_text(encoding="utf-8", errors="replace").strip()


def save_instruction(content: str) -> None:
    ensure_instruction_file().write_text(content.rstrip() + "\n", encoding="utf-8")
