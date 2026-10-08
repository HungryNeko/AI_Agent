"""Session-plan validation and explicit Markdown export helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Any

from agent.config import PROJECT_ROOT

PLAN_ROOT = PROJECT_ROOT / "data" / "plans"


@dataclass(frozen=True)
class PlanRequest:
    action: Literal["update", "finalize"]
    name: str = ""
    content: str = ""


def execute(request: PlanRequest) -> str:
    if request.action in {"update", "finalize"}:
        if not request.content.strip():
            raise ValueError("plan content is required")
        status = "ready" if request.action == "finalize" else "draft"
        return format_result(
            {
                "name": clean_name(request.name),
                "content": request.content.strip(),
                "status": status,
                "storage": "conversation",
            }
        )
    raise ValueError(f"unknown plan action: {request.action}")


def save_copy(name: str, content: str) -> dict[str, str]:
    """Persist a user-approved copy outside the active conversation."""

    if not content.strip():
        raise ValueError("plan content is required")
    PLAN_ROOT.mkdir(parents=True, exist_ok=True)
    path = plan_path(name)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return {"name": path.stem, "path": relative(path), "status": "saved"}


def plan_path(name: str) -> Path:
    clean = clean_name(name)
    return PLAN_ROOT / f"{clean[:80]}.md"


def clean_name(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-").lower()
    return clean or "implementation-plan"


def format_result(payload: dict[str, Any]) -> str:
    return "planResult:\n" + json.dumps(payload, ensure_ascii=False)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
