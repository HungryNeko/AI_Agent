"""Security levels for what the agent may do on the machine it runs on.

    L1  read-only and harmless: reading the agent's own data, RAG, history, web search. No code runs.
    L2  every tool, but anything that runs code (python, custom tools, stdio MCP servers) runs as the
        unprivileged OS user `agentcode`, so file permissions - not Python checks - keep it away from
        secrets, configuration and source. Needs Linux and a backend started as root (the Docker setup).
    L3  everything runs as the backend itself. In Docker the container is the outer boundary.

The level lives in the app settings but can only be changed through `app_settings.set_security_level`
(the UI slider's endpoint); the settings tool, the generic settings API and the model cannot touch it.
It is read at the moment a tool runs, so lowering it takes effect immediately, mid-turn included.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

LEVELS = ("L1", "L2", "L3")
DEFAULT_LEVEL = "L2"  # agent code runs as the unprivileged user; where L2 is unavailable it refuses code (see l2_status)
LOW_USER = "agentcode"
LOW_UID = 10001

L1_TOOLS = frozenset(
    {
        "webSearch",
        "nativeWebSearch",
        "webFetch",
        "curl",
        "rag",
        "history",
        "fileReader",
        "model",
        "plan",
        "question",
        "approval",
    }
)

# What fileReader may open at L1: the agent's own data and the files the user uploaded.
AGENT_READ_ROOTS = (
    "backend/runtime/uploads",
    "backend/runtime/user_data",
    "backend/runtime/conversations",
    "backend/runtime/python_runs",
    "backend/runtime/mcp_artifacts",
    "backend/runtime/subagents",
    "data/knowledge",
    "data/memory",
    "data/skills",
    "data/plans",
)

# What the in-process file editor may not touch below L3: code, configuration and deployment files.
EDITOR_PROTECTED = (
    "backend/agent/",
    "backend/tools/",
    "backend/prompts/",
    "backend/scripts/",
    "backend/pyproject.toml",
    "data/settings",
    "data/api_configs",
    "data/mcp/",
    "data/instruction.md",
    "data/custom_tools/",
    ".github/",
    "docker/",
    "docker-compose.yml",
    "scripts/",
)

# Files that hold secrets: owner-only so the unprivileged user cannot read them.
SECRET_FILES = (
    ".env",
    "backend/.env",
    "data/api_configs.json",
    "data/api_configs.local.json",
    "data/mcp/servers.json",
    "data/mcp/servers.local.json",
)


class SecurityError(PermissionError):
    pass


def normalize_level(value: Any) -> str:
    level = str(value or "").strip().upper()
    return level if level in LEVELS else DEFAULT_LEVEL


def current_level() -> str:
    """The level in force right now. Unreadable settings fail closed to L1."""

    try:
        from agent.app_settings import load_app_settings

        return normalize_level((load_app_settings().get("security") or {}).get("level"))
    except Exception:  # noqa: BLE001 - never run with more rights because the settings file is broken
        return "L1"


def tool_allowed(tool: str, level: str) -> bool:
    return level != "L1" or tool in L1_TOOLS


def require_tool(tool: str) -> None:
    level = current_level()
    if not tool_allowed(tool, level):
        raise SecurityError(f"{tool} is not available at security level {level} (read-only). Raise the level in the UI to use it.")


# ---------------------------------------------------------------- running code as the low user


def low_user() -> tuple[int, int] | None:
    """(uid, gid) of the unprivileged user, or None where it cannot be used (not Linux/root, or missing)."""

    if os.name != "posix" or not hasattr(os, "geteuid") or os.geteuid() != 0:
        return None
    try:
        import pwd

        entry = pwd.getpwnam(LOW_USER)
    except (ImportError, KeyError):
        return None
    return entry.pw_uid, entry.pw_gid


def l2_status() -> dict[str, Any]:
    if low_user() is not None:
        return {"available": True, "reason": ""}
    if os.name != "posix":
        return {"available": False, "reason": "L2 needs Linux (the Docker deployment); it is not available on Windows."}
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return {"available": False, "reason": "L2 needs the backend to start as root so it can run code as another user."}
    return {"available": False, "reason": f"the unprivileged user {LOW_USER} does not exist yet."}


def code_launch_options() -> dict[str, Any]:
    """Popen keyword arguments for starting code on behalf of the agent at the current level."""

    level = current_level()
    if level == "L1":
        raise SecurityError("code cannot run at security level L1 (read-only). Raise the level in the UI.")
    if level == "L3":
        return {}
    info = low_user()
    if info is None:
        raise SecurityError("security level L2 cannot run code here: " + l2_status()["reason"] + " Use L1 or L3.")
    uid, gid = info
    return {"user": uid, "group": gid, "extra_groups": []}


def hand_over(*paths: Path) -> None:
    """Give the unprivileged user ownership of directories it has to write to (L2 only)."""

    info = low_user()
    if info is None:
        return
    uid, gid = info
    for root in paths:
        if not root.exists():
            continue
        os.chown(root, uid, gid)
        for dirpath, dirnames, filenames in os.walk(root):
            for name in [*dirnames, *filenames]:
                try:
                    os.chown(os.path.join(dirpath, name), uid, gid, follow_symlinks=False)
                except OSError:
                    pass


def ensure_low_user() -> bool:
    """Create the unprivileged user if the backend is root and it does not exist. True when it can be used."""

    if low_user() is not None:
        return True
    if os.name != "posix" or not hasattr(os, "geteuid") or os.geteuid() != 0:
        return False
    for command in (
        ["useradd", "--system", "--no-create-home", "--uid", str(LOW_UID), "--shell", "/usr/sbin/nologin", LOW_USER],
        ["adduser", "--system", "--no-create-home", "--uid", str(LOW_UID), LOW_USER],
    ):
        try:
            subprocess.run(command, capture_output=True, check=True, timeout=20)
            break
        except (OSError, subprocess.SubprocessError):
            continue
    return low_user() is not None


def protect_file(path: Path) -> None:
    """Make a secrets file readable by its owner only."""

    if os.name != "posix":
        return
    try:
        if path.exists():
            os.chmod(path, 0o600)
    except OSError:
        pass


def harden_permissions(root: Path) -> list[str]:
    """Owner-only on every secrets file (also follows the symlinks the Docker entrypoint creates)."""

    changed = []
    for relative in SECRET_FILES:
        path = root / relative
        if path.exists():
            protect_file(path.resolve())
            changed.append(relative)
    return changed


def startup(project_root: Path) -> dict[str, Any]:
    ready = ensure_low_user()
    changed = harden_permissions(project_root) if ready else []
    return {"low_user": ready, "protected": changed, "level": current_level()}


# ---------------------------------------------------------------- path rules


def _clean(relative: str) -> str:
    clean = relative.replace("\\", "/")
    while clean.startswith("./"):
        clean = clean[2:]
    return clean.lstrip("/")


def read_allowed_at_l1(relative: str) -> bool:
    clean = _clean(relative)
    return any(clean == root or clean.startswith(root + "/") for root in AGENT_READ_ROOTS)


def editor_blocked(relative: str) -> bool:
    """True for files the in-process editor must not write below L3."""

    clean = _clean(relative).lower()
    return any(clean == rule.rstrip("/") or clean.startswith(rule) for rule in EDITOR_PROTECTED)
