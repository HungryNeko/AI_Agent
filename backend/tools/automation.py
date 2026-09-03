"""Small automation tool for model-managed workflows."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

from agent.config import PROJECT_ROOT
from tools import mcp
from tools import python as python_tool
from tools.mcp import McpRequest
from tools.settings import ToolSettings

AutomationAction = Literal["script", "mcp", "configureMcp", "reminder", "llm"]
CURRENT_AUTOMATION_PATH: ContextVar[Path | None] = ContextVar("current_automation_path", default=None)


@dataclass(frozen=True)
class AutomationRequest:
    action: AutomationAction
    title: str = ""
    prompt: str = ""
    code: str = ""
    mcp_server: str = ""
    mcp_tool: str = ""
    mcp_arguments: dict[str, Any] = field(default_factory=dict)
    mcp_config: dict[str, Any] = field(default_factory=dict)
    schedule: dict[str, Any] = field(default_factory=dict)
    target_automation: str = ""
    create_new: bool = False


@contextmanager
def automation_context(path: Path) -> Iterator[None]:
    token = CURRENT_AUTOMATION_PATH.set(path.resolve())
    try:
        yield
    finally:
        CURRENT_AUTOMATION_PATH.reset(token)


def execute(request: AutomationRequest, settings: ToolSettings) -> dict[str, Any]:
    if not settings.automation.can_model_call:
        raise ValueError("automation is disabled for the model.")
    if request.action == "script":
        return {"action": "script", "result": python_tool.run(request.code, settings.python)}
    if request.action == "mcp":
        response = mcp.execute(
            McpRequest(
                action="callTool",
                server=request.mcp_server,
                tool=request.mcp_tool,
                arguments=request.mcp_arguments,
            ),
            settings.mcp,
        )
        return {"action": "mcp", "result": response}
    if request.action == "configureMcp":
        return configure_mcp(request.mcp_config, settings)
    if request.action == "reminder":
        return save_scheduled_automation(request, settings)
    if request.action == "llm":
        return save_scheduled_automation(request, settings)
    raise ValueError(f"unknown automation action: {request.action}")


def configure_mcp(raw_config: dict[str, Any], settings: ToolSettings) -> dict[str, Any]:
    name = clean_name(str(raw_config.get("name") or ""))
    transport = str(raw_config.get("transport") or "streamable_http")
    if transport not in {"streamable_http", "stdio"}:
        raise ValueError("mcp transport must be streamable_http or stdio.")
    server: dict[str, Any] = {
        "enabled": bool(raw_config.get("enabled", True)),
        "transport": transport,
    }
    if transport == "streamable_http":
        server["url"] = str(raw_config.get("url") or "").strip()
        server["headers"] = dict(raw_config.get("headers") or {})
        server["timeout"] = float(raw_config.get("timeout") or 5)
        server["sse_read_timeout"] = float(raw_config.get("sse_read_timeout") or 300)
        if not server["url"]:
            raise ValueError("streamable_http mcp config requires url.")
    else:
        server["command"] = str(raw_config.get("command") or "").strip()
        server["args"] = list(raw_config.get("args") or [])
        server["env"] = dict(raw_config.get("env") or {})
        if not server["command"]:
            raise ValueError("stdio mcp config requires command.")

    base_path = resolve_project_path(settings.mcp.config_path)
    path = base_path.with_name(f"{base_path.stem}.local{base_path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    current = {"servers": {}}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(current.get("servers"), dict):
        current["servers"] = {}
    current["servers"][name] = server
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"action": "configureMcp", "status": "saved", "server": name, "transport": transport}


def save_scheduled_automation(request: AutomationRequest, settings: ToolSettings) -> dict[str, Any]:
    root = resolve_project_path(settings.automation.root)
    root.mkdir(parents=True, exist_ok=True)
    title = request.title.strip() or "automation"
    path = resolve_save_path(request, root, title)
    current = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(current, dict):
            current = {}
    payload = {
        **current,
        "title": title,
        "action": request.action,
        "enabled": True,
        "prompt": request.prompt,
        "code": request.code,
        "mcp_server": request.mcp_server,
        "mcp_tool": request.mcp_tool,
        "mcp_arguments": request.mcp_arguments,
        "mcp_config": request.mcp_config,
        "schedule": request.schedule,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    payload.setdefault("created_at", payload["updated_at"])
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "action": request.action,
        "status": "saved",
        "mode": "created" if not current else "updated",
        "path": relative_to_project(path),
        "schedule": request.schedule,
    }


def resolve_save_path(request: AutomationRequest, root: Path, title: str) -> Path:
    if request.target_automation:
        return resolve_automation_name(request.target_automation, root)
    current_path = CURRENT_AUTOMATION_PATH.get()
    if current_path and not request.create_new:
        resolved_current = current_path.resolve()
        if is_relative_to(resolved_current, root.resolve()):
            return resolved_current
    clean_title = clean_name(title).replace(" ", "-")
    return root / f"{int(time.time())}-{clean_title}.json"


def resolve_automation_name(name: str, root: Path) -> Path:
    clean = Path(name.replace("\\", "/")).name.strip()
    if not clean or clean in {".", ".."} or ".." in clean:
        raise ValueError("targetAutomationId must be a safe file name.")
    if Path(clean).suffix and Path(clean).suffix.lower() != ".json":
        raise ValueError("targetAutomationId must be a json file.")
    path = (root / (clean if clean.endswith(".json") else f"{clean}.json")).resolve()
    if not is_relative_to(path, root.resolve()):
        raise ValueError("targetAutomationId must stay inside automation root.")
    return path


def clean_name(name: str) -> str:
    clean = name.strip().replace("\\", "/").strip("/")
    if not clean or "/" in clean or ".." in clean:
        raise ValueError("name must be a single safe name.")
    return clean


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def relative_to_project(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved)


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
