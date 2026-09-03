"""Small automation tool for model-managed workflows."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from agent.config import PROJECT_ROOT
from tools import mcp
from tools import python as python_tool
from tools.mcp import McpRequest
from tools.settings import ToolSettings

AutomationAction = Literal["script", "mcp", "configureMcp", "reminder", "llm"]


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
        return save_reminder(request, settings)
    if request.action == "llm":
        return {
            "action": "llm",
            "nextStep": "Ask the model in the next assistant step using the prompt stored here.",
            "prompt": request.prompt,
        }
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

    path = resolve_project_path(settings.mcp.config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    current = {"servers": {}}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(current.get("servers"), dict):
        current["servers"] = {}
    current["servers"][name] = server
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"action": "configureMcp", "status": "saved", "server": name, "transport": transport}


def save_reminder(request: AutomationRequest, settings: ToolSettings) -> dict[str, Any]:
    root = resolve_project_path(settings.automation.root)
    root.mkdir(parents=True, exist_ok=True)
    title = request.title.strip() or "automation"
    clean_title = clean_name(title).replace(" ", "-")
    path = root / f"{int(time.time())}-{clean_title}.json"
    payload = {
        "title": title,
        "prompt": request.prompt,
        "schedule": request.schedule,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"action": "reminder", "status": "saved", "path": relative_to_project(path), "schedule": request.schedule}


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
    return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
