"""Create and execute user-level custom Python tools."""

from __future__ import annotations

import ast
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Any

from agent.config import PROJECT_ROOT
from tools import python as python_tool
from tools.settings import PythonSettings

CUSTOM_TOOL_ROOT = PROJECT_ROOT / "data" / "custom_tools"


@dataclass(frozen=True)
class CreateToolRequest:
    action: Literal["list", "read", "save", "delete"]
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] | None = None
    code: str = ""


def execute(request: CreateToolRequest) -> str:
    if request.action == "list":
        return result({"tools": list_tools()})
    if request.action == "read":
        return result(read_tool(request.name, include_code=True))
    if request.action == "save":
        saved = save_tool(request.name, request.description, request.parameters or {}, request.code)
        return result({"status": "saved", **saved})
    if request.action == "delete":
        return result(delete_tool(request.name))
    raise ValueError(f"unknown createTool action: {request.action}")


def list_tools() -> list[dict[str, Any]]:
    CUSTOM_TOOL_ROOT.mkdir(parents=True, exist_ok=True)
    items = []
    for manifest in sorted(CUSTOM_TOOL_ROOT.glob("*/tool.json")):
        try:
            item = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(item, dict):
            items.append({key: item.get(key) for key in ("name", "function_name", "description", "parameters")})
    return items


def read_tool(name: str, *, include_code: bool = False) -> dict[str, Any]:
    root = tool_root(name)
    manifest = root / "tool.json"
    code_path = root / "tool.py"
    if not manifest.is_file() or not code_path.is_file():
        raise ValueError(f"custom tool not found: {name}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if include_code:
        data["code"] = code_path.read_text(encoding="utf-8")
    return data


def save_tool(name: str, description: str, parameters: dict[str, Any], code: str) -> dict[str, Any]:
    clean = clean_name(name)
    if not description.strip() or not code.strip():
        raise ValueError("name, description, and code are required")
    validate_parameters(parameters)
    validate_code(code)
    root = tool_root(clean)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": clean,
        "function_name": f"custom__{clean}",
        "description": description.strip(),
        "parameters": parameters,
    }
    (root / "tool.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "tool.py").write_text(code.strip() + "\n", encoding="utf-8")
    return manifest


def delete_tool(name: str) -> dict[str, Any]:
    root = tool_root(name)
    if not root.is_dir():
        raise ValueError(f"custom tool not found: {name}")
    shutil.rmtree(root)
    return {"status": "deleted", "name": clean_name(name)}


def execute_custom(function_name: str, arguments: dict[str, Any], settings: PythonSettings, *, raw: bool = False) -> Any:
    if not function_name.startswith("custom__"):
        raise ValueError("custom tool name must start with custom__")
    item = read_tool(function_name.removeprefix("custom__"), include_code=True)
    from tools import parameterSave

    arguments = parameterSave.resolve_references(arguments)
    wrapped = (
        f"{item['code']}\n\n"
        "import json as _json\n"
        "from ai_agent_parameters import load_bindings, export_result\n"
        "_arguments = load_bindings()\n"
        "_result = run(_arguments)\n"
        + ("export_result(_result)\n" if raw else "print(_json.dumps(_result, ensure_ascii=False, default=str))\n")
    )
    execution = python_tool.run(wrapped, settings, bindings=arguments, capture_result=raw)
    if raw:
        if execution["return_code"] != 0 or "result" not in execution:
            raise ValueError(execution.get("stderr") or "custom tool returned no result")
        return execution
    return "customToolResult:\n" + json.dumps(execution, ensure_ascii=False, default=str)


def openai_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": item["function_name"],
                "description": item["description"],
                "parameters": item["parameters"],
            },
        }
        for item in list_tools()
        if item.get("function_name") and isinstance(item.get("parameters"), dict)
    ]


def clean_name(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip()).strip("_").lower()
    if not clean:
        raise ValueError("custom tool name is required")
    return clean[:48]


def tool_root(name: str) -> Path:
    return CUSTOM_TOOL_ROOT / clean_name(name)


def validate_parameters(parameters: dict[str, Any]) -> None:
    if parameters.get("type") != "object" or not isinstance(parameters.get("properties", {}), dict):
        raise ValueError("parameters must be a JSON Schema object with type=object")


def validate_code(code: str) -> None:
    tree = ast.parse(code)
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run" for node in tree.body):
        raise ValueError("custom tool code must define run(arguments)")


def result(payload: object) -> str:
    return "createToolResult:\n" + json.dumps(payload, ensure_ascii=False)
