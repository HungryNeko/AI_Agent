"""OpenAI-compatible tool schemas and tool-call parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from tools.fileEditor import FileEditRequest
from tools.mcp import McpRequest
from tools.settings import ToolSettings

ToolName = Literal["webSearch", "rag", "curl", "python", "fileEditor", "mcp"]


@dataclass(frozen=True)
class ToolRequest:
    id: str
    name: ToolName
    query: str = ""
    url: str = ""
    code: str = ""
    file_edit: FileEditRequest | None = None
    mcp_request: McpRequest | None = None


def build_openai_tools(settings: ToolSettings) -> list[dict[str, Any]]:
    """Build the `tools` payload for Chat Completions."""

    tools = []
    if settings.web_search.can_model_call:
        tools.append(
            query_tool(
                name="webSearch",
                description="Search the public web when current or external information is needed. If webSearchResult includes image URLs, include useful ones in the final answer as Markdown images using the exact URL.",
            )
        )
    if settings.rag.can_model_call:
        tools.append(
            query_tool(
                name="rag",
                description="Search local knowledge, memory, and skill files when project context or saved instructions are needed.",
            )
        )
    if settings.curl.can_model_call:
        tools.append(curl_tool())
    if settings.python.can_model_call:
        tools.append(python_tool())
    if settings.file_editor.can_model_call:
        tools.append(file_editor_tool())
    if settings.mcp.can_model_call:
        tools.append(mcp_tool())
    return tools


def parse_openai_tool_calls(
    message: dict[str, Any],
    settings: ToolSettings,
) -> list[ToolRequest]:
    """Parse assistant `tool_calls` from an OpenAI-compatible response."""

    raw_tool_calls = message.get("tool_calls") or []
    if not isinstance(raw_tool_calls, list):
        raise ValueError("assistant tool_calls must be a list.")

    return [parse_one_tool_call(raw_tool_call, settings) for raw_tool_call in raw_tool_calls]


def parse_one_tool_call(raw_tool_call: object, settings: ToolSettings) -> ToolRequest:
    if not isinstance(raw_tool_call, dict):
        raise ValueError("tool_call must be an object.")

    call_id = raw_tool_call.get("id")
    function = raw_tool_call.get("function")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("tool_call.id is required.")
    if not isinstance(function, dict):
        raise ValueError("tool_call.function is required.")

    name = function.get("name")
    if name not in {"webSearch", "rag", "curl", "python", "fileEditor", "mcp"}:
        raise ValueError(f"Unknown tool: {name}")

    arguments = function.get("arguments") or "{}"
    if not isinstance(arguments, str):
        raise ValueError("tool_call.function.arguments must be a JSON string.")

    try:
        parsed_arguments = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError("tool arguments must be valid JSON.") from exc

    if not isinstance(parsed_arguments, dict):
        raise ValueError("tool arguments must be a JSON object.")

    validate_tool_allowed(name, settings)
    if name == "curl":
        url = require_string(parsed_arguments, "url", "curl")
        return ToolRequest(id=call_id, name="curl", url=url)
    if name == "python":
        code = require_string(parsed_arguments, "code", "python")
        return ToolRequest(id=call_id, name="python", code=code)
    if name == "fileEditor":
        return ToolRequest(id=call_id, name="fileEditor", file_edit=parse_file_edit(parsed_arguments))
    if name == "mcp":
        return ToolRequest(id=call_id, name="mcp", mcp_request=parse_mcp_request(parsed_arguments))

    query = require_string(parsed_arguments, "query", "tool")
    return ToolRequest(id=call_id, name=name, query=query)


def parse_file_edit(arguments: dict[str, Any]) -> FileEditRequest:
    action = require_string(arguments, "action", "fileEditor")
    if action not in {"list", "read", "write", "replace", "insertAfter", "insertBefore", "append"}:
        raise ValueError("fileEditor action must be one of: list, read, write, replace, insertAfter, insertBefore, append.")
    return FileEditRequest(
        action=action,
        path=optional_string(arguments.get("path")),
        content=optional_string(arguments.get("content")),
        old_text=optional_string(arguments.get("oldText")),
        new_text=optional_string(arguments.get("newText")),
        anchor=optional_string(arguments.get("anchor")),
        pattern=optional_string(arguments.get("pattern")) or "**/*",
        overwrite=optional_bool(arguments.get("overwrite")),
        replace_all=optional_bool(arguments.get("replaceAll")),
        start_line=optional_int(arguments.get("startLine")),
        end_line=optional_int(arguments.get("endLine")),
        max_results=optional_int(arguments.get("maxResults")) or 80,
    )


def parse_mcp_request(arguments: dict[str, Any]) -> McpRequest:
    action = require_string(arguments, "action", "mcp")
    if action not in {"listServers", "listTools", "callTool"}:
        raise ValueError("mcp action must be one of: listServers, listTools, callTool.")
    raw_arguments = arguments.get("arguments", {})
    if raw_arguments is None:
        raw_arguments = {}
    if not isinstance(raw_arguments, dict):
        raise ValueError("mcp arguments must be an object.")
    return McpRequest(
        action=action,
        server=optional_string(arguments.get("server")),
        tool=optional_string(arguments.get("tool")),
        arguments=raw_arguments,
    )


def require_string(arguments: dict[str, Any], key: str, tool_name: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{tool_name} arguments must include non-empty string field `{key}`.")
    return value.strip()


def optional_string(value: object) -> str:
    return value if isinstance(value, str) else ""


def optional_bool(value: object) -> bool:
    return value if isinstance(value, bool) else False


def optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def validate_tool_allowed(name: str, settings: ToolSettings) -> None:
    if name == "webSearch" and not settings.web_search.can_model_call:
        raise ValueError("webSearch can only be called when web_search_mode is auto.")
    if name == "rag" and not settings.rag.can_model_call:
        raise ValueError("rag can only be called when rag_mode is on or auto.")
    if name == "curl" and not settings.curl.can_model_call:
        raise ValueError("curl can only be called when curl_mode is auto.")
    if name == "python" and not settings.python.can_model_call:
        raise ValueError("python can only be called when python_mode is auto.")
    if name == "fileEditor" and not settings.file_editor.can_model_call:
        raise ValueError("fileEditor can only be called when file_editor_mode is auto.")
    if name == "mcp" and not settings.mcp.can_model_call:
        raise ValueError("mcp can only be called when mcp_mode is auto.")


def query_tool(*, name: Literal["webSearch", "rag"], description: str) -> dict[str, Any]:
    return function_tool(
        name=name,
        description=description,
        properties={
            "query": {
                "type": "string",
                "description": "The search query.",
            }
        },
        required=["query"],
    )


def curl_tool() -> dict[str, Any]:
    return function_tool(
        name="curl",
        description="Fetch one public http(s) API URL with GET when direct JSON/text/image data is needed. If the endpoint or parameters are uncertain, search the official API documentation first. If curlResult contains image URLs or image content, include useful images in the final answer as Markdown images using the exact URL.",
        properties={
            "url": {
                "type": "string",
                "description": "Full http(s) URL to fetch, including query parameters.",
            }
        },
        required=["url"],
    )


def python_tool() -> dict[str, Any]:
    return function_tool(
        name="python",
        description="Run Python for math, statistics, data analysis, plotting, and local scripting. The current working directory is the artifact directory; save charts/files with relative names like chart.png. If pythonResult lists image files, include them in the final answer as Markdown images using the exact returned path, for example ![chart](backend/runtime/python_runs/run_x/chart.png). Local file reads, network access, imports, and normal Python introspection are available; obvious destructive operations and writes outside the artifact directory are blocked.",
        properties={
            "code": {
                "type": "string",
                "description": "Python code to run. Print concise results and save artifacts with relative filenames in the current working directory.",
            }
        },
        required=["code"],
    )


def file_editor_tool() -> dict[str, Any]:
    return function_tool(
        name="fileEditor",
        description="Edit project files using stable text anchors. Memory lives under data/memory and skills live under data/skills/<name>/SKILL.md. Write-like actions may return approvalRequired instead of applying, depending on backend approval policy. Prefer replace with exact oldText, insertBefore/insertAfter with exact anchor, write for new files, and append for simple additions. No delete, move, rename, shell, or protected-file operations are available.",
        properties={
            "action": {
                "type": "string",
                "enum": ["list", "read", "write", "replace", "insertAfter", "insertBefore", "append"],
            },
            "path": {"type": "string", "description": "Project-relative file or directory path."},
            "content": {"type": "string", "description": "Content for write/append/insert operations."},
            "oldText": {"type": "string", "description": "Exact text to replace."},
            "newText": {"type": "string", "description": "Replacement text."},
            "anchor": {"type": "string", "description": "Exact unique anchor for insertBefore/insertAfter."},
            "pattern": {"type": "string", "description": "Glob pattern for list, default **/*."},
            "overwrite": {"type": "boolean", "description": "Allow write to replace an existing file."},
            "replaceAll": {"type": "boolean", "description": "Allow replace to update all oldText matches."},
            "startLine": {"type": "integer", "description": "First 1-based line for read."},
            "endLine": {"type": "integer", "description": "Last 1-based line for read."},
            "maxResults": {"type": "integer", "description": "Maximum listed files."},
        },
        required=["action"],
    )


def mcp_tool() -> dict[str, Any]:
    return function_tool(
        name="mcp",
        description="Use configured MCP servers only. Start with listServers or listTools unless the exact server and tool are already known. Never provide URLs, headers, commands, or connection details; pass only server, tool, and arguments. If mcpResult lists image files or markdownImages, include useful images in the final answer with Markdown image syntax using the exact returned path.",
        properties={
            "action": {
                "type": "string",
                "enum": ["listServers", "listTools", "callTool"],
            },
            "server": {"type": "string", "description": "Configured MCP server name."},
            "tool": {"type": "string", "description": "MCP tool name for callTool."},
            "arguments": {"type": "object", "description": "Arguments object passed to the MCP tool."},
        },
        required=["action"],
    )


def function_tool(
    *,
    name: ToolName,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }
