"""OpenAI-compatible tool schemas and tool-call parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from tools.automation import AutomationRequest
from tools.appSettings import SettingsRequest
from tools.fileEditor import FileEditRequest
from tools.history import HistoryRequest
from tools.mcp import McpRequest
from tools.settings import ToolSettings

ToolName = Literal["webSearch", "rag", "curl", "python", "fileEditor", "mcp", "history", "automation", "settings"]


@dataclass(frozen=True)
class ToolRequest:
    id: str
    name: ToolName
    query: str = ""
    url: str = ""
    code: str = ""
    file_edit: FileEditRequest | None = None
    mcp_request: McpRequest | None = None
    history_request: HistoryRequest | None = None
    automation_request: AutomationRequest | None = None
    settings_request: SettingsRequest | None = None


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
    if settings.history.can_model_call:
        tools.append(history_tool())
    if settings.automation.can_model_call:
        tools.append(automation_tool())
        tools.append(settings_tool())
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
    if name not in {"webSearch", "rag", "curl", "python", "fileEditor", "mcp", "history", "automation", "settings"}:
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
    if name == "history":
        return ToolRequest(id=call_id, name="history", history_request=parse_history_request(parsed_arguments))
    if name == "automation":
        return ToolRequest(id=call_id, name="automation", automation_request=parse_automation_request(parsed_arguments))
    if name == "settings":
        return ToolRequest(id=call_id, name="settings", settings_request=parse_settings_request(parsed_arguments))

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


def parse_history_request(arguments: dict[str, Any]) -> HistoryRequest:
    action = require_string(arguments, "action", "history")
    if action not in {"list", "read", "search"}:
        raise ValueError("history action must be one of: list, read, search.")
    limit = optional_int(arguments.get("limit")) or 10
    return HistoryRequest(
        action=action,
        conversation_id=optional_string(arguments.get("conversationId")),
        query=optional_string(arguments.get("query")),
        limit=max(1, min(50, limit)),
    )


def parse_automation_request(arguments: dict[str, Any]) -> AutomationRequest:
    action = require_string(arguments, "action", "automation")
    if action not in {"script", "mcp", "configureMcp", "reminder", "llm"}:
        raise ValueError("automation action must be one of: script, mcp, configureMcp, reminder, llm.")
    raw_mcp_arguments = arguments.get("mcpArguments", {})
    if raw_mcp_arguments is None:
        raw_mcp_arguments = {}
    raw_mcp_config = arguments.get("mcpConfig", {})
    if raw_mcp_config is None:
        raw_mcp_config = {}
    raw_schedule = arguments.get("schedule", {})
    if raw_schedule is None:
        raw_schedule = {}
    if not isinstance(raw_mcp_arguments, dict) or not isinstance(raw_mcp_config, dict) or not isinstance(raw_schedule, dict):
        raise ValueError("automation mcpArguments, mcpConfig, and schedule must be objects.")
    return AutomationRequest(
        action=action,
        title=optional_string(arguments.get("title")),
        prompt=optional_string(arguments.get("prompt")),
        code=optional_string(arguments.get("code")),
        mcp_server=optional_string(arguments.get("mcpServer")),
        mcp_tool=optional_string(arguments.get("mcpTool")),
        mcp_arguments=raw_mcp_arguments,
        mcp_config=raw_mcp_config,
        schedule=raw_schedule,
        target_automation=optional_string(arguments.get("targetAutomationId")),
        create_new=optional_bool(arguments.get("createNew")),
    )


def parse_settings_request(arguments: dict[str, Any]) -> SettingsRequest:
    action = require_string(arguments, "action", "settings")
    if action not in {"read", "update", "replace"}:
        raise ValueError("settings action must be one of: read, update, replace.")
    raw_patch = arguments.get("patch", {})
    raw_settings = arguments.get("settings", {})
    if raw_patch is None:
        raw_patch = {}
    if raw_settings is None:
        raw_settings = {}
    if not isinstance(raw_patch, dict) or not isinstance(raw_settings, dict):
        raise ValueError("settings patch and settings must be objects.")
    return SettingsRequest(action=action, patch=raw_patch, settings=raw_settings)


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
    if name == "history" and not settings.history.can_model_call:
        raise ValueError("history can only be called when history_mode is auto.")
    if name == "automation" and not settings.automation.can_model_call:
        raise ValueError("automation can only be called when automation_mode is auto.")
    if name == "settings" and not settings.automation.can_model_call:
        raise ValueError("settings can only be called when automation_mode is auto.")


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
        description="Run Python for math, statistics, data analysis, plotting, and local scripting. The current working directory is the artifact directory; save charts/files with relative names like chart.png. For lightweight coordinate maps, `from ai_agent_maps import write_osm_scatter` is available to create OpenStreetMap/Leaflet HTML scatter-map artifacts from lat/lon points. If pythonResult lists image files, include them in the final answer as Markdown images using the exact returned path, for example ![chart](backend/runtime/python_runs/run_x/chart.png). Local file reads, network access, imports, and normal Python introspection are available; obvious destructive operations and writes outside the artifact directory are blocked.",
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
        description="Use configured MCP servers only. Start with listServers or listTools unless the exact server and tool are already known. Never provide URLs, headers, commands, or connection details; pass only server, tool, and arguments. For MCP file-byte inputs, do not inline large base64; use content_base64_from_file, body_base64_from_file, image_base64_from_file, or file_base64_from_file with an uploaded file path or /api/uploads URL so the backend injects exact bytes. For batches, pass arrays of file objects with these *_from_file fields when supported. If mcpResult lists image files or markdownImages, include useful images in the final answer with Markdown image syntax using the exact returned path.",
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


def history_tool() -> dict[str, Any]:
    return function_tool(
        name="history",
        description="List, search, or read saved conversation JSON history when exact previous conversation details are needed, especially after the active context was compressed.",
        properties={
            "action": {
                "type": "string",
                "enum": ["list", "read", "search"],
            },
            "conversationId": {"type": "string", "description": "Conversation id for read."},
            "query": {"type": "string", "description": "Text to search in saved conversations."},
            "limit": {"type": "integer", "description": "Maximum conversations to return for list/search."},
        },
        required=["action"],
    )


def automation_tool() -> dict[str, Any]:
    return function_tool(
        name="automation",
        description="Create, update, or run small automations. During an automation run, reminder/llm saves update the current automation by default; only create a separate automation when the user explicitly asks, then set createNew=true. Use script for simple Python work, mcp to call an already configured MCP tool, configureMcp to save a new MCP server from conversation details, reminder to save fixed reminders, and llm to schedule future model work. Use llm for schedules that need reasoning at execution time, including Fibonacci or custom intervals, and update the same automation schedule with previousRunAt/currentRunAt/fibIndex/nextRunAt.",
        properties={
            "action": {
                "type": "string",
                "enum": ["script", "mcp", "configureMcp", "reminder", "llm"],
            },
            "title": {"type": "string", "description": "Human-readable automation or reminder title."},
            "prompt": {"type": "string", "description": "Prompt to use for reminders or future model work."},
            "code": {"type": "string", "description": "Python code for action=script."},
            "mcpServer": {"type": "string", "description": "Configured MCP server name for action=mcp."},
            "mcpTool": {"type": "string", "description": "MCP tool name for action=mcp."},
            "mcpArguments": {"type": "object", "description": "Arguments passed to the MCP tool."},
            "mcpConfig": {
                "type": "object",
                "description": "MCP server config for action=configureMcp. Include name, enabled, transport, url/headers or command/args/env.",
            },
            "schedule": {
                "type": "object",
                "description": "Reminder schedule. Supports kind=once/interval/cron/custom, nextRunAt, intervalSeconds, cron, timezone, previousRunAt, currentRunAt.",
            },
            "targetAutomationId": {"type": "string", "description": "Existing automation JSON id to update. Omit during a running automation to update itself."},
            "createNew": {"type": "boolean", "description": "Set true only when the user explicitly wants a separate new automation instead of updating the current one."},
        },
        required=["action"],
    )


def settings_tool() -> dict[str, Any]:
    return function_tool(
        name="settings",
        description="Read or update the persistent app JSON config in data/settings.json. Use this when the user asks to remember UI or chat defaults such as theme, language, model, tool modes, search fallback, RAG toggles, approval mode, automation mode, or max tool rounds.",
        properties={
            "action": {
                "type": "string",
                "enum": ["read", "update", "replace"],
            },
            "patch": {
                "type": "object",
                "description": "Partial settings JSON for action=update, for example {\"ui\":{\"theme\":\"dark\"},\"chat\":{\"max_tool_rounds\":-1}}.",
            },
            "settings": {
                "type": "object",
                "description": "Complete settings JSON for action=replace.",
            },
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
