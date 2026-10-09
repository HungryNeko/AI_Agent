"""OpenAI-compatible tool schemas and tool-call parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from tools import createTool, parameterSave
from tools.appSettings import SettingsRequest
from tools.automation import AutomationRequest
from tools.createTool import CreateToolRequest
from tools.fileEditor import FileEditRequest
from tools.fileReader import FileReadRequest
from tools.history import HistoryRequest
from tools.mcp import McpRequest
from tools.models import ModelRequest
from tools.plan import PlanRequest
from tools.question import QuestionRequest
from tools.rag import RagRequest
from tools.settings import ToolSettings

ToolName = str
SYSTEM_TOOL_NAMES = {
    "webSearch",
    "rag",
    "curl",
    "python",
    "fileReader",
    "fileEditor",
    "mcp",
    "history",
    "automation",
    "settings",
    "question",
    "model",
    "plan",
    "createTool",
    "parameterSave",
}


@dataclass(frozen=True)
class ToolRequest:
    id: str
    name: ToolName
    query: str = ""
    url: str = ""
    code: str = ""
    rag_request: RagRequest | None = None
    file_read: FileReadRequest | None = None
    file_edit: FileEditRequest | None = None
    mcp_request: McpRequest | None = None
    history_request: HistoryRequest | None = None
    automation_request: AutomationRequest | None = None
    settings_request: SettingsRequest | None = None
    model_request: ModelRequest | None = None
    plan_request: PlanRequest | None = None
    create_tool_request: CreateToolRequest | None = None
    custom_arguments: dict[str, Any] | None = None
    question_request: QuestionRequest | None = None
    parameter_request: parameterSave.ParameterRequest | None = None
    wrapped_request: ToolRequest | None = None


def build_openai_tools(settings: ToolSettings) -> list[dict[str, Any]]:
    """Build the `tools` payload for Chat Completions."""

    tools = []
    if settings.web_search.can_model_call and settings.allows("webSearch"):
        tools.append(
            query_tool(
                name="webSearch",
                description="Search the public web when current or external information is needed. If webSearchResult includes image URLs, include useful ones in the final answer as Markdown images using the exact URL.",
            )
        )
    if settings.rag.can_model_call and settings.allows("rag"):
        tools.append(rag_tool(read_only=settings.conversation_mode != "agent"))
    if settings.curl.can_model_call and settings.allows("curl"):
        tools.append(curl_tool())
    if settings.python.can_model_call and settings.allows("python"):
        tools.append(python_tool())
    if settings.file_reader.can_model_call and settings.allows("fileReader"):
        tools.append(file_reader_tool())
    if settings.file_editor.can_model_call and settings.allows("fileEditor"):
        tools.append(file_editor_tool(read_only=settings.conversation_mode != "agent"))
    if settings.mcp.can_model_call and settings.allows("mcp"):
        tools.append(mcp_tool())
    if settings.history.can_model_call and settings.allows("history"):
        tools.append(history_tool())
    if settings.automation.can_model_call and settings.allows("automation"):
        tools.append(automation_tool())
        tools.append(settings_tool())
    tools.append(model_tool())
    tools.append(parameter_tool(read_only=settings.conversation_mode != "agent"))
    if settings.allows("plan"):
        tools.append(plan_tool())
    if settings.allows("createTool"):
        tools.append(create_tool())
        tools.extend(createTool.openai_schemas())
    if settings.question.can_model_call:
        tools.append(question_tool())
    return tools


def parse_openai_tool_calls(
    message: dict[str, Any],
    settings: ToolSettings,
) -> list[ToolRequest]:
    """Parse assistant `tool_calls` from an OpenAI-compatible response."""

    raw_tool_calls = message.get("tool_calls") or []
    if not isinstance(raw_tool_calls, list):
        raise ValueError("assistant tool_calls must be a list.")

    requests = [parse_one_tool_call(raw_tool_call, settings) for raw_tool_call in raw_tool_calls]
    if len(requests) > 1 and any(request.name == "question" for request in requests):
        raise ValueError("question must be the only tool call in an assistant step.")
    return requests


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
    if name not in SYSTEM_TOOL_NAMES and not (isinstance(name, str) and name.startswith("custom__")):
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
    if name == "parameterSave":
        item = parse_parameter_request(parsed_arguments, settings)
        wrapped = None
        if item.action == "call":
            target = item.call["tool"]
            if target not in parameterSave.DATA_TOOLS and not target.startswith("custom__"):
                raise ValueError("parameterSave can only wrap data/query tools or Python/custom tools")
            wrapped = parse_one_tool_call({"id": call_id, "function": {
                "name": target, "arguments": json.dumps(item.call["arguments"]),
            }}, settings)
            if (wrapped.file_edit and wrapped.file_edit.action not in {"list", "read"}) or (
                wrapped.rag_request and wrapped.rag_request.action != "search"
            ):
                raise ValueError("parameterSave cannot wrap file writes or RAG ingestion")
        return ToolRequest(id=call_id, name=name, parameter_request=item, wrapped_request=wrapped)
    if name == "rag":
        rag_request = parse_rag_request(parsed_arguments)
        if settings.conversation_mode != "agent" and rag_request.action != "search":
            raise ValueError(f"rag action {rag_request.action} is not available in {settings.conversation_mode} mode.")
        return ToolRequest(id=call_id, name="rag", rag_request=rag_request)
    if name.startswith("custom__"):
        return ToolRequest(id=call_id, name=name, custom_arguments=parsed_arguments)
    if name == "curl":
        url = require_string(parsed_arguments, "url", "curl")
        return ToolRequest(id=call_id, name="curl", url=url)
    if name == "python":
        code = require_string(parsed_arguments, "code", "python")
        return ToolRequest(id=call_id, name="python", code=code)
    if name == "fileReader":
        return ToolRequest(id=call_id, name="fileReader", file_read=parse_file_read(parsed_arguments))
    if name == "fileEditor":
        file_edit = parse_file_edit(parsed_arguments)
        if settings.conversation_mode != "agent" and file_edit.action not in {"list", "read"}:
            raise ValueError(f"fileEditor action {file_edit.action} is not available in {settings.conversation_mode} mode.")
        return ToolRequest(id=call_id, name="fileEditor", file_edit=file_edit)
    if name == "mcp":
        return ToolRequest(id=call_id, name="mcp", mcp_request=parse_mcp_request(parsed_arguments))
    if name == "history":
        return ToolRequest(id=call_id, name="history", history_request=parse_history_request(parsed_arguments))
    if name == "automation":
        return ToolRequest(id=call_id, name="automation", automation_request=parse_automation_request(parsed_arguments))
    if name == "settings":
        return ToolRequest(id=call_id, name="settings", settings_request=parse_settings_request(parsed_arguments))
    if name == "model":
        return ToolRequest(id=call_id, name="model", model_request=parse_model_request(parsed_arguments))
    if name == "plan":
        return ToolRequest(id=call_id, name="plan", plan_request=parse_plan_request(parsed_arguments))
    if name == "createTool":
        return ToolRequest(id=call_id, name="createTool", create_tool_request=parse_create_tool_request(parsed_arguments))
    if name == "question":
        return ToolRequest(id=call_id, name="question", question_request=parse_question_request(parsed_arguments))

    query = require_string(parsed_arguments, "query", "tool")
    return ToolRequest(id=call_id, name=name, query=query)


def parse_rag_request(arguments: dict[str, Any]) -> RagRequest:
    action = optional_string(arguments.get("action")) or "search"
    if action not in {"search", "ingest"}:
        raise ValueError("rag action must be one of: search, ingest.")
    query = optional_string(arguments.get("query")).strip()
    path = optional_string(arguments.get("path")).strip()
    if action == "search" and not query:
        raise ValueError("rag search requires a non-empty query.")
    if action == "ingest" and not path:
        raise ValueError("rag ingest requires an uploaded file path.")
    split_mode = optional_string(arguments.get("splitMode")) or "simple"
    if split_mode not in {"simple", "llm"}:
        raise ValueError("rag splitMode must be one of: simple, llm.")
    return RagRequest(
        action=action,
        query=query,
        path=path,
        name=optional_string(arguments.get("name")).strip(),
        split_mode=split_mode,
        chunk_model=optional_string(arguments.get("chunkModel")).strip(),
        overwrite=optional_bool(arguments.get("overwrite")),
    )


def parse_parameter_request(arguments: dict[str, Any], settings: ToolSettings) -> parameterSave.ParameterRequest:
    action = require_string(arguments, "action", "parameterSave")
    if action not in {"call", "list", "inspect", "delete"}:
        raise ValueError("parameterSave action must be call, list, inspect, or delete")
    allowed = {
        "call": {"action", "call", "save"}, "list": {"action"},
        "inspect": {"action", "ref", "path", "offset", "limit"}, "delete": {"action", "ref"},
    }[action]
    unknown = set(arguments) - allowed
    if unknown:
        raise ValueError(f"parameterSave {action} unsupported fields: {', '.join(sorted(unknown))}; allowed: {', '.join(sorted(allowed))}")
    if action == "delete" and settings.conversation_mode != "agent":
        raise ValueError("parameterSave delete is only available in Agent mode")
    call = arguments.get("call", {})
    if not isinstance(call, dict):
        raise ValueError("parameterSave call must be an object")
    if action == "call":
        unknown = set(call) - {"tool", "arguments"}
        if unknown:
            raise ValueError(f"parameterSave call unsupported fields: {', '.join(sorted(unknown))}; allowed: tool, arguments")
        require_string(call, "tool", "parameterSave call")
        if not isinstance(call.get("arguments"), dict):
            raise ValueError("parameterSave call.arguments must be an object")
    saves = arguments.get("save", [])
    if not isinstance(saves, list) or len(saves) > 30:
        raise ValueError("parameterSave save must be an array of at most 30 entries")
    normalized = []
    for item in saves:
        if not isinstance(item, dict):
            raise ValueError("parameterSave save entry must be an object")
        unknown = set(item) - {"name", "path", "type"}
        if unknown:
            raise ValueError(f"parameterSave save unsupported fields: {', '.join(sorted(unknown))}; allowed: name, path, type")
        name = require_string(item, "name", "parameterSave save")
        data_type = item.get("type", "auto")
        path = item.get("path", "")
        if not isinstance(data_type, str) or data_type not in {"auto", "json", "ndarray", "table", "dataframe"}:
            raise ValueError("parameterSave save.type must be auto, json, ndarray, table, or dataframe; it converts the selected value after path extraction")
        if not isinstance(path, str):
            raise ValueError("parameterSave save.path must be a dot-path string relative to the decoded tool result (a variable name for Python)")
        if action == "call" and call["tool"] == "python" and not path:
            raise ValueError("Python save entries require a variable path, e.g. values")
        normalized.append({"name": name, "path": path, "type": data_type})
    offset, limit = arguments.get("offset", 0), arguments.get("limit", 0)
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 0 <= limit <= 20:
        raise ValueError("inspect offset must be nonnegative; limit must be 0 to 20")
    ref = require_string(arguments, "ref", "parameterSave") if action in {"inspect", "delete"} else ""
    path = arguments.get("path", "")
    if not isinstance(path, str):
        raise ValueError("parameterSave path must be a string")
    return parameterSave.ParameterRequest(action, call, tuple(normalized), ref, path, offset, limit)


def parse_file_read(arguments: dict[str, Any]) -> FileReadRequest:
    return FileReadRequest(
        path=require_string(arguments, "path", "fileReader"),
        start_page=optional_int(arguments.get("startPage")),
        end_page=optional_int(arguments.get("endPage")),
        sheet=optional_string(arguments.get("sheet")),
        max_chars=optional_int(arguments.get("maxChars")),
    )


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


def parse_question_request(arguments: dict[str, Any]) -> QuestionRequest:
    question = require_string(arguments, "question", "question")
    raw_options = arguments.get("options", [])
    if raw_options is None:
        raw_options = []
    if not isinstance(raw_options, list) or any(not isinstance(item, str) for item in raw_options):
        raise ValueError("question options must be an array of strings.")
    options = tuple(dict.fromkeys(item.strip() for item in raw_options if item.strip()))
    return QuestionRequest(
        question=question,
        options=options,
        multiple=optional_bool(arguments.get("multiple")),
        title=optional_string(arguments.get("title")).strip(),
        placeholder=optional_string(arguments.get("placeholder")).strip(),
    )


def parse_model_request(arguments: dict[str, Any]) -> ModelRequest:
    action = require_string(arguments, "action", "model")
    if action not in {"list", "refresh", "switch"}:
        raise ValueError("model action must be one of: list, refresh, switch.")
    return ModelRequest(
        action=action,
        provider=optional_string(arguments.get("provider")).strip(),
        model=optional_string(arguments.get("model")).strip(),
    )


def parse_plan_request(arguments: dict[str, Any]) -> PlanRequest:
    action = require_string(arguments, "action", "plan")
    if action not in {"update", "finalize"}:
        raise ValueError("plan action must be one of: update, finalize.")
    return PlanRequest(
        action=action,
        name=optional_string(arguments.get("name")).strip(),
        content=optional_string(arguments.get("content")),
    )


def parse_create_tool_request(arguments: dict[str, Any]) -> CreateToolRequest:
    action = require_string(arguments, "action", "createTool")
    if action not in {"list", "read", "save", "delete"}:
        raise ValueError("createTool action must be one of: list, read, save, delete.")
    parameters = arguments.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("createTool parameters must be an object")
    return CreateToolRequest(
        action=action,
        name=optional_string(arguments.get("name")).strip(),
        description=optional_string(arguments.get("description")),
        parameters=parameters,
        code=optional_string(arguments.get("code")),
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
    if name == "question" and not settings.question.can_model_call:
        raise ValueError("question is disabled for this conversation.")
    base_name = "custom__*" if name.startswith("custom__") else name
    if not settings.allows(base_name):
        raise ValueError(f"{name} is not available in {settings.conversation_mode} mode.")
    if name == "webSearch" and not settings.web_search.can_model_call:
        raise ValueError("webSearch can only be called when web_search_mode is auto.")
    if name == "rag" and not settings.rag.can_model_call:
        raise ValueError("rag can only be called when rag_mode is on or auto.")
    if name == "curl" and not settings.curl.can_model_call:
        raise ValueError("curl can only be called when curl_mode is auto.")
    if name == "python" and not settings.python.can_model_call:
        raise ValueError("python can only be called when python_mode is auto.")
    if name == "fileReader" and not settings.file_reader.can_model_call:
        raise ValueError("fileReader can only be called when file_reader_mode is auto.")
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


def rag_tool(*, read_only: bool = False) -> dict[str, Any]:
    return function_tool(
        name="rag",
        description="Search local knowledge, memory, and skills. This mode is read-only." if read_only else (
            "Search local RAG data, or ingest an uploaded document into user knowledge as Markdown. "
            "For ingest, first inspect the upload with fileReader, then pass its path here; never copy "
            "the full extracted content into tool arguments. simple splitting is free and deterministic; "
            "llm splitting uses a dedicated model with strict token limits and incremental caching."
        ),
        properties={
            "action": {"type": "string", "enum": ["search"] if read_only else ["search", "ingest"]},
            "query": {"type": "string", "description": "Required for search."},
            "path": {"type": "string", "description": "Uploaded file path required for ingest."},
            "name": {"type": "string", "description": "Optional destination Markdown filename."},
            "splitMode": {"type": "string", "enum": ["simple", "llm"]},
            "chunkModel": {"type": "string", "description": "Dedicated configured model for LLM splitting."},
            "overwrite": {"type": "boolean"},
        },
        required=["action"],
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


def file_reader_tool() -> dict[str, Any]:
    return function_tool(
        name="fileReader",
        description=(
            "Read and extract text from a local project file or uploaded file without modifying it. "
            "Supports PDF, DOCX, PPTX, XLSX, HTML, CSV, Markdown, source code, and other UTF-8 text. "
            "Use startPage/endPage for PDF pages or PowerPoint slides, and sheet for one Excel worksheet. "
            "Scanned image-only PDFs need OCR and may return no text."
        ),
        properties={
            "path": {
                "type": "string",
                "description": "Project-relative path, backend/runtime/uploads path, or local /api/uploads URL.",
            },
            "startPage": {"type": "integer", "description": "Optional 1-based first PDF page or slide."},
            "endPage": {"type": "integer", "description": "Optional 1-based last PDF page or slide."},
            "sheet": {"type": "string", "description": "Optional XLSX worksheet name."},
            "maxChars": {"type": "integer", "description": "Optional bounded output size."},
        },
        required=["path"],
    )


def file_editor_tool(*, read_only: bool = False) -> dict[str, Any]:
    actions = ["list", "read"] if read_only else ["list", "read", "write", "replace", "insertAfter", "insertBefore", "append"]
    description = (
        "Read project files and list directories for understanding or planning. This mode is read-only."
        if read_only
        else "Edit project files using stable text anchors. Memory lives under data/memory and skills live under data/skills/<name>/SKILL.md. Write-like actions may return approvalRequired instead of applying, depending on backend approval policy. Prefer replace with exact oldText, insertBefore/insertAfter with exact anchor, write for new files, and append for simple additions. No delete, move, rename, shell, or protected-file operations are available."
    )
    return function_tool(
        name="fileEditor",
        description=description,
        properties={
            "action": {
                "type": "string",
                "enum": actions,
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


def question_tool() -> dict[str, Any]:
    return function_tool(
        name="question",
        description=(
            "Ask the user one necessary clarification or confirmation and pause until they reply. "
            "Use an empty options array for a free-text question. Choice questions still let the "
            "user add text, change direction, or refuse. Call this tool alone, without other tools."
        ),
        properties={
            "question": {"type": "string", "description": "The clear, specific question shown to the user."},
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional choices. Leave empty for free text.",
            },
            "multiple": {
                "type": "boolean",
                "description": "Whether the user may select more than one option.",
            },
            "title": {"type": "string", "description": "Optional short dialog title."},
            "placeholder": {"type": "string", "description": "Optional hint for the free-text field."},
        },
        required=["question", "options", "multiple"],
    )


def model_tool() -> dict[str, Any]:
    return function_tool(
        name="model",
        description=(
            "List configured models, refresh a provider's current model list from its standard "
            "OpenAI-compatible /models endpoint, or switch this conversation to another configured model. "
            "Provider API keys are private backend data and are never available through this tool."
        ),
        properties={
            "action": {"type": "string", "enum": ["list", "refresh", "switch"]},
            "provider": {"type": "string", "description": "Provider name for refresh or an unqualified model."},
            "model": {"type": "string", "description": "Configured provider:model value for switch."},
        },
        required=["action"],
    )


def parameter_tool(*, read_only: bool = False) -> dict[str, Any]:
    return function_tool(
        name="parameterSave",
        description=(
            "Wrap an enabled data tool and save selected full results before truncation. "
            "Returns only immutable conversation-scoped refs and structural metadata, not data. "
            "When the structure is unknown, save=[{name:...}] stores the whole result; then inspect it. "
            "Use list or inspect (limit=0 metadata, 1..20 preview); inspect.path describes the selected child's "
            "inferred type/columns without conversion. A table label may be a list of dicts; pythonType is the actual type. "
            "Delete removes a ref. Pass {$ref:ref, path:optional.dot.path} in MCP/custom arguments "
            "or use load_parameter(ref, path=optional.dot.path) in Python. "
            "For MCP callTool, the path root is the decoded application result: structuredContent's contents, "
            "otherwise parsed single JSON text, otherwise the MCP result. Do not prefix structuredContent. "
            "Dot paths support keys/numeric list indexes, not full JSONPath. "
            "For Python, save paths are variables in the executed code. Original tool permissions and review still apply."
        ),
        properties={
            "action": {"type": "string", "enum": ["call", "list", "inspect"] if read_only else ["call", "list", "inspect", "delete"]},
            "call": {"type": "object", "properties": {
                "tool": {"type": "string"}, "arguments": {"type": "object"},
            }, "required": ["tool", "arguments"], "additionalProperties": False},
            "save": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string", "description": "Label for a new immutable ref/version; the only required save field."},
                "path": {"type": "string", "description": "Extract BEFORE saving, relative to the decoded tool result. Omitted/empty/$ saves the whole result. Example: data.rows, not structuredContent.data.rows. For Python a variable path is required, e.g. rows."},
                "type": {"type": "string", "enum": ["auto", "json", "ndarray", "table", "dataframe"], "description": "Conversion AFTER path extraction: omitted/auto preserves Python type; json converts to JSON-compatible structures; ndarray uses NumPy; table/dataframe uses Pandas DataFrame. Does not affect path lookup."},
            }, "required": ["name"], "additionalProperties": False}},
            "ref": {"type": "string"}, "path": {"type": "string", "description": "For inspect only: child path relative to the stored value. Displays that child's inferred structure, without changing the ref or stored type; empty/$ selects the whole value."},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 0, "maximum": 20},
        },
        required=["action"],
    )


def plan_tool() -> dict[str, Any]:
    return function_tool(
        name="plan",
        description="Update the current session plan or mark it ready for user review. The plan remains in conversation state and does not change project files. Use finalize only after research and important clarifications are complete.",
        properties={
            "action": {"type": "string", "enum": ["update", "finalize"]},
            "name": {"type": "string", "description": "Short plan name."},
            "content": {"type": "string", "description": "Complete Markdown plan with scope, implementation steps, relevant files, risks, and verification."},
        },
        required=["action", "content"],
    )


def create_tool() -> dict[str, Any]:
    return function_tool(
        name="createTool",
        description="Create, update, or delete a reusable user-level Python tool. The code must define run(arguments) and is stored separately from system tools. Use custom__name after saving it.",
        properties={
            "action": {"type": "string", "enum": ["list", "read", "save", "delete"]},
            "name": {"type": "string"},
            "description": {"type": "string"},
            "parameters": {"type": "object", "description": "OpenAI function JSON Schema with type=object."},
            "code": {"type": "string", "description": "Python source defining run(arguments)."},
        },
        required=["action"],
    )


def function_tool(
    *,
    name: str,
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
