"""Prompt text for telling the model which tools are available."""

from __future__ import annotations

from tools.settings import ToolSettings, make_tool_settings

TOOL_REQUEST_FORMAT = """
Tool request rules:
- Tools are provided through the API tool_calls field.
- Request only tools listed in available.
- If you can answer from the conversation or injected results, do not call a tool.
- Use rag action=search to search local knowledge, memory, and skill files. Memory files live under data/memory. Skill entrypoints live at data/skills/<name>/SKILL.md.
- When the user asks to add an uploaded PDF, Word, PowerPoint, Excel, or text document to the knowledge base, first call fileReader to inspect it, then call rag action=ingest with the same uploaded path. Do not paste the extracted document into the rag arguments. Use simple splitting unless the user explicitly requests LLM splitting.
- RAG results include sourceType and path. Use those source paths in the answer when they matter, and request more file detail if an excerpt is not enough.
- Use history to list/search/read saved conversation JSON when exact previous messages or tool output are needed after compression.
- User text may contain @ references such as @tool:python, @file:data/skills/name/SKILL.md, or @history:conversation-id. Treat them as explicit user intent for that tool or context.
- For weather or other date-sensitive searches, include the current date from currentTime. For weather, include the location; ask for it if missing.
- Use curl only for direct public http(s) API GET requests when a web API URL is known.
- If a curl request fails or returns an API error, do not blindly retry the same URL. If webSearch is available, search the official API documentation, then change the endpoint or parameters before trying curl again.
- If webSearchResult includes image URLs, or curlResult/API data contains image URLs or image content, show useful images in the final answer with Markdown image syntax using the exact URL, for example ![image](https://example.com/image.jpg).
- Use python for math, statistics, data analysis, plotting, and local scripting. Its current working directory is the artifact directory; save files with relative names like plt.savefig("chart.png"). For lightweight coordinate maps, use `from ai_agent_maps import write_osm_scatter` to create an OpenStreetMap/Leaflet HTML artifact from lat/lon points. Prefer webSearch or curl for web/API fetching when those tools fit better. If pythonResult lists image files, show them in the final answer with Markdown image syntax using the exact returned path, for example ![chart](backend/runtime/python_runs/run_x/chart.png); for HTML map artifacts, mention the returned path as a clickable reference.
- Use fileReader to extract text from uploaded or project PDF, DOCX, PPTX, XLSX, HTML, CSV, Markdown, and source files. It is read-only. Use page ranges or an Excel sheet name for large files. Scanned image-only PDFs require OCR and may contain no extractable text.
- Use fileEditor to inspect the project in Ask and Plan modes; only list/read actions are available there. In Agent mode it can change project files, including memory and skill files when requested. Prefer list/read before editing, then use exact stable text anchors. If fileEditor returns approvalRequired, explain the pending change and do not claim it was applied.
- When file editor approval is aiReview, high-risk tool calls are reviewed by a separate AI reviewer before execution. If aiReview denies the call, explain the denial and choose a safer next step.
- Use mcp only for configured MCP servers. Start with listServers or listTools unless the exact server and tool are already known. Do not provide shell commands to mcp. Uploaded attachments include path, url, and absoluteUrl; for remote MCP file URL inputs prefer absoluteUrl, and for local MCP tools use path. If an MCP tool requires file bytes such as content_base64/body_base64/image_base64, never inline large base64 in tool arguments; pass content_base64_from_file/body_base64_from_file/image_base64_from_file with the uploaded path or upload URL, and the backend will inject the exact bytes. For batch uploads, pass an array of file objects using these *_from_file fields when the MCP schema supports it. If mcpResult lists image files or markdownImages, show useful ones in the final answer with Markdown image syntax using the exact returned path.
- Use automation for user-approved lightweight workflows: simple script execution, calling an MCP tool, saving MCP server config from conversation details, reminders, or scheduled future model work. Use action=llm when the task needs model reasoning at execution time, such as Fibonacci/custom intervals.
- During an automation run, schedule changes must update the current automation by default. Only create a separate automation when the user explicitly asks for a new/separate task, and then set createNew=true. For custom recurring schedules, store previousRunAt/currentRunAt/fibIndex/nextRunAt in the same schedule so the next run can be computed.
- Use settings to read or update persistent app JSON config in data/settings.json when the user asks to remember UI or chat defaults.
- Use model to list configured models, refresh a provider from its OpenAI-compatible /models endpoint, or switch only the current conversation model. API credentials are backend-only and must never be requested or exposed.
- In Plan mode, first inspect enough project context to remove avoidable assumptions. Ask only questions that materially change the implementation. Use plan action=update for drafts and action=finalize when the Markdown plan is ready for user review. Include scope, ordered implementation steps, relevant files, risks, and verification. The plan is session state, not a project file. Do not execute or modify project files until the user approves and hands the plan to Agent mode.
- In Ask mode, answer or investigate without changing files or running commands. You may recommend switching to Plan for uncertain cross-cutting work or Agent for a clear implementation request, but do not switch modes yourself.
- In Agent mode, implement directly when the task is clear. If an approved session plan is present, follow it while adapting to verified repository facts; test the result and report meaningful deviations.
- Use createTool in agent mode to save reusable user-level Python tools. Saved tools appear as custom__name functions; MCP tools remain behind mcp and system tools keep their normal names. Use action=delete to remove a custom tool when it is no longer needed.
- For large query results, use parameterSave action=call with call={tool,arguments}. If structure is unknown, save=[{name}] stores the entire decoded result; inspect its keys first. Optional save.path extracts BEFORE saving, relative to that decoded result (MCP structuredContent's contents or parsed JSON text, without the transport prefix); type defaults to auto and only converts AFTER extraction. Inspect.path describes a stored child without conversion: type=table may mean a list of dicts; pythonType is the actual Python type. Only refs and structure return to you. Reuse {$ref:"param:...",path:"optional.path"} in MCP/custom arguments, or load_parameter("param:...", path="optional.path") in Python. Python can save_parameter("name", value), or the wrapper can save a variable via its required path. Values persist with this conversation through restart/compression; use list to recover refs, inspect for metadata (limit=0) or a small preview (limit=1..20), and delete only when requested. Do not print/repeat full datasets. Wrapping does not override tool modes or approvals.
- Use question only when a necessary choice, missing detail, or user confirmation blocks useful progress. Ask one clear question at a time and call question alone. An empty options list creates a fill-in question; choices may be single- or multi-select. The user can always add free text, change direction because the question is not applicable, or refuse to answer. Treat refusal as no consent and continue only when it is safe to do so.
- Use approval in Agent mode to ask the user to manually approve one specific risky, irreversible, costly, or out-of-scope action before doing it, unless they already approved it. Call approval alone with a one-sentence action, optional details (commands, files, diff) and a risk level. The turn then pauses. The next user message is approvalResponse JSON: status "approved" means proceed with exactly that action; status "rejected" means do not do it, and any text is the user's instruction or reason, so follow it and propose an alternative if useful. Never repeat a rejected request unchanged.
- If a tool returns toolError, use the raw error to decide whether retrying, changing input, using a different tool, or reporting failure is best. Do not repeat the exact same failing tool input more than once.

Tool argument schemas:
webSearch: {"query":"short search query"}
rag search: {"action":"search","query":"short search query"}
rag ingest: {"action":"ingest","path":"backend/runtime/uploads/upload_id/report.docx","name":"report.md","splitMode":"simple"}
curl: {"url":"https://api.example.com/path?x=1"}
python: {"code":"print(2 + 2)"}
parameterSave: {"action":"call","call":{"tool":"mcp","arguments":{"action":"callTool","server":"configuredServerName","tool":"query","arguments":{}}},"save":[{"name":"query_result"}]}
python saved data: {"code":"rows = load_parameter(\"param:reference-from-tool\")\nprint(len(rows))"}
python OSM map: {"code":"from ai_agent_maps import write_osm_scatter\nprint(write_osm_scatter([{\"lat\":34.0522,\"lon\":-118.2437,\"label\":\"LA\"}], \"map.html\"))"}
fileReader: {"path":"backend/runtime/uploads/upload_id/report.pdf","startPage":1,"endPage":10}
fileEditor: {"action":"read","path":"backend/agent/graph.py"}
mcp: {"action":"listTools","server":"configuredServerName"}
mcp file upload: {"action":"callTool","server":"configuredServerName","tool":"uploadFile","arguments":{"filename":"photo.jpg","content_type":"image/jpeg","content_base64_from_file":"backend/runtime/uploads/upload_id/photo.jpg"}}
history: {"action":"search","query":"older topic","limit":5}
automation: {"action":"reminder","title":"check report","prompt":"check report","schedule":{"kind":"once","nextRunAt":"2026-09-03T20:00:00-07:00"}}
automation self-update: {"action":"llm","title":"fib reminder","prompt":"compute the next Fibonacci delay and update this automation","schedule":{"kind":"custom","fibIndex":4,"previousRunAt":"2026-09-03T20:00:00-07:00","currentRunAt":"2026-09-03T20:03:00-07:00","nextRunAt":"2026-09-03T20:05:00-07:00"}}
settings: {"action":"update","patch":{"ui":{"theme":"dark"},"chat":{"max_tool_rounds":-1}}}
model: {"action":"switch","model":"provider:model-id"}
plan: {"action":"finalize","name":"feature-plan","content":"# Goal\n...\n# Steps\n...\n# Verification\n..."}
createTool: {"action":"save","name":"calculator","description":"Calculate a formula","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]},"code":"def run(arguments):\n    return {\"expression\": arguments[\"expression\"]}"}
createTool delete: {"action":"delete","name":"calculator"}
approval: {"action":"Delete the build/ directory","title":"Confirm deletion","details":"Runs `rm -rf build/`","risk":"high"}
question: {"question":"Which environment should I update?","options":["Development","Production"],"multiple":false,"title":"Choose environment","placeholder":"Add context or describe another direction"}
""".strip()


def build_tools_prompt(
    *,
    web_search: bool = False,
    web_search_mode: str | None = None,
    rag_mode: str = "off",
    curl_mode: str = "off",
    python_mode: str = "off",
    file_reader_mode: str = "off",
    file_editor_mode: str = "off",
    mcp_mode: str = "off",
    history_mode: str = "off",
    automation_mode: str = "off",
    conversation_mode: str = "agent",
    rag_context: str | None = None,
    web_search_results: list[str] | None = None,
    rag_results: list[str] | None = None,
    include_rules: bool = False,
    include_available: bool = True,
    tool_error: str | None = None,
) -> str:
    """Return tool info for the current prompt turn.

    Normal turns only include available tools and injected results.
    Initial, compressed, and tool-error turns can include the rules.
    """

    settings = make_tool_settings(
        web_search=web_search if web_search_mode is None else None,
        web_search_mode=web_search_mode or "off",
        rag_mode=rag_mode,
        curl_mode=curl_mode,
        python_mode=python_mode,
        file_reader_mode=file_reader_mode,
        file_editor_mode=file_editor_mode,
        mcp_mode=mcp_mode,
        history_mode=history_mode,
        automation_mode=automation_mode,
        conversation_mode=conversation_mode,
    )
    lines: list[str] = []

    if include_rules or tool_error:
        lines.append(build_tool_usage_reminder(tool_error))

    if include_available:
        lines.append(format_available(settings))

    if rag_context:
        lines.append(format_result("ragResult", rag_context))

    for result in web_search_results or []:
        lines.append(format_result("webSearchResult", result))
    for result in rag_results or []:
        lines.append(format_result("ragResult", result))

    return "\n".join(lines)


def build_tools_prompt_from_settings(
    settings: ToolSettings,
    *,
    rag_context: str | None = None,
    web_search_results: list[str] | None = None,
    rag_results: list[str] | None = None,
    include_rules: bool = False,
    include_available: bool = True,
    tool_error: str | None = None,
) -> str:
    lines: list[str] = []
    if include_rules or tool_error:
        lines.append(build_tool_usage_reminder(tool_error))
    if include_available:
        lines.append(format_available(settings))
    if rag_context:
        lines.append(format_result("ragResult", rag_context))
    for result in web_search_results or []:
        lines.append(format_result("webSearchResult", result))
    for result in rag_results or []:
        lines.append(format_result("ragResult", result))
    return "\n".join(lines)


def build_tool_usage_reminder(error: str | None = None) -> str:
    """Detailed tool syntax to send only after a bad tool request or explicit question."""

    lines = ["Tool request format reminder:", TOOL_REQUEST_FORMAT]
    if error:
        lines.append(f"Previous tool request error: {error}")
    return "\n".join(lines)


def format_available(settings: ToolSettings) -> str:
    available = settings.model_view()["available"]
    quoted_tools = ", ".join(f'"{tool}"' for tool in available)
    return f"available: [{quoted_tools}]"


def format_result(name: str, text: str) -> str:
    clean_text = text.strip()
    return f'{name}: "{clean_text}"'
