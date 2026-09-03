"""Prompt text for telling the model which tools are available."""

from __future__ import annotations

from tools.settings import ToolSettings, make_tool_settings

TOOL_REQUEST_FORMAT = """
Tool request rules:
- Tools are provided through the API tool_calls field.
- Request only tools listed in available.
- If you can answer from the conversation or injected results, do not call a tool.
- Use rag to search local knowledge, memory, and skill files. Memory files live under data/memory. Skill entrypoints live at data/skills/<name>/SKILL.md.
- RAG results include sourceType and path. Use those source paths in the answer when they matter, and request more file detail if an excerpt is not enough.
- Use history to list/search/read saved conversation JSON when exact previous messages or tool output are needed after compression.
- User text may contain @ references such as @tool:python, @file:data/skills/name/SKILL.md, or @history:conversation-id. Treat them as explicit user intent for that tool or context.
- For weather or other date-sensitive searches, include the current date from currentTime. For weather, include the location; ask for it if missing.
- Use curl only for direct public http(s) API GET requests when a web API URL is known.
- If a curl request fails or returns an API error, do not blindly retry the same URL. If webSearch is available, search the official API documentation, then change the endpoint or parameters before trying curl again.
- If webSearchResult includes image URLs, or curlResult/API data contains image URLs or image content, show useful images in the final answer with Markdown image syntax using the exact URL, for example ![image](https://example.com/image.jpg).
- Use python for math, statistics, data analysis, plotting, and local scripting. Its current working directory is the artifact directory; save files with relative names like plt.savefig("chart.png"). Prefer webSearch or curl for web/API fetching when those tools fit better. If pythonResult lists image files, show them in the final answer with Markdown image syntax using the exact returned path, for example ![chart](backend/runtime/python_runs/run_x/chart.png).
- Use fileEditor for project file changes, including adding or updating memory and skill files when the user asks. Prefer list/read before editing. Prefer replace with exact unique oldText, or insertAfter/insertBefore with an exact unique anchor. Do not use line numbers for edits unless there is no stable text anchor. If fileEditor returns approvalRequired, explain the pending change and do not claim it was applied.
- Use mcp only for configured MCP servers. Start with listServers or listTools unless the exact server and tool are already known. Do not provide shell commands to mcp. If mcpResult lists image files or markdownImages, show useful ones in the final answer with Markdown image syntax using the exact returned path.
- Use automation for user-approved lightweight workflows: simple script execution, calling an MCP tool, saving MCP server config from conversation details, or saving reminders. For custom recurring schedules, store previousRunAt/currentRunAt/nextRunAt in schedule so the next reminder can be computed.
- Use settings to read or update persistent app JSON config in data/settings.json when the user asks to remember UI or chat defaults.
- If a tool returns toolError, use the raw error to decide whether retrying, changing input, using a different tool, or reporting failure is best. Do not repeat the exact same failing tool input more than once.

Tool argument schemas:
webSearch/rag: {"query":"short search query"}
curl: {"url":"https://api.example.com/path?x=1"}
python: {"code":"print(2 + 2)"}
fileEditor: {"action":"read","path":"backend/agent/graph.py"}
mcp: {"action":"listTools","server":"configuredServerName"}
history: {"action":"search","query":"older topic","limit":5}
automation: {"action":"reminder","title":"check report","prompt":"check report","schedule":{"kind":"once","nextRunAt":"2026-09-03T20:00:00-07:00"}}
settings: {"action":"update","patch":{"ui":{"theme":"dark"},"chat":{"max_tool_rounds":-1}}}
""".strip()


def build_tools_prompt(
    *,
    web_search: bool = False,
    web_search_mode: str | None = None,
    rag_mode: str = "off",
    curl_mode: str = "off",
    python_mode: str = "off",
    file_editor_mode: str = "off",
    mcp_mode: str = "off",
    history_mode: str = "off",
    automation_mode: str = "off",
    rag_context: str | None = None,
    web_search_results: list[str] | None = None,
    rag_results: list[str] | None = None,
    include_rules: bool = False,
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
        file_editor_mode=file_editor_mode,
        mcp_mode=mcp_mode,
        history_mode=history_mode,
        automation_mode=automation_mode,
    )
    lines: list[str] = []

    if include_rules or tool_error:
        lines.append(build_tool_usage_reminder(tool_error))

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
    tool_error: str | None = None,
) -> str:
    return build_tools_prompt(
        web_search_mode=settings.web_search.mode,
        rag_mode=settings.rag.mode,
        curl_mode=settings.curl.mode,
        python_mode=settings.python.mode,
        file_editor_mode=settings.file_editor.mode,
        mcp_mode=settings.mcp.mode,
        history_mode=settings.history.mode,
        automation_mode=settings.automation.mode,
        rag_context=rag_context,
        web_search_results=web_search_results,
        rag_results=rag_results,
        include_rules=include_rules,
        tool_error=tool_error,
    )


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
