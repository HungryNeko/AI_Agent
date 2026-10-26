"""Prompt text for telling the model which tools are available."""

from __future__ import annotations

from tools.settings import ToolSettings, make_tool_settings

TOOL_REQUEST_FORMAT = """
- Call only tools listed in available. If the conversation or injected results already answer the question, do not call a tool.
- Each tool's description is the reference for how to use that tool; these rules cover choosing and combining tools.
- User text may contain @ references such as @tool:python, @file:data/skills/name/SKILL.md, or @history:conversation-id. Treat them as explicit user intent for that tool or context.
- Local data: rag searches knowledge, memory, and skills; read the file when an excerpt is not enough. Citations from rag are checked by the system: cite what you use with the result's `cite:` tag, copied exactly, and never invent a tag or quote text that is not in the chunk. Use history for exact earlier messages or tool output after compression.
- Web: prefer nativeWebSearch and webFetch when listed, otherwise webSearch; use curl only for a known public API URL. For date-sensitive searches, include the current date from currentTime.
- Files: fileReader extracts text from documents and never changes them; fileEditor lists, reads, and edits project files. To add an uploaded document to the knowledge base, inspect it with fileReader, then call rag ingest with the same path.
- Code: use python for calculations, data analysis, plots, and scripts. Save a repeated multi-step job (for example several MCP calls in a row) as a custom tool with createTool; custom tools are called as custom__name. MCP servers are reached only through the mcp tool.
- Large results: wrap the call in parameterSave so only refs and structure come back, and never print or repeat full datasets.
- Persistent changes: settings only when the user asks to remember UI or chat defaults, automation for user-approved scheduled or repeated work, model to list, refresh, or switch models.
- Asking the user: question for a necessary choice or missing detail, approval (Agent mode) before a risky, irreversible, costly, or out-of-scope action. Call either one alone; the turn pauses until the user replies.
- Delegation: use agent (Agent mode) only for independent, non-trivial chunks of work. A message starting with systemEvent: comes from the system, not from the user.
- Approvals: when aiReview is on, a separate reviewer checks high-risk calls; if it denies one, explain why and choose a safer step. If fileEditor returns approvalRequired, describe the pending change and do not claim it was applied.
- Errors: on toolError, read the raw error and decide whether to retry with different input, use another tool, or report the failure. Never repeat the same failing input more than once.
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
    """The cross-tool rules of the first system message (plus the error when one is passed)."""

    lines = ["Tool rules:", TOOL_REQUEST_FORMAT]
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
