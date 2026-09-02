"""Prompt text for telling the model which tools are available."""

from __future__ import annotations

from tools.settings import ToolSettings, make_tool_settings


TOOL_REQUEST_FORMAT = """
Tool request rules:
- Tools are provided through the API tool_calls field.
- Request only tools listed in available.
- If you can answer from the conversation or injected results, do not call a tool.

Tool argument schema:
{"query":"short search query"}
""".strip()


def build_tools_prompt(
    *,
    web_search: bool = False,
    web_search_mode: str | None = None,
    rag_mode: str = "off",
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
