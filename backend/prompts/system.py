"""Build the system prompt sent to the model."""

from __future__ import annotations

from prompts.context import build_context_prompt, build_context_rules_prompt
from prompts.tools import build_tool_usage_reminder, build_tools_prompt


BASE_SYSTEM_PROMPT = """
You are an AI agent.

Answer directly when you have enough information.
""".strip()


def build_system_prompt(
    *,
    web_search: bool = False,
    web_search_mode: str | None = None,
    rag_mode: str = "off",
    rag_context: str | None = None,
    web_search_results: list[str] | None = None,
    rag_results: list[str] | None = None,
    conversation_summary: str | None = None,
    include_tool_rules: bool = False,
    include_context_rules: bool = False,
    tool_error: str | None = None,
) -> str:
    context_rules = build_context_rules_prompt(include_rules=include_context_rules)
    tool_rules = (
        build_tool_usage_reminder(tool_error)
        if include_tool_rules or tool_error
        else ""
    )
    context_prompt = build_context_prompt(
        conversation_summary=conversation_summary,
    )
    tools_prompt = build_tools_prompt(
        web_search=web_search,
        web_search_mode=web_search_mode,
        rag_mode=rag_mode,
        rag_context=rag_context,
        web_search_results=web_search_results,
        rag_results=rag_results,
    )
    parts = [
        BASE_SYSTEM_PROMPT,
        context_rules,
        tool_rules,
        context_prompt,
        tools_prompt,
    ]
    return "\n\n".join(part for part in parts if part).strip()
