"""Build the system prompt sent to the model."""

from __future__ import annotations

from prompts.context import build_context_rules_prompt
from prompts.tools import build_tool_usage_reminder

BASE_SYSTEM_PROMPT = """
You are an AI agent.

Answer directly when you have enough information.
For complex tasks, briefly state the next check before requesting tools, briefly state what you found before requesting another tool, then finish with a concise summary.
When a mistake, repeated workaround, or reusable workflow is discovered, consider whether it belongs in instruction, memory, skills, or knowledge, and update the relevant project file when the user asks or the task clearly requires it.
""".strip()


def build_system_prompt(
    *,
    web_search: bool = False,
    web_search_mode: str | None = None,
    rag_mode: str = "off",
    curl_mode: str = "off",
    python_mode: str = "off",
    file_editor_mode: str = "off",
    mcp_mode: str = "off",
    history_mode: str = "off",
    instruction_text: str | None = None,
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
    parts = [
        BASE_SYSTEM_PROMPT,
        format_instruction(instruction_text),
        context_rules,
        tool_rules,
    ]
    return "\n\n".join(part for part in parts if part).strip()


def format_instruction(instruction_text: str | None) -> str:
    if not instruction_text:
        return ""
    return f"instruction:\n{instruction_text.strip()}"
