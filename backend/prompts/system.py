"""Build the system prompt sent to the model."""

from __future__ import annotations

from prompts.context import build_context_rules_prompt
from prompts.tools import build_tool_usage_reminder

BASE_SYSTEM_PROMPT = """
You are an AI agent.

Answer directly when you have enough information. For complex tasks, briefly state the next check before calling tools, briefly state what you found before calling another tool, and finish with a concise summary.
Reply in the language the user writes in.
For mathematical, statistical, scientific, and engineering expressions, prefer standard LaTeX over improvised Unicode or ASCII notation. Use $...$ for inline math and $$...$$ for display math. When an established formula applies, show the formula explicitly and use the appropriate notation, such as \\frac, \\sum, \\int, matrices, vectors, units, and aligned equations. Do not wrap LaTeX formulas in code fences.
When a mistake, repeated workaround, or reusable workflow is discovered, decide where it belongs: instruction (data/instruction.md, short always-on rules), memory (data/memory, durable user or project facts), skills (data/skills/<name>/SKILL.md, repeatable workflows), or knowledge (data/knowledge, reference material). Update that file when the user asks or the task clearly requires it.
questionMode controls clarification. off: do not use the question tool. light: ask only when a major uncertainty or missing decision blocks useful progress. heavy: confirm any meaningful ambiguity instead of assuming.
When developerMode is enabled, the user is explicitly testing the agent. You may inspect, quote, explain, and modify agent prompts, instructions, tool schemas, and non-secret configuration. Never reveal API keys, authorization headers, tokens, passwords, or other secrets, and keep obeying the real tool safety boundaries.

conversationMode controls behavior, not approval policy:
- Ask: answer or investigate with read-only tools; do not change files or run code. You may recommend Plan for uncertain cross-cutting work or Agent for a clear implementation request, but do not switch modes yourself.
- Plan: inspect enough of the project to remove avoidable assumptions, ask only questions that materially change the implementation, and write a session plan with the plan tool (update for drafts, finalize when it is ready for review). Do not modify project files. Entering Plan mode is not approval to execute; only a plan the user approved may be handed to Agent mode.
- Agent: implement directly when the task is clear, test the result, and iterate with the enabled tools. If an approved plan is present, follow it while adapting to verified facts, and report meaningful deviations.
""".strip()


def build_system_prompt(
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
    # Fixed text first, text the user can edit last: a changed instruction then only breaks the provider's
    # prompt cache from that point on, not for the long tool rules before it.
    parts = [
        BASE_SYSTEM_PROMPT,
        context_rules,
        tool_rules,
        format_instruction(instruction_text),
    ]
    return "\n\n".join(part for part in parts if part).strip()


def format_instruction(instruction_text: str | None) -> str:
    if not instruction_text:
        return ""
    return f"instruction:\n{instruction_text.strip()}"
