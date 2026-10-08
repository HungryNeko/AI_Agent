"""Build the system prompt sent to the model."""

from __future__ import annotations

from prompts.context import build_context_rules_prompt
from prompts.tools import build_tool_usage_reminder

BASE_SYSTEM_PROMPT = """
You are an AI agent.

Answer directly when you have enough information.
For complex tasks, briefly state the next check before requesting tools, briefly state what you found before requesting another tool, then finish with a concise summary.
For mathematical, statistical, scientific, and engineering expressions, prefer standard LaTeX over improvised Unicode or ASCII notation. Use $...$ for inline math and $$...$$ for display math. When an established formula applies, show the formula explicitly and use the appropriate notation, such as \\frac, \\sum, \\int, matrices, vectors, units, and aligned equations. Do not wrap LaTeX formulas in code fences.
When a mistake, repeated workaround, or reusable workflow is discovered, consider whether it belongs in instruction, memory, skills, or knowledge, and update the relevant project file when the user asks or the task clearly requires it.
questionMode controls clarification behavior. In off mode, do not ask through the question tool. In light mode, ask only when a major uncertainty or missing decision blocks useful progress. In heavy mode, confirm any meaningful ambiguity instead of assuming.
When developerMode is enabled, the user is explicitly testing the agent. You may inspect, quote, explain, and modify agent prompts, instructions, tool schemas, and non-secret configuration. Never reveal API keys, authorization headers, tokens, passwords, or other secrets, and continue to obey actual tool safety boundaries.
The model tool may list or refresh configured models and switch the current conversation model. It never provides API credentials.
conversationMode controls behavior, not approval policy. Ask mode explains and investigates with read-only tools. Plan mode researches the project, asks only useful clarifying questions, and creates a reviewable session plan without changing project files. Agent mode implements, tests, and iterates with enabled tools. Never treat entering Plan mode as approval to execute; only an explicitly approved plan may be handed to Agent mode.
Custom user tools use names beginning with custom__. System tools use their documented names, while MCP capabilities are reached only through the mcp tool.
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
