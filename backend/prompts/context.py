"""Prompt text for compressed conversation context."""

from __future__ import annotations

from datetime import datetime, timezone


CONTEXT_RULES = """
Context rules:
- conversationSummary is a compressed summary, not a complete log.
- Recent messages and current tool results are more reliable than the summary.
- If exact file contents, command output, or tool details matter, use fresh context.
- currentTime is fresh for the current turn. Use it for date-sensitive work.
""".strip()


def build_context_prompt(
    *,
    conversation_summary: str | None = None,
    include_rules: bool = False,
) -> str:
    lines: list[str] = []

    if include_rules:
        lines.append(CONTEXT_RULES)
    if conversation_summary:
        lines.append(format_conversation_summary(conversation_summary))

    return "\n".join(lines)


def build_context_rules_prompt(*, include_rules: bool = False) -> str:
    if include_rules:
        return CONTEXT_RULES
    return ""


def format_conversation_summary(summary: str) -> str:
    return f'conversationSummary: "{summary.strip()}"'


def format_current_time(now: datetime | None = None) -> str:
    current = now or datetime.now().astimezone()
    current_text = current.isoformat(timespec="seconds")
    utc_text = current.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return f'currentTime: "{current_text}"\nreferenceUTC: "{utc_text}"'
