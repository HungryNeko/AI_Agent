"""How the small helper LLM calls of RAG (chunk plans, context, reranking) talk to a model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Thinking is wasted on these calls (a score, a sentence, a list of ranges), costs tokens and time, and in thinking
# mode some providers refuse a forced tool_choice. So thinking is switched off wherever the provider allows it.
NO_THINKING = "off"


def forced_tool_call(
    complete: Callable[..., dict[str, Any]],
    messages: list[dict[str, Any]],
    *,
    model: str | None,
    tool: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    """Ask for exactly one call of `tool`. A model that rejects a forced tool_choice is asked again with "auto"."""

    name = tool["function"]["name"]
    options: dict[str, Any] = {"model": model, "tools": [tool], "max_tokens": max_tokens, "reasoning_effort": NO_THINKING}
    try:
        return complete(messages, tool_choice={"type": "function", "function": {"name": name}}, **options)
    except RuntimeError as exc:
        if "tool_choice" not in str(exc):
            raise
        return complete(messages, tool_choice="auto", **options)
