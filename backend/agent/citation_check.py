"""The graph step that enforces citations on a final answer (rules and checks: tools/rag/citations.py)."""

from __future__ import annotations

from typing import Any

from agent.debug_log import log_event
from agent.reviewer import USER_MESSAGE_MARK
from tools import rag
from tools.rag import citations

ERROR_PREFIX = "citationError:"


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return " ".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    return str(content or "")


def current_turn(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The messages since the user's latest message (the one that carries `userMessage:`)."""

    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user" and USER_MESSAGE_MARK in message_text(messages[index]):
            return messages[index:]
    return messages


def searched_this_turn(messages: list[dict[str, Any]]) -> bool:
    """True when the model itself called rag in this turn and got results back."""

    return any(
        message.get("role") == "tool"
        and message_text(message).startswith("ragResult:")
        and bool(citations.parse_sources(message_text(message)))
        for message in current_turn(messages)
    )


def citation_check(state: dict[str, Any]) -> dict[str, Any]:
    """Check the final answer's citations. A failing answer goes back to the model with the reasons (at most
    `MAX_RETRIES` times per turn); after that it is delivered with a visible notice instead of silently."""

    settings = state["settings"]
    answer = str(state.get("response") or "")
    if not settings.rag.require_citations or not answer.strip() or answer.startswith("Error:"):
        return {}
    messages = list(state.get("messages") or [])
    retrieved = citations.sources_in_messages(messages)
    problems = citations.check_answer(
        answer,
        retrieved=retrieved,
        searched_now=searched_this_turn(messages),
        lookup=lambda path, chunk: rag.chunk_text(settings.rag, path, chunk),
        history_compressed=bool(state.get("conversation_summary")),
    )
    if not problems:
        return {}
    retries = int(state.get("citation_retries") or 0)
    log_event("citation.failed", problems=problems, retries=retries, answer=answer[:500])
    event = {"type": "citation_check", "tool": "rag", "status": "retry" if retries < citations.MAX_RETRIES else "failed", "text": " ".join(problems)}
    if retries < citations.MAX_RETRIES:
        available = [source.tag for source in retrieved.values()]
        messages.append({"role": "user", "content": citations.correction_message(problems, available[-12:])})
        return {"messages": messages, "citation_retries": retries + 1, "response": "", "tool_events": [event]}
    return {"response": answer + citations.failure_notice(problems), "tool_events": [event]}


def route_after_citation_check(state: dict[str, Any]) -> str:
    messages = state.get("messages") or []
    if messages and messages[-1].get("role") == "user" and message_text(messages[-1]).startswith(ERROR_PREFIX):
        return "assistant_step"
    return "conversation_end"
