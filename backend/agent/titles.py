"""Conversation titles: a short LLM-written name with a plain-text fallback."""

from __future__ import annotations

import re
import threading
from typing import Any

from agent.debug_log import log_exception
from agent.llm import complete_chat_once

TITLE_MAX_CHARS = 30
_PROMPT = (
    "You name chat conversations. Reply with ONE short title (at most 12 words, or 20 Chinese characters) "
    "that says what the conversation is about. Use the same language as the user. "
    "No quotes, no trailing punctuation, no prefix such as 'Title:'."
)


def heuristic_title(user_text: str) -> str:
    """Cheap fallback: the first meaningful sentence of the user's message."""

    text = re.sub(r"@\S+", " ", user_text or "")
    text = re.sub(r"!\[[^\]]*]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"`{1,3}", " ", text)
    text = re.sub(r"^[#>*\-\s]+", "", text, flags=re.MULTILINE)
    for line in text.splitlines():
        line = " ".join(line.split())
        if not line:
            continue
        sentence = re.split(r"[。！？!?；;\n]", line, maxsplit=1)[0].strip(" ,，、:：.")
        if sentence:
            return _clip(sentence)
    return ""


def clean_generated_title(raw: str) -> str:
    stripped = (raw or "").strip()
    text = stripped.splitlines()[0] if stripped else ""
    text = re.sub(r"^(title|标题|主题)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    text = text.strip(" \t\"'`“”‘’《》「」#*.。,，:：")
    return _clip(text)


def generate_title(user_text: str, answer_text: str, model: str | None = None) -> str:
    fallback = heuristic_title(user_text)
    user_snippet = (user_text or "").strip()[:600]
    if not user_snippet:
        return fallback
    answer_snippet = (answer_text or "").strip()[:600]
    try:
        reply = complete_chat_once(
            [
                {"role": "system", "content": _PROMPT},
                {"role": "user", "content": f"User message:\n{user_snippet}\n\nAssistant reply:\n{answer_snippet}"},
            ],
            model=model or None,
            max_tokens=60,
        )
        title = clean_generated_title(str(reply.get("content") or ""))
    except Exception as exc:  # noqa: BLE001 - titles must never break a chat turn
        log_exception("title.generate_error", exc)
        return fallback
    return title or fallback


def schedule_title(conversation_id: str, user_text: str, answer_text: str, model: str | None, store: Any) -> None:
    """Replace the provisional title in the background once the model has written a better one."""

    def work() -> None:
        title = generate_title(user_text, answer_text, model)
        if title:
            store.set_auto_title(conversation_id, title)

    threading.Thread(target=work, name=f"title-{conversation_id}", daemon=True).start()


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= TITLE_MAX_CHARS else text[: TITLE_MAX_CHARS - 1].rstrip() + "…"
