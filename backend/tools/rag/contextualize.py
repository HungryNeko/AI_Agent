"""Contextual retrieval: let an LLM situate each chunk in its document before indexing.

A chunk such as "Revenue grew 3% over the previous quarter" is useless to search without knowing whose revenue
and which quarter. A short context written from the whole document ("From ACME's Q2 2025 report, section on
results: ...") is indexed together with the chunk, for BM25 and for the vectors alike. The prompt puts the
document first and the chunk last, so a provider's prompt cache serves the document part for every chunk.
"""

from __future__ import annotations

import hashlib
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from agent.debug_log import log_event
from agent.llm import complete_chat_once
from tools.rag.aux_llm import NO_THINKING
from tools.rag.chunking import estimate_tokens

CONTEXT_PROMPT_VERSION = 1
CONTEXT_PROMPT = (
    "You write retrieval context for one chunk of a document. Write one or two short sentences (at most 60 words) "
    "that situate the chunk in the whole document so that a search can find it: name the document or its subject, "
    "the section, and any entity, date or topic the chunk depends on but does not state itself. Use the language "
    "of the chunk. Answer with the context only, with no preface and no quotation of the chunk."
)
CONTEXT_MAX_TOKENS = 160
CONTEXT_MAX_CHARS = 400
WHOLE_DOCUMENT_CHARS = 24_000  # longer documents are cut to a window around the chunk
WINDOW_CHARS = 8_000
OUTLINE_CHARS = 1_500
CALLS_PER_DOCUMENT = 100
CALLS_PER_BUILD = 300
WORKERS = 4
GIVE_UP_AFTER_FAILURES = 3  # consecutive failures with no success: the model or key is wrong, stop spending calls
HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)
MATCH_CHARS = 60


@dataclass
class ContextBudget:
    """LLM calls one index build may still make; shared by all documents of the build."""

    calls_left: int
    successes: int = 0
    failures: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def exhausted(self) -> bool:
        return self.calls_left <= 0 or (self.failures >= GIVE_UP_AFTER_FAILURES and not self.successes)


def locate_chunks(document: str, chunk_texts: list[str]) -> list[int]:
    """Where each chunk starts in the document (a chunk's overlap tail is found at or after the previous start)."""

    offsets: list[int] = []
    cursor = 0
    for text in chunk_texts:
        found = document.find(text[:MATCH_CHARS], cursor) if text else -1
        if found >= 0:
            cursor = found
        offsets.append(cursor)
    return offsets


def outline(document: str) -> list[tuple[int, int, str]]:
    return [(match.start(), len(match.group(1)), match.group(2).strip()) for match in HEADING_RE.finditer(document)]


def heading_path(headings: list[tuple[int, int, str]], offset: int) -> str:
    """`Title > Section > Subsection` of the headings in force at the offset."""

    stack: list[tuple[int, str]] = []
    for start, level, title in headings:
        if start > offset:
            break
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
    return " > ".join(title for _level, title in stack)


def document_view(document: str, offset: int, headings: list[tuple[int, int, str]]) -> str:
    """What the model reads of the document: all of it when short, otherwise outline plus a window.

    The window start is rounded to a fixed grid, so neighbouring chunks send the same text and share the
    provider's prompt cache.
    """

    if len(document) <= WHOLE_DOCUMENT_CHARS:
        return document
    start = max(0, (offset // WINDOW_CHARS) * WINDOW_CHARS - WINDOW_CHARS // 2)
    toc = "\n".join(f"{'#' * level} {title}" for _start, level, title in headings)[:OUTLINE_CHARS]
    return f"[outline]\n{toc}\n[excerpt starting at character {start}]\n{document[start : start + 2 * WINDOW_CHARS]}"


def context_key(model: str, path: str, chunk_text: str) -> str:
    digest = hashlib.sha256()
    for part in (str(CONTEXT_PROMPT_VERSION), model, path, chunk_text):
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def request_context(model: str, view: str, chunk_text: str) -> str:
    message = complete_chat_once(
        [
            {"role": "system", "content": CONTEXT_PROMPT},
            {"role": "user", "content": f"<document>\n{view}\n</document>\n\n<chunk>\n{chunk_text}\n</chunk>"},
        ],
        model=model or None,
        max_tokens=CONTEXT_MAX_TOKENS,
        reasoning_effort=NO_THINKING,
    )
    text = " ".join(str(message.get("content") or "").split())
    text = re.sub(r"^(context|上下文)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    return text[:CONTEXT_MAX_CHARS]


def contextualize_chunks(
    *,
    path: str,
    document: str,
    chunk_texts: list[str],
    offsets: list[int],
    model: str,
    cache: dict[str, str],
    budget: ContextBudget,
    report: dict[str, Any],
) -> list[str]:
    """One context per chunk: from the cache when known, otherwise from the LLM within the call budget.

    Chunks that cannot be served now get an empty context and are counted in `context_pending`, so the next
    refresh continues where this one stopped.
    """

    contexts = [""] * len(chunk_texts)
    keys = [context_key(model, path, text) for text in chunk_texts]
    misses: list[int] = []
    for index, key in enumerate(keys):
        if key in cache:
            contexts[index] = cache[key]
            report["context_reused"] += 1
        else:
            misses.append(index)
    headings = outline(document)
    served = 0
    for start in range(0, len(misses), WORKERS):
        batch = misses[start : start + WORKERS][: max(0, CALLS_PER_DOCUMENT - served)]
        with budget.lock:
            batch = batch[: max(0, budget.calls_left)]
            if not batch or budget.exhausted():
                break
            budget.calls_left -= len(batch)
        served += len(batch)

        def one(index: int) -> str:
            try:
                return request_context(model, document_view(document, offsets[index], headings), chunk_texts[index])
            except Exception as exc:  # noqa: BLE001 - context is an optimisation; the chunk is indexed without it
                log_event("rag.context_error", path=path, error=str(exc)[:200])
                return ""

        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            results = list(pool.map(one, batch))
        for index, context in zip(batch, results, strict=True):
            report["context_calls"] += 1
            report["context_input_tokens"] += estimate_tokens(document_view(document, offsets[index], headings)) + estimate_tokens(chunk_texts[index]) + 80
            if context:
                cache[keys[index]] = context
                contexts[index] = context
                budget.successes += 1
            else:
                report["context_failures"] += 1
                budget.failures += 1
    report["context_pending"] += sum(1 for index in misses if not contexts[index])
    return contexts
