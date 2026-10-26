"""Enforced citations: a code-level check that an answer built on RAG results points at real sources.

A rag result carries a `cite:` tag, `[path#chunk]`. When the model answers after searching, this module checks the
answer against what was really retrieved, so a prompt is not the only thing standing between the model and an
invented source:

- every `[path#chunk]` in the answer must be a chunk a search returned in this conversation (or a chunk that exists
  in the index, when the history was compressed and the result is no longer visible);
- a text quoted right before a citation must really occur in that chunk;
- an answer that follows a search must cite at least one source, or say `Sources: none used`.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

MAX_RETRIES = 2
MIN_QUOTE_CHARS = 12  # shorter quoted texts are not checked (they match almost anything)
MIN_QUOTE_PART = 8

CITE_RE = re.compile(r"\[([^\[\]#\n]{3,300}?)#(\d{1,5})\]")
SOURCE_BLOCK_RE = re.compile(
    r"(?ms)^cite: \[(?P<path>[^\[\]#\n]+)#(?P<chunk>\d+)\]\n(?:[A-Za-z]+: [^\n]*\n)*?content: (?P<text>.*?)"
    r"(?=^(?:\d+\. )?sourceType: |^note: |\Z)"
)
QUOTE_RE = re.compile(
    r"(?P<open>[\"“「『])(?P<quote>[^\"”」』\n]{%d,600}?)(?P<close>[\"”」』])"
    r"[\s.,;:。，；：]*(?P<tag>\[[^\[\]#\n]{3,300}?#\d{1,5}\])" % MIN_QUOTE_CHARS
)
BLOCKQUOTE_RE = re.compile(r"(?m)^>\s*(?P<quote>[^\n\[]{%d,600}?)\s*(?P<tag>\[[^\[\]#\n]{3,300}?#\d{1,5}\])\s*$" % MIN_QUOTE_CHARS)
ELLIPSIS_RE = re.compile(r"\.\.\.|…|\[\s*(?:\.\.\.|…)\s*\]")
NO_SOURCE_RE = re.compile(r"(?im)^\W*(?:sources?\s*:\s*none(?:\s+used)?|来源\s*[:：]\s*(?:无|未使用))\W*$")
PATH_HINT_RE = re.compile(r"/|\.(?:md|txt|json|ya?ml)$", re.IGNORECASE)
CODE_RE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)  # a tag inside code is text about a tag, not a citation


@dataclass(frozen=True)
class Source:
    path: str
    chunk: int
    text: str

    @property
    def tag(self) -> str:
        return cite_tag(self.path, self.chunk)


def cite_tag(path: str, chunk: int) -> str:
    return f"[{path}#{chunk}]"


def squash(text: str) -> str:
    """Lower-case letters and digits only, so a quote still matches after spacing, punctuation or width changes."""

    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())


def parse_sources(text: str) -> list[Source]:
    """The sources in a formatted rag result (`ragResult:` or `autoRagResult:`)."""

    return [Source(match["path"], int(match["chunk"]), match["text"].strip()) for match in SOURCE_BLOCK_RE.finditer(text)]


def sources_in_messages(messages: list[dict]) -> dict[tuple[str, int], Source]:
    """Every source a search returned in these messages (tool results and automatically injected context)."""

    found: dict[tuple[str, int], Source] = {}
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
        if not isinstance(content, str):
            continue
        searched = message.get("role") == "tool" and content.startswith("ragResult:")
        injected = message.get("role") == "user" and ("autoRagResult:" in content or "ragResult:" in content)
        if searched or injected:
            for source in parse_sources(content):
                found[(source.path, source.chunk)] = source
    return found


def citations_in(answer: str) -> list[tuple[str, int]]:
    """Citation-looking tags `[path#n]`; a bracket that does not look like a file path is not a citation."""

    return [(match[1], int(match[2])) for match in CITE_RE.finditer(answer) if PATH_HINT_RE.search(match[1])]


def quotes_in(answer: str) -> list[tuple[str, str]]:
    """(quoted text, tag) for each text quoted directly before a citation."""

    pairs = [(match["quote"], match["tag"]) for match in QUOTE_RE.finditer(answer)]
    pairs += [(match["quote"], match["tag"]) for match in BLOCKQUOTE_RE.finditer(answer)]
    return pairs


def quote_problem(quote: str, source_text: str) -> bool:
    """True when some part of the quote (split at an ellipsis) is not in the source text."""

    haystack = squash(source_text)
    parts = [squash(part) for part in ELLIPSIS_RE.split(quote)]
    return any(len(part) >= MIN_QUOTE_PART and part not in haystack for part in parts)


def check_answer(
    answer: str,
    *,
    retrieved: dict[tuple[str, int], Source],
    searched_now: bool,
    lookup: Callable[[str, int], str | None],
    history_compressed: bool,
) -> list[str]:
    """What is wrong with the citations of an answer, as short instructions for the model; empty when it is fine.

    `retrieved`: sources searches returned in this conversation. `searched_now`: the model called rag in this turn
    and got results, so the answer must cite. `lookup(path, chunk)`: the chunk text from the index, or None.
    """

    problems: list[str] = []
    prose = CODE_RE.sub(" ", answer)
    cited = list(dict.fromkeys(citations_in(prose)))
    texts: dict[tuple[str, int], str] = {}
    for key in cited:
        tag = cite_tag(*key)
        if key in retrieved:
            texts[key] = retrieved[key].text
            continue
        indexed = lookup(*key)
        if indexed is None:
            problems.append(f"{tag} does not exist. Cite only the `cite:` tags that rag results show, copied exactly.")
        elif history_compressed:
            texts[key] = indexed  # the result was compressed away, but the chunk is real
        else:
            problems.append(f"{tag} exists but no search returned it in this conversation. Search for it with rag before citing it.")
    for quote, tag in quotes_in(prose):
        match = CITE_RE.fullmatch(tag)
        key = (match[1], int(match[2])) if match else None
        if key in texts and quote_problem(quote, texts[key]):
            problems.append(f'The quoted text "{quote[:80]}" is not in {tag}. Quote exactly or paraphrase without quotation marks.')
    if searched_now and not cited and not NO_SOURCE_RE.search(answer):
        problems.append(
            "The answer follows a rag search but cites nothing. Cite each fact taken from the results as [path#chunk] "
            'using their `cite:` tags, or, if no result was relevant, say so and end with the line "Sources: none used".'
        )
    return problems


def correction_message(problems: list[str], available: list[str]) -> str:
    listed = ", ".join(available[:12]) or "none"
    return (
        "citationError: your answer was not delivered because its citations failed the system check.\n"
        + "\n".join(f"- {problem}" for problem in problems)
        + f"\nCitable tags from this conversation: {listed}\nWrite the complete answer again with correct citations."
    )


def failure_notice(problems: list[str]) -> str:
    return "\n\n> Citation check failed, treat the sources with care: " + " ".join(problems)
