"""Token-conscious simple and LLM-assisted chunk planning for local RAG."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from agent.llm import complete_chat_once

SplitMode = Literal["simple", "llm"]

CHUNK_PROMPT_VERSION = 1
LLM_MIN_SOURCE_CHARS = 1_800
LLM_MAX_SOURCE_CHARS = 60_000
LLM_UNIT_CHARS = 6_000
LLM_MAX_BLOCKS_PER_UNIT = 48
LLM_MAX_CALLS_PER_DOCUMENT = 10
LLM_MAX_OUTPUT_TOKENS = 700


@dataclass(frozen=True)
class PreparedChunk:
    text: str
    title: str = ""


@dataclass
class ChunkingMetrics:
    llm_calls: int = 0
    llm_input_characters: int = 0
    estimated_input_tokens: int = 0
    reused_units: int = 0
    fallbacks: int = 0
    fallback_reason: str = ""

    def model_view(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "llm_input_characters": self.llm_input_characters,
            "estimated_input_tokens": self.estimated_input_tokens,
            "reused_llm_units": self.reused_units,
            "llm_fallbacks": self.fallbacks,
            "fallback_reason": self.fallback_reason,
        }


def split_document(
    text: str,
    *,
    mode: SplitMode,
    model: str,
    max_chars: int,
    overlap_chars: int,
    unit_cache: dict[str, list[dict[str, str]]],
) -> tuple[list[PreparedChunk], ChunkingMetrics]:
    """Split one document while reusing unchanged LLM-planned units."""

    clean = text.strip()
    metrics = ChunkingMetrics()
    if not clean:
        return [], metrics
    if mode == "simple" or len(clean) < LLM_MIN_SOURCE_CHARS:
        return simple_split(clean, max_chars=max_chars, overlap_chars=overlap_chars), metrics
    if len(clean) > LLM_MAX_SOURCE_CHARS:
        metrics.fallbacks = 1
        metrics.fallback_reason = (
            f"source has {len(clean)} characters; LLM mode is capped at "
            f"{LLM_MAX_SOURCE_CHARS} to control token cost"
        )
        return simple_split(clean, max_chars=max_chars, overlap_chars=overlap_chars), metrics

    chunks: list[PreparedChunk] = []
    for unit in make_llm_units(clean):
        unit_text = join_blocks(unit)
        if len(unit_text) <= max_chars:
            chunks.append(PreparedChunk(unit_text))
            continue
        cache_key = unit_cache_key(unit_text, model=model, max_chars=max_chars)
        cached = decode_cached_chunks(unit_cache.get(cache_key))
        if cached:
            chunks.extend(cached)
            metrics.reused_units += 1
            continue
        if metrics.llm_calls >= LLM_MAX_CALLS_PER_DOCUMENT:
            metrics.fallbacks += 1
            metrics.fallback_reason = "per-document LLM call limit reached"
            chunks.extend(simple_split(unit_text, max_chars=max_chars, overlap_chars=overlap_chars))
            continue
        try:
            planned = request_chunk_plan(unit, model=model, max_chars=max_chars)
        except Exception:  # noqa: BLE001 - an optional optimization must safely fall back
            planned = []
        metrics.llm_calls += 1
        metrics.llm_input_characters += len(unit_text)
        metrics.estimated_input_tokens += estimate_tokens(unit_text) + 200
        if not planned:
            metrics.fallbacks += 1
            metrics.fallback_reason = "LLM returned no valid contiguous chunk plan"
            planned = simple_split(unit_text, max_chars=max_chars, overlap_chars=overlap_chars)
        unit_cache[cache_key] = [
            {"text": item.text, "title": item.title} for item in planned
        ]
        chunks.extend(planned)
    return chunks, metrics


def simple_split(text: str, *, max_chars: int, overlap_chars: int) -> list[PreparedChunk]:
    """Paragraph-aware deterministic splitting with bounded character overlap."""

    maximum = max(500, int(max_chars))
    overlap = min(max(0, int(overlap_chars)), maximum // 3)
    blocks = semantic_blocks(text, max_block_chars=maximum)
    if not blocks:
        return []
    chunks: list[PreparedChunk] = []
    current = ""
    for block in blocks:
        candidate = block if not current else f"{current}\n\n{block}"
        if current and len(candidate) > maximum:
            chunks.append(PreparedChunk(current.strip()))
            prefix = current[-overlap:].lstrip() if overlap else ""
            if prefix and len(prefix) + len(block) + 2 <= maximum:
                current = f"{prefix}\n\n{block}".strip()
            else:
                current = block
        else:
            current = candidate
    if current.strip():
        chunks.append(PreparedChunk(current.strip()))
    return chunks


def semantic_blocks(text: str, *, max_block_chars: int) -> list[str]:
    raw_blocks = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    blocks: list[str] = []
    for block in raw_blocks:
        if len(block) <= max_block_chars:
            blocks.append(block)
            continue
        sentences = [item.strip() for item in re.split(r"(?<=[。！？.!?])", block) if item.strip()]
        current = ""
        for sentence in sentences or [block]:
            if len(sentence) > max_block_chars:
                if current:
                    blocks.append(current)
                    current = ""
                blocks.extend(
                    sentence[start : start + max_block_chars]
                    for start in range(0, len(sentence), max_block_chars)
                )
                continue
            candidate = sentence if not current else f"{current} {sentence}"
            if current and len(candidate) > max_block_chars:
                blocks.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            blocks.append(current)
    return blocks


def make_llm_units(text: str) -> list[list[str]]:
    blocks = semantic_blocks(text, max_block_chars=1_200)
    units: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for block in blocks:
        starts_section = bool(re.match(r"^#{1,6}\s", block))
        would_overflow = current and (
            current_chars + len(block) + 2 > LLM_UNIT_CHARS
            or len(current) >= LLM_MAX_BLOCKS_PER_UNIT
        )
        if current and (starts_section or would_overflow):
            units.append(current)
            current = []
            current_chars = 0
        current.append(block)
        current_chars += len(block) + 2
    if current:
        units.append(current)
    return units


def request_chunk_plan(blocks: list[str], *, model: str, max_chars: int) -> list[PreparedChunk]:
    numbered = "\n\n".join(f"<BLOCK {index}>\n{block}" for index, block in enumerate(blocks))
    message = complete_chat_once(
        [
            {
                "role": "system",
                "content": (
                    "You plan semantic RAG chunk boundaries. Call submit_chunk_plan exactly once. "
                    "Never repeat or rewrite source text. Group every block exactly once, in order, "
                    "using contiguous inclusive 0-based ranges. Prefer topic-complete chunks and keep "
                    f"each group near 600-{max_chars} source characters. Titles must be under 60 characters."
                ),
            },
            {"role": "user", "content": numbered},
        ],
        model=model or None,
        tools=[chunk_plan_tool()],
        tool_choice={"type": "function", "function": {"name": "submit_chunk_plan"}},
        max_tokens=LLM_MAX_OUTPUT_TOKENS,
    )
    arguments = find_plan_arguments(message)
    groups = arguments.get("groups") if isinstance(arguments, dict) else None
    if not isinstance(groups, list):
        return []
    return validate_plan(groups, blocks, max_chars=max_chars)


def chunk_plan_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "submit_chunk_plan",
            "description": "Return only semantic block ranges and short titles; never return source text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "groups": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "startBlock": {"type": "integer"},
                                "endBlock": {"type": "integer"},
                                "title": {"type": "string"},
                            },
                            "required": ["startBlock", "endBlock", "title"],
                        },
                    }
                },
                "required": ["groups"],
            },
        },
    }


def find_plan_arguments(message: dict[str, Any]) -> dict[str, Any]:
    for call in message.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or function.get("name") != "submit_chunk_plan":
            continue
        raw = function.get("arguments")
        if not isinstance(raw, str):
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def validate_plan(
    groups: list[Any],
    blocks: list[str],
    *,
    max_chars: int,
) -> list[PreparedChunk]:
    chunks: list[PreparedChunk] = []
    expected_start = 0
    for group in groups:
        if not isinstance(group, dict):
            return []
        start = group.get("startBlock")
        end = group.get("endBlock")
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
            return []
        if start != expected_start or end < start or end >= len(blocks):
            return []
        text = join_blocks(blocks[start : end + 1])
        if len(text) > max_chars * 3 // 2:
            return []
        title = str(group.get("title") or "").strip()[:60]
        chunks.append(PreparedChunk(text=text, title=title))
        expected_start = end + 1
    if expected_start != len(blocks):
        return []
    return chunks


def unit_cache_key(text: str, *, model: str, max_chars: int) -> str:
    digest = hashlib.sha256()
    digest.update(str(CHUNK_PROMPT_VERSION).encode("ascii"))
    digest.update(model.encode("utf-8"))
    digest.update(str(max_chars).encode("ascii"))
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def decode_cached_chunks(value: object) -> list[PreparedChunk]:
    if not isinstance(value, list):
        return []
    chunks: list[PreparedChunk] = []
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            return []
        chunks.append(PreparedChunk(item["text"], str(item.get("title") or "")))
    return chunks


def join_blocks(blocks: list[str]) -> str:
    return "\n\n".join(block.strip() for block in blocks if block.strip()).strip()


def estimate_tokens(text: str) -> int:
    cjk = sum(1 for char in text if "\u3400" <= char <= "\u9fff")
    return cjk + (len(text) - cjk + 3) // 4
