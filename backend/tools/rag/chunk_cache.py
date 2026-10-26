"""The on-disk cache that makes refreshing the index incremental.

`<index>.chunks.json` holds, per document, its chunk plan (keyed by content hash and chunking options), the plans
the LLM chunker made for parts of documents, and the contexts the LLM wrote for chunks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.rag import paths
from tools.rag.chunking import PreparedChunk
from tools.settings import RagSettings

CHUNK_CACHE_VERSION = 1
MAX_LLM_UNITS = 2_000
MAX_CONTEXTS = 20_000


def cache_path(settings: RagSettings) -> Path:
    index_path = paths.resolve_project_path(settings.index_path, allow_missing=True)
    return index_path.with_name(f"{index_path.stem}.chunks.json")


def empty_cache() -> dict[str, Any]:
    return {"version": CHUNK_CACHE_VERSION, "documents": {}, "llm_units": {}, "contexts": {}}


def load_cache(settings: RagSettings) -> dict[str, Any]:
    try:
        data = json.loads(cache_path(settings).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_cache()
    if not isinstance(data, dict) or data.get("version") != CHUNK_CACHE_VERSION:
        return empty_cache()
    return {**empty_cache(), **{key: value for key, value in data.items() if isinstance(value, dict) or key == "version"}}


def save_cache(settings: RagSettings, documents: dict[str, Any], llm_units: dict[str, Any], contexts: dict[str, str]) -> None:
    data = {
        "version": CHUNK_CACHE_VERSION,
        "documents": documents,
        "llm_units": dict(list(llm_units.items())[-MAX_LLM_UNITS:]),
        "contexts": dict(list(contexts.items())[-MAX_CONTEXTS:]),
    }
    paths.write_atomic(cache_path(settings), json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def cached_document_chunks(value: object, expected: dict[str, Any]) -> list[PreparedChunk]:
    if not isinstance(value, dict):
        return []
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        return []
    raw_chunks = value.get("chunks")
    if not isinstance(raw_chunks, list):
        return []
    chunks: list[PreparedChunk] = []
    for item in raw_chunks:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            return []
        chunks.append(PreparedChunk(item["text"], str(item.get("title") or "")))
    return chunks
