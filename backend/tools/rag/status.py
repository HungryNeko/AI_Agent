"""Refresh the index and report what happened."""

from __future__ import annotations

from typing import Any

from tools.rag import paths
from tools.rag.documents import iter_documents
from tools.rag.index import load_index, ensure_index
from tools.rag.models import BuildOptions
from tools.settings import RagSettings

REUSED_REPORT = {
    "documents_processed": 0,
    "documents_removed": 0,
    "llm_calls": 0,
    "llm_input_characters": 0,
    "estimated_input_tokens": 0,
    "reused_llm_units": 0,
    "llm_fallbacks": 0,
    "fallback_reasons": [],
    "context_calls": 0,
    "context_reused": 0,
    "context_failures": 0,
    "context_input_tokens": 0,
    "embedded_texts": 0,
}


def index_status(
    settings: RagSettings,
    options: BuildOptions | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Bring the index up to date (calling the LLM for chunk plans and contexts where switched on) and describe it."""

    previous = load_index(settings)
    index = ensure_index(settings, options, force=force)
    reused = (
        not force
        and previous is not None
        and previous.signature == index.signature
        and previous.built_at == index.built_at
    )
    documents = iter_documents(settings, read_text=False)
    report = {**index.report, **REUSED_REPORT, "documents_reused": len(documents)} if reused else index.report
    counts = {"knowledge": 0, "memory": 0, "skill": 0}
    for chunk in index.chunks:
        counts[chunk.source_type] = counts.get(chunk.source_type, 0) + 1
    return {
        "status": "ready",
        "index": "hybrid",
        "retrievers": ["bm25", "vector"],
        "embedding": index.embedding_label,
        "embedding_fallback": index.embedding_fallback,
        "split_mode": index.options.split_mode,
        "chunk_model": index.options.chunk_model,
        "contextual": index.options.contextual,
        "context_model": index.options.context_model if index.options.contextual == "llm" else "",
        "context_pending": index.pending_contexts,
        "document_count": len(documents),
        "chunk_count": len(index.chunks),
        "sources": counts,
        "built_at": index.built_at,
        "path": paths.relative_path(paths.resolve_project_path(settings.index_path, allow_missing=True)),
        "index_reused": reused,
        **report,
    }
