"""From documents to indexed chunks: split, add the section path, optionally add LLM context."""

from __future__ import annotations

import hashlib
from typing import Any

from tools.rag import chunk_cache, contextualize, paths
from tools.rag.chunking import ChunkingMetrics, split_document
from tools.rag.models import BuildOptions, RagChunk, RagDocument
from tools.settings import RagSettings


def new_report(options: BuildOptions) -> dict[str, Any]:
    return {
        "documents_reused": 0,
        "documents_processed": 0,
        "documents_removed": 0,
        "llm_calls": 0,
        "llm_input_characters": 0,
        "estimated_input_tokens": 0,
        "reused_llm_units": 0,
        "llm_fallbacks": 0,
        "fallback_reasons": [],
        "contextual": options.contextual,
        "context_model": options.context_model if options.contextual == "llm" else "",
        "context_calls": 0,
        "context_reused": 0,
        "context_failures": 0,
        "context_pending": 0,
        "context_input_tokens": 0,
    }


def merge_chunking_metrics(report: dict[str, Any], metrics: ChunkingMetrics) -> None:
    values = metrics.model_view()
    for key in ("llm_calls", "llm_input_characters", "estimated_input_tokens", "reused_llm_units", "llm_fallbacks"):
        report[key] += int(values[key])
    reason = str(values.get("fallback_reason") or "")
    if reason and reason not in report["fallback_reasons"]:
        report["fallback_reasons"].append(reason)


def prepare_chunks(
    documents: list[RagDocument],
    settings: RagSettings,
    options: BuildOptions,
    *,
    force: bool,
    offline: bool,
) -> tuple[list[RagChunk], dict[str, Any]]:
    """Chunk every document, reusing what the cache already knows.

    `offline` builds make no LLM call for contexts (the search path must not stall on them); chunks without a
    cached context are counted as pending and filled in by the next refresh.
    """

    cache = chunk_cache.load_cache(settings)
    cached_documents = cache["documents"]
    unit_cache: dict[str, Any] = {} if force else dict(cache["llm_units"])
    context_cache: dict[str, str] = {} if force else dict(cache["contexts"])
    report = new_report(options)
    use_context = options.contextual == "llm"
    budget = contextualize.ContextBudget(calls_left=0 if offline or not use_context else contextualize.CALLS_PER_BUILD)
    max_chars = max(500, int(settings.max_chunk_chars))
    overlap = min(max(0, int(settings.chunk_overlap_chars)), max_chars // 2)
    chunks: list[RagChunk] = []
    next_documents: dict[str, dict[str, Any]] = {}
    for document in documents:
        text = document.text.strip()
        if not text:
            continue
        path = paths.relative_path(document.path)
        expected = {
            "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "split_mode": options.split_mode,
            "chunk_model": options.chunk_model,
            "max_chars": max_chars,
            "overlap_chars": overlap,
        }
        prepared = [] if force else chunk_cache.cached_document_chunks(cached_documents.get(path), expected)
        if prepared:
            report["documents_reused"] += 1
        else:
            prepared, metrics = split_document(
                text,
                mode=options.split_mode,
                model=options.chunk_model,
                max_chars=max_chars,
                overlap_chars=overlap,
                unit_cache=unit_cache,
            )
            report["documents_processed"] += 1
            merge_chunking_metrics(report, metrics)
        next_documents[path] = {
            **expected,
            "source_type": document.source_type,
            "chunks": [{"text": item.text, "title": item.title} for item in prepared],
        }
        chunks.extend(index_chunks(document.source_type, path, text, prepared, options, context_cache, budget, report, use_context))
    report["documents_removed"] = len(set(cached_documents) - set(next_documents))
    chunk_cache.save_cache(settings, next_documents, unit_cache, context_cache)
    return chunks, report


def index_chunks(
    source_type: str,
    path: str,
    document_text: str,
    prepared: list[Any],
    options: BuildOptions,
    context_cache: dict[str, str],
    budget: contextualize.ContextBudget,
    report: dict[str, Any],
    use_context: bool,
) -> list[RagChunk]:
    texts = [item.text for item in prepared]
    offsets = contextualize.locate_chunks(document_text, texts)
    headings = contextualize.outline(document_text)
    contexts = [""] * len(texts)
    if use_context:
        contexts = contextualize.contextualize_chunks(
            path=path,
            document=document_text,
            chunk_texts=texts,
            offsets=offsets,
            model=options.context_model,
            cache=context_cache,
            budget=budget,
            report=report,
        )
    result: list[RagChunk] = []
    for index, item in enumerate(prepared):
        chunk_text = item.text
        if item.title and not chunk_text.lstrip().startswith("#"):
            chunk_text = f"## {item.title}\n\n{chunk_text}"
        result.append(
            RagChunk(
                source_type=source_type,
                path=path,
                chunk_index=index,
                text=chunk_text,
                headings=contextualize.heading_path(headings, offsets[index]),
                context=contexts[index],
            )
        )
    return result
