"""Search: BM25 and vectors in parallel, fused with reciprocal rank fusion, optionally reranked."""

from __future__ import annotations

from tools.rag import fusion
from tools.rag.citations import cite_tag
from tools.rag.index import RagIndex, ensure_index, load_index
from tools.rag.options import normalize_method, normalize_sources
from tools.rag.rerank import rerank
from tools.rag.models import RagChunk, RagRequest, ScoredChunk, SearchOutcome
from tools.settings import RagSettings

MAX_LIMIT = 10
MIN_POOL = 20
MAX_POOL = 50


def auto_context(user_message: str, settings: RagSettings) -> str | None:
    """RAG context added to a user turn automatically (hybrid fusion, never a reranker: it runs on every turn)."""

    if not settings.can_auto_include_context:
        return None
    results = search(user_message, settings, rerank_results=False)
    if not results:
        return None
    return "autoRagResult:\n" + "\n\n".join(results)


def search(
    query: str,
    settings: RagSettings,
    *,
    method: str = "",
    rerank_results: bool | None = None,
    limit: int | None = None,
    sources: tuple[str, ...] = (),
) -> list[str]:
    """The formatted results of a search over the persisted index."""

    if not settings.can_model_call:
        raise ValueError("rag search is disabled for the model.")
    outcome = search_chunks(query, settings, method=method, rerank_results=rerank_results, limit=limit, sources=sources)
    return [format_chunk(chunk) for chunk in outcome.results]


def search_chunks(
    query: str,
    settings: RagSettings,
    *,
    method: str = "",
    rerank_results: bool | None = None,
    limit: int | None = None,
    sources: tuple[str, ...] = (),
) -> SearchOutcome:
    clean_query = query.strip()
    chosen = normalize_method(method or settings.default_method)
    outcome = SearchOutcome(method=chosen)
    if not clean_query:
        return outcome
    index = ensure_index(settings, offline=True)
    if not index.chunks:
        return outcome
    wanted = max(1, min(MAX_LIMIT, int(limit or settings.max_results)))
    pool = max(MIN_POOL, min(MAX_POOL, wanted * 4, settings.candidates))
    allowed = allowed_positions(index.chunks, normalize_sources(sources))
    rankings = retrieve(index, clean_query, chosen, pool, allowed, settings, outcome)
    candidates = fuse(index, rankings, retrievers=len(outcome.retrievers))
    if not candidates:
        return outcome
    mode = reranker_mode(settings, rerank_results)
    if mode:
        try:
            candidates = rerank(clean_query, candidates, mode=mode, model=settings.rerank_model)
            outcome.reranker = mode
        except Exception as exc:  # noqa: BLE001 - a failing reranker must not lose the search
            outcome.notes.append(f"rerank ({mode}) unavailable: {str(exc)[:160]}; results are in fused order")
    outcome.results = candidates[:wanted]
    return outcome


def chunk_text(settings: RagSettings, path: str, chunk: int) -> str | None:
    """The text of one indexed chunk, or None. Never rebuilds the index: it only answers "does this chunk exist"."""

    index = load_index(settings)
    if index is None:
        return None
    return next((item.text for item in index.chunks if item.path == path and item.chunk_index == chunk), None)


def allowed_positions(chunks: list[RagChunk], sources: tuple[str, ...]) -> set[int] | None:
    if not sources:
        return None
    return {position for position, chunk in enumerate(chunks) if chunk.source_type in sources}


def retrieve(
    index: RagIndex,
    query: str,
    method: str,
    pool: int,
    allowed: set[int] | None,
    settings: RagSettings,
    outcome: SearchOutcome,
) -> dict[str, list[int]]:
    """Ranked chunk positions per retriever that the method uses (`outcome.retrievers` lists the ones that ran)."""

    rankings: dict[str, list[int]] = {}

    def keyword() -> None:
        if index.bm25 is not None:
            rankings["bm25"] = [position for position, _score in index.bm25.search(query, pool, allowed)]
            outcome.retrievers.append("bm25")

    if method in {"hybrid", "bm25"}:
        keyword()
    if method in {"hybrid", "vector"} and index.vectors is not None:
        try:
            hits = index.vectors.query(query, pool if allowed is None else len(index.chunks))
        except Exception as exc:  # noqa: BLE001 - e.g. the embedding provider is down: keyword search still works
            outcome.notes.append(f"vector search unavailable ({str(exc)[:160]}); used bm25 only")
            if "bm25" not in rankings:
                keyword()
        else:
            floor = max(0.0, float(settings.min_similarity))
            rankings["vector"] = [position for position, score in hits if score >= floor and (allowed is None or position in allowed)][:pool]
            outcome.retrievers.append(f"vector[{index.embedding_label}]")
    return {name: positions for name, positions in rankings.items() if positions}


def fuse(index: RagIndex, rankings: dict[str, list[int]], *, retrievers: int) -> list[ScoredChunk]:
    fused = fusion.reciprocal_rank_fusion(rankings, retrievers=retrievers)
    return [
        ScoredChunk(
            source_type=index.chunks[position].source_type,
            path=index.chunks[position].path,
            chunk_index=index.chunks[position].chunk_index,
            score=score,
            text=index.chunks[position].text,
            context=index.chunks[position].context,
            headings=index.chunks[position].headings,
            via=tuple(sorted(via.items(), key=lambda item: item[1])),
        )
        for position, score, via in fused
    ]


def reranker_mode(settings: RagSettings, requested: bool | None) -> str:
    """"llm" or "api" when this search should be reranked, else "". An explicit request beats the setting."""

    if requested is False:
        return ""
    if settings.rerank != "off":
        return settings.rerank
    return "llm" if requested else ""


def run_search(request: RagRequest, settings: RagSettings) -> str:
    """The `ragResult:` text for a rag tool call: which retrievers ran, any notes, then the results."""

    if not settings.can_model_call:
        raise ValueError("rag search is disabled for the model.")
    outcome = search_chunks(
        request.query,
        settings,
        method=request.method,
        rerank_results=request.rerank,
        limit=request.limit,
        sources=request.sources,
    )
    return format_outcome(outcome)


def format_outcome(outcome: SearchOutcome) -> str:
    if not outcome.results:
        lines = ['ragResult: "no results"']
    else:
        lines = ["ragResult:"]
        lines.append(f"retrieval: {outcome.method} ({' + '.join(outcome.retrievers) or 'none'}), reranker: {outcome.reranker or 'none'}")
    lines.extend(f"note: {note}" for note in outcome.notes)
    for number, chunk in enumerate(outcome.results, start=1):
        lines.append(f"{number}. {format_chunk(chunk)}")
    return "\n".join(lines)


def format_chunk(chunk: ScoredChunk) -> str:
    lines = [
        f"sourceType: {chunk.source_type}",
        f"path: {chunk.path}",
        f"chunk: {chunk.chunk_index}",
        f"cite: {cite_tag(chunk.path, chunk.chunk_index)}",
        f"score: {chunk.score:.4f}{' (reranked)' if chunk.reranked else ''}",
    ]
    if chunk.via:
        lines.append("foundBy: " + ", ".join(f"{name} #{rank}" for name, rank in chunk.via))
    if chunk.headings:
        lines.append(f"section: {chunk.headings}")
    if chunk.context:
        lines.append(f"context: {chunk.context}")
    lines.append(f"content: {chunk.text}")
    return "\n".join(lines)
