"""Reciprocal rank fusion of several ranked lists."""

from __future__ import annotations

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: dict[str, list[int]],
    *,
    k: int = RRF_K,
    retrievers: int | None = None,
) -> list[tuple[int, float, dict[str, int]]]:
    """Merge ranked chunk positions from several retrievers.

    Each retriever gives a position `1 / (k + rank)`; the sum decides the order. Ranks, not scores, are
    combined, because BM25 scores and cosine similarities are on different scales and cannot be added.
    Returns (position, fused score, {retriever: 1-based rank}), best first. The fused score is divided by the
    best score a chunk could get (first in every retriever that ran), so it is between 0 and 1: a chunk only one of
    two retrievers found cannot score above 0.5, even when the other found nothing at all.
    """

    fused: dict[int, float] = {}
    seen_in: dict[int, dict[str, int]] = {}
    for name, positions in rankings.items():
        for rank, position in enumerate(positions, start=1):
            fused[position] = fused.get(position, 0.0) + 1.0 / (k + rank)
            seen_in.setdefault(position, {})[name] = rank
    ceiling = max(retrievers or 0, len(rankings)) / (k + 1) if rankings else 1.0
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return [(position, score / ceiling, seen_in[position]) for position, score in ordered]
