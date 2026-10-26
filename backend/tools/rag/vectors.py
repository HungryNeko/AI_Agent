"""The vector side of hybrid search: local character n-gram vectors, or provider embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from agent.debug_log import log_event
from tools.rag.api_embeddings import embed_texts, embed_with_cache
from tools.rag.models import LOCAL_EMBEDDING
from tools.settings import RagSettings


def top_scores(scores: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """Positions of the highest positive scores, best first."""

    if not scores.size:
        return []
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [(int(position), float(scores[position])) for position in order if scores[position] > 0]


class LocalVectors:
    """TF-IDF over character 2-5-grams.

    It needs no model or download and matches word parts, typos and Chinese substrings, but it is lexical:
    it does not know that two differently worded sentences mean the same. A real embedding model does.
    """

    kind = "local"
    label = LOCAL_EMBEDDING

    def __init__(self, texts: list[str]) -> None:
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True, sublinear_tf=True, norm="l2")
        self.matrix = self.vectorizer.fit_transform(texts)

    def query(self, text: str, top_k: int) -> list[tuple[int, float]]:
        scores = (self.matrix @ self.vectorizer.transform([text]).T).toarray().ravel()
        return top_scores(scores, top_k)


class ApiVectors:
    """Dense embeddings of the chunks from a provider's embedding model; the query is embedded when searching."""

    kind = "api"

    def __init__(self, model: str, matrix: np.ndarray) -> None:
        self.model = model
        self.matrix = matrix

    @property
    def label(self) -> str:
        return self.model

    def query(self, text: str, top_k: int) -> list[tuple[int, float]]:
        vector = embed_texts(self.model, [text])[0]
        return top_scores(self.matrix @ vector, top_k)


def build_vectors(
    texts: list[str],
    embedding_model: str,
    settings: RagSettings,
    *,
    force: bool,
) -> tuple[LocalVectors | ApiVectors, dict[str, Any]]:
    """Vectors for the chunk texts. A failing provider falls back to the local vectors, and says so."""

    model = embedding_model.strip()
    if not model:
        return LocalVectors(texts), {"embedding_fallback": ""}
    try:
        matrix, report = embed_with_cache(model, texts, settings, force=force)
        return ApiVectors(model, matrix), {**report, "embedding_fallback": ""}
    except Exception as exc:  # noqa: BLE001 - search must keep working without the provider
        log_event("rag.embedding_fallback", model=model, error=str(exc))
        return LocalVectors(texts), {"embedding_fallback": f"{model}: {str(exc)[:200]}"}
