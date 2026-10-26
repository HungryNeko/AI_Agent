"""Dense vectors from a provider's OpenAI-compatible `/embeddings` endpoint."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from tools.rag import paths
from tools.rag.providers import post_json
from tools.settings import RagSettings

BATCH_SIZE = 48
MAX_TEXT_CHARS = 6_000
CACHE_MAGIC = b"AI_AGENT_RAG_EMBEDDINGS_V1\n"


def embed_texts(model: str, texts: list[str]) -> np.ndarray:
    """Unit-length vectors for the texts (rows in the same order)."""

    rows: list[np.ndarray] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = [text[:MAX_TEXT_CHARS] or " " for text in texts[start : start + BATCH_SIZE]]
        data = post_json(model, "/embeddings", {"input": batch})
        items = data.get("data")
        if not isinstance(items, list) or len(items) != len(batch):
            raise RuntimeError("embeddings response does not match the request.")
        for item in sorted(items, key=lambda entry: int(entry.get("index", 0))):
            vector = np.asarray(item.get("embedding"), dtype=np.float32)
            if vector.ndim != 1 or not vector.size:
                raise RuntimeError("embeddings response has an invalid vector.")
            rows.append(vector)
    matrix = np.vstack(rows) if rows else np.zeros((0, 0), dtype=np.float32)
    if matrix.size:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
    return matrix


def text_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\n{text[:MAX_TEXT_CHARS]}".encode()).hexdigest()


def cache_path(settings: RagSettings) -> Path:
    index_path = paths.resolve_project_path(settings.index_path, allow_missing=True)
    return index_path.with_name(f"{index_path.stem}.embeddings.pkl")


def load_cache(settings: RagSettings) -> dict[str, np.ndarray]:
    path = cache_path(settings)
    try:
        with path.open("rb") as file:
            if file.read(len(CACHE_MAGIC)) != CACHE_MAGIC:
                return {}
            cache = pickle.load(file)  # noqa: S301 - a cache this application wrote itself
    except Exception:  # noqa: BLE001 - a bad cache is only a cache miss
        return {}
    return cache if isinstance(cache, dict) else {}


def save_cache(settings: RagSettings, cache: dict[str, np.ndarray]) -> None:
    paths.write_atomic(cache_path(settings), CACHE_MAGIC + pickle.dumps(cache))


def embed_with_cache(model: str, texts: list[str], settings: RagSettings, *, force: bool) -> tuple[np.ndarray, dict[str, Any]]:
    """Embed every text, calling the provider only for texts whose vector is not cached.

    The cache keeps exactly the vectors of the current index, so deleted chunks drop out of it.
    """

    cache = {} if force else load_cache(settings)
    keys = [text_key(model, text) for text in texts]
    missing = [index for index, key in enumerate(keys) if key not in cache]
    if missing:
        fresh = embed_texts(model, [texts[index] for index in missing])
        for index, vector in zip(missing, fresh, strict=True):
            cache[keys[index]] = vector
    matrix = np.vstack([cache[key] for key in keys]) if keys else np.zeros((0, 0), dtype=np.float32)
    save_cache(settings, {key: cache[key] for key in keys})
    return matrix, {"embedded_texts": len(missing), "embedding_reused": len(keys) - len(missing)}
