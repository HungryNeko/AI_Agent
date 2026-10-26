"""The persisted index: chunks, their BM25 index and their vectors, built incrementally."""

from __future__ import annotations

import hashlib
import pickle
import threading
from dataclasses import dataclass
from typing import Any

from tools.rag import paths
from tools.rag.bm25 import Bm25Index
from tools.rag.documents import iter_documents
from tools.rag.options import resolved_options
from tools.rag.pipeline import prepare_chunks
from tools.rag.models import BuildOptions, RagChunk, RagDocument
from tools.rag.vectors import ApiVectors, LocalVectors, build_vectors
from tools.settings import RagSettings

INDEX_VERSION = 5
INDEX_MAGIC = b"AI_AGENT_RAG_INDEX_V5\n"
_BUILD_LOCK = threading.Lock()
_LOADED: dict[str, tuple[tuple[int, int], "RagIndex"]] = {}


@dataclass
class RagIndex:
    version: int
    built_at: str
    signature: str
    options: BuildOptions
    report: dict[str, Any]
    chunks: list[RagChunk]
    bm25: Bm25Index | None
    vectors: LocalVectors | ApiVectors | None
    pending_contexts: int = 0  # chunks still waiting for an LLM context (a refresh makes them)
    embedding_fallback: str = ""  # why the configured embedding model was not used

    @property
    def embedding_label(self) -> str:
        return self.vectors.label if self.vectors is not None else self.options.embedding_label

    @property
    def unfinished(self) -> bool:
        """True when a refresh could still improve this index (it is retried then, never while searching)."""

        return self.pending_contexts > 0 or bool(self.embedding_fallback)


def documents_signature(documents: list[RagDocument], settings: RagSettings, options: BuildOptions) -> str:
    digest = hashlib.sha256()
    for part in (
        INDEX_VERSION,
        settings.include_knowledge,
        settings.include_memory,
        settings.include_skills,
        options.split_mode,
        options.chunk_model,
        options.contextual,
        options.context_model,
        options.embedding_model,
        settings.max_file_bytes,
        settings.max_chunk_chars,
        settings.chunk_overlap_chars,
    ):
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    for document in documents:
        for part in (document.source_type, document.path.as_posix(), document.mtime_ns, document.size):
            digest.update(str(part).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def load_index(settings: RagSettings) -> RagIndex | None:
    """The saved index. Searches happen often and the file can be large, so it is unpickled once per version."""

    path = paths.resolve_project_path(settings.index_path, allow_missing=True)
    try:
        stat = path.stat()
    except OSError:
        return None
    stamp = (stat.st_mtime_ns, stat.st_size)
    remembered = _LOADED.get(str(path))
    if remembered is not None and remembered[0] == stamp:
        return remembered[1]
    try:
        with path.open("rb") as file:
            if file.read(len(INDEX_MAGIC)) != INDEX_MAGIC:
                return None
            index = pickle.load(file)  # noqa: S301 - the index is a cache this application wrote itself
    # The index is a disposable cache. Any incompatible or corrupt cache is rebuilt instead of blocking RAG.
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(index, RagIndex):
        return None
    _LOADED[str(path)] = (stamp, index)
    return index


def save_index(index: RagIndex, settings: RagSettings) -> None:
    path = paths.resolve_project_path(settings.index_path, allow_missing=True)
    paths.write_atomic(path, INDEX_MAGIC + pickle.dumps(index))


def build_index(settings: RagSettings, options: BuildOptions, *, force: bool = False, offline: bool = False) -> RagIndex:
    options = resolved_options(options)
    documents = iter_documents(settings)
    chunks, report = prepare_chunks(documents, settings, options, force=force, offline=offline)
    texts = [chunk.search_text for chunk in chunks]
    bm25 = Bm25Index.build(texts) if chunks else None
    vectors, vector_report = build_vectors(texts, options.embedding_model, settings, force=force) if chunks else (None, {"embedding_fallback": ""})
    report.update({key: value for key, value in vector_report.items() if key != "embedding_fallback"})
    index = RagIndex(
        version=INDEX_VERSION,
        built_at=paths.utc_now(),
        signature=documents_signature(documents, settings, options),
        options=options,
        report=report,
        chunks=chunks,
        bm25=bm25,
        vectors=vectors,
        pending_contexts=int(report["context_pending"]),
        embedding_fallback=str(vector_report.get("embedding_fallback") or ""),
    )
    save_index(index, settings)
    return index


def ensure_index(
    settings: RagSettings,
    options: BuildOptions | None = None,
    *,
    force: bool = False,
    offline: bool = False,
) -> RagIndex:
    """The current index, rebuilt first when the documents or the build options changed.

    Without explicit options a stale index is rebuilt the way it was built. `offline` is for searching: it never
    calls an LLM for contexts and never retries unfinished work, so a search stays as fast as the index allows.
    """

    existing = load_index(settings)
    chosen = options or (existing.options if existing is not None else BuildOptions())
    wanted = resolved_options(chosen)
    signature = documents_signature(iter_documents(settings, read_text=False), settings, wanted)
    stale = (
        force
        or existing is None
        or existing.version != INDEX_VERSION
        or existing.signature != signature
        or (not offline and existing.unfinished)
    )
    if not stale:
        return existing  # type: ignore[return-value]
    with _BUILD_LOCK:
        # Another thread may have finished the same rebuild while this one waited for the lock.
        current = load_index(settings)
        if current is not None and not force and current.signature == signature and current.version == INDEX_VERSION and (offline or not current.unfinished):
            return current
        return build_index(settings, chosen, force=force, offline=offline)
