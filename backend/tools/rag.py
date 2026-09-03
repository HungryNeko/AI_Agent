"""Local vector RAG over knowledge, memory, and skills."""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

from tools import memory, skills
from tools.settings import RagSettings

TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml"}


@dataclass(frozen=True)
class RagDocument:
    source_type: str
    path: Path
    text: str


@dataclass(frozen=True)
class RagChunk:
    source_type: str
    path: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class ScoredChunk:
    source_type: str
    path: str
    chunk_index: int
    score: float
    text: str


@dataclass(frozen=True)
class VectorIndex:
    version: int
    built_at: str
    signature: str
    chunks: list[RagChunk]
    vectorizer: Any
    matrix: Any


INDEX_VERSION = 1


def auto_context(user_message: str, settings: RagSettings) -> str | None:
    """Return automatically injected RAG context when enabled."""

    if not settings.can_auto_include_context:
        return None
    results = search(user_message, settings)
    if not results:
        return None
    return "autoRagResult:\n" + "\n\n".join(results)


def search(query: str, settings: RagSettings) -> list[str]:
    """Search the persisted vector index."""

    if not settings.can_model_call:
        raise ValueError("rag search is disabled for the model.")
    chunks = search_chunks(query, settings)
    return [format_chunk(chunk) for chunk in chunks]


def search_chunks(query: str, settings: RagSettings) -> list[ScoredChunk]:
    clean_query = query.strip()
    if not clean_query:
        return []

    index = load_or_rebuild_index(settings)
    if not index.chunks or index.matrix is None or index.vectorizer is None:
        return []

    query_vector = index.vectorizer.transform([clean_query])
    scores = (index.matrix @ query_vector.T).toarray().ravel()
    min_score = max(0.0, float(settings.min_similarity))
    ranked: list[ScoredChunk] = []
    for position, score in enumerate(scores):
        if score < min_score:
            continue
        chunk = index.chunks[position]
        ranked.append(
            ScoredChunk(
                source_type=chunk.source_type,
                path=chunk.path,
                chunk_index=chunk.chunk_index,
                score=float(score),
                text=chunk.text,
            )
        )
    ranked.sort(key=lambda item: (-item.score, item.path, item.chunk_index))
    return ranked[: settings.max_results]


def rebuild_index(settings: RagSettings) -> VectorIndex:
    documents = iter_documents(settings)
    chunks = chunk_documents(documents, settings)
    signature = documents_signature(documents, settings)
    vectorizer = None
    matrix = None
    if chunks:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            lowercase=True,
            sublinear_tf=True,
            norm="l2",
        )
        matrix = vectorizer.fit_transform(vector_text(chunk) for chunk in chunks)
    index = VectorIndex(
        version=INDEX_VERSION,
        built_at=utc_now(),
        signature=signature,
        chunks=chunks,
        vectorizer=vectorizer,
        matrix=matrix,
    )
    save_index(index, settings)
    return index


def load_or_rebuild_index(settings: RagSettings) -> VectorIndex:
    documents = iter_documents(settings)
    signature = documents_signature(documents, settings)
    index = load_index(settings)
    if (
        index is None
        or index.version != INDEX_VERSION
        or index.signature != signature
    ):
        return rebuild_index(settings)
    return index


def load_index(settings: RagSettings) -> VectorIndex | None:
    path = resolve_project_path(settings.index_path, allow_missing=True)
    if not path.is_file():
        return None
    with path.open("rb") as file:
        index = pickle.load(file)
    if not isinstance(index, VectorIndex):
        return None
    return index


def save_index(index: VectorIndex, settings: RagSettings) -> None:
    path = resolve_project_path(settings.index_path, allow_missing=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(index, file)


def index_status(settings: RagSettings) -> dict[str, object]:
    index = load_or_rebuild_index(settings)
    counts = {"knowledge": 0, "memory": 0, "skill": 0}
    for chunk in index.chunks:
        counts[chunk.source_type] = counts.get(chunk.source_type, 0) + 1
    return {
        "status": "ready",
        "index": "local-vector",
        "embedding": "tfidf-char-ngram",
        "document_count": len(iter_documents(settings)),
        "chunk_count": len(index.chunks),
        "sources": counts,
        "built_at": index.built_at,
        "path": relative_path(resolve_project_path(settings.index_path, allow_missing=True)),
    }


def iter_documents(settings: RagSettings) -> list[RagDocument]:
    documents: list[RagDocument] = []
    if settings.include_knowledge:
        documents.extend(load_documents_from_roots("knowledge", [settings.knowledge_root, settings.user_knowledge_root], settings))
    if settings.include_memory:
        documents.extend(
            RagDocument("memory", path, memory.read_text_file(path, max_bytes=settings.max_file_bytes))
            for path in iter_memory_roots([settings.memory_root, settings.user_memory_root])
        )
    if settings.include_skills:
        documents.extend(
            RagDocument("skill", path, skills.read_text_file(path, max_bytes=settings.max_file_bytes))
            for path in iter_skill_roots([settings.skills_root, settings.user_skills_root])
        )
    return documents


def load_documents_from_roots(source_type: str, roots: list[str], settings: RagSettings) -> list[RagDocument]:
    documents: list[RagDocument] = []
    seen: set[Path] = set()
    for root_text in roots:
        root = resolve_project_path(root_text, allow_missing=True)
        for document in load_documents(source_type, root, settings):
            if document.path in seen:
                continue
            seen.add(document.path)
            documents.append(document)
    return documents


def iter_memory_roots(roots: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for path in memory.iter_memory_files(root):
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)
    return paths


def iter_skill_roots(roots: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for path in skills.iter_skill_files(root):
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)
    return paths


def load_documents(source_type: str, root: Path, settings: RagSettings) -> list[RagDocument]:
    if not root.exists():
        return []
    return [
        RagDocument(source_type, path, read_text_file(path, max_bytes=settings.max_file_bytes))
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
    ]


def chunk_documents(documents: list[RagDocument], settings: RagSettings) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    max_chars = max(500, int(settings.max_chunk_chars))
    overlap = min(max(0, int(settings.chunk_overlap_chars)), max_chars // 2)
    step = max_chars - overlap
    for document in documents:
        text = document.text.strip()
        if not text:
            continue
        path = relative_path(document.path)
        for index, start in enumerate(range(0, len(text), step)):
            chunk_text = text[start : start + max_chars].strip()
            if chunk_text:
                chunks.append(
                    RagChunk(
                        source_type=document.source_type,
                        path=path,
                        chunk_index=index,
                        text=chunk_text,
                    )
                )
            if start + max_chars >= len(text):
                break
    return chunks


def documents_signature(documents: list[RagDocument], settings: RagSettings) -> str:
    digest = hashlib.sha256()
    digest.update(str(INDEX_VERSION).encode("utf-8"))
    digest.update(str(settings.include_knowledge).encode("utf-8"))
    digest.update(str(settings.include_memory).encode("utf-8"))
    digest.update(str(settings.include_skills).encode("utf-8"))
    digest.update(str(settings.max_chunk_chars).encode("utf-8"))
    digest.update(str(settings.chunk_overlap_chars).encode("utf-8"))
    for document in documents:
        stat = document.path.stat()
        digest.update(document.source_type.encode("utf-8"))
        digest.update(relative_path(document.path).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
    return digest.hexdigest()


def vector_text(chunk: RagChunk) -> str:
    return f"{chunk.source_type}\n{chunk.path}\n{chunk.text}"


def format_chunk(chunk: ScoredChunk) -> str:
    return (
        f"sourceType: {chunk.source_type}\n"
        f"path: {chunk.path}\n"
        f"chunk: {chunk.chunk_index}\n"
        f"score: {chunk.score:.4f}\n"
        f"content: {chunk.text}"
    )


def read_text_file(path: Path, *, max_bytes: int) -> str:
    if path.stat().st_size > max_bytes:
        return path.read_bytes()[:max_bytes].decode("utf-8", errors="replace")
    return path.read_text(encoding="utf-8", errors="replace")


def resolve_project_path(path_text: str, *, allow_missing: bool = False) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = project_root() / path
    resolved = path.resolve() if path.exists() else path.parent.resolve() / path.name
    if not allow_missing and not resolved.exists():
        raise ValueError(f"path does not exist: {path_text}")
    return resolved


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root())).replace("\\", "/")
    except ValueError:
        return str(path)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
