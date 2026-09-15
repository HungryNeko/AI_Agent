"""Local vector RAG over knowledge, memory, and skills."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

from tools import memory, skills
from tools.fileReader import FileReadRequest
from tools.fileReader import read as read_file
from tools.fileReader import resolve_path as resolve_reader_path
from tools.rag_chunking import ChunkingMetrics, PreparedChunk, SplitMode, split_document
from tools.settings import FileReaderSettings, RagSettings

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
class RagRequest:
    action: Literal["search", "ingest"] = "search"
    query: str = ""
    path: str = ""
    name: str = ""
    split_mode: SplitMode = "simple"
    chunk_model: str = ""
    overwrite: bool = False


@dataclass(frozen=True)
class VectorIndex:
    version: int
    built_at: str
    signature: str
    embedding_model: str
    split_mode: SplitMode
    chunk_model: str
    chunking_report: dict[str, Any]
    chunks: list[RagChunk]
    embeddings: Any


INDEX_VERSION = 3
INDEX_MAGIC = b"AI_AGENT_RAG_INDEX_V3\n"
CHUNK_CACHE_VERSION = 1
MAX_INGESTED_CHARACTERS = 500_000


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


def ingest_uploaded_file(request: RagRequest, settings: RagSettings) -> dict[str, Any]:
    """Extract one uploaded document, save Markdown, and incrementally refresh RAG."""

    if request.action != "ingest":
        raise ValueError("rag ingestion requires action=ingest.")
    split_mode = normalize_split_mode(request.split_mode)
    reader_settings = FileReaderSettings(
        mode="auto",
        root=str(project_root()),
        max_file_bytes=25_000_000,
        max_output_chars=MAX_INGESTED_CHARACTERS,
    )
    source = resolve_reader_path(request.path, reader_settings)
    uploads = (project_root() / "backend" / "runtime" / "uploads").resolve()
    try:
        source.relative_to(uploads)
    except ValueError as exc:
        raise ValueError("rag ingest accepts uploaded files only.") from exc
    extracted = read_file(
        FileReadRequest(path=str(source), max_chars=MAX_INGESTED_CHARACTERS),
        reader_settings,
    )
    content = str(extracted.get("content") or "").strip()
    if not content:
        if extracted.get("needsOcr"):
            raise ValueError("uploaded document has no extractable text and requires OCR.")
        raise ValueError("uploaded document contains no readable text.")

    knowledge_root = resolve_project_path(settings.user_knowledge_root, allow_missing=True)
    knowledge_root.mkdir(parents=True, exist_ok=True)
    filename = clean_knowledge_filename(request.name or source.stem)
    target = (knowledge_root / filename).resolve()
    if not is_relative_to(target, knowledge_root):
        raise ValueError("knowledge filename must stay inside the user knowledge directory.")
    if target.exists() and not request.overwrite:
        raise ValueError(f"knowledge file already exists: {filename}")
    markdown = (
        f"# {Path(filename).stem}\n\n"
        f"> Imported from `{source.name}` using fileReader.\n\n"
        f"{content.rstrip()}\n"
    )
    target.write_text(markdown, encoding="utf-8")
    status = index_status(
        settings,
        split_mode=split_mode,
        chunk_model=request.chunk_model,
    )
    return {
        "status": "saved",
        "source": relative_path(source),
        "path": relative_path(target),
        "format": extracted.get("format"),
        "extracted_characters": extracted.get("extractedCharacters", len(content)),
        "truncated": bool(extracted.get("truncated")),
        "index": status,
    }


def search_chunks(query: str, settings: RagSettings) -> list[ScoredChunk]:
    clean_query = query.strip()
    if not clean_query:
        return []

    index = load_or_rebuild_index(settings)
    if not index.chunks or index.embeddings is None:
        return []

    query_vector = encode_texts(
        [clean_query],
        model_name=settings.embedding_model,
        input_type="query",
    )[0]
    scores = np.asarray(index.embeddings) @ query_vector
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


def rebuild_index(
    settings: RagSettings,
    *,
    split_mode: SplitMode = "simple",
    chunk_model: str = "",
    force_chunking: bool = False,
) -> VectorIndex:
    if split_mode == "llm":
        chunk_model = resolve_chunk_model(chunk_model)
    documents = iter_documents(settings)
    chunks, chunking_report = chunk_documents_cached(
        documents,
        settings,
        split_mode=split_mode,
        chunk_model=chunk_model,
        force=force_chunking,
    )
    signature = documents_signature(
        documents,
        settings,
        split_mode=split_mode,
        chunk_model=chunk_model,
    )
    embeddings = None
    if chunks:
        embeddings = encode_texts(
            [vector_text(chunk) for chunk in chunks],
            model_name=settings.embedding_model,
            input_type="passage",
        )
    index = VectorIndex(
        version=INDEX_VERSION,
        built_at=utc_now(),
        signature=signature,
        embedding_model=settings.embedding_model,
        split_mode=split_mode,
        chunk_model=chunk_model,
        chunking_report=chunking_report,
        chunks=chunks,
        embeddings=embeddings,
    )
    save_index(index, settings)
    return index


def load_or_rebuild_index(
    settings: RagSettings,
    *,
    split_mode: SplitMode | None = None,
    chunk_model: str = "",
    force: bool = False,
) -> VectorIndex:
    documents = iter_documents(settings)
    index = load_index(settings)
    selected_mode: SplitMode = split_mode or normalize_split_mode(
        getattr(index, "split_mode", "simple") if index else "simple"
    )
    selected_model = chunk_model.strip()
    if selected_mode == "llm" and not selected_model and index is not None:
        selected_model = str(getattr(index, "chunk_model", "") or "")
    if selected_mode == "llm":
        selected_model = resolve_chunk_model(selected_model)
    signature = documents_signature(
        documents,
        settings,
        split_mode=selected_mode,
        chunk_model=selected_model,
    )
    if (
        force
        or index is None
        or getattr(index, "version", None) != INDEX_VERSION
        or getattr(index, "embedding_model", None) != settings.embedding_model
        or getattr(index, "signature", None) != signature
    ):
        return rebuild_index(
            settings,
            split_mode=selected_mode,
            chunk_model=selected_model,
            force_chunking=force,
        )
    return index


def load_index(settings: RagSettings) -> VectorIndex | None:
    path = resolve_project_path(settings.index_path, allow_missing=True)
    if not path.is_file():
        return None
    try:
        with path.open("rb") as file:
            if file.read(len(INDEX_MAGIC)) != INDEX_MAGIC:
                return None
            index = pickle.load(file)
    # The index is a disposable cache. Old TF-IDF pickles may import sklearn while
    # unpickling, so any incompatible or corrupt cache should be rebuilt in place.
    except Exception:  # noqa: BLE001 - incompatible caches must never block a rebuild
        return None
    if not isinstance(index, VectorIndex):
        return None
    return index


def save_index(index: VectorIndex, settings: RagSettings) -> None:
    path = resolve_project_path(settings.index_path, allow_missing=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(INDEX_MAGIC)
        pickle.dump(index, file)


def index_status(
    settings: RagSettings,
    *,
    split_mode: SplitMode | None = None,
    chunk_model: str = "",
    force: bool = False,
) -> dict[str, object]:
    previous = load_index(settings)
    index = load_or_rebuild_index(
        settings,
        split_mode=split_mode,
        chunk_model=chunk_model,
        force=force,
    )
    reused_existing_index = (
        not force
        and previous is not None
        and getattr(previous, "signature", None) == index.signature
        and getattr(previous, "built_at", None) == index.built_at
    )
    operation_report = index.chunking_report
    if reused_existing_index:
        operation_report = {
            "documents_reused": len(iter_documents(settings)),
            "documents_processed": 0,
            "documents_removed": 0,
            "llm_calls": 0,
            "llm_input_characters": 0,
            "estimated_input_tokens": 0,
            "reused_llm_units": 0,
            "llm_fallbacks": 0,
            "fallback_reasons": [],
        }
    counts = {"knowledge": 0, "memory": 0, "skill": 0}
    for chunk in index.chunks:
        counts[chunk.source_type] = counts.get(chunk.source_type, 0) + 1
    return {
        "status": "ready",
        "index": "local-vector",
        "embedding": settings.embedding_model,
        "split_mode": index.split_mode,
        "chunk_model": index.chunk_model,
        "document_count": len(iter_documents(settings)),
        "chunk_count": len(index.chunks),
        "sources": counts,
        "built_at": index.built_at,
        "path": relative_path(resolve_project_path(settings.index_path, allow_missing=True)),
        "index_reused": reused_existing_index,
        **operation_report,
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
    chunks, _report = chunk_documents_cached(
        documents,
        settings,
        split_mode="simple",
        chunk_model="",
        force=False,
    )
    return chunks


def chunk_documents_cached(
    documents: list[RagDocument],
    settings: RagSettings,
    *,
    split_mode: SplitMode,
    chunk_model: str,
    force: bool,
) -> tuple[list[RagChunk], dict[str, Any]]:
    cache = load_chunk_cache(settings)
    cached_documents = cache.get("documents") if isinstance(cache.get("documents"), dict) else {}
    raw_unit_cache = cache.get("llm_units") if isinstance(cache.get("llm_units"), dict) else {}
    unit_cache: dict[str, list[dict[str, str]]] = {} if force else dict(raw_unit_cache)
    chunks: list[RagChunk] = []
    next_documents: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {
        "documents_reused": 0,
        "documents_processed": 0,
        "documents_removed": 0,
        "llm_calls": 0,
        "llm_input_characters": 0,
        "estimated_input_tokens": 0,
        "reused_llm_units": 0,
        "llm_fallbacks": 0,
        "fallback_reasons": [],
    }
    max_chars = max(500, int(settings.max_chunk_chars))
    overlap = min(max(0, int(settings.chunk_overlap_chars)), max_chars // 2)
    for document in documents:
        text = document.text.strip()
        if not text:
            continue
        path = relative_path(document.path)
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        expected = {
            "content_hash": content_hash,
            "split_mode": split_mode,
            "chunk_model": chunk_model,
            "max_chars": max_chars,
            "overlap_chars": overlap,
        }
        cached = cached_documents.get(path) if isinstance(cached_documents, dict) else None
        prepared = cached_document_chunks(cached, expected) if not force else []
        metrics = ChunkingMetrics()
        if prepared:
            report["documents_reused"] += 1
        else:
            prepared, metrics = split_document(
                text,
                mode=split_mode,
                model=chunk_model,
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
        for index, item in enumerate(prepared):
            chunk_text = item.text
            if item.title and not chunk_text.lstrip().startswith("#"):
                chunk_text = f"## {item.title}\n\n{chunk_text}"
            chunks.append(
                RagChunk(
                    source_type=document.source_type,
                    path=path,
                    chunk_index=index,
                    text=chunk_text,
                )
            )
    previous_paths = set(cached_documents) if isinstance(cached_documents, dict) else set()
    report["documents_removed"] = len(previous_paths - set(next_documents))
    trimmed_units = dict(list(unit_cache.items())[-2_000:])
    save_chunk_cache(
        settings,
        {
            "version": CHUNK_CACHE_VERSION,
            "documents": next_documents,
            "llm_units": trimmed_units,
        },
    )
    return chunks, report


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


def merge_chunking_metrics(report: dict[str, Any], metrics: ChunkingMetrics) -> None:
    values = metrics.model_view()
    for key in [
        "llm_calls",
        "llm_input_characters",
        "estimated_input_tokens",
        "reused_llm_units",
        "llm_fallbacks",
    ]:
        report[key] += int(values[key])
    reason = str(values.get("fallback_reason") or "")
    if reason and reason not in report["fallback_reasons"]:
        report["fallback_reasons"].append(reason)


def chunk_cache_path(settings: RagSettings) -> Path:
    index_path = resolve_project_path(settings.index_path, allow_missing=True)
    return index_path.with_name(f"{index_path.stem}.chunks.json")


def load_chunk_cache(settings: RagSettings) -> dict[str, Any]:
    path = chunk_cache_path(settings)
    if not path.is_file():
        return {"version": CHUNK_CACHE_VERSION, "documents": {}, "llm_units": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": CHUNK_CACHE_VERSION, "documents": {}, "llm_units": {}}
    if not isinstance(data, dict) or data.get("version") != CHUNK_CACHE_VERSION:
        return {"version": CHUNK_CACHE_VERSION, "documents": {}, "llm_units": {}}
    return data


def save_chunk_cache(settings: RagSettings, cache: dict[str, Any]) -> None:
    path = chunk_cache_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def documents_signature(
    documents: list[RagDocument],
    settings: RagSettings,
    *,
    split_mode: SplitMode,
    chunk_model: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(INDEX_VERSION).encode("utf-8"))
    digest.update(str(settings.include_knowledge).encode("utf-8"))
    digest.update(str(settings.include_memory).encode("utf-8"))
    digest.update(str(settings.include_skills).encode("utf-8"))
    digest.update(settings.embedding_model.encode("utf-8"))
    digest.update(split_mode.encode("ascii"))
    digest.update(chunk_model.encode("utf-8"))
    digest.update(str(settings.max_file_bytes).encode("utf-8"))
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


def encode_texts(
    texts: list[str],
    *,
    model_name: str,
    input_type: str,
) -> np.ndarray:
    """Encode normalized E5 query or passage vectors."""

    if input_type not in {"query", "passage"}:
        raise ValueError("input_type must be query or passage.")
    prefixed = [f"{input_type}: {text.strip()}" for text in texts]
    model = get_embedding_model(model_name)
    embeddings = model.encode(
        prefixed,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != len(texts):
        raise RuntimeError("embedding model returned an unexpected vector shape.")
    return array


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str) -> Any:
    """Load and cache the local Sentence Transformers embedding model."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for RAG embeddings; install backend dependencies."
        ) from exc
    return SentenceTransformer(model_name, device="cpu")


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


def normalize_split_mode(value: str) -> SplitMode:
    mode = value.strip().lower()
    if mode not in {"simple", "llm"}:
        raise ValueError("split_mode must be one of: simple, llm.")
    return mode


def resolve_chunk_model(value: str) -> str:
    clean = value.strip()
    if clean:
        return clean
    from agent.config import get_model_config

    config = get_model_config()
    return f"{config.provider}:{config.model_id}"


def clean_knowledge_filename(value: str) -> str:
    clean = Path(value.replace("\\", "/")).name.strip()
    if clean.lower().endswith(".md"):
        clean = clean[:-3]
    clean = re.sub(r"[^\w\-. ()\[\]\u4e00-\u9fff]+", "-", clean, flags=re.UNICODE).strip(" .-")
    if not clean or clean in {".", ".."}:
        raise ValueError("knowledge filename is invalid.")
    return f"{clean}.md"


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root())).replace("\\", "/")
    except ValueError:
        return str(path)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
