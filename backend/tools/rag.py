"""Small file-backed RAG over knowledge, memory, and skills."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tools import memory, skills
from tools.settings import RagSettings

TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml"}


@dataclass(frozen=True)
class RagDocument:
    source_type: str
    path: Path
    text: str


@dataclass(frozen=True)
class ScoredChunk:
    source_type: str
    path: str
    score: float
    text: str


def auto_context(user_message: str, settings: RagSettings) -> str | None:
    """Return automatically injected RAG context when enabled."""

    if not settings.can_auto_include_context:
        return None
    results = search(user_message, settings)
    if not results:
        return None
    return "autoRagResult:\n" + "\n\n".join(results)


def search(query: str, settings: RagSettings) -> list[str]:
    """Search local knowledge, memory, and skill documents."""

    if not settings.can_model_call:
        raise ValueError("rag search is disabled for the model.")
    chunks = search_chunks(query, settings)
    return [format_chunk(chunk) for chunk in chunks]


def search_chunks(query: str, settings: RagSettings) -> list[ScoredChunk]:
    terms = tokenize(query)
    if not terms:
        return []

    scored: list[ScoredChunk] = []
    for document in iter_documents(settings):
        score = score_document(query, terms, document)
        if score <= 0:
            continue
        scored.append(
            ScoredChunk(
                source_type=document.source_type,
                path=relative_path(document.path),
                score=score,
                text=best_excerpt(document.text, terms, settings.max_chunk_chars),
            )
        )
    scored.sort(key=lambda item: (-item.score, item.path))
    return scored[: settings.max_results]


def iter_documents(settings: RagSettings) -> list[RagDocument]:
    documents: list[RagDocument] = []
    documents.extend(load_documents("knowledge", resolve_project_path(settings.knowledge_root), settings))
    documents.extend(
        RagDocument("memory", path, memory.read_text_file(path, max_bytes=settings.max_file_bytes))
        for path in memory.iter_memory_files(settings.memory_root)
    )
    documents.extend(
        RagDocument("skill", path, skills.read_text_file(path, max_bytes=settings.max_file_bytes))
        for path in skills.iter_skill_files(settings.skills_root)
    )
    return documents


def load_documents(source_type: str, root: Path, settings: RagSettings) -> list[RagDocument]:
    if not root.exists():
        return []
    return [
        RagDocument(source_type, path, read_text_file(path, max_bytes=settings.max_file_bytes))
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
    ]


def score_document(query: str, terms: list[str], document: RagDocument) -> float:
    text = document.text.lower()
    path = relative_path(document.path).lower()
    lowered_query = query.strip().lower()
    score = 0.0
    if lowered_query and lowered_query in text:
        score += 8.0
    if lowered_query and lowered_query in path:
        score += 4.0
    for term in terms:
        score += text.count(term)
        if term in path:
            score += 3.0
    if document.source_type == "skill" and document.path.name.upper() == "SKILL.MD":
        score += 0.5
    return score


def best_excerpt(text: str, terms: list[str], max_chars: int) -> str:
    clean = text.strip()
    if len(clean) <= max_chars:
        return clean
    lowered = clean.lower()
    positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
    start = max(0, min(positions) - max_chars // 4) if positions else 0
    end = min(len(clean), start + max_chars)
    excerpt = clean[start:end].strip()
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(clean):
        excerpt += "..."
    return excerpt


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    ascii_terms = re.findall(r"[a-z0-9_\-]{2,}", lowered)
    cjk_terms = re.findall(r"[\u4e00-\u9fff]", lowered)
    seen: set[str] = set()
    terms: list[str] = []
    for term in ascii_terms + cjk_terms:
        if term not in seen:
            terms.append(term)
            seen.add(term)
    return terms


def format_chunk(chunk: ScoredChunk) -> str:
    return (
        f"sourceType: {chunk.source_type}\n"
        f"path: {chunk.path}\n"
        f"score: {chunk.score:.2f}\n"
        f"content: {chunk.text}"
    )


def read_text_file(path: Path, *, max_bytes: int) -> str:
    if path.stat().st_size > max_bytes:
        return path.read_bytes()[:max_bytes].decode("utf-8", errors="replace")
    return path.read_text(encoding="utf-8", errors="replace")


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = project_root() / path
    root = project_root()
    resolved = path.resolve()
    resolved.relative_to(root)
    return resolved


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root())).replace("\\", "/")
    except ValueError:
        return str(path)