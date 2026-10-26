"""Find the documents that feed the index: knowledge, memory and skills."""

from __future__ import annotations

from pathlib import Path

from tools import memory, skills
from tools.rag import paths
from tools.rag.models import RagDocument
from tools.settings import RagSettings

TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml"}


def iter_documents(settings: RagSettings, *, read_text: bool = True) -> list[RagDocument]:
    """Every source document. `read_text=False` leaves the text empty: enough to tell whether anything changed."""

    def text_of(read, path: Path) -> str:
        return read(path, max_bytes=settings.max_file_bytes) if read_text else ""

    def document(source_type: str, read, path: Path) -> RagDocument:
        stat = path.stat()
        return RagDocument(source_type, path, text_of(read, path), stat.st_mtime_ns, stat.st_size)

    documents: list[RagDocument] = []
    if settings.include_knowledge:
        documents.extend(load_from_roots("knowledge", [settings.knowledge_root, settings.user_knowledge_root], settings, read_text))
    if settings.include_memory:
        documents.extend(
            document("memory", memory.read_text_file, path)
            for path in unique_paths(memory.iter_memory_files, [settings.memory_root, settings.user_memory_root])
        )
    if settings.include_skills:
        documents.extend(
            document("skill", skills.read_text_file, path)
            for path in unique_paths(skills.iter_skill_files, [settings.skills_root, settings.user_skills_root])
        )
    return documents


def unique_paths(lister, roots: list[str]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for path in lister(root):
            if path not in seen:
                seen.add(path)
                found.append(path)
    return found


def load_from_roots(source_type: str, roots: list[str], settings: RagSettings, read_text: bool) -> list[RagDocument]:
    documents: list[RagDocument] = []
    seen: set[Path] = set()
    for root_text in roots:
        root = paths.resolve_project_path(root_text, allow_missing=True)
        for document in load_documents(source_type, root, settings, read_text):
            if document.path not in seen:
                seen.add(document.path)
                documents.append(document)
    return documents


def load_documents(source_type: str, root: Path, settings: RagSettings, read_text: bool) -> list[RagDocument]:
    if not root.exists():
        return []
    documents = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():  # the suffix is free, is_file is a system call
            continue
        stat = path.stat()
        text = paths.read_text_file(path, max_bytes=settings.max_file_bytes) if read_text else ""
        documents.append(RagDocument(source_type, path, text, stat.st_mtime_ns, stat.st_size))
    return documents
