"""Ingest an uploaded document into user knowledge as Markdown, then refresh the index."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tools.fileReader import FileReadRequest
from tools.fileReader import read as read_file
from tools.fileReader import resolve_path as resolve_reader_path
from tools.rag import paths
from tools.rag.models import BuildOptions, RagRequest
from tools.rag.options import normalize_split_mode, saved_build_options
from tools.rag.status import index_status
from tools.settings import FileReaderSettings, RagSettings

MAX_INGESTED_CHARACTERS = 500_000


def ingest_uploaded_file(request: RagRequest, settings: RagSettings, options: BuildOptions | None = None) -> dict[str, Any]:
    """Extract one uploaded document, save it as Markdown, and incrementally refresh RAG.

    Without explicit `options`, the saved RAG settings apply, with the request's split mode and model on top.
    """

    if request.action != "ingest":
        raise ValueError("rag ingestion requires action=ingest.")
    split_mode = normalize_split_mode(request.split_mode)
    root = paths.project_root()
    reader_settings = FileReaderSettings(
        mode="auto",
        root=str(root),
        max_file_bytes=25_000_000,
        max_output_chars=MAX_INGESTED_CHARACTERS,
    )
    source = resolve_reader_path(request.path, reader_settings)
    uploads = (root / "backend" / "runtime" / "uploads").resolve()
    if not paths.is_relative_to(source, uploads):
        raise ValueError("rag ingest accepts uploaded files only.")
    extracted = read_file(FileReadRequest(path=str(source), max_chars=MAX_INGESTED_CHARACTERS), reader_settings)
    content = str(extracted.get("content") or "").strip()
    if not content:
        if extracted.get("needsOcr"):
            raise ValueError("uploaded document has no extractable text and requires OCR.")
        raise ValueError("uploaded document contains no readable text.")

    knowledge_root = paths.resolve_project_path(settings.user_knowledge_root, allow_missing=True)
    knowledge_root.mkdir(parents=True, exist_ok=True)
    filename = clean_knowledge_filename(request.name or source.stem)
    target = (knowledge_root / filename).resolve()
    if not paths.is_relative_to(target, knowledge_root):
        raise ValueError("knowledge filename must stay inside the user knowledge directory.")
    if target.exists() and not request.overwrite:
        raise ValueError(f"knowledge file already exists: {filename}")
    target.write_text(
        f"# {Path(filename).stem}\n\n> Imported from `{source.name}` using fileReader.\n\n{content.rstrip()}\n",
        encoding="utf-8",
    )
    status = index_status(settings, options or saved_build_options(split_mode, request.chunk_model))
    return {
        "status": "saved",
        "source": paths.relative_path(source),
        "path": paths.relative_path(target),
        "format": extracted.get("format"),
        "extracted_characters": extracted.get("extractedCharacters", len(content)),
        "truncated": bool(extracted.get("truncated")),
        "index": status,
    }


def clean_knowledge_filename(value: str) -> str:
    clean = Path(value.replace("\\", "/")).name.strip()
    if clean.lower().endswith(".md"):
        clean = clean[:-3]
    clean = re.sub(r"[^\w\-. ()\[\]一-鿿]+", "-", clean, flags=re.UNICODE).strip(" .-")
    if not clean or clean in {".", ".."}:
        raise ValueError("knowledge filename is invalid.")
    return f"{clean}.md"
