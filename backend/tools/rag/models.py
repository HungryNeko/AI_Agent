"""Data types shared by the RAG modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

SplitMode = Literal["simple", "llm"]
SEARCH_METHODS = ("hybrid", "vector", "bm25")
SOURCE_TYPES = ("knowledge", "memory", "skill")
LOCAL_EMBEDDING = "tfidf-char-ngram"


@dataclass(frozen=True)
class RagDocument:
    source_type: str
    path: Path
    text: str
    mtime_ns: int = 0  # taken when the file was listed: together with the size they tell whether it changed
    size: int = 0


@dataclass(frozen=True)
class RagChunk:
    """One indexed piece of a document.

    `text` is what the model reads. `headings` (the section path) and `context` (a sentence the LLM wrote to
    situate the chunk in its document) are added only to the text that is indexed, so both BM25 and the
    vectors can match words the chunk itself never states.
    """

    source_type: str
    path: str
    chunk_index: int
    text: str
    headings: str = ""
    context: str = ""

    @property
    def search_text(self) -> str:
        parts = (self.source_type, self.path, self.headings, self.context, self.text)
        return "\n".join(part for part in parts if part)


@dataclass(frozen=True)
class ScoredChunk:
    source_type: str
    path: str
    chunk_index: int
    score: float
    text: str
    context: str = ""
    headings: str = ""
    via: tuple[tuple[str, int], ...] = ()  # (retriever, 1-based rank) for each retriever that found it
    reranked: bool = False


@dataclass(frozen=True)
class RagRequest:
    action: Literal["search", "ingest"] = "search"
    query: str = ""
    path: str = ""
    name: str = ""
    split_mode: SplitMode = "simple"
    chunk_model: str = ""
    overwrite: bool = False
    method: str = ""  # "" means the default (hybrid)
    rerank: bool | None = None  # None means the user's setting
    limit: int | None = None
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class BuildOptions:
    """How the index is built. The index remembers them, so a stale index is rebuilt the same way."""

    split_mode: SplitMode = "simple"
    chunk_model: str = ""
    contextual: Literal["off", "llm"] = "off"
    context_model: str = ""
    embedding_model: str = ""  # "" = the local character n-gram vectors, otherwise "provider:model"

    @property
    def embedding_label(self) -> str:
        return self.embedding_model.strip() or LOCAL_EMBEDDING


@dataclass
class SearchOutcome:
    results: list[ScoredChunk] = field(default_factory=list)
    method: str = "hybrid"
    retrievers: list[str] = field(default_factory=list)
    reranker: str = ""  # "", "llm" or "api"
    notes: list[str] = field(default_factory=list)
