"""Local hybrid RAG over knowledge, memory and skills.

Pipeline: documents are split into chunks (simple or LLM-planned), each chunk can get an LLM-written context
that situates it in its document, and the context plus the chunk are indexed twice: for BM25 keyword search and
as vectors (local character n-grams, or a provider's embedding model). A search runs BM25, vectors or both,
fuses the rankings with reciprocal rank fusion and can rerank the best candidates with a model.

Modules: `models` (types), `paths`, `documents`, `chunking`, `chunk_cache`, `contextualize`, `pipeline`,
`bm25`, `vectors` + `api_embeddings`, `fusion`, `rerank`, `index`, `query` (search), `ingest`, `status`, `options`.
See docs/rag-ingestion.md.
"""

from tools.rag.index import INDEX_MAGIC, INDEX_VERSION, RagIndex, build_index, ensure_index, load_index
from tools.rag.ingest import clean_knowledge_filename, ingest_uploaded_file
from tools.rag.models import (
    SEARCH_METHODS,
    SOURCE_TYPES,
    BuildOptions,
    RagChunk,
    RagDocument,
    RagRequest,
    ScoredChunk,
    SearchOutcome,
)
from tools.rag.options import normalize_method, normalize_sources, normalize_split_mode, saved_build_options
from tools.rag.query import auto_context, chunk_text, format_chunk, run_search, search, search_chunks
from tools.rag.status import index_status

__all__ = [
    "INDEX_MAGIC",
    "INDEX_VERSION",
    "SEARCH_METHODS",
    "SOURCE_TYPES",
    "BuildOptions",
    "RagChunk",
    "RagDocument",
    "RagIndex",
    "RagRequest",
    "ScoredChunk",
    "SearchOutcome",
    "auto_context",
    "chunk_text",
    "build_index",
    "clean_knowledge_filename",
    "ensure_index",
    "format_chunk",
    "index_status",
    "ingest_uploaded_file",
    "load_index",
    "normalize_method",
    "normalize_sources",
    "normalize_split_mode",
    "run_search",
    "saved_build_options",
    "search",
    "search_chunks",
]
