# RAG ingestion route

This document describes the ingestion path used by the Data UI and the agent. It is
application behavior, not development-agent memory.

## Uploaded document route

```text
PDF / DOCX / PPTX / XLSX / HTML / text upload
  -> POST /api/uploads
  -> fileReader extracts exact source text (read-only, bounded)
  -> POST /api/rag/ingest or rag action=ingest
  -> backend/runtime/user_data/knowledge/<name>.md
  -> simple or LLM chunk planner
  -> multilingual-e5-small passage embeddings
  -> backend/runtime/rag_index/index.pkl
```

The agent should call `fileReader` first so it can verify the attachment, then call
`rag` with `action=ingest` and the original upload path. It must not copy the full
document into tool arguments. The ingestion backend invokes `fileReader` again to save
the complete extracted text rather than a possibly truncated model-visible preview.
Extraction is capped at 500,000 returned characters and the RAG source reader at
2,000,000 UTF-8 bytes; the API reports `truncated` when the extraction cap is reached.

## Chunk modes

### Simple

- Runs locally and consumes no LLM tokens.
- Splits on headings and paragraph boundaries where possible.
- Splits oversized paragraphs on sentence boundaries and then hard character limits.
- Applies bounded overlap for retrieval continuity.

### LLM

- Uses the dedicated model selected in the Data page.
- Files shorter than 1,800 characters use simple splitting because an LLM call adds
  little value.
- Files longer than 60,000 characters automatically use simple splitting to cap cost.
- Other files are grouped into stable heading/paragraph units of at most about 6,000
  characters and 48 blocks.
- The LLM receives numbered blocks and must call `submit_chunk_plan`. It returns only
  contiguous block ranges and titles, never the source text.
- Each response is capped at 700 output tokens and each document at 10 LLM calls.
- Invalid plans and model failures fall back to simple splitting without blocking RAG.

## Incremental refresh

`backend/runtime/rag_index/index.chunks.json` stores content hashes, per-document
chunks, and reusable LLM unit plans. The cache is ignored by Git.

- Unchanged documents reuse their complete chunk plan.
- A changed document is processed again, but unchanged LLM units reuse their hashes.
- Adding a document processes only the new document.
- Deleting a document removes only its document entry; no LLM call is needed for the
  remaining documents.
- Changing chunk mode, chunk model, or chunk sizing intentionally invalidates the
  affected cached plans.
- `force=true` on `/api/rag/reindex` bypasses chunk-plan reuse.

The E5 vector index has its own format/version marker. Old TF-IDF or incompatible
pickle files are never deserialized and are rebuilt safely.

## APIs

- `POST /api/rag/ingest`: upload path to user-knowledge Markdown and index.
- `POST /api/rag/reindex`: refresh with `split_mode`, `chunk_model`, and optional
  `force`.
- `PUT /api/data/file`, `POST /api/data/import`, and
  `POST /api/data/file/rename`: save and refresh using the selected chunk mode.
- `DELETE /api/data/file`: delete a writable user file and incrementally refresh.

Refresh responses expose processed/reused/removed document counts, LLM call count,
estimated input tokens, reused LLM units, and fallback reasons for UI visibility.
