# RAG: ingestion, indexing and search

This document describes how documents get into the local RAG index and how a search ranks them. It is application
behavior, not development-agent memory. The code is the package `backend/tools/rag/`; each module has one job.

## Overview

```text
document (knowledge / memory / skill, or an upload)
  -> chunks               simple or LLM-planned split                       rag/chunking.py
  -> chunk context        optional: an LLM situates each chunk in its doc   rag/contextualize.py
  -> indexed text         source + path + section path + context + chunk    rag/models.py (RagChunk.search_text)
       +-> BM25 index     keyword search                                    rag/bm25.py
       +-> vectors        local character n-grams, or an embedding model    rag/vectors.py, rag/api_embeddings.py
search
  -> BM25 and/or vector top candidates
  -> reciprocal rank fusion                                                 rag/fusion.py
  -> optional rerank by an LLM or a /rerank endpoint                        rag/rerank.py
  -> top results with their section and context                             rag/query.py
```

| Module | Job |
| --- | --- |
| `models.py` | `RagChunk`, `ScoredChunk`, `RagRequest`, `BuildOptions`, `SearchOutcome` |
| `paths.py` | project paths, atomic writes (the one place tests patch) |
| `documents.py` | find the documents: knowledge, memory, skills |
| `chunking.py`, `chunk_cache.py` | split documents; the incremental cache |
| `contextualize.py` | contextual retrieval (section path + LLM context) |
| `pipeline.py` | documents to indexed chunks, using the cache |
| `bm25.py` | tokenizer and BM25 |
| `vectors.py`, `api_embeddings.py`, `providers.py` | local vectors, provider embeddings, provider HTTP calls |
| `fusion.py`, `rerank.py` | rank fusion, reranking |
| `index.py` | the persisted index, rebuilt incrementally |
| `query.py` | search, formatting, automatic context |
| `citations.py` | enforced-citation rules: parse sources, check citations and quotes |
| `ingest.py`, `status.py` | upload to Markdown; refresh and report |
| `options.py` | option validation and the saved build options |

## Uploaded document route

```text
PDF / DOCX / PPTX / XLSX / HTML / text upload
  -> POST /api/uploads
  -> fileReader extracts exact source text (read-only, bounded)
  -> POST /api/rag/ingest or rag action=ingest
  -> backend/runtime/user_data/knowledge/<name>.md
  -> the indexing pipeline above
  -> backend/runtime/rag_index/index.pkl
```

The agent should call `fileReader` first so it can verify the attachment, then call `rag` with `action=ingest` and
the original upload path. It must not copy the full document into tool arguments. The ingestion backend invokes
`fileReader` again to save the complete extracted text rather than a possibly truncated model-visible preview.
Extraction is capped at 500,000 returned characters and the RAG source reader at 2,000,000 UTF-8 bytes; the API
reports `truncated` when the extraction cap is reached.

## Chunking

### Simple

Runs locally and consumes no LLM tokens. Splits on headings and paragraph boundaries where possible, oversized
paragraphs on sentence boundaries and then hard character limits, with bounded overlap.

### LLM

- Uses the model selected in the Data page. Files under 1,800 characters use simple splitting; files over 60,000
  characters fall back to it to cap cost.
- Other files are grouped into units of at most about 6,000 characters and 48 blocks. The LLM receives numbered
  blocks and must call `submit_chunk_plan` with contiguous block ranges and titles, never source text.
- At most 700 output tokens per response and 10 calls per document. Invalid plans and failures fall back to simple
  splitting without blocking RAG.

## Contextual retrieval

A chunk such as "Revenue grew 3% over the previous quarter" cannot be found by a search for the company or the
quarter it is about. Two kinds of context are added to the text that is **indexed** (BM25 and vectors); the chunk
text the model reads is not changed:

- **Section path** (always, free): the headings above the chunk, `Title > Section > Subsection`.
- **LLM context** (Data page, "Chunk context" = LLM): for each chunk the model reads the document and writes one or
  two sentences (at most about 60 words) that name the subject, the section and anything the chunk depends on but
  does not state. Search results show both as `section:` and `context:`.

Cost and safety:

- The prompt sends the document first and the chunk last, so a provider's prompt cache serves the document part
  for every chunk. A document over 24,000 characters is cut to its outline plus a window around the chunk; the
  window start is rounded to a grid so neighbouring chunks share the same prefix.
- Contexts are cached per chunk (model, path, chunk text), so a refresh only pays for new or changed text.
- Limits: 100 calls per document and 300 per build, 4 in parallel. What does not fit is counted in
  `context_pending`; the next refresh continues. If the first calls all fail (a wrong model or key) it stops instead
  of paying for every chunk.
- A search never calls the LLM for contexts. If a stale index must be rebuilt while searching, chunks without a
  cached context are indexed without one and finished by the next refresh.

## Retrieval

| Retriever | Finds | Weak at |
| --- | --- | --- |
| `bm25` | exact names, identifiers, error codes, rare words (`approvalRequired`, `E_ORD_17`) | paraphrases and synonyms |
| `vector` | similar meaning (with an embedding model) | exact identifiers and numbers |
| `hybrid` (default) | both, fused | adds nothing when one retriever has no signal |

BM25 splits identifiers into their parts and indexes Chinese as characters and pairs. **Vectors** come from:

- **Local** (default): TF-IDF over character 2-5-grams. No model or download. It matches similar-looking words, word
  forms, typos and Chinese substrings, but it is lexical: it does not know that "feline" and "kitten" are related.
- **An embedding model** (Data page, "Vector model"): a `provider:model` whose provider serves the OpenAI-compatible
  `/embeddings` endpoint (OpenAI, OpenRouter, Ollama, DashScope, SiliconFlow, Jina, ...). Add the embedding model id
  to that provider on the Models page, then select it. Vectors are cached per text. If the provider fails, the index
  is built with local vectors and says why (`embedding_fallback`); a refresh retries. If it fails while searching,
  that search uses BM25 only and says so.

### Fusion

Reciprocal rank fusion: each retriever gives a chunk `1 / (60 + rank)` and the sum decides the order. Ranks are
combined, not scores, because BM25 scores and cosine similarities are on different scales. The reported score is
scaled to 0-1 by the best a chunk could get from the retrievers that ran, so a chunk only one of two retrievers
found cannot score above 0.5.

### Reranking

BM25 and vectors score a chunk against the query separately. A reranker reads query and chunk together and judges
how well it answers, which is more precise and slower, so only the top candidates go through it.

- **LLM**: one call scores the top 12 candidates 0-10 (forced tool call `submit_scores`; JSON in the text is
  accepted from models without tools). Candidate text is treated as data: the prompt tells the model not to follow
  instructions found in it.
- The helper LLM calls of RAG (chunk plans, contexts, reranking) switch thinking off (`rag/aux_llm.py`): a score or a
  sentence does not need it, and some providers refuse a forced tool call in thinking mode; if one still does, the call
  is repeated with the tool choice on auto. A small, fast model is a good choice for all three.
- **API**: the provider's `/rerank` endpoint (Cohere, Jina, SiliconFlow, ...), top 30. Needs a rerank model.
- Set in the Data page ("Rerank"). The model can also ask for a rerank on one search (`rerank: true`), which uses
  the LLM reranker when none is set. Automatic context injection never reranks. A failing reranker keeps the fused
  order and says so in a `note:`.

### What the model can choose

`rag action=search` takes `method` (`hybrid`, `bm25`, `vector`), `rerank`, `limit` (1-10) and `sources`
(`knowledge`, `memory`, `skill`). Without them the saved defaults apply. The result lists the retrievers that ran,
the reranker, and for each chunk which retriever found it and at which rank (`foundBy: bm25 #1, vector #4`).

## Enforced citations

Telling the model to cite is not enough: it can invent a source or a quote. After the model gives a final answer, the
graph step `citation_check` (`backend/agent/citation_check.py`, rules in `backend/tools/rag/citations.py`) checks it
in code. Every rag result carries a `cite:` tag, `[path#chunk]`; the checks are:

| Check | Fails when |
| --- | --- |
| source exists | the answer cites a tag that is no chunk of the index |
| source was retrieved | the chunk exists but no search returned it in this conversation (after context compression, an existing chunk is accepted) |
| exact quotes | text in quotation marks or a `>` block right before a citation is not in that chunk (case, spacing, punctuation and width are ignored; `...` splits a quote into parts) |
| cites something | the model itself called rag this turn and got results, but the answer cites nothing and does not end with `Sources: none used` |

A failing answer is **not delivered**. The model receives a `citationError:` message that lists the problems and the
tags it may cite, and writes the answer again, at most twice per turn. If it still fails, the answer is delivered
with a visible `Citation check failed` notice, never silently. Details:

- Context automatically injected by `rag_mode=on` can be cited, but is never required (otherwise every small talk
  would need a source). Only a search the model chose makes citations mandatory.
- The check is deterministic and checks that a source is real and that a quote is verbatim. It cannot tell whether a
  claim is *supported* by the chunk it cites; that is what the reranker and the model's own reading are for.
- The Data page switch "Citation check" (`rag_search.require_citations`, default on) turns it off.
- The UI shows checked citations as small tags; each retry or failure is an event `citation_check` in the activity log.

## Incremental refresh

`backend/runtime/rag_index/` holds `index.pkl` (the index), `index.chunks.json` (per-document chunk plans, LLM chunk
plans and chunk contexts) and `index.embeddings.pkl` (provider vectors of the current chunks). All are ignored by
Git and written atomically.

- Unchanged documents reuse their chunk plan; a changed document is processed again, but unchanged LLM units and
  chunk contexts are reused. Adding or deleting a document touches only that document.
- Changing a build option (chunk mode or model, context on/off or model, vector model) rebuilds the index. Chunk
  plans made with another split mode or chunk model, and contexts made by another model, are not reused.
- `force=true` on `/api/rag/reindex` bypasses every cache.
- An index remembers how it was built. A search that finds it stale rebuilds it the same way, without LLM calls for
  contexts. A refresh (the Data page, ingestion, saving a file) uses the saved settings and finishes unfinished
  work: pending contexts and a failed embedding provider.
- The index file has a format marker; an incompatible or corrupt file is never trusted and is rebuilt.

## Settings

Saved with the app settings and read again for every refresh and every turn:

| Key | Values | Meaning |
| --- | --- | --- |
| `rag_ingestion.split_mode` / `model` | `simple`, `llm` / model | how documents are split |
| `rag_ingestion.contextual` / `context_model` | `off`, `llm` / model | LLM context for each chunk |
| `rag_ingestion.embedding_model` | `""` or `provider:model` | `""` = local vectors |
| `rag_search.default_method` | `hybrid`, `bm25`, `vector` | when the model names no method |
| `rag_search.rerank` / `rerank_model` | `off`, `llm`, `api` / model | reranker applied to every search |
| `rag_search.require_citations` | `true`, `false` | enforce and check citations (above) |

## APIs

- `POST /api/rag/ingest`: upload path to user-knowledge Markdown and index.
- `POST /api/rag/reindex`: refresh with the saved settings (`force` optional; `split_mode` and `chunk_model` may
  still override the saved chunking).
- `PUT /api/data/file`, `POST /api/data/import`, `POST /api/data/file/rename`, `DELETE /api/data/file`: save and
  refresh with the saved settings.
- `GET /api/prompts` lists every prompt, including the context and rerank prompts.

Refresh responses report processed, reused and removed documents, LLM calls for chunking and for contexts (with
estimated input tokens), pending contexts, embedded texts, the embedding in use and any fallback reason.
