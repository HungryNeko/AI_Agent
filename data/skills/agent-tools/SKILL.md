# Agent Tools

Use this skill when a task may need local tools, conversation history, RAG, MCP, uploads, or user-managed instruction files.

## Core Rules

- Keep the fixed system prompt stable for prompt-cache reuse. Put changing time, RAG hits, references, tool status, and current user text in the per-turn user message.
- Load `data/instruction.md` only into the first system message of a conversation. Do not inject it again every turn.
- Treat `data/skills` as system skills: visible to users, read-only in the UI, and maintained by developers.
- Treat `backend/runtime/user_data/skills`, `backend/runtime/user_data/memory`, and `backend/runtime/user_data/knowledge` as user data. These files are editable in the UI and ignored by git.
- After creating, renaming, importing, or editing memory, skill, or knowledge Markdown files, rebuild the local vector index so RAG can find the latest content.

## Tool Use

- `rag` action=search searches knowledge, memory, and skills using the local TF-IDF character n-gram index. For an uploaded document that the user wants in the knowledge base, inspect it with `fileReader`, then use `rag` action=ingest with the upload path; do not repeat the extracted source in tool arguments. Use simple splitting unless the user explicitly requests LLM splitting. Include source path and source type in search results so the model can ask for more detail when needed.
- `history` reads saved JSON conversations under `backend/runtime/conversations`; use it when compressed context is missing older details.
- `mcp` calls configured external services. Uploaded file paths may be passed to MCP tools when a server supports file input. If an MCP tool needs base64 file bytes, pass `content_base64_from_file`, `body_base64_from_file`, `image_base64_from_file`, or `file_base64_from_file` with a `backend/runtime/uploads/...` path or `/api/uploads/...` URL; the backend injects exact base64 bytes at call time.
- `python` is for local analysis and artifact generation. Keep outputs bounded and save generated artifacts under backend runtime paths.
- `fileReader` extracts bounded text from supported uploaded or project documents without modifying them. Use it for PDF, DOCX, PPTX, XLSX, HTML, CSV, Markdown, and source files.
- `fileEditor` reads and writes workspace files according to the configured approval mode.
- `webSearch` is for current or uncertain external facts.

## parameterSave / Python References

- Wrap large queries with `parameterSave` before executing them so raw data stays out of model context. An empty `list` means this conversation has no saved values, not that the tool is unavailable.
- Save the whole result when its structure is unknown: `{"action":"call","call":{"tool":"mcp","arguments":{"action":"callTool","server":"configuredServerName","tool":"query","arguments":{}}},"save":[{"name":"query_result"}]}`. Only `name` is required; the result contains an immutable `param:...` ref and structural metadata.
- Optional `save.path` extracts a child BEFORE saving. The root is the decoded application result, not the MCP transport envelope: `structuredContent` is unwrapped, or a single JSON text content block is parsed. For an application object with keys `content`, `data`, `success`, `error`, use `data.rows`, not `structuredContent.data.rows`. Omitting `path`, or using empty/`$`, saves the whole result. Python wrappers require a variable path such as `rows`.
- Optional `save.type` converts the selected value AFTER extraction: `auto` (default) preserves type, `json` produces JSON-compatible structures, `ndarray` creates a NumPy array, and `table`/`dataframe` creates a Pandas DataFrame. Changing `type` cannot repair a wrong path. Unknown fields are rejected, not silently ignored.
- `{"action":"inspect","ref":"param:..."}` returns structure only. Add `path="data.rows"` to describe that child's inferred type and columns. Inspection never converts stored data: `type=table` can describe a list of dicts, while `pythonType=list` reports its real Python type. `limit=1..20` explicitly requests a bounded preview; leave `limit=0` to avoid revealing row values.
- Python helpers are available without imports: `obj = load_parameter(ref)` loads the entire stored value; `rows = load_parameter(ref, path="data.rows")` selects a child. They preserve actual stored types. `save_parameter("summary", value, data_type="auto")` returns a new ref; only successful runs publish it to conversation storage. Use `pd.DataFrame(rows)` explicitly if needed.
- MCP and custom-tool arguments can use `{"$ref":"param:...","path":"data.rows"}`; the backend resolves the value without printing it. Dot paths support dictionary keys and numeric list indexes (`data.rows.0`), not brackets or arbitrary JSONPath expressions. Path errors show available keys/structure, never row values.
- Values belong to one conversation and survive restart/compression; `list` recovers refs, `delete` removes them when requested, and branches keep independent copies. Original tool availability and approvals still apply. Do not print entire datasets just to pass them between tools.

## Attachments

- Uploaded text files should be included as a bounded text preview in the current user message.
- Uploaded images should be sent as OpenAI-compatible `image_url` content blocks when the selected model supports vision input.
- Binary non-image files should be listed with filename, path, MIME type, and size, allowing the model to decide whether MCP or another tool should process them.
