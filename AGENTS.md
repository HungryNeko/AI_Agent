# AI Agent Repository Guide

This file is development context for coding agents working on this repository. It is
not runtime memory for the AI Agent application. Application memory lives under
`data/memory` and `backend/runtime/user_data/memory`.

## Project purpose

This is a learning-oriented but fairly complete local AI agent application. It combines:

- a Python 3.11+ FastAPI backend;
- a LangGraph tool-calling conversation loop;
- an OpenAI-compatible Chat Completions client;
- a React/Vite frontend;
- persistent conversations, uploads, local RAG, MCP connections, configurable tools,
  scheduled automations, and project-scoped file editing.

The product UI has six main areas: Chat, Data, Model Config, Automation, MCP, and
System Settings. Chinese is the default UI language.

## High-level request flow

```text
React UI
  -> POST /api/chat/stream (SSE)
  -> agent.server.build_chat_state()
  -> agent.graph.stream_turn()
  -> first_state
  -> conversation_begin
  -> assistant_step
       -> conversation_end                 when the model gives a final answer
       -> tool_call -> assistant_step      when the model emits tool_calls
       -> tool_error -> assistant_step     when parsing/execution needs correction
  -> save conversation JSON under backend/runtime/conversations
```

The graph uses native OpenAI-compatible `tools` and `tool_calls`. Do not replace this
with model-written JSON embedded in normal assistant text. The graph has no persistent
LangGraph checkpointer: the frontend carries the public chat state between turns, and
the server separately saves full conversation JSON.

## Important paths

### Backend

- `backend/agent/graph.py`: canonical agent state, LangGraph nodes, routing, streaming
  event conversion, tool budgets, and approval hooks.
- `backend/agent/server.py`: FastAPI app and all `/api/*` endpoints.
- `backend/agent/llm.py`: one OpenAI-compatible `/chat/completions` request.
- `backend/agent/config.py`: model/provider config plus `.env` loading.
- `backend/agent/session_store.py`: one JSON file per saved conversation and context
  compression.
- `backend/agent/automation_runner.py`: background scheduler started and stopped by the
  FastAPI lifespan.
- `backend/agent/reviewer.py`: optional AI review for high-risk tool requests.
- `backend/prompts/`: fixed system/tool rules and small per-turn dynamic context.
- `backend/tools/request.py`: tool schemas and parsed request types.
- `backend/tools/settings.py`: backend-only tool modes, limits, and paths.
- `backend/tools/executor.py`: central dispatcher and tool-result formatting.

### Frontend

- `frontend/src/main.jsx`: the entire React UI, API calls, SSE parsing, conversation
  state, uploads, mentions, settings, automation, and MCP configuration.
- `frontend/src/styles.css`: all UI styling.
- `frontend/vite.config.js`: Vite runs on `127.0.0.1:5173`; `/api` proxies to
  `http://127.0.0.1:8012` unless `VITE_BACKEND_PROXY` overrides it.

### Versioned data and local runtime data

- `data/api_configs.example.json`, `data/settings.example.json`,
  `data/instruction.example.md`, and `data/mcp/servers.example.json`: tracked safe
  templates. The corresponding non-example files and all `*.local.json` overrides are
  local-only and ignored by Git.
- `data/knowledge`, `data/memory`, `data/skills`: system RAG source locations. Local
  memory Markdown is ignored by Git; developer-maintained skills use
  `data/skills/<name>/SKILL.md`.
- `backend/runtime/user_data/{knowledge,memory,skills}`: UI-editable user RAG data,
  ignored by Git.
- `backend/runtime/conversations`: saved full conversation/event JSON.
- `backend/runtime/{uploads,python_runs,mcp_artifacts}`: uploaded/generated artifacts.
- `backend/runtime/logs`: append-only redacted JSONL debug logs.
- `backend/runtime/{automations,automation_runs}`: scheduled task definitions and run
  records.

Tracked system RAG files are read-only through the Data UI. User data under
`backend/runtime/user_data` is writable. After memory, knowledge, or skills change,
rebuild the RAG index so the new content is searchable.

## Built-in tools

- `webSearch`: DuckDuckGo (`ddgs`) by default; optional SearXNG or Tavily.
- `rag`: local `intfloat/multilingual-e5-small` dense-vector index with cosine-similarity
  search across system and user knowledge, memory, and skills. Its upload-to-Markdown,
  simple/LLM chunking, token limits, and incremental cache route are documented in
  `docs/rag-ingestion.md`.
- `curl`: bounded public HTTP(S) GET only; credentials in URLs, localhost, and private
  IP targets are blocked.
- `python`: calculations, analysis, local reads/network use, plots, and artifacts under
  `backend/runtime/python_runs`; obvious destructive behavior and direct writes outside
  the run directory are blocked.
- `fileReader`: read-only text extraction for uploaded or project PDF, DOCX, PPTX,
  XLSX, HTML, CSV, Markdown, source code, and other UTF-8 text files.
- `fileEditor`: project-scoped `list`, `read`, `write`, `replace`, `insertAfter`,
  `insertBefore`, and `append`; no delete/move/rename. It protects `.git`, secrets,
  runtime output, and dependency directories.
- `mcp`: lists/calls configured stdio or Streamable HTTP MCP servers. Runtime callers
  cannot supply arbitrary server commands. Uploaded files can be converted to base64
  for supported MCP arguments, and returned images become local artifacts.
- `history`: lists, searches, and reads saved conversation JSON.
- `automation`: creates/updates scheduled reminder, LLM, Python, or MCP work.
- `settings`: lets an enabled automation-capable agent update application settings.

Tool availability is controlled by `ToolSettings`, not only prompts. A per-turn
`max_tool_rounds` of `0` disables calls, a positive value limits rounds, and `-1` is
unlimited. When a finite budget is exhausted, the backend makes a final no-tools model
request instead of ending with a graph recursion error.

File editor approval modes are `readOnly`, `manual`, `auto`, and `aiReview`. In
`manual`, a mutating edit returns a diff preview with `approvalRequired` and does not
write. `aiReview` sends high-risk requests to `agent/reviewer.py`; high-risk requests
include Python, mutating file edits, MCP `callTool`, mutating automation actions, and
settings updates.

## Configuration and secrets

Use Python 3.11 or newer; the code uses `datetime.UTC` and `typing.NotRequired`.

1. Copy `backend/.env.example` to `backend/.env`.
2. Set `DEEPSEEK_API_KEY` or the environment variable named by the selected provider.
3. Keep secrets and private configuration out of tracked files. Local state is loaded
   from these ignored files:
   - `data/api_configs.json`
   - `data/api_configs.local.json`
   - `data/settings.json`
   - `data/settings.local.json`
   - `data/instruction.md`
   - `data/mcp/servers.json`
   - `data/mcp/servers.local.json`

The default provider/model is DeepSeek `deepseek-chat`. OpenAI, OpenRouter, Ollama, or
other OpenAI-compatible providers can be added through model configuration.

## Development commands

From the repository root:

```bash
# Install backend and test dependencies into a Python 3.11+ environment
python -m pip install -e './backend[dev]'

# Backend on the port expected by the default Vite proxy
AI_AGENT_BACKEND_PORT=8012 python backend/scripts/server.py

# CLI agent
python backend/scripts/simple_chat.py --loop

# Backend tests
python -m pytest backend/tests

# Frontend
npm --prefix frontend ci
npm --prefix frontend run dev
npm --prefix frontend run build
```

On Windows, `./start_dev.ps1` starts the backend on port 8012 and the frontend on
5173. Running `backend/scripts/server.py` without `AI_AGENT_BACKEND_PORT` uses port
8010, which does not match the default Vite proxy; set the environment variable or
override `VITE_BACKEND_PROXY`.

On macOS, `./start_dev.sh` starts both services in the current terminal. It shuts down
both process groups when either service exits, the user presses `Ctrl+C`, or the shell
receives a terminal-close signal, so Python/npm child processes are not left running.
It prefers `.venv/bin/python` and supports an explicit `AI_AGENT_PYTHON` override.

## API groups

`backend/agent/server.py` is the source of truth. Major endpoint groups are:

- health/version/models/config/settings;
- streaming chat and stop;
- instruction and saved conversations (list/read/rename/delete/compress);
- uploads and allowed artifact serving;
- system/user data listing, import, edit, and rename;
- RAG reindex;
- automation definitions and run history;
- MCP list/test/upsert/import/delete.

When adding an API capability, update both the Pydantic request model/server route and
the matching frontend call. Preserve SSE event types already consumed by the UI:
`assistant_progress`, `tool_call`, `approval_required`, `ai_review`, `error`, `stopped`,
`settings_changed`, and final `assistant`.

## Change guidelines

- Read `data/skills/agent-backend/SKILL.md` before substantial graph/backend changes.
- Read `data/skills/agent-tools/SKILL.md` before changing tools, RAG, MCP, attachments,
  history, or user-managed data.
- Keep the fixed first system prompt stable for prompt-cache reuse. Put time, current
  tool availability, references, RAG hits, attachments, and the user message in the
  per-turn user content.
- Preserve bounded outputs, project-root checks, secret redaction, SSRF protection, and
  file approval semantics when changing tools.
- Treat `backend/runtime` and ignored local config as user-owned local state. Do not
  delete or commit them.
- Preserve unrelated dirty-worktree files. Inspect `git status` before editing.
- Add or update focused tests under `backend/tests` for backend behavior. The frontend
  currently has no automated test suite, so at minimum run its production build.

## Verification snapshot (2026-09-15)

- `python -m compileall -q backend/agent backend/tools backend/prompts backend/scripts`
  passes in the current checkout.
- File reader, graph, tool, prompt, and LLM tests pass: `92 passed`.
- The full backend suite reaches `168 passed, 1 failed` when dependencies are supplied
  from a temporary Python 3.11 test environment. The remaining pre-existing MCP server
  assertion expects an omitted `headers` key, while the endpoint returns `headers: {}`;
  it is unrelated to `fileReader`.
- The RAG index uses `intfloat/multilingual-e5-small`; real-model verification returns
  normalized 384-dimensional vectors. The current simple-mode local index has 5
  documents and 7 chunks; an unchanged refresh reused all 5 documents with zero LLM
  calls. LLM chunking and ingestion behavior is covered in `docs/rag-ingestion.md`.
- `npm --prefix frontend run build` passes with Vite 8.2.2.
