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

The product UI has seven main areas: Chat, Data, Model Config, Automation, MCP, Tools,
and System Settings. Chinese is the default UI language.

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
- `backend/agent/subagents.py` / `backend/agent/wakeups.py` / `backend/agent/runs.py`: parallel sub-agents,
  the per-conversation turn gate, live event bus, wake-up turns, and the cancel/pause registries. User turns
  and wake turns share one gate per conversation; saved chat state carries a `rev` so a client that missed a
  wake turn cannot overwrite its history.
- `backend/agent/automation_runner.py`: background scheduler started and stopped by the
  FastAPI lifespan.
- `backend/agent/reviewer.py`: optional AI review for high-risk tool requests.
- `backend/prompts/`: fixed system/tool rules and small per-turn dynamic context.
- `backend/tools/request.py`: tool schemas and parsed request types.
- `backend/tools/settings.py`: backend-only tool modes, limits, and paths.
- `backend/tools/executor.py`: central dispatcher and tool-result formatting.

### Frontend

- `frontend/src/main.jsx`: the small React application bootstrap.
- `frontend/src/App.jsx`: the React UI, API calls, SSE parsing, conversation state,
  uploads, mentions, settings, automation, and MCP configuration.
- `frontend/src/ui.jsx`: shared UI primitives: `PopPill` (pill button + popover), `Pill`, `MenuItem`, `StepSlider`,
  `Segmented`, and the resizable panes (`usePane`, `Resizer`, `splitColumns`; widths and collapsed state are kept in
  localStorage).
- `frontend/src/styles.css` imports `styles/tokens.css` (every colour; light/dark/system), `base.css`, `chat.css` and
  `pages.css`. Components never hardcode a colour: use the tokens so both themes stay in step.
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

- `webSearch`: DuckDuckGo (`ddgs`) by default; optional SearXNG or Tavily. The chat setting `web_search_mode` has four steps:
  `off`, `tool` (the webSearch tool; `auto` is the old name), `native` (the provider's own web search/fetch, switched on
  through the provider's `native_web` config: a preset such as `openrouter`/`dashscope` or `{"body": {...}, "tools": [...]}`
  merged into the request, or `endpoints` for providers that host search/fetch as separate authenticated endpoints:
  DeepSeek's `POST /v1/tools/search` and `/v1/tools/fetch` are used automatically for api.deepseek.com and become the
  `nativeWebSearch` and `webFetch` tools, see `tools/nativeWeb.py`) and `both`.
- `rag`: local TF-IDF character n-gram index with cosine-similarity
  search across system and user knowledge, memory, and skills. Its upload-to-Markdown,
  simple/LLM chunking, token limits, and incremental cache route are documented in
  `docs/rag-ingestion.md`.
- `curl`: bounded public HTTP(S) GET only; credentials in URLs, localhost, and private
  IP targets are blocked.
- `python`: calculations, analysis, local reads/network use, plots, and artifacts under
  `backend/runtime/python_runs`; obvious destructive behavior and direct writes outside
  the run directory are blocked. Each call gets a fresh run directory; state that must outlive a call goes in
  the persistent `WORKSPACE` folder (`python_runs/_workspaces/<conversation or sub-agent>`, writable, removed
  when the conversation is deleted).
  Code run by `python` and by custom tools can call back into the backend through `tools/bridge.py` (file-based RPC, no
  port): `call_mcp`, `list_mcp_servers`, `list_mcp_tools`, `ask_llm` (one completion) and `run_agent` (a full agent loop on a
  custom prompt, depth-limited). Time spent waiting on those calls does not count against the python timeout.
- `fileReader`: read-only text extraction for uploaded or project PDF, DOCX, PPTX,
  XLSX, HTML, CSV, Markdown, source code, and other UTF-8 text files.
- `fileEditor` / `fileReader` take per-file read/write locks (`tools/filelock.py`) shared by the main agent and
  all sub-agents; waiting is bounded and the error names the holder. A whole-file `write` is also refused when
  the file changed after that agent last read it (lost-update protection, per agent); `replace`/`insert`/`append`
  work on the current text under the lock.
- `fileEditor`: project-scoped `list`, `read`, `write`, `replace`, `insertAfter`,
  `insertBefore`, and `append`; no delete/move/rename. It protects `.git`, secrets,
  runtime output, and dependency directories.
- `mcp`: lists/calls configured stdio or Streamable HTTP MCP servers. Runtime callers
  cannot supply arbitrary server commands. Uploaded files can be converted to base64
  for supported MCP arguments, and returned images become local artifacts.
- `history`: lists, searches, and reads saved conversation JSON.
- `automation`: creates/updates scheduled reminder, LLM, Python, or MCP work.
- `settings`: lets an enabled automation-capable agent update application settings.
- `question`: requests clarification and ends the turn until the user responds.
- `approval`: Agent-mode manual approval; asks the user to approve or reject one action (with optional text),
  ends the turn, and the reply returns as an `approvalResponse` message. It reuses `question_pending`
  with `kind: "approval"` and emits the `approval_request` SSE event.
- `agent`: Agent-mode sub-agents (`spawn`, `send`, `status`, `wait`, `result`, `cancel`). Each sub-agent is a
  normal `graph.stream_turn` loop in its own thread with its own context, so several really run in parallel
  (limits in `agent/subagents.py`); they cannot spawn agents, ask questions or request approval. `wait=false`
  returns at once; blocking calls cap at 90s. `spawn` requires a stated `taskList` and a `taskId` per task
  (shown in the UI, statuses derived from the agents). Foreground agents (wait=true) die with the turn that
  created them, together with their child processes (`tools/procs.py` tracks python runs); background agents
  (wait=false, or a foreground wait that timed out) are deliberate: they survive the turn and the stop button
  and are recorded in `_background.json`. Deleting the conversation or stopping the server kills everything.
  When one finishes and nobody was waiting for it, `agent/wakeups.py` starts a server-side turn in that conversation
  with a `systemEvent:` message, so the agent is woken instead of polling. Transcripts live under
  `backend/runtime/subagents/<conversation>/`.
- `model`: lists/refreshes provider catalogs or switches the current conversation model,
  without exposing backend credentials.
- `plan`: updates/finalizes a session plan for review; explicit exports use `data/plans`.
- `createTool`: manages user-level Python tools under `data/custom_tools`, exposed as
  `custom__name` functions in Agent mode.

Ask and Plan are read-only: keep `fileReader`, fileEditor list/read, and RAG search
available, but block RAG ingestion and execution tools. Plan additionally allows session
plan updates. Preserve `tools_announced` when rebuilding state: inject the textual
available list on the first turn and after tool request errors, not every turn. Native
OpenAI schemas still accompany every model request.

Tool availability is controlled by `ToolSettings`, not only prompts. A per-turn
`max_tool_rounds` of `0` disables calls, a positive value limits rounds, and `-1` is
unlimited. When a finite budget is exhausted, the backend makes a final no-tools model
request instead of ending with a graph recursion error.

File editor approval modes are `readOnly`, `manual`, `auto`, and `aiReview` (the composer's approval pill). `auto`
involves no model: everything runs directly. In `manual`, a mutating edit is not written: the turn pauses with an
`approval_request` (`source: "fileEditor"`, the exact edits in `pending_edits`, their diffs in `details`). The
pending edits are saved with the conversation state, and when the user approves, `agent/edit_approval.py` applies
the saved edits on the server (never the client's copy) and appends the results to the `approvalResponse` the model
reads, so the model does not redo them. Sub-agents cannot ask and keep the plain `approvalRequired` result.
`aiReview` sends high-risk requests to `agent/reviewer.py` (Python and custom tools, mutating file edits, MCP
`callTool`, mutating automation actions, settings updates, RAG ingest); the reviewer sees the current request, the
conversation's first and most recent earlier requests, the model's step and the call, and a failure denies.

## Security levels

The UI has a three-step slider (`PUT /api/security/level`, stored as `security.level` in the app settings, default L3):

- **L1**: read-only and harmless. Only the tools in `security.L1_TOOLS` exist (file reader limited to the agent's own data and
  uploads, RAG search, history, web search/fetch, model, plan, question, approval). No code runs, so no python, custom tools,
  MCP, automation, settings tool, file editor or sub-agents.
- **L2**: every tool, but anything that runs code (python, custom tools, stdio MCP servers, automation scripts) runs as the
  unprivileged OS user `agentcode` with a scrubbed environment. Needs Linux and a backend started as root (the Docker setup);
  elsewhere it refuses to run code instead of running it unrestricted. Secrets files are owner-only (`security.harden_permissions`),
  and the in-process file editor cannot write code, configuration or deployment files below L3.
- **L3**: everything runs as the backend itself; in Docker the container is the outer boundary.

The level is read when a tool runs (`executor.security_block`, `python.run`, `mcp.run_session`), so lowering it works mid-turn,
and for sub-agents, wake turns, automations and the bridge. Only `app_settings.set_security_level` changes it: the settings tool,
`PUT/PATCH /api/settings` and the client's chat state cannot. `agent/netguard.py` refuses every API request whose TCP connection
belongs to `agentcode`, so agent code cannot call the unauthenticated API (provider config, MCP servers, the level) through the
backend's privileges. Do not promise more than this: without an OS-level sandbox, free-form code is only as contained as its
user's file permissions.

## Models: sub-agents and thinking effort

- `settings.subagent_model` (chat option, UI: agents pill) is the default model of spawned sub-agents, `run_agent` and `ask_llm`;
  a task may name its own `model` in `agent spawn`. There is no automatic escalation: a failing agent can switch by itself.
- `reasoning_effort` / `subagent_reasoning_effort` (`default`, `off`, `low`, `medium`, `high`, `max`) is sent in the provider's own
  format (`llm.apply_reasoning`, picked by the provider's `reasoning` field or its base URL; `default` sends nothing) and is
  retried without the parameter when the provider answers 400.
- DeepSeek thinking mode answers 400 ("reasoning_content must be passed back") when an assistant tool-call message has none, which
  happens after switching model or effort in the middle of a turn. `llm.with_reasoning_content` gives those messages an empty one
  in the outgoing request only (stored history is untouched).

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
- streaming chat, pause/resume, and stop;
- instruction and saved conversations (list/read/rename/delete/compress/branch/plan);
- uploads and allowed artifact serving;
- system/user data listing, import, edit, and rename;
- `GET /api/prompts` (read-only catalog of every prompt and tool schema sent to a model, built by
  `agent/prompt_catalog.py`; shown under Data > Prompts). A new prompt string belongs in a named constant that the
  catalog imports, so the page always shows the text the code really uses;
- RAG reindex;
- automation definitions and run history;
- MCP list/test/upsert/import/delete.
- provider model refresh and user-created tool management;
- `GET /api/conversations/{id}/live` (SSE for sub-agent updates and wake turns) and `/subagents` endpoints.

When adding an API capability, update both the Pydantic request model/server route and
the matching frontend call. Preserve SSE event types already consumed by the UI:
`assistant_progress`, `tool_call`, `approval_required`, `approval_request`, `subagent_update`, `wake_started`, `wake_finished`, `ai_review`, `error`, `stopped`,
`settings_changed`, and final `assistant`.

## Change guidelines

- Read `data/skills/agent-backend/SKILL.md` before substantial graph/backend changes.
- Read `data/skills/agent-tools/SKILL.md` before changing tools, RAG, MCP, attachments,
  history, or user-managed data.
- Prompts are English only. How to use one tool belongs in that tool's description (`tools/request.py`); the tool
  rules in `prompts/tools.py` only cover choosing and combining tools, so nothing is written twice. The system prompt
  is fixed text first and the user's instruction last.
- Keep the fixed first system prompt stable for prompt-cache reuse. Put time, current
  tool availability, references, RAG hits, attachments, and the user message in the
  per-turn user content.
- Prompt cache: providers match the exact start of a request (tools, system prompt, then messages in order). Never
  rewrite earlier messages, never drop or reorder `tools` within a conversation (the tool budget uses
  `tool_choice: "none"` instead), and never inject the same text into every turn: results, the plan and context are
  injected once and then live in the history (`injected_results`, `plan_sent`). `backend/tests/test_prompt_cache.py`
  replays turns and checks the prefix. The final `assistant` event carries `usage` (prompt, cached, completion) which the
  UI shows as the cache-hit line.
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
- The RAG index uses local TF-IDF character n-grams without a model download or neural
  inference step. The current simple-mode local index has 5
  documents and 7 chunks; an unchanged refresh reused all 5 documents with zero LLM
  calls. LLM chunking and ingestion behavior is covered in `docs/rag-ingestion.md`.
- `npm --prefix frontend run build` passes with Vite 8.2.2.
