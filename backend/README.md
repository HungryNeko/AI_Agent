# AI Agent Backend

This backend is intentionally small while learning LangGraph.

## Structure

```text
backend/
  agent/
    config.py      # load local model config with a public example fallback
    llm.py         # call OpenAI-compatible /chat/completions
    graph.py       # LangGraph state flow and tool loop
    cli.py         # plain argparse CLI and chat loop
    server.py      # FastAPI routes and SSE delivery to React
    session_store.py # saved conversations, branches, and context compression
    app_settings.py  # persistent defaults
  prompts/
    context.py     # compressed context summary prompt
    system.py      # build the system prompt
    tools.py       # build the available-tool prompt section
  tools/
    settings.py    # system-side tool config
    request.py     # OpenAI tool schemas and tool_call parsing
    executor.py    # execute tool calls and format tool results
    WebSearch.py   # web search
    curl.py        # direct public HTTP API GET tool
    python.py      # Python analysis/plotting/local scripting tool
    fileEditor.py  # project-scoped anchor-based file editor
    fileReader.py  # read-only PDF/Office/text extraction
    memory.py      # file-backed memory helpers
    skills.py      # file-backed skill helpers
    rag.py         # search data/knowledge, data/memory, data/skills
    mcp.py         # configured MCP stdio / Streamable HTTP / SSE client
    plan.py        # session-plan validation and Markdown export
    question.py    # ask the user a question and pause the turn
    models.py      # provider model catalog (list/refresh/switch)
    createTool.py  # create and run user-level custom Python tools
    automation.py  # scheduled automation runner
    history.py     # read saved conversation history
    appSettings.py # persistent app settings storage
  scripts/
    simple_chat.py # run the CLI with python directly
```


## Data Files

```text
data/
  api_configs.example.json # tracked safe model/provider defaults
  api_configs.json         # optional local base config, ignored by Git
  api_configs.local.json   # UI-managed local override, ignored by Git
  settings.example.json    # tracked safe application defaults
  instruction.example.md   # tracked safe instruction template
  knowledge/       # local docs and reference notes for rag
  memory/          # local durable project memory, ignored except README
  skills/          # skill folders, each with SKILL.md
  mcp/servers.example.json # tracked empty MCP template
  mcp/servers.local.json   # configured MCP servers, ignored by Git
  plans/           # saved plan Markdown exports
  custom_tools/    # user-created custom Python tools (tool.json + tool.py)
```

`rag` searches `data/knowledge`, `data/memory`, and `data/skills`. When
`--rag-mode on` is enabled, matching local context is automatically injected.
When `--rag-mode auto` is enabled, the model may call `rag` itself.

Memory and skills are ordinary project files. To update them, let the model use
`fileEditor` so the same approval policy applies:

```powershell
python backend\scripts\simple_chat.py --rag-mode auto --file-editor-mode auto --file-editor-approval manual "Search project memory, then propose a memory update about today's decision."
```

MCP servers are stored locally in `data/mcp/servers.local.json`. The model can list
servers, list tools, or call a configured tool, but it cannot provide a server
command at runtime:

```powershell
python backend\scripts\simple_chat.py --mcp-mode auto "Use mcp to list configured servers."
```
## API Server

For the React frontend, start the FastAPI server from the repo root:

```powershell
$env:AI_AGENT_BACKEND_PORT = "8012"
conda run --no-capture-output -n sde python backend\scripts\server.py
```

The server exposes chat streaming, data file editing, skill import, and MCP configuration endpoints under `/api/*`.

Without `AI_AGENT_BACKEND_PORT`, the server uses `8010`, while Vite expects `8012`.
On Windows, `powershell -ExecutionPolicy Bypass -File .\start_dev.ps1` starts both.

On macOS, start the backend and frontend together from the repo root:

```bash
./start_dev.sh
```

Press `Ctrl+C` or close the terminal to stop both services and their child processes.
The script prefers `.venv/bin/python`, or you can set `AI_AGENT_PYTHON` to another
Python 3.11+ interpreter with the backend dependencies installed.

## Run

Put this in `backend/.env`:

```env
DEEPSEEK_API_KEY=your_key
# Optional only when using --web-search-provider tavily
TAVILY_API_KEY=your_tavily_key
# Optional only when using --web-search-provider searxng
SEARXNG_QUERY_URL=http://localhost:8080/search?q=<query>&format=json
```

Then run from the repo root:

```powershell
python backend\scripts\simple_chat.py "hello"
```

Loop mode:

```powershell
python backend\scripts\simple_chat.py --loop
```

Web search mode lets the model decide when to call `webSearch`. The default
provider is DuckDuckGo through `ddgs`, so no search API key is required:

```powershell
python backend\scripts\simple_chat.py --loop --web-search-mode auto
```

For weather through web search, include a location in the question, for example:

```powershell
python backend\scripts\simple_chat.py --web-search-mode auto "What is the weather in Los Angeles today?"
```

Direct public API calls use the `curl` tool. This is useful when a task needs a
public JSON/text API. If the endpoint or parameters are uncertain, let the model
search the official documentation first, then call `curl` with the direct URL:

```powershell
python backend\scripts\simple_chat.py --web-search-mode auto --curl-mode auto "Search for the official current weather API documentation, then use curl to fetch current weather for New York City. Show the endpoint you used and summarize the result."
```

If Python `httpx` times out, the tool falls back to system `curl.exe`/`curl`
with a 20 second default timeout. If a tool execution fails, the raw error is
sent to the model. The model may decide whether a retry, official-doc lookup,
changed endpoint, changed parameters, or different source makes sense. The
frontend-selected `max_tool_rounds` controls the loop: `0` disables tools for
the turn, and `-1` allows unlimited rounds. If a non-negative limit is reached,
the backend makes one final model call without tools so the user gets a final
explanation instead of a hard graph error.

Python mode lets the model run Python for math, statistics, data analysis,
plotting, and local scripting. The Python current working directory is the
artifact directory, so code should save generated files with relative names such
as `chart.png`. Generated files are written under
`backend/runtime/python_runs/...` and returned as artifact paths:

```powershell
python backend\scripts\simple_chat.py --python-mode auto "Use python to calculate the mean of [2, 4, 6] and plot y=x^2 for x=1..5."
```

The Python tool allows normal imports, local file reads, network access, and
standard Python introspection. It still blocks obvious destructive operations
and direct writes outside the artifact directory; use `fileEditor` for project
file changes that should follow the editor approval policy.

File reader mode extracts bounded text from uploaded or project files without
modifying them. It supports PDF, DOCX, PPTX, XLSX, HTML, CSV, Markdown, source
code, and other UTF-8 text formats. PDF/PowerPoint page ranges and a single
Excel sheet can be selected for large documents. Legacy `.doc`, `.ppt`, and
`.xls` files should be converted to their modern formats first; scanned PDFs
need OCR before they contain extractable text.

File editor mode lets the model inspect and edit project files with stable text
anchors. It supports `list`, `read`, `write`, `replace`, `insertAfter`,
`insertBefore`, and `append`. It does not expose delete/move/rename operations,
and it blocks paths outside the project root plus protected paths such as `.git`,
`.env`, `backend/runtime`, and `node_modules`.

Write permission is controlled separately with `--file-editor-approval`:
- `manual` validates the edit and returns a diff preview, but does not write.
- `auto` is the default and applies allowed writes immediately.
- `readOnly` allows `list`/`read` but never applies writes.

The frontend/backend also support `aiReview` for separate review of high-risk calls.
Approval policy is independent of Ask/Plan/Agent tool availability.

Read a file:

```powershell
python backend\scripts\simple_chat.py --file-editor-mode auto "Use fileEditor to read backend/README.md lines 1 to 8 and summarize what this backend contains."
```

Preview a write without applying it:

```powershell
python backend\scripts\simple_chat.py --file-editor-mode auto --file-editor-approval manual "Use fileEditor to create backend/tmp_demo.txt with content hello."
```

Apply allowed writes immediately:

```powershell
python backend\scripts\simple_chat.py --file-editor-mode auto --file-editor-approval auto "Use fileEditor to create backend/tmp_demo.txt with content hello."
```

For code changes, prefer `replace` with exact unique `oldText`, or
`insertAfter`/`insertBefore` with an exact unique `anchor`. Use line numbers for
reading chunks, not for editing, unless there is no stable text anchor. If the
tool returns `approvalRequired`, the edit was only previewed and has not been applied.

In loop mode, you can enable both model-decided web search and direct API GET:

```powershell
python backend\scripts\simple_chat.py --loop --web-search-mode auto --curl-mode auto --python-mode auto --file-editor-mode auto --file-editor-approval manual
```

The CLI uses LangGraph streaming. For complex tasks, the model can emit brief
progress text before a tool call and again between tool calls. The backend also
prints a stable tool event before execution, for example:

```text
AI> I will search for the forecast source first.
[tool] webSearch: Los Angeles weather
AI> I found a direct API endpoint. I will fetch it now.
[tool] curl: https://api.example.com/v1/resource?...
AI> Summary: ...
```

The reusable backend entrypoint for a future frontend is `agent.graph.stream_turn()`.
It yields `assistant_progress`, `tool_call`, `approval_required`, `error`, and final `assistant` events.

Optional providers:

```powershell
python backend\scripts\simple_chat.py --loop --web-search-mode auto --web-search-provider searxng
python backend\scripts\simple_chat.py --loop --web-search-mode auto --web-search-provider tavily
```

Only the first turn announces the textual `available` list. Later turns keep dynamic
context small; a tool request error supplies the allowed list again:

```text
available: ["webSearch", "rag", "curl", "python", "fileEditor", "mcp"]
conversationSummary: "..."
webSearchResult: "..."
ragResult: "..."
```

Actual tool calling uses OpenAI-compatible `tools` and `tool_calls`, not
model-written JSON in normal text.

Backend-only settings such as max results, timeout, and similarity threshold
stay in `backend/tools/settings.py`.

## Debug Logs

The backend writes append-only JSONL debug logs under
`backend/runtime/logs/agent-YYYYMMDD.jsonl`. These logs include chat stream
starts, LLM requests and responses, tool calls and results, MCP calls, and
exception tracebacks. Sensitive fields such as authorization headers, API keys,
tokens, secrets, and large base64/body payloads are redacted or omitted.

Set `AI_AGENT_LOG_DIR` to write logs somewhere else:

```powershell
$env:AI_AGENT_LOG_DIR = "D:\tmp\ai-agent-logs"
```

The first system prompt includes fixed rules plus `data/instruction.md`.
Later turns refresh current time, mode, RAG context, and optional
compressed summary, but they do not re-inject the instruction file or textual tool
list. Tool availability is enforced by the current OpenAI schemas and request parser.
Conversation JSON is saved under
`backend/runtime/conversations`; compression shortens active model context but
does not delete the full saved history, which the `history` tool can read.
RAG builds a local TF-IDF character n-gram index at
`backend/runtime/rag_index/index.pkl` and searches it with cosine similarity. It has no
model download or neural inference step, so searches remain fast on CPU-only servers.
The `/api/rag/reindex` endpoint
rebuilds that vector index after instruction, memory, skill, or knowledge files
change. The Data page supports local simple splitting and bounded LLM-assisted
splitting with a dedicated configured model. Upload ingestion uses `fileReader`, saves
the extracted source as user-knowledge Markdown, and reuses per-document and per-unit
chunk caches so small additions or deletions do not re-run unrelated LLM work. See
`../docs/rag-ingestion.md` for the complete route and cost limits.

Current chain:

```text
simple_chat.py -> agent/cli.py -> agent/graph.py -> agent/llm.py -> model
```

Core graph shape:

```text
conversation_begin -> assistant_step
assistant_step -> conversation_end   # final answer
assistant_step -> tool_call          # content + tool_calls, or tool_calls only
tool_call -> assistant_step          # model reads tool results and decides next step
tool_call -> tool_error              # bad tool request or blocked tool
tool_error -> assistant_step
```

`response` stores only the final answer. Tool-call prefaces and between-tool
notes are streamed as `assistant_progress` events.

## Conversation Modes and User Tools

The UI selects `ask`, `plan`, or `agent`. Ask and Plan expose `fileReader`, read-only
`fileEditor`, and retrieval tools, but disallow Python, MCP calls, custom tools, and RAG
ingestion. Plan additionally exposes `plan` to update/finalize a session draft; explicit
Save copy exports it to `data/plans`. Approve and implement switches the UI to Agent
and sends an implementation request. A plan approval does not override file permissions.

`question` returns a waiting event and ends the turn. The next user response resumes the
conversation. Question strength is `off`, `light`, or `heavy`; the UI always offers free
text, refusal, redirection, and minimize/restore. The CLI shows the question and accepts
the response as the next normal input.

`model` lists or refreshes configured model catalogs and switches the current conversation
model. Provider keys remain backend-only. Refresh requires an OpenAI-compatible `/models`
endpoint; manual configuration is available when a provider does not support it.

`createTool` saves a user tool as `data/custom_tools/<name>/tool.py` plus `tool.json`.
Code defines `run(arguments)` and uses an object JSON Schema. Saved functions are exposed
as `custom__name`; built-in tools keep their names and MCP calls use `mcp`. These local
execution helpers are not a hardened sandbox, so use trusted code.

Branches summarize a selected history prefix into a new conversation. Pause/stop act
at event boundaries and do not interrupt or roll back a tool already running. Queue
and insertion are UI workflows; queued inputs are not durable across page reloads.

## Verification

From the repository root, using `sde`:

```powershell
conda run --no-capture-output -n sde python -m pytest backend/tests
npm.cmd run build --prefix frontend
git diff --check
```
