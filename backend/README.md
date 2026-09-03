# AI Agent Backend

This backend is intentionally small while learning LangGraph.

## Structure

```text
backend/
  agent/
    config.py      # load .env and data/api_configs.json
    llm.py         # call OpenAI-compatible /chat/completions
    graph.py       # LangGraph state flow and tool loop
    cli.py         # plain argparse CLI and chat loop
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
    memory.py      # file-backed memory helpers
    skills.py      # file-backed skill helpers
    rag.py         # search data/knowledge, data/memory, data/skills
    mcp.py         # configured MCP stdio client
  scripts/
    simple_chat.py # run the CLI with python directly
```


## Data Files

```text
data/
  api_configs.json
  knowledge/       # local docs and reference notes for rag
  memory/          # durable project memory files
  skills/          # skill folders, each with SKILL.md
  mcp/servers.json # configured MCP servers
```

`rag` searches `data/knowledge`, `data/memory`, and `data/skills`. When
`--rag-mode on` is enabled, matching local context is automatically injected.
When `--rag-mode auto` is enabled, the model may call `rag` itself.

Memory and skills are ordinary project files. To update them, let the model use
`fileEditor` so the same approval policy applies:

```powershell
python backend\scripts\simple_chat.py --rag-mode auto --file-editor-mode auto --file-editor-approval manual "Search project memory, then propose a memory update about today's decision."
```

MCP servers are configured only in `data/mcp/servers.json`. The model can list
servers, list tools, or call a configured tool, but it cannot provide a server
command at runtime:

```powershell
python backend\scripts\simple_chat.py --mcp-mode auto "Use mcp to list configured servers."
```
## API Server

For the React frontend, start the FastAPI server from the repo root:

```powershell
conda run --no-capture-output -n sde python backend\scripts\server.py
```

The server exposes chat streaming, data file editing, skill import, and MCP configuration endpoints under `/api/*`.

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
changed endpoint, changed parameters, or different source makes sense; the
backend only blocks the exact same tool input after two attempts in one user
turn. If `max_tool_rounds` is reached, the backend makes one final model call
without tools so the user gets a final explanation instead of a hard graph error.

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

File editor mode lets the model inspect and edit project files with stable text
anchors. It supports `list`, `read`, `write`, `replace`, `insertAfter`,
`insertBefore`, and `append`. It does not expose delete/move/rename operations,
and it blocks paths outside the project root plus protected paths such as `.git`,
`.env`, `backend/runtime`, and `node_modules`.

Write permission is controlled separately with `--file-editor-approval`:
- `manual` is the default: validate the edit and return a diff preview, but do not write.
- `auto` applies allowed writes immediately, like approving the agent to edit.
- `readOnly` allows `list`/`read` but never applies writes.

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

Normal turns keep dynamic prompt text small:

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

The first system prompt includes the fixed rules at the top for better prompt
cache reuse. Later turns reuse conversation history and only add small dynamic
context lines.

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
