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
    WebSearch.py   # web search placeholder
    rag.py         # RAG placeholder
  scripts/
    simple_chat.py # run the CLI with python directly
```

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

Optional providers:

```powershell
python backend\scripts\simple_chat.py --loop --web-search-mode auto --web-search-provider searxng
python backend\scripts\simple_chat.py --loop --web-search-mode auto --web-search-provider tavily
```

Normal turns keep dynamic prompt text small:

```text
available: ["webSearch", "rag"]
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
