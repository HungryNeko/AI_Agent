# Agent Tools

Use this skill when a task may need local tools, conversation history, RAG, MCP, uploads, or user-managed instruction files.

## Core Rules

- Keep the fixed system prompt stable for prompt-cache reuse. Put changing time, RAG hits, references, tool status, and current user text in the per-turn user message.
- Load `data/instruction.md` only into the first system message of a conversation. Do not inject it again every turn.
- Treat `data/skills` as system skills: visible to users, read-only in the UI, and maintained by developers.
- Treat `backend/runtime/user_data/skills`, `backend/runtime/user_data/memory`, and `backend/runtime/user_data/knowledge` as user data. These files are editable in the UI and ignored by git.
- After creating, renaming, importing, or editing memory, skill, or knowledge Markdown files, rebuild the local vector index so RAG can find the latest content.

## Tool Use

- `rag` searches knowledge, memory, and skills. Include source path and source type in results so the model can ask for more detail when needed.
- `history` reads saved JSON conversations under `backend/runtime/conversations`; use it when compressed context is missing older details.
- `mcp` calls configured external services. Uploaded file paths may be passed to MCP tools when a server supports file input.
- `python` is for local analysis and artifact generation. Keep outputs bounded and save generated artifacts under backend runtime paths.
- `fileEditor` reads and writes workspace files according to the configured approval mode.
- `webSearch` is for current or uncertain external facts.

## Attachments

- Uploaded text files should be included as a bounded text preview in the current user message.
- Uploaded images should be sent as OpenAI-compatible `image_url` content blocks when the selected model supports vision input.
- Binary non-image files should be listed with filename, path, MIME type, and size, allowing the model to decide whether MCP or another tool should process them.
