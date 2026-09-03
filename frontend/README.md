# AI Agent Frontend

Minimal React test panel for the backend agent.

## Run

Start the backend API from the repo root:

```powershell
conda run --no-capture-output -n sde python backend\scripts\server.py
```

Start the React dev server:

```powershell
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173.

## Panels

- Chat: stream assistant progress, tool calls, tool errors, approval previews, and final answers.
- Data: import skills and edit `data/skills`, `data/memory`, and `data/knowledge` text files.
- MCP: add/edit configured MCP stdio servers with form fields instead of editing JSON manually.