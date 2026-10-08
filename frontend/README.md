# AI Agent Frontend

React/Vite interface for chat, data and RAG management, model configuration,
automation, MCP, reusable tools, and system settings.

## Run

The recommended command starts the frontend and backend together from the repository
root on Windows (using Conda `sde`):

```powershell
powershell -ExecutionPolicy Bypass -File .\start_dev.ps1
```

On macOS/Linux:

```bash
./start_dev.sh
```

To run only the frontend:

```bash
npm ci --prefix frontend
npm --prefix frontend run dev
```

Open <http://127.0.0.1:5173>. The Vite development server proxies `/api` to the
backend at `http://127.0.0.1:8012` by default.

For project architecture, backend setup, and verification commands, see the root
[`README.md`](../README.md).

## Source layout

- `src/main.jsx`: React bootstrap only.
- `src/App.jsx`: application state, API integration, and feature views.
- `src/styles.css`: shared visual system and responsive layout.

## Controls

- Chat: compact composer toolbar for Ask/Plan/Agent, model selection, uploads, approval,
  queue/insert, pause/resume, and stop. Completed operation details collapse while the
  final Markdown answer remains visible. Responses support images, tables, and formulas.
- Plan: review the session draft, request revisions, reject, save a Markdown copy, or
  approve and implement in Agent mode. Switching modes alone does not start execution.
- Questions: single/multiple choice or free text, with decline, redirection, and a
  minimizable dialog. Settings select skipped/light/heavy clarification and developer mode.
- Data: import/edit user knowledge, memory, skills, and instructions; system sources are
  read-only in this UI. Rebuild RAG after changes; document ingestion supports simple or
  bounded LLM splitting.
- Models: configure provider URLs and credentials, refresh supported model lists, and
  choose a conversation model. Credentials are not available to model tools.
- MCP: enter a URL and header key/value pairs for Streamable HTTP, save, edit, or test
  the connection. Only stdio transport requires a local command.
- Tools: create/edit reusable Python tools with `run(arguments)` and an object schema.
- Automation/System: manage scheduled work, saved defaults, appearance, and permissions.

Pause/stop take effect between stream events, not inside a running request/tool.
Branches reconstruct summarized context, rather than replaying an exact checkpoint.
Queued inputs remain only in the current page session.

Build check: `npm --prefix frontend run build`. There is no automated frontend test suite.
