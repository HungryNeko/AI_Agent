# AI Agent Frontend

React/Vite interface for chat, data and RAG management, model configuration,
automation, MCP, and system settings.

## Run

The recommended command starts the frontend and backend together from the repository
root:

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
