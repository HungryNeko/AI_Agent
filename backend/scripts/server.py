"""Run the FastAPI backend for the React frontend."""

from __future__ import annotations

import sys
import os
from pathlib import Path

import uvicorn

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

if __name__ == "__main__":
    port = int(os.environ.get("AI_AGENT_BACKEND_PORT", "8010"))
    uvicorn.run("agent.server:app", host="127.0.0.1", port=port, reload=True)
