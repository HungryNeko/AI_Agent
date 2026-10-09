#!/bin/sh
set -eu

LOCAL_DATA_DIR="${AI_AGENT_LOCAL_DATA_DIR:-/app/local-data}"
RUNTIME_DIR="${AI_AGENT_RUNTIME_DIR:-/app/backend/runtime}"
LEASE_MCP_URL="${LEASE_MCP_URL:-http://web:5000/mcp}"

mkdir -p "$LOCAL_DATA_DIR/mcp" "$RUNTIME_DIR" /app/data/mcp
ln -sfn "$LOCAL_DATA_DIR/api_configs.local.json" /app/data/api_configs.local.json
ln -sfn "$LOCAL_DATA_DIR/settings.local.json" /app/data/settings.local.json
ln -sfn "$LOCAL_DATA_DIR/mcp/servers.local.json" /app/data/mcp/servers.local.json

if [ ! -f "$LOCAL_DATA_DIR/mcp/servers.local.json" ]; then
  python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ.get("AI_AGENT_LOCAL_DATA_DIR", "/app/local-data")) / "mcp" / "servers.local.json"
token = os.environ.get("RENT_MCP_TOKEN", "").strip()
headers = {"Authorization": f"Bearer {token}"} if token else {}
config = {
    "servers": {
        "Rent": {
            "enabled": True,
            "transport": "streamable_http",
            "url": os.environ.get("LEASE_MCP_URL", "http://web:5000/mcp"),
            "headers": headers,
            "timeout": 5.0,
            "sse_read_timeout": 300.0,
        }
    }
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
fi

exec python -m uvicorn agent.server:app --host 0.0.0.0 --port "${AI_AGENT_BACKEND_PORT:-8010}"
