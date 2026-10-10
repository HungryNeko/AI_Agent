#!/bin/sh
set -eu

LOCAL_DATA_DIR="${AI_AGENT_LOCAL_DATA_DIR:-/app/local-data}"
RUNTIME_DIR="${AI_AGENT_RUNTIME_DIR:-/app/backend/runtime}"
LEASE_MCP_URL="${LEASE_MCP_URL:-http://web:5000/mcp}"

mkdir -p "$LOCAL_DATA_DIR/mcp" "$RUNTIME_DIR" /app/data/mcp
ln -sfn "$LOCAL_DATA_DIR/api_configs.local.json" /app/data/api_configs.local.json
ln -sfn "$LOCAL_DATA_DIR/settings.local.json" /app/data/settings.local.json
ln -sfn "$LOCAL_DATA_DIR/mcp/servers.local.json" /app/data/mcp/servers.local.json

# Runtime-editable files that live under data/ must survive image rebuilds.
# Keep them in the mounted local-data volume and link them into the image tree.
# instruction.md: seed from the image copy the first time so nothing is lost.
if [ ! -e "$LOCAL_DATA_DIR/instruction.md" ] && [ -f /app/data/instruction.md ] && [ ! -L /app/data/instruction.md ]; then
  cp -p /app/data/instruction.md "$LOCAL_DATA_DIR/instruction.md"
fi
ln -sfn "$LOCAL_DATA_DIR/instruction.md" /app/data/instruction.md
for dir in custom_tools plans; do
  mkdir -p "$LOCAL_DATA_DIR/$dir"
  if [ ! -L "/app/data/$dir" ]; then
    cp -an "/app/data/$dir/." "$LOCAL_DATA_DIR/$dir/" 2>/dev/null || true
    rm -rf "/app/data/$dir"
  fi
  ln -sfn "$LOCAL_DATA_DIR/$dir" "/app/data/$dir"
done

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
