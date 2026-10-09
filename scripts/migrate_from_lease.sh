#!/bin/sh
# One-time migration of the agent's server-side state from the old embedded
# deployment (Lease-Management-System/AI/agent) into this repo's docker-data/.
#
# Copies (never overwrites anything that already exists here):
#   AI/agent/data/{api_configs,settings}.local.json, mcp/servers.local.json
#   AI/agent/runtime/                     (conversations, uploads, ...)
#   AI-related variables from the lease .env  -> ./.env (only if ./.env is missing)
#
# Safe to re-run; scripts/update_server.sh calls it until the marker exists.
set -eu

APP_DIR="${APP_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}"
LEGACY_LEASE_DIR="${LEGACY_LEASE_DIR:-/root/github/Lease-Management-System}"
DATA_DIR="$APP_DIR/docker-data"
MARKER="$DATA_DIR/.migrated-from-lease"

ENV_KEYS="AI_AGENT_DEFAULT_MODEL LLM_MODEL LLM_API_KEY AI_API_KEY DEEPSEEK_API_KEY OPENAI_API_KEY OPENROUTER_API_KEY ALIYUN_1875083326719091_API_KEY RENT_MCP_TOKEN NPM_REGISTRY"

if [ -f "$MARKER" ]; then
  echo "Lease migration already done ($MARKER)."
  exit 0
fi

legacy_agent="$LEGACY_LEASE_DIR/AI/agent"
if [ ! -d "$legacy_agent" ] && [ ! -f "$LEGACY_LEASE_DIR/.env" ]; then
  echo "No legacy agent data under $LEGACY_LEASE_DIR, skip migration."
  mkdir -p "$DATA_DIR"
  touch "$MARKER"
  exit 0
fi

echo "Migrating agent state from $LEGACY_LEASE_DIR ..."
mkdir -p "$DATA_DIR/local-data/mcp" "$DATA_DIR/runtime"

copy_if_missing() {
  from="$1"
  to="$2"
  if [ -f "$from" ] && [ ! -e "$to" ]; then
    cp -p "$from" "$to"
    echo "  copied $from -> $to"
  fi
}

copy_if_missing "$legacy_agent/data/api_configs.local.json" "$DATA_DIR/local-data/api_configs.local.json"
copy_if_missing "$legacy_agent/data/settings.local.json" "$DATA_DIR/local-data/settings.local.json"
copy_if_missing "$legacy_agent/data/mcp/servers.local.json" "$DATA_DIR/local-data/mcp/servers.local.json"

if [ -d "$legacy_agent/runtime" ]; then
  # -n: keep anything already present in the new runtime dir.
  cp -an "$legacy_agent/runtime/." "$DATA_DIR/runtime/"
  echo "  copied runtime $legacy_agent/runtime -> $DATA_DIR/runtime"
fi

if [ ! -f "$APP_DIR/.env" ] && [ -f "$LEGACY_LEASE_DIR/.env" ]; then
  : > "$APP_DIR/.env"
  chmod 600 "$APP_DIR/.env"
  for key in $ENV_KEYS; do
    line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?$key=" "$LEGACY_LEASE_DIR/.env" | tail -n 1 || true)"
    if [ -n "$line" ]; then
      printf '%s\n' "$line" | sed -E 's/^[[:space:]]*export[[:space:]]+//' >> "$APP_DIR/.env"
      echo "  .env: copied $key"
    fi
  done
fi

touch "$MARKER"
echo "Lease migration complete."
