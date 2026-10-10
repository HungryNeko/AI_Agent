#!/bin/sh
# Server-side deploy for the AI agent (backend + frontend containers).
# The agent exposes no host port; the Lease-Management-System gateway proxies
# /ai-agent/ to it over the shared docker network.
set -eu

APP_DIR="${APP_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
SHARED_NETWORK="${SHARED_NETWORK:-lease-ai-net}"

cd "$APP_DIR"

echo "App dir: $APP_DIR"
echo "Compose file: $COMPOSE_FILE"

if [ -d .git ]; then
  echo "Fetching latest code..."
  git fetch --prune origin
  current_branch="$(git rev-parse --abbrev-ref HEAD)"
  # Match the remote exactly. A fast-forward-only pull aborts whenever the remote history was
  # rewritten; runtime data lives in ignored paths (docker-data, .env) and is not touched.
  git reset --hard "origin/$current_branch"
else
  echo "No .git directory found, skip git pull."
fi

if ! docker network inspect "$SHARED_NETWORK" >/dev/null 2>&1; then
  echo "Creating shared docker network: $SHARED_NETWORK"
  docker network create "$SHARED_NETWORK"
fi

sh scripts/migrate_from_lease.sh

mkdir -p \
  "$APP_DIR/docker-data/local-data/mcp" \
  "$APP_DIR/docker-data/runtime"

# Rescue files an older image kept only inside the container (they were lost on
# every rebuild). Runs before the rebuild; never overwrites existing data.
backend_container="$(docker compose -f "$COMPOSE_FILE" ps -q ai-agent-backend 2>/dev/null || true)"
if [ -n "$backend_container" ]; then
  for name in instruction.md custom_tools plans; do
    if [ ! -e "$APP_DIR/docker-data/local-data/$name" ]; then
      docker cp "$backend_container:/app/data/$name" "$APP_DIR/docker-data/local-data/$name" 2>/dev/null         && echo "Rescued $name from running container" || true
    fi
  done
fi

echo "Building and updating Docker containers..."
docker compose -f "$COMPOSE_FILE" up -d --build --remove-orphans

echo "Deploy update complete."
