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
  git fetch --all --prune
  current_branch="$(git rev-parse --abbrev-ref HEAD)"
  git pull --ff-only origin "$current_branch"
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

echo "Building and updating Docker containers..."
docker compose -f "$COMPOSE_FILE" up -d --build --remove-orphans

echo "Deploy update complete."
