#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT="${AI_AGENT_BACKEND_PORT:-8012}"
FRONTEND_PORT="5173"
BACKEND_PID=""
FRONTEND_PID=""

if [[ -n "${AI_AGENT_PYTHON:-}" ]]; then
    PYTHON_BIN="$AI_AGENT_PYTHON"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

# Give each background command its own process group. This lets cleanup stop
# Python/npm together with any child processes that they launch.
set -m

stop_process_group() {
    local process_group_id="$1"

    if [[ -z "$process_group_id" ]] || ! kill -0 -- "-$process_group_id" 2>/dev/null; then
        return
    fi

    kill -TERM -- "-$process_group_id" 2>/dev/null || true
}

process_group_is_running() {
    local process_group_id="$1"

    [[ -n "$process_group_id" ]] && kill -0 -- "-$process_group_id" 2>/dev/null
}

cleanup() {
    local attempt

    trap - EXIT INT TERM HUP
    set +e

    stop_process_group "$FRONTEND_PID"
    stop_process_group "$BACKEND_PID"

    # Give Vite/Uvicorn a moment to shut down cleanly before forcing any
    # surviving child process to exit.
    for attempt in {1..20}; do
        if ! process_group_is_running "$FRONTEND_PID" && \
           ! process_group_is_running "$BACKEND_PID"; then
            break
        fi
        sleep 0.1
    done

    if [[ -n "$FRONTEND_PID" ]]; then
        kill -KILL -- "-$FRONTEND_PID" 2>/dev/null || true
        wait "$FRONTEND_PID" 2>/dev/null || true
    fi
    if [[ -n "$BACKEND_PID" ]]; then
        kill -KILL -- "-$BACKEND_PID" 2>/dev/null || true
        wait "$BACKEND_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: Python was not found: $PYTHON_BIN" >&2
    exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
    echo "Error: npm was not found in PATH." >&2
    exit 1
fi
if [[ ! -d "$ROOT_DIR/frontend/node_modules" ]]; then
    echo "Error: frontend dependencies are missing. Run: npm --prefix frontend install" >&2
    exit 1
fi
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
    echo "Error: the backend requires Python 3.11 or newer." >&2
    exit 1
fi
if ! "$PYTHON_BIN" -c 'import fastapi, langgraph, uvicorn' >/dev/null 2>&1; then
    echo "Error: backend dependencies are missing for $PYTHON_BIN" >&2
    echo "Create the project environment with:" >&2
    echo "  python3 -m venv .venv" >&2
    echo "  .venv/bin/python -m pip install -e './backend[dev]'" >&2
    exit 1
fi

cd "$ROOT_DIR"

echo "Starting backend:  http://127.0.0.1:${BACKEND_PORT}"
AI_AGENT_BACKEND_PORT="$BACKEND_PORT" \
    "$PYTHON_BIN" backend/scripts/server.py &
BACKEND_PID=$!

echo "Starting frontend: http://127.0.0.1:${FRONTEND_PORT}/"
VITE_BACKEND_PROXY="${VITE_BACKEND_PROXY:-http://127.0.0.1:${BACKEND_PORT}}" \
    npm --prefix frontend run dev &
FRONTEND_PID=$!

echo "Press Ctrl+C to stop both services."

# macOS ships Bash 3.2, which has no `wait -n`, so poll both tracked jobs.
while true; do
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        if wait "$BACKEND_PID"; then
            status=0
        else
            status=$?
        fi
        echo "Backend stopped; shutting down the frontend."
        exit "$status"
    fi

    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        if wait "$FRONTEND_PID"; then
            status=0
        else
            status=$?
        fi
        echo "Frontend stopped; shutting down the backend."
        exit "$status"
    fi

    sleep 0.5
done
