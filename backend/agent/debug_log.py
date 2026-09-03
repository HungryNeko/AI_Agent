"""Append-only JSONL debug logging for backend calls and errors."""

from __future__ import annotations

import json
import os
import threading
import traceback
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAX_STRING_CHARS = 50_000
SENSITIVE_KEY_PARTS = ("authorization", "api_key", "apikey", "secret", "token", "password")
LARGE_PAYLOAD_KEY_PARTS = ("base64", "b64")
_LOCK = threading.Lock()


def log_event(event: str, **fields: Any) -> None:
    """Write one debug event. Logging must never break the agent."""

    try:
        now = datetime.now(UTC)
        record = {
            "ts": now.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "event": event,
            **sanitize(fields),
        }
        path = log_path(now)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception:
        return


def log_exception(event: str, exc: BaseException, **fields: Any) -> None:
    log_event(
        event,
        error_type=type(exc).__name__,
        error=str(exc),
        traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        **fields,
    )


def log_path(now: datetime | None = None) -> Path:
    current = now or datetime.now(UTC)
    return log_dir() / f"agent-{current.strftime('%Y%m%d')}.jsonl"


def log_dir() -> Path:
    configured = os.environ.get("AI_AGENT_LOG_DIR")
    if configured:
        return Path(configured).resolve()
    return project_root() / "backend" / "runtime" / "logs"


def sanitize(value: Any, *, key: str = "") -> Any:
    if is_sensitive_key(key):
        return "<redacted>"
    if is_large_payload_key(key) and isinstance(value, str):
        return f"<omitted {len(value)} chars>"
    if is_dataclass(value) and not isinstance(value, type):
        return sanitize(asdict(value), key=key)
    if isinstance(value, dict):
        return {str(item_key): sanitize(item_value, key=str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize(item) for item in value]
    if isinstance(value, bytes):
        return f"<bytes {len(value)}>"
    if isinstance(value, str):
        return trim_string(value)
    return value


def is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def is_large_payload_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in LARGE_PAYLOAD_KEY_PARTS)


def trim_string(value: str) -> str:
    if len(value) <= MAX_STRING_CHARS:
        return value
    return value[:MAX_STRING_CHARS] + f"\n...truncated {len(value) - MAX_STRING_CHARS} chars..."


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
