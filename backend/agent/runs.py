"""Cancel/pause registries for running turns, shared by user turns and background wake turns."""

from __future__ import annotations

from threading import Lock

_LOCK = Lock()
_CANCELLED: set[str] = set()
_PAUSED: set[str] = set()


def mark_cancelled_run(run_id: str) -> None:
    with _LOCK:
        _CANCELLED.add(run_id)


def is_cancelled_run(run_id: str) -> bool:
    with _LOCK:
        return run_id in _CANCELLED


def clear_cancelled_run(run_id: str) -> None:
    with _LOCK:
        _CANCELLED.discard(run_id)


def set_run_paused(run_id: str, paused: bool) -> None:
    with _LOCK:
        if paused:
            _PAUSED.add(run_id)
        else:
            _PAUSED.discard(run_id)


def is_paused_run(run_id: str) -> bool:
    with _LOCK:
        return run_id in _PAUSED
