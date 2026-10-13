"""Per-file read/write locks shared by the main agent and all sub-agents.

Readers share a file, a writer has it alone. Every fileEditor/fileReader operation takes the
matching lock for its duration, so a `replace`/`append` read-modify-write is atomic and nobody
reads a half-written file. Waiting is bounded: after the timeout the tool returns a clear error
naming who holds the file instead of hanging the agent.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from tools import procs

DEFAULT_TIMEOUT_SECONDS = 30.0


class FileLockTimeout(ValueError):
    pass


class RWLock:
    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.readers: dict[str, int] = {}
        self.writer: str = ""
        self.writer_depth = 0


_LOCKS: dict[str, RWLock] = {}
_REGISTRY_LOCK = threading.Lock()


def _key(path: Path | str) -> str:
    resolved = str(Path(path).resolve())
    return resolved.lower() if os.name == "nt" else resolved


def _lock_for(key: str) -> RWLock:
    with _REGISTRY_LOCK:
        return _LOCKS.setdefault(key, RWLock())


@contextlib.contextmanager
def locked(path: Path | str, *, write: bool, timeout: float | None = None, owner: str = "") -> Iterator[None]:
    holder = owner or procs.current_label()
    timeout = DEFAULT_TIMEOUT_SECONDS if timeout is None else timeout
    lock = _lock_for(_key(path))
    deadline = time.monotonic() + timeout
    with lock.cond:
        while True:
            other_writer = bool(lock.writer) and lock.writer != holder
            other_readers = any(name != holder for name in lock.readers)
            if write and not other_writer and not other_readers:
                lock.writer = holder
                lock.writer_depth += 1
                break
            if not write and not other_writer:
                lock.readers[holder] = lock.readers.get(holder, 0) + 1
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                who = lock.writer if other_writer else ", ".join(name for name in lock.readers if name != holder)
                raise FileLockTimeout(
                    f"{Path(path).name} is locked by {who} ({'writing' if other_writer else 'reading'}); "
                    "retry in a moment or work on a different file."
                )
            lock.cond.wait(min(0.5, remaining))
    try:
        yield
    finally:
        with lock.cond:
            if write:
                lock.writer_depth -= 1
                if lock.writer_depth <= 0:
                    lock.writer, lock.writer_depth = "", 0
            else:
                lock.readers[holder] -= 1
                if lock.readers[holder] <= 0:
                    del lock.readers[holder]
            lock.cond.notify_all()


def snapshot() -> list[dict[str, object]]:
    """Currently held locks (for diagnostics and tests)."""

    with _REGISTRY_LOCK:
        items = list(_LOCKS.items())
    held = []
    for key, lock in items:
        with lock.cond:
            if lock.writer or lock.readers:
                held.append({"path": key, "writer": lock.writer, "readers": sorted(lock.readers)})
    return held


# ---------------------------------------------------------------- stale-write protection
#
# Locks make each operation atomic, but a whole-file `write` is still a lost update when an agent
# writes back text it read earlier and someone else changed the file in between. So every agent has
# a "seen" digest per file (what it last read or wrote) and a full overwrite is refused when the file
# no longer matches it. `replace`/`insert`/`append` work on the current text under the lock and are
# not affected.

_SEEN: dict[tuple[str, str], str] = {}
_LAST_WRITER: dict[str, tuple[str, str]] = {}  # file -> (digest, label of the agent that wrote it)
_VERSION_LOCK = threading.Lock()


class StaleFileError(ValueError):
    pass


def digest(path: Path | str) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def note_seen(owner: str, path: Path | str, value: str | None) -> None:
    if value is None:
        return
    with _VERSION_LOCK:
        _SEEN[(owner, _key(path))] = value


def note_written(owner: str, label: str, path: Path | str, before: str | None) -> None:
    """Record a change made by `owner`; its own view stays current only if it was current before."""

    after = digest(path)
    if after is None or after == before:
        return
    key = _key(path)
    with _VERSION_LOCK:
        _LAST_WRITER[key] = (after, label)
        if _SEEN.get((owner, key)) == before or before is None:
            _SEEN[(owner, key)] = after


def check_fresh(owner: str, path: Path | str) -> None:
    """Refuse a whole-file overwrite based on a stale (or never fetched) view of changed content."""

    current = digest(path)
    if current is None:
        return  # new file
    key = _key(path)
    with _VERSION_LOCK:
        seen = _SEEN.get((owner, key))
        last = _LAST_WRITER.get(key)
    other_changed = last is not None and last[0] == current
    if (seen is not None and seen != current) or (seen is None and other_changed):
        who = last[1] if last else "someone else"
        raise StaleFileError(
            f"{Path(path).name} was changed by {who} after you last read it (or you never read its current "
            "content), so overwriting it would erase their edits. Read it again, merge your change into the "
            "new content, then write again; or use replace/append, which work on the current text."
        )
