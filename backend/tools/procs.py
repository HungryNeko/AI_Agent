"""Tracked child processes, so that stopping an agent also stops what it started.

`python.run` starts its interpreter through `run_tracked`. Each thread may declare an owner
(a sub-agent); `kill_owner` then terminates the whole process tree of everything that owner
is currently running, instead of leaving a training script running after its agent was cancelled.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
from collections.abc import Iterator
from typing import Any

_local = threading.local()
_LOCK = threading.Lock()
_PROCS: dict[str, set[subprocess.Popen]] = {}
_CANCELLED: set[str] = set()  # owners that were killed: they may not start new processes


def set_owner(owner: str, label: str = "") -> None:
    _local.owner = owner
    _local.label = label or owner


def current_owner() -> str:
    return getattr(_local, "owner", "")


def current_label() -> str:
    return getattr(_local, "label", "") or "main"


@contextlib.contextmanager
def owner_scope(owner: str, label: str = "") -> Iterator[None]:
    previous = (current_owner(), getattr(_local, "label", ""))
    set_owner(owner, label)
    try:
        yield
    finally:
        _local.owner, _local.label = previous


def kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, check=False)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    with contextlib.suppress(OSError):
        proc.kill()


def run_tracked(
    args: list[str], *, timeout: float | None = None, capture_output: bool = True, **kwargs: Any
) -> subprocess.CompletedProcess:
    """Like `subprocess.run`, but the process is registered under the current owner while it runs."""

    kwargs.pop("check", None)
    if capture_output:
        kwargs.setdefault("stdout", subprocess.PIPE)
        kwargs.setdefault("stderr", subprocess.PIPE)
    if os.name == "nt":
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    owner = current_owner()
    with _LOCK:
        if owner and owner in _CANCELLED:
            raise RuntimeError("cancelled: this agent was stopped, so it may not start new processes")
        proc = subprocess.Popen(args, **kwargs)
        if owner:
            _PROCS.setdefault(owner, set()).add(proc)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        proc.communicate()
        raise
    except BaseException:
        kill_tree(proc)
        raise
    finally:
        if owner:
            with _LOCK:
                _PROCS.get(owner, set()).discard(proc)
    return subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)


def clear_owner(owner: str) -> None:
    """The owner may run processes again (a stopped agent that is being continued)."""

    with _LOCK:
        _CANCELLED.discard(owner)


def kill_owner(owner: str) -> int:
    with _LOCK:
        _CANCELLED.add(owner)
        procs = list(_PROCS.get(owner, ()))
    for proc in procs:
        kill_tree(proc)
    return len(procs)


def kill_all() -> int:
    with _LOCK:
        _CANCELLED.update(_PROCS)
        procs = [proc for group in _PROCS.values() for proc in group]
    for proc in procs:
        kill_tree(proc)
    return len(procs)


def running_for(owner: str) -> int:
    with _LOCK:
        return sum(1 for proc in _PROCS.get(owner, ()) if proc.poll() is None)
