"""Conversation runtime: turn gate, live event bus and background wake-ups.

Background work (sub-agents today, anything else later) keeps running after a turn ends or
the user presses stop. When it finishes while nobody is waiting for it, it calls
`schedule_wake`: the backend then starts a new turn in that conversation by itself, with a
special `systemEvent:` message, and publishes the turn's events to everyone watching the
conversation through the live stream. The user never has to prompt the agent and the agent
never has to sit in a polling loop.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from agent import session_store
from agent.debug_log import log_event, log_exception
from agent.runs import clear_cancelled_run, is_cancelled_run, is_paused_run, set_run_paused

# A turn that stopped reporting for this long (client vanished mid-turn) no longer blocks the gate.
STALE_GATE_SECONDS = 900.0
# Agents finishing close together are reported in one wake turn instead of one turn each.
WAKE_DEBOUNCE_SECONDS = 2.0
SYSTEM_EVENT_PREFIX = "systemEvent:\n"


class TurnGate:
    """Serializes turns of one conversation: one user turn or one wake turn at a time."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._owner: str | None = None
        self._beat = 0.0

    def _held(self) -> bool:
        return self._owner is not None and time.monotonic() - self._beat < STALE_GATE_SECONDS

    def acquire(self, owner: str, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._held():
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(1.0 if remaining is None else min(1.0, remaining))
            self._owner = owner
            self._beat = time.monotonic()
            return True

    def touch(self, owner: str) -> None:
        with self._cond:
            if self._owner == owner:
                self._beat = time.monotonic()

    def release(self, owner: str) -> None:
        with self._cond:
            if self._owner == owner:
                self._owner = None
                self._cond.notify_all()


class Runtime:
    def __init__(self) -> None:
        self.gate = TurnGate()
        self.lock = threading.Lock()
        self.pending: list[dict[str, Any]] = []
        self.drainer_active = False
        self.subscribers: list[Callable[[dict[str, Any]], None]] = []


_RUNTIMES: dict[str, Runtime] = {}
_RUNTIMES_LOCK = threading.Lock()


def runtime(conversation_id: str) -> Runtime:
    with _RUNTIMES_LOCK:
        if conversation_id not in _RUNTIMES:
            _RUNTIMES[conversation_id] = Runtime()
        return _RUNTIMES[conversation_id]


# ---------------------------------------------------------------- live event bus


def subscribe(conversation_id: str, callback: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
    """Register a thread-safe callback for live events; returns the unsubscribe function."""

    rt = runtime(conversation_id)
    with rt.lock:
        rt.subscribers.append(callback)

    def unsubscribe() -> None:
        with rt.lock:
            if callback in rt.subscribers:
                rt.subscribers.remove(callback)

    return unsubscribe


def publish(conversation_id: str, event: dict[str, Any]) -> None:
    rt = runtime(conversation_id)
    with rt.lock:
        callbacks = list(rt.subscribers)
    for callback in callbacks:
        try:
            callback(event)
        except Exception as exc:  # noqa: BLE001 - a broken subscriber must not stop the turn
            log_exception("wake.publish_error", exc)


# ---------------------------------------------------------------- wake-ups


def schedule_wake(conversation_id: str, item: dict[str, Any]) -> None:
    """Ask for a new turn in this conversation because background work finished."""

    rt = runtime(conversation_id)
    with rt.lock:
        rt.pending.append(item)
    log_event("wake.scheduled", conversation_id=conversation_id, item=item)
    kick(conversation_id)


def kick(conversation_id: str) -> None:
    """Start the drainer thread if wake-ups are waiting (also called after every user turn)."""

    rt = runtime(conversation_id)
    with rt.lock:
        if rt.drainer_active or not rt.pending:
            return
        rt.drainer_active = True
    threading.Thread(target=_drain, args=(conversation_id,), daemon=True, name=f"wake-{conversation_id}").start()


def _drain(conversation_id: str) -> None:
    rt = runtime(conversation_id)
    owner = f"wake-{uuid.uuid4().hex[:8]}"
    try:
        while True:
            with rt.lock:
                if not rt.pending:
                    rt.drainer_active = False
                    return
            if WAKE_DEBOUNCE_SECONDS > 0:
                time.sleep(WAKE_DEBOUNCE_SECONDS)
            rt.gate.acquire(owner)
            try:
                with rt.lock:
                    items = list(rt.pending)
                    rt.pending.clear()
                if not items:
                    continue
                if not run_wake_turn(conversation_id, items, owner):
                    # Not runnable right now (e.g. a question awaits the user): try again after their next turn.
                    with rt.lock:
                        rt.pending[:0] = items
                        rt.drainer_active = False
                    return
            finally:
                rt.gate.release(owner)
    except Exception as exc:  # noqa: BLE001
        log_exception("wake.drain_error", exc, conversation_id=conversation_id)
        with rt.lock:
            rt.drainer_active = False


def build_wake_message(items: list[dict[str, Any]]) -> tuple[str, str]:
    """The special command sent to the model, plus the short text shown in the chat."""

    names = ", ".join(str(item.get("name") or item.get("id")) for item in items)
    payload = {
        "type": "subagent_finished",
        "agents": items,
        "note": (
            "These background agents finished while you were not waiting for them, and the user did not send "
            "this message. Review the results: continue with agent send to refine or redo, or summarize for the "
            "user. Agents still running will wake you again when they finish."
        ),
    }
    return SYSTEM_EVENT_PREFIX + json.dumps(payload, ensure_ascii=False), f"后台任务完成：{names}"


def run_wake_turn(conversation_id: str, items: list[dict[str, Any]], owner: str) -> bool:
    """Run one server-initiated turn. Returns False when it had to be deferred."""

    from agent import graph

    path = session_store.conversation_path(conversation_id)
    if not path.exists():
        return True  # the conversation was deleted; nothing to wake
    conversation = session_store.read_conversation(conversation_id)
    state = dict(conversation.get("state") or {})
    if state.get("question_pending"):
        return False
    for key in ("settings", "message"):
        state.pop(key, None)
    run_id = uuid.uuid4().hex
    clear_cancelled_run(run_id)
    base_rev = int(state.get("rev") or 0)
    state["conversation_id"] = conversation_id
    state["run_id"] = run_id
    message, display = build_wake_message(items)

    turn_events: list[dict[str, Any]] = []
    final_state: dict[str, Any] = {}
    publish(conversation_id, {"type": "wake_started", "run_id": run_id, "conversation_id": conversation_id, "text": display})
    publish(conversation_id, {"type": "user", "text": display, "source": "wake", "run_id": run_id, "conversation_id": conversation_id})

    def emit(event: dict[str, Any]) -> None:
        event = {**event, "conversation_id": conversation_id, "run_id": run_id}
        turn_events.append(event)
        publish(conversation_id, event)

    try:
        for event in graph.stream_turn(state, message):
            rt = runtime(conversation_id)
            rt.gate.touch(owner)
            while is_paused_run(run_id) and not is_cancelled_run(run_id):
                time.sleep(0.1)
                rt.gate.touch(owner)
            if is_cancelled_run(run_id):
                emit({"type": "stopped", "text": "AI output stopped."})
                break
            if event.get("type") == "assistant" and isinstance(event.get("state"), dict):
                public = {k: v for k, v in event["state"].items() if k not in {"settings", "message"}}
                public["rev"] = base_rev + 1
                public.pop("run_id", None)
                final_state = public
                event = {**event, "state": public}
            emit(event)
    except Exception as exc:  # noqa: BLE001
        log_exception("wake.turn_error", exc, conversation_id=conversation_id)
        emit({"type": "error", "text": str(exc), "terminal": True})
    finally:
        set_run_paused(run_id, False)
        clear_cancelled_run(run_id)
        from agent import subagents

        subagents.MANAGER.close_run(conversation_id, run_id)

    saved_events = [{k: v for k, v in event.items() if k != "state"} for event in turn_events]
    session_store.save_turn(
        conversation_id,
        user_text=display,
        turn_events=saved_events,
        state=final_state or {key: value for key, value in state.items() if key != "run_id"},
        user_extra={"source": "wake"},
    )
    publish(conversation_id, {"type": "wake_finished", "run_id": run_id, "conversation_id": conversation_id})
    return True
