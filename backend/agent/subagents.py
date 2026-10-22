"""Parallel background sub-agents.

Each sub-agent is a normal agent loop (`graph.stream_turn`) running in its own thread with its
own context, and its python runs are tracked child processes.

Lifetime rules:

* A **foreground** agent (spawned with wait=true) belongs to the turn that created it. When that
  turn closes (finished, failed, stopped) and the agent is still running, it is killed together
  with its child processes.
* A **background** agent (spawned with wait=false, or a foreground one whose blocking wait timed
  out and was handed over) is deliberate: it survives the turn and the stop button. It is recorded
  in `_background.json`, and when it finishes while nobody waits for it the conversation is woken
  through `wakeups.schedule_wake`.
* Deleting the conversation or shutting the server down kills everything.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent import wakeups
from agent.config import PROJECT_ROOT
from agent.debug_log import log_event, log_exception
from agent.runs import is_cancelled_run
from tools import procs
from tools.subagent import MAX_BLOCKING_SECONDS, SubagentRequest, SubagentTask, TaskItem

ROOT = PROJECT_ROOT / "backend" / "runtime" / "subagents"
MAX_PER_CONVERSATION = 4  # running or queued at once
MAX_RUNNING_TOTAL = 8  # real threads executing at once, across conversations
AGENT_TIMEOUT_SECONDS = 3600.0
RESULT_LIMIT_CHARS = 8000  # what the main agent sees of a result
EVENT_LIMIT = 300
FINISHED = {"done", "failed", "cancelled", "timeout", "interrupted"}
ACTIVE = {"queued", "running"}
BLOCKED_TOOLS = ("agent", "approval", "plan")  # no recursion, nobody to ask, no plan mode

_SLOTS = threading.BoundedSemaphore(MAX_RUNNING_TOTAL)

TASK_WRAPPER = (
    "You are a sub-agent working for a main agent, not talking to the user. Start by writing a short task list "
    "(the steps you will take), then complete the task on your own with your tools, and finish with a concise "
    "final report: what you did, the key results or numbers, relevant file paths, and any caveats. Your final "
    "message is the only thing the main agent sees, so make the report complete on its own. Do not ask "
    "questions; make reasonable assumptions and state them. Other sub-agents may be running in parallel; files "
    "are protected by read/write locks, so if a file is reported as locked, work on something else first and "
    "retry. Only touch files the task needs.\n\n"
    "Task name: {name}\nTask:\n{prompt}"
)


def stream_turn(state: dict[str, Any], message: str):
    """Indirection so tests can replace the model loop."""

    from agent import graph

    return graph.stream_turn(state, message)  # type: ignore[arg-type]


@dataclass
class SubAgent:
    id: str
    conversation_id: str
    name: str
    prompt: str
    tools: str = "full"
    model: str = ""
    max_rounds: int = 30
    python_timeout: float = 1800.0
    task_id: str = ""
    background: bool = False  # deliberately left running after its parent turn closes
    spawned_by: str = ""  # run id of the turn that owns it
    list_run: str = ""  # run id of the task list it belongs to
    kill_reason: str = ""
    status: str = "queued"
    result: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    rounds: int = 0
    activity: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] | None = None
    watchers: int = 0  # blocking calls that will hand the result to the model
    delivered: bool = False  # the model has seen the final result
    cancel: threading.Event = field(default_factory=threading.Event)

    @property
    def owner_key(self) -> str:
        return f"agent:{self.conversation_id}:{self.id}"

    def summary(self, *, with_result: bool = False, limit: int = 300) -> dict[str, Any]:
        end = self.finished_at or time.time()
        data: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "taskId": self.task_id,
            "status": self.status,
            "background": self.background,
            "tools": self.tools,
            "elapsedSeconds": round(end - (self.started_at or self.created_at), 1),
            "startedAt": self.started_at or self.created_at,
            "finishedAt": self.finished_at,
            "rounds": self.rounds,
            "activity": self.activity,
        }
        if self.kill_reason:
            data["killReason"] = self.kill_reason
        if self.error:
            data["error"] = self.error[:1000]
        if self.status in FINISHED and self.result:
            data["result"] = self.result[: (RESULT_LIMIT_CHARS if with_result else limit)]
            if with_result and len(self.result) > RESULT_LIMIT_CHARS:
                data["resultTruncated"] = True
        return data

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "name": self.name,
            "prompt": self.prompt,
            "tools": self.tools,
            "model": self.model,
            "max_rounds": self.max_rounds,
            "python_timeout": self.python_timeout,
            "task_id": self.task_id,
            "background": self.background,
            "spawned_by": self.spawned_by,
            "list_run": self.list_run,
            "kill_reason": self.kill_reason,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "rounds": self.rounds,
            "activity": self.activity,
            "events": self.events,
            "state": self.state,
        }


class SubagentManager:
    def __init__(self) -> None:
        self._agents: dict[str, dict[str, SubAgent]] = {}
        self._tasks: dict[str, dict[str, Any]] = {}  # conversation -> {"run": run id, "items": {task id: title}}
        self._lock = threading.RLock()

    # ------------------------------------------------------------ lookup / persistence

    def _dir(self, conversation_id: str) -> Path:
        return ROOT / conversation_id

    def _write_json(self, conversation_id: str, filename: str, data: Any) -> None:
        try:
            directory = self._dir(conversation_id)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / filename).write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
        except OSError as exc:
            log_exception("subagent.save_error", exc, conversation_id=conversation_id, file=filename)

    def _save(self, agent: SubAgent, status: str | None = None) -> None:
        data = agent.to_json()
        if status:
            data["status"] = status
        self._write_json(agent.conversation_id, f"{agent.id}.json", data)

    def _save_registry(self, conversation_id: str) -> None:
        """`_background.json`: which agents run (or ran) deliberately in the background, and which were killed."""

        agents = list(self._agents.get(conversation_id, {}).values())
        self._write_json(
            conversation_id,
            "_background.json",
            {
                "background": [
                    {
                        "id": a.id,
                        "name": a.name,
                        "status": a.status,
                        "spawnedByRun": a.spawned_by,
                        "startedAt": a.started_at or a.created_at,
                        "finishedAt": a.finished_at,
                    }
                    for a in agents
                    if a.background
                ],
                "killed": [
                    {"id": a.id, "name": a.name, "reason": a.kill_reason, "at": a.finished_at}
                    for a in agents
                    if a.kill_reason
                ],
            },
        )

    def registry(self, conversation_id: str) -> dict[str, Any]:
        agents = self.list_agents(conversation_id)
        return {
            "background": [a.summary() for a in agents if a.background],
            "killed": [a.summary() for a in agents if a.kill_reason],
        }

    def _load(self, conversation_id: str, agent_id: str) -> SubAgent | None:
        path = self._dir(conversation_id) / f"{agent_id}.json"
        if agent_id.startswith("_") or "/" in agent_id or "\\" in agent_id or ".." in agent_id or not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        agent = SubAgent(
            id=data["id"],
            conversation_id=conversation_id,
            name=data.get("name", data["id"]),
            prompt=data.get("prompt", ""),
            tools=data.get("tools", "full"),
            model=data.get("model", ""),
            max_rounds=int(data.get("max_rounds", 30)),
            python_timeout=float(data.get("python_timeout", 1800.0)),
            task_id=data.get("task_id", ""),
            background=bool(data.get("background", False)),
            spawned_by=data.get("spawned_by", ""),
            list_run=data.get("list_run", ""),
            kill_reason=data.get("kill_reason", ""),
            status=data.get("status", "interrupted"),
            result=data.get("result", ""),
            error=data.get("error", ""),
            created_at=data.get("created_at", 0.0),
            started_at=data.get("started_at", 0.0),
            finished_at=data.get("finished_at", 0.0),
            rounds=int(data.get("rounds", 0)),
            activity=data.get("activity", ""),
            events=list(data.get("events") or []),
            state=data.get("state"),
            delivered=True,
        )
        if agent.status not in FINISHED:
            # The server restarted while it ran: its thread is gone.
            agent.status = "interrupted"
            agent.error = agent.error or "interrupted by a server restart"
        return agent

    def get(self, conversation_id: str, agent_id: str) -> SubAgent | None:
        with self._lock:
            agent = self._agents.get(conversation_id, {}).get(agent_id)
            if agent is None:
                agent = self._load(conversation_id, agent_id)
                if agent is not None:
                    self._agents.setdefault(conversation_id, {})[agent_id] = agent
            return agent

    def list_agents(self, conversation_id: str) -> list[SubAgent]:
        with self._lock:
            directory = self._dir(conversation_id)
            if directory.is_dir():
                for path in directory.glob("*.json"):
                    if not path.stem.startswith("_"):
                        self.get(conversation_id, path.stem)
            return sorted(self._agents.get(conversation_id, {}).values(), key=lambda a: a.created_at)

    def require(self, conversation_id: str, agent_id: str) -> SubAgent:
        agent = self.get(conversation_id, agent_id)
        if agent is None:
            raise ValueError(f"unknown agent id: {agent_id}")
        return agent

    # ------------------------------------------------------------ task list

    def add_tasks(self, conversation_id: str, items: tuple[TaskItem, ...], run_id: str = "") -> None:
        """Record the task list stated by a turn.

        A task list belongs to the turn that stated it: a later turn starts a fresh list instead of
        inheriting (and mislabeling) the ids of an earlier one. Within one turn, repeated spawns add to
        the list and a repeated id takes the newest title.
        """

        with self._lock:
            current = self._tasks.get(conversation_id) or self._load_tasks(conversation_id)
            if current.get("run") != run_id:
                current = {"run": run_id, "items": {}}
            for item in items:
                current["items"][item.id] = item.title
            self._tasks[conversation_id] = current
            self._write_json(
                conversation_id,
                "_tasklist.json",
                {"run": current["run"], "items": [{"id": k, "title": v} for k, v in current["items"].items()]},
            )

    def _load_tasks(self, conversation_id: str) -> dict[str, Any]:
        path = self._dir(conversation_id) / "_tasklist.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):  # older format without a run id
                data = {"run": "", "items": data}
            return {"run": data.get("run", ""), "items": {i["id"]: i["title"] for i in data["items"]}}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return {"run": "", "items": {}}

    def task_list(self, conversation_id: str) -> list[dict[str, Any]]:
        """The current task list with a status derived from the agents working on each item."""

        with self._lock:
            current = self._tasks.get(conversation_id) or self._load_tasks(conversation_id)
            agents = self.list_agents(conversation_id)
        result = []
        for task_id, title in current["items"].items():
            linked = [a for a in agents if a.task_id == task_id and a.list_run == current["run"]]
            status = "pending"
            if linked:
                latest = linked[-1]
                status = (
                    "running" if latest.status in ACTIVE
                    else "done" if latest.status == "done"
                    else "cancelled" if latest.status == "cancelled"
                    else "failed"
                )
            result.append({"id": task_id, "title": title, "status": status, "agents": [a.id for a in linked]})
        return result

    def _publish_tasks(self, conversation_id: str) -> None:
        wakeups.publish(conversation_id, {"type": "tasklist_update", "tasks": self.task_list(conversation_id)})

    # ------------------------------------------------------------ lifecycle

    def spawn(
        self,
        conversation_id: str,
        tasks: tuple[SubagentTask, ...],
        settings: Any,
        *,
        watch: bool,
        task_list: tuple[TaskItem, ...] = (),
    ) -> list[SubAgent]:
        run_id = str(getattr(settings, "run_id", "") or "")
        with self._lock:
            active = sum(1 for a in self._agents.get(conversation_id, {}).values() if a.status in ACTIVE)
            if active + len(tasks) > MAX_PER_CONVERSATION:
                raise ValueError(
                    f"at most {MAX_PER_CONVERSATION} agents may run at once in a conversation "
                    f"({active} active); wait for or cancel some first."
                )
            created: list[SubAgent] = []
            for task in tasks:
                agent = SubAgent(
                    id=uuid.uuid4().hex[:8],
                    conversation_id=conversation_id,
                    name=task.name,
                    prompt=task.prompt,
                    task_id=task.task_id,
                    tools=task.tools,
                    model=task.model or str(getattr(settings, "subagent_model", "") or "") or str(getattr(settings, "model", "") or ""),
                    max_rounds=task.max_rounds,
                    python_timeout=task.python_timeout,
                    background=not watch,
                    spawned_by=run_id,
                    list_run=run_id,
                    watchers=1 if watch else 0,
                )
                agent.state = self._initial_state(agent, settings)
                self._agents.setdefault(conversation_id, {})[agent.id] = agent
                created.append(agent)
        self.add_tasks(conversation_id, task_list, run_id)
        self._save_registry(conversation_id)
        for agent in created:
            if agent.background:
                log_event("subagent.background", agent=agent.id, name=agent.name, conversation_id=conversation_id)
            self._start(agent, TASK_WRAPPER.format(name=agent.name, prompt=agent.prompt))
        self._publish_tasks(conversation_id)
        return created

    def _initial_state(self, agent: SubAgent, settings: Any) -> dict[str, Any]:
        return build_state(
            settings,
            conversation_id=f"{agent.conversation_id}-{agent.id}",
            tools=agent.tools,
            model=agent.model,
            max_rounds=agent.max_rounds,
            python_timeout=agent.python_timeout,
        )

    def _start(self, agent: SubAgent, message: str) -> None:
        threading.Thread(target=self._run, args=(agent, message), daemon=True, name=f"subagent-{agent.id}").start()

    def _publish(self, agent: SubAgent) -> None:
        wakeups.publish(agent.conversation_id, {"type": "subagent_update", "agent": agent.summary()})

    def _run(self, agent: SubAgent, message: str) -> None:
        procs.set_owner(agent.owner_key, agent.name)  # its python runs are killed with it; file locks show its name
        if not agent.cancel.is_set():
            procs.clear_owner(agent.owner_key)
        with _SLOTS:
            if agent.cancel.is_set():
                self._finish(agent, "cancelled", agent.kill_reason)
                return
            agent.status = "running"
            agent.started_at = agent.started_at or time.time()
            self._publish(agent)
            deadline = time.monotonic() + AGENT_TIMEOUT_SECONDS
            status, error = "done", ""
            try:
                state = dict(agent.state or {})
                for event in stream_turn(state, message):
                    if agent.cancel.is_set():
                        status, error = "cancelled", agent.kill_reason
                        break
                    if time.monotonic() > deadline:
                        status, error = "timeout", f"exceeded {int(AGENT_TIMEOUT_SECONDS)}s"
                        break
                    self._record(agent, event)
                if agent.cancel.is_set() and status == "done":
                    status, error = "cancelled", agent.kill_reason
                if status == "done" and not agent.result:
                    status, error = "failed", "the agent produced no final answer"
            except Exception as exc:  # noqa: BLE001 - a crashing agent must not take the server down
                log_exception("subagent.error", exc, agent=agent.id)
                status, error = ("cancelled", agent.kill_reason) if agent.cancel.is_set() else ("failed", str(exc))
            finally:
                procs.kill_owner(agent.owner_key)  # nothing it started outlives it
            self._finish(agent, status, error)

    def _record(self, agent: SubAgent, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "assistant":
            agent.result = str(event.get("text") or "").strip()
            raw = event.get("state")
            if isinstance(raw, dict):
                agent.state = {k: v for k, v in raw.items() if k not in {"settings", "message"}}
            return
        text = str(event.get("text") or event.get("query") or event.get("url") or "")[:500]
        if kind == "tool_call":
            agent.rounds += 1
            agent.activity = text or str(event.get("tool") or "")
            self._publish(agent)
        elif kind == "assistant_progress" and text:
            agent.activity = text[:200]
        agent.events.append({"type": kind, "text": text, "tool": event.get("tool", "")})
        del agent.events[:-EVENT_LIMIT]

    def _finish(self, agent: SubAgent, status: str, error: str = "") -> None:
        with self._lock:
            agent.error = error
            agent.finished_at = time.time()
        # Persist first so that whoever sees the finished status can also load it from disk.
        self._save(agent, status)
        with self._lock:
            agent.status = status
            wake = status in {"done", "failed", "timeout"} and agent.watchers == 0 and not agent.delivered
        self._save_registry(agent.conversation_id)
        self._publish(agent)
        self._publish_tasks(agent.conversation_id)
        log_event("subagent.finished", agent=agent.id, status=status, conversation_id=agent.conversation_id)
        if wake:
            self._wake(agent)

    def _wake(self, agent: SubAgent) -> None:
        with self._lock:
            if agent.delivered:
                return
            agent.delivered = True  # the wake message carries the result
        wakeups.schedule_wake(agent.conversation_id, agent.summary(with_result=True))

    def send(self, conversation_id: str, agent_id: str, message: str, *, watch: bool, run_id: str = "") -> SubAgent:
        with self._lock:
            agent = self.require(conversation_id, agent_id)
            if agent.status in ACTIVE:
                raise ValueError(f"agent {agent_id} is still running; wait for it or cancel it first.")
            if not agent.state:
                raise ValueError(f"agent {agent_id} has no saved context to continue.")
            agent.status = "queued"
            agent.result = ""
            agent.error = ""
            agent.kill_reason = ""
            agent.finished_at = 0.0
            agent.delivered = False
            agent.watchers = 1 if watch else 0
            agent.background = not watch
            agent.spawned_by = run_id or agent.spawned_by
            agent.cancel = threading.Event()
        self._save_registry(conversation_id)
        self._start(agent, message)
        self._publish_tasks(conversation_id)
        return agent

    # ------------------------------------------------------------ stopping

    def _kill(self, agent: SubAgent, reason: str) -> None:
        """Stop an agent and every process it started."""

        with self._lock:
            if agent.status in FINISHED:
                return
            agent.kill_reason = reason
            agent.cancel.set()
            if agent.status == "queued":
                agent.status = "cancelled"
        killed = procs.kill_owner(agent.owner_key)
        log_event("subagent.killed", agent=agent.id, name=agent.name, reason=reason, processes=killed)
        self._save_registry(agent.conversation_id)

    def cancel(self, conversation_id: str, agent_id: str) -> SubAgent:
        agent = self.require(conversation_id, agent_id)
        self._kill(agent, "cancelled on request")
        return agent

    def close_run(self, conversation_id: str, run_id: str) -> list[SubAgent]:
        """The turn `run_id` is over: kill the foreground agents it started that are still running.

        Background agents are deliberate and stay (they are listed in `_background.json`).
        """

        if not run_id:
            return []
        with self._lock:
            victims = [
                a
                for a in self._agents.get(conversation_id, {}).values()
                if a.spawned_by == run_id and not a.background and a.status in ACTIVE
            ]
        for agent in victims:
            self._kill(agent, "its parent turn closed while it was still running")
        return victims

    def close_conversation(self, conversation_id: str) -> None:
        """The conversation is gone: kill everything, background agents included, and drop the records."""

        with self._lock:
            agents = list(self._agents.get(conversation_id, {}).values())
        for agent in agents:
            self._kill(agent, "conversation deleted")
        with self._lock:
            self._agents.pop(conversation_id, None)
            self._tasks.pop(conversation_id, None)
        shutil.rmtree(self._dir(conversation_id), ignore_errors=True)

    def shutdown(self) -> None:
        """Server stopping: kill every running agent and child process, and record what was running."""

        with self._lock:
            running = [a for items in self._agents.values() for a in items.values() if a.status in ACTIVE]
        for agent in running:
            agent.kill_reason = "server shutdown"
            agent.cancel.set()
            agent.error = "interrupted by server shutdown"
            self._save(agent, "interrupted")
            self._save_registry(agent.conversation_id)
        procs.kill_all()

    # ------------------------------------------------------------ waiting

    def _hold(self, agent: SubAgent) -> None:
        with self._lock:
            agent.watchers += 1

    def _release(self, agent: SubAgent) -> None:
        with self._lock:
            agent.watchers = max(0, agent.watchers - 1)
            late = agent.status in {"done", "failed", "timeout"} and agent.watchers == 0 and not agent.delivered
        if late:
            self._wake(agent)

    def wait(
        self,
        conversation_id: str,
        agents: list[SubAgent],
        timeout: float,
        run_id: str = "",
        *,
        initial_hold: bool = False,
    ) -> list[SubAgent]:
        """Block until all given agents finish, the timeout passes, or the turn is stopped.

        Agents still running when the *timeout* passes are handed over to the background (the model is
        told they will wake it); if the turn was *stopped* instead they stay foreground and are killed
        when the turn closes.
        """

        if not initial_hold:
            for agent in agents:
                self._hold(agent)
        deadline = time.monotonic() + min(max(timeout, 0.0), MAX_BLOCKING_SECONDS)
        stopped = False
        try:
            while time.monotonic() < deadline:
                if all(a.status in FINISHED for a in agents):
                    break
                if run_id and is_cancelled_run(run_id):
                    stopped = True
                    break
                time.sleep(0.25)
            with self._lock:
                for agent in agents:
                    if agent.status in FINISHED:
                        agent.delivered = True  # handed to the model right now
                    elif not stopped and not agent.background:
                        agent.background = True  # deliberately continues without the turn
                        log_event("subagent.background", agent=agent.id, name=agent.name, promoted=True)
            self._save_registry(conversation_id)
        finally:
            for agent in agents:
                self._release(agent)
        return agents


MANAGER = SubagentManager()


def build_state(
    settings: Any,
    *,
    conversation_id: str,
    tools: str,
    model: str,
    max_rounds: int,
    python_timeout: float,
) -> dict[str, Any]:
    """Chat state for an agent started by another agent (sub-agent) or by python code (run_agent)."""

    state: dict[str, Any] = {
        "message": "",
        "conversation_id": conversation_id,
        "conversation_mode": "agent" if tools == "full" else "ask",
        "web_search_mode": settings.web_search.ui_mode,
        "web_search_provider": settings.web_search.provider,
        "rag_mode": settings.rag.mode,
        "rag_include_knowledge": settings.rag.include_knowledge,
        "rag_include_memory": settings.rag.include_memory,
        "rag_include_skills": settings.rag.include_skills,
        "curl_mode": settings.curl.mode,
        "python_mode": settings.python.mode,
        "python_timeout_seconds": python_timeout,
        "file_reader_mode": settings.file_reader.mode,
        "file_editor_mode": settings.file_editor.mode,
        "file_editor_approval": settings.file_editor.approval,
        "mcp_mode": settings.mcp.mode,
        "history_mode": settings.history.mode,
        "automation_mode": "off",
        "question_mode": "off",
        "max_tool_rounds": max_rounds,
        "blocked_tools": list(BLOCKED_TOOLS),
        "bridge_depth": int(getattr(settings, "bridge_depth", 0) or 0) + 1,
        "reasoning_effort": str(getattr(settings, "subagent_reasoning", "") or "default"),
        "subagent_model": str(getattr(settings, "subagent_model", "") or ""),
        "subagent_reasoning_effort": str(getattr(settings, "subagent_reasoning", "") or "default"),
    }
    if model:
        state["model"] = model
    return state


def run_inline_agent(
    settings: Any,
    prompt: str,
    *,
    tools: str = "full",
    model: str = "",
    max_rounds: int = 20,
    python_timeout: float = 1800.0,
    should_stop: Any = None,
) -> str:
    """Run an agent loop right here on a custom prompt and return its final answer (used by `run_agent`)."""

    state = build_state(
        settings,
        conversation_id=f"{getattr(settings, 'conversation_id', '') or 'inline'}-inline-{uuid.uuid4().hex[:6]}",
        tools=tools,
        model=model or str(getattr(settings, "subagent_model", "") or "") or str(getattr(settings, "model", "") or ""),
        max_rounds=max_rounds,
        python_timeout=python_timeout,
    )
    answer = ""
    for event in stream_turn(state, prompt):
        if should_stop is not None and should_stop():
            raise RuntimeError("stopped")
        if event.get("type") == "assistant":
            answer = str(event.get("text") or "").strip()
    if not answer:
        raise RuntimeError("the agent produced no final answer")
    return answer


# ---------------------------------------------------------------- tool entry point


def handle(request: SubagentRequest, settings: Any) -> str:
    conversation_id = str(getattr(settings, "conversation_id", "") or "")
    if not conversation_id:
        raise ValueError("agent needs a conversation context.")
    run_id = str(getattr(settings, "run_id", "") or "")
    manager = MANAGER
    action = request.action

    if action == "spawn":
        agents = manager.spawn(conversation_id, request.tasks, settings, watch=request.wait, task_list=request.task_list)
        if request.wait:
            manager.wait(conversation_id, agents, request.timeout, run_id, initial_hold=True)
            return format_result(agents, manager, conversation_id, with_result=True, note=_blocking_note(agents))
        return format_result(
            agents,
            manager,
            conversation_id,
            note=(
                "Started in the background (recorded as background agents: they keep running even if this turn "
                "ends or is stopped). You will be woken with a systemEvent when they finish; you may keep working "
                "or end your turn. Use agent wait/status only if you need results sooner."
            ),
        )

    if action == "send":
        agent = manager.send(conversation_id, request.ids[0], request.message, watch=request.wait, run_id=run_id)
        if request.wait:
            manager.wait(conversation_id, [agent], request.timeout, run_id, initial_hold=True)
            return format_result([agent], manager, conversation_id, with_result=True, note=_blocking_note([agent]))
        return format_result([agent], manager, conversation_id, note="Continuing in the background; you will be woken when it finishes.")

    if action == "status":
        agents = _select(manager, conversation_id, request.ids)
        return format_result(agents, manager, conversation_id, note=f"{len(agents)} agent(s).")

    if action == "wait":
        agents = _select(manager, conversation_id, request.ids, running_only=not request.ids)
        if not agents:
            return format_result([], manager, conversation_id, note="No running agents.")
        manager.wait(conversation_id, agents, request.timeout, run_id)
        return format_result(agents, manager, conversation_id, with_result=True, note=_blocking_note(agents))

    if action == "result":
        agents = _select(manager, conversation_id, request.ids)
        with manager._lock:
            for agent in agents:
                if agent.status in FINISHED:
                    agent.delivered = True
        return format_result(agents, manager, conversation_id, with_result=True)

    if action == "cancel":
        agents = [manager.cancel(conversation_id, agent_id) for agent_id in request.ids]
        return format_result(agents, manager, conversation_id, note="Cancellation requested; its child processes are being stopped.")

    raise ValueError(f"unknown agent action: {action}")


def _select(manager: SubagentManager, conversation_id: str, ids: tuple[str, ...], running_only: bool = False) -> list[SubAgent]:
    if ids:
        return [manager.require(conversation_id, agent_id) for agent_id in ids]
    agents = manager.list_agents(conversation_id)
    return [a for a in agents if a.status in ACTIVE] if running_only else agents


def _blocking_note(agents: list[SubAgent]) -> str:
    running = [a.name for a in agents if a.status not in FINISHED]
    if running:
        return (
            f"Still running: {', '.join(running)}. They were handed over to the background (recorded) and wake you "
            "with a systemEvent when finished; end your turn or do other work instead of polling."
        )
    return "All agents finished. Review the results; use agent send to refine or redo any of them."


def format_result(
    agents: list[SubAgent],
    manager: SubagentManager,
    conversation_id: str,
    *,
    with_result: bool = False,
    note: str = "",
) -> str:
    payload: dict[str, Any] = {
        "taskList": manager.task_list(conversation_id),
        "agents": [a.summary(with_result=with_result) for a in agents],
    }
    if note:
        payload["note"] = note
    return "agentResult:\n" + json.dumps(payload, ensure_ascii=False)
