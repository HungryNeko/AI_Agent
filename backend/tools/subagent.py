"""Sub-agent tool: the main agent delegates tasks to parallel background agents.

The tool only describes requests. Running, waiting and waking the conversation lives in
`agent.subagents` / `agent.wakeups` (imported lazily because those modules import the graph,
which imports the tool layer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SubagentAction = Literal["spawn", "send", "status", "wait", "result", "cancel"]
SUBAGENT_ACTIONS: tuple[str, ...] = ("spawn", "send", "status", "wait", "result", "cancel")
ToolsMode = Literal["full", "readonly"]

MAX_TASKS_PER_CALL = 6
MAX_TASK_LIST_ITEMS = 20
MAX_TASK_PROMPT_CHARS = 20_000
MAX_BLOCKING_SECONDS = 90.0  # a blocking call must stay shorter than typical proxy read timeouts


@dataclass(frozen=True)
class TaskItem:
    id: str
    title: str


@dataclass(frozen=True)
class SubagentTask:
    name: str
    prompt: str
    task_id: str = ""
    tools: ToolsMode = "full"
    model: str = ""
    max_rounds: int = 30
    python_timeout: float = 1800.0


@dataclass(frozen=True)
class SubagentRequest:
    action: SubagentAction
    task_list: tuple[TaskItem, ...] = ()
    tasks: tuple[SubagentTask, ...] = ()
    ids: tuple[str, ...] = ()
    message: str = ""
    wait: bool = True
    timeout: float = MAX_BLOCKING_SECONDS
    extra: dict[str, object] = field(default_factory=dict)


def execute(request: SubagentRequest, settings) -> str:
    from agent import subagents

    return subagents.handle(request, settings)
