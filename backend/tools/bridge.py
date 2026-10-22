"""Let Python code (the python tool and custom tools) call MCP tools and the model.

Python runs in a child process, so it cannot call back into the backend directly. The child
writes JSON request files into a per-run directory and the parent, which is waiting for it
anyway, serves them in threads and writes the answers next to them. Nothing listens on a port,
so it works the same on a dev machine and in Docker.

Inside the child, `ai_agent_bridge` provides:

    call_mcp(server, tool, arguments=None)   -> the tool's data (structured content, parsed JSON or text)
    list_mcp_servers() / list_mcp_tools(server)
    ask_llm(prompt, system="", model="", max_tokens=0) -> str    one completion, no tools
    run_agent(prompt, tools="full", model="", max_rounds=20) -> str   a full agent loop with your own prompt

The time a run spends waiting for these calls does not count against the python timeout.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tools import procs, security

MAX_BRIDGE_DEPTH = 2  # an agent started from python may itself start one more through python, not further
POLL_SECONDS = 0.05
RESULT_LIMIT_CHARS = 60_000


class ToolBridge:
    def __init__(self, settings: Any, *, should_stop: Any = None) -> None:
        self.settings = settings
        self.should_stop = should_stop or (lambda: False)
        self.closed = threading.Event()
        self._lock = threading.Lock()
        self._finished_seconds = 0.0
        self._active: dict[str, float] = {}
        self._owner = procs.current_owner()
        self._label = procs.current_label()

    # ------------------------------------------------------------ time accounting

    def busy_seconds(self) -> float:
        """Seconds the child has spent waiting on bridge calls (running ones included)."""

        now = time.monotonic()
        with self._lock:
            return self._finished_seconds + sum(now - started for started in self._active.values())

    # ------------------------------------------------------------ serving

    @contextlib.contextmanager
    def serve(self, directory: Path) -> Iterator[None]:
        directory.mkdir(parents=True, exist_ok=True)
        stop = threading.Event()
        thread = threading.Thread(target=self._poll, args=(directory, stop), daemon=True, name="tool-bridge")
        thread.start()
        try:
            yield
        finally:
            stop.set()
            self.closed.set()  # in-flight agent runs wind down

    def _poll(self, directory: Path, stop: threading.Event) -> None:
        seen: set[str] = set()
        while not stop.is_set():
            try:
                for path in directory.glob("req-*.json"):
                    if path.name not in seen:
                        seen.add(path.name)
                        threading.Thread(target=self._serve_one, args=(path,), daemon=True, name="tool-bridge-call").start()
            except OSError:
                pass
            stop.wait(POLL_SECONDS)

    def _serve_one(self, path: Path) -> None:
        request_id = path.name[len("req-") : -len(".json")]
        procs.set_owner(self._owner, f"{self._label}/tool") if self._owner else None
        started = time.monotonic()
        with self._lock:
            self._active[request_id] = started
        try:
            try:
                request = json.loads(path.read_text(encoding="utf-8"))
                response: dict[str, Any] = {"result": self.handle(str(request.get("method") or ""), request.get("params") or {})}
            except Exception as exc:  # noqa: BLE001 - any failure goes back to the calling code as an exception
                response = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            with self._lock:
                self._active.pop(request_id, None)
                self._finished_seconds += time.monotonic() - started
        try:
            text = json.dumps(response, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as exc:
            text = json.dumps({"error": f"result is not JSON serializable: {exc}"})
        temporary = path.with_name(f"resp-{request_id}.json.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path.with_name(f"resp-{request_id}.json"))

    # ------------------------------------------------------------ the calls

    def handle(self, method: str, params: dict[str, Any]) -> Any:
        if method == "mcp.call":
            return self._mcp("callTool", params)
        if method == "mcp.list_tools":
            return self._mcp("listTools", params)
        if method == "mcp.list_servers":
            return self._mcp("listServers", params)
        if method == "llm.ask":
            return self._ask(params)
        if method == "agent.run":
            return self._agent(params)
        raise ValueError(f"unknown bridge method: {method}")

    def _mcp(self, action: str, params: dict[str, Any]) -> Any:
        from tools.executor import execute_raw_data_tool
        from tools.mcp import McpRequest
        from tools.request import ToolRequest

        security.require_tool("mcp")
        if not self.settings.mcp.can_model_call:
            raise ValueError("MCP is switched off in this conversation.")
        arguments = params.get("arguments")
        request = McpRequest(
            action=action,  # type: ignore[arg-type]
            server=str(params.get("server") or ""),
            tool=str(params.get("tool") or ""),
            arguments=arguments if isinstance(arguments, dict) else {},
        )
        value, _files = execute_raw_data_tool(ToolRequest(id="bridge", name="mcp", mcp_request=request), self.settings)
        return value

    def _ask(self, params: dict[str, Any]) -> str:
        from agent.llm import complete_chat_once

        prompt = str(params.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("ask_llm needs a prompt.")
        messages: list[dict[str, Any]] = []
        if str(params.get("system") or "").strip():
            messages.append({"role": "system", "content": str(params["system"])})
        messages.append({"role": "user", "content": prompt})
        max_tokens = int(params.get("max_tokens") or 0)
        extra: dict[str, Any] = {}
        effort = str(getattr(self.settings, "subagent_reasoning", "") or "default")
        if effort != "default":
            extra["reasoning_effort"] = effort
        reply = complete_chat_once(
            messages,
            model=str(params.get("model") or "")
            or str(getattr(self.settings, "subagent_model", "") or "")
            or str(getattr(self.settings, "model", "") or "")
            or None,
            max_tokens=max_tokens or None,
            **extra,
        )
        return str(reply.get("content") or "")

    def _agent(self, params: dict[str, Any]) -> str:
        from agent import subagents

        depth = int(getattr(self.settings, "bridge_depth", 0) or 0)
        if depth >= MAX_BRIDGE_DEPTH:
            raise ValueError(f"run_agent is nested too deeply (limit {MAX_BRIDGE_DEPTH}).")
        prompt = str(params.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("run_agent needs a prompt.")
        tools = str(params.get("tools") or "full")
        if tools not in {"full", "readonly"}:
            raise ValueError("run_agent tools must be full or readonly.")
        rounds = int(params.get("max_rounds") or 20)
        return subagents.run_inline_agent(
            self.settings,
            prompt,
            tools=tools,
            model=str(params.get("model") or ""),
            max_rounds=min(max(rounds, 1), 100),
            should_stop=lambda: self.closed.is_set() or bool(self.should_stop()),
        )[:RESULT_LIMIT_CHARS]


# ---------------------------------------------------------------- child side

BRIDGE_HELPER_SOURCE = r'''
"""Call MCP tools and the model from Python code run by the AI Agent."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

__all__ = ["call_mcp", "list_mcp_servers", "list_mcp_tools", "ask_llm", "run_agent"]


def _rpc(method, params, timeout):
    directory = os.environ.get("AI_AGENT_BRIDGE_DIR")
    if not directory:
        raise RuntimeError("calling MCP or the model from Python is not available in this context")
    base = Path(directory)
    request_id = uuid.uuid4().hex
    temporary = base / f"req-{request_id}.json.tmp"
    temporary.write_text(json.dumps({"method": method, "params": params}, ensure_ascii=False), encoding="utf-8")
    temporary.replace(base / f"req-{request_id}.json")
    response_path = base / f"resp-{request_id}.json"
    deadline = time.monotonic() + timeout
    delay = 0.02
    while not response_path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"{method} did not finish within {timeout} seconds")
        time.sleep(delay)
        delay = min(delay * 1.5, 0.25)
    response = json.loads(response_path.read_text(encoding="utf-8"))
    if "error" in response:
        raise RuntimeError(f"{method} failed: {response['error']}")
    return response.get("result")


def call_mcp(server, tool, arguments=None, timeout=1800):
    """Call one tool of a configured MCP server and return its data (structured content, parsed JSON or text)."""

    return _rpc("mcp.call", {"server": server, "tool": tool, "arguments": arguments or {}}, timeout)


def list_mcp_servers(timeout=120):
    return _rpc("mcp.list_servers", {}, timeout)


def list_mcp_tools(server, timeout=120):
    return _rpc("mcp.list_tools", {"server": server}, timeout)


def ask_llm(prompt, system="", model="", max_tokens=0, timeout=600):
    """One model completion with your own prompt; returns the text. No tools are involved."""

    return _rpc("llm.ask", {"prompt": prompt, "system": system, "model": model, "max_tokens": max_tokens}, timeout)


def run_agent(prompt, tools="full", model="", max_rounds=20, timeout=1800):
    """Run a full agent loop (with tools) on your own prompt and return its final answer.

    tools: "full" (code, files, MCP, web...) or "readonly".
    """

    return _rpc("agent.run", {"prompt": prompt, "tools": tools, "model": model, "max_rounds": max_rounds}, timeout)
'''
