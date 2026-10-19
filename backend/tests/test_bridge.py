import json
import subprocess
from dataclasses import replace

import pytest

from agent import llm, subagents
from tools import createTool, executor
from tools import python as python_tool
from tools.bridge import ToolBridge
from tools.createTool import CreateToolRequest
from tools.request import ToolRequest
from tools.settings import make_tool_settings


def settings_for(tmp_path, *, timeout=60.0, depth=0, mcp_mode="auto"):
    base = make_tool_settings(mcp_mode=mcp_mode, python_timeout_seconds=timeout)
    python = replace(base.python, artifact_root=str(tmp_path / "runs"))
    return replace(base, python=python, conversation_id="conv1", model="p:m", bridge_depth=depth)


def run(code, settings):
    return python_tool.run(code, settings.python, workspace_key=settings.conversation_id, bridge=ToolBridge(settings))


@pytest.fixture
def fake_mcp(monkeypatch):
    calls = []

    def fake_raw(request, settings):
        item = request.mcp_request
        calls.append((item.action, item.server, item.tool, item.arguments))
        if item.action == "listTools":
            return {"tools": [{"name": "get_rent"}]}, []
        if item.action == "listServers":
            return ["Rent"], []
        return {"echo": item.arguments, "tool": item.tool}, []

    monkeypatch.setattr(executor, "execute_raw_data_tool", fake_raw)
    return calls


def test_python_code_can_call_mcp_tools_in_sequence(tmp_path, fake_mcp):
    result = run(
        "rows = call_mcp('Rent', 'get_rent', {'room': 3})\n"
        "saved = call_mcp('Rent', 'update_rent', {'room': 3, 'rent': rows['echo']['room'] * 1000})\n"
        "print(list_mcp_servers(), list_mcp_tools('Rent'), saved['echo'])",
        settings_for(tmp_path),
    )
    assert result["return_code"] == 0, result["stderr"]
    assert [call[:3] for call in fake_mcp] == [
        ("callTool", "Rent", "get_rent"),
        ("callTool", "Rent", "update_rent"),
        ("listServers", "", ""),
        ("listTools", "Rent", ""),
    ]
    assert "{'room': 3, 'rent': 3000}" in result["stdout"]


def test_mcp_switched_off_fails_inside_the_code_not_silently(tmp_path, fake_mcp):
    result = run("try:\n    call_mcp('Rent', 'x')\nexcept RuntimeError as exc:\n    print('ERR', exc)", settings_for(tmp_path, mcp_mode="off"))
    assert "ERR" in result["stdout"] and "switched off" in result["stdout"] and fake_mcp == []


def test_python_code_can_ask_the_model_with_its_own_prompt(tmp_path, monkeypatch):
    seen = []

    def fake_complete(messages, *, model=None, tools=None, max_tokens=None, **kwargs):
        seen.append((messages, model, max_tokens))
        return {"role": "assistant", "content": "SUMMARY: " + messages[-1]["content"][:20]}

    monkeypatch.setattr(llm, "complete_chat_once", fake_complete)
    result = run("print(ask_llm('Summarise the lease terms', system='Be brief', max_tokens=50))", settings_for(tmp_path))
    assert "SUMMARY: Summarise the leas" in result["stdout"]
    messages, model, max_tokens = seen[0]
    assert [m["role"] for m in messages] == ["system", "user"] and model == "p:m" and max_tokens == 50


def test_python_code_can_run_a_full_agent_on_its_own_prompt(tmp_path, monkeypatch):
    seen = []

    def fake_stream(state, message):
        seen.append((state, message))
        yield {"type": "tool_call", "text": "tool: webSearch"}
        yield {"type": "assistant", "text": "AGENT ANSWER", "state": dict(state)}

    monkeypatch.setattr(subagents, "stream_turn", fake_stream)
    result = run("print(run_agent('Research the market and report', tools='readonly', max_rounds=7))", settings_for(tmp_path))
    assert "AGENT ANSWER" in result["stdout"], result["stderr"]
    state, message = seen[0]
    assert message == "Research the market and report"
    assert state["conversation_mode"] == "ask" and state["max_tool_rounds"] == 7 and state["bridge_depth"] == 1
    assert {"agent", "approval"} <= set(state["blocked_tools"]) and state["question_mode"] == "off"


def test_nested_agent_calls_are_limited(tmp_path, monkeypatch):
    monkeypatch.setattr(subagents, "stream_turn", lambda state, message: iter([{"type": "assistant", "text": "x", "state": {}}]))
    result = run("try:\n    run_agent('again')\nexcept RuntimeError as exc:\n    print('ERR', exc)", settings_for(tmp_path, depth=2))
    assert "nested too deeply" in result["stdout"]


def test_time_spent_waiting_for_the_model_does_not_count_against_the_python_timeout(tmp_path, monkeypatch):
    import time

    def slow(messages, **kwargs):
        time.sleep(3.0)
        return {"role": "assistant", "content": "late but fine"}

    monkeypatch.setattr(llm, "complete_chat_once", slow)
    cfg = settings_for(tmp_path, timeout=1.5)
    result = run("print(ask_llm('slow question'))", cfg)
    assert result["return_code"] == 0 and "late but fine" in result["stdout"]
    # ...while code that just computes for too long is still stopped by the timeout
    with pytest.raises(subprocess.TimeoutExpired):
        run("import time\ntime.sleep(10)", cfg)


def test_calls_from_several_threads_are_served_at_once(tmp_path, monkeypatch):
    import threading

    barrier = threading.Barrier(2)

    def meet(messages, **kwargs):
        barrier.wait(timeout=5)  # both calls must be in flight together
        return {"role": "assistant", "content": messages[-1]["content"]}

    monkeypatch.setattr(llm, "complete_chat_once", meet)
    result = run(
        "import threading\nout = {}\n"
        "def go(name):\n    out[name] = ask_llm(name)\n"
        "threads = [threading.Thread(target=go, args=(n,)) for n in ('a', 'b')]\n"
        "[t.start() for t in threads]\n[t.join() for t in threads]\nprint(sorted(out.items()))",
        settings_for(tmp_path),
    )
    assert "[('a', 'a'), ('b', 'b')]" in result["stdout"], result["stderr"]


def test_without_a_bridge_the_helpers_explain_themselves(tmp_path):
    cfg = settings_for(tmp_path)
    result = python_tool.run("try:\n    ask_llm('x')\nexcept RuntimeError as exc:\n    print('ERR', exc)", cfg.python)
    assert "not available in this context" in result["stdout"]


def test_a_custom_tool_chains_mcp_calls_and_an_llm_step(tmp_path, monkeypatch, fake_mcp):
    monkeypatch.setattr(createTool, "CUSTOM_TOOL_ROOT", tmp_path / "custom_tools")
    monkeypatch.setattr(
        llm,
        "complete_chat_once",
        lambda messages, **kw: {"role": "assistant", "content": "NOTICE for " + messages[-1]["content"][-6:]},
    )
    createTool.execute(
        CreateToolRequest(
            action="save",
            name="raise_rent",
            description="Read a room, raise its rent, and draft a notice",
            parameters={"type": "object", "properties": {"room": {"type": "integer"}}, "required": ["room"]},
            code=(
                "def run(arguments):\n"
                "    room = arguments['room']\n"
                "    current = call_mcp('Rent', 'get_rent', {'room': room})\n"
                "    call_mcp('Rent', 'update_rent', {'room': room, 'rent': 1200})\n"
                "    notice = ask_llm(f'Write a rent notice for room {room}')\n"
                "    return {'before': current['echo'], 'notice': notice}\n"
            ),
        )
    )
    text = executor.execute_tool(
        ToolRequest(id="1", name="custom__raise_rent", custom_arguments={"room": 7}), settings_for(tmp_path)
    )
    payload = json.loads(text.split("\n", 1)[1])
    assert payload["return_code"] == 0, payload["stderr"]
    assert json.loads(payload["stdout"]) == {"before": {"room": 7}, "notice": "NOTICE for room 7"}
    assert [call[2] for call in fake_mcp] == ["get_rent", "update_rent"]
