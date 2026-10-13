import json
import threading
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from agent import server, session_store, subagents, wakeups
from tools.request import build_openai_tools, parse_openai_tool_calls
from tools.settings import make_tool_settings
from tools.subagent import SubagentRequest, SubagentTask

REAL_SCHEDULE_WAKE = wakeups.schedule_wake


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "CONVERSATION_ROOT", tmp_path / "conversations")
    monkeypatch.setattr(subagents, "ROOT", tmp_path / "subagents")
    monkeypatch.setattr(subagents, "MANAGER", subagents.SubagentManager())
    wakeups._RUNTIMES.clear()
    monkeypatch.setattr(wakeups, "WAKE_DEBOUNCE_SECONDS", 0.0)
    woken = []
    monkeypatch.setattr(wakeups, "schedule_wake", lambda cid, item: woken.append((cid, item)))
    return woken


def settings(cid="c1", **kw):
    return replace(make_tool_settings(), conversation_id=cid, run_id="run1", **kw)


def make_fake(delay=0.0, barrier=None, fail_for=(), seen=None):
    def fake_stream(state, message):
        if seen is not None:
            seen.append((dict(state), message))
        if barrier is not None:
            barrier.wait(timeout=5)  # both agents must be running at the same time to pass
        yield {"type": "tool_call", "tool": "python", "text": "python: train"}
        time.sleep(delay)
        if any(name in message for name in fail_for):
            raise RuntimeError("boom")
        final = dict(state)
        final["messages"] = [*state.get("messages", []), {"role": "assistant", "content": "report"}]
        yield {"type": "assistant", "text": f"report for: {message[-30:]}", "state": final}

    return fake_stream


def spawn(tasks, wait, timeout=5.0, cid="c1"):
    request = SubagentRequest(action="spawn", tasks=tuple(tasks), wait=wait, timeout=timeout)
    return json.loads(subagents.handle(request, settings(cid)).split("\n", 1)[1])


def test_agents_really_run_in_parallel(monkeypatch):
    barrier = threading.Barrier(2)
    monkeypatch.setattr(subagents, "stream_turn", make_fake(barrier=barrier))
    started = time.monotonic()
    out = spawn([SubagentTask("a", "train model A"), SubagentTask("b", "train model B")], wait=True)
    assert time.monotonic() - started < 4
    assert [a["status"] for a in out["agents"]] == ["done", "done"]
    assert all("report for" in a["result"] for a in out["agents"])


def test_blocking_spawn_hands_result_over_without_waking(monkeypatch, isolated):
    monkeypatch.setattr(subagents, "stream_turn", make_fake())
    spawn([SubagentTask("a", "task")], wait=True)
    time.sleep(0.2)
    assert isolated == []


def test_background_agent_wakes_the_conversation_when_it_finishes(monkeypatch, isolated):
    monkeypatch.setattr(subagents, "stream_turn", make_fake(delay=0.3))
    out = spawn([SubagentTask("trainer", "train")], wait=False)
    assert out["agents"][0]["status"] in {"queued", "running"}
    deadline = time.monotonic() + 5
    while not isolated and time.monotonic() < deadline:
        time.sleep(0.05)
    cid, item = isolated[0]
    assert cid == "c1" and item["name"] == "trainer" and item["status"] == "done" and "report for" in item["result"]


def test_late_finish_after_a_timed_out_wait_still_wakes(monkeypatch, isolated):
    monkeypatch.setattr(subagents, "stream_turn", make_fake(delay=0.6))
    out = spawn([SubagentTask("slow", "x")], wait=True, timeout=0.2)
    assert out["agents"][0]["status"] != "done"
    deadline = time.monotonic() + 5
    while not isolated and time.monotonic() < deadline:
        time.sleep(0.05)
    assert isolated and isolated[0][1]["name"] == "slow"


def test_failure_is_reported_and_wakes(monkeypatch, isolated):
    monkeypatch.setattr(subagents, "stream_turn", make_fake(fail_for=("explode",)))
    out = spawn([SubagentTask("bad", "explode")], wait=True)
    assert out["agents"][0]["status"] == "failed" and "boom" in out["agents"][0]["error"]


def test_send_continues_with_the_same_context(monkeypatch):
    seen = []
    monkeypatch.setattr(subagents, "stream_turn", make_fake(seen=seen))
    first = spawn([SubagentTask("a", "do it")], wait=True)["agents"][0]
    request = SubagentRequest(action="send", ids=(first["id"],), message="also check X", wait=True, timeout=5.0)
    out = json.loads(subagents.handle(request, settings()).split("\n", 1)[1])
    assert out["agents"][0]["status"] == "done"
    state, message = seen[-1]
    assert message == "also check X"
    assert any(m.get("content") == "report" for m in state["messages"])  # earlier context carried over


def test_cancel_stops_a_running_agent(monkeypatch):
    def endless(state, message):
        for _ in range(200):
            time.sleep(0.02)
            yield {"type": "assistant_progress", "text": "working"}

    monkeypatch.setattr(subagents, "stream_turn", endless)
    agent = spawn([SubagentTask("a", "x")], wait=False)["agents"][0]
    time.sleep(0.1)
    subagents.handle(SubagentRequest(action="cancel", ids=(agent["id"],)), settings())
    deadline = time.monotonic() + 3
    while subagents.MANAGER.require("c1", agent["id"]).status != "cancelled" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert subagents.MANAGER.require("c1", agent["id"]).status == "cancelled"


def test_sub_agent_state_blocks_recursion_questions_and_respects_readonly(monkeypatch):
    seen = []
    monkeypatch.setattr(subagents, "stream_turn", make_fake(seen=seen))
    spawn([SubagentTask("r", "look", tools="readonly"), SubagentTask("w", "work", python_timeout=900.0)], wait=True)
    states = {m[-12:]: s for s, m in seen}
    readonly = next(s for s in states.values() if s["conversation_mode"] == "ask")
    full = next(s for s in states.values() if s["conversation_mode"] == "agent")
    assert set(full["blocked_tools"]) >= {"agent", "approval"}
    assert full["question_mode"] == "off" and full["python_timeout_seconds"] == 900.0
    assert readonly["conversation_mode"] == "ask"


def test_concurrency_limit_per_conversation(monkeypatch):
    monkeypatch.setattr(subagents, "stream_turn", make_fake(delay=0.5))
    spawn([SubagentTask(f"a{i}", "x") for i in range(subagents.MAX_PER_CONVERSATION)], wait=False)
    with pytest.raises(ValueError, match="at most"):
        spawn([SubagentTask("extra", "x")], wait=False)


def test_agent_tool_only_in_agent_mode_and_not_for_sub_agents():
    names = lambda s: [t["function"]["name"] for t in build_openai_tools(s)]
    assert "agent" in names(make_tool_settings())
    assert "agent" not in names(make_tool_settings(conversation_mode="ask"))
    assert "agent" not in names(make_tool_settings(conversation_mode="plan"))
    assert "agent" not in names(replace(make_tool_settings(), blocked_tools=("agent",)))


def test_request_parsing_and_validation():
    call = lambda args: {"tool_calls": [{"id": "1", "type": "function", "function": {"name": "agent", "arguments": json.dumps(args)}}]}
    listed = {"taskList": [{"id": "t1", "title": "train"}]}
    parsed = parse_openai_tool_calls(call({"action": "spawn", **listed, "tasks": [{"prompt": "p", "tools": "readonly", "maxRounds": 500}], "wait": False}), make_tool_settings())
    request = parsed[0].subagent_request
    assert request.wait is False and request.tasks[0].tools == "readonly" and request.tasks[0].max_rounds == 100
    assert request.tasks[0].name == "agent-1" and request.tasks[0].task_id == "t1"  # mapped by position
    assert [item.title for item in request.task_list] == ["train"]
    for bad in (
        {"action": "spawn"},
        {"action": "spawn", "tasks": [{"prompt": "p"}]},  # no task list stated first
        {"action": "spawn", "taskList": [], "tasks": [{"prompt": "p"}]},
        {"action": "spawn", **listed, "tasks": [{"prompt": "p", "taskId": "zzz"}]},
        {"action": "spawn", "taskList": ["a", "b"], "tasks": [{"prompt": "p"}]},  # cannot map by position
        {"action": "spawn", **listed, "tasks": [{"prompt": "p", "tools": "root"}]},
        {"action": "send", "id": "x"},
        {"action": "cancel"},
        {"action": "dance"},
    ):
        with pytest.raises(ValueError):
            parse_openai_tool_calls(call(bad), make_tool_settings())


def test_agents_are_persisted_and_interrupted_ones_are_marked(monkeypatch):
    monkeypatch.setattr(subagents, "stream_turn", make_fake())
    agent = spawn([SubagentTask("a", "x")], wait=True)["agents"][0]
    fresh = subagents.SubagentManager()
    loaded = fresh.require("c1", agent["id"])
    assert loaded.status == "done" and loaded.result
    path = subagents.ROOT / "c1" / f"{agent['id']}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["status"] = "running"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert subagents.SubagentManager().require("c1", agent["id"]).status == "interrupted"


# ------------------------------------------------------------------ wake-ups


def seed_conversation(cid="w1", **state_extra):
    session_store.save_turn(
        cid,
        user_text="start the training",
        turn_events=[{"type": "assistant", "text": "started"}],
        state={"messages": [{"role": "system", "content": "rules"}], "initialized": True, **state_extra},
    )


def patch_graph(monkeypatch, fake):
    from agent import graph

    monkeypatch.setattr(graph, "stream_turn", fake)


def test_wake_turn_runs_server_side_saves_and_publishes(monkeypatch):
    seen = {}

    def fake_turn(state, message):
        seen["message"] = message
        yield {"type": "assistant", "text": "Training finished: 93% accuracy.", "state": {**state, "response": "ok"}}

    patch_graph(monkeypatch, fake_turn)
    seed_conversation()
    events = []
    wakeups.subscribe("w1", events.append)
    item = {"id": "a1", "name": "trainer", "status": "done", "result": "93% accuracy"}
    assert wakeups.run_wake_turn("w1", [item], "owner") is True

    assert seen["message"].startswith("systemEvent:") and "93% accuracy" in seen["message"]
    assert [e["type"] for e in events] == ["wake_started", "user", "assistant", "wake_finished"]
    assert events[1]["source"] == "wake"
    saved = session_store.read_conversation("w1")
    assert saved["events"][-2]["source"] == "wake" and saved["events"][-1]["text"].startswith("Training finished")
    assert saved["state"]["rev"] == 1 and "run_id" not in saved["state"]


def test_wake_is_deferred_while_a_question_is_pending():
    seed_conversation(question_pending={"question": "which?"})
    assert wakeups.run_wake_turn("w1", [{"id": "x", "name": "n"}], "owner") is False


def test_schedule_wake_waits_for_the_running_user_turn_then_runs(monkeypatch):
    ran = threading.Event()

    def fake_turn(state, message):
        ran.set()
        yield {"type": "assistant", "text": "woken", "state": dict(state)}

    patch_graph(monkeypatch, fake_turn)
    monkeypatch.setattr(wakeups, "schedule_wake", REAL_SCHEDULE_WAKE)
    seed_conversation()
    gate = wakeups.runtime("w1").gate
    assert gate.acquire("user-turn")  # a user turn is running
    wakeups.schedule_wake("w1", {"id": "a", "name": "n", "status": "done", "result": "r"})
    time.sleep(0.4)
    assert not ran.is_set()  # blocked behind the user turn
    gate.release("user-turn")
    assert ran.wait(5)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and session_store.read_conversation("w1")["state"].get("rev") != 1:
        time.sleep(0.05)
    assert session_store.read_conversation("w1")["state"]["rev"] == 1


def test_gate_ignores_a_stale_owner(monkeypatch):
    gate = wakeups.TurnGate()
    assert gate.acquire("a")
    assert not gate.acquire("b", timeout=0.1)
    monkeypatch.setattr(wakeups, "STALE_GATE_SECONDS", 0.0)
    assert gate.acquire("b", timeout=0.1)


# ------------------------------------------------------------------ chat_stream integration


def sse_events(response):
    events = []
    for block in response.text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def test_chat_stream_stamps_revisions_and_adopts_newer_saved_history(monkeypatch):
    seen_states = []

    def fake_turn(state, message):
        seen_states.append(dict(state))
        yield {"type": "assistant", "text": "hi", "state": {**state, "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": message}]}}

    monkeypatch.setattr(server, "stream_turn", fake_turn)
    client = TestClient(server.app)
    body = {"message": "hello", "conversation_id": "cs1"}
    first = sse_events(client.post("/api/chat/stream", json=body))
    final = next(e for e in first if e["type"] == "assistant")
    assert final["state"]["rev"] == 1 and "run_id" not in final["state"]
    assert session_store.read_conversation("cs1")["state"]["rev"] == 1

    # a background wake turn advances the saved conversation to rev 2 with new history
    saved = session_store.read_conversation("cs1")
    saved["state"]["rev"] = 2
    saved["state"]["messages"] = [{"role": "system", "content": "s"}, {"role": "user", "content": "from wake"}]
    session_store.write_conversation(saved)

    # the client still holds rev 1: the server must not overwrite the wake turn's history
    stale = {"message": "next", "conversation_id": "cs1", "state": final["state"]}
    second = sse_events(client.post("/api/chat/stream", json=stale))
    assert any(m.get("content") == "from wake" for m in seen_states[-1]["messages"])
    assert next(e for e in second if e["type"] == "assistant")["state"]["rev"] == 3


# ------------------------------------------------------------------ task list, lifetimes, processes, file locks


def responsive(delay_steps=100, step=0.05):
    def fake_stream(state, message):
        for _ in range(delay_steps):
            time.sleep(step)
            yield {"type": "assistant_progress", "text": "working"}
        yield {"type": "assistant", "text": "finished", "state": dict(state)}

    return fake_stream


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def spawn_listed(tasks, wait, timeout=5.0, cid="c1", listing=None):
    from tools.subagent import TaskItem

    items = listing or (TaskItem("t1", "first"), TaskItem("t2", "second"))
    request = SubagentRequest(action="spawn", task_list=items, tasks=tuple(tasks), wait=wait, timeout=timeout)
    return json.loads(subagents.handle(request, settings(cid)).split("\n", 1)[1])


def test_task_list_is_recorded_with_statuses_and_published(monkeypatch):
    monkeypatch.setattr(subagents, "stream_turn", make_fake(delay=0.2))
    events = []
    wakeups.subscribe("c1", events.append)
    out = spawn_listed([SubagentTask("a", "work", task_id="t1")], wait=True)
    statuses = {t["id"]: t["status"] for t in out["taskList"]}
    assert statuses == {"t1": "done", "t2": "pending"}
    assert any(e["type"] == "tasklist_update" for e in events)
    assert {t["id"] for t in subagents.SubagentManager().task_list("c1")} == {"t1", "t2"}  # persisted


def test_foreground_agent_is_killed_when_its_turn_stops(monkeypatch):
    from agent import runs

    monkeypatch.setattr(subagents, "stream_turn", responsive())
    threading.Timer(0.3, lambda: runs.mark_cancelled_run("run1")).start()
    out = spawn_listed([SubagentTask("fg", "x", task_id="t1")], wait=True, timeout=30.0)
    assert out["agents"][0]["status"] in {"queued", "running"}  # the turn was stopped while waiting
    killed = subagents.MANAGER.close_run("c1", "run1")  # the turn is closing
    assert [a.name for a in killed] == ["fg"]
    agent_id = out["agents"][0]["id"]
    assert wait_for(lambda: subagents.MANAGER.require("c1", agent_id).status == "cancelled")
    agent = subagents.MANAGER.require("c1", agent_id)
    assert "parent turn closed" in agent.kill_reason and not agent.background
    runs.clear_cancelled_run("run1")


def test_background_agent_survives_its_turn_and_is_recorded(monkeypatch, isolated):
    monkeypatch.setattr(subagents, "stream_turn", responsive(delay_steps=10))
    out = spawn_listed([SubagentTask("bg", "x", task_id="t1")], wait=False)
    agent_id = out["agents"][0]["id"]
    assert out["agents"][0]["background"] is True
    assert subagents.MANAGER.close_run("c1", "run1") == []  # deliberate: not killed
    registry = json.loads((subagents.ROOT / "c1" / "_background.json").read_text(encoding="utf-8"))
    assert [a["id"] for a in registry["background"]] == [agent_id] and registry["killed"] == []
    assert wait_for(lambda: bool(isolated))
    assert subagents.MANAGER.require("c1", agent_id).status == "done"
    assert subagents.MANAGER.registry("c1")["background"][0]["name"] == "bg"


def test_timed_out_blocking_wait_hands_the_agent_over_to_the_background(monkeypatch, isolated):
    monkeypatch.setattr(subagents, "stream_turn", responsive(delay_steps=12))
    out = spawn_listed([SubagentTask("slow", "x", task_id="t1")], wait=True, timeout=0.2)
    agent_id = out["agents"][0]["id"]
    assert subagents.MANAGER.require("c1", agent_id).background is True  # recorded as background
    assert subagents.MANAGER.close_run("c1", "run1") == []
    assert wait_for(lambda: bool(isolated))  # still wakes the conversation when it finishes


def test_cancel_kills_the_agents_child_processes(monkeypatch):
    import sys

    from tools import procs

    started = threading.Event()

    def fake_stream(state, message):
        started.set()
        result = procs.run_tracked([sys.executable, "-c", "import time; time.sleep(60)"], timeout=120)
        yield {"type": "tool_call", "tool": "python", "text": f"python exited {result.returncode}"}
        yield {"type": "assistant", "text": "done", "state": dict(state)}

    monkeypatch.setattr(subagents, "stream_turn", fake_stream)
    out = spawn_listed([SubagentTask("trainer", "x", task_id="t1")], wait=False)
    agent = subagents.MANAGER.require("c1", out["agents"][0]["id"])
    assert started.wait(5) and wait_for(lambda: procs.running_for(agent.owner_key) == 1)
    started_at = time.monotonic()
    subagents.handle(SubagentRequest(action="cancel", ids=(agent.id,)), settings())
    assert wait_for(lambda: agent.status == "cancelled", timeout=10)
    assert time.monotonic() - started_at < 8  # not the 60s the process wanted
    assert procs.running_for(agent.owner_key) == 0
    # a stopped agent may not start new processes
    procs.set_owner(agent.owner_key)
    with pytest.raises(RuntimeError, match="cancelled"):
        procs.run_tracked([sys.executable, "-c", "pass"])
    procs.set_owner("")


def test_deleting_the_conversation_and_shutdown_kill_everything(monkeypatch):
    monkeypatch.setattr(subagents, "stream_turn", responsive())
    first = spawn_listed([SubagentTask("bg", "x", task_id="t1")], wait=False)["agents"][0]["id"]
    subagents.MANAGER.close_conversation("c1")
    assert not (subagents.ROOT / "c1").exists() and subagents.MANAGER.get("c1", first) is None

    second = spawn_listed([SubagentTask("bg2", "x", task_id="t1")], wait=False, cid="c2")["agents"][0]["id"]
    subagents.MANAGER.shutdown()
    saved = json.loads((subagents.ROOT / "c2" / f"{second}.json").read_text(encoding="utf-8"))
    assert saved["status"] == "interrupted" and saved["kill_reason"] == "server shutdown"
    registry = json.loads((subagents.ROOT / "c2" / "_background.json").read_text(encoding="utf-8"))
    assert registry["killed"][0]["name"] == "bg2"


def test_file_locks_share_readers_exclude_writers_and_name_the_holder(tmp_path):
    from tools import filelock

    target = tmp_path / "data.txt"
    target.write_text("x", encoding="utf-8")
    inside = threading.Event()
    release = threading.Event()

    def hold(owner, write):
        with filelock.locked(target, write=write, owner=owner):
            inside.set()
            release.wait(5)

    with filelock.locked(target, write=False, owner="reader-1"), filelock.locked(target, write=False, owner="reader-2"):
        with pytest.raises(filelock.FileLockTimeout, match="reader"):
            with filelock.locked(target, write=True, timeout=0.2, owner="writer"):
                pass
    thread = threading.Thread(target=hold, args=("trainer-a", True))
    thread.start()
    assert inside.wait(5)
    with pytest.raises(filelock.FileLockTimeout, match="trainer-a"):
        with filelock.locked(target, write=False, timeout=0.2, owner="trainer-b"):
            pass
    with pytest.raises(filelock.FileLockTimeout, match="writing"):
        with filelock.locked(target, write=True, timeout=0.2, owner="trainer-b"):
            pass
    release.set()
    thread.join()
    with filelock.locked(target, write=True, owner="trainer-b"):
        with filelock.locked(target, write=True, owner="trainer-b"):  # re-entrant for its holder
            pass
    assert filelock.snapshot() == []


def test_file_tools_take_the_locks(tmp_path, monkeypatch):
    from tools import filelock, fileEditor, fileReader
    from tools.fileEditor import FileEditRequest
    from tools.fileReader import FileReadRequest
    from tools.settings import FileEditorSettings, FileReaderSettings

    target = tmp_path / "notes.txt"
    target.write_text("hello", encoding="utf-8")
    monkeypatch.setattr(filelock, "DEFAULT_TIMEOUT_SECONDS", 0.2)
    editor = FileEditorSettings(root=str(tmp_path))
    reader = FileReaderSettings(root=str(tmp_path))
    inside, release = threading.Event(), threading.Event()

    def hold():
        with filelock.locked(target, write=True, owner="trainer-a"):
            inside.set()
            release.wait(5)

    thread = threading.Thread(target=hold)
    thread.start()
    assert inside.wait(5)
    with pytest.raises(ValueError, match="trainer-a"):
        fileEditor.execute(FileEditRequest(action="append", path="notes.txt", content="!"), editor)
    with pytest.raises(ValueError, match="trainer-a"):
        fileReader.read(FileReadRequest(path=str(target)), reader)
    assert target.read_text(encoding="utf-8") == "hello"  # untouched while locked
    release.set()
    thread.join()
    fileEditor.execute(FileEditRequest(action="append", path="notes.txt", content="!"), editor)
    assert target.read_text(encoding="utf-8").endswith("!")


# ------------------------------------------------------------------ lost updates and task-list ids


def lost_update_scenario(tmp_path, *, guarded=True, iterations=10):
    """Two agents each do `iterations` x (read -> +1 -> write back whole file) on one counter file."""

    from tools import fileEditor, procs
    from tools.fileEditor import FileEditRequest
    from tools.settings import FileEditorSettings

    settings_ = FileEditorSettings(root=str(tmp_path))
    target = tmp_path / "counter.txt"
    target.write_text("value: 0\n", encoding="utf-8")
    conflicts = []

    def agent(name):
        with procs.owner_scope(f"agent:test:{name}", name):
            done = 0
            while done < iterations:
                read = fileEditor.execute(FileEditRequest(action="read", path="counter.txt"), settings_)
                lines = read["content"].splitlines()
                value = int(lines[0].split(":")[1]) + 1
                time.sleep(0.01)  # both agents are now holding a snapshot of the same version
                content = "\n".join([f"value: {value}", *lines[1:], f"{name}-{done + 1}"]) + "\n"
                try:
                    fileEditor.execute(FileEditRequest(action="write", path="counter.txt", content=content, overwrite=True), settings_)
                    done += 1
                except ValueError as exc:
                    if "changed by" not in str(exc):
                        raise
                    conflicts.append(name)  # re-read and redo the step, as the error message says

    threads = [threading.Thread(target=agent, args=(n,)) for n in ("A", "B")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    lines = target.read_text(encoding="utf-8").splitlines()
    return int(lines[0].split(":")[1]), lines[1:], conflicts


def test_concurrent_read_modify_write_loses_no_updates(tmp_path):
    value, log, conflicts = lost_update_scenario(tmp_path)
    assert value == 20
    assert len(log) == 20 and sorted(entry for entry in log if entry.startswith("A")) == sorted(f"A-{i}" for i in range(1, 11))
    assert conflicts, "the interleaving must have produced conflicts that were detected and retried"


def test_overwrite_is_refused_when_the_file_changed_since_it_was_read(tmp_path):
    from tools import fileEditor, procs
    from tools.fileEditor import FileEditRequest
    from tools.settings import FileEditorSettings

    settings_ = FileEditorSettings(root=str(tmp_path))
    (tmp_path / "f.txt").write_text("one\n", encoding="utf-8")
    run = lambda owner, **kw: procs.owner_scope(owner, owner)

    with run("agent:x:a"):
        fileEditor.execute(FileEditRequest(action="read", path="f.txt"), settings_)
    with run("agent:x:b"):
        fileEditor.execute(FileEditRequest(action="read", path="f.txt"), settings_)
        fileEditor.execute(FileEditRequest(action="write", path="f.txt", content="two\n", overwrite=True), settings_)
    with run("agent:x:a"):
        with pytest.raises(ValueError, match="changed by agent:x:b"):
            fileEditor.execute(FileEditRequest(action="write", path="f.txt", content="mine\n", overwrite=True), settings_)
        # replace/append work on the current text, so they stay possible without re-reading
        fileEditor.execute(FileEditRequest(action="append", path="f.txt", content="three\n"), settings_)
        # ...but the agent's view is now stale, so a whole-file overwrite still needs a fresh read
        with pytest.raises(ValueError, match="changed by"):
            fileEditor.execute(FileEditRequest(action="write", path="f.txt", content="mine\n", overwrite=True), settings_)
        fileEditor.execute(FileEditRequest(action="read", path="f.txt"), settings_)
        fileEditor.execute(FileEditRequest(action="write", path="f.txt", content="merged\n", overwrite=True), settings_)
    assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "merged\n"


def test_single_agent_flows_are_not_slowed_down_by_the_check(tmp_path):
    from tools import fileEditor
    from tools.fileEditor import FileEditRequest
    from tools.settings import FileEditorSettings

    settings_ = FileEditorSettings(root=str(tmp_path))
    fileEditor.execute(FileEditRequest(action="write", path="n.txt", content="a\n"), settings_)
    fileEditor.execute(FileEditRequest(action="write", path="n.txt", content="b\n", overwrite=True), settings_)  # own writes
    fileEditor.execute(FileEditRequest(action="replace", path="n.txt", old_text="b", new_text="c"), settings_)
    fileEditor.execute(FileEditRequest(action="write", path="n.txt", content="d\n", overwrite=True), settings_)
    assert (tmp_path / "n.txt").read_text(encoding="utf-8") == "d\n"


def test_a_new_turn_gets_a_fresh_task_list_instead_of_reusing_old_ids(monkeypatch):
    from tools.subagent import TaskItem

    monkeypatch.setattr(subagents, "stream_turn", make_fake())
    first = replace(settings(), run_id="turn-1")
    second = replace(settings(), run_id="turn-2")
    request = lambda title: SubagentRequest(
        action="spawn", task_list=(TaskItem("t1", title),), tasks=(SubagentTask("a", "x", task_id="t1"),), wait=True, timeout=5.0
    )
    out1 = json.loads(subagents.handle(request("Check Los Angeles weather"), first).split("\n", 1)[1])
    assert [t["title"] for t in out1["taskList"]] == ["Check Los Angeles weather"]
    out2 = json.loads(subagents.handle(request("Concurrent write test"), second).split("\n", 1)[1])
    assert [(t["id"], t["title"], t["status"]) for t in out2["taskList"]] == [("t1", "Concurrent write test", "done")]
    # within one turn a repeated id takes the newest title and keeps accumulating
    out3 = json.loads(subagents.handle(request("Renamed in the same turn"), second).split("\n", 1)[1])
    assert [t["title"] for t in out3["taskList"]] == ["Renamed in the same turn"]
