import json
from datetime import UTC, datetime

from agent import automation_runner
from tools.automation import AutomationRequest, automation_context, execute as execute_automation
from tools.settings import make_tool_settings


def test_runner_executes_once_reminder_and_records_result(tmp_path):
    automation_root = tmp_path / "automations"
    run_root = tmp_path / "runs"
    automation_root.mkdir()
    path = automation_root / "task.json"
    path.write_text(
        json.dumps(
            {
                "title": "Ping",
                "action": "reminder",
                "enabled": True,
                "prompt": "ping me",
                "schedule": {"kind": "once", "nextRunAt": "2026-09-03T10:00:00Z"},
            }
        ),
        encoding="utf-8",
    )

    records = automation_runner.run_due_automations(
        automation_root,
        run_root,
        now=datetime(2026, 9, 3, 10, 0, 1, tzinfo=UTC),
    )

    assert records[0]["status"] == "ok"
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["enabled"] is False
    assert updated["schedule"]["nextRunAt"] == ""
    assert list(run_root.glob("runs-*.jsonl"))


def test_runner_reschedules_interval(tmp_path):
    automation_root = tmp_path / "automations"
    run_root = tmp_path / "runs"
    automation_root.mkdir()
    path = automation_root / "interval.json"
    path.write_text(
        json.dumps(
            {
                "title": "Interval",
                "action": "reminder",
                "enabled": True,
                "prompt": "repeat",
                "schedule": {
                    "kind": "interval",
                    "nextRunAt": "2026-09-03T10:00:00Z",
                    "intervalSeconds": 30,
                },
            }
        ),
        encoding="utf-8",
    )

    automation_runner.run_due_automations(
        automation_root,
        run_root,
        now=datetime(2026, 9, 3, 10, 0, 1, tzinfo=UTC),
    )

    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["enabled"] is True
    assert updated["schedule"]["nextRunAt"] == "2026-09-03T10:00:31Z"
    assert updated["last_run"]["status"] == "ok"


def test_automation_context_updates_current_file_by_default(tmp_path):
    path = tmp_path / "self.json"
    path.write_text(
        json.dumps({"title": "Self", "action": "llm", "schedule": {"kind": "custom", "fibIndex": 2}}),
        encoding="utf-8",
    )
    settings = make_tool_settings(automation_mode="auto", automation_root=str(tmp_path))

    with automation_context(path):
        result = execute_automation(
            AutomationRequest(
                action="llm",
                title="Self",
                prompt="continue",
                schedule={"kind": "custom", "fibIndex": 3, "nextRunAt": "2026-09-03T10:05:00Z"},
            ),
            settings,
        )

    updated = json.loads(path.read_text(encoding="utf-8"))
    assert result["mode"] == "updated"
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert updated["schedule"]["fibIndex"] == 3
    assert updated["schedule"]["nextRunAt"] == "2026-09-03T10:05:00Z"


def test_llm_runner_uses_dedicated_conversation_and_preserves_self_update(tmp_path, monkeypatch):
    automation_root = tmp_path / "automations"
    run_root = tmp_path / "runs"
    conversation_root = tmp_path / "conversations"
    automation_root.mkdir()
    monkeypatch.setattr(automation_runner.session_store, "CONVERSATION_ROOT", conversation_root)
    path = automation_root / "fib.json"
    path.write_text(
        json.dumps(
            {
                "title": "Fib",
                "action": "llm",
                "enabled": True,
                "prompt": "update fibonacci schedule",
                "schedule": {"kind": "custom", "fibIndex": 1, "nextRunAt": "2026-09-03T10:00:00Z"},
            }
        ),
        encoding="utf-8",
    )

    def fake_stream_turn(state, message):
        assert "automationId: fib.json" in message
        execute_automation(
            AutomationRequest(
                action="llm",
                title="Fib",
                prompt="next run",
                schedule={"kind": "custom", "fibIndex": 2, "nextRunAt": "2026-09-03T10:01:00Z"},
            ),
            make_tool_settings(automation_mode="auto", automation_root=str(automation_root)),
        )
        yield {"type": "assistant", "text": "updated self", "state": state}

    monkeypatch.setattr("agent.graph.stream_turn", fake_stream_turn)
    monkeypatch.setattr("agent.graph.new_chat_state", lambda **kwargs: {"messages": []})

    records = automation_runner.run_due_automations(
        automation_root,
        run_root,
        now=datetime(2026, 9, 3, 10, 0, 1, tzinfo=UTC),
    )

    updated = json.loads(path.read_text(encoding="utf-8"))
    assert records[0]["status"] == "ok"
    assert len(list(automation_root.glob("*.json"))) == 1
    assert updated["conversation_id"] == "automation-fib"
    assert updated["schedule"]["fibIndex"] == 2
    assert updated["schedule"]["nextRunAt"] == "2026-09-03T10:01:00Z"
    conversation = automation_runner.session_store.read_conversation("automation-fib")
    assert conversation["title"] == "自动化: Fib"
    assert "updated self" in json.dumps(conversation["events"], ensure_ascii=False)
