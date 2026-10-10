import pytest
from fastapi.testclient import TestClient

from agent import server, session_store, titles


@pytest.fixture(autouse=True)
def isolated_conversations(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "CONVERSATION_ROOT", tmp_path / "conversations")


def make_state(turns):
    messages = [{"role": "system", "content": "rules"}]
    for n in range(turns):
        messages.append({"role": "user", "content": f"u{n}"})
        messages.append({"role": "assistant", "content": f"a{n}"})
    return {"messages": messages, "initialized": True, "tools_announced": True}


def seed(turns=3):
    for n in range(turns):
        session_store.save_turn(
            "c1",
            user_text=f"question {n}",
            turn_events=[{"type": "assistant", "text": f"answer {n}"}],
            state=make_state(n + 1),
        )


def test_truncate_cuts_messages_at_turn_boundary():
    seed()
    # events: u0 a0 u1 a1 u2 a2 -> edit the second question (index 2)
    conversation = session_store.truncate_conversation("c1", 2)
    assert [e["text"] for e in conversation["events"]] == ["question 0", "answer 0"]
    assert [m["content"] for m in conversation["state"]["messages"]] == ["rules", "u0", "a0"]
    assert conversation["state"]["question_pending"] is None


def test_truncate_first_turn_resets_tools_announced():
    seed()
    conversation = session_store.truncate_conversation("c1", 0)
    assert conversation["events"] == []
    assert conversation["state"]["tools_announced"] is False
    assert [m["role"] for m in conversation["state"]["messages"]] == ["system"]


def test_truncate_rejects_non_user_events():
    seed()
    with pytest.raises(ValueError):
        session_store.truncate_conversation("c1", 1)


def test_truncate_after_compression_falls_back_to_summary():
    seed()
    session_store.compress_conversation("c1")
    conversation = session_store.truncate_conversation("c1", 4)
    assert "question 1" in conversation["state"]["conversation_summary"]
    assert "question 2" not in conversation["state"]["conversation_summary"]


def test_branch_before_user_message_and_after_answer():
    seed()
    before = session_store.branch_conversation("c1", 2, before=True)
    assert [e["text"] for e in before["events"]] == ["question 0", "answer 0"]
    assert before["id"] != "c1"
    after = session_store.branch_conversation("c1", 3)
    assert len(after["events"]) == 4
    assert [m["content"] for m in after["state"]["messages"]] == ["rules", "u0", "a0", "u1", "a1"]


def test_branch_restores_plan_from_events():
    session_store.save_turn(
        "p1",
        user_text="plan it",
        turn_events=[
            {"type": "plan_ready", "name": "demo", "content": "# Plan", "status": "ready"},
            {"type": "assistant", "text": "ok"},
        ],
        state={"messages": [{"role": "system", "content": "rules"}, {"role": "user", "content": "u"}], "plan": {"name": "demo"}},
    )
    session_store.save_turn("p1", user_text="more", turn_events=[{"type": "assistant", "text": "x"}], state={})
    branch = session_store.branch_conversation("p1", 3)
    assert branch["state"]["plan"]["content"] == "# Plan"
    early = session_store.branch_conversation("p1", 0, before=True)
    assert "plan" not in early["state"]


def test_paged_conversation_returns_newest_first_window():
    seed()
    client = TestClient(server.app)
    page = client.get("/api/conversations/c1", params={"limit": 2, "state": "false"}).json()
    assert [e["text"] for e in page["events"]] == ["question 2", "answer 2"]
    assert page["event_offset"] == 4 and page["total_events"] == 6 and page["has_more"] is True
    assert "state" not in page
    older = client.get("/api/conversations/c1", params={"limit": 10, "before": 4}).json()
    assert older["event_offset"] == 0 and older["has_more"] is False
    assert len(older["events"]) == 4


def test_truncate_and_plan_delete_endpoints():
    seed()
    client = TestClient(server.app)
    assert client.post("/api/conversations/c1/truncate", json={"event_index": 1}).status_code == 400
    assert client.post("/api/conversations/c1/truncate", json={"event_index": 4}).status_code == 200
    assert len(client.get("/api/conversations/c1").json()["events"]) == 4
    conversation = session_store.read_conversation("c1")
    conversation["state"]["plan"] = {"name": "x", "content": "y", "status": "ready"}
    session_store.write_conversation(conversation)
    assert client.delete("/api/conversations/c1/plan").json()["state"]["plan"] is None


def test_first_turn_gets_readable_provisional_title_and_user_rename_wins():
    session_store.save_turn(
        "t1",
        user_text="@file:a.md 帮我分析一下这个季度的销售数据。然后给出建议",
        turn_events=[{"type": "assistant", "text": "好的，我来看看。"}],
        state={},
    )
    stored = session_store.read_conversation("t1")
    assert stored["title"] == "帮我分析一下这个季度的销售数据"
    assert stored["title_auto"] is True
    session_store.set_auto_title("t1", "季度销售分析")
    assert session_store.read_conversation("t1")["title"] == "季度销售分析"
    session_store.rename_conversation("t1", "我的名字")
    session_store.set_auto_title("t1", "不应覆盖")
    assert session_store.read_conversation("t1")["title"] == "我的名字"


def test_generate_title_cleans_model_output_and_falls_back(monkeypatch):
    monkeypatch.setattr(titles, "complete_chat_once", lambda *a, **k: {"content": '标题：「季度销售分析」\n多余'})
    assert titles.generate_title("分析销售", "好的") == "季度销售分析"

    def boom(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(titles, "complete_chat_once", boom)
    assert titles.generate_title("请解释 Python 的装饰器。谢谢", "") == "请解释 Python 的装饰器"


def test_plans_are_a_listable_deletable_data_kind(tmp_path, monkeypatch):
    plans = tmp_path / "plans"
    plans.mkdir()
    (plans / "demo.md").write_text("# plan", encoding="utf-8")
    monkeypatch.setattr(server, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(server, "ALLOWED_DATA_ROOTS", {"plans": plans})

    def no_reindex(*args, **kwargs):
        raise AssertionError("plans must not trigger a RAG reindex")

    monkeypatch.setattr(server, "refresh_rag", no_reindex)
    client = TestClient(server.app)
    items = client.get("/api/data/files", params={"kind": "plans"}).json()["items"]
    assert items == [{"path": "plans/demo.md", "name": "demo.md", "scope": "user", "writable": True}]
    assert client.delete("/api/data/file", params={"path": "plans/demo.md"}).json()["status"] == "deleted"
    assert not (plans / "demo.md").exists()


def test_since_returns_the_tail_from_an_index():
    seed()
    client = TestClient(server.app)
    page = client.get("/api/conversations/c1", params={"since": 3, "state": "false"}).json()
    assert page["event_offset"] == 3 and len(page["events"]) == 3 and page["has_more"] is True
