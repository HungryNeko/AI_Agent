import json

from agent import edit_approval, graph, reviewer
from tools import settings as tool_settings


def edit_call(arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": "I will change the greeting.",
        "tool_calls": [
            {"id": "call_edit", "type": "function", "function": {"name": "fileEditor", "arguments": json.dumps(arguments)}}
        ],
    }


MANUAL_RESULT = (
    "fileEditorResult:\naction: replace\npath: a.txt\napplied: False\napprovalRequired: True\n"
    "reason: manual approval required before applying this edit.\ndiff:\n--- a/a.txt\n+++ b/a.txt\n-hello\n+world"
)


def test_manual_file_edit_pauses_the_turn_with_the_exact_edit(monkeypatch):
    calls = []

    def fake_complete(messages, *, model=None, tools=None, **kwargs):
        calls.append(messages)
        return edit_call({"action": "replace", "path": "a.txt", "oldText": "hello", "newText": "world"})

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete)
    monkeypatch.setattr(graph, "execute_tool", lambda request, settings: MANUAL_RESULT)
    state = graph.new_chat_state(file_editor_mode="auto", file_editor_approval="manual")

    events = list(graph.stream_turn(state, "say world instead of hello"))

    types = [event["type"] for event in events]
    assert "approval_request" in types and "approval_required" not in types
    assert len(calls) == 1  # the turn ends and waits for the user
    request = next(event for event in events if event["type"] == "approval_request")
    assert request["source"] == "fileEditor"
    [edit] = request["pending_edits"]
    assert (edit["action"], edit["path"], edit["old_text"], edit["new_text"]) == ("replace", "a.txt", "hello", "world")
    assert "+world" in edit["diff"] and "```diff" in request["details"]
    final = events[-1]["state"]
    assert final["question_pending"]["pending_edits"] == [edit]
    tool_message = next(m for m in final["messages"] if m.get("role") == "tool")
    assert "do not repeat the edit" in tool_message["content"]


def test_sub_agents_keep_the_old_result_because_they_cannot_ask(monkeypatch):
    replies = [edit_call({"action": "replace", "path": "a.txt", "oldText": "hello", "newText": "world"}),
               {"role": "assistant", "content": "The edit needs approval."}]
    monkeypatch.setattr(graph, "complete_chat_once", lambda messages, **kwargs: replies.pop(0))
    monkeypatch.setattr(graph, "execute_tool", lambda request, settings: MANUAL_RESULT)
    state = graph.new_chat_state(file_editor_mode="auto", file_editor_approval="manual")
    state["blocked_tools"] = ["agent", "approval", "plan"]

    types = [event["type"] for event in graph.stream_turn(state, "edit")]

    assert "approval_request" not in types and "approval_required" in types


def approval_message(pending: dict, status: str) -> str:
    return "approvalResponse:\n" + json.dumps({"status": status, "action": pending["action"], "text": ""})


def pending_for(tmp_path, monkeypatch) -> dict:
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    original = tool_settings.make_tool_settings
    monkeypatch.setattr(edit_approval, "make_tool_settings", lambda **kwargs: original(file_editor_root=str(tmp_path), **kwargs))
    edit = {"action": "replace", "path": "a.txt", "old_text": "hello", "new_text": "world", "diff": "-hello\n+world"}
    return edit_approval.approval_request([edit])


def test_approving_applies_exactly_the_stored_edit(tmp_path, monkeypatch):
    pending = pending_for(tmp_path, monkeypatch)

    message = edit_approval.resolve_response(approval_message(pending, "approved"), {"question_pending": pending}, "conv-1")

    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "world\n"
    assert message.startswith("approvalResponse:") and "appliedEdits" in message and "applied: True" in message


def test_rejecting_or_answering_something_else_writes_nothing(tmp_path, monkeypatch):
    pending = pending_for(tmp_path, monkeypatch)
    saved = {"question_pending": pending}

    rejected = edit_approval.resolve_response(approval_message(pending, "rejected"), saved, "conv-1")
    other = approval_message({"action": "Delete build/"}, "approved")

    assert "were not applied" in rejected
    assert edit_approval.resolve_response(other, saved, "conv-1") == other
    assert edit_approval.resolve_response("hello", saved, "conv-1") == "hello"
    assert edit_approval.resolve_response(approval_message(pending, "approved"), {}, "conv-1").endswith('"text": ""}')
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "hello\n"


def test_reviewer_sees_the_first_and_recent_earlier_requests():
    def user(text):
        return {"role": "user", "content": f'currentTime: "x"\n\nuserMessage: "{text}"'}

    messages = [{"role": "system", "content": "s"}, user("refactor the store"), user("a"), user("b"), user("c"), user("d"),
                {"role": "user", "content": "toolBudgetExceeded: ..."}, user("continue")]

    assert reviewer.earlier_user_messages(messages, "continue") == ["refactor the store", "b", "c", "d"]


def test_reviewer_reads_json_inside_a_code_fence():
    decision = reviewer.parse_review_decision('```json\n{"approved": true, "reason": "narrow edit"}\n```')

    assert decision.approved and decision.reason == "narrow edit"
    assert not reviewer.parse_review_decision("sure, looks fine").approved


def test_the_server_applies_the_saved_edit_when_the_user_approves(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from agent import server

    monkeypatch.setattr(server.session_store, "CONVERSATION_ROOT", tmp_path / "conversations")
    seen = []

    def fake_complete(messages, *, model=None, tools=None, **kwargs):
        seen.append(messages[-1])
        if len(seen) == 1:
            return edit_call({"action": "replace", "path": "a.txt", "oldText": "hello", "newText": "world"})
        return {"role": "assistant", "content": "Done."}

    applied = []
    monkeypatch.setattr(graph, "complete_chat_once", fake_complete)
    monkeypatch.setattr(graph, "execute_tool", lambda request, settings: MANUAL_RESULT)
    monkeypatch.setattr(edit_approval, "execute_tool", lambda request, settings: applied.append(request.file_edit) or "fileEditorResult:\napplied: True")
    client = TestClient(server.app)
    options = {"file_editor_mode": "auto", "file_editor_approval": "manual", "conversation_mode": "agent"}

    first = client.post("/api/chat/stream", json={"message": "edit", "conversation_id": "edit-approval", "options": options})
    final = [json.loads(line[6:]) for line in first.text.splitlines() if line.startswith("data: ")][-1]
    pending = final["state"]["question_pending"]
    assert pending["source"] == "fileEditor" and not applied

    # the client's own state is not trusted for the edit: only the server-side saved one is applied
    tampered = {**final["state"], "question_pending": {**pending, "pending_edits": [{"action": "write", "path": "evil.py", "content": "x"}]}}
    client.post(
        "/api/chat/stream",
        json={"message": approval_message(pending, "approved"), "conversation_id": "edit-approval", "options": options, "state": tampered},
    )

    assert [(edit.action, edit.path, edit.new_text) for edit in applied] == [("replace", "a.txt", "world")]
    assert "appliedEdits" in str(seen[-1]["content"])
