import json

from agent import debug_log


def test_debug_log_writes_jsonl_and_redacts_sensitive_values(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_AGENT_LOG_DIR", str(tmp_path))

    debug_log.log_event(
        "test.event",
        headers={"Authorization": "Bearer secret"},
        body_base64="abc123",
        message="hello",
    )

    [path] = list(tmp_path.glob("agent-*.jsonl"))
    [line] = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(line)

    assert record["event"] == "test.event"
    assert record["headers"]["Authorization"] == "<redacted>"
    assert record["body_base64"] == "<omitted 6 chars>"
    assert record["message"] == "hello"
