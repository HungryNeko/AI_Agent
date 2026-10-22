from fastapi.testclient import TestClient

from agent import server
from agent.prompt_catalog import build_prompt_catalog
from prompts.system import BASE_SYSTEM_PROMPT
from prompts.tools import TOOL_REQUEST_FORMAT


def test_catalog_lists_every_prompt_group_with_both_languages():
    catalog = build_prompt_catalog()
    items = {entry["id"]: entry for entry in catalog["items"]}
    assert {entry["group"] for entry in catalog["items"]} == set(catalog["groups"])
    for entry in items.values():
        assert entry["title"] and entry["title_zh"] and entry["note"] and entry["note_zh"] and entry["content"]
    # the texts are the real constants, not copies
    assert items["system.base"]["content"] == BASE_SYSTEM_PROMPT
    assert items["system.tools"]["content"] == TOOL_REQUEST_FORMAT
    assert BASE_SYSTEM_PROMPT in items["system.assembled"]["content"]
    assert TOOL_REQUEST_FORMAT in items["system.assembled"]["content"]


def test_catalog_includes_the_native_tool_schemas():
    names = {entry["id"] for entry in build_prompt_catalog()["items"] if entry["group"] == "tools"}
    assert {"tool.python", "tool.fileEditor", "tool.agent", "tool.plan", "tool.question"} <= names
    python_schema = next(e for e in build_prompt_catalog()["items"] if e["id"] == "tool.python")
    assert '"code"' in python_schema["content"]


def test_prompts_endpoint_is_read_only():
    client = TestClient(server.app)
    response = client.get("/api/prompts")
    assert response.status_code == 200
    assert response.json()["items"]
    assert client.put("/api/prompts", json={}).status_code == 405
    assert client.post("/api/prompts", json={}).status_code == 405
