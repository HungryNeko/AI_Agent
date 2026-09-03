from fastapi.testclient import TestClient

from agent import server


def test_data_file_endpoints_use_allowed_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "PROJECT_ROOT", tmp_path)
    knowledge = tmp_path / "knowledge"
    memory = tmp_path / "memory"
    skills = tmp_path / "skills"
    for path in [knowledge, memory, skills]:
        path.mkdir()
    monkeypatch.setattr(
        server,
        "ALLOWED_DATA_ROOTS",
        {"knowledge": knowledge, "memory": memory, "skills": skills},
    )

    client = TestClient(server.app)
    response = client.put(
        "/api/data/file",
        json={"path": str(memory / "MEMORY.md"), "content": "remember compact prompts"},
    )
    assert response.status_code == 200

    list_response = client.get("/api/data/files", params={"kind": "memory"})
    assert list_response.status_code == 200
    assert list_response.json()["files"] == ["memory/MEMORY.md"]

    read_response = client.get("/api/data/file", params={"path": str(memory / "MEMORY.md")})
    assert read_response.json()["content"] == "remember compact prompts"


def test_skill_import_writes_skill_entrypoint(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "PROJECT_ROOT", tmp_path)
    skills = tmp_path / "skills"
    skills.mkdir()
    monkeypatch.setattr(server, "ALLOWED_DATA_ROOTS", {"skills": skills})

    client = TestClient(server.app)
    response = client.post("/api/skills/import", json={"name": "demo", "content": "# Demo"})

    assert response.status_code == 200
    assert response.json()["path"] == "skills/demo/SKILL.md"
    assert (skills / "demo" / "SKILL.md").read_text(encoding="utf-8") == "# Demo"


def test_mcp_server_form_endpoint_updates_config(tmp_path, monkeypatch):
    config = tmp_path / "servers.json"
    monkeypatch.setattr(server, "MCP_CONFIG_PATH", config)

    client = TestClient(server.app)
    response = client.put(
        "/api/mcp/servers/rent",
        json={
            "name": "rent",
            "enabled": True,
            "transport": "streamable_http",
            "url": "[http://127.0.0.1:5050/mcp](http://127.0.0.1:5050/mcp)",
            "timeout": 5,
            "sse_read_timeout": 300,
        },
    )

    assert response.status_code == 200
    data = client.get("/api/mcp/servers").json()
    assert data["servers"]["rent"] == {
        "enabled": True,
        "transport": "streamable_http",
        "url": "http://127.0.0.1:5050/mcp",
        "timeout": 5,
        "sse_read_timeout": 300,
    }


def test_mcp_server_endpoint_can_use_path_name(tmp_path, monkeypatch):
    config = tmp_path / "servers.json"
    monkeypatch.setattr(server, "MCP_CONFIG_PATH", config)

    client = TestClient(server.app)
    response = client.put(
        "/api/mcp/servers/rent",
        json={
            "transport": "streamable_http",
            "url": "http://127.0.0.1:5050/mcp",
            "timeout": 5,
            "sse_read_timeout": 300,
        },
    )

    assert response.status_code == 200
    assert client.get("/api/mcp/servers").json()["servers"]["rent"]["url"] == "http://127.0.0.1:5050/mcp"


def test_mcp_test_endpoint_uses_form_config(monkeypatch):
    seen = {}

    def fake_test_server_config(config, settings):
        seen["config"] = config
        return {"action": "testConnection", "response": '{"result":{"tools":[]}}'}

    monkeypatch.setattr(server.mcp_tool, "test_server_config", fake_test_server_config)

    client = TestClient(server.app)
    response = client.post(
        "/api/mcp/test",
        json={
            "transport": "streamable_http",
            "url": "http://127.0.0.1:5050/mcp",
            "timeout": 5,
            "sse_read_timeout": 300,
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert seen["config"]["url"] == "http://127.0.0.1:5050/mcp"


def test_artifact_endpoint_serves_python_run_files(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "PROJECT_ROOT", tmp_path)
    artifact_root = tmp_path / "backend" / "runtime" / "python_runs"
    artifact_root.mkdir(parents=True)
    image = artifact_root / "chart.png"
    image.write_bytes(b"fake-png")
    monkeypatch.setattr(server, "ARTIFACT_ROOTS", [artifact_root])

    client = TestClient(server.app)
    response = client.get("/api/artifact", params={"path": "backend/runtime/python_runs/chart.png"})

    assert response.status_code == 200
    assert response.content == b"fake-png"


def test_instruction_endpoint_reads_and_writes(monkeypatch, tmp_path):
    instruction = tmp_path / "instruction.md"
    monkeypatch.setattr(server, "load_instruction", lambda: instruction.read_text(encoding="utf-8") if instruction.exists() else "")

    def fake_save_instruction(content):
        instruction.write_text(content + "\n", encoding="utf-8")

    monkeypatch.setattr(server, "save_instruction", fake_save_instruction)
    client = TestClient(server.app)

    response = client.put("/api/instruction", json={"content": "Use concise answers."})

    assert response.status_code == 200
    assert client.get("/api/instruction").json()["content"].strip() == "Use concise answers."


def test_conversation_endpoints_use_json_store(tmp_path, monkeypatch):
    monkeypatch.setattr(server.session_store, "CONVERSATION_ROOT", tmp_path)
    conversation_id = server.session_store.create_conversation_id()
    server.session_store.save_turn(
        conversation_id,
        user_text="hello history",
        turn_events=[{"type": "assistant", "text": "hi"}],
        state={"messages": [{"role": "system", "content": "system"}]},
    )

    client = TestClient(server.app)
    listed = client.get("/api/conversations").json()["conversations"]
    compressed = client.post(f"/api/conversations/{conversation_id}/compress").json()

    assert listed[0]["id"] == conversation_id
    assert compressed["status"] == "compressed"
    assert compressed["state"]["conversation_summary"]


def test_conversation_can_be_renamed_and_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(server.session_store, "CONVERSATION_ROOT", tmp_path)
    conversation_id = server.session_store.create_conversation_id()
    server.session_store.save_turn(conversation_id, user_text="hello", turn_events=[], state={})
    client = TestClient(server.app)

    renamed = client.patch(f"/api/conversations/{conversation_id}", json={"title": "New title"})
    deleted = client.delete(f"/api/conversations/{conversation_id}")

    assert renamed.status_code == 200
    assert renamed.json()["title"] == "New title"
    assert deleted.status_code == 200
    assert server.session_store.list_conversations() == []


def test_config_endpoint_reads_and_writes(tmp_path, monkeypatch):
    config_path = tmp_path / "api_configs.json"
    config_path.write_text('{"default_provider":"demo","default_model":"m","providers":{"demo":{"base_url":"http://x","models":["m"]}}}', encoding="utf-8")
    monkeypatch.setattr(server, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(server, "CONFIG_PATH", config_path, raising=False)
    monkeypatch.setattr("agent.config.CONFIG_PATH", config_path)
    client = TestClient(server.app)

    response = client.put(
        "/api/config",
        json={"config": {"default_provider": "demo", "default_model": "m2", "providers": {"demo": {"base_url": "http://x", "models": ["m2"]}}}},
    )

    assert response.status_code == 200
    assert client.get("/api/config").json()["config"]["default_model"] == "m2"


def test_settings_endpoint_persists_json_config(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr("agent.app_settings.SETTINGS_PATH", settings_path)
    client = TestClient(server.app)

    response = client.patch(
        "/api/settings",
        json={"patch": {"ui": {"theme": "dark", "language": "en"}, "chat": {"max_tool_rounds": -1}}},
    )

    assert response.status_code == 200
    data = client.get("/api/settings").json()
    assert data["path"] == "data/settings.json"
    assert data["settings"]["ui"] == {"theme": "dark", "language": "en"}
    assert data["settings"]["chat"]["max_tool_rounds"] == -1
    assert settings_path.exists()


def test_rag_reindex_endpoint_reports_vector_status():
    client = TestClient(server.app)
    response = client.post("/api/rag/reindex")

    assert response.status_code == 200
    assert response.json()["index"] == "local-vector"


def test_upload_endpoint_stores_runtime_file(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "PROJECT_ROOT", tmp_path)
    client = TestClient(server.app)

    response = client.post("/api/uploads", params={"filename": "note.txt"}, content=b"hello")

    assert response.status_code == 200
    path = response.json()["path"]
    assert path == "backend/runtime/uploads/note.txt"
    assert (tmp_path / path).read_bytes() == b"hello"
