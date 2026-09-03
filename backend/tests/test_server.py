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
