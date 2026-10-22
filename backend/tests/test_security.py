import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent import app_settings, graph, netguard, server
from tools import executor, mcp, security
from tools import python as python_tool
from tools.fileEditor import FileEditRequest
from tools import fileEditor, fileReader
from tools.fileReader import FileReadRequest
from tools.request import ToolRequest, build_openai_tools
from tools.settings import FileEditorSettings, FileReaderSettings, PythonSettings, make_tool_settings


@pytest.fixture
def level(monkeypatch):
    """Pin the level without touching the real settings file."""

    holder = {"value": "L3"}
    monkeypatch.setattr(security, "current_level", lambda: holder["value"])
    return holder


@pytest.fixture
def settings_files(tmp_path, monkeypatch):
    monkeypatch.setattr(app_settings, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(app_settings, "local_settings_path", lambda: tmp_path / "settings.local.json")
    return tmp_path


# ------------------------------------------------------------------ the level itself


def test_levels_normalise_and_default_to_the_previous_behaviour():
    assert [security.normalize_level(v) for v in ("l1", " L2 ", "L3", "x", None)] == ["L1", "L2", "L3", "L3", "L3"]
    assert security.DEFAULT_LEVEL == "L3"


def test_unreadable_settings_fail_closed(monkeypatch):
    def broken():
        raise ValueError("bad json")

    monkeypatch.setattr(app_settings, "load_app_settings", broken)
    assert security.current_level() == "L1"


def test_l1_allows_only_harmless_tools():
    assert all(security.tool_allowed(t, "L1") for t in security.L1_TOOLS)
    for tool in ("python", "mcp", "fileEditor", "createTool", "custom__*", "automation", "settings", "agent", "parameterSave"):
        assert not security.tool_allowed(tool, "L1"), tool
        assert security.tool_allowed(tool, "L2") and security.tool_allowed(tool, "L3")


def test_the_model_only_sees_the_tools_of_the_level(level):
    names = lambda lv: [t["function"]["name"] for t in build_openai_tools(replace(make_tool_settings(), security_level=lv))]
    assert set(names("L1")) <= security.L1_TOOLS | {"model"}
    assert "python" in names("L2") and "python" in names("L3")
    assert "python" not in names("L1") and "fileEditor" not in names("L1") and "mcp" not in names("L1")


def test_the_graph_reads_the_level_from_the_server_not_from_client_state(level, monkeypatch):
    seen = []

    def fake_complete(messages, *, model=None, tools=None, **kw):
        seen.append([t["function"]["name"] for t in tools or []])
        return {"role": "assistant", "content": "ok"}

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete)
    level["value"] = "L1"
    state = graph.new_chat_state()
    state["security_level"] = "L3"  # a client claiming more rights changes nothing
    list(graph.stream_turn(state, "hi"))
    assert "python" not in seen[0] and "fileEditor" not in seen[0]


def test_lowering_the_level_blocks_a_tool_call_in_the_middle_of_a_turn(level):
    cfg = make_tool_settings()
    request = ToolRequest(id="1", name="python", code="print(1)")
    level["value"] = "L1"
    assert executor.execute_tool(request, cfg).startswith('toolError: "python is not available at security level L1')
    custom = ToolRequest(id="2", name="custom__anything", custom_arguments={})
    assert "not available at security level L1" in executor.execute_tool(custom, cfg)
    ingest = ToolRequest(id="3", name="rag", rag_request=__import__("tools.rag", fromlist=["RagRequest"]).RagRequest(action="ingest", path="x.md"))
    assert "rag ingest is not available" in executor.execute_tool(ingest, cfg)


# ------------------------------------------------------------------ nobody but the slider can change it


def test_the_level_survives_every_generic_settings_write(settings_files):
    app_settings.set_security_level("L1")
    assert app_settings.load_app_settings()["security"]["level"] == "L1"
    with pytest.raises(ValueError, match="security"):
        app_settings.patch_app_settings({"security": {"level": "L3"}})
    app_settings.save_app_settings({"chat": {"model": "x"}, "security": {"level": "L3"}})  # a full replace cannot raise it
    assert app_settings.load_app_settings()["security"]["level"] == "L1"
    from tools import appSettings

    with pytest.raises(ValueError):
        appSettings.execute(appSettings.SettingsRequest(action="update", patch={"security": {"level": "L3"}}))
    appSettings.execute(appSettings.SettingsRequest(action="replace", settings={"security": {"level": "L3"}}))
    assert app_settings.load_app_settings()["security"]["level"] == "L1"
    client = TestClient(server.app)
    assert client.patch("/api/settings", json={"patch": {"security": {"level": "L3"}}}).status_code == 400
    client.put("/api/settings", json={"settings": {"security": {"level": "L3"}}})
    assert app_settings.load_app_settings()["security"]["level"] == "L1"


def test_the_slider_endpoint_sets_and_reports_the_level(settings_files, monkeypatch):
    client = TestClient(server.app)
    assert client.put("/api/security/level", json={"level": "L1"}).json()["level"] == "L1"
    assert client.get("/api/security").json()["level"] == "L1"
    assert client.put("/api/security/level", json={"level": "L9"}).status_code == 422
    monkeypatch.setattr(security, "l2_status", lambda: {"available": False, "reason": "needs Linux"})
    refused = client.put("/api/security/level", json={"level": "L2"})
    assert refused.status_code == 400 and "needs Linux" in refused.json()["detail"]
    monkeypatch.setattr(security, "l2_status", lambda: {"available": True, "reason": ""})
    assert client.put("/api/security/level", json={"level": "L2"}).json()["level"] == "L2"


# ------------------------------------------------------------------ file tools


def make_project(tmp_path):
    for relative in ("data/knowledge/note.md", "data/settings.json", "backend/runtime/uploads/u1/a.txt", "src/app.py", "backend/tools/x.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content", encoding="utf-8")
    return tmp_path


def test_l1_reads_only_the_agents_own_data(tmp_path, level):
    root = make_project(tmp_path)
    cfg = FileReaderSettings(root=str(root))
    level["value"] = "L1"
    assert fileReader.read(FileReadRequest(path="data/knowledge/note.md"), cfg)["content"] == "content"
    assert fileReader.read(FileReadRequest(path="backend/runtime/uploads/u1/a.txt"), cfg)["content"] == "content"
    for refused in ("data/settings.json", "src/app.py", "backend/tools/x.py"):
        with pytest.raises(ValueError, match="L1"):
            fileReader.read(FileReadRequest(path=refused), cfg)
    level["value"] = "L2"
    assert fileReader.read(FileReadRequest(path="src/app.py"), cfg)["content"] == "content"


def test_code_and_configuration_cannot_be_edited_below_l3(tmp_path, level):
    root = make_project(tmp_path)
    cfg = FileEditorSettings(root=str(root))
    for lv in ("L2",):
        level["value"] = lv
        for blocked in ("backend/tools/x.py", "data/settings.json"):
            with pytest.raises(ValueError, match="below security level L3"):
                fileEditor.execute(FileEditRequest(action="append", path=blocked, content="!"), cfg)
        fileEditor.execute(FileEditRequest(action="append", path="src/app.py", content="!"), cfg)  # ordinary project files are fine
        assert fileEditor.execute(FileEditRequest(action="read", path="backend/tools/x.py"), cfg)["content"] == "content"
    level["value"] = "L3"
    fileEditor.execute(FileEditRequest(action="append", path="backend/tools/x.py", content="!"), cfg)
    assert (root / "backend/tools/x.py").read_text(encoding="utf-8") == "content!"


def test_editor_rules_do_not_miss_dotfiles_or_case_variants():
    assert security.editor_blocked(".github/workflows/deploy.yml")
    assert security.editor_blocked("Backend/Tools/x.py") and security.editor_blocked("./data/settings.local.json")
    assert not security.editor_blocked("frontend/src/App.jsx") and not security.editor_blocked("notes.md")
    assert security.read_allowed_at_l1("./data/knowledge/a.md") and not security.read_allowed_at_l1("data/knowledgeX/a.md")


# ------------------------------------------------------------------ running code


def python_cfg(tmp_path):
    return PythonSettings(mode="auto", artifact_root=str(tmp_path / "runs"), timeout_seconds=60.0)


def test_no_code_runs_at_l1(tmp_path, level):
    level["value"] = "L1"
    with pytest.raises(security.SecurityError, match="L1"):
        python_tool.run("print(1)", python_cfg(tmp_path))
    with pytest.raises(security.SecurityError):
        mcp.run_session({"transport": "stdio", "command": "x"}, make_tool_settings().mcp, method="tools/list", params={})
    assert python_tool.run.__name__ == "run"


def test_l2_without_the_low_user_refuses_instead_of_running_unrestricted(tmp_path, level, monkeypatch):
    level["value"] = "L2"
    monkeypatch.setattr(security, "low_user", lambda: None)
    with pytest.raises(security.SecurityError, match="L2 cannot run code here"):
        python_tool.run("print(1)", python_cfg(tmp_path))


def test_l2_starts_code_as_the_unprivileged_user_with_a_clean_environment(tmp_path, level, monkeypatch):
    level["value"] = "L2"
    monkeypatch.setattr(security, "low_user", lambda: (4242, 4343))
    handed = []
    monkeypatch.setattr(security, "hand_over", lambda *paths: handed.extend(paths))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret")
    captured = {}

    def fake_run(args, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(args, 0, "ok", "")

    monkeypatch.setattr(python_tool.procs, "run_tracked", fake_run)
    result = python_tool.run("print(1)", python_cfg(tmp_path), workspace_key="conv1")
    assert result["return_code"] == 0
    assert (captured["user"], captured["group"], captured["extra_groups"]) == (4242, 4343, [])
    assert "sk-secret" not in json.dumps(captured["env"]) and captured["env"]["HOME"] == result["artifact_dir"]
    assert Path(result["artifact_dir"]) in handed and any("_workspaces" in str(p) for p in handed)


def test_l3_runs_code_as_the_backend_itself(tmp_path, level):
    level["value"] = "L3"
    assert python_tool.run("print(40 + 2)", python_cfg(tmp_path))["stdout"].strip() == "42"
    assert security.code_launch_options() == {}


def test_stdio_mcp_servers_get_no_copy_of_the_backends_secrets_when_unprivileged(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret")
    monkeypatch.setenv("PATH", "/usr/bin")
    server_def = {"command": "x", "env": {"RENT_TOKEN": "${DEEPSEEK_API_KEY}", "PLAIN": "1"}}
    scrubbed = mcp.build_env({"command": "x", "env": {"PLAIN": "1"}}, scrub=True)
    assert "DEEPSEEK_API_KEY" not in scrubbed and scrubbed["PLAIN"] == "1" and scrubbed["PATH"] == "/usr/bin"
    assert mcp.build_env(server_def, scrub=False)["RENT_TOKEN"] == "sk-secret"  # configured values still work


def test_the_bridge_cannot_reach_mcp_at_l1(level):
    from tools.bridge import ToolBridge

    level["value"] = "L1"
    with pytest.raises(security.SecurityError):
        ToolBridge(make_tool_settings())._mcp("listServers", {})


# ------------------------------------------------------------------ the local API guard

# Lines as the kernel writes them: addresses are little-endian words, the uid is the 8th column.
TCP_HEADER = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"


def tcp_row(local, remote, uid):
    return f"   0: {local} {remote} 01 00000000:00000000 00:00000000 00000000  {uid}        0 12345 1 0 100 4 0\n"


def test_proc_address_spelling():
    assert netguard._proc_hex("127.0.0.1", 8010)[0] == "0100007F:1F4A"
    assert netguard._proc_hex("127.0.0.1", 8010)[1] == "0000000000000000FFFF00000100007F:1F4A"
    assert netguard._proc_hex("::1", 80)[0] == "00000000000000000000000001000000:0050"


def test_the_owner_of_a_loopback_connection_is_found(tmp_path):
    (tmp_path / "tcp").write_text(
        TCP_HEADER
        + tcp_row("0100007F:1F4A", "0100007F:D431", 0)  # the backend's end of the connection
        + tcp_row("0100007F:D431", "0100007F:1F4A", 10001),  # the client's end, owned by agentcode
    )
    (tmp_path / "tcp6").write_text(TCP_HEADER)
    assert netguard.peer_uid(("127.0.0.1", 0xD431), ("127.0.0.1", 8010), tmp_path) == 10001
    assert netguard.peer_uid(("127.0.0.1", 40000), ("127.0.0.1", 8010), tmp_path) is None  # not in the table


def test_ipv6_mapped_connections_are_found_too(tmp_path):
    (tmp_path / "tcp").write_text(TCP_HEADER)
    (tmp_path / "tcp6").write_text(
        TCP_HEADER + tcp_row("0000000000000000FFFF00000100007F:D431", "0000000000000000FFFF00000100007F:1F4A", 10001)
    )
    assert netguard.peer_uid(("127.0.0.1", 0xD431), ("127.0.0.1", 8010), tmp_path) == 10001


def test_only_the_low_users_local_connections_are_refused(tmp_path, monkeypatch):
    (tmp_path / "tcp").write_text(
        TCP_HEADER
        + tcp_row("0100007F:D431", "0100007F:1F4A", 10001)
        + tcp_row("0100007F:D432", "0100007F:1F4A", 0)
    )
    (tmp_path / "tcp6").write_text(TCP_HEADER)
    server_end = ("127.0.0.1", 8010)
    assert netguard.should_refuse(("127.0.0.1", 0xD431), server_end, 10001, tmp_path) is True
    assert netguard.should_refuse(("127.0.0.1", 0xD432), server_end, 10001, tmp_path) is False  # the backend's own user
    assert netguard.should_refuse(("172.20.0.5", 5000), server_end, 10001, tmp_path) is False  # the gateway container
    assert netguard.should_refuse(("127.0.0.1", 0xD431), server_end, None, tmp_path) is False  # no low user: nothing to guard


def test_requests_from_agent_code_get_403_and_everyone_else_is_served(monkeypatch):
    client = TestClient(server.app)
    monkeypatch.setattr(netguard, "refusal", lambda scope: True)
    refused = client.get("/api/security")
    assert refused.status_code == 403 and "agent code" in refused.json()["detail"]
    assert client.put("/api/security/level", json={"level": "L3"}).status_code == 403
    monkeypatch.setattr(netguard, "refusal", lambda scope: False)
    assert client.get("/api/security").status_code == 200


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc/net/tcp")
def test_a_real_loopback_connection_is_attributed_to_its_owner():
    import socket

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    client = socket.create_connection(listener.getsockname())
    accepted, peer = listener.accept()
    try:
        assert netguard.peer_uid(peer, listener.getsockname()) == os.getuid()
    finally:
        client.close(), accepted.close(), listener.close()


# ------------------------------------------------------------------ setup helpers


def test_startup_helpers_are_harmless_where_they_cannot_apply(tmp_path, monkeypatch):
    if os.name != "posix" or os.geteuid() != 0:
        assert security.ensure_low_user() is False
        assert security.l2_status()["available"] is False and security.l2_status()["reason"]
    (tmp_path / ".env").write_text("KEY=1", encoding="utf-8")
    assert ".env" in security.harden_permissions(tmp_path)
    if os.name == "posix":
        assert (tmp_path / ".env").stat().st_mode & 0o077 == 0
