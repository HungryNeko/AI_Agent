import json
import sys

import httpx
import pytest

from tools import mcp
from tools.mcp import McpRequest
from tools.settings import McpSettings

ONE_PIXEL_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def write_config(path, command, args, *, enabled=True):
    path.write_text(
        json.dumps(
            {
                "servers": {
                    "local": {
                        "enabled": enabled,
                        "command": command,
                        "args": args,
                        "env": {},
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def write_server(path):
    path.write_text(
        """
import json
import sys

for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    if method == "initialize":
        print(json.dumps({"jsonrpc":"2.0","id":msg["id"],"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"test","version":"1"}}}), flush=True)
    elif method == "tools/list":
        print(json.dumps({"jsonrpc":"2.0","id":msg["id"],"result":{"tools":[{"name":"echo","description":"Echo text","inputSchema":{"type":"object"}}]}}), flush=True)
    elif method == "tools/call":
        params = msg.get("params", {})
        text = params.get("arguments", {}).get("text", "")
        print(json.dumps({"jsonrpc":"2.0","id":msg["id"],"result":{"content":[{"type":"text","text":text}]}}), flush=True)
""".strip(),
        encoding="utf-8",
    )


def test_mcp_lists_configured_servers(tmp_path):
    config = tmp_path / "servers.json"
    write_config(config, sys.executable, ["server.py"], enabled=False)
    settings = McpSettings(mode="auto", config_path=str(config))

    result = mcp.execute(McpRequest(action="listServers"), settings)

    assert result["servers"] == [
        {
            "name": "local",
            "enabled": False,
            "transport": "stdio",
            "url": "",
            "command": sys.executable,
            "args": ["server.py"],
        }
    ]


def test_mcp_lists_tools_and_calls_tool(tmp_path):
    server = tmp_path / "server.py"
    config = tmp_path / "servers.json"
    write_server(server)
    write_config(config, sys.executable, [str(server)])
    settings = McpSettings(mode="auto", config_path=str(config), timeout_seconds=5)

    tools_result = mcp.execute(McpRequest(action="listTools", server="local"), settings)
    call_result = mcp.execute(
        McpRequest(action="callTool", server="local", tool="echo", arguments={"text": "hello"}),
        settings,
    )

    assert "echo" in tools_result["response"]
    assert "hello" in call_result["response"]


def test_mcp_streamable_http_lists_tools(tmp_path, monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        data = json.loads(request.content.decode("utf-8"))
        method = data.get("method")
        if method == "initialize":
            return httpx.Response(
                200,
                headers={"content-type": "application/json", "mcp-session-id": "session-1"},
                json={
                    "jsonrpc": "2.0",
                    "id": data["id"],
                    "result": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "test", "version": "1"},
                    },
                },
            )
        if method == "notifications/initialized":
            assert request.headers["mcp-session-id"] == "session-1"
            return httpx.Response(202)
        if method == "tools/list":
            assert request.headers["authorization"] == "Bearer test"
            assert request.headers["mcp-session-id"] == "session-1"
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={
                    "jsonrpc": "2.0",
                    "id": data["id"],
                    "result": {
                        "tools": [
                            {
                                "name": "echo",
                                "description": "Echo text",
                                "inputSchema": {"type": "object"},
                            }
                        ]
                    },
                },
            )
        raise AssertionError(f"unexpected method: {method}")

    real_client = httpx.Client
    transport = httpx.MockTransport(handler)

    def make_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(mcp.httpx, "Client", make_client)
    config = tmp_path / "servers.json"
    config.write_text(
        json.dumps(
            {
                "servers": {
                    "remote": {
                        "enabled": True,
                        "transport": "streamable_http",
                        "url": "http://mcp.test/mcp",
                        "headers": {"Authorization": "Bearer test"},
                        "timeout": 5,
                        "sse_read_timeout": 300,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = mcp.execute(
        McpRequest(action="listTools", server="remote"),
        McpSettings(mode="auto", config_path=str(config)),
    )

    assert "echo" in result["response"]
    assert [json.loads(request.content.decode("utf-8")).get("method") for request in requests] == [
        "initialize",
        "notifications/initialized",
        "tools/list",
    ]


def test_mcp_loads_local_config_overlay(tmp_path):
    config = tmp_path / "servers.json"
    local_config = tmp_path / "servers.local.json"
    config.write_text(json.dumps({"servers": {"public": {"enabled": False, "command": "python", "args": []}}}), encoding="utf-8")
    local_config.write_text(json.dumps({"servers": {"private": {"enabled": False, "command": "node", "args": []}}}), encoding="utf-8")

    servers = mcp.load_config(McpSettings(mode="auto", config_path=str(config)))

    assert set(servers) == {"public", "private"}


def test_mcp_expands_env_placeholders_in_headers(monkeypatch):
    monkeypatch.setenv("RENT_MCP_TOKEN", "test-token")

    headers = mcp.build_http_headers(
        {
            "transport": "streamable_http",
            "url": "http://mcp.test/mcp",
            "headers": {"Authorization": "Bearer ${RENT_MCP_TOKEN}"},
        },
    )

    assert headers["Authorization"] == "Bearer test-token"


def test_mcp_rejects_disabled_server(tmp_path):
    config = tmp_path / "servers.json"
    write_config(config, sys.executable, ["server.py"], enabled=False)
    settings = McpSettings(mode="auto", config_path=str(config))

    with pytest.raises(ValueError, match="disabled"):
        mcp.execute(McpRequest(action="listTools", server="local"), settings)


def test_mcp_extracts_standard_image_content_to_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp, "project_root", lambda: tmp_path)
    response = {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "content": [
                {
                    "type": "image",
                    "data": ONE_PIXEL_PNG_BASE64,
                    "mimeType": "image/png",
                }
            ]
        },
    }

    files = mcp.extract_image_artifacts(response)

    assert len(files) == 1
    assert files[0].startswith("backend/runtime/mcp_artifacts/")
    assert files[0].endswith(".png")
    assert (tmp_path / files[0]).read_bytes().startswith(b"\x89PNG")


def test_mcp_extracts_body_base64_inside_text_json(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp, "project_root", lambda: tmp_path)
    text = json.dumps(
        {
            "status_code": 200,
            "headers": {"content-type": "image/png"},
            "body_base64": ONE_PIXEL_PNG_BASE64,
        }
    )
    response = {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {"content": [{"type": "text", "text": text}]},
    }

    files = mcp.extract_image_artifacts(response)
    redacted = mcp.redact_image_payloads(response)

    assert len(files) == 1
    assert (tmp_path / files[0]).is_file()
    assert ONE_PIXEL_PNG_BASE64 not in json.dumps(redacted)
    assert "image base64 omitted" in json.dumps(redacted)
