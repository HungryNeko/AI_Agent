"""Minimal MCP client for configured servers."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx

from agent.debug_log import log_event, log_exception
from tools.settings import McpSettings

McpAction = Literal["listServers", "listTools", "callTool"]
PROTOCOL_VERSION = "2025-03-26"
IMAGE_BASE64_KEYS = {"body_base64", "image_base64", "b64_json"}
IMAGE_EXTENSIONS_BY_CONTENT_TYPE = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
}
ENV_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class McpRequest:
    action: McpAction
    server: str = ""
    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


def execute(request: McpRequest, settings: McpSettings) -> dict[str, Any]:
    log_event("mcp.request", action=request.action, server=request.server, tool=request.tool, arguments=request.arguments)
    try:
        if not settings.can_model_call:
            raise ValueError("mcp is disabled for the model.")
        if request.action == "listServers":
            result = list_servers(settings)
        elif request.action == "listTools":
            result = list_tools(request.server, settings)
        elif request.action == "callTool":
            result = call_tool(request.server, request.tool, request.arguments, settings)
        else:
            raise ValueError(f"unknown mcp action: {request.action}")
    except Exception as exc:
        log_exception(
            "mcp.error",
            exc,
            action=request.action,
            server=request.server,
            tool=request.tool,
            arguments=request.arguments,
        )
        raise
    log_event("mcp.response", action=request.action, server=request.server, tool=request.tool, response=result)
    return result


def list_servers(settings: McpSettings) -> dict[str, Any]:
    config = load_config(settings)
    servers = []
    for name, server in sorted(config.items()):
        servers.append(
            {
                "name": name,
                "enabled": bool(server.get("enabled", True)),
                "transport": server.get("transport", "stdio"),
                "url": server.get("url", ""),
                "command": server.get("command", ""),
                "args": server.get("args", []),
            }
        )
    return {"action": "listServers", "servers": servers}


def list_tools(server_name: str, settings: McpSettings) -> dict[str, Any]:
    server = require_server(server_name, settings)
    response = run_session(server, settings, method="tools/list", params={})
    return {"action": "listTools", "server": server_name, "response": trim_json(response, settings.max_output_chars)}


def call_tool(server_name: str, tool_name: str, arguments: dict[str, Any], settings: McpSettings) -> dict[str, Any]:
    if not tool_name.strip():
        raise ValueError("mcp callTool requires tool.")
    server = require_server(server_name, settings)
    response = run_session(
        server,
        settings,
        method="tools/call",
        params={"name": tool_name, "arguments": arguments},
    )
    files = extract_image_artifacts(response)
    response_for_model = redact_image_payloads(response) if files else response
    return {
        "action": "callTool",
        "server": server_name,
        "tool": tool_name,
        "response": trim_json(response_for_model, settings.max_output_chars),
        "files": files,
    }


def load_config(settings: McpSettings) -> dict[str, dict[str, Any]]:
    path = resolve_project_path(settings.config_path)
    data: dict[str, Any] = {"servers": {}}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    local_path = path.with_name(f"{path.stem}.local{path.suffix}")
    if local_path.exists():
        data = merge_mcp_config(data, json.loads(local_path.read_text(encoding="utf-8")))
    servers = data.get("servers") if isinstance(data, dict) else None
    if not isinstance(servers, dict):
        raise ValueError("mcp config must contain an object field `servers`.")
    clean: dict[str, dict[str, Any]] = {}
    for name, server in servers.items():
        if isinstance(name, str) and isinstance(server, dict):
            clean[name] = server
    return clean


def merge_mcp_config(base: dict[str, Any], local: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    base_servers = base.get("servers") if isinstance(base.get("servers"), dict) else {}
    local_servers = local.get("servers") if isinstance(local.get("servers"), dict) else {}
    merged["servers"] = {**base_servers, **local_servers}
    return merged


def require_server(name: str, settings: McpSettings) -> dict[str, Any]:
    server_name = name.strip()
    if not server_name:
        raise ValueError("mcp action requires server.")
    servers = load_config(settings)
    if server_name not in servers:
        raise ValueError(f"mcp server is not configured: {server_name}")
    server = servers[server_name]
    if not server.get("enabled", True):
        raise ValueError(f"mcp server is disabled: {server_name}")
    transport = str(server.get("transport") or "stdio")
    if transport == "streamable_http":
        if not isinstance(server.get("url"), str) or not server["url"].strip():
            raise ValueError(f"mcp server has no url: {server_name}")
        return server
    if transport != "stdio":
        raise ValueError(f"unsupported mcp transport for {server_name}: {transport}")
    if not isinstance(server.get("command"), str) or not server["command"].strip():
        raise ValueError(f"mcp server has no command: {server_name}")
    return server


def run_session(server: dict[str, Any], settings: McpSettings, *, method: str, params: dict[str, Any]) -> dict[str, Any]:
    if str(server.get("transport") or "stdio") == "streamable_http":
        return run_streamable_http_session(server, settings, method=method, params=params)
    return run_stdio_session(server, settings, method=method, params=params)


def test_server_config(server: dict[str, Any], settings: McpSettings) -> dict[str, Any]:
    log_event("mcp.test.request", server=server)
    if not server.get("enabled", True):
        raise ValueError("mcp server is disabled.")
    try:
        response = run_session(server, settings, method="tools/list", params={})
    except Exception as exc:
        log_exception("mcp.test.error", exc, server=server)
        raise
    result = {"action": "testConnection", "response": trim_json(response, settings.max_output_chars)}
    log_event("mcp.test.response", server=server, response=result)
    return result


def run_stdio_session(
    server: dict[str, Any],
    settings: McpSettings,
    *,
    method: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    process = subprocess.Popen(
        build_command(server),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        env=build_env(server),
    )
    try:
        send_message(
            process,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "ai-agent-backend", "version": "0.1.0"},
                },
            },
        )
        initialize_response = read_response(process, 1, timeout_seconds=settings.timeout_seconds)
        if "error" in initialize_response:
            return initialize_response
        send_message(process, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        send_message(process, {"jsonrpc": "2.0", "id": 2, "method": method, "params": params})
        return read_response(process, 2, timeout_seconds=settings.timeout_seconds)
    finally:
        terminate_process(process)


def run_streamable_http_session(
    server: dict[str, Any],
    settings: McpSettings,
    *,
    method: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    session_id = ""
    with httpx.Client(timeout=http_timeout(server, settings), follow_redirects=True) as client:
        initialize_response = post_json_rpc(
            client,
            server,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": str(server.get("protocol_version") or PROTOCOL_VERSION),
                    "capabilities": {},
                    "clientInfo": {"name": "ai-agent-backend", "version": "0.1.0"},
                },
            },
            expected_id=1,
        )
        session_id = str(initialize_response.pop("_session_id", ""))
        if "error" in initialize_response:
            return initialize_response
        post_json_rpc(
            client,
            server,
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            session_id=session_id,
        )
        return post_json_rpc(
            client,
            server,
            {"jsonrpc": "2.0", "id": 2, "method": method, "params": params},
            expected_id=2,
            session_id=session_id,
        )


def post_json_rpc(
    client: httpx.Client,
    server: dict[str, Any],
    message: dict[str, Any],
    *,
    expected_id: int | None = None,
    session_id: str = "",
) -> dict[str, Any]:
    with client.stream(
        "POST",
        str(server["url"]).strip(),
        headers=build_http_headers(server, session_id=session_id),
        json=message,
    ) as response:
        response.raise_for_status()
        returned_session_id = response.headers.get("mcp-session-id", "")
        if expected_id is None:
            return {"status": response.status_code}
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            parsed = read_sse_response(response, expected_id)
        else:
            parsed = read_json_response(response, expected_id)
        if returned_session_id:
            parsed["_session_id"] = returned_session_id
        return parsed


def build_http_headers(server: dict[str, Any], *, session_id: str = "") -> dict[str, str]:
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "MCP-Protocol-Version": str(server.get("protocol_version") or PROTOCOL_VERSION),
    }
    configured = server.get("headers", {})
    if configured is not None:
        if not isinstance(configured, dict):
            raise ValueError("mcp server headers must be an object.")
        for key, value in configured.items():
            if isinstance(key, str) and isinstance(value, str) and key.strip():
                headers[key.strip()] = expand_env_placeholders(value)
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    return headers


def read_json_response(response: httpx.Response, expected_id: int) -> dict[str, Any]:
    text = response.read().decode(response.encoding or "utf-8", errors="replace").strip()
    if not text:
        raise TimeoutError(f"mcp streamable_http returned an empty response for id {expected_id}.")
    for message in parse_json_messages(text):
        if message.get("id") == expected_id:
            return message
    raise TimeoutError(f"mcp streamable_http did not return response id {expected_id}. body={text[:1000]}")


def read_sse_response(response: httpx.Response, expected_id: int) -> dict[str, Any]:
    data_lines: list[str] = []
    for line in response.iter_lines():
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())
            continue
        if line.strip():
            continue
        if not data_lines:
            continue
        parsed = parse_sse_data(data_lines)
        data_lines = []
        if parsed.get("id") == expected_id:
            return parsed
    if data_lines:
        parsed = parse_sse_data(data_lines)
        if parsed.get("id") == expected_id:
            return parsed
    raise TimeoutError(f"mcp streamable_http did not return response id {expected_id}.")


def parse_json_messages(text: str) -> list[dict[str, Any]]:
    data = json.loads(text)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def parse_sse_data(data_lines: list[str]) -> dict[str, Any]:
    try:
        parsed = json.loads("\n".join(data_lines))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"mcp streamable_http returned invalid SSE JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("mcp streamable_http SSE data must be a JSON object.")
    return parsed


def http_timeout(server: dict[str, Any], settings: McpSettings) -> httpx.Timeout:
    connect_timeout = float(server.get("timeout") or settings.timeout_seconds)
    read_timeout = float(server.get("sse_read_timeout") or settings.timeout_seconds)
    return httpx.Timeout(
        connect=connect_timeout,
        read=read_timeout,
        write=connect_timeout,
        pool=connect_timeout,
    )


def build_command(server: dict[str, Any]) -> list[str]:
    command = str(server["command"]).strip()
    args = server.get("args", [])
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise ValueError("mcp server args must be a list of strings.")
    return [command, *args]


def build_env(server: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    configured = server.get("env", {})
    if configured is None:
        return env
    if not isinstance(configured, dict):
        raise ValueError("mcp server env must be an object.")
    for key, value in configured.items():
        if isinstance(key, str) and isinstance(value, str):
            env[key] = expand_env_placeholders(value)
    return env


def expand_env_placeholders(value: str) -> str:
    return ENV_PLACEHOLDER_RE.sub(lambda match: os.getenv(match.group(1), ""), value)


def send_message(process: subprocess.Popen[str], message: dict[str, Any]) -> None:
    if process.stdin is None:
        raise RuntimeError("mcp process stdin is closed.")
    process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
    process.stdin.flush()


def read_response(process: subprocess.Popen[str], expected_id: int, *, timeout_seconds: float) -> dict[str, Any]:
    if process.stdout is None:
        raise RuntimeError("mcp process stdout is closed.")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("id") == expected_id:
            return message
    stderr = ""
    if process.stderr is not None:
        try:
            stderr = process.stderr.read()[:1000]
        except OSError:
            stderr = ""
    raise TimeoutError(f"mcp server did not return response id {expected_id}. stderr={stderr}")


def terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()


def extract_image_artifacts(response: dict[str, Any]) -> list[str]:
    candidates: list[tuple[str, str]] = []
    collect_image_candidates(response, candidates)
    if not candidates:
        return []

    files: list[str] = []
    seen: set[str] = set()
    output_dir: Path | None = None
    for index, (encoded, content_type) in enumerate(candidates, start=1):
        image_bytes = decode_base64_data(encoded)
        if not image_bytes:
            continue
        digest = hashlib.sha256(image_bytes).hexdigest()[:12]
        if digest in seen:
            continue
        seen.add(digest)
        if output_dir is None:
            output_dir = make_mcp_artifact_dir()
        extension = image_extension(content_type, image_bytes)
        path = output_dir / f"image_{index:02d}_{digest}{extension}"
        path.write_bytes(image_bytes)
        files.append(relative_to_project(path))
    return files


def collect_image_candidates(value: Any, candidates: list[tuple[str, str]]) -> None:
    if isinstance(value, dict):
        content_type = dict_content_type(value)
        if str(value.get("type") or "").lower() == "image" and isinstance(value.get("data"), str):
            candidates.append((value["data"], content_type))
        for key in IMAGE_BASE64_KEYS:
            if isinstance(value.get(key), str) and is_image_candidate(value, content_type):
                candidates.append((value[key], content_type))
        for item in value.values():
            collect_image_candidates(item, candidates)
        return

    if isinstance(value, list):
        for item in value:
            collect_image_candidates(item, candidates)
        return

    if isinstance(value, str) and "base64" in value.lower():
        parsed = parse_possible_json(value)
        if parsed is not None:
            collect_image_candidates(parsed, candidates)


def dict_content_type(value: dict[str, Any]) -> str:
    for key in ("content_type", "contentType", "mime_type", "mimeType"):
        content_type = value.get(key)
        if isinstance(content_type, str) and content_type.strip():
            return content_type.strip()
    headers = value.get("headers")
    if isinstance(headers, dict):
        for key, header_value in headers.items():
            if isinstance(key, str) and key.lower() == "content-type" and isinstance(header_value, str):
                return header_value.strip()
    return ""


def is_image_candidate(value: dict[str, Any], content_type: str) -> bool:
    if content_type.lower().split(";", 1)[0].strip().startswith("image/"):
        return True
    for key in ("filename", "file_name", "name", "path", "url"):
        text = value.get(key)
        if isinstance(text, str) and image_extension_from_name(text):
            return True
    return False


def parse_possible_json(text: str) -> Any | None:
    clean = text.strip()
    if not clean.startswith(("{", "[")):
        return None
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return None


def decode_base64_data(encoded: str) -> bytes:
    data = encoded.strip()
    if data.startswith("data:image/") and "," in data:
        data = data.split(",", 1)[1]
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return b""


def image_extension(content_type: str, image_bytes: bytes) -> str:
    media_type = content_type.lower().split(";", 1)[0].strip()
    if media_type in IMAGE_EXTENSIONS_BY_CONTENT_TYPE:
        return IMAGE_EXTENSIONS_BY_CONTENT_TYPE[media_type]
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return ".gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    if image_bytes.lstrip().startswith(b"<svg"):
        return ".svg"
    return ".bin"


def image_extension_from_name(name: str) -> str:
    suffix = Path(name.split("?", 1)[0]).suffix.lower()
    return suffix if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"} else ""


def redact_image_payloads(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        is_image_content = str(value.get("type") or "").lower() == "image"
        for key, item in value.items():
            if key in IMAGE_BASE64_KEYS and isinstance(item, str):
                redacted[key] = "<image base64 omitted; see files>"
            elif key == "data" and is_image_content and isinstance(item, str):
                redacted[key] = "<image base64 omitted; see files>"
            else:
                redacted[key] = redact_image_payloads(item)
        return redacted
    if isinstance(value, list):
        return [redact_image_payloads(item) for item in value]
    if isinstance(value, str) and "base64" in value.lower():
        parsed = parse_possible_json(value)
        if parsed is not None:
            return json.dumps(redact_image_payloads(parsed), ensure_ascii=False)
    return value


def make_mcp_artifact_dir() -> Path:
    base_dir = project_root() / "backend" / "runtime" / "mcp_artifacts"
    suffix = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000:06d}"
    run_dir = base_dir / f"run_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def relative_to_project(path: Path) -> str:
    return str(path.resolve().relative_to(project_root())).replace("\\", "/")


def trim_json(value: dict[str, Any], max_chars: int) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...mcp response truncated..."


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
