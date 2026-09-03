"""Small FastAPI server for the React test frontend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent.config import load_config
from agent.graph import ChatState, stream_turn
from tools import mcp as mcp_tool
from tools.mcp import McpRequest
from tools.settings import McpSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
ALLOWED_DATA_ROOTS = {
    "knowledge": DATA_ROOT / "knowledge",
    "memory": DATA_ROOT / "memory",
    "skills": DATA_ROOT / "skills",
}
MCP_CONFIG_PATH = DATA_ROOT / "mcp" / "servers.json"
TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml"}
ARTIFACT_ROOTS = [
    PROJECT_ROOT / "backend" / "runtime" / "python_runs",
    PROJECT_ROOT / "backend" / "runtime" / "mcp_artifacts",
]

app = FastAPI(title="AI Agent Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatOptions(BaseModel):
    model: str | None = None
    system_prompt: str | None = None
    web_search_mode: Literal["off", "auto"] = "auto"
    web_search_provider: Literal["duckduckgo", "searxng", "tavily"] = "duckduckgo"
    rag_mode: Literal["off", "on", "auto"] = "auto"
    curl_mode: Literal["off", "auto"] = "auto"
    python_mode: Literal["off", "auto"] = "auto"
    file_editor_mode: Literal["off", "auto"] = "auto"
    file_editor_approval: Literal["readOnly", "manual", "auto"] = "auto"
    mcp_mode: Literal["off", "auto"] = "auto"
    max_tool_rounds: int = Field(default=20, ge=1, le=20)


class ChatRequest(BaseModel):
    message: str
    state: dict[str, Any] | None = None
    options: ChatOptions = ChatOptions()


class DataFilePayload(BaseModel):
    path: str
    content: str


class SkillPayload(BaseModel):
    name: str
    content: str


class McpServerPayload(BaseModel):
    name: str = ""
    enabled: bool = True
    transport: Literal["streamable_http", "stdio"] = "streamable_http"
    url: str = ""
    headers: dict[str, str] = {}
    timeout: float = 5
    sse_read_timeout: float = 300
    command: str = ""
    args: list[str] = []
    env: dict[str, str] = {}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/version")
def version() -> dict[str, str]:
    return {
        "status": "ok",
        "backend": "ai-agent-backend",
        "mcpSchema": "streamable_http",
        "source": str(Path(__file__).resolve()),
    }


@app.get("/api/artifact")
def artifact(path: str = Query(...)) -> FileResponse:
    resolved = resolve_artifact_path(path)
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(resolved)


@app.get("/api/models")
def models() -> dict[str, Any]:
    config = load_config()
    providers = config.get("providers", {})
    items: list[dict[str, str]] = []
    if isinstance(providers, dict):
        for provider_name, provider in providers.items():
            if not isinstance(provider, dict):
                continue
            raw_models = provider.get("models", [])
            if not isinstance(raw_models, list):
                continue
            for item in raw_models:
                if isinstance(item, str):
                    items.append({"label": f"{provider_name}:{item}", "value": f"{provider_name}:{item}"})
                elif isinstance(item, dict) and isinstance(item.get("id"), str):
                    alias = item.get("alias") if isinstance(item.get("alias"), str) else item["id"]
                    items.append({"label": f"{provider_name}:{alias}", "value": f"{provider_name}:{alias}"})
    return {
        "defaultProvider": config.get("default_provider"),
        "defaultModel": config.get("default_model"),
        "models": items,
    }


@app.post("/api/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    state = build_chat_state(payload)

    def events():
        try:
            for event in stream_turn(state, payload.message):
                yield encode_sse(event)
        except Exception as exc:
            yield encode_sse({"type": "error", "text": str(exc)})

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/api/data/files")
def list_data_files(kind: Literal["knowledge", "memory", "skills"] = Query(...)) -> dict[str, Any]:
    root = allowed_root(kind)
    root.mkdir(parents=True, exist_ok=True)
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            files.append(relative_to_project(path))
    return {"kind": kind, "files": files}


@app.get("/api/data/file")
def read_data_file(path: str = Query(...)) -> dict[str, str]:
    resolved = resolve_data_path(path)
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return {"path": relative_to_project(resolved), "content": resolved.read_text(encoding="utf-8", errors="replace")}


@app.put("/api/data/file")
def write_data_file(payload: DataFilePayload) -> dict[str, str]:
    resolved = resolve_data_path(payload.path, allow_missing=True)
    if resolved.suffix.lower() not in TEXT_SUFFIXES:
        raise HTTPException(status_code=400, detail="unsupported text file type")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(payload.content, encoding="utf-8")
    return {"path": relative_to_project(resolved), "status": "saved"}


@app.post("/api/skills/import")
def import_skill(payload: SkillPayload) -> dict[str, str]:
    name = clean_single_name(payload.name)
    path = ALLOWED_DATA_ROOTS["skills"] / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.content, encoding="utf-8")
    return {"path": relative_to_project(path), "status": "saved"}


@app.get("/api/mcp/servers")
def get_mcp_servers() -> dict[str, Any]:
    return load_mcp_config()


@app.post("/api/mcp/test")
def test_mcp_config(payload: McpServerPayload) -> dict[str, Any]:
    try:
        result = mcp_tool.test_server_config(
            mcp_server_config_from_payload(payload),
            McpSettings(mode="auto", config_path=str(MCP_CONFIG_PATH)),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"status": "ok", "result": result}


@app.post("/api/mcp/servers/{name}/test")
def test_saved_mcp_server(name: str) -> dict[str, Any]:
    server_name = clean_single_name(name)
    try:
        result = mcp_tool.execute(
            McpRequest(action="listTools", server=server_name),
            McpSettings(mode="auto", config_path=str(MCP_CONFIG_PATH)),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"status": "ok", "result": result}


@app.put("/api/mcp/servers/{name}")
def upsert_mcp_server(name: str, payload: McpServerPayload) -> dict[str, Any]:
    path_name = clean_single_name(name)
    body_name = clean_single_name(payload.name or name)
    if path_name != body_name:
        raise HTTPException(status_code=400, detail="path name and body name must match")
    config = load_mcp_config()
    config.setdefault("servers", {})[body_name] = mcp_server_config_from_payload(payload)
    save_mcp_config(config)
    return config


@app.delete("/api/mcp/servers/{name}")
def delete_mcp_server(name: str) -> dict[str, Any]:
    server_name = clean_single_name(name)
    config = load_mcp_config()
    config.setdefault("servers", {}).pop(server_name, None)
    save_mcp_config(config)
    return config


def build_chat_state(payload: ChatRequest) -> ChatState:
    previous = payload.state or {}
    state: ChatState = {
        "message": "",
        "web_search_mode": payload.options.web_search_mode,
        "web_search_provider": payload.options.web_search_provider,
        "rag_mode": payload.options.rag_mode,
        "curl_mode": payload.options.curl_mode,
        "python_mode": payload.options.python_mode,
        "file_editor_mode": payload.options.file_editor_mode,
        "file_editor_approval": payload.options.file_editor_approval,
        "mcp_mode": payload.options.mcp_mode,
        "max_tool_rounds": payload.options.max_tool_rounds,
    }
    for key in ["messages", "initialized", "conversation_summary", "web_search_results", "rag_results"]:
        if key in previous:
            state[key] = previous[key]  # type: ignore[literal-required]
    if payload.options.model:
        state["model"] = payload.options.model
    if payload.options.system_prompt:
        state["system_prompt"] = payload.options.system_prompt
    return state


def encode_sse(event: dict[str, Any]) -> str:
    clean = dict(event)
    if clean.get("type") == "assistant" and isinstance(clean.get("state"), dict):
        clean["state"] = public_state(clean["state"])
    return "data: " + json.dumps(clean, ensure_ascii=False, default=str) + "\n\n"


def public_state(state: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if key not in {"settings", "message"}}


def allowed_root(kind: str) -> Path:
    if kind not in ALLOWED_DATA_ROOTS:
        raise HTTPException(status_code=400, detail="unknown data kind")
    return ALLOWED_DATA_ROOTS[kind].resolve()


def resolve_artifact_path(path_text: str) -> Path:
    raw = Path(path_text)
    path = raw if raw.is_absolute() else PROJECT_ROOT / raw
    resolved = path.resolve()
    if not any(is_relative_to(resolved, root.resolve()) for root in ARTIFACT_ROOTS):
        raise HTTPException(status_code=400, detail="artifact path is not allowed")
    return resolved


def resolve_data_path(path_text: str, *, allow_missing: bool = False) -> Path:
    raw = Path(path_text)
    path = raw if raw.is_absolute() else PROJECT_ROOT / raw
    resolved = path.resolve() if path.exists() else path.parent.resolve() / path.name
    if not any(is_relative_to(resolved, root.resolve()) for root in ALLOWED_DATA_ROOTS.values()):
        raise HTTPException(status_code=400, detail="path must stay inside data/knowledge, data/memory, or data/skills")
    if not allow_missing and not resolved.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return resolved


def load_mcp_config() -> dict[str, Any]:
    MCP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MCP_CONFIG_PATH.exists():
        return {"servers": {}}
    data = json.loads(MCP_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("servers"), dict):
        raise HTTPException(status_code=500, detail="invalid data/mcp/servers.json")
    return data


def save_mcp_config(config: dict[str, Any]) -> None:
    MCP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    MCP_CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mcp_server_config_from_payload(payload: McpServerPayload) -> dict[str, Any]:
    server_config: dict[str, Any] = {"enabled": payload.enabled, "transport": payload.transport}
    if payload.transport == "streamable_http":
        server_config.update(
            {
                "url": clean_url(payload.url),
                "headers": payload.headers,
                "timeout": payload.timeout,
                "sse_read_timeout": payload.sse_read_timeout,
            }
        )
    else:
        server_config.update(
            {
                "command": payload.command.strip(),
                "args": payload.args,
                "env": payload.env,
            }
        )
    return server_config


def clean_single_name(name: str) -> str:
    clean = name.strip().replace("\\", "/").strip("/")
    if not clean or "/" in clean or ".." in clean:
        raise HTTPException(status_code=400, detail="name must be a single folder/config name")
    return clean


def clean_url(url: str) -> str:
    value = url.strip()
    if value.startswith("[") and "](" in value and value.endswith(")"):
        value = value.split("](", 1)[1][:-1].strip()
    if not value.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="streamable_http url must start with http:// or https://")
    return value


def relative_to_project(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
