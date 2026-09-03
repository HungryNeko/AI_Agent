"""Small FastAPI server for the React test frontend."""

from __future__ import annotations

import json
import mimetypes
import subprocess
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent.app_settings import load_app_settings, patch_app_settings, save_app_settings
from agent import session_store
from agent.config import load_config
from agent.debug_log import log_event, log_exception
from agent.graph import ChatState, stream_turn
from agent.instructions import load_instruction, save_instruction
from tools import mcp as mcp_tool
from tools import rag
from tools.mcp import McpRequest
from tools.settings import McpSettings, make_tool_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
SYSTEM_DATA_ROOTS = {
    "knowledge": DATA_ROOT / "knowledge",
    "memory": DATA_ROOT / "memory",
    "skills": DATA_ROOT / "skills",
}
USER_DATA_ROOT = PROJECT_ROOT / "backend" / "runtime" / "user_data"
USER_DATA_ROOTS = {
    "knowledge": USER_DATA_ROOT / "knowledge",
    "memory": USER_DATA_ROOT / "memory",
    "skills": USER_DATA_ROOT / "skills",
}
ALLOWED_DATA_ROOTS = {kind: (SYSTEM_DATA_ROOTS[kind], USER_DATA_ROOTS[kind]) for kind in SYSTEM_DATA_ROOTS}
MCP_CONFIG_PATH = DATA_ROOT / "mcp" / "servers.json"
TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml"}
ARTIFACT_ROOTS = [
    PROJECT_ROOT / "backend" / "runtime" / "python_runs",
    PROJECT_ROOT / "backend" / "runtime" / "mcp_artifacts",
    PROJECT_ROOT / "backend" / "runtime" / "uploads",
]

app = FastAPI(title="AI Agent Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_http_request(request: Request, call_next):
    log_event("http.request", method=request.method, path=request.url.path, query=str(request.url.query))
    try:
        response = await call_next(request)
    except Exception as exc:
        log_exception(
            "http.error",
            exc,
            method=request.method,
            path=request.url.path,
            query=str(request.url.query),
        )
        raise
    log_event(
        "http.response",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
    )
    return response


class ChatOptions(BaseModel):
    model: str | None = None
    system_prompt: str | None = None
    web_search_mode: Literal["off", "auto"] = "auto"
    web_search_provider: Literal["duckduckgo", "searxng", "tavily"] = "duckduckgo"
    web_search_auto_switch: bool = False
    rag_mode: Literal["off", "on", "auto"] = "auto"
    rag_include_knowledge: bool = True
    rag_include_memory: bool = True
    rag_include_skills: bool = True
    curl_mode: Literal["off", "auto"] = "auto"
    python_mode: Literal["off", "auto"] = "auto"
    file_editor_mode: Literal["off", "auto"] = "auto"
    file_editor_approval: Literal["readOnly", "manual", "auto"] = "auto"
    mcp_mode: Literal["off", "auto"] = "auto"
    history_mode: Literal["off", "auto"] = "auto"
    automation_mode: Literal["off", "auto"] = "off"
    max_tool_rounds: int = Field(default=20, ge=-1)


class AttachmentPayload(BaseModel):
    path: str
    filename: str = ""
    content_type: str = ""


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    state: dict[str, Any] | None = None
    options: ChatOptions | None = None
    attachments: list[AttachmentPayload] = []


class DataFilePayload(BaseModel):
    path: str
    content: str


class InstructionPayload(BaseModel):
    content: str


class ConfigPayload(BaseModel):
    config: dict[str, Any]


class SettingsPayload(BaseModel):
    settings: dict[str, Any]


class SettingsPatchPayload(BaseModel):
    patch: dict[str, Any]


class RenamePayload(BaseModel):
    title: str


class SkillPayload(BaseModel):
    name: str
    content: str


class DataImportPayload(BaseModel):
    kind: Literal["knowledge", "memory", "skills"]
    name: str
    content: str


class DataRenamePayload(BaseModel):
    path: str
    new_name: str


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


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    return {"path": "data/api_configs.json", "config": load_config()}


@app.put("/api/config")
def put_config(payload: ConfigPayload) -> dict[str, Any]:
    if not isinstance(payload.config.get("providers"), dict):
        raise HTTPException(status_code=400, detail="config.providers must be an object")
    path = DATA_ROOT / "api_configs.json"
    path.write_text(json.dumps(payload.config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"path": "data/api_configs.json", "status": "saved", "config": load_config()}


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    try:
        settings = load_app_settings()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"path": "data/settings.json", "settings": settings}


@app.put("/api/settings")
def put_settings(payload: SettingsPayload) -> dict[str, Any]:
    try:
        settings = save_app_settings(payload.settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"path": "data/settings.json", "status": "saved", "settings": settings}


@app.patch("/api/settings")
def patch_settings(payload: SettingsPatchPayload) -> dict[str, Any]:
    try:
        settings = patch_app_settings(payload.patch)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"path": "data/settings.json", "status": "saved", "settings": settings}


@app.post("/api/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    log_event("http.chat_stream", message=payload.message, options=model_to_dict(payload.options) if payload.options else {})
    state = build_chat_state(payload)
    conversation_id = payload.conversation_id or session_store.create_conversation_id()
    try:
        session_store.conversation_path(conversation_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def events():
        turn_events: list[dict[str, Any]] = []
        final_state: dict[str, Any] = {}
        try:
            for event in stream_turn(state, payload.message):
                event_with_id = {**event, "conversation_id": conversation_id}
                turn_events.append(event_with_id)
                if event.get("type") == "assistant" and isinstance(event.get("state"), dict):
                    final_state = public_state(event["state"])
                yield encode_sse(event_with_id)
            session_store.save_turn(
                conversation_id,
                user_text=payload.message,
                turn_events=turn_events,
                state=final_state or public_state(state),
                attachments=[model_to_dict(item) for item in payload.attachments],
            )
        except Exception as exc:  # noqa: BLE001
            log_exception("http.chat_stream_error", exc, message=payload.message)
            error_event = {"type": "error", "text": str(exc), "conversation_id": conversation_id}
            turn_events.append(error_event)
            session_store.save_turn(
                conversation_id,
                user_text=payload.message,
                turn_events=turn_events,
                state=final_state or public_state(state),
                attachments=[model_to_dict(item) for item in payload.attachments],
            )
            yield encode_sse(error_event)

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/api/instruction")
def get_instruction() -> dict[str, str]:
    return {"path": "data/instruction.md", "content": load_instruction()}


@app.put("/api/instruction")
def put_instruction(payload: InstructionPayload) -> dict[str, str]:
    save_instruction(payload.content)
    return {"path": "data/instruction.md", "status": "saved", "content": load_instruction()}


@app.get("/api/conversations")
def conversations(limit: int = Query(50, ge=1, le=100), query: str = Query("")) -> dict[str, Any]:
    return {"conversations": session_store.list_conversations(limit=limit, query=query)}


@app.get("/api/conversations/{conversation_id}")
def conversation(conversation_id: str) -> dict[str, Any]:
    try:
        return session_store.read_conversation(conversation_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/conversations/{conversation_id}/compress")
def compress_conversation(conversation_id: str) -> dict[str, Any]:
    try:
        conversation = session_store.compress_conversation(conversation_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "compressed", **conversation}


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, payload: RenamePayload) -> dict[str, Any]:
    try:
        conversation = session_store.rename_conversation(conversation_id, payload.title)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "renamed", **conversation}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict[str, str]:
    try:
        session_store.delete_conversation(conversation_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "deleted"}


@app.post("/api/rag/reindex")
def reindex_rag() -> dict[str, Any]:
    settings = make_tool_settings(rag_mode="auto")
    return rag.index_status(settings.rag)


@app.post("/api/uploads")
async def upload_file(request: Request, filename: str = Query("upload.bin")) -> dict[str, str]:
    root = PROJECT_ROOT / "backend" / "runtime" / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    clean_name = clean_upload_name(filename)
    path = root / clean_name
    suffix = 1
    while path.exists():
        path = root / f"{Path(clean_name).stem}-{suffix}{Path(clean_name).suffix}"
        suffix += 1
    path.write_bytes(await request.body())
    content_type = request.headers.get("content-type") or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return {"path": relative_to_project(path), "filename": path.name, "content_type": content_type}


@app.get("/api/data/files")
def list_data_files(kind: Literal["knowledge", "memory", "skills"] = Query(...)) -> dict[str, Any]:
    items = list_data_file_items(kind)
    return {"kind": kind, "files": [item["path"] for item in items], "items": items}


@app.get("/api/data/file")
def read_data_file(path: str = Query(...)) -> dict[str, Any]:
    resolved, scope, writable = resolve_data_path(path)
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return {
        "path": relative_to_project(resolved),
        "content": resolved.read_text(encoding="utf-8", errors="replace"),
        "scope": scope,
        "writable": writable,
    }


@app.put("/api/data/file")
def write_data_file(payload: DataFilePayload) -> dict[str, str]:
    resolved, _, writable = resolve_data_path(payload.path, allow_missing=True)
    if not writable:
        raise HTTPException(status_code=403, detail="system files are read-only")
    if resolved.suffix.lower() not in TEXT_SUFFIXES:
        raise HTTPException(status_code=400, detail="unsupported text file type")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(payload.content, encoding="utf-8")
    return {"path": relative_to_project(resolved), "status": "saved"}


@app.post("/api/data/import")
def import_data_file(payload: DataImportPayload) -> dict[str, str]:
    kind = payload.kind
    name = clean_import_name(payload.name)
    root = allowed_root(kind)
    if kind == "skills":
        path = root / name / "SKILL.md"
    else:
        filename = name if Path(name).suffix else f"{name}.md"
        path = root / filename
    resolved = path.resolve()
    if not is_relative_to(resolved, root.resolve()):
        raise HTTPException(status_code=400, detail="path must stay inside user data")
    if resolved.suffix.lower() not in TEXT_SUFFIXES:
        raise HTTPException(status_code=400, detail="unsupported text file type")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(payload.content, encoding="utf-8")
    rag.index_status(make_tool_settings(rag_mode="auto").rag)
    return {"path": relative_to_project(resolved), "status": "saved"}


@app.post("/api/data/file/rename")
def rename_data_file(payload: DataRenamePayload) -> dict[str, str]:
    resolved, _, writable = resolve_data_path(payload.path)
    if not writable:
        raise HTTPException(status_code=403, detail="system files are read-only")
    if resolved.suffix.lower() != ".md":
        raise HTTPException(status_code=400, detail="only markdown files can be renamed")
    new_name = clean_import_name(payload.new_name)
    if Path(new_name).suffix and Path(new_name).suffix.lower() != ".md":
        raise HTTPException(status_code=400, detail="new name must be markdown")
    target = resolved.with_name(new_name if new_name.endswith(".md") else f"{new_name}.md").resolve()
    writable_roots = [
        root.resolve()
        for kind in ALLOWED_DATA_ROOTS
        for _, root, writable_root in iter_data_roots(kind)
        if writable_root
    ]
    if not any(is_relative_to(target, root) for root in writable_roots):
        raise HTTPException(status_code=400, detail="path must stay inside user data")
    if target.exists():
        raise HTTPException(status_code=409, detail="target already exists")
    resolved.rename(target)
    rag.index_status(make_tool_settings(rag_mode="auto").rag)
    return {"path": relative_to_project(target), "status": "renamed"}


@app.post("/api/skills/import")
def import_skill(payload: SkillPayload) -> dict[str, str]:
    name = clean_single_name(payload.name)
    path = allowed_root("skills") / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.content, encoding="utf-8")
    rag.index_status(make_tool_settings(rag_mode="auto").rag)
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


@app.post("/api/mcp/import")
def import_mcp_server(payload: McpServerPayload) -> dict[str, Any]:
    server_name = clean_single_name(payload.name)
    config = load_mcp_config()
    config.setdefault("servers", {})[server_name] = mcp_server_config_from_payload(payload)
    save_mcp_config(config)
    return {"status": "saved", "server": server_name, "config": config}


@app.delete("/api/mcp/servers/{name}")
def delete_mcp_server(name: str) -> dict[str, Any]:
    server_name = clean_single_name(name)
    config = load_mcp_config()
    config.setdefault("servers", {}).pop(server_name, None)
    save_mcp_config(config)
    return config


def build_chat_state(payload: ChatRequest) -> ChatState:
    previous = payload.state or {}
    saved_options = load_app_settings().get("chat", {})
    request_options = model_to_dict(payload.options) if payload.options else {}
    options = merge_chat_options(saved_options, request_options)
    state: ChatState = {
        "message": "",
        "web_search_mode": options["web_search_mode"],
        "web_search_provider": options["web_search_provider"],
        "web_search_auto_switch": options["web_search_auto_switch"],
        "rag_mode": options["rag_mode"],
        "rag_include_knowledge": options["rag_include_knowledge"],
        "rag_include_memory": options["rag_include_memory"],
        "rag_include_skills": options["rag_include_skills"],
        "curl_mode": options["curl_mode"],
        "python_mode": options["python_mode"],
        "file_editor_mode": options["file_editor_mode"],
        "file_editor_approval": options["file_editor_approval"],
        "mcp_mode": options["mcp_mode"],
        "history_mode": options["history_mode"],
        "automation_mode": options["automation_mode"],
        "max_tool_rounds": options["max_tool_rounds"],
        "attachments": [model_to_dict(item) for item in payload.attachments],
    }
    for key in ["messages", "initialized", "conversation_summary", "web_search_results", "rag_results"]:
        if key in previous:
            state[key] = previous[key]  # type: ignore[literal-required]
    if options.get("model"):
        state["model"] = options["model"]
    if options.get("system_prompt"):
        state["system_prompt"] = options["system_prompt"]
    return state


def merge_chat_options(saved_options: dict[str, Any], request_options: dict[str, Any]) -> dict[str, Any]:
    merged = dict(saved_options)
    for key, value in request_options.items():
        if value not in {None, ""}:
            merged[key] = value
    return merged


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


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
    roots = ALLOWED_DATA_ROOTS[kind]
    if isinstance(roots, tuple):
        return roots[-1].resolve()
    return roots.resolve()


def resolve_artifact_path(path_text: str) -> Path:
    raw = Path(path_text)
    path = raw if raw.is_absolute() else PROJECT_ROOT / raw
    resolved = path.resolve()
    if not any(is_relative_to(resolved, root.resolve()) for root in ARTIFACT_ROOTS):
        raise HTTPException(status_code=400, detail="artifact path is not allowed")
    return resolved


def list_data_file_items(kind: str) -> list[dict[str, Any]]:
    if kind not in ALLOWED_DATA_ROOTS:
        raise HTTPException(status_code=400, detail="unknown data kind")
    items: list[dict[str, Any]] = []
    for scope, root, writable in iter_data_roots(kind):
        root.mkdir(parents=True, exist_ok=True)
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                item_scope, item_writable = classify_data_file(path, scope, writable)
                items.append(
                    {
                        "path": relative_to_project(path),
                        "name": path.name,
                        "scope": item_scope,
                        "writable": item_writable,
                    }
                )
    return items


def resolve_data_path(path_text: str, *, allow_missing: bool = False) -> tuple[Path, str, bool]:
    raw = Path(path_text)
    path = raw if raw.is_absolute() else PROJECT_ROOT / raw
    resolved = path.resolve() if path.exists() else path.parent.resolve() / path.name
    for kind in ALLOWED_DATA_ROOTS:
        for scope, root, writable in iter_data_roots(kind):
            if is_relative_to(resolved, root.resolve()):
                if not allow_missing and not resolved.exists():
                    raise HTTPException(status_code=404, detail="file not found")
                item_scope, item_writable = classify_data_file(resolved, scope, writable)
                return resolved, item_scope, item_writable
    allowed = "data/knowledge, data/memory, data/skills, or backend/runtime/user_data"
    raise HTTPException(status_code=400, detail=f"path must stay inside {allowed}")


def iter_data_roots(kind: str) -> list[tuple[str, Path, bool]]:
    roots = ALLOWED_DATA_ROOTS[kind]
    if isinstance(roots, tuple):
        return [("system", roots[0], False), ("user", roots[1], True)]
    return [("user", roots, True)]


def classify_data_file(path: Path, scope: str, writable: bool) -> tuple[str, bool]:
    if scope != "system":
        return scope, writable
    if is_git_ignored(path):
        return "user", True
    return "system", False


def is_git_ignored(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", str(path.resolve())],
            cwd=PROJECT_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


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


def clean_import_name(name: str) -> str:
    clean = Path(name.replace("\\", "/")).name.strip()
    if not clean or clean in {".", ".."} or ".." in clean:
        raise HTTPException(status_code=400, detail="invalid import name")
    return clean


def clean_upload_name(name: str) -> str:
    clean = Path(name.replace("\\", "/")).name.strip()
    if not clean or clean in {".", ".."}:
        raise HTTPException(status_code=400, detail="invalid upload filename")
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
