"""Small FastAPI server for the React test frontend."""

from __future__ import annotations

import json
import mimetypes
import re
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any, AsyncIterator, Literal
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent.automation_runner import list_run_records, start_runner, stop_runner
from agent.app_settings import load_app_settings, patch_app_settings, save_app_settings
from agent import session_store
from agent.config import load_config, save_config
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
MCP_LOCAL_CONFIG_PATH = DATA_ROOT / "mcp" / "servers.local.json"
TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml"}
ARTIFACT_ROOTS = [
    PROJECT_ROOT / "backend" / "runtime" / "python_runs",
    PROJECT_ROOT / "backend" / "runtime" / "mcp_artifacts",
    PROJECT_ROOT / "backend" / "runtime" / "uploads",
]
_CANCELLED_RUNS: set[str] = set()
_CANCELLED_RUNS_LOCK = Lock()


def mark_cancelled_run(run_id: str) -> None:
    with _CANCELLED_RUNS_LOCK:
        _CANCELLED_RUNS.add(run_id)


def is_cancelled_run(run_id: str) -> bool:
    with _CANCELLED_RUNS_LOCK:
        return run_id in _CANCELLED_RUNS


def clear_cancelled_run(run_id: str) -> None:
    with _CANCELLED_RUNS_LOCK:
        _CANCELLED_RUNS.discard(run_id)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    start_runner()
    try:
        yield
    finally:
        stop_runner()


app = FastAPI(title="AI Agent Backend", version="0.1.0", lifespan=lifespan)
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
    file_editor_approval: Literal["readOnly", "manual", "auto", "aiReview"] = "auto"
    mcp_mode: Literal["off", "auto"] = "auto"
    history_mode: Literal["off", "auto"] = "auto"
    automation_mode: Literal["off", "auto"] = "off"
    max_tool_rounds: int = Field(default=20, ge=-1)


class AttachmentPayload(BaseModel):
    path: str
    filename: str = ""
    content_type: str = ""
    url: str = ""
    absolute_url: str = ""
    size: int = 0


class ChatRequest(BaseModel):
    message: str
    run_id: str | None = None
    conversation_id: str | None = None
    state: dict[str, Any] | None = None
    options: ChatOptions | None = None
    attachments: list[AttachmentPayload] = []


class StopChatRequest(BaseModel):
    run_id: str


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


class AutomationPayload(BaseModel):
    title: str = ""
    action: Literal["script", "mcp", "configureMcp", "reminder", "llm"] = "reminder"
    enabled: bool = True
    prompt: str = ""
    code: str = ""
    mcp_server: str = ""
    mcp_tool: str = ""
    mcp_arguments: dict[str, Any] = {}
    mcp_config: dict[str, Any] = {}
    schedule: dict[str, Any] = {}


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


@app.get("/api")
@app.get("/api/")
def api_root() -> dict[str, str]:
    return {"status": "ok", "health": "/api/health"}


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
    merged = save_config(payload.config, DATA_ROOT / "api_configs.local.json")
    return {"path": "data/api_configs.local.json", "status": "saved", "config": merged}


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    try:
        settings = load_app_settings()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"path": "data/settings.local.json", "settings": settings}


@app.put("/api/settings")
def put_settings(payload: SettingsPayload) -> dict[str, Any]:
    try:
        settings = save_app_settings(payload.settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"path": "data/settings.local.json", "status": "saved", "settings": settings}


@app.patch("/api/settings")
def patch_settings(payload: SettingsPatchPayload) -> dict[str, Any]:
    try:
        settings = patch_app_settings(payload.patch)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"path": "data/settings.local.json", "status": "saved", "settings": settings}


@app.post("/api/chat/stop")
def stop_chat(payload: StopChatRequest) -> dict[str, str]:
    run_id = payload.run_id.strip()
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required")
    mark_cancelled_run(run_id)
    log_event("http.chat_stop", run_id=run_id)
    return {"status": "stopping", "run_id": run_id}


@app.post("/api/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    if not payload.message.strip() and not payload.attachments:
        raise HTTPException(status_code=400, detail="message or attachment is required")

    log_event("http.chat_stream", message=payload.message, options=model_to_dict(payload.options) if payload.options else {})
    state = build_chat_state(payload)
    conversation_id = payload.conversation_id or session_store.create_conversation_id()
    try:
        session_store.conversation_path(conversation_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    run_id = (payload.run_id or uuid.uuid4().hex).strip() or uuid.uuid4().hex
    clear_cancelled_run(run_id)

    def events():
        turn_events: list[dict[str, Any]] = []
        final_state: dict[str, Any] = {}
        try:
            for event in stream_turn(state, payload.message):
                if is_cancelled_run(run_id):
                    stopped_event = {
                        "type": "stopped",
                        "text": "AI output stopped.",
                        "conversation_id": conversation_id,
                        "run_id": run_id,
                    }
                    turn_events.append(stopped_event)
                    yield encode_sse(stopped_event)
                    break
                event_with_id = {**event, "conversation_id": conversation_id, "run_id": run_id}
                turn_events.append(event_with_id)
                if event.get("type") == "assistant" and isinstance(event.get("state"), dict):
                    final_state = public_state(event["state"])
                yield encode_sse(event_with_id)
                if is_cancelled_run(run_id):
                    stopped_event = {
                        "type": "stopped",
                        "text": "AI output stopped.",
                        "conversation_id": conversation_id,
                        "run_id": run_id,
                    }
                    turn_events.append(stopped_event)
                    yield encode_sse(stopped_event)
                    break
            session_store.save_turn(
                conversation_id,
                user_text=payload.message,
                turn_events=turn_events,
                state=final_state or public_state(state),
                attachments=[model_to_dict(item) for item in payload.attachments],
            )
        except Exception as exc:  # noqa: BLE001
            log_exception("http.chat_stream_error", exc, message=payload.message)
            error_event = {"type": "error", "text": str(exc), "conversation_id": conversation_id, "run_id": run_id}
            turn_events.append(error_event)
            session_store.save_turn(
                conversation_id,
                user_text=payload.message,
                turn_events=turn_events,
                state=final_state or public_state(state),
                attachments=[model_to_dict(item) for item in payload.attachments],
            )
            yield encode_sse(error_event)
        finally:
            clear_cancelled_run(run_id)

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
async def upload_file(request: Request, filename: str = Query("upload.bin")) -> dict[str, Any]:
    root = upload_root()
    clean_name = clean_upload_name(filename)
    upload_id = uuid.uuid4().hex
    upload_dir = root / upload_id
    upload_dir.mkdir(parents=True, exist_ok=False)
    path = upload_dir / clean_name
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="upload body is empty")
    path.write_bytes(body)
    content_type = request.headers.get("content-type") or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    url = f"/api/uploads/{upload_id}/{quote(path.name)}"
    return {
        "id": upload_id,
        "path": relative_to_project(path),
        "filename": path.name,
        "content_type": content_type,
        "size": path.stat().st_size,
        "url": url,
        "absolute_url": absolute_request_url(request, url),
    }


@app.get("/api/uploads/{upload_id}/{filename}")
def uploaded_file(upload_id: str, filename: str) -> FileResponse:
    resolved = resolve_uploaded_file_path(upload_id, filename)
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="upload not found")
    return FileResponse(resolved)


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


@app.get("/api/automations")
def list_automations() -> dict[str, Any]:
    root = automation_root()
    root.mkdir(parents=True, exist_ok=True)
    items = [automation_summary(path) for path in sorted(root.glob("*.json"), reverse=True)]
    return {"items": items}


@app.post("/api/automations")
def create_automation(payload: AutomationPayload) -> dict[str, Any]:
    root = automation_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / new_automation_filename(payload.title or payload.action)
    write_automation(path, payload)
    return {"status": "saved", "item": automation_summary(path)}


@app.get("/api/automations/{automation_id}")
def read_automation(automation_id: str) -> dict[str, Any]:
    path = resolve_automation_path(automation_id)
    return {
        "item": automation_summary(path),
        "content": read_automation_payload(path),
        "runs": list_run_records(automation_run_root(), automation_id=path.name, limit=20),
    }


@app.get("/api/automation-runs")
def automation_runs(limit: int = Query(50, ge=1, le=200), automation_id: str = Query("")) -> dict[str, Any]:
    clean_id = clean_automation_name(automation_id) if automation_id else None
    return {"items": list_run_records(automation_run_root(), automation_id=clean_id, limit=limit)}


@app.put("/api/automations/{automation_id}")
def update_automation(automation_id: str, payload: AutomationPayload) -> dict[str, Any]:
    path = resolve_automation_path(automation_id, allow_missing=True)
    write_automation(path, payload)
    return {"status": "saved", "item": automation_summary(path)}


@app.delete("/api/automations/{automation_id}")
def delete_automation(automation_id: str) -> dict[str, str]:
    path = resolve_automation_path(automation_id)
    path.unlink()
    return {"status": "deleted"}


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
    data: dict[str, Any] = {"servers": {}}
    if MCP_CONFIG_PATH.exists():
        data = json.loads(MCP_CONFIG_PATH.read_text(encoding="utf-8"))
    if MCP_LOCAL_CONFIG_PATH.exists():
        local = json.loads(MCP_LOCAL_CONFIG_PATH.read_text(encoding="utf-8"))
        base_servers = data.get("servers") if isinstance(data.get("servers"), dict) else {}
        local_servers = local.get("servers") if isinstance(local.get("servers"), dict) else {}
        data["servers"] = {**base_servers, **local_servers}
    if not isinstance(data, dict) or not isinstance(data.get("servers"), dict):
        raise HTTPException(status_code=500, detail="invalid data/mcp/servers.json")
    return data


def save_mcp_config(config: dict[str, Any]) -> None:
    MCP_LOCAL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    MCP_LOCAL_CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def clean_upload_id(upload_id: str) -> str:
    clean = upload_id.strip()
    if not re.fullmatch(r"[a-fA-F0-9]{32}", clean):
        raise HTTPException(status_code=400, detail="invalid upload id")
    return clean.lower()


def upload_root() -> Path:
    return (PROJECT_ROOT / "backend" / "runtime" / "uploads").resolve()


def resolve_uploaded_file_path(upload_id: str, filename: str) -> Path:
    root = upload_root()
    path = (root / clean_upload_id(upload_id) / clean_upload_name(filename)).resolve()
    if not is_relative_to(path, root):
        raise HTTPException(status_code=400, detail="upload path is not allowed")
    return path


def absolute_request_url(request: Request, path: str) -> str:
    host = (request.headers.get("x-forwarded-host") or request.headers.get("host") or "").split(",", 1)[0].strip()
    proto = (request.headers.get("x-forwarded-proto") or request.url.scheme).split(",", 1)[0].strip()
    prefix = (request.headers.get("x-forwarded-prefix") or "").rstrip("/")
    if host:
        return f"{proto}://{host}{prefix}{path}"
    return str(request.base_url).rstrip("/") + path


def automation_root() -> Path:
    return (PROJECT_ROOT / "backend" / "runtime" / "automations").resolve()


def automation_run_root() -> Path:
    return (PROJECT_ROOT / "backend" / "runtime" / "automation_runs").resolve()


def resolve_automation_path(automation_id: str, *, allow_missing: bool = False) -> Path:
    name = clean_automation_name(automation_id)
    path = (automation_root() / name).resolve()
    if not is_relative_to(path, automation_root()):
        raise HTTPException(status_code=400, detail="automation path is not allowed")
    if not allow_missing and not path.is_file():
        raise HTTPException(status_code=404, detail="automation not found")
    return path


def clean_automation_name(name: str) -> str:
    clean = Path(name.replace("\\", "/")).name.strip()
    if not clean or clean in {".", ".."} or ".." in clean:
        raise HTTPException(status_code=400, detail="invalid automation name")
    if Path(clean).suffix and Path(clean).suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail="automation must be a json file")
    return clean if clean.lower().endswith(".json") else f"{clean}.json"


def new_automation_filename(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", title.strip().lower()).strip("-") or "automation"
    return f"{int(time.time())}-{slug[:48]}.json"


def write_automation(path: Path, payload: AutomationPayload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = read_automation_payload(path) if path.exists() else {}
    data = {
        **current,
        "title": payload.title.strip() or payload.action,
        "action": payload.action,
        "enabled": payload.enabled,
        "prompt": payload.prompt,
        "code": payload.code,
        "mcp_server": payload.mcp_server,
        "mcp_tool": payload.mcp_tool,
        "mcp_arguments": payload.mcp_arguments,
        "mcp_config": payload.mcp_config,
        "schedule": payload.schedule,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    data.setdefault("created_at", data["updated_at"])
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_automation_payload(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"invalid automation json: {path.name}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail=f"invalid automation json: {path.name}")
    return data


def automation_summary(path: Path) -> dict[str, Any]:
    data = read_automation_payload(path)
    schedule = data.get("schedule") if isinstance(data.get("schedule"), dict) else {}
    recent_runs = list_run_records(automation_run_root(), automation_id=path.name, limit=3)
    return {
        "id": path.name,
        "path": relative_to_project(path),
        "title": str(data.get("title") or path.stem),
        "action": str(data.get("action") or "reminder"),
        "enabled": bool(data.get("enabled", True)),
        "prompt": str(data.get("prompt") or ""),
        "schedule": schedule,
        "schedule_kind": str(schedule.get("kind") or ""),
        "next_run_at": str(schedule.get("nextRunAt") or ""),
        "last_run": data.get("last_run") if isinstance(data.get("last_run"), dict) else {},
        "recent_runs": recent_runs,
        "run_log": "backend/runtime/automation_runs/runs-YYYYMMDD.jsonl",
        "conversation_id": str(data.get("conversation_id") or ""),
        "created_at": str(data.get("created_at") or ""),
        "updated_at": str(data.get("updated_at") or ""),
    }


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
