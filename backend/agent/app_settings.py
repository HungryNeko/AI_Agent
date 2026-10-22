"""Persistent UI and chat defaults stored as JSON."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from agent.config import PROJECT_ROOT

SETTINGS_PATH = PROJECT_ROOT / "data" / "settings.json"
# How hard the model should think. "default" sends nothing, so the provider's own default applies.
REASONING_EFFORTS = {"default", "off", "low", "medium", "high", "max"}

DEFAULT_SETTINGS: dict[str, Any] = {
    "ui": {
        "theme": "system",
        "language": "zh",
    },
    "chat": {
        "model": "",
        "system_prompt": "",
        "web_search_mode": "auto",
        "web_search_provider": "duckduckgo",
        "web_search_auto_switch": True,
        "rag_mode": "auto",
        "rag_include_knowledge": True,
        "rag_include_memory": True,
        "rag_include_skills": True,
        "curl_mode": "auto",
        "python_mode": "auto",
        "file_reader_mode": "auto",
        "file_editor_mode": "auto",
        "file_editor_approval": "auto",
        "mcp_mode": "auto",
        "history_mode": "auto",
        "automation_mode": "auto",
        "question_mode": "light",
        "developer_mode": False,
        "conversation_mode": "agent",
        "max_tool_rounds": 20,
        "subagent_model": "",
        "reasoning_effort": "default",
        "subagent_reasoning_effort": "default",
    },
    "rag_ingestion": {
        "split_mode": "simple",
        "model": "",
    },
    # Changed only through set_security_level (the UI slider); never by the settings tool or API.
    "security": {"level": "L3"},
}


def load_app_settings() -> dict[str, Any]:
    raw = copy.deepcopy(DEFAULT_SETTINGS)
    if SETTINGS_PATH.exists():
        with SETTINGS_PATH.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        if not isinstance(raw, dict):
            raise ValueError("data/settings.json must be a JSON object.")
    local_path = local_settings_path()
    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as file:
            local = json.load(file)
        if not isinstance(local, dict):
            raise ValueError("data/settings.local.json must be a JSON object.")
        raw = deep_merge(raw, local)
    return normalize_app_settings(raw)


def save_app_settings(settings: dict[str, Any], path: Path | None = None, *, allow_security: bool = False) -> dict[str, Any]:
    normalized = normalize_app_settings(settings)
    if not allow_security:
        # Whatever was submitted, the stored security level stays as it is.
        try:
            normalized["security"] = load_app_settings()["security"]
        except (OSError, ValueError):
            normalized["security"] = copy.deepcopy(DEFAULT_SETTINGS["security"])
    target = path or local_settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return load_app_settings()


def patch_app_settings(patch: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise ValueError("settings patch must be an object.")
    if "security" in patch:
        raise ValueError("the security level cannot be changed through settings; use the security slider.")
    current = load_app_settings()
    merged = deep_merge(current, patch)
    return save_app_settings(merged)


def set_security_level(level: str) -> dict[str, Any]:
    clean = str(level or "").strip().upper()
    if clean not in {"L1", "L2", "L3"}:
        raise ValueError("security level must be L1, L2 or L3.")
    current = load_app_settings()
    current["security"] = {"level": clean}
    return save_app_settings(current, allow_security=True)


def local_settings_path() -> Path:
    return SETTINGS_PATH.with_name("settings.local.json")


def normalize_app_settings(raw: dict[str, Any]) -> dict[str, Any]:
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    settings = deep_merge(settings, raw)
    ui = settings["ui"]
    chat = settings["chat"]
    rag_ingestion = settings["rag_ingestion"]

    if ui.get("theme") not in {"system", "light", "dark"}:
        ui["theme"] = DEFAULT_SETTINGS["ui"]["theme"]
    if ui.get("language") not in {"zh", "en", "both"}:
        ui["language"] = DEFAULT_SETTINGS["ui"]["language"]

    for key in [
        "model",
        "system_prompt",
        "subagent_model",
    ]:
        chat[key] = str(chat.get(key, DEFAULT_SETTINGS["chat"][key]) or "")
    for key, allowed in {
        "web_search_mode": {"off", "tool", "native", "both", "auto"},
        "web_search_provider": {"duckduckgo", "searxng", "tavily"},
        "rag_mode": {"off", "on", "auto"},
        "curl_mode": {"off", "auto"},
        "python_mode": {"off", "auto"},
        "file_reader_mode": {"off", "auto"},
        "file_editor_mode": {"off", "auto"},
        "file_editor_approval": {"readOnly", "manual", "auto", "aiReview"},
        "mcp_mode": {"off", "auto"},
        "history_mode": {"off", "auto"},
        "automation_mode": {"off", "auto"},
        "question_mode": {"off", "light", "heavy"},
        "conversation_mode": {"ask", "plan", "agent"},
        "reasoning_effort": REASONING_EFFORTS,
        "subagent_reasoning_effort": REASONING_EFFORTS,
    }.items():
        value = str(chat.get(key, DEFAULT_SETTINGS["chat"][key]) or "")
        if key == "conversation_mode" and value == "chat":
            value = "ask"
        chat[key] = value if value in allowed else DEFAULT_SETTINGS["chat"][key]

    for key in [
        "rag_include_knowledge",
        "rag_include_memory",
        "rag_include_skills",
        "web_search_auto_switch",
        "developer_mode",
    ]:
        chat[key] = bool(chat.get(key, DEFAULT_SETTINGS["chat"][key]))

    try:
        chat["max_tool_rounds"] = int(chat.get("max_tool_rounds", DEFAULT_SETTINGS["chat"]["max_tool_rounds"]))
    except (TypeError, ValueError):
        chat["max_tool_rounds"] = DEFAULT_SETTINGS["chat"]["max_tool_rounds"]
    if chat["max_tool_rounds"] < -1:
        chat["max_tool_rounds"] = -1

    security = settings.get("security")
    level = str((security or {}).get("level") if isinstance(security, dict) else "").strip().upper()
    settings["security"] = {"level": level if level in {"L1", "L2", "L3"} else DEFAULT_SETTINGS["security"]["level"]}

    if rag_ingestion.get("split_mode") not in {"simple", "llm"}:
        rag_ingestion["split_mode"] = DEFAULT_SETTINGS["rag_ingestion"]["split_mode"]
    rag_ingestion["model"] = str(rag_ingestion.get("model") or "")

    return settings


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
