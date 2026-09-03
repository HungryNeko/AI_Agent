"""Persistent UI and chat defaults stored as JSON."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from agent.config import PROJECT_ROOT

SETTINGS_PATH = PROJECT_ROOT / "data" / "settings.json"

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
        "file_editor_mode": "auto",
        "file_editor_approval": "auto",
        "mcp_mode": "auto",
        "history_mode": "auto",
        "automation_mode": "auto",
        "max_tool_rounds": 20,
    },
}


def load_app_settings() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        save_app_settings(DEFAULT_SETTINGS)
        return copy.deepcopy(DEFAULT_SETTINGS)
    with SETTINGS_PATH.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError("data/settings.json must be a JSON object.")
    return normalize_app_settings(raw)


def save_app_settings(settings: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_app_settings(settings)
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return normalized


def patch_app_settings(patch: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise ValueError("settings patch must be an object.")
    current = load_app_settings()
    merged = deep_merge(current, patch)
    return save_app_settings(merged)


def normalize_app_settings(raw: dict[str, Any]) -> dict[str, Any]:
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    settings = deep_merge(settings, raw)
    ui = settings["ui"]
    chat = settings["chat"]

    if ui.get("theme") not in {"system", "light", "dark"}:
        ui["theme"] = DEFAULT_SETTINGS["ui"]["theme"]
    if ui.get("language") not in {"zh", "en", "both"}:
        ui["language"] = DEFAULT_SETTINGS["ui"]["language"]

    for key in [
        "model",
        "system_prompt",
    ]:
        chat[key] = str(chat.get(key, DEFAULT_SETTINGS["chat"][key]) or "")
    for key, allowed in {
        "web_search_mode": {"off", "auto"},
        "web_search_provider": {"duckduckgo", "searxng", "tavily"},
        "rag_mode": {"off", "on", "auto"},
        "curl_mode": {"off", "auto"},
        "python_mode": {"off", "auto"},
        "file_editor_mode": {"off", "auto"},
        "file_editor_approval": {"readOnly", "manual", "auto"},
        "mcp_mode": {"off", "auto"},
        "history_mode": {"off", "auto"},
        "automation_mode": {"off", "auto"},
    }.items():
        value = str(chat.get(key, DEFAULT_SETTINGS["chat"][key]) or "")
        chat[key] = value if value in allowed else DEFAULT_SETTINGS["chat"][key]

    for key in ["rag_include_knowledge", "rag_include_memory", "rag_include_skills", "web_search_auto_switch"]:
        chat[key] = bool(chat.get(key, DEFAULT_SETTINGS["chat"][key]))

    try:
        chat["max_tool_rounds"] = int(chat.get("max_tool_rounds", DEFAULT_SETTINGS["chat"]["max_tool_rounds"]))
    except (TypeError, ValueError):
        chat["max_tool_rounds"] = DEFAULT_SETTINGS["chat"]["max_tool_rounds"]
    if chat["max_tool_rounds"] < -1:
        chat["max_tool_rounds"] = -1

    return settings


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
