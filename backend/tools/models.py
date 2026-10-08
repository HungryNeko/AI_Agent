"""Manage OpenAI-compatible provider model catalogs without exposing secrets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Any

import httpx

from agent.config import (
    get_model_config,
    list_model_items,
    load_config,
    local_config_path,
    save_config,
)


@dataclass(frozen=True)
class ModelRequest:
    action: Literal["list", "refresh", "switch"]
    provider: str = ""
    model: str = ""


def execute(request: ModelRequest) -> str:
    if request.action == "list":
        return format_result({"models": list_model_items()})
    if request.action == "refresh":
        return format_result(refresh_models(request.provider))
    if request.action == "switch":
        return format_result({"selected": resolve_model_selection(request)})
    raise ValueError(f"unknown model action: {request.action}")


def refresh_models(provider_name: str = "") -> dict[str, Any]:
    config = load_config()
    providers = config.get("providers", {})
    if not isinstance(providers, dict):
        raise ValueError("model config providers must be an object")
    name = provider_name.strip() or str(config.get("default_provider") or "")
    provider = providers.get(name)
    if not name or not isinstance(provider, dict):
        raise ValueError(f"unknown provider: {name or '(empty)'}")

    model_config = get_model_config(f"{name}:{provider.get('default_model') or 'model-list'}")
    headers = {"Accept": "application/json"}
    if model_config.api_key_value:
        headers["Authorization"] = f"Bearer {model_config.api_key_value}"
    response = httpx.get(
        f"{model_config.base_url}/models",
        headers=headers,
        timeout=float(provider.get("timeout") or 15),
        follow_redirects=True,
    )
    response.raise_for_status()
    payload = response.json()
    raw_models = payload.get("data", []) if isinstance(payload, dict) else []
    ids = sorted({model_id for item in raw_models if (model_id := read_model_id(item))})
    if not ids:
        raise ValueError("provider /models response did not contain any model ids")

    old_models = provider.get("models", [])
    metadata = {
        item.get("id"): item
        for item in old_models
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    } if isinstance(old_models, list) else {}
    provider["models"] = [{**metadata[model_id], "id": model_id} if model_id in metadata else model_id for model_id in ids]
    save_config(config, local_config_path())
    return {"provider": name, "models": ids, "count": len(ids)}


def resolve_model_selection(request: ModelRequest) -> str:
    raw_model = request.model.strip()
    if not raw_model:
        raise ValueError("model is required for switch")
    if ":" in raw_model:
        provider_name, model_name = raw_model.split(":", 1)
    else:
        config = load_config()
        provider_name = request.provider.strip() or str(config.get("default_provider") or "")
        model_name = raw_model
    configured = {item["value"] for item in list_model_items()}
    selected = f"{provider_name}:{model_name}"
    if configured and selected not in configured:
        raise ValueError(f"model is not in the configured model list: {selected}")
    get_model_config(selected)
    return selected


def format_result(payload: dict[str, Any]) -> str:
    return "modelResult:\n" + json.dumps(payload, ensure_ascii=False)


def read_model_id(item: object) -> str:
    value = item.get("id") if isinstance(item, dict) else item
    return value.strip() if isinstance(value, str) else ""
