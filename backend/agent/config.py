"""Load model settings from .env and data/api_configs.json."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
CONFIG_PATH = PROJECT_ROOT / "data" / "api_configs.json"
@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model_id: str
    base_url: str
    api_key_env: str | None = None
    api_key: str | None = None

    @property
    def api_key_value(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None


def load_environment() -> None:
    load_dotenv(BACKEND_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / ".env")


def load_config() -> dict[str, Any]:
    load_environment()
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("data/api_configs.json must be a JSON object.")
    return data


def get_model_config(model: str | None = None) -> ModelConfig:
    data = load_config()
    providers = data.get("providers", {})
    if not isinstance(providers, dict):
        raise ValueError("`providers` must be an object in data/api_configs.json.")

    default_provider = str(data.get("default_provider") or "deepseek")
    requested_model = model or os.getenv("AI_AGENT_DEFAULT_MODEL") or data.get("default_model")
    if not requested_model:
        raise ValueError("No model selected.")

    if ":" in requested_model:
        provider_name, model_name = requested_model.split(":", 1)
    else:
        provider_name, model_name = default_provider, str(requested_model)

    provider = providers.get(provider_name)
    if not isinstance(provider, dict):
        raise ValueError(f"Unknown provider: {provider_name}")

    model_id = find_model_id(provider, model_name)
    base_url = str(provider.get("base_url") or provider.get("url") or "").rstrip("/")
    if not base_url:
        raise ValueError(f"Provider {provider_name} is missing base_url.")

    return ModelConfig(
        provider=provider_name,
        model_id=model_id,
        base_url=base_url,
        api_key_env=optional_string(provider.get("api_key_env")),
        api_key=optional_string(provider.get("api_key")),
    )


def find_model_id(provider: dict[str, Any], model_name: str) -> str:
    models = provider.get("models", [])
    if isinstance(models, list):
        for item in models:
            if isinstance(item, str) and item == model_name:
                return item
            if isinstance(item, dict):
                model_id = item.get("id")
                alias = item.get("alias")
                if model_name in {model_id, alias} and isinstance(model_id, str):
                    return model_id

    default_model = provider.get("default_model")
    if model_name == default_model and isinstance(default_model, str):
        return default_model

    return model_name


def load_system_prompt(
    *,
    web_search: bool = False,
    web_search_mode: str | None = None,
    rag_mode: str = "off",
    rag_context: str | None = None,
    web_search_results: list[str] | None = None,
    rag_results: list[str] | None = None,
    conversation_summary: str | None = None,
    include_tool_rules: bool = False,
    include_context_rules: bool = False,
    tool_error: str | None = None,
) -> str:
    from prompts.system import build_system_prompt

    return build_system_prompt(
        web_search=web_search,
        web_search_mode=web_search_mode,
        rag_mode=rag_mode,
        rag_context=rag_context,
        web_search_results=web_search_results,
        rag_results=rag_results,
        conversation_summary=conversation_summary,
        include_tool_rules=include_tool_rules,
        include_context_rules=include_context_rules,
        tool_error=tool_error,
    )


def optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
