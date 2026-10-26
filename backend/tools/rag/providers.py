"""Calls to a configured provider's non-chat endpoints (`/embeddings`, `/rerank`)."""

from __future__ import annotations

import time
from typing import Any

import httpx

from agent.config import get_model_config
from agent.debug_log import log_event

RETRY_STATUSES = {429, 500, 502, 503, 504}


def post_json(model: str, path: str, payload: dict[str, Any], *, timeout: float = 60.0, retries: int = 2) -> dict[str, Any]:
    """POST to `<provider base url><path>` for a `provider:model` and return the JSON object."""

    config = get_model_config(model)
    api_key = config.api_key_value
    if not api_key and config.provider != "ollama":
        raise ValueError(f"Missing API key for provider {config.provider}. Set {config.api_key_env or 'its key'} in backend/.env.")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {**payload, "model": config.model_id}
    last = ""
    for attempt in range(retries + 1):
        try:
            response = httpx.post(f"{config.base_url}{path}", headers=headers, json=body, timeout=timeout)
        except httpx.HTTPError as exc:
            last = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code < 400:
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError(f"{path} returned non-object JSON.")
                return data
            last = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code not in RETRY_STATUSES:
                break
        if attempt < retries:
            time.sleep(1.0 + attempt)
    log_event("rag.provider_error", provider=config.provider, model=config.model_id, path=path, error=last)
    raise RuntimeError(f"{config.provider}{path} failed: {last}")
