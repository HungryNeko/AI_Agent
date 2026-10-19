"""Provider-hosted web search and page fetch (DeepSeek Search / Fetch API).

Some providers do not switch search on inside the chat request; they offer search and fetch as separate
authenticated endpoints. Those are exposed to the model as the `nativeWebSearch` and `webFetch` tools and
called with the provider's own API key, so the answer comes from the provider's engine instead of a
third-party one. Providers that do switch it on inside the request (OpenRouter, DashScope) are handled
in `agent.llm` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import httpx

MAX_RESULTS = 20
SUMMARY_CHARS = 900
FETCH_CHARS = 24_000
TIMEOUT_SECONDS = 40.0


@dataclass(frozen=True)
class NativeWebRequest:
    kind: Literal["search", "fetch"]
    query: str = ""
    url: str = ""
    count: int = 5


def endpoint_url(base_url: str, path: str) -> str:
    """`https://api.deepseek.com/v1` + `/v1/tools/search` -> `https://api.deepseek.com/v1/tools/search`."""

    root = base_url.rstrip("/")
    if root.endswith("/v1") and path.startswith("/v1/"):
        root = root[: -len("/v1")]
    return root + path


def execute(request: NativeWebRequest, settings: Any) -> str:
    from agent.config import get_model_config

    config = get_model_config(str(getattr(settings, "model", "") or "") or None)
    endpoints = (config.native_web or {}).get("endpoints") or {}
    path = endpoints.get(request.kind)
    if not path:
        raise ValueError(f"the current provider has no native web {request.kind} configured.")
    api_key = config.api_key_value
    if not api_key:
        raise ValueError(f"missing API key for provider {config.provider}.")
    if request.kind == "search":
        body: dict[str, Any] = {"query": request.query, "count": max(1, min(request.count, MAX_RESULTS))}
    else:
        body = {"url": request.url}
    response = httpx.post(
        endpoint_url(config.base_url, path),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=TIMEOUT_SECONDS,
    )
    if response.status_code >= 400:
        raise ValueError(describe_error(response))
    data = response.json()
    return format_search(data) if request.kind == "search" else format_fetch(data, request.url)


def describe_error(response: httpx.Response) -> str:
    try:
        error = response.json().get("error") or {}
        message = f"{error.get('message') or response.text[:300]}"
    except (ValueError, AttributeError):
        message = response.text[:300]
    hints = {
        401: "the API key was rejected",
        402: "insufficient balance",
        403: "the page is blocked by the provider's policy",
        404: "no usable content was found",
        429: "rate limit reached (5 requests per second); wait a moment",
    }
    hint = hints.get(response.status_code)
    return f"native web request failed (HTTP {response.status_code}): {message}" + (f" [{hint}]" if hint else "")


def format_search(data: Any) -> str:
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list) or not results:
        return "nativeWebSearchResult:\nno results."
    lines = ["nativeWebSearchResult:"]
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("summary") or item.get("snippet") or item.get("content") or "").strip()
        head = f"{index}. {str(item.get('title') or '').strip()} - {item.get('url', '')}"
        meta = " | ".join(str(item[key]) for key in ("site_name", "date") if item.get(key))
        lines.append(head + (f" ({meta})" if meta else ""))
        if text:
            lines.append("   " + text[:SUMMARY_CHARS].replace("\n", " "))
    return "\n".join(lines)


def format_fetch(data: Any, url: str) -> str:
    if not isinstance(data, dict):
        return "webFetchResult:\nno content."
    content = str(data.get("content") or "").strip()
    truncated = len(content) > FETCH_CHARS
    lines = ["webFetchResult:", f"url: {url}"]
    if data.get("title"):
        lines.append(f"title: {data['title']}")
    lines.append("content: " + (content[:FETCH_CHARS] if content else "(empty)"))
    if truncated:
        lines.append(f"(truncated to {FETCH_CHARS} characters of {len(content)})")
    return "\n".join(lines)
