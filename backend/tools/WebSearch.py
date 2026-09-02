"""Web search tool implementation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from dotenv import load_dotenv
import httpx

from tools.settings import WebSearchSettings


def search(query: str, settings: WebSearchSettings) -> list[dict[str, str]]:
    """Search the web using the configured provider."""

    if settings.provider == "duckduckgo":
        return duckduckgo_search(query, settings)
    if settings.provider == "searxng":
        return searxng_search(query, settings)
    if settings.provider == "tavily":
        return tavily_search(query, settings)
    raise ValueError(f"Unsupported web search provider: {settings.provider}")


def duckduckgo_search(query: str, settings: WebSearchSettings) -> list[dict[str, str]]:
    """Search DuckDuckGo through the ddgs package. No API key is required."""

    DDGS = import_ddgs()
    with DDGS() as client:
        raw_results = client.text(
            query,
            safesearch="moderate",
            max_results=settings.max_results,
            backend="duckduckgo",
        )
    return normalize_duckduckgo_results(list(raw_results or []))


def searxng_search(query: str, settings: WebSearchSettings) -> list[dict[str, str]]:
    """Search a SearXNG instance. No API key is required unless the instance adds one."""

    load_tool_env()
    query_url = os.getenv("SEARXNG_QUERY_URL")
    if query_url:
        url = query_url.replace("<query>", quote_plus(query))
    else:
        url = f"{settings.base_url.rstrip('/')}/search"

    response = httpx.get(
        url,
        params=None if query_url else {"q": query, "format": "json"},
        timeout=settings.timeout_seconds,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"SearXNG search error {response.status_code}: {response.text}")

    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("SearXNG returned non-object JSON.")
    return normalize_searxng_results(data)[: settings.max_results]


def tavily_search(query: str, settings: WebSearchSettings) -> list[dict[str, str]]:
    api_key = get_api_key(settings.api_key_env)
    if not api_key:
        raise ValueError(f"Missing web search API key. Set {settings.api_key_env} in backend/.env.")

    response = httpx.post(
        f"{settings.base_url.rstrip('/')}/search",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "max_results": settings.max_results,
            "search_depth": settings.search_depth,
            "topic": settings.topic,
            "include_answer": settings.include_answer,
        },
        timeout=settings.timeout_seconds,
    )

    if response.status_code >= 400:
        raise RuntimeError(f"Tavily search error {response.status_code}: {response.text}")

    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Tavily returned non-object JSON.")

    return normalize_tavily_results(data)


def normalize_tavily_results(data: dict[str, Any]) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []

    answer = data.get("answer")
    if isinstance(answer, str) and answer.strip():
        results.append(
            {
                "title": "Tavily answer",
                "url": "",
                "snippet": answer.strip(),
            }
        )

    raw_results = data.get("results", [])
    if not isinstance(raw_results, list):
        return results

    for item in raw_results:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": text_or_empty(item.get("title")),
                "url": text_or_empty(item.get("url")),
                "snippet": text_or_empty(item.get("content") or item.get("snippet")),
                "score": text_or_empty(item.get("score")),
            }
        )

    return results


def normalize_duckduckgo_results(raw_results: list[dict[str, Any]]) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": text_or_empty(item.get("title")),
                "url": text_or_empty(item.get("href") or item.get("url")),
                "snippet": text_or_empty(item.get("body") or item.get("snippet")),
            }
        )
    return results


def normalize_searxng_results(data: dict[str, Any]) -> list[dict[str, str]]:
    raw_results = data.get("results", [])
    if not isinstance(raw_results, list):
        return []

    results: list[dict[str, str]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": text_or_empty(item.get("title")),
                "url": text_or_empty(item.get("url")),
                "snippet": text_or_empty(item.get("content")),
                "score": text_or_empty(item.get("score")),
            }
        )
    return results


def get_api_key(env_name: str) -> str | None:
    load_tool_env()
    value = os.getenv(env_name)
    return value.strip() if value else None


def load_tool_env() -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / "backend" / ".env")
    load_dotenv(project_root / ".env")


def import_ddgs():
    try:
        from ddgs import DDGS
    except ModuleNotFoundError as exc:
        raise ValueError("DuckDuckGo search needs `ddgs`. Install it with: pip install ddgs") from exc
    return DDGS


def text_or_empty(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()
