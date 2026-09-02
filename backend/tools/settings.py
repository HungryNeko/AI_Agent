"""System-side tool settings.

The model should not see most of these values. They are backend controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


RagMode = Literal["off", "on", "auto"]
WebSearchMode = Literal["off", "auto"]
WebSearchProvider = Literal["duckduckgo", "searxng", "tavily"]


@dataclass(frozen=True)
class WebSearchSettings:
    mode: WebSearchMode = "off"
    provider: WebSearchProvider = "duckduckgo"
    api_key_env: str = "TAVILY_API_KEY"
    base_url: str = ""
    max_results: int = 5
    timeout_seconds: float = 10.0
    search_depth: str = "basic"
    topic: str = "general"
    include_answer: bool = False

    @property
    def can_model_call(self) -> bool:
        return self.mode == "auto"


@dataclass(frozen=True)
class RagSettings:
    mode: RagMode = "off"
    min_similarity: float = 0.75
    max_results: int = 5
    auto_include: bool = False

    @property
    def can_model_call(self) -> bool:
        return self.mode in {"on", "auto"}

    @property
    def can_auto_include_context(self) -> bool:
        return self.mode == "on" or self.auto_include


@dataclass(frozen=True)
class ToolSettings:
    web_search: WebSearchSettings = WebSearchSettings()
    rag: RagSettings = RagSettings()

    def model_view(self) -> dict[str, list[str]]:
        """Return only what the model needs to know."""

        available = []
        if self.web_search.can_model_call:
            available.append("webSearch")
        if self.rag.can_model_call:
            available.append("rag")
        return {"available": available}


def make_tool_settings(
    *,
    web_search: bool | None = None,
    web_search_mode: str = "off",
    rag_mode: str = "off",
    web_search_provider: WebSearchProvider = "duckduckgo",
    web_search_base_url: str | None = None,
    web_search_max_results: int = 5,
    web_search_timeout_seconds: float = 10.0,
    web_search_depth: str = "basic",
    web_search_topic: str = "general",
    web_search_include_answer: bool = False,
    rag_min_similarity: float = 0.75,
    rag_max_results: int = 5,
) -> ToolSettings:
    if web_search is not None:
        web_search_mode = "auto" if web_search else "off"

    provider = normalize_web_search_provider(web_search_provider)
    return ToolSettings(
        web_search=WebSearchSettings(
            mode=normalize_web_search_mode(web_search_mode),
            provider=provider,
            base_url=default_web_search_base_url(provider, web_search_base_url),
            max_results=web_search_max_results,
            timeout_seconds=web_search_timeout_seconds,
            search_depth=web_search_depth,
            topic=web_search_topic,
            include_answer=web_search_include_answer,
        ),
        rag=RagSettings(
            mode=normalize_rag_mode(rag_mode),
            min_similarity=rag_min_similarity,
            max_results=rag_max_results,
        ),
    )


def normalize_web_search_mode(web_search_mode: str) -> WebSearchMode:
    mode = web_search_mode.strip().lower()
    if mode not in {"off", "auto"}:
        raise ValueError("web_search_mode must be one of: off, auto.")
    return mode


def normalize_web_search_provider(provider: str) -> WebSearchProvider:
    value = provider.strip().lower()
    if value not in {"duckduckgo", "searxng", "tavily"}:
        raise ValueError("web_search_provider must be one of: duckduckgo, searxng, tavily.")
    return value


def default_web_search_base_url(
    provider: WebSearchProvider,
    configured_base_url: str | None,
) -> str:
    if configured_base_url:
        return configured_base_url.rstrip("/")
    if provider == "searxng":
        return "http://localhost:8080"
    if provider == "tavily":
        return "https://api.tavily.com"
    return ""


def normalize_rag_mode(rag_mode: str) -> RagMode:
    mode = rag_mode.strip().lower()
    if mode not in {"off", "on", "auto"}:
        raise ValueError("rag_mode must be one of: off, on, auto.")
    return mode
