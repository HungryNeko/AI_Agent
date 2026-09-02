"""Execute model-requested tools."""

from __future__ import annotations

from typing import Any

from tools import WebSearch
from tools import rag
from tools.request import ToolRequest
from tools.settings import ToolSettings


def execute_tool(request: ToolRequest, settings: ToolSettings) -> str:
    """Run one tool call and return content for a `role=tool` message."""

    try:
        if request.name == "webSearch":
            results = WebSearch.search(request.query, settings.web_search)
            return format_web_search_results(results)
        if request.name == "rag":
            results = rag.search(request.query, settings.rag)
            return format_rag_results(results)
    except NotImplementedError as exc:
        return f'toolError: "{exc}"'
    except Exception as exc:
        return f'toolError: "{request.name} failed: {exc}"'

    return f'toolError: "unknown tool: {request.name}"'


def format_web_search_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return 'webSearchResult: "no results"'
    lines = ["webSearchResult:"]
    for index, item in enumerate(results, start=1):
        title = item.get("title") or "untitled"
        url = item.get("url") or ""
        snippet = item.get("snippet") or item.get("content") or ""
        lines.append(f"{index}. {title} {url} {snippet}".strip())
    return "\n".join(lines)


def format_rag_results(results: list[str]) -> str:
    if not results:
        return 'ragResult: "no results"'
    lines = ["ragResult:"]
    for index, item in enumerate(results, start=1):
        lines.append(f"{index}. {item}")
    return "\n".join(lines)
