"""OpenAI-compatible tool schemas and tool-call parsing."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal

from tools.settings import ToolSettings


ToolName = Literal["webSearch", "rag"]


@dataclass(frozen=True)
class ToolRequest:
    id: str
    name: ToolName
    query: str


def build_openai_tools(settings: ToolSettings) -> list[dict[str, Any]]:
    """Build the `tools` payload for Chat Completions."""

    tools = []
    if settings.web_search.can_model_call:
        tools.append(
            function_tool(
                name="webSearch",
                description="Search the public web when current or external information is needed.",
            )
        )
    if settings.rag.can_model_call:
        tools.append(
            function_tool(
                name="rag",
                description="Search local/private knowledge when project or stored context is needed.",
            )
        )
    return tools


def parse_openai_tool_calls(
    message: dict[str, Any],
    settings: ToolSettings,
) -> list[ToolRequest]:
    """Parse assistant `tool_calls` from an OpenAI-compatible response."""

    raw_tool_calls = message.get("tool_calls") or []
    if not isinstance(raw_tool_calls, list):
        raise ValueError("assistant tool_calls must be a list.")

    return [parse_one_tool_call(raw_tool_call, settings) for raw_tool_call in raw_tool_calls]


def parse_one_tool_call(raw_tool_call: object, settings: ToolSettings) -> ToolRequest:
    if not isinstance(raw_tool_call, dict):
        raise ValueError("tool_call must be an object.")

    call_id = raw_tool_call.get("id")
    function = raw_tool_call.get("function")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("tool_call.id is required.")
    if not isinstance(function, dict):
        raise ValueError("tool_call.function is required.")

    name = function.get("name")
    if name not in {"webSearch", "rag"}:
        raise ValueError(f"Unknown tool: {name}")

    arguments = function.get("arguments") or "{}"
    if not isinstance(arguments, str):
        raise ValueError("tool_call.function.arguments must be a JSON string.")

    try:
        parsed_arguments = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError("tool arguments must be valid JSON.") from exc

    if not isinstance(parsed_arguments, dict):
        raise ValueError("tool arguments must be a JSON object.")

    query = parsed_arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("tool arguments must include non-empty string field `query`.")

    validate_tool_allowed(name, settings)
    return ToolRequest(id=call_id, name=name, query=query.strip())


def validate_tool_allowed(name: str, settings: ToolSettings) -> None:
    if name == "webSearch" and not settings.web_search.can_model_call:
        raise ValueError("webSearch can only be called when web_search_mode is auto.")
    if name == "rag" and not settings.rag.can_model_call:
        raise ValueError("rag can only be called when rag_mode is on or auto.")


def function_tool(*, name: ToolName, description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
