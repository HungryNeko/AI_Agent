"""Small OpenAI-compatible Chat Completions client."""

from __future__ import annotations

from typing import Any

import httpx

from agent.config import get_model_config


def chat(
    message: str,
    *,
    model: str | None = None,
    system_prompt: str | None = None,
    web_search: bool = False,
    web_search_mode: str | None = None,
    web_search_provider: str = "duckduckgo",
    web_search_base_url: str | None = None,
    rag_mode: str = "auto",
    curl: bool = False,
    curl_mode: str = "auto",
    python: bool = False,
    python_mode: str = "auto",
    file_editor: bool = False,
    file_editor_mode: str = "auto",
    file_editor_approval: str = "auto",
    mcp: bool = False,
    mcp_mode: str = "auto",
    rag_context: str | None = None,
    web_search_results: list[str] | None = None,
    rag_results: list[str] | None = None,
    conversation_summary: str | None = None,
    include_tool_rules: bool = False,
    include_context_rules: bool = False,
    tool_error: str | None = None,
    max_tool_rounds: int = 20,
) -> str:
    """Compatibility wrapper for one user message.

    The real agent loop lives in agent.graph. Keeping this function makes older
    tests and quick imports continue to work.
    """

    from agent.graph import run_agent

    return run_agent(
        message,
        model=model,
        system_prompt=system_prompt,
        web_search=web_search,
        web_search_mode=web_search_mode or "auto",
        web_search_provider=web_search_provider,
        web_search_base_url=web_search_base_url,
        rag_mode=rag_mode,
        curl=curl,
        curl_mode=curl_mode,
        python=python,
        python_mode=python_mode,
        file_editor=file_editor,
        file_editor_mode=file_editor_mode,
        file_editor_approval=file_editor_approval,
        mcp=mcp,
        mcp_mode=mcp_mode,
        rag_context=rag_context,
        web_search_results=web_search_results,
        rag_results=rag_results,
        conversation_summary=conversation_summary,
        include_tool_rules=include_tool_rules,
        include_context_rules=include_context_rules,
        tool_error=tool_error,
        max_tool_rounds=max_tool_rounds,
    )


def complete_chat_once(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Send one request and return the assistant message."""

    config = get_model_config(model)
    api_key = config.api_key_value
    if not api_key and config.provider != "ollama":
        raise ValueError(f"Missing API key. Set {config.api_key_env} in backend/.env.")

    payload = build_chat_payload(config.model_id, messages, tools=tools)
    data = post_chat_completion(config.base_url, api_key, payload)
    return read_assistant_message(data)


def build_chat_payload(
    model_id: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return payload


def post_chat_completion(
    base_url: str,
    api_key: str | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers=make_headers(api_key),
        json=payload,
        timeout=60,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"LLM API error {response.status_code}: {response.text}")

    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("LLM API returned non-object JSON.")
    return data


def read_assistant_message(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM API response has no choices.")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("LLM API choice is invalid.")
    message = first.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("LLM API choice has no assistant message.")
    return message


def make_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers
