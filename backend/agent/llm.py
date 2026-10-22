"""Small OpenAI-compatible Chat Completions client."""

from __future__ import annotations

import threading
from typing import Any

import httpx

from agent.config import get_model_config, models_to_try
from agent.debug_log import log_event, log_exception


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
    rag_include_knowledge: bool = True,
    rag_include_memory: bool = True,
    rag_include_skills: bool = True,
    curl: bool = False,
    curl_mode: str = "auto",
    python: bool = False,
    python_mode: str = "auto",
    file_reader: bool = False,
    file_reader_mode: str = "auto",
    file_editor: bool = False,
    file_editor_mode: str = "auto",
    file_editor_approval: str = "auto",
    mcp: bool = False,
    mcp_mode: str = "auto",
    history: bool = False,
    history_mode: str = "off",
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
        rag_include_knowledge=rag_include_knowledge,
        rag_include_memory=rag_include_memory,
        rag_include_skills=rag_include_skills,
        curl=curl,
        curl_mode=curl_mode,
        python=python,
        python_mode=python_mode,
        file_reader=file_reader,
        file_reader_mode=file_reader_mode,
        file_editor=file_editor,
        file_editor_mode=file_editor_mode,
        file_editor_approval=file_editor_approval,
        mcp=mcp,
        mcp_mode=mcp_mode,
        history=history,
        history_mode=history_mode,
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
    tool_choice: str | dict[str, Any] | None = None,
    max_tokens: int | None = None,
    native_web: bool = False,
    reasoning_effort: str = "default",
) -> dict[str, Any]:
    """Send one request and return the assistant message.

    The default model falls back along the user's priority chain when a request fails;
    an explicitly chosen non-default model is tried alone.
    """

    candidates = models_to_try(model)
    last_error: Exception | None = None
    for index, candidate in enumerate(candidates):
        try:
            extra: dict[str, Any] = {"native_web": True} if native_web else {}
            if reasoning_effort and reasoning_effort != "default":
                extra["reasoning_effort"] = reasoning_effort
            return complete_chat_with_model(
                messages,
                model=candidate,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                **extra,
            )
        except Exception as exc:  # noqa: BLE001 - any failure of this model moves on to the next
            last_error = exc
            if index + 1 < len(candidates):
                log_event("llm.fallback", failed=candidate, next=candidates[index + 1], error=str(exc))
    assert last_error is not None
    raise last_error


def complete_chat_with_model(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    max_tokens: int | None = None,
    native_web: bool = False,
    reasoning_effort: str = "default",
) -> dict[str, Any]:
    """Send one request to exactly one model."""

    config = get_model_config(model)
    api_key = config.api_key_value
    if not api_key and config.provider != "ollama":
        raise ValueError(f"Missing API key. Set {config.api_key_env} in backend/.env.")

    payload = build_chat_payload(
        config.model_id,
        messages,
        tools=tools,
        tool_choice=tool_choice,
        max_tokens=max_tokens,
    )
    if native_web:
        if getattr(config, "native_web", None):
            apply_native_web(payload, config.native_web)
        else:
            log_event("llm.native_web_unavailable", provider=config.provider, model=config.model_id)
    log_event(
        "llm.request",
        provider=config.provider,
        base_url=config.base_url,
        model=config.model_id,
        message_count=len(messages),
        tool_count=len(tools or []),
        payload=payload,
    )
    reasoning_keys = apply_reasoning(payload, getattr(config, "reasoning", ""), reasoning_effort)
    if getattr(config, "reasoning", "") == "deepseek":
        payload["messages"] = with_reasoning_content(payload["messages"])
    try:
        try:
            data = post_chat_completion(config.base_url, api_key, payload)
        except RuntimeError as exc:
            # A model that does not take the thinking setting answers 400: try again without it.
            if not reasoning_keys or "error 400" not in str(exc):
                raise
            log_event("llm.reasoning_rejected", provider=config.provider, model=config.model_id, error=str(exc)[:300])
            for key in reasoning_keys:
                payload.pop(key, None)
            data = post_chat_completion(config.base_url, api_key, payload)
    except Exception as exc:
        log_exception(
            "llm.error",
            exc,
            provider=config.provider,
            base_url=config.base_url,
            model=config.model_id,
            message_count=len(messages),
            tool_count=len(tools or []),
        )
        raise
    _USAGE.last = normalize_usage(data.get("usage"))
    log_event(
        "llm.response",
        provider=config.provider,
        base_url=config.base_url,
        model=config.model_id,
        response=data,
        usage=_USAGE.last,
    )
    return read_assistant_message(data)


def build_chat_payload(
    model_id: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"
    if max_tokens is not None:
        payload["max_tokens"] = max(1, int(max_tokens))
    return payload


_USAGE = threading.local()


def normalize_usage(raw: Any) -> dict[str, int] | None:
    """Token usage in one shape across providers.

    `cached` is the part of the prompt served from the provider's prompt cache; it is left out when the
    provider does not report it (DeepSeek: prompt_cache_hit_tokens, OpenAI-style: prompt_tokens_details
    .cached_tokens, Anthropic-style: cache_read_input_tokens).
    """

    if not isinstance(raw, dict):
        return None
    usage: dict[str, int] = {
        "prompt": int(raw.get("prompt_tokens") or 0),
        "completion": int(raw.get("completion_tokens") or 0),
    }
    details = raw.get("prompt_tokens_details")
    cached = raw.get("prompt_cache_hit_tokens")
    if cached is None and isinstance(details, dict):
        cached = details.get("cached_tokens")
    if cached is None:
        cached = raw.get("cache_read_input_tokens")
    if cached is not None:
        usage["cached"] = int(cached or 0)
    return usage


def consume_usage() -> dict[str, int] | None:
    """Usage of the last model request made on this thread (cleared by reading it)."""

    usage = getattr(_USAGE, "last", None)
    _USAGE.last = None
    return usage


def add_usage(total: dict[str, int] | None, usage: dict[str, int] | None) -> dict[str, int] | None:
    if not usage:
        return total
    merged = dict(total or {"prompt": 0, "completion": 0, "requests": 0})
    merged["prompt"] = merged.get("prompt", 0) + usage.get("prompt", 0)
    merged["completion"] = merged.get("completion", 0) + usage.get("completion", 0)
    merged["requests"] = merged.get("requests", 0) + 1
    if "cached" in usage:
        merged["cached"] = merged.get("cached", 0) + usage["cached"]
    return merged


REASONING_LEVELS = {"low", "medium", "high", "max"}
DASHSCOPE_BUDGETS = {"low": 1024, "medium": 4096, "high": 16384, "max": 32768}


def apply_reasoning(payload: dict[str, Any], kind: str, effort: str) -> list[str]:
    """Put the thinking effort into the request the way this provider expects; returns the keys it added."""

    if not effort or effort == "default" or kind in {"", "none"}:
        return []
    if effort != "off" and effort not in REASONING_LEVELS:
        return []
    before = set(payload)
    if kind == "deepseek":
        if effort == "off":
            payload["thinking"] = {"type": "disabled"}
        else:
            payload["thinking"] = {"type": "enabled"}
            payload["reasoning_effort"] = {"low": "low", "medium": "high", "high": "high", "max": "max"}[effort]
    elif kind == "openai":
        if effort == "off":
            return []  # always-reasoning models cannot be switched off; leave the provider default
        payload["reasoning_effort"] = "high" if effort == "max" else effort
    elif kind == "openrouter":
        payload["reasoning"] = {"effort": "none" if effort == "off" else effort}
    elif kind == "dashscope":
        payload["enable_thinking"] = effort != "off"
        if effort != "off":
            payload["thinking_budget"] = DASHSCOPE_BUDGETS[effort]
    return [key for key in payload if key not in before]


def with_reasoning_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """DeepSeek thinking mode rejects assistant tool-call messages that carry no `reasoning_content`.

    Such messages exist whenever the conversation switched model or thinking effort mid-turn (the earlier
    calls were made without thinking), so give them an empty one. Stored history is left untouched.
    """

    fixed = []
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls") and "reasoning_content" not in message:
            message = {**message, "reasoning_content": ""}
        fixed.append(message)
    return fixed


PROTECTED_PAYLOAD_KEYS = {"model", "messages", "stream"}


def apply_native_web(payload: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    """Switch on a provider's own web search: merge its request fields and tools into the payload."""

    for key, value in (native.get("body") or {}).items():
        if key not in PROTECTED_PAYLOAD_KEYS:
            payload[key] = value
    extra_tools = native.get("tools") or []
    if extra_tools:
        payload["tools"] = [*(payload.get("tools") or []), *extra_tools]
        payload.setdefault("tool_choice", "auto")
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
