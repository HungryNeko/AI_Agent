"""LangGraph flow for the small command-line agent demo."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.config import load_system_prompt
from agent.llm import complete_chat_once
from prompts.context import build_context_prompt
from prompts.tools import build_tool_usage_reminder, build_tools_prompt_from_settings
from tools import rag
from tools.executor import execute_tool
from tools.request import build_openai_tools, parse_openai_tool_calls
from tools.settings import ToolSettings, make_tool_settings


Message = dict[str, Any]
Route = Literal["model_respond", "tool_call", "tool_error", "conversation_end"]


class ChatState(TypedDict):
    message: str
    messages: NotRequired[list[Message]]
    model: NotRequired[str]
    system_prompt: NotRequired[str]
    web_search: NotRequired[bool]
    web_search_mode: NotRequired[str]
    web_search_provider: NotRequired[str]
    web_search_base_url: NotRequired[str]
    rag_mode: NotRequired[str]
    rag_context: NotRequired[str]
    web_search_results: NotRequired[list[str]]
    rag_results: NotRequired[list[str]]
    conversation_summary: NotRequired[str]
    include_tool_rules: NotRequired[bool]
    include_context_rules: NotRequired[bool]
    tool_error: NotRequired[str]
    settings: NotRequired[ToolSettings]
    initialized: NotRequired[bool]
    tool_rounds: NotRequired[int]
    max_tool_rounds: NotRequired[int]
    response: NotRequired[str]


def first_state(state: ChatState) -> dict[str, Any]:
    """Initialize backend-only settings and the fixed first system prompt."""

    settings = make_tool_settings(
        web_search=state.get("web_search") if "web_search_mode" not in state else None,
        web_search_mode=state.get("web_search_mode", "off"),
        web_search_provider=state.get("web_search_provider", "duckduckgo"),
        web_search_base_url=state.get("web_search_base_url"),
        rag_mode=state.get("rag_mode", "off"),
    )

    if state.get("initialized") and state.get("messages"):
        return {"settings": settings}

    system_prompt = state.get("system_prompt") or load_system_prompt(
        web_search_mode=settings.web_search.mode,
        rag_mode=settings.rag.mode,
        rag_context=state.get("rag_context"),
        web_search_results=state.get("web_search_results"),
        rag_results=state.get("rag_results"),
        conversation_summary=state.get("conversation_summary"),
        include_tool_rules=True,
        include_context_rules=True,
        tool_error=state.get("tool_error"),
    )

    return {
        "messages": [{"role": "system", "content": system_prompt}],
        "settings": settings,
        "initialized": True,
        "tool_rounds": 0,
        "response": "",
    }


def conversation_begin(state: ChatState) -> dict[str, Any]:
    """Add one user turn plus the small dynamic context visible this turn."""

    settings = state["settings"]
    rag_results = list(state.get("rag_results") or [])
    auto_rag_context = rag.auto_context(state["message"], settings.rag)
    if auto_rag_context:
        rag_results.append(auto_rag_context)

    context_text = build_context_prompt(
        conversation_summary=state.get("conversation_summary"),
    )
    tool_text = build_tools_prompt_from_settings(
        settings,
        rag_context=state.get("rag_context"),
        web_search_results=state.get("web_search_results"),
        rag_results=rag_results,
    )
    user_parts = [part for part in [context_text, tool_text, f'userMessage: "{state["message"]}"'] if part]
    messages = list(state.get("messages") or [])
    messages.append({"role": "user", "content": "\n\n".join(user_parts)})

    return {
        "messages": messages,
        "rag_results": rag_results,
        "tool_error": "",
        "tool_rounds": 0,
        "response": "",
    }


def model_respond(state: ChatState) -> dict[str, Any]:
    """Ask the model once. Routing decides whether this is final or needs tools."""

    if state.get("tool_rounds", 0) >= state.get("max_tool_rounds", 3):
        return {"response": "Error: model kept requesting tools after max_tool_rounds."}

    settings = state["settings"]
    assistant_message = complete_chat_once(
        state["messages"],
        model=state.get("model"),
        tools=build_openai_tools(settings),
    )
    messages = list(state["messages"])
    messages.append(assistant_message)

    return {
        "messages": messages,
        "response": str(assistant_message.get("content") or ""),
    }


def route_after_model(state: ChatState) -> Route:
    last_message = last_assistant_message(state)
    if state.get("response", "").startswith("Error:"):
        return "conversation_end"
    if (last_message.get("tool_calls") or []) and state.get("tool_rounds", 0) < state.get(
        "max_tool_rounds", 3
    ):
        return "tool_call"
    return "conversation_end"


def tool_call(state: ChatState) -> dict[str, Any]:
    """Run all tool calls from the last assistant message."""

    settings = state["settings"]
    try:
        tool_requests = parse_openai_tool_calls(last_assistant_message(state), settings)
    except ValueError as exc:
        return {"tool_error": str(exc)}

    messages = list(state["messages"])
    web_search_results = list(state.get("web_search_results") or [])
    rag_results = list(state.get("rag_results") or [])

    for request in tool_requests:
        result = execute_tool(request, settings)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": request.id,
                "content": result,
            }
        )
        if request.name == "webSearch":
            web_search_results.append(result)
        if request.name == "rag":
            rag_results.append(result)

    return {
        "messages": messages,
        "web_search_results": web_search_results,
        "rag_results": rag_results,
        "tool_rounds": state.get("tool_rounds", 0) + 1,
        "tool_error": "",
    }


def route_after_tool_call(state: ChatState) -> Route:
    if state.get("tool_error"):
        return "tool_error"
    return "model_respond"


def tool_error(state: ChatState) -> dict[str, Any]:
    """Return a tool error message so the model can correct its next step."""

    error = state.get("tool_error") or "unknown tool error"
    reminder = build_tool_usage_reminder(error)
    messages = list(state["messages"])
    tool_call_ids = read_last_tool_call_ids(state)

    if tool_call_ids:
        for tool_call_id in tool_call_ids:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": reminder,
                }
            )
    else:
        messages.append({"role": "user", "content": reminder})

    return {
        "messages": messages,
        "tool_error": "",
        "tool_rounds": state.get("tool_rounds", 0) + 1,
    }


def conversation_end(state: ChatState) -> dict[str, str]:
    return {"response": state.get("response", "")}


def build_graph():
    graph = StateGraph(ChatState)
    graph.add_node("first_state", first_state)
    graph.add_node("conversation_begin", conversation_begin)
    graph.add_node("model_respond", model_respond)
    graph.add_node("tool_call", tool_call)
    graph.add_node("tool_error", tool_error)
    graph.add_node("conversation_end", conversation_end)

    graph.add_edge(START, "first_state")
    graph.add_edge("first_state", "conversation_begin")
    graph.add_edge("conversation_begin", "model_respond")
    graph.add_conditional_edges(
        "model_respond",
        route_after_model,
        {
            "tool_call": "tool_call",
            "conversation_end": "conversation_end",
        },
    )
    graph.add_conditional_edges(
        "tool_call",
        route_after_tool_call,
        {
            "model_respond": "model_respond",
            "tool_error": "tool_error",
        },
    )
    graph.add_edge("tool_error", "model_respond")
    graph.add_edge("conversation_end", END)
    return graph.compile()


def new_chat_state(
    *,
    model: str | None = None,
    system_prompt: str | None = None,
    web_search: bool = False,
    web_search_mode: str = "off",
    web_search_provider: str = "duckduckgo",
    web_search_base_url: str | None = None,
    rag_mode: str = "off",
    rag_context: str | None = None,
    web_search_results: list[str] | None = None,
    rag_results: list[str] | None = None,
    conversation_summary: str | None = None,
    include_tool_rules: bool = False,
    include_context_rules: bool = False,
    tool_error: str | None = None,
    max_tool_rounds: int = 3,
) -> ChatState:
    resolved_web_search_mode = "auto" if web_search and web_search_mode == "off" else web_search_mode
    state: ChatState = {
        "message": "",
        "web_search": web_search,
        "web_search_mode": resolved_web_search_mode,
        "web_search_provider": web_search_provider,
        "rag_mode": rag_mode,
        "max_tool_rounds": max_tool_rounds,
    }
    if web_search_base_url:
        state["web_search_base_url"] = web_search_base_url
    if model:
        state["model"] = model
    if system_prompt:
        state["system_prompt"] = system_prompt
    if rag_context:
        state["rag_context"] = rag_context
    if web_search_results:
        state["web_search_results"] = web_search_results
    if rag_results:
        state["rag_results"] = rag_results
    if conversation_summary:
        state["conversation_summary"] = conversation_summary
    if include_tool_rules:
        state["include_tool_rules"] = include_tool_rules
    if include_context_rules:
        state["include_context_rules"] = include_context_rules
    if tool_error:
        state["tool_error"] = tool_error
    return state


def run_turn(state: ChatState, message: str) -> ChatState:
    app = build_graph()
    next_state = dict(state)
    next_state["message"] = message
    return app.invoke(next_state)


def run_agent(
    message: str,
    *,
    model: str | None = None,
    system_prompt: str | None = None,
    web_search: bool = False,
    web_search_mode: str = "off",
    web_search_provider: str = "duckduckgo",
    web_search_base_url: str | None = None,
    rag_mode: str = "off",
    rag_context: str | None = None,
    web_search_results: list[str] | None = None,
    rag_results: list[str] | None = None,
    conversation_summary: str | None = None,
    include_tool_rules: bool = False,
    include_context_rules: bool = False,
    tool_error: str | None = None,
    max_tool_rounds: int = 3,
) -> str:
    state = new_chat_state(
        model=model,
        system_prompt=system_prompt,
        web_search=web_search,
        web_search_mode=web_search_mode,
        web_search_provider=web_search_provider,
        web_search_base_url=web_search_base_url,
        rag_mode=rag_mode,
        rag_context=rag_context,
        web_search_results=web_search_results,
        rag_results=rag_results,
        conversation_summary=conversation_summary,
        include_tool_rules=include_tool_rules,
        include_context_rules=include_context_rules,
        tool_error=tool_error,
        max_tool_rounds=max_tool_rounds,
    )
    result = run_turn(state, message)
    return result.get("response", "")


def last_assistant_message(state: ChatState) -> Message:
    for message in reversed(state.get("messages") or []):
        if message.get("role") == "assistant":
            return message
    return {}


def read_last_tool_call_ids(state: ChatState) -> list[str]:
    ids: list[str] = []
    for tool_call in last_assistant_message(state).get("tool_calls") or []:
        if isinstance(tool_call, dict) and isinstance(tool_call.get("id"), str):
            ids.append(tool_call["id"])
    return ids


# first_state: initialize fixed prompt rules -> conversation_begin
# conversation_begin: previous messages + user message + auto tool context + available tools -> model_respond
# model_respond: model answers or requests tools -> conversation_end or tool_call
# conversation_end: return the answer, ready for the next CLI input
# tool_call: execute one or more tool calls and return results to the model -> model_respond or tool_error
# tool_error: tell the model why the call failed and remind it of the right format -> model_respond
# compact: future node; summarize old messages when the context becomes too long -> conversation_begin
