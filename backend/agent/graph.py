"""LangGraph flow for the small command-line agent demo."""

from __future__ import annotations

import base64
import mimetypes
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.config import PROJECT_ROOT, load_system_prompt
from agent.debug_log import log_event, log_exception
from agent.instructions import load_instruction
from agent.llm import complete_chat_once
from agent.references import resolve_reference_context
from prompts.context import build_context_prompt, format_current_time
from prompts.tools import build_tool_usage_reminder, build_tools_prompt_from_settings
from tools import rag
from tools.executor import execute_tool
from tools.request import ToolRequest, build_openai_tools, parse_openai_tool_calls
from tools.settings import ToolSettings, make_tool_settings

Message = dict[str, Any]
Route = Literal["assistant_step", "tool_call", "tool_error", "conversation_end"]


class AgentEvent(TypedDict, total=False):
    type: str
    text: str
    tool: str
    query: str
    url: str
    code: str
    state: ChatState


class ChatState(TypedDict):
    message: str
    messages: NotRequired[list[Message]]
    model: NotRequired[str]
    system_prompt: NotRequired[str]
    web_search: NotRequired[bool]
    web_search_mode: NotRequired[str]
    web_search_provider: NotRequired[str]
    web_search_auto_switch: NotRequired[bool]
    web_search_base_url: NotRequired[str]
    rag_mode: NotRequired[str]
    rag_include_knowledge: NotRequired[bool]
    rag_include_memory: NotRequired[bool]
    rag_include_skills: NotRequired[bool]
    curl: NotRequired[bool]
    curl_mode: NotRequired[str]
    python: NotRequired[bool]
    python_mode: NotRequired[str]
    file_editor: NotRequired[bool]
    file_editor_mode: NotRequired[str]
    file_editor_approval: NotRequired[str]
    mcp: NotRequired[bool]
    mcp_mode: NotRequired[str]
    history: NotRequired[bool]
    history_mode: NotRequired[str]
    automation: NotRequired[bool]
    automation_mode: NotRequired[str]
    attachments: NotRequired[list[dict[str, Any]]]
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
    tool_events: NotRequired[list[AgentEvent]]


def first_state(state: ChatState) -> dict[str, Any]:
    """Initialize backend-only settings and the fixed first system prompt."""

    settings = make_tool_settings(
        web_search=state.get("web_search") if "web_search_mode" not in state else None,
        web_search_mode=state.get("web_search_mode", "auto"),
        web_search_provider=state.get("web_search_provider", "duckduckgo"),
        web_search_auto_switch=state.get("web_search_auto_switch", False),
        web_search_base_url=state.get("web_search_base_url"),
        rag_mode=state.get("rag_mode", "auto"),
        rag_include_knowledge=state.get("rag_include_knowledge", True),
        rag_include_memory=state.get("rag_include_memory", True),
        rag_include_skills=state.get("rag_include_skills", True),
        curl=state.get("curl") if "curl_mode" not in state else None,
        curl_mode=state.get("curl_mode", "auto"),
        python=state.get("python") if "python_mode" not in state else None,
        python_mode=state.get("python_mode", "auto"),
        file_editor=state.get("file_editor") if "file_editor_mode" not in state else None,
        file_editor_mode=state.get("file_editor_mode", "auto"),
        file_editor_approval=state.get("file_editor_approval", "auto"),
        mcp=state.get("mcp") if "mcp_mode" not in state else None,
        mcp_mode=state.get("mcp_mode", "auto"),
        history=state.get("history") if "history_mode" not in state else None,
        history_mode=state.get("history_mode", "off"),
        automation=state.get("automation") if "automation_mode" not in state else None,
        automation_mode=state.get("automation_mode", "off"),
    )

    if state.get("initialized") and state.get("messages"):
        return {"settings": settings}

    system_prompt = load_system_prompt(
        web_search_mode=settings.web_search.mode,
        rag_mode=settings.rag.mode,
        curl_mode=settings.curl.mode,
        python_mode=settings.python.mode,
        file_editor_mode=settings.file_editor.mode,
        mcp_mode=settings.mcp.mode,
        history_mode=settings.history.mode,
        instruction_text=load_instruction(),
        include_tool_rules=True,
        include_context_rules=True,
    )
    if state.get("system_prompt"):
        system_prompt = f"{system_prompt}\n\nuserSystemPrompt:\n{state['system_prompt'].strip()}"

    return {
        "messages": [{"role": "system", "content": system_prompt}],
        "settings": settings,
        "initialized": True,
        "tool_rounds": 0,
        "response": "",
        "tool_events": [],
    }


def conversation_begin(state: ChatState) -> dict[str, Any]:
    """Add one user turn plus the small dynamic context visible this turn."""

    settings = state["settings"]
    log_event(
        "turn.begin",
        message=state["message"],
        model=state.get("model"),
        max_tool_rounds=state.get("max_tool_rounds", 20),
        tools=settings.model_view()["available"],
    )
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
    time_text = format_current_time()
    reference_text = resolve_reference_context(state["message"])
    attachment_text, image_parts = build_attachment_context(state.get("attachments") or [])
    user_parts = [
        part
        for part in [
            time_text,
            context_text,
            tool_text,
            reference_text,
            attachment_text,
            f'userMessage: "{state["message"]}"',
        ]
        if part
    ]
    messages = list(state.get("messages") or [])
    user_text = "\n\n".join(user_parts)
    content: str | list[dict[str, Any]]
    if image_parts:
        content = [{"type": "text", "text": user_text}, *image_parts]
    else:
        content = user_text
    messages.append({"role": "user", "content": content})

    return {
        "messages": messages,
        "rag_results": rag_results,
        "tool_error": "",
        "tool_rounds": 0,
        "response": "",
        "tool_events": [],
    }


def build_attachment_context(attachments: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    lines: list[str] = []
    image_parts: list[dict[str, Any]] = []
    if not attachments:
        return "", []
    lines.append("uploadedAttachments:")
    for index, attachment in enumerate(attachments, start=1):
        path_text = str(attachment.get("path") or "")
        filename = str(attachment.get("filename") or Path(path_text).name or f"upload-{index}")
        content_type = str(attachment.get("content_type") or mimetypes.guess_type(filename)[0] or "")
        try:
            path = resolve_upload_path(path_text)
        except ValueError as exc:
            lines.append(f"- {filename}: unavailable ({exc})")
            continue
        lines.append(
            f"- {filename}\n"
            f"  path: {relative_to_project(path)}\n"
            f"  mime: {content_type or 'application/octet-stream'}\n"
            f"  sizeBytes: {path.stat().st_size}"
        )
        if content_type.startswith("image/"):
            image_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{content_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}",
                    },
                }
            )
        elif path.suffix.lower() in {".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".tsv", ".log", ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css"}:
            text = path.read_text(encoding="utf-8", errors="replace")[:12000]
            lines.append(f"  textPreview:\n{text}")
    return "\n".join(lines), image_parts


def resolve_upload_path(path_text: str) -> Path:
    root = (PROJECT_ROOT / "backend" / "runtime" / "uploads").resolve()
    raw = Path(path_text)
    path = raw if raw.is_absolute() else PROJECT_ROOT / raw
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("path is outside backend/runtime/uploads") from exc
    if not resolved.is_file():
        raise ValueError("file not found")
    return resolved


def relative_to_project(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def assistant_step(state: ChatState) -> dict[str, Any]:
    """Ask the model once. Routing decides whether this is final or needs tools."""

    settings = state["settings"]
    messages = list(state["messages"])
    tools = build_openai_tools(settings)
    if tool_budget_reached(state):
        log_event(
            "tool_budget.exceeded",
            tool_rounds=state.get("tool_rounds", 0),
            max_tool_rounds=state.get("max_tool_rounds", 20),
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "toolBudgetExceeded: max_tool_rounds reached for this user turn. "
                    "No more tool calls are available now. Give the best final answer "
                    "from the tool results and raw tool errors above. If live data is "
                    "unavailable, say that clearly."
                ),
            }
        )
        tools = None

    request_messages = list(messages)
    assistant_message = complete_chat_once(
        request_messages,
        model=state.get("model"),
        tools=tools,
    )
    messages = request_messages + [assistant_message]

    has_tool_calls = bool(assistant_message.get("tool_calls") or [])
    if tools is None and has_tool_calls:
        return {
            "messages": messages,
            "response": str(assistant_message.get("content") or "Unable to produce a final answer."),
        }

    return {
        "messages": messages,
        "response": "" if has_tool_calls else str(assistant_message.get("content") or ""),
    }


def route_after_assistant_step(state: ChatState) -> Route:
    last_message = last_assistant_message(state)
    if state.get("response", "").startswith("Error:"):
        return "conversation_end"
    if (last_message.get("tool_calls") or []) and tool_budget_allows(state):
        return "tool_call"
    return "conversation_end"


def tool_call(state: ChatState) -> dict[str, Any]:
    """Run all tool calls from the last assistant message."""

    settings = state["settings"]
    try:
        tool_requests = parse_openai_tool_calls(last_assistant_message(state), settings)
    except ValueError as exc:
        log_exception("tool.parse_error", exc, assistant_message=last_assistant_message(state))
        return {"tool_error": str(exc)}

    messages = list(state["messages"])
    web_search_results = list(state.get("web_search_results") or [])
    rag_results = list(state.get("rag_results") or [])
    tool_events: list[AgentEvent] = []

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
        if result.startswith("toolError:"):
            tool_events.append({"type": "error", "text": result})
        elif request.name == "settings":
            tool_events.append({"type": "settings_changed", "tool": "settings", "text": result})
        elif request.name == "fileEditor" and file_editor_approval_required(result):
            tool_events.append({"type": "approval_required", "tool": "fileEditor", "text": result})

    return {
        "messages": messages,
        "web_search_results": web_search_results,
        "rag_results": rag_results,
        "tool_events": tool_events,
        "tool_rounds": state.get("tool_rounds", 0) + 1,
        "tool_error": "",
    }

def route_after_tool_call(state: ChatState) -> Route:
    if state.get("tool_error"):
        return "tool_error"
    return "assistant_step"


def tool_error(state: ChatState) -> dict[str, Any]:
    """Return a tool error message so the model can correct its next step."""

    error = state.get("tool_error") or "unknown tool error"
    log_event("tool.error_prompt", error=error)
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
    graph.add_node("assistant_step", assistant_step)
    graph.add_node("tool_call", tool_call)
    graph.add_node("tool_error", tool_error)
    graph.add_node("conversation_end", conversation_end)

    graph.add_edge(START, "first_state")
    graph.add_edge("first_state", "conversation_begin")
    graph.add_edge("conversation_begin", "assistant_step")
    graph.add_conditional_edges(
        "assistant_step",
        route_after_assistant_step,
        {
            "tool_call": "tool_call",
            "conversation_end": "conversation_end",
        },
    )
    graph.add_conditional_edges(
        "tool_call",
        route_after_tool_call,
        {
            "assistant_step": "assistant_step",
            "tool_error": "tool_error",
        },
    )
    graph.add_edge("tool_error", "assistant_step")
    graph.add_edge("conversation_end", END)
    return graph.compile()


def new_chat_state(
    *,
    model: str | None = None,
    system_prompt: str | None = None,
    web_search: bool = False,
    web_search_mode: str = "auto",
    web_search_provider: str = "duckduckgo",
    web_search_auto_switch: bool = False,
    web_search_base_url: str | None = None,
    rag_mode: str = "auto",
    rag_include_knowledge: bool = True,
    rag_include_memory: bool = True,
    rag_include_skills: bool = True,
    curl: bool = False,
    curl_mode: str = "auto",
    python: bool = False,
    python_mode: str = "auto",
    file_editor: bool = False,
    file_editor_mode: str = "auto",
    file_editor_approval: str = "auto",
    mcp: bool = False,
    mcp_mode: str = "auto",
    history: bool = False,
    history_mode: str = "off",
    automation: bool = False,
    automation_mode: str = "off",
    rag_context: str | None = None,
    web_search_results: list[str] | None = None,
    rag_results: list[str] | None = None,
    conversation_summary: str | None = None,
    include_tool_rules: bool = False,
    include_context_rules: bool = False,
    tool_error: str | None = None,
    max_tool_rounds: int = 20,
) -> ChatState:
    resolved_web_search_mode = "auto" if web_search and web_search_mode == "off" else web_search_mode
    resolved_curl_mode = "auto" if curl and curl_mode == "off" else curl_mode
    resolved_python_mode = "auto" if python and python_mode == "off" else python_mode
    resolved_file_editor_mode = "auto" if file_editor and file_editor_mode == "off" else file_editor_mode
    resolved_mcp_mode = "auto" if mcp and mcp_mode == "off" else mcp_mode
    resolved_history_mode = "auto" if history and history_mode == "off" else history_mode
    resolved_automation_mode = "auto" if automation and automation_mode == "off" else automation_mode
    state: ChatState = {
        "message": "",
        "web_search": web_search,
        "web_search_mode": resolved_web_search_mode,
        "web_search_provider": web_search_provider,
        "web_search_auto_switch": web_search_auto_switch,
        "rag_mode": rag_mode,
        "rag_include_knowledge": rag_include_knowledge,
        "rag_include_memory": rag_include_memory,
        "rag_include_skills": rag_include_skills,
        "curl": curl,
        "curl_mode": resolved_curl_mode,
        "python": python,
        "python_mode": resolved_python_mode,
        "file_editor": file_editor,
        "file_editor_mode": resolved_file_editor_mode,
        "file_editor_approval": file_editor_approval,
        "mcp": mcp,
        "mcp_mode": resolved_mcp_mode,
        "history": history,
        "history_mode": resolved_history_mode,
        "automation": automation,
        "automation_mode": resolved_automation_mode,
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


def stream_turn(state: ChatState, message: str) -> Iterator[AgentEvent]:
    """Run one turn and yield simple UI events from LangGraph streaming."""

    app = build_graph()
    current_state: ChatState = dict(state)
    current_state["message"] = message

    try:
        for chunk in app.stream(current_state, stream_mode="updates"):
            if not isinstance(chunk, dict):
                continue
            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                current_state.update(update)
                if node == "assistant_step":
                    yield from describe_assistant_progress_events(current_state)
                    yield from describe_tool_call_events(current_state)
                if node == "tool_call":
                    yield from update.get("tool_events") or []
                if update.get("tool_error"):
                    yield {"type": "error", "text": str(update["tool_error"])}
    except Exception as exc:
        log_exception("turn.stream_error", exc, message=message, state=current_state)
        raise

    log_event("turn.end", response=current_state.get("response", ""), state=current_state)
    yield {
        "type": "assistant",
        "text": current_state.get("response", ""),
        "state": current_state,
    }


def run_agent(
    message: str,
    *,
    model: str | None = None,
    system_prompt: str | None = None,
    web_search: bool = False,
    web_search_mode: str = "auto",
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
    state = new_chat_state(
        model=model,
        system_prompt=system_prompt,
        web_search=web_search,
        web_search_mode=web_search_mode,
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
    result = run_turn(state, message)
    return result.get("response", "")


def tool_call_key(request: ToolRequest) -> str:
    if request.name in {"webSearch", "rag"}:
        return f"{request.name}:{request.query}"
    if request.name == "curl":
        return f"{request.name}:{request.url}"
    if request.name == "python":
        return f"{request.name}:{request.code}"
    if request.name == "fileEditor" and request.file_edit:
        edit = request.file_edit
        return f"{request.name}:{edit.action}:{edit.path}:{edit.old_text}:{edit.new_text}:{edit.anchor}:{edit.content}"
    if request.name == "mcp" and request.mcp_request:
        mcp_request = request.mcp_request
        return f"{request.name}:{mcp_request.action}:{mcp_request.server}:{mcp_request.tool}:{mcp_request.arguments}"
    return request.name


def tool_budget_allows(state: ChatState) -> bool:
    max_rounds = int(state.get("max_tool_rounds", 20))
    return max_rounds < 0 or state.get("tool_rounds", 0) < max_rounds


def tool_budget_reached(state: ChatState) -> bool:
    return not tool_budget_allows(state)



def describe_file_edit_target(request: ToolRequest) -> str:
    if not request.file_edit:
        return "missing file_edit"
    edit = request.file_edit
    return f"{edit.action} {edit.path}".strip()


def describe_mcp_target(request: ToolRequest) -> str:
    if not request.mcp_request:
        return "missing mcp_request"
    item = request.mcp_request
    return f"{item.action} {item.server} {item.tool}".strip()


def format_repeated_tool_call_error(request: ToolRequest, previous_count: int) -> str:
    target = describe_tool_call_target(request)
    return (
        f'toolError: "repeated tool call blocked: {request.name} already used '
        f'{previous_count} times with this exact input this turn: {target}. '
        "Use a different source/input or answer from available results."
        '"'
    )

def describe_tool_call_target(request: ToolRequest) -> str:
    if request.name == "curl":
        return request.url
    if request.name == "python":
        return request.code
    if request.name == "fileEditor":
        return describe_file_edit_target(request)
    if request.name == "mcp":
        return describe_mcp_target(request)
    return request.query


def file_editor_approval_required(result: str) -> bool:
    return result.startswith("fileEditorResult:") and "approvalRequired: True" in result


def describe_assistant_progress_events(state: ChatState) -> Iterator[AgentEvent]:
    assistant_message = last_assistant_message(state)
    if not assistant_message.get("tool_calls"):
        return
    text = str(assistant_message.get("content") or "").strip()
    if text:
        yield {"type": "assistant_progress", "text": text}


def describe_tool_call_events(state: ChatState) -> Iterator[AgentEvent]:
    settings = state.get("settings")
    if not settings:
        return
    try:
        requests = parse_openai_tool_calls(last_assistant_message(state), settings)
    except ValueError:
        return
    for request in requests:
        yield describe_tool_request(request)


def describe_tool_request(request: ToolRequest) -> AgentEvent:
    if request.name == "webSearch":
        return {
            "type": "tool_call",
            "tool": request.name,
            "query": request.query,
            "text": f"webSearch: {request.query}",
        }
    if request.name == "rag":
        return {
            "type": "tool_call",
            "tool": request.name,
            "query": request.query,
            "text": f"rag: {request.query}",
        }
    if request.name == "curl":
        return {
            "type": "tool_call",
            "tool": request.name,
            "url": request.url,
            "text": f"curl: {request.url}",
        }
    if request.name == "python":
        first_line = request.code.splitlines()[0] if request.code.splitlines() else ""
        return {
            "type": "tool_call",
            "tool": request.name,
            "code": request.code,
            "text": f"python: {first_line[:120]}",
        }
    if request.name == "fileEditor" and request.file_edit:
        edit = request.file_edit
        return {
            "type": "tool_call",
            "tool": request.name,
            "action": edit.action,
            "path": edit.path,
            "text": f"fileEditor: {edit.action} {edit.path}".strip(),
        }
    if request.name == "mcp" and request.mcp_request:
        item = request.mcp_request
        target = item.tool or item.server
        return {
            "type": "tool_call",
            "tool": request.name,
            "action": item.action,
            "text": f"mcp: {item.action} {target}".strip(),
        }
    if request.name == "history" and request.history_request:
        item = request.history_request
        target = item.conversation_id or item.query or ""
        return {
            "type": "tool_call",
            "tool": request.name,
            "action": item.action,
            "text": f"history: {item.action} {target}".strip(),
        }
    if request.name == "automation" and request.automation_request:
        item = request.automation_request
        target = item.title or item.mcp_tool or item.mcp_server or ""
        return {
            "type": "tool_call",
            "tool": request.name,
            "action": item.action,
            "text": f"automation: {item.action} {target}".strip(),
        }
    return {
        "type": "tool_call",
        "tool": request.name,
        "text": f"tool: {request.name}",
    }


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
# conversation_begin: previous messages + user message + auto tool context + available tools -> assistant_step
# assistant_step: model answers or requests tools -> conversation_end or tool_call
# conversation_end: return the answer, ready for the next CLI input
# tool_call: execute one or more tool calls and return results to the model -> assistant_step or tool_error
# tool_error: tell the model why the call failed and remind it of the right format -> assistant_step
# compact: future node; summarize old messages when the context becomes too long -> conversation_begin
