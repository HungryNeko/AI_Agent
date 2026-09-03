"""Plain argparse CLI. No Typer, no Rich, no folded fancy output."""

from __future__ import annotations

import argparse
import ctypes
import os
import sys

from agent.graph import ChatState, new_chat_state, stream_turn


def fix_windows_encoding() -> None:
    if os.name == "nt":
        ctypes.windll.kernel32.SetConsoleCP(65001)
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the simple LangGraph agent demo.")
    parser.add_argument("message", nargs="*")
    parser.add_argument("-m", "--model", default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--loop", action="store_true", help="Keep chatting until exit/quit/q.")
    parser.add_argument("--web-search", action="store_true", help="Alias for --web-search-mode auto.")
    parser.add_argument("--web-search-mode", choices=["off", "auto"], default="auto")
    parser.add_argument(
        "--web-search-provider",
        choices=["duckduckgo", "searxng", "tavily"],
        default="duckduckgo",
    )
    parser.add_argument("--web-search-base-url", default=None)
    parser.add_argument("--rag-mode", choices=["off", "on", "auto"], default="auto")
    parser.add_argument("--rag-knowledge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rag-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rag-skills", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--curl", action="store_true", help="Alias for --curl-mode auto.")
    parser.add_argument("--curl-mode", choices=["off", "auto"], default="auto")
    parser.add_argument("--python", action="store_true", help="Alias for --python-mode auto.")
    parser.add_argument("--python-mode", choices=["off", "auto"], default="auto")
    parser.add_argument("--file-editor", action="store_true", help="Alias for --file-editor-mode auto.")
    parser.add_argument("--file-editor-mode", choices=["off", "auto"], default="auto")
    parser.add_argument("--file-editor-approval", choices=["readOnly", "manual", "auto"], default="auto")
    parser.add_argument("--mcp", action="store_true", help="Alias for --mcp-mode auto.")
    parser.add_argument("--mcp-mode", choices=["off", "auto"], default="auto")
    parser.add_argument("--history", action="store_true", help="Alias for --history-mode auto.")
    parser.add_argument("--history-mode", choices=["off", "auto"], default="off")
    parser.add_argument("--rag-context", default=None)
    parser.add_argument("--web-search-result", action="append", default=None)
    parser.add_argument("--rag-result", action="append", default=None)
    parser.add_argument("--conversation-summary", default=None)
    parser.add_argument("--tool-rules", action="store_true")
    parser.add_argument("--context-rules", action="store_true")
    parser.add_argument("--tool-error", default=None)
    parser.add_argument("--max-tool-rounds", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    fix_windows_encoding()
    args = parse_args()
    message = " ".join(args.message)

    try:
        state = make_state(args)
        if args.loop or not message:
            run_loop(args, state)
            return 0

        run_streamed_turn(state, message, assistant_prefix="")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def make_state(args: argparse.Namespace) -> ChatState:
    return new_chat_state(
        model=args.model,
        system_prompt=args.system_prompt,
        web_search=args.web_search,
        web_search_mode=args.web_search_mode,
        web_search_provider=args.web_search_provider,
        web_search_base_url=args.web_search_base_url,
        rag_mode=args.rag_mode,
        rag_include_knowledge=args.rag_knowledge,
        rag_include_memory=args.rag_memory,
        rag_include_skills=args.rag_skills,
        curl=args.curl,
        curl_mode=args.curl_mode,
        python=args.python,
        python_mode=args.python_mode,
        file_editor=args.file_editor,
        file_editor_mode=args.file_editor_mode,
        file_editor_approval=args.file_editor_approval,
        mcp=args.mcp,
        mcp_mode=args.mcp_mode,
        history=args.history,
        history_mode=args.history_mode,
        rag_context=args.rag_context,
        web_search_results=args.web_search_result,
        rag_results=args.rag_result,
        conversation_summary=args.conversation_summary,
        include_tool_rules=args.tool_rules,
        include_context_rules=args.context_rules,
        tool_error=args.tool_error,
        max_tool_rounds=args.max_tool_rounds,
    )


def run_loop(args: argparse.Namespace, state: ChatState) -> None:
    print("AI agent CLI. Type exit, quit, or q to stop.")
    while True:
        try:
            message = input("You> ").strip()
        except EOFError:
            print()
            return

        if message.lower() in {"exit", "quit", "q"}:
            return
        if not message:
            continue

        state = run_streamed_turn(state, message, assistant_prefix="AI> ")


def run_streamed_turn(state: ChatState, message: str, *, assistant_prefix: str) -> ChatState:
    next_state = state
    for event in stream_turn(state, message):
        event_type = event.get("type")
        if event_type == "assistant_progress":
            print(f"{assistant_prefix}{event.get('text', '')}")
        elif event_type == "tool_call":
            print(f"[tool] {event.get('text', '')}")
        elif event_type == "error":
            print(f"[tool error] {event.get('text', '')}")
        elif event_type == "approval_required":
            print(f"[approval] {event.get('text', '')}")
        elif event_type == "assistant":
            next_state = event.get("state", next_state)
            print(f"{assistant_prefix}{event.get('text', '')}")
    return next_state


if __name__ == "__main__":
    raise SystemExit(main())
