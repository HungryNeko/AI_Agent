"""Read-only list of every prompt the application sends to a model, for debugging.

Nothing here is editable: the texts are the constants the code really uses, the assembled system prompt is
built the same way the graph builds it, and the tool schemas are the ones sent in the `tools` field.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from agent.automation_runner import AUTOMATION_RULES
from agent.config import load_system_prompt
from agent.instructions import load_instruction
from agent.reviewer import REVIEWER_PROMPT
from agent.subagents import TASK_WRAPPER
from agent.titles import _PROMPT as TITLE_PROMPT
from prompts.context import CONTEXT_RULES, TOOL_BUDGET_NOTICE, format_current_time
from prompts.system import BASE_SYSTEM_PROMPT
from prompts.tools import TOOL_REQUEST_FORMAT, format_available
from tools.rag.chunking import CHUNK_PLAN_PROMPT
from tools.rag.citations import correction_message
from tools.rag.contextualize import CONTEXT_PROMPT
from tools.rag.rerank import RERANK_PROMPT
from tools.request import build_openai_tools
from tools.settings import make_tool_settings

GROUPS = ("system", "turn", "other", "tools")


def item(group: str, item_id: str, title: str, title_zh: str, note: str, note_zh: str, content: str) -> dict[str, str]:
    return {
        "group": group,
        "id": item_id,
        "title": title,
        "title_zh": title_zh,
        "note": note,
        "note_zh": note_zh,
        "content": content,
    }


def full_tool_settings(mode: str):
    """Tool settings with everything switched on, as a chat with every tool enabled would have them."""

    settings = make_tool_settings(
        web_search_mode="both",
        rag_mode="auto",
        curl_mode="auto",
        python_mode="auto",
        file_reader_mode="auto",
        file_editor_mode="auto",
        mcp_mode="auto",
        history_mode="auto",
        automation_mode="auto",
        conversation_mode=mode,
    )
    return replace(settings, native_tools=("search", "fetch"), security_level="L3")


def tool_items() -> list[dict[str, str]]:
    seen: dict[str, dict[str, Any]] = {}
    for mode in ("agent", "plan"):
        for tool in build_openai_tools(full_tool_settings(mode)):
            function = tool.get("function") or {}
            seen.setdefault(str(function.get("name") or tool.get("type")), tool)
    items = []
    for name, tool in seen.items():
        function = tool.get("function") or {}
        description = str(function.get("description") or "").strip()
        schema = json.dumps(function.get("parameters") or {}, indent=2, ensure_ascii=False)
        items.append(
            item(
                "tools",
                f"tool.{name}",
                name,
                name,
                "Native tool schema sent in the `tools` field of every request.",
                "原生工具定义，每次请求都会放进 `tools` 字段。",
                f"{description}\n\nparameters:\n{schema}",
            )
        )
    return items


def build_prompt_catalog() -> dict[str, Any]:
    first_turn = full_tool_settings("agent")
    instruction = load_instruction()
    assembled = load_system_prompt(
        web_search_mode="both",
        rag_mode="auto",
        curl_mode="auto",
        python_mode="auto",
        file_reader_mode="auto",
        file_editor_mode="auto",
        mcp_mode="auto",
        history_mode="auto",
        instruction_text=instruction,
        include_tool_rules=True,
        include_context_rules=True,
    )
    turn_example = "\n\n".join(
        [
            format_current_time(),
            'conversationMode: "agent"',
            'questionMode: "light"',
            'developerMode: "enabled"',
            'nativeWebSearch: "enabled"',
            'conversationSummary: "..."',
            format_available(first_turn),
            "<reference, attachment and plan blocks, only when present>",
            'userMessage: "..."',
        ]
    )
    items = [
        item(
            "system",
            "system.assembled",
            "Assembled system prompt",
            "完整系统提示词",
            "The first message of a new conversation, exactly as it is built: base prompt, instruction, context rules and tool rules. A per-chat user system prompt is appended as userSystemPrompt.",
            "新对话的第一条消息，和实际构造的完全一致：基础提示词、指令、上下文规则、工具规则。聊天里单独设置的系统提示词会作为 userSystemPrompt 追加在后面。",
            assembled,
        ),
        item("system", "system.base", "Base system prompt", "基础系统提示词", "backend/prompts/system.py", "backend/prompts/system.py", BASE_SYSTEM_PROMPT),
        item(
            "system",
            "system.instruction",
            "Instruction",
            "指令",
            "data/instruction.md, edited on this page under Instruction.",
            "data/instruction.md，在本页“指令”分类里编辑。",
            instruction or "(empty)",
        ),
        item("system", "system.context", "Context rules", "上下文规则", "backend/prompts/context.py", "backend/prompts/context.py", CONTEXT_RULES),
        item(
            "system",
            "system.tools",
            "Tool request rules",
            "工具使用规则",
            "backend/prompts/tools.py; part of the first system message of every conversation (a tool error only returns the error and the available list).",
            "backend/prompts/tools.py；每个对话第一条系统消息的一部分（工具出错时只返回错误和可用工具列表，不会重发）。",
            TOOL_REQUEST_FORMAT,
        ),
        item(
            "turn",
            "turn.layout",
            "Per-turn user message",
            "每轮用户消息",
            "What each user turn looks like. Only small dynamic context goes here, so the system prompt stays cacheable.",
            "每轮用户消息的格式。只放少量动态内容，这样系统提示词才能命中缓存。",
            turn_example,
        ),
        item(
            "turn",
            "turn.budget",
            "Tool budget notice",
            "工具次数用尽提示",
            "Sent when max_tool_rounds is used up. The tools stay in the request, with tool_choice set to none.",
            "max_tool_rounds 用完时发送。工具定义仍保留在请求里，只是 tool_choice 设为 none。",
            TOOL_BUDGET_NOTICE,
        ),
        item(
            "turn",
            "turn.citation",
            "Citation check",
            "引文检查",
            "agent/citation_check.py; when an answer after a rag search cites a source that was never returned, quotes text that is not in the chunk, or cites nothing, the answer is withheld and this goes back to the model (at most twice per turn).",
            "agent/citation_check.py；检索后的答案引用了从未返回过的来源、引文不在原文里、或完全没有引用时，答案不会发出，而是把这条反馈给模型（每轮最多两次）。",
            correction_message(["[knowledge/a.md#9] does not exist. Cite only the `cite:` tags that rag results show, copied exactly."], ["[knowledge/a.md#0]"]),
        ),
        item(
            "turn",
            "turn.error",
            "Tool error reminder",
            "工具错误提示",
            "The result of a bad tool call; the raw error follows toolError.",
            "工具调用出错时返回给模型的内容，toolError 后面是原始错误。",
            f"toolError: <error>\n{format_available(first_turn)}",
        ),
        item(
            "other",
            "other.subagent",
            "Sub-agent task wrapper",
            "子代理任务包装",
            "backend/agent/subagents.py; wraps the prompt of every spawned task.",
            "backend/agent/subagents.py；每个子代理任务的提示词都会被它包起来。",
            TASK_WRAPPER,
        ),
        item(
            "other",
            "other.reviewer",
            "AI reviewer",
            "AI 审查员",
            "backend/agent/reviewer.py; used when file editor approval is AI review.",
            "backend/agent/reviewer.py；文件编辑批准设为“AI 审查”时使用。",
            REVIEWER_PROMPT,
        ),
        item("other", "other.title", "Conversation title", "对话标题", "backend/agent/titles.py", "backend/agent/titles.py", TITLE_PROMPT),
        item(
            "other",
            "other.automation",
            "Automation run",
            "自动化运行",
            "backend/agent/automation_runner.py; the user message of an LLM automation run, followed by the task.",
            "backend/agent/automation_runner.py；LLM 类自动化运行时的用户消息，后面接任务内容。",
            AUTOMATION_RULES,
        ),
        item(
            "other",
            "other.chunking",
            "RAG chunk planner",
            "RAG 切分规划",
            "backend/tools/rag/chunking.py; used when ingestion splits with an LLM.",
            "backend/tools/rag/chunking.py；用 LLM 切分文档时使用。",
            CHUNK_PLAN_PROMPT,
        ),
        item(
            "other",
            "other.rag_context",
            "RAG chunk context",
            "RAG 分块上下文",
            "backend/tools/rag/contextualize.py; with contextual retrieval on, each chunk is sent after its document and the answer is indexed with the chunk.",
            "backend/tools/rag/contextualize.py；开启上下文检索后，每个分块连同整篇文档发给模型，生成的上下文会和分块一起被索引。",
            CONTEXT_PROMPT,
        ),
        item(
            "other",
            "other.rag_rerank",
            "RAG reranker",
            "RAG 重排",
            "backend/tools/rag/rerank.py; used when the LLM reranker is on (or a search asks for rerank).",
            "backend/tools/rag/rerank.py；开启 LLM 重排（或某次检索要求重排）时使用。",
            RERANK_PROMPT,
        ),
        *tool_items(),
    ]
    return {"groups": list(GROUPS), "items": items}
