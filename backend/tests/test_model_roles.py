from dataclasses import replace

import pytest

from agent import app_settings, config, graph, llm, server, subagents
from tools.settings import make_tool_settings
from tools.subagent import SubagentTask


# ------------------------------------------------------------------ thinking effort per provider


@pytest.mark.parametrize(
    "kind, effort, expected",
    [
        ("deepseek", "off", {"thinking": {"type": "disabled"}}),
        ("deepseek", "low", {"thinking": {"type": "enabled"}, "reasoning_effort": "low"}),
        ("deepseek", "medium", {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}),
        ("deepseek", "max", {"thinking": {"type": "enabled"}, "reasoning_effort": "max"}),
        ("openai", "medium", {"reasoning_effort": "medium"}),
        ("openai", "max", {"reasoning_effort": "high"}),
        ("openai", "off", {}),  # always-reasoning models cannot be switched off
        ("openrouter", "high", {"reasoning": {"effort": "high"}}),
        ("openrouter", "off", {"reasoning": {"effort": "none"}}),
        ("dashscope", "off", {"enable_thinking": False}),
        ("dashscope", "high", {"enable_thinking": True, "thinking_budget": 16384}),
        ("none", "high", {}),
        ("", "high", {}),
        ("deepseek", "default", {}),
        ("deepseek", "bogus", {}),
    ],
)
def test_thinking_effort_is_sent_in_each_providers_own_fields(kind, effort, expected):
    payload = {"model": "m", "messages": []}
    added = llm.apply_reasoning(payload, kind, effort)
    assert {k: v for k, v in payload.items() if k not in {"model", "messages"}} == expected
    assert sorted(added) == sorted(expected)


def test_provider_kind_is_guessed_from_the_url_and_can_be_overridden():
    assert config.resolve_reasoning_kind({"base_url": "https://api.deepseek.com/v1"}) == "deepseek"
    assert config.resolve_reasoning_kind({"base_url": "https://openrouter.ai/api/v1"}) == "openrouter"
    assert config.resolve_reasoning_kind({"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"}) == "dashscope"
    assert config.resolve_reasoning_kind({"base_url": "http://localhost:11434/v1"}) == "none"
    assert config.resolve_reasoning_kind({"base_url": "https://api.deepseek.com/v1", "reasoning": "none"}) == "none"
    assert config.resolve_reasoning_kind({"base_url": "http://x", "reasoning": "OpenAI"}) == "openai"


def provider_setup(monkeypatch, kind, replies):
    cfg = config.ModelConfig(provider="p", model_id="m", base_url="http://x", api_key="k", reasoning=kind)
    monkeypatch.setattr(llm, "get_model_config", lambda model=None: cfg)
    monkeypatch.setattr(llm, "models_to_try", lambda model=None: [model])
    sent = []

    def fake_post(base_url, api_key, payload):
        sent.append(dict(payload))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(llm, "post_chat_completion", fake_post)
    return sent


OK = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}


def test_request_carries_the_thinking_effort_only_when_a_level_is_chosen(monkeypatch):
    sent = provider_setup(monkeypatch, "deepseek", [OK, OK])
    llm.complete_chat_once([{"role": "user", "content": "x"}], reasoning_effort="high")
    llm.complete_chat_once([{"role": "user", "content": "x"}])
    assert sent[0]["thinking"] == {"type": "enabled"} and sent[0]["reasoning_effort"] == "high"
    assert "thinking" not in sent[1] and "reasoning_effort" not in sent[1]


def test_deepseek_tool_calls_made_without_thinking_get_an_empty_reasoning_content(monkeypatch):
    # Switching model or effort mid-turn leaves tool-call messages without reasoning_content; DeepSeek's thinking
    # mode answers 400 for those. Only the outgoing copy is changed, never the stored history.
    history = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
        {"role": "assistant", "content": "kept", "reasoning_content": "thought", "tool_calls": [{"id": "c2"}]},
        {"role": "assistant", "content": "plain"},
    ]
    sent = provider_setup(monkeypatch, "deepseek", [OK])
    llm.complete_chat_once(history, reasoning_effort="high")
    out = sent[0]["messages"]
    assert out[1]["reasoning_content"] == ""
    assert out[3]["reasoning_content"] == "thought"
    assert "reasoning_content" not in out[4]
    assert "reasoning_content" not in history[1]
    sent = provider_setup(monkeypatch, "openai", [OK])
    llm.complete_chat_once(history)
    assert "reasoning_content" not in sent[0]["messages"][1]


def test_a_model_that_rejects_the_thinking_setting_is_retried_without_it(monkeypatch):
    sent = provider_setup(monkeypatch, "openai", [RuntimeError("LLM API error 400: unknown parameter reasoning_effort"), OK])
    reply = llm.complete_chat_once([{"role": "user", "content": "x"}], reasoning_effort="high")
    assert reply["content"] == "ok" and len(sent) == 2
    assert "reasoning_effort" in sent[0] or sent[0].get("reasoning_effort") == "high"
    assert "reasoning_effort" not in sent[1]
    # other failures are not swallowed
    provider_setup(monkeypatch, "openai", [RuntimeError("LLM API error 500: boom")])
    with pytest.raises(RuntimeError, match="500"):
        llm.complete_chat_once([{"role": "user", "content": "x"}], reasoning_effort="high")


def test_graph_passes_the_chosen_effort_to_the_model(monkeypatch):
    seen = []

    def fake_complete(messages, *, model=None, tools=None, **kwargs):
        seen.append(kwargs)
        return {"role": "assistant", "content": "ok"}

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete)
    list(graph.stream_turn(graph.new_chat_state(), "hi"))
    state = graph.new_chat_state()
    state["reasoning_effort"] = "high"
    list(graph.stream_turn(state, "hi"))
    assert seen == [{}, {"reasoning_effort": "high"}]


# ------------------------------------------------------------------ settings and options


def test_settings_keep_valid_values_and_fall_back_for_bad_ones():
    ok = app_settings.normalize_app_settings(
        {"chat": {"subagent_model": "p:small", "reasoning_effort": "max", "subagent_reasoning_effort": "low"}}
    )["chat"]
    assert (ok["subagent_model"], ok["reasoning_effort"], ok["subagent_reasoning_effort"]) == ("p:small", "max", "low")
    bad = app_settings.normalize_app_settings({"chat": {"reasoning_effort": "extreme", "subagent_reasoning_effort": 7}})["chat"]
    assert bad["reasoning_effort"] == "default" and bad["subagent_reasoning_effort"] == "default"
    assert app_settings.normalize_app_settings({})["chat"]["subagent_model"] == ""


def test_chat_options_reach_the_graph_state():
    options = server.ChatOptions(subagent_model="p:small", reasoning_effort="high", subagent_reasoning_effort="low")
    payload = server.ChatRequest(message="hi", options=options)
    state = server.build_chat_state(payload)
    assert (state["subagent_model"], state["reasoning_effort"], state["subagent_reasoning_effort"]) == ("p:small", "high", "low")
    with pytest.raises(ValueError):
        server.ChatOptions(reasoning_effort="extreme")


# ------------------------------------------------------------------ sub-agent model


def spawn_settings(**kw):
    return replace(make_tool_settings(), conversation_id="c1", model="p:big", **kw)


def test_sub_agents_use_the_sub_agent_model_unless_a_task_names_one(monkeypatch, tmp_path):
    monkeypatch.setattr(subagents, "ROOT", tmp_path)
    manager = subagents.SubagentManager()
    monkeypatch.setattr(manager, "_start", lambda agent, message: None)
    cfg = spawn_settings(subagent_model="p:small", subagent_reasoning="low")
    default, explicit = manager.spawn(
        "c1", (SubagentTask("a", "x"), SubagentTask("b", "y", model="p:strong")), cfg, watch=False
    )
    assert default.model == "p:small" and explicit.model == "p:strong"
    assert default.state["model"] == "p:small" and default.state["reasoning_effort"] == "low"
    # without a sub-agent model they follow the main model
    plain = manager.spawn("c2", (SubagentTask("c", "z"),), replace(spawn_settings(), conversation_id="c2"), watch=False)[0]
    assert plain.model == "p:big" and plain.state["reasoning_effort"] == "default"


def test_inline_agents_and_ask_llm_follow_the_sub_agent_model(monkeypatch):
    seen = {}

    def fake_stream(state, message):
        seen["state"] = state
        yield {"type": "assistant", "text": "answer", "state": state}

    monkeypatch.setattr(subagents, "stream_turn", fake_stream)
    cfg = spawn_settings(subagent_model="p:small", subagent_reasoning="medium")
    assert subagents.run_inline_agent(cfg, "task") == "answer"
    assert seen["state"]["model"] == "p:small" and seen["state"]["reasoning_effort"] == "medium"
    assert subagents.run_inline_agent(cfg, "task", model="p:other") == "answer" and seen["state"]["model"] == "p:other"

    from tools.bridge import ToolBridge

    calls = []
    monkeypatch.setattr(llm, "complete_chat_once", lambda messages, **kw: calls.append(kw) or {"content": "text"})
    ToolBridge(cfg)._ask({"prompt": "hello"})
    assert calls[0]["model"] == "p:small" and calls[0]["reasoning_effort"] == "medium"
