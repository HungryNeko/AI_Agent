"""The provider's prompt cache matches on the exact start of the request: tools, system prompt, then the
messages in order. These tests replay a multi-turn conversation the way the frontend does (state goes
through JSON between turns) and check that each request starts with the previous one, byte for byte."""

import json

import pytest

from agent import graph


def message_with_tool_call(call_id="call_1", code="print(1)"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": code})}}
        ],
    }


class RequestLog(list):
    def __init__(self):
        super().__init__()
        self.script = {"replies": []}


@pytest.fixture
def requests_log(monkeypatch):
    log = RequestLog()
    script = log.script

    def fake_complete(messages, *, model=None, tools=None, tool_choice=None, **kwargs):
        log.append(
            {
                "messages": json.loads(json.dumps(messages)),
                "tools": json.loads(json.dumps(tools)) if tools is not None else None,
                "tool_choice": tool_choice,
            }
        )
        reply = script["replies"].pop(0) if script["replies"] else {"role": "assistant", "content": "done"}
        return reply

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete)
    return log


def run_turn(state, message):
    events = list(graph.stream_turn(state, message))
    final = next(event for event in events if event["type"] == "assistant")
    public = {k: v for k, v in final["state"].items() if k not in {"settings", "message"}}
    return json.loads(json.dumps(public, default=str))  # what the frontend sends back next turn


def assert_cache_friendly(log):
    first = log[0]
    for earlier, later in zip(log, log[1:]):
        assert later["tools"] == first["tools"], "the tools array must not change between requests"
        assert later["messages"][: len(earlier["messages"])] == earlier["messages"], "history must stay a byte-identical prefix"


def test_conversation_requests_extend_each_other_across_turns_and_tool_rounds(requests_log):
    requests_log.script["replies"] = [
        message_with_tool_call(),
        {"role": "assistant", "content": "first answer"},
        {"role": "assistant", "content": "second answer"},
        message_with_tool_call("call_2", "print(2)"),
        {"role": "assistant", "content": "third answer"},
    ]
    state = graph.new_chat_state(python_mode="auto")
    for text in ("compute 1", "and then?", "compute 2"):
        state = run_turn(state, text)
    assert len(requests_log) == 5
    assert_cache_friendly(requests_log)


def test_exhausting_the_tool_budget_keeps_the_prompt_prefix(requests_log):
    """The last request of a turn that ran out of tool rounds used to drop the tools array, which
    changes the very start of the prompt and misses the cache for the whole conversation."""

    requests_log.script["replies"] = [
        message_with_tool_call("call_1"),  # uses the only tool round
        {"role": "assistant", "content": "best effort answer"},  # the budget is gone for this request
    ]
    state = graph.new_chat_state(python_mode="auto", max_tool_rounds=1)
    run_turn(state, "keep computing")
    assert len(requests_log) == 2
    assert_cache_friendly(requests_log)
    assert requests_log[-1]["tool_choice"] == "none" and requests_log[0]["tool_choice"] is None


def test_switching_conversation_mode_is_the_known_cache_break(requests_log):
    state = graph.new_chat_state(python_mode="auto")
    state = run_turn(state, "hello")
    state["conversation_mode"] = "ask"
    run_turn(state, "again")
    assert requests_log[0]["tools"] != requests_log[1]["tools"]  # documented trade-off, not a bug


# ------------------------------------------------------------------ measuring the hit rate


def test_usage_is_normalised_across_providers():
    from agent.llm import add_usage, normalize_usage

    deepseek = {"prompt_tokens": 1000, "completion_tokens": 50, "prompt_cache_hit_tokens": 896, "prompt_cache_miss_tokens": 104}
    openai = {"prompt_tokens": 2000, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 1536}}
    anthropic_style = {"prompt_tokens": 300, "completion_tokens": 5, "cache_read_input_tokens": 256}
    silent = {"prompt_tokens": 400, "completion_tokens": 20}
    assert normalize_usage(deepseek) == {"prompt": 1000, "completion": 50, "cached": 896}
    assert normalize_usage(openai)["cached"] == 1536
    assert normalize_usage(anthropic_style)["cached"] == 256
    assert "cached" not in normalize_usage(silent)  # not reported is different from 0% hit
    assert normalize_usage(None) is None
    total = add_usage(add_usage(None, normalize_usage(deepseek)), normalize_usage(openai))
    assert total == {"prompt": 3000, "completion": 60, "requests": 2, "cached": 2432}


def test_a_model_request_records_usage_for_the_caller(monkeypatch):
    from agent import config, llm

    cfg = config.ModelConfig(provider="p", model_id="m", base_url="http://x", api_key="k")
    monkeypatch.setattr(llm, "get_model_config", lambda model=None: cfg)
    monkeypatch.setattr(llm, "models_to_try", lambda model=None: [model])
    monkeypatch.setattr(
        llm,
        "post_chat_completion",
        lambda *args: {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 7, "prompt_cache_hit_tokens": 448},
        },
    )
    llm.complete_chat_once([{"role": "user", "content": "x"}])
    assert llm.consume_usage() == {"prompt": 500, "completion": 7, "cached": 448}
    assert llm.consume_usage() is None  # read once


def test_the_final_event_of_a_turn_carries_the_summed_usage(monkeypatch, requests_log):
    answers = iter([
        {"prompt": 1000, "completion": 20, "cached": 0},
        {"prompt": 1100, "completion": 30, "cached": 1000},
    ])
    monkeypatch.setattr(graph, "consume_usage", lambda: next(answers, None))
    requests_log.script["replies"] = [message_with_tool_call(), {"role": "assistant", "content": "done"}]
    events = list(graph.stream_turn(graph.new_chat_state(python_mode="auto"), "compute"))
    assert events[-1]["usage"] == {"prompt": 2100, "completion": 50, "cached": 1000, "requests": 2}
    # a turn without any reported usage simply has no usage key
    monkeypatch.setattr(graph, "consume_usage", lambda: None)
    assert "usage" not in list(graph.stream_turn(graph.new_chat_state(), "hi"))[-1]


# ------------------------------------------------------------------ nothing is injected twice


def user_texts(request):
    return [m["content"] for m in request["messages"] if m["role"] == "user"]


def test_tool_results_are_not_injected_again_in_later_turns(monkeypatch, requests_log):
    """A web search result is already in the history as a tool message. It used to be copied into the user
    message of every later turn as well, growing the prompt each turn with text that is all cache misses."""

    monkeypatch.setattr(graph, "execute_tool", lambda request, settings: "webSearchResult: SUNNY-MARKER")
    call = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "webSearch", "arguments": json.dumps({"query": "weather"})}}
    ]}
    requests_log.script["replies"] = [call, {"role": "assistant", "content": "sunny"}]
    state = graph.new_chat_state(web_search_mode="tool")
    for text in ("weather?", "thanks", "and tomorrow?"):
        state = run_turn(state, text)
    last = requests_log[-1]
    carriers = [t for t in user_texts(last) if "SUNNY-MARKER" in t]
    assert carriers == []  # only the tool message carries it
    assert sum(1 for m in last["messages"] if m["role"] == "tool" and "SUNNY-MARKER" in m["content"]) == 1
    assert_cache_friendly(requests_log)


def test_results_given_up_front_are_shown_once(requests_log):
    state = graph.new_chat_state(rag_results=["PRELOADED-NOTE"])
    for text in ("first", "second", "third"):
        state = run_turn(state, text)
    shown = [sum("PRELOADED-NOTE" in t for t in user_texts(request)) for request in requests_log]
    assert shown == [1, 1, 1]  # one copy in the history, however many turns follow


def test_the_plan_is_repeated_only_when_it_changes(requests_log):
    state = graph.new_chat_state()
    state["plan"] = {"name": "p", "content": "# Steps\n1. migrate", "status": "ready"}
    state = run_turn(state, "start")
    state = run_turn(state, "continue")
    assert sum("activePlan" in t for t in user_texts(requests_log[-1])) == 1
    state["plan"] = {"name": "p", "content": "# Steps\n1. migrate\n2. verify", "status": "approved"}
    state = run_turn(state, "after the change")
    assert sum("activePlan" in t for t in user_texts(requests_log[-1])) == 2


def test_compression_and_rewinding_show_results_and_plan_again(requests_log):
    from agent import session_store

    state = graph.new_chat_state()
    state["plan"] = {"name": "p", "content": "# Steps\n1. migrate", "status": "ready"}
    state = run_turn(state, "start")
    assert "plan_sent" in state
    compacted = session_store.compact_state(state, [{"type": "user", "text": "start"}])
    assert "plan_sent" not in compacted and "injected_results" not in compacted
    run_turn(compacted, "next")
    assert sum("activePlan" in t for t in user_texts(requests_log[-1])) == 1  # the old copy is gone, so it is sent again
