import pytest

from agent import config, llm


def make_config(order, cutoff):
    return {
        "default_provider": "p",
        "providers": {"p": {"base_url": "http://x", "api_key": "k", "models": ["m1", "m2", "m3", "m4"]}},
        "model_priority": order,
        "model_priority_cutoff": cutoff,
    }


def test_chain_is_the_first_cutoff_models_in_the_user_order():
    # models 1,2,3,4 sorted as 3,2,4,1 and cut after three -> 3,2,4; model 1 is never tried
    data = make_config(["p:m3", "p:m2", "p:m4", "p:m1"], 3)
    assert config.default_model_chain(data) == ["p:m3", "p:m2", "p:m4"]


def test_chain_skips_removed_models_without_using_up_slots_and_needs_a_cutoff():
    data = make_config(["p:gone", "p:m2", "p:m1"], 2)
    assert config.default_model_chain(data) == ["p:m2", "p:m1"]
    assert config.default_model_chain(make_config(["p:m1"], None)) == []
    assert config.default_model_chain(make_config(["p:m1"], 0)) == []


def test_sync_default_points_at_chain_head():
    data = config.sync_default_from_priority(make_config(["p:m3", "p:m2"], 2))
    assert (data["default_provider"], data["default_model"]) == ("p", "m3")


def install_chain(monkeypatch, chain, failing):
    calls = []
    monkeypatch.setattr(config, "default_model_chain", lambda data=None: list(chain))
    monkeypatch.setattr(config, "same_model", lambda a, b: a == b)
    monkeypatch.delenv("AI_AGENT_DEFAULT_MODEL", raising=False)

    def fake_single(messages, *, model=None, tools=None, tool_choice=None, max_tokens=None):
        calls.append(model)
        if model in failing:
            raise RuntimeError(f"{model} down")
        return {"role": "assistant", "content": f"from {model}"}

    monkeypatch.setattr(llm, "complete_chat_with_model", fake_single)
    return calls


def test_default_model_falls_back_down_the_chain(monkeypatch):
    calls = install_chain(monkeypatch, ["p:m3", "p:m2", "p:m4"], failing={"p:m3"})
    assert llm.complete_chat_once([], model=None)["content"] == "from p:m2"
    assert calls == ["p:m3", "p:m2"]


def test_chain_stops_at_the_cutoff_and_raises_the_last_error(monkeypatch):
    calls = install_chain(monkeypatch, ["p:m3", "p:m2", "p:m4"], failing={"p:m3", "p:m2", "p:m4"})
    with pytest.raises(RuntimeError, match="m4 down"):
        llm.complete_chat_once([], model="p:m3")
    assert calls == ["p:m3", "p:m2", "p:m4"]  # m1 sits below the cutoff and is never tried


def test_explicit_non_default_model_never_falls_back(monkeypatch):
    calls = install_chain(monkeypatch, ["p:m3", "p:m2"], failing={"p:m1"})
    with pytest.raises(RuntimeError):
        llm.complete_chat_once([], model="p:m1")
    assert calls == ["p:m1"]


def test_env_override_bypasses_chain(monkeypatch):
    calls = install_chain(monkeypatch, ["p:m3", "p:m2"], failing={None})
    monkeypatch.setenv("AI_AGENT_DEFAULT_MODEL", "p:m3")
    with pytest.raises(RuntimeError):
        llm.complete_chat_once([], model=None)
    assert calls == [None]
