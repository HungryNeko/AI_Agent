import pytest
from fastapi.testclient import TestClient

from agent import app_settings, config, graph, llm, server
from tools.request import build_openai_tools
from tools.settings import WebSearchSettings, make_tool_settings


@pytest.mark.parametrize(
    "mode, tool_on, native_on, ui_mode",
    [
        ("off", False, False, "off"),
        ("tool", True, False, "tool"),
        ("auto", True, False, "tool"),  # the older name for "tool"
        ("native", False, True, "native"),
        ("both", True, True, "both"),
        (" Both ", True, True, "both"),
    ],
)
def test_four_step_web_search_setting_splits_into_tool_and_native(mode, tool_on, native_on, ui_mode):
    settings = make_tool_settings(web_search_mode=mode)
    assert settings.web_search.can_model_call is tool_on
    assert settings.web_search.native is native_on
    assert settings.web_search.ui_mode == ui_mode
    names = [tool["function"]["name"] for tool in build_openai_tools(settings)]
    assert ("webSearch" in names) is tool_on
    assert ("webSearch" in settings.model_view()["available"]) is tool_on


def test_invalid_web_search_mode_is_rejected():
    with pytest.raises(ValueError, match="off, tool, native, both"):
        make_tool_settings(web_search_mode="sometimes")
    assert WebSearchSettings().ui_mode == "tool"


def test_native_web_presets_and_custom_definitions():
    assert config.resolve_native_web({"native_web": "openrouter"}) == {"body": {"plugins": [{"id": "web"}]}}
    assert config.resolve_native_web({"native_web": "DashScope"})["body"] == {"enable_search": True}
    custom = config.resolve_native_web(
        {"native_web": {"body": {"search": True}, "tools": [{"type": "web_search_preview"}]}}
    )
    assert custom == {"body": {"search": True}, "tools": [{"type": "web_search_preview"}]}
    assert config.resolve_native_web({"native_web": "nope"}) is None
    assert config.resolve_native_web({"native_web": {}}) is None
    assert config.resolve_native_web({}) is None
    data = {"providers": {"a": {"native_web": "openrouter"}, "b": {}, "c": {"native_web": {"body": {"x": 1}}}}}
    assert config.native_web_providers(data) == ["a", "c"]


def test_apply_native_web_merges_body_and_tools_but_not_protected_keys():
    payload = {"model": "m", "messages": [], "tools": [{"type": "function", "function": {"name": "t"}}]}
    llm.apply_native_web(
        payload,
        {"body": {"enable_search": True, "model": "evil", "stream": True}, "tools": [{"type": "web_search_preview"}]},
    )
    assert payload["enable_search"] is True and payload["model"] == "m" and "stream" not in payload
    assert [t.get("type") for t in payload["tools"]] == ["function", "web_search_preview"]
    bare = llm.apply_native_web({"model": "m", "messages": []}, {"body": {}, "tools": [{"type": "web_search_preview"}]})
    assert bare["tools"] == [{"type": "web_search_preview"}] and bare["tool_choice"] == "auto"


def fake_provider(monkeypatch, native_web):
    sent = []
    cfg = config.ModelConfig(
        provider="p", model_id="m", base_url="http://x", api_key="k", native_web=native_web
    )
    monkeypatch.setattr(llm, "get_model_config", lambda model=None: cfg)
    monkeypatch.setattr(llm, "models_to_try", lambda model=None: [model])

    def fake_post(base_url, api_key, payload):
        sent.append(payload)
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(llm, "post_chat_completion", fake_post)
    return sent


def test_native_search_is_added_to_the_request_only_when_asked_for_and_configured(monkeypatch):
    sent = fake_provider(monkeypatch, {"body": {"plugins": [{"id": "web"}]}, "tools": []})
    llm.complete_chat_once([{"role": "user", "content": "x"}], native_web=True)
    llm.complete_chat_once([{"role": "user", "content": "x"}])
    assert sent[0]["plugins"] == [{"id": "web"}]
    assert "plugins" not in sent[1]

    sent = fake_provider(monkeypatch, None)  # provider without native search: request is left alone
    llm.complete_chat_once([{"role": "user", "content": "x"}], native_web=True)
    assert set(sent[0]) == {"model", "messages"}


@pytest.fixture
def capture_graph(monkeypatch):
    calls = []

    def fake_complete(messages, *, model=None, tools=None, **kwargs):
        calls.append({"kwargs": kwargs, "context": messages[-1]["content"]})
        return {"role": "assistant", "content": "done"}

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete)
    return calls


def test_graph_asks_for_native_search_in_native_and_both_modes(monkeypatch, capture_graph):
    monkeypatch.setattr(graph, "native_web_supported", lambda model=None: True)
    for mode in ("native", "both"):
        list(graph.stream_turn(graph.new_chat_state(web_search_mode=mode), "latest news?"))
    assert [call["kwargs"] for call in capture_graph] == [{"native_web": True}, {"native_web": True}]
    assert all('nativeWebSearch: "enabled"' in call["context"] for call in capture_graph)


def test_graph_leaves_the_request_alone_without_native_mode(capture_graph):
    for mode in ("off", "tool"):
        list(graph.stream_turn(graph.new_chat_state(web_search_mode=mode), "hi"))
    assert [call["kwargs"] for call in capture_graph] == [{}, {}]
    assert all("nativeWebSearch" not in call["context"] for call in capture_graph)


def test_graph_tells_the_model_when_the_provider_has_no_native_search(monkeypatch, capture_graph):
    monkeypatch.setattr(graph, "native_web_supported", lambda model=None: False)
    list(graph.stream_turn(graph.new_chat_state(web_search_mode="native"), "hi"))
    assert "unavailable" in capture_graph[0]["context"]


def test_chat_options_settings_and_models_endpoint_accept_the_new_values(monkeypatch):
    for mode in ("off", "tool", "native", "both", "auto"):
        assert server.ChatOptions(web_search_mode=mode).web_search_mode == mode
    cleaned = app_settings.normalize_settings({"chat": {"web_search_mode": "both"}}) if hasattr(app_settings, "normalize_settings") else None
    if cleaned is not None:
        assert cleaned["chat"]["web_search_mode"] == "both"
    fake = {
        "default_provider": "p",
        "default_model": "m",
        "providers": {"p": {"base_url": "http://x", "models": ["m"], "native_web": "openrouter"}, "q": {"models": ["z"]}},
    }
    monkeypatch.setattr(server, "load_config", lambda: fake)
    assert TestClient(server.app).get("/api/models").json()["nativeWebProviders"] == ["p"]
