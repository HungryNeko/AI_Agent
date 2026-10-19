import json

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


# ------------------------------------------------------------------ DeepSeek: hosted search/fetch endpoints


def deepseek_config(**extra):
    return config.ModelConfig(
        provider="deepseek",
        model_id="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key="sk-test",
        native_web=config.resolve_native_web({"base_url": "https://api.deepseek.com/v1", **extra}),
    )


def test_deepseek_works_out_of_the_box_and_other_providers_do_not():
    assert config.resolve_native_web({"base_url": "https://api.deepseek.com/v1"})["endpoints"] == {
        "search": "/v1/tools/search",
        "fetch": "/v1/tools/fetch",
    }
    assert config.resolve_native_web({"base_url": "https://api.openai.com/v1"}) is None
    assert config.resolve_native_web({"base_url": "https://api.deepseek.com/v1", "native_web": "dashscope"})["body"] == {"enable_search": True}
    custom = config.resolve_native_web({"native_web": {"endpoints": {"search": "/s", "fetch": "nope", "other": "/x"}}})
    assert custom["endpoints"] == {"search": "/s"}  # only known kinds with absolute paths
    assert config.native_web_providers({"providers": {"deepseek": {"base_url": "https://api.deepseek.com/v1"}, "x": {}}}) == ["deepseek"]


def test_endpoint_urls_do_not_double_the_version_segment():
    from tools import nativeWeb

    assert nativeWeb.endpoint_url("https://api.deepseek.com/v1", "/v1/tools/search") == "https://api.deepseek.com/v1/tools/search"
    assert nativeWeb.endpoint_url("https://api.deepseek.com", "/v1/tools/fetch") == "https://api.deepseek.com/v1/tools/fetch"
    assert nativeWeb.endpoint_url("https://host/api/", "/search") == "https://host/api/search"


def test_hosted_tools_appear_only_in_native_modes_for_a_provider_that_has_them():
    from dataclasses import replace

    names = lambda s: [t["function"]["name"] for t in build_openai_tools(s)]
    base = make_tool_settings(web_search_mode="native")
    assert "nativeWebSearch" not in names(base) and "webFetch" not in names(base)  # no endpoints known
    native = replace(base, native_tools=("search", "fetch"))
    assert names(native)[:2] == ["nativeWebSearch", "webFetch"]
    assert "webSearch" not in names(native)  # native only: the third-party search tool is gone
    both = replace(make_tool_settings(web_search_mode="both"), native_tools=("search",))
    assert "webSearch" in names(both) and "nativeWebSearch" in names(both) and "webFetch" not in names(both)
    assert {"nativeWebSearch", "webFetch"} <= set(native.model_view()["available"])
    for mode in ("ask", "plan"):  # read-only modes may browse too
        assert "webFetch" in names(replace(native, conversation_mode=mode))


def test_hosted_tool_calls_are_parsed_and_gated():
    from dataclasses import replace

    from tools.request import parse_openai_tool_calls

    call = lambda name, args: {"tool_calls": [{"id": "1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
    native = replace(make_tool_settings(web_search_mode="native"), native_tools=("search", "fetch"))
    search = parse_openai_tool_calls(call("nativeWebSearch", {"query": "q", "count": 99}), native)[0].native_web_request
    assert (search.kind, search.query, search.count) == ("search", "q", 20)
    fetch = parse_openai_tool_calls(call("webFetch", {"url": "https://example.com"}), native)[0].native_web_request
    assert (fetch.kind, fetch.url) == ("fetch", "https://example.com")
    with pytest.raises(ValueError, match="not available"):
        parse_openai_tool_calls(call("webFetch", {"url": "https://example.com"}), make_tool_settings())
    with pytest.raises(ValueError, match="query"):
        parse_openai_tool_calls(call("nativeWebSearch", {}), native)


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, json.dumps(body)

    def json(self):
        return self._body


def run_hosted(monkeypatch, request, response):
    from dataclasses import replace

    from tools import nativeWeb

    sent = []
    monkeypatch.setattr(config, "get_model_config", lambda model=None: deepseek_config())
    monkeypatch.setattr(nativeWeb.httpx, "post", lambda url, **kw: sent.append((url, kw)) or response)
    settings = replace(make_tool_settings(web_search_mode="native"), native_tools=("search", "fetch"), model="deepseek:deepseek-chat")
    return nativeWeb.execute(request, settings), sent


def test_search_calls_the_provider_endpoint_with_the_provider_key(monkeypatch):
    from tools.nativeWeb import NativeWebRequest

    body = {"results": [{"title": "DeepSeek V4", "url": "https://x.example/a", "summary": "S" * 2000, "site_name": "X", "date": "2026-10-01"}]}
    text, sent = run_hosted(monkeypatch, NativeWebRequest(kind="search", query="DeepSeek V4 release", count=3), FakeResponse(200, body))
    url, kw = sent[0]
    assert url == "https://api.deepseek.com/v1/tools/search"
    assert kw["headers"]["Authorization"] == "Bearer sk-test" and kw["json"] == {"query": "DeepSeek V4 release", "count": 3}
    assert text.startswith("nativeWebSearchResult:") and "1. DeepSeek V4 - https://x.example/a (X | 2026-10-01)" in text
    assert "S" * 900 in text and "S" * 901 not in text  # summaries are bounded
    empty, _ = run_hosted(monkeypatch, NativeWebRequest(kind="search", query="nothing"), FakeResponse(200, {"results": []}))
    assert "no results" in empty


def test_fetch_returns_bounded_markdown(monkeypatch):
    from tools import nativeWeb
    from tools.nativeWeb import NativeWebRequest

    page = {"title": "Article", "content": "# Heading\n" + "x" * 50_000}
    text, sent = run_hosted(monkeypatch, NativeWebRequest(kind="fetch", url="https://example.com/a"), FakeResponse(200, page))
    assert sent[0][0] == "https://api.deepseek.com/v1/tools/fetch" and sent[0][1]["json"] == {"url": "https://example.com/a"}
    assert text.startswith("webFetchResult:\nurl: https://example.com/a\ntitle: Article") and "truncated" in text
    assert len(text) < nativeWeb.FETCH_CHARS + 300


@pytest.mark.parametrize("status, hint", [(402, "insufficient balance"), (429, "rate limit"), (403, "blocked"), (401, "rejected")])
def test_provider_errors_become_readable_tool_errors(monkeypatch, status, hint):
    from tools.nativeWeb import NativeWebRequest

    error = {"error": {"message": "boom", "type": "x", "code": "c"}}
    with pytest.raises(ValueError, match=hint):
        run_hosted(monkeypatch, NativeWebRequest(kind="fetch", url="https://example.com"), FakeResponse(status, error))


def test_graph_offers_the_hosted_tools_when_the_current_provider_has_them(monkeypatch):
    captured = []

    def fake_complete(messages, *, model=None, tools=None, **kwargs):
        captured.append([t["function"]["name"] for t in tools or []])
        return {"role": "assistant", "content": "ok"}

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete)
    monkeypatch.setattr(config, "get_model_config", lambda model=None: deepseek_config())
    monkeypatch.setattr(graph, "native_web_supported", lambda model=None: True)
    list(graph.stream_turn(graph.new_chat_state(web_search_mode="native", model="deepseek:deepseek-chat"), "latest news?"))
    list(graph.stream_turn(graph.new_chat_state(web_search_mode="tool", model="deepseek:deepseek-chat"), "latest news?"))
    assert captured[0][:2] == ["nativeWebSearch", "webFetch"] and "webSearch" not in captured[0]
    assert "webSearch" in captured[1] and "nativeWebSearch" not in captured[1]
