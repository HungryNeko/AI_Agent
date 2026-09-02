import pytest

from tools.WebSearch import normalize_tavily_results, search
from tools.settings import WebSearchSettings, make_tool_settings


def test_default_web_search_provider_is_duckduckgo():
    settings = make_tool_settings(web_search_mode="auto")

    assert settings.web_search.provider == "duckduckgo"


def test_normalize_tavily_results():
    results = normalize_tavily_results(
        {
            "answer": "Short answer.",
            "results": [
                {
                    "title": "Title",
                    "url": "https://example.com",
                    "content": "Snippet",
                    "score": 0.9,
                }
            ],
        }
    )

    assert results == [
        {"title": "Tavily answer", "url": "", "snippet": "Short answer."},
        {
            "title": "Title",
            "url": "https://example.com",
            "snippet": "Snippet",
            "score": "0.9",
        },
    ]


def test_search_requires_tavily_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        search("LangGraph", WebSearchSettings(mode="auto", provider="tavily"))


def test_search_calls_duckduckgo(monkeypatch):
    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def text(self, query, *, safesearch, max_results, backend):
            return [
                {
                    "title": "LangGraph",
                    "href": "https://langchain-ai.github.io/langgraph/",
                    "body": "Build stateful agents.",
                }
            ]

    monkeypatch.setattr("tools.WebSearch.import_ddgs", lambda: FakeDDGS)

    results = search("LangGraph", WebSearchSettings(mode="auto", provider="duckduckgo"))

    assert results == [
        {
            "title": "LangGraph",
            "url": "https://langchain-ai.github.io/langgraph/",
            "snippet": "Build stateful agents.",
        }
    ]


def test_search_calls_tavily(monkeypatch):
    calls = {}

    class FakeResponse:
        status_code = 200
        text = "ok"

        def json(self):
            return {
                "results": [
                    {
                        "title": "LangGraph",
                        "url": "https://langchain-ai.github.io/langgraph/",
                        "content": "Build stateful agents.",
                    }
                ]
            }

    def fake_post(url, *, headers, json, timeout):
        calls["url"] = url
        calls["headers"] = headers
        calls["json"] = json
        calls["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setattr("tools.WebSearch.httpx.post", fake_post)

    settings = make_tool_settings(
        web_search_mode="auto",
        web_search_provider="tavily",
        web_search_max_results=3,
    ).web_search
    results = search("LangGraph", settings)

    assert calls["url"] == "https://api.tavily.com/search"
    assert calls["headers"]["Authorization"] == "Bearer tvly-test"
    assert calls["json"]["query"] == "LangGraph"
    assert calls["json"]["max_results"] == 3
    assert results[0]["title"] == "LangGraph"


def test_search_calls_searxng(monkeypatch):
    calls = {}

    class FakeResponse:
        status_code = 200
        text = "ok"

        def json(self):
            return {
                "results": [
                    {
                        "title": "LangGraph",
                        "url": "https://langchain-ai.github.io/langgraph/",
                        "content": "Build stateful agents.",
                    }
                ]
            }

    def fake_get(url, *, params, timeout):
        calls["url"] = url
        calls["params"] = params
        calls["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("tools.WebSearch.httpx.get", fake_get)

    results = search(
        "LangGraph",
        WebSearchSettings(
            mode="auto",
            provider="searxng",
            base_url="http://localhost:8080",
            max_results=3,
        ),
    )

    assert calls["url"] == "http://localhost:8080/search"
    assert calls["params"] == {"q": "LangGraph", "format": "json"}
    assert results[0]["title"] == "LangGraph"
