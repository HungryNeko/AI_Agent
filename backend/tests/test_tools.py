import pytest

from tools.request import build_openai_tools, parse_openai_tool_calls
from tools.settings import make_tool_settings


def test_model_view_is_small():
    settings = make_tool_settings(web_search_mode="auto", rag_mode="auto")

    assert settings.model_view() == {"available": ["webSearch", "rag"]}


def test_build_openai_tools_from_enabled_settings():
    settings = make_tool_settings(web_search_mode="auto", rag_mode="on")

    tool_names = [tool["function"]["name"] for tool in build_openai_tools(settings)]

    assert tool_names == ["webSearch", "rag"]


def test_parse_web_search_tool_call_when_enabled():
    settings = make_tool_settings(web_search_mode="auto", rag_mode="off")

    requests = parse_openai_tool_calls(
        {
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "webSearch",
                        "arguments": '{"query":"LangGraph docs"}',
                    },
                }
            ]
        },
        settings,
    )

    request = requests[0]
    assert request is not None
    assert request.id == "call_1"
    assert request.name == "webSearch"
    assert request.query == "LangGraph docs"


def test_rag_request_requires_auto_mode():
    settings = make_tool_settings(web_search_mode="off", rag_mode="off")

    with pytest.raises(ValueError, match="rag can only be called"):
        parse_openai_tool_calls(
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "rag",
                            "arguments": '{"query":"project notes"}',
                        },
                    }
                ]
            },
            settings,
        )


def test_web_search_has_no_on_mode():
    with pytest.raises(ValueError, match="web_search_mode"):
        make_tool_settings(web_search_mode="on")


def test_rag_on_allows_model_call_and_auto_include():
    settings = make_tool_settings(rag_mode="on")

    assert settings.rag.can_model_call is True
    assert settings.rag.can_auto_include_context is True
