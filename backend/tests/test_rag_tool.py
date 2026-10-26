import json

import pytest

from agent import app_settings, graph
from tools.request import build_openai_tools, parse_openai_tool_calls
from tools.settings import make_tool_settings, normalize_rag_method, normalize_rag_rerank


def call(arguments: dict, *, mode: str = "agent"):
    settings = make_tool_settings(rag_mode="auto", conversation_mode=mode)
    message = {"tool_calls": [{"id": "c1", "type": "function", "function": {"name": "rag", "arguments": json.dumps(arguments)}}]}
    return parse_openai_tool_calls(message, settings)[0].rag_request


def test_the_model_can_choose_method_rerank_limit_and_sources():
    request = call({"action": "search", "query": "x", "method": "bm25", "rerank": True, "limit": 3, "sources": ["memory", "skills"]})

    assert (request.method, request.rerank, request.limit, request.sources) == ("bm25", True, 3, ("memory", "skill"))


def test_search_defaults_leave_the_choices_to_the_settings():
    request = call({"query": "x"})

    assert (request.action, request.method, request.rerank, request.limit, request.sources) == ("search", "", None, None, ())


def test_limit_is_clamped_and_bad_values_are_rejected():
    assert call({"query": "x", "limit": 99}).limit == 10
    assert call({"query": "x", "limit": 0}).limit is None
    assert call({"query": "x", "rerank": "yes"}).rerank is None  # only a real boolean counts
    with pytest.raises(ValueError, match="method"):
        call({"query": "x", "method": "semantic"})
    with pytest.raises(ValueError, match="sources"):
        call({"query": "x", "sources": ["web"]})
    with pytest.raises(ValueError, match="sources must be an array"):
        call({"query": "x", "sources": "memory"})


def test_ingest_does_not_take_search_options():
    request = call({"action": "ingest", "path": "backend/runtime/uploads/x/a.pdf", "method": "bm25", "limit": 2})

    assert (request.method, request.limit, request.sources) == ("", None, ())


def test_the_tool_schema_offers_the_search_choices_and_read_only_modes_keep_them():
    for mode in ("agent", "ask"):
        tool = next(t for t in build_openai_tools(make_tool_settings(rag_mode="auto", conversation_mode=mode)) if t["function"]["name"] == "rag")
        properties = tool["function"]["parameters"]["properties"]
        assert properties["method"]["enum"] == ["hybrid", "vector", "bm25"]
        assert {"rerank", "limit", "sources"} <= set(properties)
        assert ("path" in properties) == (mode == "agent")
        assert "bm25" in tool["function"]["description"] and "hybrid" in tool["function"]["description"]


def test_settings_validate_the_reranker_and_the_default_method():
    assert normalize_rag_rerank("LLM") == "llm" and normalize_rag_method("") == "hybrid"
    with pytest.raises(ValueError):
        normalize_rag_rerank("cross")
    with pytest.raises(ValueError):
        normalize_rag_method("dense")
    rag = make_tool_settings(rag_rerank="api", rag_rerank_model=" jina:r ", rag_default_method="vector").rag
    assert (rag.rerank, rag.rerank_model, rag.default_method) == ("api", "jina:r", "vector")


def test_each_turn_reads_the_saved_rag_search_settings():
    app_settings.save_app_settings({"rag_search": {"rerank": "llm", "rerank_model": "p:m", "default_method": "bm25"}})

    options = graph.saved_rag_search_options()
    state = graph.first_state(graph.new_chat_state())
    rag = state["settings"].rag

    assert options == {"rag_rerank": "llm", "rag_rerank_model": "p:m", "rag_default_method": "bm25", "rag_require_citations": True}
    assert (rag.rerank, rag.rerank_model, rag.default_method) == ("llm", "p:m", "bm25")


def test_invalid_saved_search_settings_fall_back_to_the_defaults():
    normalized = app_settings.normalize_app_settings({"rag_search": {"rerank": "x", "default_method": "y"}, "rag_ingestion": {"contextual": "z"}})

    assert normalized["rag_search"] == {"rerank": "off", "rerank_model": "", "default_method": "hybrid", "require_citations": True}
    assert normalized["rag_ingestion"]["contextual"] == "off"
