from prompts.system import build_system_prompt
from prompts.tools import build_tools_prompt


def test_rag_off_hides_rag_tool():
    prompt = build_tools_prompt(web_search_mode="auto", rag_mode="off")

    assert "webSearch" in prompt
    assert "rag" not in prompt
    assert "Web search mode" not in prompt
    assert "RAG mode" not in prompt


def test_rag_auto_exposes_rag_tool():
    prompt = build_tools_prompt(rag_mode="auto")

    assert "rag" in prompt
    assert "RAG mode" not in prompt


def test_rag_on_injects_context_and_exposes_tool():
    prompt = build_system_prompt(rag_mode="on", rag_context="Document says hello.")

    assert "Document says hello." in prompt
    assert "rag" in prompt
    assert 'ragResult: "Document says hello."' in prompt


def test_tool_rules_are_optional():
    short_prompt = build_system_prompt(web_search=True, rag_mode="auto")
    detailed_prompt = build_system_prompt(
        web_search=True,
        rag_mode="auto",
        include_tool_rules=True,
    )

    assert "Tool argument schema" not in short_prompt
    assert "Tool argument schema" in detailed_prompt


def test_tool_error_includes_rules():
    prompt = build_system_prompt(web_search_mode="auto", tool_error="bad json")

    assert "Previous tool request error: bad json" in prompt
    assert "Tool argument schema" in prompt


def test_injected_results_are_prefixed():
    prompt = build_tools_prompt(
        web_search_mode="auto",
        rag_mode="auto",
        web_search_results=["Search says A."],
        rag_results=["RAG says B."],
    )

    assert 'webSearchResult: "Search says A."' in prompt
    assert 'ragResult: "RAG says B."' in prompt


def test_context_summary_without_rules_by_default():
    prompt = build_system_prompt(conversation_summary="Previous work was summarized.")

    assert 'conversationSummary: "Previous work was summarized."' in prompt
    assert "Context rules:" not in prompt


def test_context_rules_are_optional():
    prompt = build_system_prompt(
        conversation_summary="Previous work was summarized.",
        include_context_rules=True,
    )

    assert "Context rules:" in prompt
    assert "not a complete log" in prompt


def test_static_rules_come_before_dynamic_context():
    prompt = build_system_prompt(
        conversation_summary="Previous work was summarized.",
        include_context_rules=True,
        include_tool_rules=True,
        web_search_mode="auto",
        rag_mode="auto",
        web_search_results=["Search says A."],
    )

    base_index = prompt.index("You are an AI agent.")
    context_rules_index = prompt.index("Context rules:")
    tool_rules_index = prompt.index("Tool request format reminder:")
    summary_index = prompt.index("conversationSummary:")
    available_index = prompt.index("available:")
    result_index = prompt.index("webSearchResult:")

    assert base_index < context_rules_index < tool_rules_index
    assert tool_rules_index < summary_index < available_index < result_index
