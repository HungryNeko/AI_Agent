from datetime import datetime, timezone

from prompts.context import build_context_prompt, format_current_time
from prompts.system import build_system_prompt
from prompts.tools import build_tools_prompt
from tools.request import curl_tool, file_editor_tool, mcp_tool, python_tool


def test_system_prompt_prefers_latex_for_formulas():
    prompt = build_system_prompt()

    assert "prefer standard LaTeX" in prompt
    assert "$...$ for inline math" in prompt
    assert "$$...$$ for display math" in prompt
    assert "\\frac" in prompt
    assert "Do not wrap LaTeX formulas in code fences" in prompt


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


def test_curl_auto_exposes_curl_tool():
    prompt = build_tools_prompt(curl_mode="auto")

    assert 'available: ["curl", "createTool", "custom__*", "model", "parameterSave", "question", "approval", "agent"]' in prompt



def test_curl_failure_guidance_lives_in_the_curl_tool():
    description = curl_tool()["function"]["description"]

    assert "official API documentation" in description
    assert "change the endpoint or parameters" in description
    assert "Open-Meteo" not in description

def test_rag_on_injects_context_and_exposes_tool():
    prompt = build_tools_prompt(rag_mode="on", rag_context="Document says hello.")

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

    assert "Tool rules:" not in short_prompt
    assert "Tool rules:" in detailed_prompt


def test_tool_error_includes_rules():
    prompt = build_system_prompt(web_search_mode="auto", tool_error="bad json")

    assert "Previous tool request error: bad json" in prompt
    assert "Tool rules:" in prompt


def test_injected_results_are_prefixed():
    prompt = build_tools_prompt(
        web_search_mode="auto",
        rag_mode="auto",
        web_search_results=["Search says A."],
        rag_results=["RAG says B."],
    )

    assert 'webSearchResult: "Search says A."' in prompt
    assert 'ragResult: "RAG says B."' in prompt


def test_available_tools_can_be_omitted_without_hiding_injected_results():
    from prompts.tools import build_tools_prompt_from_settings
    from tools.settings import make_tool_settings

    prompt = build_tools_prompt_from_settings(
        make_tool_settings(web_search_mode="auto"),
        web_search_results=["Search says A."],
        include_available=False,
    )

    assert "available:" not in prompt
    assert 'webSearchResult: "Search says A."' in prompt


def test_context_summary_without_rules_by_default():
    prompt = build_system_prompt(conversation_summary="Previous work was summarized.")

    assert 'conversationSummary: "Previous work was summarized."' not in prompt
    assert "Context rules:" not in prompt


def test_context_rules_are_optional():
    prompt = build_system_prompt(
        conversation_summary="Previous work was summarized.",
        include_context_rules=True,
    )

    assert "Context rules:" in prompt
    assert "not a complete log" in prompt
    assert 'conversationSummary: "Previous work was summarized."' not in prompt


def test_system_prompt_keeps_dynamic_context_out_for_prompt_cache():
    system_prompt = build_system_prompt(
        conversation_summary="Previous work was summarized.",
        include_context_rules=True,
        include_tool_rules=True,
        web_search_mode="auto",
        rag_mode="auto",
        web_search_results=["Search says A."],
    )
    turn_prompt = "\n\n".join(
        [
            build_context_prompt(conversation_summary="Previous work was summarized."),
            build_tools_prompt(
                web_search_mode="auto",
                rag_mode="auto",
                web_search_results=["Search says A."],
            ),
        ]
    )

    base_index = system_prompt.index("You are an AI agent.")
    context_rules_index = system_prompt.index("Context rules:")
    tool_rules_index = system_prompt.index("Tool rules:")

    assert base_index < context_rules_index < tool_rules_index
    assert "conversationSummary:" not in system_prompt
    assert "available:" not in system_prompt
    assert "webSearchResult:" not in system_prompt
    assert 'conversationSummary: "Previous work was summarized."' in turn_prompt
    assert "available:" in turn_prompt
    assert "webSearchResult:" in turn_prompt

def test_current_time_prompt_has_local_and_utc_reference():
    prompt = format_current_time(datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc))

    assert 'currentTime: "2026-09-02T12:00:00+00:00"' in prompt
    assert 'referenceUTC: "2026-09-02T12:00:00Z"' in prompt


def test_python_auto_exposes_python_tool_without_rules():
    prompt = build_tools_prompt(python_mode="auto")

    assert 'available: ["python", "createTool", "custom__*", "model", "parameterSave", "question", "approval", "agent"]' in prompt
    assert "artifact directory" not in prompt


def test_per_tool_guidance_lives_in_the_tool_descriptions_not_in_the_rules():
    rules = build_tools_prompt(include_rules=True, include_available=False)
    python_text = python_tool()["function"]["description"]
    mcp_text = mcp_tool()["function"]["description"]
    editor_text = file_editor_tool()["function"]["description"]

    assert "current working directory is the artifact directory" in python_text
    assert "write_osm_scatter" in python_text and "Markdown images" in python_text
    assert "content_base64_from_file" in mcp_text and "absoluteUrl" in mcp_text
    assert "approvalRequired" in editor_text and "List and read before editing" in editor_text
    # the rules choose and combine tools; they do not repeat each tool's mechanics or old JSON examples
    for detail in ("write_osm_scatter", "content_base64_from_file", "Tool argument schemas", '{"action"'):
        assert detail not in rules


def test_file_editor_auto_exposes_tool_without_rules():
    prompt = build_tools_prompt(file_editor_mode="auto")

    assert 'available: ["fileEditor", "createTool", "custom__*", "model", "parameterSave", "question", "approval", "agent"]' in prompt
    assert "oldText" not in prompt


def test_file_reader_auto_exposes_read_only_document_tool():
    prompt = build_tools_prompt(file_reader_mode="auto")

    assert 'available: ["fileReader", "createTool", "custom__*", "model", "parameterSave", "question", "approval", "agent"]' in prompt

    detailed = build_tools_prompt(file_reader_mode="auto", include_rules=True)
    assert "fileReader extracts text" in detailed


def test_user_editable_text_comes_after_the_fixed_rules():
    prompt = build_system_prompt(
        instruction_text="Use short answers.",
        include_tool_rules=True,
        include_context_rules=True,
    )

    assert prompt.index("Context rules:") < prompt.index("Tool rules:") < prompt.index("instruction:\nUse short answers.")
    assert prompt.endswith("Use short answers.")
