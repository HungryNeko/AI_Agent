from datetime import datetime, timezone

from prompts.context import build_context_prompt, format_current_time
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


def test_curl_auto_exposes_curl_tool():
    prompt = build_tools_prompt(curl_mode="auto")

    assert 'available: ["curl"]' in prompt



def test_tool_rules_handle_curl_failure_generically():
    prompt = build_tools_prompt(web_search_mode="auto", curl_mode="auto", include_rules=True)

    assert "official API documentation" in prompt
    assert "change the endpoint or parameters" in prompt
    assert "Open-Meteo" not in prompt

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
    tool_rules_index = system_prompt.index("Tool request format reminder:")

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

    assert 'available: ["python"]' in prompt
    assert "artifact directory" not in prompt


def test_tool_rules_include_python_artifact_guidance():
    prompt = build_tools_prompt(python_mode="auto", include_rules=True)

    assert "current working directory is the artifact directory" in prompt
    assert "Markdown image syntax" in prompt
    assert 'python: {"code"' in prompt


def test_tool_rules_include_web_and_api_image_guidance():
    prompt = build_tools_prompt(web_search_mode="auto", curl_mode="auto", include_rules=True)

    assert "webSearchResult includes image URLs" in prompt
    assert "curlResult/API data contains image URLs or image content" in prompt
    assert "![image](https://example.com/image.jpg)" in prompt

def test_file_editor_auto_exposes_tool_without_rules():
    prompt = build_tools_prompt(file_editor_mode="auto")

    assert 'available: ["fileEditor"]' in prompt
    assert "oldText" not in prompt


def test_tool_rules_include_file_editor_anchor_guidance():
    prompt = build_tools_prompt(file_editor_mode="auto", include_rules=True)

    assert "Use fileEditor for project file changes" in prompt
    assert 'fileEditor: {"action":"read"' in prompt
