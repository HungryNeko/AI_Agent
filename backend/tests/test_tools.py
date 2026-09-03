import pytest

from tools.executor import (
    format_curl_result,
    format_file_editor_result,
    format_mcp_result,
    format_python_result,
    format_web_search_results,
    execute_tool,
)
from tools.request import build_openai_tools, parse_openai_tool_calls
from tools.settings import make_tool_settings


def test_model_view_is_small():
    settings = make_tool_settings(
        web_search_mode="auto",
        rag_mode="auto",
        curl_mode="off",
        python_mode="off",
        file_editor_mode="off",
        mcp_mode="off",
    )

    assert settings.model_view() == {"available": ["webSearch", "rag"]}


def test_model_view_includes_curl_only_when_enabled():
    settings = make_tool_settings(
        web_search_mode="auto",
        rag_mode="auto",
        curl_mode="auto",
        python_mode="off",
        file_editor_mode="off",
        mcp_mode="off",
    )

    assert settings.model_view() == {"available": ["webSearch", "rag", "curl"]}


def test_build_openai_tools_from_enabled_settings():
    settings = make_tool_settings(
        web_search_mode="auto",
        rag_mode="on",
        curl_mode="off",
        python_mode="off",
        file_editor_mode="off",
        mcp_mode="off",
    )

    tool_names = [tool["function"]["name"] for tool in build_openai_tools(settings)]

    assert tool_names == ["webSearch", "rag"]


def test_build_openai_tools_includes_curl_when_enabled():
    settings = make_tool_settings(
        web_search_mode="off",
        rag_mode="off",
        curl_mode="auto",
        python_mode="off",
        file_editor_mode="off",
        mcp_mode="off",
    )

    tool_names = [tool["function"]["name"] for tool in build_openai_tools(settings)]

    assert tool_names == ["curl"]


def test_build_openai_tools_includes_history_when_enabled():
    settings = make_tool_settings(
        web_search_mode="off",
        rag_mode="off",
        curl_mode="off",
        python_mode="off",
        file_editor_mode="off",
        mcp_mode="off",
        history_mode="auto",
    )

    tool_names = [tool["function"]["name"] for tool in build_openai_tools(settings)]

    assert tool_names == ["history"]


def test_settings_tool_is_available_with_automation_mode():
    settings = make_tool_settings(
        web_search_mode="off",
        rag_mode="off",
        curl_mode="off",
        python_mode="off",
        file_editor_mode="off",
        mcp_mode="off",
        history_mode="off",
        automation_mode="auto",
    )

    tool_names = [tool["function"]["name"] for tool in build_openai_tools(settings)]

    assert tool_names == ["automation", "settings"]


def test_settings_tool_updates_persistent_json(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr("agent.app_settings.SETTINGS_PATH", settings_path)
    tool_settings = make_tool_settings(
        web_search_mode="off",
        rag_mode="off",
        curl_mode="off",
        python_mode="off",
        file_editor_mode="off",
        mcp_mode="off",
        history_mode="off",
        automation_mode="auto",
    )

    [request] = parse_openai_tool_calls(
        {
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "settings",
                        "arguments": '{"action":"update","patch":{"ui":{"theme":"dark"},"chat":{"max_tool_rounds":-1}}}',
                    },
                }
            ]
        },
        tool_settings,
    )
    result = execute_tool(request, tool_settings)

    assert result.startswith("settingsResult:")
    assert '"theme": "dark"' in result
    assert '"max_tool_rounds": -1' in result



def test_curl_tool_description_points_to_official_docs_when_uncertain():
    settings = make_tool_settings(
        web_search_mode="off",
        rag_mode="off",
        curl_mode="auto",
        python_mode="off",
        file_editor_mode="off",
        mcp_mode="off",
    )

    [tool] = build_openai_tools(settings)
    description = tool["function"]["description"]

    assert "official API documentation" in description
    assert "Open-Meteo" not in description


def test_web_search_tool_description_mentions_image_markdown():
    settings = make_tool_settings(
        web_search_mode="auto",
        rag_mode="off",
        curl_mode="off",
        python_mode="off",
        file_editor_mode="off",
        mcp_mode="off",
    )

    [tool] = build_openai_tools(settings)
    description = tool["function"]["description"]

    assert "image URLs" in description
    assert "Markdown images" in description

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


def test_parse_curl_tool_call_when_enabled():
    settings = make_tool_settings(curl_mode="auto")

    requests = parse_openai_tool_calls(
        {
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "curl",
                        "arguments": (
                            '{"url":"https://api.example.com/v1/weather?'
                            'latitude=34.05&longitude=-118.24&current=true"}'
                        ),
                    },
                }
            ]
        },
        settings,
    )

    request = requests[0]
    assert request.name == "curl"
    assert request.url.startswith("https://api.example.com/v1/weather")


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


def test_curl_request_requires_auto_mode():
    settings = make_tool_settings(curl_mode="off")

    with pytest.raises(ValueError, match="curl can only be called"):
        parse_openai_tool_calls(
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "curl",
                            "arguments": '{"url":"https://api.example.com/v1/weather"}',
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

def test_curl_http_error_formats_as_tool_error():
    result = format_curl_result(
        {
            "url": "https://api.example.com/bad",
            "status_code": 400,
            "content_type": "application/json",
            "body": '{"reason":"bad request"}',
            "truncated": False,
        }
    )

    assert result.startswith("toolError:")
    assert "HTTP 400" in result
    assert "bad request" in result

def test_curl_api_error_body_formats_as_tool_error():
    result = format_curl_result(
        {
            "url": "https://api.example.com/v1/weather",
            "status_code": 200,
            "content_type": "text/plain",
            "body": "Unexpected error while streaming data: timeoutReached",
            "truncated": False,
        }
    )

    assert result.startswith("toolError:")
    assert "timeoutReached" in result


def test_web_search_result_formats_image_markdown():
    result = format_web_search_results(
        [
            {
                "title": "Example",
                "url": "https://example.com/page",
                "snippet": "A result with an image.",
                "image": "https://example.com/image.jpg",
            }
        ]
    )

    assert "image: https://example.com/image.jpg" in result
    assert "markdownImages:" in result
    assert "![web image](https://example.com/image.jpg)" in result


def test_curl_result_formats_image_url_markdown():
    result = format_curl_result(
        {
            "url": "https://api.example.com/photos",
            "status_code": 200,
            "content_type": "application/json",
            "body": '{"image":"https://cdn.example.com/photo.png"}',
            "truncated": False,
        }
    )

    assert "markdownImages:" in result
    assert "![api image](https://cdn.example.com/photo.png)" in result


def test_curl_result_formats_image_content_markdown():
    result = format_curl_result(
        {
            "url": "https://api.example.com/photo",
            "status_code": 200,
            "content_type": "image/png",
            "body": "",
            "truncated": False,
        }
    )

    assert "markdownImages:" in result
    assert "![api image](https://api.example.com/photo)" in result


def test_model_view_includes_python_only_when_enabled():
    settings = make_tool_settings(
        web_search_mode="off",
        rag_mode="off",
        curl_mode="off",
        python_mode="auto",
        file_editor_mode="off",
        mcp_mode="off",
    )

    assert settings.model_view() == {"available": ["python"]}


def test_build_openai_tools_includes_python_when_enabled():
    settings = make_tool_settings(
        web_search_mode="off",
        rag_mode="off",
        curl_mode="off",
        python_mode="auto",
        file_editor_mode="off",
        mcp_mode="off",
    )

    [tool] = build_openai_tools(settings)

    assert tool["function"]["name"] == "python"
    assert "current working directory is the artifact directory" in tool["function"]["description"]
    assert "Markdown images" in tool["function"]["description"]


def test_parse_python_tool_call_when_enabled():
    settings = make_tool_settings(python_mode="auto")

    [request] = parse_openai_tool_calls(
        {
            "tool_calls": [
                {
                    "id": "call_py",
                    "type": "function",
                    "function": {
                        "name": "python",
                        "arguments": '{"code":"print(2 + 2)"}',
                    },
                }
            ]
        },
        settings,
    )

    assert request.id == "call_py"
    assert request.name == "python"
    assert request.code == "print(2 + 2)"


def test_python_request_requires_auto_mode():
    settings = make_tool_settings(python_mode="off")

    with pytest.raises(ValueError, match="python can only be called"):
        parse_openai_tool_calls(
            {
                "tool_calls": [
                    {
                        "id": "call_py",
                        "type": "function",
                        "function": {
                            "name": "python",
                            "arguments": '{"code":"print(2 + 2)"}',
                        },
                    }
                ]
            },
            settings,
        )

def test_python_result_formats_artifacts():
    result = format_python_result(
        {
            "return_code": 0,
            "stdout": "mean=4\n",
            "stderr": "",
            "artifact_dir": "backend/runtime/python_runs/run_1",
            "files": ["backend/runtime/python_runs/run_1/chart.png"],
        }
    )

    assert result.startswith("pythonResult:")
    assert "artifactDir: backend/runtime/python_runs/run_1" in result
    assert "chart.png" in result
    assert "markdownImages:" in result
    assert "![artifact](backend/runtime/python_runs/run_1/chart.png)" in result
    assert "stdout: mean=4" in result


def test_python_nonzero_formats_as_tool_error():
    result = format_python_result(
        {
            "return_code": 1,
            "stdout": "",
            "stderr": "boom",
            "artifact_dir": "backend/runtime/python_runs/run_1",
            "files": [],
        }
    )

    assert result.startswith("toolError:")
    assert "returnCode: 1" in result
    assert "stderr: boom" in result


def test_mcp_result_formats_image_artifacts():
    result = format_mcp_result(
        {
            "action": "callTool",
            "server": "rent",
            "tool": "api_request",
            "response": '{"body_base64":"<image base64 omitted; see files>"}',
            "files": ["backend/runtime/mcp_artifacts/run_1/image_01.png"],
        }
    )

    assert result.startswith("mcpResult:")
    assert "files:" in result
    assert "markdownImages:" in result
    assert "![mcp artifact](backend/runtime/mcp_artifacts/run_1/image_01.png)" in result


def test_file_editor_approval_defaults_to_auto_and_normalizes_aliases():
    settings = make_tool_settings(file_editor_mode="auto")
    read_only_settings = make_tool_settings(file_editor_mode="auto", file_editor_approval="read-only")

    assert settings.file_editor.approval == "auto"
    assert read_only_settings.file_editor.approval == "readOnly"

    with pytest.raises(ValueError, match="file_editor_approval"):
        make_tool_settings(file_editor_mode="auto", file_editor_approval="always")

def test_model_view_includes_file_editor_only_when_enabled():
    settings = make_tool_settings(
        web_search_mode="off",
        rag_mode="off",
        curl_mode="off",
        python_mode="off",
        file_editor_mode="auto",
        mcp_mode="off",
    )

    assert settings.model_view() == {"available": ["fileEditor"]}


def test_build_openai_tools_includes_file_editor_when_enabled():
    settings = make_tool_settings(
        web_search_mode="off",
        rag_mode="off",
        curl_mode="off",
        python_mode="off",
        file_editor_mode="auto",
        mcp_mode="off",
    )

    [tool] = build_openai_tools(settings)

    assert tool["function"]["name"] == "fileEditor"
    assert "replace" in tool["function"]["description"]
    assert "approvalRequired" in tool["function"]["description"]


def test_parse_file_editor_tool_call_when_enabled():
    settings = make_tool_settings(file_editor_mode="auto")

    [request] = parse_openai_tool_calls(
        {
            "tool_calls": [
                {
                    "id": "call_file",
                    "type": "function",
                    "function": {
                        "name": "fileEditor",
                        "arguments": '{"action":"replace","path":"a.py","oldText":"x","newText":"y"}',
                    },
                }
            ]
        },
        settings,
    )

    assert request.id == "call_file"
    assert request.name == "fileEditor"
    assert request.file_edit is not None
    assert request.file_edit.action == "replace"
    assert request.file_edit.path == "a.py"
    assert request.file_edit.old_text == "x"
    assert request.file_edit.new_text == "y"


def test_file_editor_request_requires_auto_mode():
    settings = make_tool_settings(file_editor_mode="off")

    with pytest.raises(ValueError, match="fileEditor can only be called"):
        parse_openai_tool_calls(
            {
                "tool_calls": [
                    {
                        "id": "call_file",
                        "type": "function",
                        "function": {
                            "name": "fileEditor",
                            "arguments": '{"action":"read","path":"a.py"}',
                        },
                    }
                ]
            },
            settings,
        )


def test_file_editor_result_formats_content_and_files():
    result = format_file_editor_result(
        {"action": "read", "path": "a.py", "files": ["a.py"], "content": "print('hi')\n", "diff": "--- a.py\n+++ a.py\n"}
    )

    assert result.startswith("fileEditorResult:")
    assert "path: a.py" in result
    assert "- a.py" in result
    assert "content:\nprint('hi')" in result
    assert "diff:\n--- a.py" in result


def test_model_view_includes_mcp_only_when_enabled():
    settings = make_tool_settings(
        web_search_mode="off",
        rag_mode="off",
        curl_mode="off",
        python_mode="off",
        file_editor_mode="off",
        mcp_mode="auto",
    )

    assert settings.model_view() == {"available": ["mcp"]}


def test_build_openai_tools_includes_mcp_when_enabled():
    settings = make_tool_settings(
        web_search_mode="off",
        rag_mode="off",
        curl_mode="off",
        python_mode="off",
        file_editor_mode="off",
        mcp_mode="auto",
    )

    [tool] = build_openai_tools(settings)

    assert tool["function"]["name"] == "mcp"
    assert "configured MCP servers" in tool["function"]["description"]


def test_parse_mcp_tool_call_when_enabled():
    settings = make_tool_settings(mcp_mode="auto")

    [request] = parse_openai_tool_calls(
        {
            "tool_calls": [
                {
                    "id": "call_mcp",
                    "type": "function",
                    "function": {
                        "name": "mcp",
                        "arguments": '{"action":"callTool","server":"local","tool":"echo","arguments":{"text":"hi"}}',
                    },
                }
            ]
        },
        settings,
    )

    assert request.id == "call_mcp"
    assert request.name == "mcp"
    assert request.mcp_request is not None
    assert request.mcp_request.action == "callTool"
    assert request.mcp_request.server == "local"
    assert request.mcp_request.tool == "echo"
    assert request.mcp_request.arguments == {"text": "hi"}


def test_mcp_request_requires_auto_mode():
    settings = make_tool_settings(mcp_mode="off")

    with pytest.raises(ValueError, match="mcp can only be called"):
        parse_openai_tool_calls(
            {
                "tool_calls": [
                    {
                        "id": "call_mcp",
                        "type": "function",
                        "function": {
                            "name": "mcp",
                            "arguments": '{"action":"listServers"}',
                        },
                    }
                ]
            },
            settings,
        )


def test_invalid_mcp_mode_is_rejected():
    with pytest.raises(ValueError, match="mcp_mode"):
        make_tool_settings(mcp_mode="on")
