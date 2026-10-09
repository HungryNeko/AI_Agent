"""Execute model-requested tools."""

from __future__ import annotations

import json
import re
from typing import Any

from agent.debug_log import log_event, log_exception
from tools import (
    WebSearch,
    appSettings,
    automation,
    createTool,
    curl,
    fileEditor,
    fileReader,
    history,
    mcp,
    models,
    parameterSave,
    plan,
    question,
    rag,
)
from tools import python as python_tool
from tools.request import ToolRequest, parse_one_tool_call
from tools.settings import ToolSettings

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")
IMAGE_URL_RE = re.compile(
    r"https?://[^\s<>'\")]+?\.(?:png|jpg|jpeg|gif|webp|svg)(?:\?[^\s<>'\")]*)?",
    re.IGNORECASE,
)


def execute_tool(request: ToolRequest, settings: ToolSettings) -> str:
    """Run one tool call and return content for a `role=tool` message."""

    log_event("tool.request", tool=request.name, request=request)
    try:
        with parameterSave.conversation_scope(settings.conversation_id):
            result_text = execute_tool_uncaught(request, settings)
    except NotImplementedError as exc:
        result_text = f'toolError: "{exc}"'
        log_exception("tool.error", exc, tool=request.name, request=request, result=result_text)
    except Exception as exc:
        result_text = f'toolError: "{request.name} failed: {exc}"'
        log_exception("tool.error", exc, tool=request.name, request=request, result=result_text)

    log_event(
        "tool.response",
        tool=request.name,
        request=request,
        is_error=result_text.startswith("toolError:"),
        result=result_text,
    )
    return result_text


def execute_tool_uncaught(request: ToolRequest, settings: ToolSettings) -> str:
    if request.name == "parameterSave":
        return execute_parameter_request(request, settings)
    if request.name.startswith("custom__"):
        return createTool.execute_custom(request.name, request.custom_arguments or {}, settings.python)
    if request.name == "webSearch":
        results = WebSearch.search(request.query, settings.web_search)
        return format_web_search_results(results)
    if request.name == "rag":
        if request.rag_request is None:
            return 'toolError: "rag request is missing rag_request."'
        if request.rag_request.action == "ingest":
            result = rag.ingest_uploaded_file(request.rag_request, settings.rag)
            return "ragIngestResult:\n" + json.dumps(result, ensure_ascii=False, indent=2)
        results = rag.search(request.rag_request.query, settings.rag)
        return format_rag_results(results)
    if request.name == "curl":
        result = curl.get(request.url, settings.curl)
        return format_curl_result(result)
    if request.name == "python":
        result = python_tool.run(request.code, settings.python)
        return format_python_result(result)
    if request.name == "fileReader":
        if request.file_read is None:
            return 'toolError: "fileReader request is missing file_read."'
        result = fileReader.read(request.file_read, settings.file_reader)
        return format_file_reader_result(result)
    if request.name == "fileEditor":
        if request.file_edit is None:
            return 'toolError: "fileEditor request is missing file_edit."'
        result = fileEditor.execute(request.file_edit, settings.file_editor)
        return format_file_editor_result(result)
    if request.name == "mcp":
        if request.mcp_request is None:
            return 'toolError: "mcp request is missing mcp_request."'
        result = mcp.execute(request.mcp_request, settings.mcp)
        return format_mcp_result(result)
    if request.name == "history":
        if request.history_request is None:
            return 'toolError: "history request is missing history_request."'
        return history.execute(request.history_request)
    if request.name == "automation":
        if request.automation_request is None:
            return 'toolError: "automation request is missing automation_request."'
        result = automation.execute(request.automation_request, settings)
        return format_automation_result(result)
    if request.name == "settings":
        if request.settings_request is None:
            return 'toolError: "settings request is missing settings_request."'
        return "settingsResult:\n" + appSettings.execute(request.settings_request)
    if request.name == "model":
        if request.model_request is None:
            return 'toolError: "model request is missing model_request."'
        return models.execute(request.model_request)
    if request.name == "plan":
        if request.plan_request is None:
            return 'toolError: "plan request is missing plan_request."'
        return plan.execute(request.plan_request)
    if request.name == "createTool":
        if request.create_tool_request is None:
            return 'toolError: "createTool request is missing create_tool_request."'
        return createTool.execute(request.create_tool_request)
    if request.name == "question":
        if request.question_request is None:
            return 'toolError: "question request is missing question_request."'
        return question.waiting_result(request.question_request)
    return f'toolError: "unknown tool: {request.name}"'


def execute_parameter_request(request: ToolRequest, settings: ToolSettings) -> str:
    item = request.parameter_request
    if item is None:
        raise ValueError("parameterSave request is missing")
    if item.action == "list":
        result = {"parameters": parameterSave.list_parameters()}
    elif item.action == "inspect":
        result = parameterSave.inspect_parameter(item.ref, path=item.path, offset=item.offset, limit=item.limit)
    elif item.action == "delete":
        result = parameterSave.delete_parameter(item.ref)
    else:
        # Re-parse at dispatch too: wrapping must never bypass current tool permissions.
        inner = parse_one_tool_call({"id": request.id, "function": {
            "name": item.call["tool"], "arguments": json.dumps(item.call["arguments"]),
        }}, settings)
        if inner.name == "python":
            code = inner.code
            if item.save:
                code += "\nfrom ai_agent_parameters import select_path\n"
                for spec in item.save:
                    code += f"save_parameter({spec['name']!r}, select_path(globals(), {spec['path']!r}), {spec['type']!r})\n"
            execution = python_tool.run(code, settings.python)
            if execution["return_code"] != 0:
                raise ValueError(execution["stderr"] or "Python execution failed")
            if not item.save:
                return format_python_result(execution)
            if len(execution.get("parameters", [])) < len(item.save):
                raise ValueError("Python finished before exporting the requested variables")
            result = {"saved": execution.get("parameters", []), "files": execution["files"]}
        elif not item.save:
            return execute_tool_uncaught(inner, settings)
        else:
            value, files = execute_raw_data_tool(inner, settings)
            saved = parameterSave.store_many([
                (spec["name"], parameterSave.select_path(value, spec["path"]), spec["type"])
                for spec in item.save
            ])
            result = {"saved": saved, "files": files}
    return "parameterSaveResult:\n" + json.dumps(result, ensure_ascii=False)


def execute_raw_data_tool(request: ToolRequest, settings: ToolSettings) -> tuple[Any, list[str]]:
    if request.name.startswith("custom__"):
        result = createTool.execute_custom(request.name, request.custom_arguments or {}, settings.python, raw=True)
        return result["result"], result["files"]
    if request.name == "mcp" and request.mcp_request:
        item = request.mcp_request
        if item.action == "listServers":
            return mcp.list_servers(settings.mcp), []
        if item.action == "listTools":
            return mcp.list_tools(item.server, settings.mcp, raw=True)["response"], []
        result = mcp.call_tool(item.server, item.tool, item.arguments, settings.mcp, raw=True)
        response = result["response"]
        payload = response.get("result", {})
        if response.get("error") or payload.get("isError"):
            raise ValueError("MCP error: " + json.dumps(response.get("error") or payload, ensure_ascii=False)[:4000])
        if "structuredContent" in payload:
            return payload["structuredContent"], result["files"]
        content = payload.get("content", [])
        if len(content) == 1 and content[0].get("type") == "text":
            text = content[0].get("text", "")
            try:
                return json.loads(text), result["files"]
            except json.JSONDecodeError:
                return text, result["files"]
        return payload, result["files"]
    if request.name == "webSearch":
        return WebSearch.search(request.query, settings.web_search), []
    if request.name == "rag" and request.rag_request and request.rag_request.action == "search":
        return rag.search(request.rag_request.query, settings.rag), []
    if request.name == "curl":
        result = curl.get(request.url, settings.curl)
        if format_curl_result(result).startswith("toolError:"):
            raise ValueError(format_curl_result(result))
        body = result.get("body", "")
        try:
            return json.loads(body), []
        except json.JSONDecodeError:
            return result, []
    if request.name == "fileReader" and request.file_read:
        return fileReader.read(request.file_read, settings.file_reader), []
    if request.name == "fileEditor" and request.file_edit and request.file_edit.action in {"list", "read"}:
        return fileEditor.execute(request.file_edit, settings.file_editor), []
    raise ValueError("this tool cannot provide a saved data result")


def format_web_search_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return 'webSearchResult: "no results"'
    lines = ["webSearchResult:"]
    image_urls = []
    for index, item in enumerate(results, start=1):
        title = item.get("title") or "untitled"
        url = item.get("url") or ""
        snippet = item.get("snippet") or item.get("content") or ""
        image = item.get("image") or item.get("thumbnail") or item.get("img_src") or ""
        line = f"{index}. {title} {url} {snippet}".strip()
        if image:
            line += f"\nimage: {image}"
            image_urls.append(str(image))
        lines.append(line)
    markdown_images = markdown_image_lines("web image", image_urls)
    if markdown_images:
        lines.append("markdownImages:")
        lines.extend(markdown_images)
    return "\n".join(lines)


def format_rag_results(results: list[str]) -> str:
    if not results:
        return 'ragResult: "no results"'
    lines = ["ragResult:"]
    for index, item in enumerate(results, start=1):
        lines.append(f"{index}. {item}")
    return "\n".join(lines)


def body_looks_like_api_error(body: str) -> bool:
    lowered = body.lower()
    return (
        "unexpected error while streaming data" in lowered
        or "timeoutreached" in lowered
        or lowered.startswith("error:")
    )


def format_curl_result(result: dict[str, Any]) -> str:
    body = str(result.get("body") or "").strip()
    status_code = int(result.get("status_code") or 0)
    url = str(result.get("url") or "")
    content_type = str(result.get("content_type") or "")
    if status_code >= 400 or status_code == 0 or body_looks_like_api_error(body):
        return (
            f'toolError: "curl returned HTTP {status_code}\n'
            f"url: {url}\n"
            f"contentType: {content_type}\n"
            f"body: {body[:1000]}\""
        )
    image_urls = []
    if is_image_content_type(content_type) and url:
        image_urls.append(url)
    image_urls.extend(extract_image_urls(body))

    lines = [
        "curlResult:",
        f"status: {status_code}",
        f"url: {url}",
        f"contentType: {content_type}",
        f"truncated: {str(bool(result.get('truncated'))).lower()}",
        f"body: {body}",
    ]
    markdown_images = markdown_image_lines("api image", image_urls)
    if markdown_images:
        lines.append("markdownImages:")
        lines.extend(markdown_images)
    return "\n".join(lines)


def format_python_result(result: dict[str, Any]) -> str:
    return_code = int(result.get("return_code") or 0)
    stdout = str(result.get("stdout") or "").strip()
    stderr = str(result.get("stderr") or "").strip()
    artifact_dir = str(result.get("artifact_dir") or "")
    files = result.get("files") or []
    if not isinstance(files, list):
        files = []

    prefix = "pythonResult" if return_code == 0 else "toolError"
    lines = [f"{prefix}:", f"returnCode: {return_code}", f"artifactDir: {artifact_dir}"]
    if result.get("parameters"):
        lines.append("savedParameters: " + json.dumps(result["parameters"], ensure_ascii=False))
    if files:
        lines.append("files:")
        lines.extend(f"- {path}" for path in files)
        image_files = [path for path in files if is_image_artifact(str(path))]
        if image_files:
            lines.append("markdownImages:")
            lines.extend(f"![artifact]({path})" for path in image_files)
    if stdout:
        lines.append(f"stdout: {stdout}")
    if stderr:
        lines.append(f"stderr: {stderr}")
    if return_code != 0 and not stderr:
        lines.append("stderr: Python exited with a non-zero status.")
    return "\n".join(lines)


def is_image_artifact(path: str) -> bool:
    return path.lower().endswith(IMAGE_SUFFIXES)


def is_image_content_type(content_type: str) -> bool:
    return content_type.lower().split(";", 1)[0].strip().startswith("image/")


def extract_image_urls(text: str) -> list[str]:
    urls = []
    seen = set()
    for match in IMAGE_URL_RE.findall(text):
        url = match.rstrip(".,;")
        if url not in seen:
            urls.append(url)
            seen.add(url)
    return urls


def markdown_image_lines(label: str, urls: list[str]) -> list[str]:
    lines = []
    seen = set()
    for url in urls:
        url = url.strip()
        if not url or url in seen:
            continue
        lines.append(f"![{label}]({url})")
        seen.add(url)
    return lines


def format_file_editor_result(result: dict[str, Any]) -> str:
    lines = ["fileEditorResult:"]
    for key, value in result.items():
        if key == "files" and isinstance(value, list):
            lines.append("files:")
            lines.extend(f"- {path}" for path in value)
            continue
        if key in {"content", "diff"}:
            lines.append(f"{key}:")
            lines.append(str(value))
            continue
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def format_file_reader_result(result: dict[str, Any]) -> str:
    lines = ["fileReaderResult:"]
    for key, value in result.items():
        if key == "content":
            lines.append("content:")
            lines.append(str(value))
            continue
        if isinstance(value, list):
            lines.append(f"{key}: {json_dumps(value)}")
            continue
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def format_mcp_result(result: dict[str, Any]) -> str:
    lines = ["mcpResult:"]
    for key, value in result.items():
        if key == "servers" and isinstance(value, list):
            lines.append("servers:")
            for server in value:
                if isinstance(server, dict):
                    lines.append(
                        f"- {server.get('name')} enabled={server.get('enabled')} "
                        f"transport={server.get('transport')} url={server.get('url')} "
                        f"command={server.get('command')}"
                    )
            continue
        if key == "files" and isinstance(value, list):
            lines.append("files:")
            lines.extend(f"- {path}" for path in value)
            image_files = [path for path in value if is_image_artifact(str(path))]
            if image_files:
                lines.append("markdownImages:")
                lines.extend(f"![mcp artifact]({path})" for path in image_files)
            continue
        if key == "response":
            lines.append("response:")
            lines.append(str(value))
            continue
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def format_automation_result(result: dict[str, Any]) -> str:
    lines = ["automationResult:"]
    for key, value in result.items():
        if key == "result" and isinstance(value, dict):
            lines.append("result:")
            lines.append(json_dumps(value))
            continue
        if isinstance(value, (dict, list)):
            lines.append(f"{key}:")
            lines.append(json_dumps(value))
            continue
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2, default=str)
