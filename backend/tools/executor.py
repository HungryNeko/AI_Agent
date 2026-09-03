"""Execute model-requested tools."""

from __future__ import annotations

import re
from typing import Any

from tools import WebSearch, curl, fileEditor, mcp, rag
from tools import python as python_tool
from tools.request import ToolRequest
from tools.settings import ToolSettings

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")
IMAGE_URL_RE = re.compile(
    r"https?://[^\s<>'\")]+?\.(?:png|jpg|jpeg|gif|webp|svg)(?:\?[^\s<>'\")]*)?",
    re.IGNORECASE,
)


def execute_tool(request: ToolRequest, settings: ToolSettings) -> str:
    """Run one tool call and return content for a `role=tool` message."""

    try:
        if request.name == "webSearch":
            results = WebSearch.search(request.query, settings.web_search)
            return format_web_search_results(results)
        if request.name == "rag":
            results = rag.search(request.query, settings.rag)
            return format_rag_results(results)
        if request.name == "curl":
            result = curl.get(request.url, settings.curl)
            return format_curl_result(result)
        if request.name == "python":
            result = python_tool.run(request.code, settings.python)
            return format_python_result(result)
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
    except NotImplementedError as exc:
        return f'toolError: "{exc}"'
    except Exception as exc:
        return f'toolError: "{request.name} failed: {exc}"'

    return f'toolError: "unknown tool: {request.name}"'


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
