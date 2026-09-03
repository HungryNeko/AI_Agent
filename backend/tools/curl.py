"""Small HTTP GET tool for direct API calls."""

from __future__ import annotations

import ipaddress
import re
import shutil
import subprocess
from typing import Any
from urllib.parse import urlparse

import httpx

from tools.settings import CurlSettings


BLOCKED_HOSTS = {"localhost", "localhost.localdomain"}
CURL_STATUS_MARKER = "\n__AI_AGENT_CURL_STATUS__:"
CURL_TYPE_MARKER = "\n__AI_AGENT_CURL_CONTENT_TYPE__:"
CURL_URL_MARKER = "\n__AI_AGENT_CURL_URL__:"


def get(url: str, settings: CurlSettings) -> dict[str, Any]:
    """Fetch an http(s) URL and return bounded text content."""

    safe_url = validate_url(url)
    try:
        return httpx_get(safe_url, settings)
    except httpx.TimeoutException:
        return system_curl_get(safe_url, settings)


def httpx_get(url: str, settings: CurlSettings) -> dict[str, Any]:
    response = httpx.get(
        url,
        timeout=settings.timeout_seconds,
        follow_redirects=True,
        headers={"Accept": "application/json, text/plain, */*"},
    )
    body_bytes = response.content[: settings.max_bytes]
    truncated = len(response.content) > settings.max_bytes
    return {
        "url": str(response.url),
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type", ""),
        "body": body_bytes.decode(response.encoding or "utf-8", errors="replace"),
        "truncated": truncated,
    }


def system_curl_get(url: str, settings: CurlSettings) -> dict[str, Any]:
    curl_path = shutil.which("curl.exe") or shutil.which("curl")
    if not curl_path:
        raise RuntimeError("httpx timed out and system curl was not found.")

    command = [
        curl_path,
        "-L",
        "--silent",
        "--show-error",
        "--max-time",
        str(max(1, int(settings.timeout_seconds))),
        "--header",
        "Accept: application/json, text/plain, */*",
        "--write-out",
        (
            f"{CURL_STATUS_MARKER}%{{http_code}}"
            f"{CURL_TYPE_MARKER}%{{content_type}}"
            f"{CURL_URL_MARKER}%{{url_effective}}"
        ),
        url,
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=False,
        timeout=settings.timeout_seconds + 2,
        check=False,
    )
    output = completed.stdout.decode("utf-8", errors="replace")
    error = completed.stderr.decode("utf-8", errors="replace").strip()
    if completed.returncode != 0:
        raise RuntimeError(f"system curl failed: {error or completed.returncode}")

    body, status_code, content_type, effective_url = split_curl_output(output)
    body_bytes = body.encode("utf-8", errors="replace")[: settings.max_bytes]
    truncated = len(body_bytes) < len(body.encode("utf-8", errors="replace"))
    return {
        "url": effective_url or url,
        "status_code": status_code,
        "content_type": content_type,
        "body": body_bytes.decode("utf-8", errors="replace"),
        "truncated": truncated,
    }


def split_curl_output(output: str) -> tuple[str, int, str, str]:
    status_index = output.rfind(CURL_STATUS_MARKER)
    type_index = output.rfind(CURL_TYPE_MARKER)
    url_index = output.rfind(CURL_URL_MARKER)
    if status_index < 0 or type_index < 0 or url_index < 0:
        return output, 0, "", ""

    body = output[:status_index]
    status_text = output[status_index + len(CURL_STATUS_MARKER) : type_index].strip()
    content_type = output[type_index + len(CURL_TYPE_MARKER) : url_index].strip()
    effective_url = output[url_index + len(CURL_URL_MARKER) :].strip()
    try:
        status_code = int(status_text)
    except ValueError:
        status_code = 0
    return body, status_code, content_type, effective_url


def validate_url(url: str) -> str:
    normalized_url = extract_url(url)
    parsed = urlparse(normalized_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("curl only allows http(s) URLs.")
    if not parsed.hostname:
        raise ValueError("curl URL must include a host.")
    if parsed.username or parsed.password:
        raise ValueError("curl URL must not include credentials.")

    host = parsed.hostname.lower().strip("[]")
    if host in BLOCKED_HOSTS or host.endswith(".local"):
        raise ValueError("curl does not allow localhost/private hosts.")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return normalized_url

    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
        raise ValueError("curl does not allow localhost/private IP addresses.")
    return normalized_url


def extract_url(text: str) -> str:
    clean_text = text.strip()
    markdown_match = re.fullmatch(r"\[[^\]]+\]\((https?://[^)]+)\)", clean_text)
    if markdown_match:
        return markdown_match.group(1).strip()
    return clean_text
