import pytest

from tools import curl
from tools.settings import CurlSettings


def test_validate_url_blocks_localhost():
    with pytest.raises(ValueError, match="localhost"):
        curl.validate_url("http://localhost:8000")


def test_validate_url_blocks_private_ip():
    with pytest.raises(ValueError, match="private"):
        curl.validate_url("http://127.0.0.1")


def test_curl_get_calls_httpx(monkeypatch):
    class FakeResponse:
        content = b'{"temperature": 23}'
        encoding = "utf-8"
        url = "https://api.open-meteo.com/v1/forecast"
        status_code = 200
        headers = {"content-type": "application/json"}

    def fake_get(url, *, timeout, follow_redirects, headers):
        assert url.startswith("https://api.open-meteo.com")
        assert timeout == 20.0
        assert follow_redirects is True
        assert headers["Accept"].startswith("application/json")
        return FakeResponse()

    monkeypatch.setattr(curl.httpx, "get", fake_get)

    result = curl.get(
        "https://api.open-meteo.com/v1/forecast?latitude=34.05&longitude=-118.24",
        CurlSettings(mode="auto"),
    )

    assert result["status_code"] == 200
    assert '"temperature": 23' in result["body"]

def test_validate_url_extracts_markdown_link():
    url = curl.validate_url(
        "[weather](https://api.open-meteo.com/v1/forecast?latitude=40&longitude=-74)"
    )

    assert url == "https://api.open-meteo.com/v1/forecast?latitude=40&longitude=-74"


def test_curl_get_falls_back_to_system_curl_on_httpx_timeout(monkeypatch):
    def fake_httpx_get(url, settings):
        raise curl.httpx.ReadTimeout("slow")

    def fake_system_curl_get(url, settings):
        return {
            "url": url,
            "status_code": 200,
            "content_type": "application/json",
            "body": '{"ok": true}',
            "truncated": False,
        }

    monkeypatch.setattr(curl, "httpx_get", fake_httpx_get)
    monkeypatch.setattr(curl, "system_curl_get", fake_system_curl_get)

    result = curl.get("https://api.open-meteo.com/v1/forecast", CurlSettings(mode="auto"))

    assert result["status_code"] == 200
    assert result["body"] == '{"ok": true}'


def test_split_curl_output_reads_markers():
    output = (
        '{"ok":true}'
        "\n__AI_AGENT_CURL_STATUS__:200"
        "\n__AI_AGENT_CURL_CONTENT_TYPE__:application/json"
        "\n__AI_AGENT_CURL_URL__:https://api.example.com/data"
    )

    body, status_code, content_type, url = curl.split_curl_output(output)

    assert body == '{"ok":true}'
    assert status_code == 200
    assert content_type == "application/json"
    assert url == "https://api.example.com/data"
