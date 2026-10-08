import json

from agent import config
from tools import models


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"data": [{"id": "z-model"}, {"id": "a-model"}]}


def test_refresh_models_uses_private_key_and_saves_catalog(tmp_path, monkeypatch):
    config_path = tmp_path / "api_configs.json"
    config_path.write_text(
        json.dumps(
            {
                "default_provider": "demo",
                "default_model": "old",
                "providers": {
                    "demo": {
                        "base_url": "https://models.example/v1",
                        "api_key": "secret-value",
                        "models": [{"id": "a-model", "supports_tools": True}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_PATH", config_path)
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        return FakeResponse()

    monkeypatch.setattr(models.httpx, "get", fake_get)
    result = models.refresh_models("demo")

    assert result == {"provider": "demo", "models": ["a-model", "z-model"], "count": 2}
    assert captured["url"] == "https://models.example/v1/models"
    assert captured["headers"]["Authorization"] == "Bearer secret-value"
    saved = json.loads((tmp_path / "api_configs.local.json").read_text(encoding="utf-8"))
    assert saved["providers"]["demo"]["api_key"] == "secret-value"
    assert saved["providers"]["demo"]["models"][0]["supports_tools"] is True


def test_public_config_never_exposes_direct_api_key(monkeypatch):
    monkeypatch.setenv("DEMO_KEY", "environment-secret")
    public = config.public_config(
        {
            "providers": {
                "direct": {"api_key": "direct-secret", "models": []},
                "env": {"api_key_env": "DEMO_KEY", "models": []},
            }
        }
    )

    assert "api_key" not in public["providers"]["direct"]
    assert public["providers"]["direct"]["has_api_key"] is True
    assert public["providers"]["env"]["has_api_key"] is True
