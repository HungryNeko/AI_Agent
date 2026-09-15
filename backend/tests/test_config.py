from agent import config as config_module
from agent.config import get_model_config


def test_get_default_model_from_json():
    config = get_model_config()

    assert config.provider == "deepseek"
    assert config.model_id == "deepseek-chat"


def test_get_model_by_alias():
    config = get_model_config("deepseek-reasoner")

    assert config.provider == "deepseek"
    assert config.model_id == "deepseek-reasoner"


def test_model_config_uses_public_example_when_local_base_is_missing(tmp_path, monkeypatch):
    missing = tmp_path / "api_configs.json"
    example = tmp_path / "api_configs.example.json"
    example.write_text(
        '{"default_provider":"demo","default_model":"demo-model",'
        '"providers":{"demo":{"base_url":"https://example.com/v1","models":[]}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", missing)
    monkeypatch.setattr(config_module, "CONFIG_EXAMPLE_PATH", example)

    config = config_module.get_model_config()

    assert config.provider == "demo"
    assert config.model_id == "demo-model"
    assert config.base_url == "https://example.com/v1"
