from agent.config import get_model_config


def test_get_default_model_from_json():
    config = get_model_config()

    assert config.provider == "deepseek"
    assert config.model_id == "deepseek-chat"


def test_get_model_by_alias():
    config = get_model_config("deepseek-reasoner")

    assert config.provider == "deepseek"
    assert config.model_id == "deepseek-reasoner"
