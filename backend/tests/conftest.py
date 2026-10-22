from pathlib import Path
import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


import pytest


@pytest.fixture(autouse=True)
def isolated_app_settings(tmp_path, monkeypatch):
    """Tests must not depend on the developer's own settings (for instance a security level set in Docker)."""

    from agent import app_settings

    monkeypatch.setattr(app_settings, "SETTINGS_PATH", tmp_path / "isolated-settings.json")
