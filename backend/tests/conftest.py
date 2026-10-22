from pathlib import Path
import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


import pytest


@pytest.fixture(autouse=True)
def isolated_app_settings(tmp_path_factory, monkeypatch):
    """Tests must not depend on the developer's own settings (for instance a security level set in Docker)."""

    try:
        from agent import app_settings
    except ImportError:
        return  # backend dependencies are not installed (the CI deployment-config check): nothing to isolate

    # A directory of its own: many tests treat every JSON file in tmp_path as their data.
    path = tmp_path_factory.mktemp("app-settings") / "settings.json"
    # Tests that run code expect the backend's own rights; the security tests set their levels themselves.
    path.write_text('{"security": {"level": "L3"}}', encoding="utf-8")
    monkeypatch.setattr(app_settings, "SETTINGS_PATH", path)
