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


class RagEnv:
    """A throw-away project for RAG tests: every source lives in tmp_path, nothing from the real data folders."""

    def __init__(self, root):
        self.root = root

    def settings(self, **overrides):
        from tools.settings import RagSettings

        values = {
            "mode": "auto",
            "knowledge_root": str(self.root / "knowledge"),
            "user_knowledge_root": str(self.root / "user_knowledge"),
            "memory_root": str(self.root / "memory"),
            "user_memory_root": str(self.root / "user_memory"),
            "skills_root": str(self.root / "skills"),
            "user_skills_root": str(self.root / "user_skills"),
            "index_path": str(self.root / "index" / "index.pkl"),
        }
        values.update(overrides)
        return RagSettings(**values)

    def write(self, kind: str, name: str, text: str):
        if kind == "skill":
            path = self.root / "skills" / name / "SKILL.md"
        else:
            path = self.root / kind / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path


@pytest.fixture
def rag_env(tmp_path, monkeypatch):
    from tools.rag import paths

    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)
    return RagEnv(tmp_path)
