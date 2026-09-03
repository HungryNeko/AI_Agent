from tools import rag
from tools.settings import RagSettings


def test_rag_searches_knowledge_memory_and_skills(tmp_path):
    knowledge = tmp_path / "knowledge"
    memory = tmp_path / "memory"
    skills = tmp_path / "skills" / "backend"
    knowledge.mkdir()
    memory.mkdir()
    skills.mkdir(parents=True)
    (knowledge / "api.md").write_text("OpenAI compatible chat completions notes", encoding="utf-8")
    (memory / "MEMORY.md").write_text("User prefers compact agent prompts", encoding="utf-8")
    (skills / "SKILL.md").write_text("Use LangGraph assistant_step tool_call loop", encoding="utf-8")

    settings = RagSettings(
        mode="auto",
        max_results=5,
        knowledge_root=str(knowledge),
        memory_root=str(memory),
        skills_root=str(tmp_path / "skills"),
    )

    results = rag.search("LangGraph compact prompts", settings)

    joined = "\n".join(results)
    assert "sourceType: skill" in joined
    assert "sourceType: memory" in joined
    assert "data" not in joined or "content:" in joined


def test_rag_auto_context_uses_same_search(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "MEMORY.md").write_text("approvalRequired means preview only", encoding="utf-8")
    settings = RagSettings(
        mode="on",
        max_results=1,
        knowledge_root=str(tmp_path / "missing_knowledge"),
        memory_root=str(memory),
        skills_root=str(tmp_path / "missing_skills"),
    )

    context = rag.auto_context("approvalRequired", settings)

    assert context is not None
    assert context.startswith("autoRagResult:")
    assert "preview only" in context


def test_rag_disabled_blocks_model_search(tmp_path):
    settings = RagSettings(mode="off", memory_root=str(tmp_path))

    try:
        rag.search("anything", settings)
    except ValueError as exc:
        assert "disabled" in str(exc)
    else:
        raise AssertionError("rag.search should fail when disabled")