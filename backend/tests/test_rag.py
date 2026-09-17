import pytest

from tools import rag
from tools.rag import RagRequest
from tools.rag_chunking import ChunkingMetrics, PreparedChunk
from tools.settings import RagSettings


@pytest.fixture(autouse=True)
def isolated_project_root(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "project_root", lambda: tmp_path)


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
        index_path=str(tmp_path / "index.pkl"),
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
        index_path=str(tmp_path / "index.pkl"),
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


def test_rag_rebuilds_persistent_tfidf_index(tmp_path):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "api.md").write_text("vector search stores chunks", encoding="utf-8")
    index_path = tmp_path / "rag_index.pkl"
    settings = RagSettings(
        mode="auto",
        knowledge_root=str(knowledge),
        user_knowledge_root=str(tmp_path / "missing_user_knowledge"),
        memory_root=str(tmp_path / "missing_memory"),
        user_memory_root=str(tmp_path / "missing_user_memory"),
        skills_root=str(tmp_path / "missing_skills"),
        user_skills_root=str(tmp_path / "missing_user_skills"),
        index_path=str(index_path),
    )

    status = rag.index_status(settings)
    results = rag.search("vector chunks", settings)

    assert status["index"] == "local-vector"
    assert status["embedding"] == "tfidf-char-ngram"
    assert status["chunk_count"] == 1
    assert index_path.is_file()
    assert "sourceType: knowledge" in "\n".join(results)


def test_rag_replaces_incompatible_index_without_unpickling_it(tmp_path):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "note.md").write_text("新向量索引", encoding="utf-8")
    index_path = tmp_path / "index.pkl"
    index_path.write_bytes(b"incompatible dense index that must not be loaded")
    settings = RagSettings(
        knowledge_root=str(knowledge),
        memory_root=str(tmp_path / "missing_memory"),
        skills_root=str(tmp_path / "missing_skills"),
        index_path=str(index_path),
    )

    status = rag.index_status(settings)

    assert status["embedding"] == "tfidf-char-ngram"
    assert index_path.read_bytes().startswith(rag.INDEX_MAGIC)


def test_rag_llm_chunk_cache_only_reprocesses_changed_documents(tmp_path, monkeypatch):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    first = knowledge / "first.md"
    second = knowledge / "second.md"
    first.write_text("first version", encoding="utf-8")
    second.write_text("unchanged", encoding="utf-8")
    calls = []

    def fake_split(text, **kwargs):
        calls.append(text)
        return [PreparedChunk(text, "planned")], ChunkingMetrics(llm_calls=1)

    monkeypatch.setattr(rag, "split_document", fake_split)
    settings = RagSettings(
        knowledge_root=str(knowledge),
        user_knowledge_root=str(tmp_path / "missing_user_knowledge"),
        memory_root=str(tmp_path / "missing_memory"),
        user_memory_root=str(tmp_path / "missing_user_memory"),
        skills_root=str(tmp_path / "missing_skills"),
        user_skills_root=str(tmp_path / "missing_user_skills"),
        index_path=str(tmp_path / "index.pkl"),
    )

    initial = rag.index_status(settings, split_mode="llm", chunk_model="chunk-model")
    first.write_text("first version updated", encoding="utf-8")
    updated = rag.index_status(settings, split_mode="llm", chunk_model="chunk-model")
    second.unlink()
    deleted = rag.index_status(settings, split_mode="llm", chunk_model="chunk-model")

    assert len(calls) == 3
    assert initial["documents_processed"] == 2
    assert updated["documents_processed"] == 1
    assert updated["documents_reused"] == 1
    assert deleted["documents_processed"] == 0
    assert deleted["documents_reused"] == 1
    assert deleted["documents_removed"] == 1
    assert deleted["llm_calls"] == 0


def test_rag_ingests_uploaded_docx_as_markdown(tmp_path, monkeypatch):
    from docx import Document

    upload = tmp_path / "backend" / "runtime" / "uploads" / ("a" * 32) / "report.docx"
    upload.parent.mkdir(parents=True)
    document = Document()
    document.add_heading("Quarterly Report", level=1)
    document.add_paragraph("Revenue increased in the current quarter.")
    document.save(upload)
    knowledge = tmp_path / "user_knowledge"
    settings = RagSettings(
        knowledge_root=str(tmp_path / "missing_knowledge"),
        user_knowledge_root=str(knowledge),
        memory_root=str(tmp_path / "missing_memory"),
        user_memory_root=str(tmp_path / "missing_user_memory"),
        skills_root=str(tmp_path / "missing_skills"),
        user_skills_root=str(tmp_path / "missing_user_skills"),
        index_path=str(tmp_path / "index.pkl"),
    )
    monkeypatch.setattr(rag, "project_root", lambda: tmp_path)

    result = rag.ingest_uploaded_file(
        RagRequest(
            action="ingest",
            path="backend/runtime/uploads/" + "a" * 32 + "/report.docx",
            name="quarterly-report.md",
            split_mode="simple",
        ),
        settings,
    )

    saved = knowledge / "quarterly-report.md"
    assert result["path"].endswith("user_knowledge/quarterly-report.md")
    assert result["format"] == "docx"
    assert saved.is_file()
    assert "Imported from `report.docx` using fileReader" in saved.read_text(encoding="utf-8")
    assert "Revenue increased" in saved.read_text(encoding="utf-8")
    assert result["index"]["split_mode"] == "simple"
