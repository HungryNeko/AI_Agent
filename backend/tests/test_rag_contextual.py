import pytest

from tools import rag
from tools.rag import contextualize
from tools.rag.models import BuildOptions

REPORT_DOC = """# ACME Annual Report

## Results

Revenue grew 3% over the previous quarter, driven by subscriptions.

## Outlook

The company expects steady demand next year.
"""


class FakeLlm:
    def __init__(self, answer="From the ACME annual report, section on results: quarterly revenue growth."):
        self.answer = answer
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if isinstance(self.answer, Exception):
            raise self.answer
        return {"content": self.answer}


@pytest.fixture
def fake_llm(monkeypatch):
    llm = FakeLlm()
    monkeypatch.setattr(contextualize, "complete_chat_once", llm)
    return llm


CONTEXTUAL = BuildOptions(contextual="llm", context_model="p:ctx")


def test_headings_become_a_section_path_for_every_chunk():
    headings = contextualize.outline(REPORT_DOC)
    offset = REPORT_DOC.index("Revenue")

    assert contextualize.heading_path(headings, offset) == "ACME Annual Report > Results"
    assert contextualize.heading_path(headings, REPORT_DOC.index("The company")) == "ACME Annual Report > Outlook"
    assert contextualize.heading_path(headings, 0) == "ACME Annual Report"
    assert contextualize.heading_path([], 5) == ""


def test_the_context_prompt_puts_the_document_first_so_the_prompt_cache_serves_it(fake_llm):
    context = contextualize.request_context("p:ctx", REPORT_DOC, "Revenue grew 3%")

    messages, kwargs = fake_llm.calls[0]
    user = messages[1]["content"]
    assert messages[0]["content"] == contextualize.CONTEXT_PROMPT
    assert user.startswith("<document>\n# ACME Annual Report") and user.endswith("<chunk>\nRevenue grew 3%\n</chunk>")
    assert kwargs["model"] == "p:ctx" and kwargs["max_tokens"] == contextualize.CONTEXT_MAX_TOKENS
    assert kwargs["reasoning_effort"] == "off"  # no thinking for one sentence of context
    assert context.startswith("From the ACME annual report")


def test_long_documents_are_cut_to_an_outline_and_a_grid_aligned_window():
    document = "# Title\n\n" + ("filler sentence. " * 3000)
    headings = contextualize.outline(document)

    near = contextualize.document_view(document, 10_000, headings)
    also_near = contextualize.document_view(document, 12_000, headings)

    assert near == also_near  # neighbouring chunks send the same text, so the provider's cache is hit
    assert near.startswith("[outline]\n# Title") and len(near) < 2 * contextualize.WINDOW_CHARS + 2_000


def test_the_context_is_indexed_so_a_chunk_is_found_by_words_it_never_states(rag_env, fake_llm):
    rag_env.write("knowledge", "report.md", REPORT_DOC)
    settings = rag_env.settings()

    status = rag.index_status(settings, CONTEXTUAL)
    plain = rag.search_chunks("ACME quarterly revenue", settings, method="bm25").results

    assert status["contextual"] == "llm" and status["context_calls"] == status["chunk_count"] > 0
    assert plain and "ACME" in plain[0].context
    assert "ACME" not in plain[0].text.replace("# ACME Annual Report", "")  # the words came from the context
    assert "context: From the ACME annual report" in rag.search("ACME quarterly revenue", settings)[0]


def test_without_contextual_retrieval_that_chunk_is_not_found_by_context_words(rag_env, fake_llm):
    rag_env.write("knowledge", "report.md", "Revenue grew 3% over the previous quarter.")
    settings = rag_env.settings()

    rag.index_status(settings, BuildOptions())

    assert fake_llm.calls == []
    assert rag.search_chunks("ACME", settings, method="bm25").results == []


def test_contexts_are_cached_per_chunk_so_a_refresh_only_pays_for_new_text(rag_env, fake_llm):
    rag_env.write("knowledge", "a.md", "# A\n\nAlpha text about apples.")
    second = rag_env.write("knowledge", "b.md", "# B\n\nBeta text about bananas.")
    settings = rag_env.settings()

    first = rag.index_status(settings, CONTEXTUAL)
    second.write_text("# B\n\nBeta text about blueberries.", encoding="utf-8")
    refreshed = rag.index_status(settings, CONTEXTUAL)

    assert first["context_calls"] == 2
    assert refreshed["context_calls"] == 1 and refreshed["context_reused"] == 1
    assert len(fake_llm.calls) == 3


def test_a_failing_model_stops_after_a_few_calls_and_the_chunks_stay_pending(rag_env, monkeypatch):
    llm = FakeLlm(RuntimeError("401 bad key"))
    monkeypatch.setattr(contextualize, "complete_chat_once", llm)
    for index in range(12):
        rag_env.write("knowledge", f"doc{index}.md", f"# Doc {index}\n\nText {index}.")
    settings = rag_env.settings()

    status = rag.index_status(settings, CONTEXTUAL)

    assert len(llm.calls) < 12  # gave up instead of paying for every chunk
    assert status["context_failures"] >= contextualize.GIVE_UP_AFTER_FAILURES
    assert status["context_pending"] == 12 and status["chunk_count"] == 12  # still searchable, without context
    assert rag.search("Text 3", settings)


def test_searching_never_calls_the_llm_for_contexts_but_a_refresh_finishes_the_job(rag_env, monkeypatch):
    llm = FakeLlm(RuntimeError("down"))
    monkeypatch.setattr(contextualize, "complete_chat_once", llm)
    note = rag_env.write("knowledge", "a.md", "# A\n\nAlpha.")
    settings = rag_env.settings()
    rag.index_status(settings, CONTEXTUAL)
    note.write_text("# A\n\nAlpha changed.", encoding="utf-8")
    calls_before = len(llm.calls)

    assert rag.search("Alpha changed", settings)  # the stale index is rebuilt for the search, offline
    assert len(llm.calls) == calls_before
    assert rag.load_index(settings).pending_contexts == 1

    working = FakeLlm()
    monkeypatch.setattr(contextualize, "complete_chat_once", working)
    refreshed = rag.index_status(settings)  # the index remembers its options, and unfinished work is retried

    assert len(working.calls) == 1 and refreshed["context_pending"] == 0
    assert "context: From the ACME" in rag.search("Alpha changed", settings)[0]


def test_the_per_build_budget_defers_the_rest(rag_env, fake_llm, monkeypatch):
    monkeypatch.setattr(contextualize, "CALLS_PER_BUILD", 3)
    for index in range(5):
        rag_env.write("knowledge", f"doc{index}.md", f"# Doc {index}\n\nText {index}.")

    status = rag.index_status(rag_env.settings(), CONTEXTUAL)

    assert status["context_calls"] == 3 and status["context_pending"] == 2


def test_chunks_are_located_even_when_they_start_with_an_overlap_tail():
    texts = ["first block of the document.", "block of the document. second block here.", "second block here. third."]
    document = "first block of the document. second block here. third."

    offsets = contextualize.locate_chunks(document, texts)

    assert offsets == sorted(offsets) and offsets[0] == 0
