import json

import pytest

from tools import rag
from tools.rag import api_embeddings, query, rerank
from tools.rag.models import BuildOptions, RagRequest

CONCEPTS = {
    "cats": {"cat", "cats", "kitten", "feline"},
    "cars": {"car", "cars", "vehicle", "automobile"},
    "tax": {"tax", "taxes", "levy"},
}


def concept_vector(text: str) -> list[float]:
    words = set(text.lower().replace(".", " ").split())
    return [float(len(words & group)) for group in CONCEPTS.values()] + [0.01]


class FakeEmbeddings:
    """A tiny embedding model where synonyms share a dimension, so similarity is about meaning, not words."""

    def __init__(self):
        self.calls = []
        self.fail = None

    def __call__(self, model, path, payload):
        self.calls.append((model, path, len(payload["input"])))
        if self.fail:
            raise RuntimeError(self.fail)
        return {"data": [{"index": index, "embedding": concept_vector(text)} for index, text in enumerate(payload["input"])]}


@pytest.fixture
def embeddings(monkeypatch):
    fake = FakeEmbeddings()
    monkeypatch.setattr(api_embeddings, "post_json", fake)
    return fake


def corpus(rag_env, **overrides):
    rag_env.write("knowledge", "pets.md", "The kitten sleeps all day.")
    rag_env.write("knowledge", "autos.md", "A vehicle needs fuel.")
    rag_env.write("memory", "errors.md", "Error code E_ORD_17 means the order was rejected.")
    return rag_env.settings(**overrides)


API = BuildOptions(embedding_model="p:embed")


def test_vector_search_with_an_embedding_model_matches_meaning_not_words(rag_env, embeddings):
    settings = corpus(rag_env)

    status = rag.index_status(settings, API)
    by_vector = rag.search_chunks("feline", settings, method="vector")
    by_keywords = rag.search_chunks("feline", settings, method="bm25")

    assert status["embedding"] == "p:embed" and status["embedded_texts"] == 3 and status["embedding_fallback"] == ""
    assert by_vector.results[0].path == "knowledge/pets.md" and by_vector.retrievers == ["vector[p:embed]"]
    assert by_keywords.results == []  # no word in common: only the vectors know


def test_bm25_finds_exact_identifiers_that_vectors_blur(rag_env, embeddings):
    settings = corpus(rag_env)
    rag.index_status(settings, API)

    outcome = rag.search_chunks("E_ORD_17", settings, method="bm25")

    assert [chunk.path for chunk in outcome.results] == ["memory/errors.md"]
    assert outcome.results[0].via == (("bm25", 1),)


def test_hybrid_fuses_both_retrievers_and_reports_who_found_what(rag_env, embeddings):
    settings = corpus(rag_env)
    rag.index_status(settings, API)

    outcome = rag.search_chunks("kitten feline", settings, limit=3)

    assert outcome.method == "hybrid" and outcome.retrievers == ["bm25", "vector[p:embed]"]
    top = outcome.results[0]
    assert top.path == "knowledge/pets.md" and dict(top.via) == {"bm25": 1, "vector": 1} and top.score == 1.0
    text = rag.run_search(RagRequest(query="kitten feline", limit=3), settings)
    assert "retrieval: hybrid (bm25 + vector[p:embed]), reranker: none" in text
    assert "foundBy: bm25 #1, vector #1" in text


def test_the_embedding_cache_means_only_new_chunks_are_embedded(rag_env, embeddings):
    settings = corpus(rag_env)
    rag.index_status(settings, API)
    rag_env.write("knowledge", "taxes.md", "The levy rose again.")

    refreshed = rag.index_status(settings, API)

    assert refreshed["embedded_texts"] == 1 and refreshed["embedding_reused"] == 3
    assert [call[2] for call in embeddings.calls] == [3, 1]


def test_a_failing_embedding_provider_falls_back_to_local_vectors_and_a_refresh_retries(rag_env, embeddings):
    settings = corpus(rag_env)
    embeddings.fail = "HTTP 401"

    status = rag.index_status(settings, API)

    assert status["embedding"] == "tfidf-char-ngram" and "p:embed" in status["embedding_fallback"]
    assert rag.search("kitten", settings)  # search keeps working
    calls = len(embeddings.calls)
    rag.search("kitten", settings)
    assert len(embeddings.calls) == calls  # searching does not retry the provider

    embeddings.fail = None
    healed = rag.index_status(settings)  # a refresh does, with the options the index remembers
    assert healed["embedding"] == "p:embed" and healed["embedding_fallback"] == ""


def test_a_query_time_embedding_failure_degrades_to_bm25_with_a_note(rag_env, embeddings):
    settings = corpus(rag_env)
    rag.index_status(settings, API)
    embeddings.fail = "HTTP 503"

    outcome = rag.search_chunks("order rejected", settings)
    only_vector = rag.search_chunks("order rejected", settings, method="vector")

    assert outcome.results and outcome.retrievers == ["bm25"]
    assert any("vector search unavailable" in note and "used bm25 only" in note for note in outcome.notes)
    assert only_vector.results and only_vector.retrievers == ["bm25"]  # a vector-only request still gets an answer
    assert "note: vector search unavailable" in rag.run_search(RagRequest(query="order rejected", method="vector"), settings)


def test_sources_limit_and_method_are_applied(rag_env, embeddings):
    settings = corpus(rag_env)
    rag.index_status(settings, API)

    only_memory = rag.search_chunks("order vehicle kitten", settings, sources=("memory",))
    one = rag.search_chunks("kitten vehicle order", settings, limit=1)

    assert {chunk.source_type for chunk in only_memory.results} == {"memory"}
    assert len(one.results) == 1
    with pytest.raises(ValueError, match="method"):
        rag.search_chunks("x", settings, method="semantic")
    with pytest.raises(ValueError, match="sources"):
        rag.search_chunks("x", settings, sources=("web",))


def test_no_results_and_empty_queries(rag_env):
    settings = corpus(rag_env)

    assert rag.search_chunks("   ", settings).results == []
    assert rag.run_search(RagRequest(query="向量索引"), settings).startswith('ragResult: "no results"')


def test_the_default_method_comes_from_the_settings(rag_env, embeddings):
    settings = corpus(rag_env, default_method="bm25")
    rag.index_status(settings, API)

    assert rag.search_chunks("kitten", settings).method == "bm25"
    assert rag.search_chunks("kitten", settings, method="vector").method == "vector"


def test_local_vectors_still_work_without_any_model(rag_env):
    settings = corpus(rag_env)

    status = rag.index_status(settings)
    typo = rag.search_chunks("kittens", settings, method="vector")  # character n-grams tolerate word forms

    assert status["embedding"] == "tfidf-char-ngram"
    assert typo.results and typo.results[0].path == "knowledge/pets.md"


# ---------------------------------------------------------------- reranking inside a search


def tool_scores(*scores):
    arguments = json.dumps({"scores": [{"id": index, "score": value} for index, value in scores]})
    return {"tool_calls": [{"function": {"name": "submit_scores", "arguments": arguments}}]}


def test_reranker_mode_follows_the_setting_unless_the_request_overrides_it(rag_env):
    off, llm = rag_env.settings(), rag_env.settings(rerank="llm", rerank_model="p:m")

    assert [query.reranker_mode(off, None), query.reranker_mode(off, True), query.reranker_mode(off, False)] == ["", "llm", ""]
    assert [query.reranker_mode(llm, None), query.reranker_mode(llm, True), query.reranker_mode(llm, False)] == ["llm", "llm", ""]
    assert query.reranker_mode(rag_env.settings(rerank="api"), True) == "api"


def test_a_search_can_ask_for_a_rerank_and_the_reranker_decides_the_order(rag_env, monkeypatch):
    rag_env.write("knowledge", "a.md", "apple pie recipe with cinnamon")
    rag_env.write("knowledge", "b.md", "apple orchard visit notes")
    settings = rag_env.settings()
    rag.index_status(settings)
    seen = []

    def fake(messages, **kwargs):
        seen.append(kwargs["model"])
        return tool_scores((0, 1), (1, 10))  # the reranker prefers whatever the fusion put second

    monkeypatch.setattr(rerank, "complete_chat_once", fake)
    plain = rag.search_chunks("apple", settings, rerank_results=False)
    reranked = rag.search_chunks("apple", settings, rerank_results=True)

    assert reranked.reranker == "llm" and [c.path for c in reranked.results] == [c.path for c in reversed(plain.results)]
    assert all(chunk.reranked for chunk in reranked.results) and seen == [None]
    text = rag.run_search(RagRequest(query="apple", rerank=True), settings)
    assert "reranker: llm" in text and "(reranked)" in text


def test_a_failing_reranker_keeps_the_fused_results_and_says_so(rag_env, monkeypatch):
    rag_env.write("knowledge", "a.md", "apple pie recipe")
    settings = rag_env.settings(rerank="llm")
    rag.index_status(settings)

    def broken(messages, **kwargs):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(rerank, "complete_chat_once", broken)

    outcome = rag.search_chunks("apple", settings)

    assert outcome.results and outcome.reranker == "" and not outcome.results[0].reranked
    assert "rerank (llm) unavailable: quota exceeded" in outcome.notes[0]


def test_the_automatic_context_never_pays_for_a_reranker(rag_env, monkeypatch):
    rag_env.write("memory", "m.md", "approvalRequired means preview only")
    settings = rag_env.settings(mode="on", rerank="llm")
    monkeypatch.setattr(rerank, "complete_chat_once", lambda *args, **kwargs: pytest.fail("auto context must not rerank"))

    context = rag.auto_context("approvalRequired", settings)

    assert context and "preview only" in context
