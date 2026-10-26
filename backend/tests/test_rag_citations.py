import json

import pytest

from agent import citation_check, graph
from tools import rag
from tools.rag import citations
from tools.rag.citations import Source

RESULT = """ragResult:
retrieval: hybrid (bm25 + vector[tfidf-char-ngram]), reranker: none
note: something
1. sourceType: knowledge
path: knowledge/api.md
chunk: 3
cite: [knowledge/api.md#3]
score: 1.0000
foundBy: bm25 #1, vector #2
section: API > Limits
context: About the API limits.
content: Extraction is capped at 500,000 returned characters.
The reader stops after that.
2. sourceType: memory
path: memory/MEMORY.md
chunk: 0
cite: [memory/MEMORY.md#0]
score: 0.4000
content: The user prefers compact agent prompts."""


def retrieved():
    return citations.sources_in_messages([{"role": "tool", "content": RESULT}])


def check(answer, *, searched=True, known=None, compressed=False):
    return citations.check_answer(
        answer,
        retrieved=retrieved(),
        searched_now=searched,
        lookup=lambda path, chunk: (known or {}).get((path, chunk)),
        history_compressed=compressed,
    )


def test_results_are_parsed_into_citable_sources():
    sources = retrieved()

    assert set(sources) == {("knowledge/api.md", 3), ("memory/MEMORY.md", 0)}
    assert sources[("knowledge/api.md", 3)].text == "Extraction is capped at 500,000 returned characters.\nThe reader stops after that."
    assert sources[("memory/MEMORY.md", 0)].tag == "[memory/MEMORY.md#0]"


def test_auto_injected_context_counts_as_retrieved_but_other_messages_do_not():
    messages = [
        {"role": "user", "content": "autoRagResult:\n" + RESULT.split("1. ", 1)[1].replace("2. ", "\n\n")},
        {"role": "assistant", "content": "I saw [memory/other.md#4] somewhere"},
        {"role": "tool", "content": "curlResult: cite: [x/y.md#1]\ncontent: no"},
    ]

    assert set(citations.sources_in_messages(messages)) == {("knowledge/api.md", 3), ("memory/MEMORY.md", 0)}


def test_a_correct_answer_passes():
    answer = 'Extraction stops at 500,000 characters [knowledge/api.md#3], and "The user prefers compact agent prompts" [memory/MEMORY.md#0].'

    assert check(answer) == []


def test_an_invented_source_is_rejected():
    problems = check("It is 10 MB [knowledge/limits.md#7].")

    assert len(problems) == 1 and "[knowledge/limits.md#7] does not exist" in problems[0]


def test_a_real_chunk_that_was_never_retrieved_is_rejected_unless_history_was_compressed():
    known = {("knowledge/other.md", 1): "Some other text in the index."}

    assert "no search returned it" in check("See [knowledge/other.md#1].", known=known)[0]
    assert check("See [knowledge/other.md#1].", known=known, compressed=True) == []


def test_a_misquote_is_caught_but_formatting_differences_are_not():
    bad = check('The doc says "extraction is capped at ten million characters" [knowledge/api.md#3].')
    good = check('The doc says "Extraction  is CAPPED at 500,000 returned characters" [knowledge/api.md#3].')
    ellipsis = check('It says "Extraction is capped ... The reader stops after that" [knowledge/api.md#3].')
    blockquote = check("> The reader stops after that and more [knowledge/api.md#3]")

    assert len(bad) == 1 and "is not in [knowledge/api.md#3]" in bad[0]
    assert good == [] and ellipsis == []
    assert len(blockquote) == 1  # "and more" is not in the chunk


def test_short_quotes_are_not_checked_and_quotes_without_a_citation_are_ignored():
    assert check('The word "capped" [knowledge/api.md#3] is used.') == []
    assert check('He said "this sentence is not in any chunk at all" and left. See [knowledge/api.md#3].') == []


def test_an_answer_after_a_search_must_cite_or_say_it_used_no_source():
    assert "cites nothing" in check("The limit is 500,000.")[0]
    assert check("Nothing relevant was found.\n\nSources: none used") == []
    assert check("没有相关内容。\n来源：无") == []
    assert check("The limit is 500,000.", searched=False) == []  # no search this turn: nothing to enforce


def test_a_tag_inside_code_is_text_about_a_tag_not_a_citation():
    discussing = "I cannot cite `[knowledge/fake.md#4]`, it does not exist.\n```\n[knowledge/fake.md#5]\n```\nSources: none used"

    assert check(discussing) == []
    assert "does not exist" in check("It is so [knowledge/fake.md#4].")[0]


def test_brackets_that_are_not_file_paths_are_not_citations():
    assert citations.citations_in("array[idx#2] and [see note#3] and [a/b.md#4] and [Title](https://x.y#5)") == [("a/b.md", 4)]


def test_the_correction_message_names_the_problems_and_the_citable_tags():
    text = citations.correction_message(["[a/b.md#1] does not exist."], ["[knowledge/api.md#3]"])

    assert text.startswith("citationError:") and "- [a/b.md#1] does not exist." in text and "[knowledge/api.md#3]" in text


# ---------------------------------------------------------------- inside the agent loop


def tool_call(arguments):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "rag", "arguments": json.dumps(arguments)}}]}


class ScriptedModel:
    """Replies in order and remembers what it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, messages, **kwargs):
        self.seen.append([dict(message) for message in messages])
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return {"role": "assistant", "content": reply} if isinstance(reply, str) else reply


@pytest.fixture
def knowledge(rag_env, monkeypatch):
    rag_env.write("knowledge", "api.md", "# API\n\nExtraction is capped at 500,000 returned characters.")
    settings = rag_env.settings()
    rag.index_status(settings)
    original = graph.make_tool_settings
    monkeypatch.setattr(graph, "make_tool_settings", lambda **kwargs: original(**{**kwargs, **{
        "rag_knowledge_root": settings.knowledge_root, "rag_user_knowledge_root": settings.user_knowledge_root,
        "rag_memory_root": settings.memory_root, "rag_user_memory_root": settings.user_memory_root,
        "rag_skills_root": settings.skills_root, "rag_user_skills_root": settings.user_skills_root,
        "rag_index_path": settings.index_path}}))
    return settings


def run(model, **state_options):
    state = graph.new_chat_state(rag_mode="auto", conversation_mode="agent", python_mode="off", **state_options)
    return list(graph.stream_turn(state, "what is the extraction limit?"))


def test_a_fabricated_citation_is_sent_back_and_the_corrected_answer_is_delivered(knowledge, monkeypatch):
    model = ScriptedModel(
        tool_call({"action": "search", "query": "extraction capped"}),
        "The limit is 10 MB [knowledge/api.md#9].",
        "The limit is 500,000 characters [knowledge/api.md#0].",
    )
    monkeypatch.setattr(graph, "complete_chat_once", model)

    events = run(model)

    final = events[-1]
    assert final["text"] == "The limit is 500,000 characters [knowledge/api.md#0]."
    retry = [event for event in events if event["type"] == "citation_check"]
    assert len(retry) == 1 and retry[0]["status"] == "retry" and "[knowledge/api.md#9] does not exist" in retry[0]["text"]
    correction = model.seen[-1][-1]
    assert correction["role"] == "user" and correction["content"].startswith("citationError:") and "[knowledge/api.md#0]" in correction["content"]
    assert not any(event["type"] == "assistant_progress" and "10 MB" in event["text"] for event in events)  # the bad answer is never shown


def test_an_answer_that_never_gets_fixed_is_delivered_with_a_notice(knowledge, monkeypatch):
    model = ScriptedModel(tool_call({"action": "search", "query": "extraction capped"}), "It is 10 MB [knowledge/api.md#9].")
    monkeypatch.setattr(graph, "complete_chat_once", model)

    events = run(model)

    assert events[-1]["text"].startswith("It is 10 MB [knowledge/api.md#9].")
    assert "Citation check failed" in events[-1]["text"] and "[knowledge/api.md#9] does not exist" in events[-1]["text"]
    assert [event["status"] for event in events if event["type"] == "citation_check"] == ["retry", "retry", "failed"]
    assert len(model.seen) == 1 + 1 + citations.MAX_RETRIES  # the search, the first answer, and the two rewrites


def test_a_search_without_any_citation_in_the_answer_is_sent_back(knowledge, monkeypatch):
    model = ScriptedModel(tool_call({"action": "search", "query": "extraction capped"}), "It is capped at 500,000 characters.", "Capped at 500,000 [knowledge/api.md#0].")
    monkeypatch.setattr(graph, "complete_chat_once", model)

    events = run(model)

    assert events[-1]["text"] == "Capped at 500,000 [knowledge/api.md#0]."


def test_turns_without_a_search_are_not_enforced_and_the_setting_switches_it_off(knowledge, monkeypatch):
    chat = ScriptedModel("Hello! No sources involved.")
    monkeypatch.setattr(graph, "complete_chat_once", chat)
    assert run(chat)[-1]["text"] == "Hello! No sources involved."
    assert len(chat.seen) == 1

    off = ScriptedModel(tool_call({"action": "search", "query": "extraction"}), "Uncited answer.")
    monkeypatch.setattr(graph, "complete_chat_once", off)
    original = graph.saved_rag_search_options
    monkeypatch.setattr(graph, "saved_rag_search_options", lambda: {**original(), "rag_require_citations": False})
    assert run(off)[-1]["text"] == "Uncited answer."


def test_citing_an_unretrieved_but_real_chunk_is_rejected_in_the_loop(knowledge, rag_env, monkeypatch):
    rag_env.write("knowledge", "secret.md", "# Secret\n\nThe launch code is 1234.")
    rag.index_status(knowledge)
    model = ScriptedModel(
        tool_call({"action": "search", "query": "extraction capped", "sources": ["knowledge"], "limit": 1}),
        "Per the notes [knowledge/secret.md#0] the launch code is 1234.",
        "Extraction is capped [knowledge/api.md#0].",
    )
    monkeypatch.setattr(graph, "complete_chat_once", model)

    events = run(model)

    assert any("no search returned it" in event["text"] for event in events if event["type"] == "citation_check")
    assert events[-1]["text"] == "Extraction is capped [knowledge/api.md#0]."


def test_the_router_only_loops_back_for_a_correction_message():
    assert citation_check.route_after_citation_check({"messages": [{"role": "user", "content": "citationError: x"}]}) == "assistant_step"
    assert citation_check.route_after_citation_check({"messages": [{"role": "assistant", "content": "ok"}]}) == "conversation_end"
    assert citation_check.route_after_citation_check({"messages": []}) == "conversation_end"
