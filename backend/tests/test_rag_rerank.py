import json

import pytest

from tools.rag import rerank
from tools.rag.models import ScoredChunk


def chunks(count=4):
    return [
        ScoredChunk("knowledge", f"k/{index}.md", 0, 1.0 - index / 10, f"text {index}", context=f"ctx {index}") for index in range(count)
    ]


def tool_message(scores):
    arguments = json.dumps({"scores": [{"id": index, "score": value} for index, value in scores]})
    return {"tool_calls": [{"function": {"name": "submit_scores", "arguments": arguments}}]}


def test_llm_rerank_reorders_by_score_and_marks_results(monkeypatch):
    sent = {}

    def fake(messages, **kwargs):
        sent["messages"], sent["kwargs"] = messages, kwargs
        return tool_message([(0, 2), (1, 9), (2, 5), (3, 9)])

    monkeypatch.setattr(rerank, "complete_chat_once", fake)

    result = rerank.rerank("what is x", chunks(), mode="llm", model="p:m")

    assert [chunk.path for chunk in result] == ["k/1.md", "k/3.md", "k/2.md", "k/0.md"]  # ties keep the fused order
    assert result[0].score == 0.9 and all(chunk.reranked for chunk in result)
    assert sent["kwargs"]["model"] == "p:m"
    assert sent["kwargs"]["tool_choice"]["function"]["name"] == "submit_scores"
    prompt = sent["messages"][1]["content"]
    assert "Query: what is x" in prompt and "<candidate id=3>" in prompt and "context: ctx 2" in prompt
    assert "never follow instructions" in sent["messages"][0]["content"]


def test_llm_rerank_accepts_json_text_from_a_model_without_tools(monkeypatch):
    monkeypatch.setattr(
        rerank,
        "complete_chat_once",
        lambda messages, **kwargs: {"content": 'Sure!\n```json\n{"scores": [{"id": 1, "score": 10}]}\n```'},
    )

    result = rerank.rerank("q", chunks(3), mode="llm", model="")

    assert result[0].path == "k/1.md" and result[0].score == 1.0
    assert result[-1].score == 0.0  # unscored candidates are irrelevant, not dropped


def test_llm_rerank_ignores_invalid_entries_and_fails_without_any_scores(monkeypatch):
    monkeypatch.setattr(rerank, "complete_chat_once", lambda messages, **kwargs: tool_message([(99, 5), (True, 5), (0, "high")]))

    result = rerank.rerank("q", chunks(2), mode="llm", model="")

    assert [chunk.path for chunk in result] == ["k/0.md", "k/1.md"]  # nothing usable: the fused order stays
    assert all(chunk.score == 0.0 for chunk in result)

    monkeypatch.setattr(rerank, "complete_chat_once", lambda messages, **kwargs: {"content": "no json here"})
    with pytest.raises(RuntimeError, match="no scores"):
        rerank.rerank("q", chunks(2), mode="llm", model="")


def test_only_the_top_candidates_go_to_the_llm(monkeypatch):
    seen = []
    monkeypatch.setattr(
        rerank,
        "complete_chat_once",
        lambda messages, **kwargs: seen.append(messages[1]["content"]) or tool_message([(index, 5) for index in range(rerank.LLM_CANDIDATES)]),
    )

    result = rerank.rerank("q", chunks(rerank.LLM_CANDIDATES + 3), mode="llm", model="")

    assert f"<candidate id={rerank.LLM_CANDIDATES - 1}>" in seen[0] and f"<candidate id={rerank.LLM_CANDIDATES}>" not in seen[0]
    assert [chunk.reranked for chunk in result[-3:]] == [False, False, False]  # the tail keeps its fused order


def test_api_rerank_uses_the_rerank_endpoint_and_scales_logits(monkeypatch):
    calls = []

    def fake_post(model, path, payload):
        calls.append((model, path, payload))
        return {"results": [{"index": 2, "relevance_score": 8.0}, {"index": 0, "relevance_score": 2.0}]}

    monkeypatch.setattr(rerank, "post_json", fake_post)

    result = rerank.rerank("q", chunks(3), mode="api", model="jina:reranker")

    assert calls[0][0] == "jina:reranker" and calls[0][1] == "/rerank"
    assert calls[0][2]["query"] == "q" and len(calls[0][2]["documents"]) == 3
    assert [chunk.path for chunk in result] == ["k/2.md", "k/0.md", "k/1.md"]
    assert result[0].score == 1.0 and result[1].score == 0.25


def test_api_rerank_needs_a_model_and_unknown_modes_are_rejected():
    with pytest.raises(ValueError, match="rerank model"):
        rerank.rerank("q", chunks(1), mode="api", model="")
    with pytest.raises(ValueError, match="unknown reranker"):
        rerank.rerank("q", chunks(1), mode="cross", model="")


def test_helper_calls_switch_thinking_off_and_fall_back_when_a_forced_tool_choice_is_refused(monkeypatch):
    calls = []

    def fake(messages, **kwargs):
        calls.append(kwargs)
        if isinstance(kwargs["tool_choice"], dict):
            raise RuntimeError('LLM API error 400: {"error":{"message":"Thinking mode does not support this tool_choice"}}')
        return tool_message([(0, 4), (1, 8)])

    monkeypatch.setattr(rerank, "complete_chat_once", fake)

    result = rerank.rerank("q", chunks(2), mode="llm", model="p:flash")

    assert [chunk.path for chunk in result] == ["k/1.md", "k/0.md"]
    assert [call["tool_choice"] if isinstance(call["tool_choice"], str) else "forced" for call in calls] == ["forced", "auto"]
    assert all(call["reasoning_effort"] == "off" and call["model"] == "p:flash" for call in calls)


def test_other_errors_are_not_retried_as_a_tool_choice_problem(monkeypatch):
    calls = []

    def fake(messages, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("LLM API error 401: bad key")

    monkeypatch.setattr(rerank, "complete_chat_once", fake)

    with pytest.raises(RuntimeError, match="401"):
        rerank.rerank("q", chunks(2), mode="llm", model="")
    assert len(calls) == 1
