from tools import rag_chunking


def test_llm_chunking_uses_forced_boundary_tool_and_reuses_unit_cache(monkeypatch):
    calls = []

    def fake_complete(messages, *, model=None, tools=None, tool_choice=None, max_tokens=None):
        calls.append(
            {
                "messages": messages,
                "model": model,
                "tools": tools,
                "tool_choice": tool_choice,
                "max_tokens": max_tokens,
            }
        )
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "plan-1",
                    "type": "function",
                    "function": {
                        "name": "submit_chunk_plan",
                        "arguments": (
                            '{"groups":['
                            '{"startBlock":0,"endBlock":1,"title":"First"},'
                            '{"startBlock":2,"endBlock":3,"title":"Second"},'
                            '{"startBlock":4,"endBlock":4,"title":"Third"}'
                            "]}"
                        ),
                    },
                }
            ],
        }

    monkeypatch.setattr(rag_chunking, "complete_chat_once", fake_complete)
    text = "\n\n".join(f"Paragraph {index} " + (chr(65 + index) * 440) for index in range(5))
    cache = {}

    first, first_metrics = rag_chunking.split_document(
        text,
        mode="llm",
        model="deepseek:deepseek-chat",
        max_chars=900,
        overlap_chars=100,
        unit_cache=cache,
    )
    second, second_metrics = rag_chunking.split_document(
        text,
        mode="llm",
        model="deepseek:deepseek-chat",
        max_chars=900,
        overlap_chars=100,
        unit_cache=cache,
    )

    assert [item.title for item in first] == ["First", "Second", "Third"]
    assert second == first
    assert first_metrics.llm_calls == 1
    assert second_metrics.llm_calls == 0
    assert second_metrics.reused_units == 1
    assert len(calls) == 1
    assert calls[0]["tool_choice"]["function"]["name"] == "submit_chunk_plan"
    assert calls[0]["max_tokens"] == rag_chunking.LLM_MAX_OUTPUT_TOKENS
    assert "Never repeat or rewrite source text" in calls[0]["messages"][0]["content"]


def test_llm_chunking_falls_back_for_oversized_source_without_calling_model(monkeypatch):
    monkeypatch.setattr(
        rag_chunking,
        "complete_chat_once",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM must not be called")),
    )

    chunks, metrics = rag_chunking.split_document(
        "大" * (rag_chunking.LLM_MAX_SOURCE_CHARS + 1),
        mode="llm",
        model="deepseek:deepseek-chat",
        max_chars=2_000,
        overlap_chars=200,
        unit_cache={},
    )

    assert len(chunks) > 1
    assert metrics.llm_calls == 0
    assert metrics.fallbacks == 1
    assert "capped" in metrics.fallback_reason
