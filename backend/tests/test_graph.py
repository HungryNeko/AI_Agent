from agent import graph


def test_run_agent_calls_model_once(monkeypatch):
    payloads = []

    def fake_complete_chat_once(messages, *, model=None, tools=None):
        payloads.append({"messages": messages, "model": model, "tools": tools})
        return {"role": "assistant", "content": "hi"}

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete_chat_once)

    result = graph.run_agent(
        "hello",
        model="deepseek-chat",
        web_search=True,
        rag_mode="auto",
        conversation_summary="summary",
    )

    assert result == "hi"
    assert payloads[0]["model"] == "deepseek-chat"
    assert [tool["function"]["name"] for tool in payloads[0]["tools"]] == ["webSearch", "rag"]
    assert payloads[0]["messages"][0]["role"] == "system"
    assert "Tool request format reminder:" in payloads[0]["messages"][0]["content"]
    assert payloads[0]["messages"][-1]["role"] == "user"
    assert 'conversationSummary: "summary"' in payloads[0]["messages"][-1]["content"]
    assert 'available: ["webSearch", "rag"]' in payloads[0]["messages"][-1]["content"]


def test_graph_executes_tool_call_and_loops_to_final_answer(monkeypatch):
    calls = []

    def fake_complete_chat_once(messages, *, model=None, tools=None):
        calls.append(messages)
        if len(calls) == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "webSearch",
                            "arguments": '{"query":"LangGraph"}',
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "final answer"}

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete_chat_once)
    monkeypatch.setattr(graph, "execute_tool", lambda request, settings: "webSearchResult: result")

    result = graph.run_agent("hello", web_search_mode="auto")

    assert result == "final answer"
    assert calls[1][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "webSearchResult: result",
    }


def test_run_turn_keeps_previous_messages(monkeypatch):
    def fake_complete_chat_once(messages, *, model=None, tools=None):
        return {"role": "assistant", "content": f"seen {len(messages)}"}

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete_chat_once)

    state = graph.new_chat_state()
    state = graph.run_turn(state, "first")
    state = graph.run_turn(state, "second")

    assert state["response"] == "seen 4"
    assert [message["role"] for message in state["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
