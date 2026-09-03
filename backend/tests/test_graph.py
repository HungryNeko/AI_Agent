from agent import graph


def test_run_agent_calls_model_once(monkeypatch):
    payloads = []

    def fake_complete_chat_once(messages, *, model=None, tools=None):
        payloads.append({"messages": messages, "model": model, "tools": tools})
        return {"role": "assistant", "content": "hi"}

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete_chat_once)
    monkeypatch.setattr(
        graph,
        "format_current_time",
        lambda: 'currentTime: "2026-09-02T12:00:00-07:00"\nreferenceUTC: "2026-09-02T19:00:00Z"',
    )

    result = graph.run_agent(
        "hello",
        model="deepseek-chat",
        web_search=True,
        rag_mode="auto",
        curl_mode="off",
        python_mode="off",
        file_editor_mode="off",
        mcp_mode="off",
        conversation_summary="summary",
    )

    assert result == "hi"
    assert payloads[0]["model"] == "deepseek-chat"
    assert [tool["function"]["name"] for tool in payloads[0]["tools"]] == ["webSearch", "rag"]
    assert payloads[0]["messages"][0]["role"] == "system"
    assert "Tool request format reminder:" in payloads[0]["messages"][0]["content"]
    assert payloads[0]["messages"][-1]["role"] == "user"
    assert 'currentTime: "2026-09-02T12:00:00-07:00"' in payloads[0]["messages"][-1]["content"]
    assert 'conversationSummary: "summary"' in payloads[0]["messages"][-1]["content"]
    assert 'available: ["webSearch", "rag"]' in payloads[0]["messages"][-1]["content"]


def test_assistant_step_keeps_tool_call_content_out_of_final_response(monkeypatch):
    def fake_complete_chat_once(messages, *, model=None, tools=None):
        return {
            "role": "assistant",
            "content": "I will check with a tool first.",
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

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete_chat_once)

    update = graph.assistant_step(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "settings": graph.make_tool_settings(web_search_mode="auto"),
        }
    )

    assert update["response"] == ""
    assert update["messages"][-1]["content"] == "I will check with a tool first."

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


def test_instruction_is_only_in_initial_system_prompt(monkeypatch):
    payloads = []

    def fake_complete_chat_once(messages, *, model=None, tools=None):
        payloads.append(messages)
        return {"role": "assistant", "content": "ok"}

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete_chat_once)
    monkeypatch.setattr(graph, "load_instruction", lambda: "Use short answers.")

    state = graph.new_chat_state()
    state = graph.run_turn(state, "first")
    graph.run_turn(state, "second")

    assert "instruction:\nUse short answers." in payloads[0][0]["content"]
    assert "instruction:\nUse short answers." not in payloads[0][-1]["content"]
    assert "instruction:\nUse short answers." not in payloads[1][-1]["content"]

def test_stream_turn_emits_tool_call_before_final_answer(monkeypatch):
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
                            "arguments": '{"query":"Los Angeles weather"}',
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "final answer"}

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete_chat_once)
    monkeypatch.setattr(graph, "execute_tool", lambda request, settings: "webSearchResult: result")

    events = list(graph.stream_turn(graph.new_chat_state(web_search_mode="auto"), "hello"))

    assert events[0] == {
        "type": "tool_call",
        "tool": "webSearch",
        "query": "Los Angeles weather",
        "text": "webSearch: Los Angeles weather",
    }
    assert events[-1]["type"] == "assistant"
    assert events[-1]["text"] == "final answer"
    assert events[-1]["state"]["response"] == "final answer"


def test_stream_turn_emits_settings_changed_event(monkeypatch):
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
                            "name": "settings",
                            "arguments": '{"action":"update","patch":{"ui":{"theme":"dark"}}}',
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "settings updated"}

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete_chat_once)
    monkeypatch.setattr(graph, "execute_tool", lambda request, settings: 'settingsResult:\n{"status":"saved"}')

    events = list(graph.stream_turn(graph.new_chat_state(automation_mode="auto"), "dark theme"))

    assert {"type": "settings_changed", "tool": "settings", "text": 'settingsResult:\n{"status":"saved"}'} in events
    assert events[-1]["text"] == "settings updated"


def test_stream_turn_emits_assistant_progress_before_tool_call(monkeypatch):
    calls = []

    def fake_complete_chat_once(messages, *, model=None, tools=None):
        calls.append(messages)
        if len(calls) == 1:
            return {
                "role": "assistant",
                "content": "I will check current weather data first.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "curl",
                            "arguments": '{"url":"https://api.open-meteo.com/v1/forecast"}',
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "The API returned current weather data."}

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete_chat_once)
    monkeypatch.setattr(graph, "execute_tool", lambda request, settings: "curlResult: result")

    events = list(graph.stream_turn(graph.new_chat_state(curl_mode="auto"), "hello"))

    assert events[0] == {
        "type": "assistant_progress",
        "text": "I will check current weather data first.",
    }
    assert events[1] == {
        "type": "tool_call",
        "tool": "curl",
        "url": "https://api.open-meteo.com/v1/forecast",
        "text": "curl: https://api.open-meteo.com/v1/forecast",
    }
    assert events[-1]["type"] == "assistant"
    assert events[-1]["text"] == "The API returned current weather data."

def test_stream_turn_emits_progress_between_multiple_tool_calls(monkeypatch):
    calls = []

    def fake_complete_chat_once(messages, *, model=None, tools=None):
        calls.append(messages)
        if len(calls) == 1:
            return {
                "role": "assistant",
                "content": "I will search for the forecast source first.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "webSearch",
                            "arguments": '{"query":"Open-Meteo Los Angeles current weather API"}',
                        },
                    }
                ],
            }
        if len(calls) == 2:
            return {
                "role": "assistant",
                "content": "I found a direct API endpoint. I will fetch it now.",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "curl",
                            "arguments": '{"url":"https://api.open-meteo.com/v1/forecast"}',
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "Summary: current weather data is available."}

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete_chat_once)
    monkeypatch.setattr(graph, "execute_tool", lambda request, settings: f"{request.name}Result")

    events = list(
        graph.stream_turn(
            graph.new_chat_state(web_search_mode="auto", curl_mode="auto"),
            "weather",
        )
    )

    assert [event["type"] for event in events] == [
        "assistant_progress",
        "tool_call",
        "assistant_progress",
        "tool_call",
        "assistant",
    ]
    assert events[0]["text"] == "I will search for the forecast source first."
    assert events[2]["text"] == "I found a direct API endpoint. I will fetch it now."
    assert events[-1]["text"] == "Summary: current weather data is available."

def test_stream_turn_emits_executor_tool_error_event_and_allows_model_retry(monkeypatch):
    calls = []
    tool_payloads = []

    def fake_complete_chat_once(messages, *, model=None, tools=None):
        calls.append(messages)
        tool_payloads.append(tools)
        if len(calls) == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "curl",
                            "arguments": '{"url":"https://api.open-meteo.com/v1/forecast"}',
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "The weather API request failed, so I cannot provide live weather."}

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete_chat_once)
    monkeypatch.setattr(graph, "execute_tool", lambda request, settings: 'toolError: "curl failed: raw timeout"')

    events = list(graph.stream_turn(graph.new_chat_state(curl_mode="auto"), "hello"))

    assert tool_payloads[0]
    assert tool_payloads[1]
    assert any(
        event.get("type") == "error" and "curl failed: raw timeout" in event.get("text", "")
        for event in events
    )
    assert events[-1]["text"] == "The weather API request failed, so I cannot provide live weather."


def test_repeated_tool_call_is_not_blocked_by_backend(monkeypatch):
    calls = []
    executed = []

    def fake_complete_chat_once(messages, *, model=None, tools=None):
        calls.append(messages)
        if len(calls) in {1, 2, 3}:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{len(calls)}",
                        "type": "function",
                        "function": {
                            "name": "curl",
                            "arguments": '{"url":"https://api.open-meteo.com/v1/forecast"}',
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "Stopped retrying the same request."}

    def fake_execute_tool(request, settings):
        executed.append(request.url)
        return 'toolError: "curl failed: timeout"'

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete_chat_once)
    monkeypatch.setattr(graph, "execute_tool", fake_execute_tool)

    events = list(graph.stream_turn(graph.new_chat_state(curl_mode="auto", max_tool_rounds=4), "hello"))

    assert executed == [
        "https://api.open-meteo.com/v1/forecast",
        "https://api.open-meteo.com/v1/forecast",
        "https://api.open-meteo.com/v1/forecast",
    ]
    assert not any("repeated tool call blocked" in event.get("text", "") for event in events)
    assert events[-1]["text"] == "Stopped retrying the same request."

def test_max_tool_rounds_forces_final_answer_without_tools(monkeypatch):
    tool_payloads = []
    calls = []

    def fake_complete_chat_once(messages, *, model=None, tools=None):
        calls.append(messages)
        tool_payloads.append(tools)
        if len(calls) == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "curl",
                            "arguments": '{"url":"https://api.open-meteo.com/v1/forecast"}',
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "The API did not return live data, so I cannot provide current weather."}

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete_chat_once)
    monkeypatch.setattr(graph, "execute_tool", lambda request, settings: 'toolError: "curl returned HTTP 503"')

    events = list(graph.stream_turn(graph.new_chat_state(curl_mode="auto", max_tool_rounds=1), "weather"))

    assert tool_payloads[0]
    assert tool_payloads[1] is None
    assert "toolBudgetExceeded" in calls[1][-1]["content"]
    assert events[-1]["type"] == "assistant"
    assert events[-1]["text"] == "The API did not return live data, so I cannot provide current weather."

def test_stream_turn_emits_tool_error_event(monkeypatch):
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
                            "name": "curl",
                            "arguments": '{"url":"https://api.open-meteo.com/v1/forecast"}',
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "fixed answer"}

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete_chat_once)

    events = list(graph.stream_turn(graph.new_chat_state(curl_mode="off"), "hello"))

    assert any(
        event.get("type") == "error" and "curl can only be called" in event.get("text", "")
        for event in events
    )
    assert events[-1]["text"] == "fixed answer"


def test_stream_turn_emits_python_tool_call(monkeypatch):
    calls = []

    def fake_complete_chat_once(messages, *, model=None, tools=None):
        calls.append(messages)
        if len(calls) == 1:
            return {
                "role": "assistant",
                "content": "I will calculate this with Python.",
                "tool_calls": [
                    {
                        "id": "call_py",
                        "type": "function",
                        "function": {
                            "name": "python",
                            "arguments": '{"code":"print(2 + 2)"}',
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "The result is 4."}

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete_chat_once)
    monkeypatch.setattr(graph, "execute_tool", lambda request, settings: "pythonResult:\nstdout: 4")

    events = list(graph.stream_turn(graph.new_chat_state(python_mode="auto"), "calculate"))

    assert events[0] == {
        "type": "assistant_progress",
        "text": "I will calculate this with Python.",
    }
    assert events[1] == {
        "type": "tool_call",
        "tool": "python",
        "code": "print(2 + 2)",
        "text": "python: print(2 + 2)",
    }
    assert events[-1]["text"] == "The result is 4."

def test_stream_turn_emits_file_editor_tool_call(monkeypatch):
    calls = []

    def fake_complete_chat_once(messages, *, model=None, tools=None):
        calls.append(messages)
        if len(calls) == 1:
            return {
                "role": "assistant",
                "content": "I will edit the file with a stable anchor.",
                "tool_calls": [
                    {
                        "id": "call_file",
                        "type": "function",
                        "function": {
                            "name": "fileEditor",
                            "arguments": '{"action":"replace","path":"a.py","oldText":"x","newText":"y"}',
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "Updated a.py."}

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete_chat_once)
    monkeypatch.setattr(graph, "execute_tool", lambda request, settings: "fileEditorResult:\naction: replace")

    events = list(graph.stream_turn(graph.new_chat_state(file_editor_mode="auto"), "edit"))

    assert events[0] == {
        "type": "assistant_progress",
        "text": "I will edit the file with a stable anchor.",
    }
    assert events[1] == {
        "type": "tool_call",
        "tool": "fileEditor",
        "action": "replace",
        "path": "a.py",
        "text": "fileEditor: replace a.py",
    }
    assert events[-1]["text"] == "Updated a.py."

def test_stream_turn_emits_file_editor_approval_required_event(monkeypatch):
    calls = []

    def fake_complete_chat_once(messages, *, model=None, tools=None):
        calls.append(messages)
        if len(calls) == 1:
            return {
                "role": "assistant",
                "content": "I will prepare the edit for approval.",
                "tool_calls": [
                    {
                        "id": "call_file",
                        "type": "function",
                        "function": {
                            "name": "fileEditor",
                            "arguments": '{"action":"write","path":"a.py","content":"print(1)\\n"}',
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "The edit needs approval before it is applied."}

    monkeypatch.setattr(graph, "complete_chat_once", fake_complete_chat_once)
    monkeypatch.setattr(
        graph,
        "execute_tool",
        lambda request, settings: "fileEditorResult:\naction: write\napprovalRequired: True\ndiff:\n--- a.py",
    )

    events = list(graph.stream_turn(graph.new_chat_state(file_editor_mode="auto"), "edit"))

    assert [event["type"] for event in events] == [
        "assistant_progress",
        "tool_call",
        "approval_required",
        "assistant",
    ]
    assert events[2]["tool"] == "fileEditor"
    assert "approvalRequired: True" in events[2]["text"]
    assert events[-1]["text"] == "The edit needs approval before it is applied."
