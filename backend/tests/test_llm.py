from agent import graph, llm


def test_chat_sends_openai_tools_and_returns_content(monkeypatch):
    payloads = []

    def fake_post_chat_completion(base_url, api_key, payload):
        payloads.append(payload)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "done",
                    }
                }
            ]
        }

    monkeypatch.setattr(llm, "post_chat_completion", fake_post_chat_completion)
    monkeypatch.setattr(
        llm,
        "get_model_config",
        lambda model=None: type(
            "Config",
            (),
            {
                "provider": "deepseek",
                "model_id": "deepseek-chat",
                "base_url": "https://api.deepseek.com/v1",
                "api_key_value": "key",
            },
        )(),
    )

    result = llm.chat(
        "hello",
        web_search_mode="auto",
        rag_mode="auto",
        curl_mode="off",
        python_mode="off",
        file_editor_mode="off",
        mcp_mode="off",
    )

    assert result == "done"
    assert payloads[0]["tool_choice"] == "auto"
    assert [tool["function"]["name"] for tool in payloads[0]["tools"]] == ["webSearch", "rag"]
    assert 'conversationSummary: "' not in payloads[0]["messages"][0]["content"]


def test_chat_can_include_context_summary(monkeypatch):
    payloads = []

    def fake_post_chat_completion(base_url, api_key, payload):
        payloads.append(payload)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "done",
                    }
                }
            ]
        }

    monkeypatch.setattr(llm, "post_chat_completion", fake_post_chat_completion)
    monkeypatch.setattr(
        llm,
        "get_model_config",
        lambda model=None: type(
            "Config",
            (),
            {
                "provider": "deepseek",
                "model_id": "deepseek-chat",
                "base_url": "https://api.deepseek.com/v1",
                "api_key_value": "key",
            },
        )(),
    )

    result = llm.chat(
        "hello",
        conversation_summary="Earlier turns were summarized.",
        include_context_rules=True,
    )

    assert result == "done"
    system_prompt = payloads[0]["messages"][0]["content"]
    assert 'conversationSummary: "Earlier turns were summarized."' in system_prompt
    assert "Context rules:" in system_prompt


def test_chat_executes_tool_call_and_sends_tool_result(monkeypatch):
    payloads = []

    def fake_post_chat_completion(base_url, api_key, payload):
        payloads.append(payload)
        if len(payloads) == 1:
            return {
                "choices": [
                    {
                        "message": {
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
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "final answer",
                    }
                }
            ]
        }

    monkeypatch.setattr(llm, "post_chat_completion", fake_post_chat_completion)
    monkeypatch.setattr(graph, "execute_tool", lambda request, settings: "webSearchResult: result")
    monkeypatch.setattr(
        llm,
        "get_model_config",
        lambda model=None: type(
            "Config",
            (),
            {
                "provider": "deepseek",
                "model_id": "deepseek-chat",
                "base_url": "https://api.deepseek.com/v1",
                "api_key_value": "key",
            },
        )(),
    )

    result = llm.chat("hello", web_search_mode="auto")

    assert result == "final answer"
    assert payloads[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "webSearchResult: result",
    }
