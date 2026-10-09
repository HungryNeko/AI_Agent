import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent import graph, server, session_store
from tools import createTool, mcp, parameterSave
from tools.executor import execute_tool
from tools.request import parameter_tool, parse_openai_tool_calls
from tools.settings import make_tool_settings


@pytest.fixture(autouse=True)
def local_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(parameterSave, "PARAMETER_ROOT", tmp_path / "parameters")
    monkeypatch.setattr(session_store, "CONVERSATION_ROOT", tmp_path / "conversations")
    monkeypatch.setattr(createTool, "CUSTOM_TOOL_ROOT", tmp_path / "custom_tools")


def tool_call(name, arguments, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def settings_for(tmp_path, **options):
    settings = make_tool_settings(**options)
    return replace(
        settings,
        conversation_id="chat-one",
        python=replace(
            settings.python,
            artifact_root=str(tmp_path / "python_runs"),
            timeout_seconds=45,
        ),
    )


def run_parameter(settings, arguments):
    [request] = parse_openai_tool_calls(
        {"tool_calls": [tool_call("parameterSave", arguments)]}, settings
    )
    output = execute_tool(request, settings)
    assert output.startswith("parameterSaveResult:"), output
    return json.loads(output.split("\n", 1)[1])


def test_types_versions_reload_and_conversation_isolation():
    array = np.arange(12, dtype=np.int32).reshape(3, 4)
    frame = pd.DataFrame({"rent": [100, 200], "label": ["a", "b"]})
    with parameterSave.conversation_scope("first"):
        first = parameterSave.store_many(
            [
                ("values", {"tuple": (1, None, True), "nan": float("nan")}, "auto"),
                ("array", array, "auto"),
                ("frame", frame, "auto"),
            ]
        )
        second = parameterSave.store_many([("values", [9], "auto")])[0]
        assert second["version"] == 2
    with parameterSave.conversation_scope("second"):
        with pytest.raises(ValueError, match="not available"):
            parameterSave.load_parameter(first[0]["ref"])
    # A new scope reads from disk rather than process-local values.
    with parameterSave.conversation_scope("first"):
        assert parameterSave.load_parameter(first[0]["ref"])["tuple"] == (1, None, True)
        np.testing.assert_array_equal(parameterSave.load_parameter(first[1]["ref"]), array)
        assert parameterSave.load_parameter(first[1]["ref"]).dtype == array.dtype
        pd.testing.assert_frame_equal(parameterSave.load_parameter(first[2]["ref"]), frame)
        parameterSave.delete_parameter(second["ref"])
        third = parameterSave.store_many([("values", [10], "auto")])[0]
        assert third["version"] == 3


def test_inspect_child_infers_structure_without_converting_loaded_value():
    rows = [{"asking_price": 1000}, {"asking_price": 2000}]
    value = {"content": "", "data": {"rows": rows}, "success": True, "error": None}
    with parameterSave.conversation_scope("inspection"):
        ref = parameterSave.save_parameter("rent_prices", value)
        assert parameterSave.inspect_parameter(ref)["type"] == "dict"
        child = parameterSave.inspect_parameter(ref, path="data.rows")
        assert child["type"] == "table"
        assert child["pythonType"] == "list"
        assert child["columns"] == ["asking_price"]
        assert child["rowCount"] == 2
        assert child["path"] == "data.rows"
        assert "preview" not in child
        assert parameterSave.load_parameter(ref) == value
        assert parameterSave.load_parameter(ref, path="data.rows") == rows
        assert isinstance(parameterSave.load_parameter(ref, path="data.rows"), list)
        assert parameterSave.load_parameter(ref, "$.data.rows.0.asking_price") == 1000
        assert parameterSave.resolve_references({"$ref": ref, "path": "data.rows"}) == rows


@pytest.mark.parametrize(
    "path",
    [
        "structuredContent.data.rows",
        "data.missing",
        "data.rows.9",
        "data.rows.bad",
        "data.rows.0.asking_price.child",
    ],
)
def test_path_errors_show_keys_and_failed_segment_without_data_values(path):
    root = {
        "content": "DO_NOT_REVEAL_VALUE",
        "data": {"rows": [{"asking_price": 123456789}]},
        "success": True,
        "error": None,
    }
    with pytest.raises(ValueError) as caught:
        parameterSave.select_path(root, path)
    message = str(caught.value)
    assert f"unknown path {path!r}" in message
    assert 'top-level keys: ["content", "data", "success", "error"]' in message
    assert "failed at" in message
    assert "current structure" in message
    assert "DO_NOT_REVEAL_VALUE" not in message
    assert "123456789" not in message


def test_invalid_path_type_and_root_list_diagnostics():
    with pytest.raises(ValueError, match="path must be a string"):
        parameterSave.select_path({}, ["rows"])
    with pytest.raises(ValueError, match='root structure: .*"length": 2'):
        parameterSave.select_path([100, 200], "missing")


@pytest.mark.parametrize(
    "save",
    [
        {"name": "rows", "reflect": "data.rows"},
        {"name": "rows", "format": "json"},
        {"name": "rows", "type": []},
        {"name": "rows", "path": ["data", "rows"]},
    ],
)
def test_invalid_save_fields_are_rejected_before_tool_execution(tmp_path, save):
    arguments = {
        "action": "call",
        "call": {
            "tool": "mcp",
            "arguments": {"action": "callTool", "server": "demo", "tool": "query", "arguments": {}},
        },
        "save": [save],
    }
    with pytest.raises(ValueError, match="parameterSave save"):
        parse_openai_tool_calls(
            {"tool_calls": [tool_call("parameterSave", arguments)]}, settings_for(tmp_path)
        )


@pytest.mark.parametrize(
    "arguments",
    [
        {"action": "list", "path": "rows"},
        {"action": "inspect", "ref": "param:" + "a" * 32, "save": [{"name": "ignored"}]},
        {"action": "call", "call": {"tool": "mcp", "arguments": {}, "reflect": "rows"}},
    ],
)
def test_inapplicable_or_unknown_wrapper_fields_are_rejected(tmp_path, arguments):
    with pytest.raises(ValueError, match="unsupported fields"):
        parse_openai_tool_calls(
            {"tool_calls": [tool_call("parameterSave", arguments)]}, settings_for(tmp_path)
        )


def test_parameter_schema_documents_paths_and_type_inference():
    function = parameter_tool()["function"]
    assert "Do not prefix structuredContent" in function["description"]
    assert "pythonType" in function["description"]
    properties = function["parameters"]["properties"]
    save = properties["save"]["items"]
    assert save["required"] == ["name"]
    assert save["additionalProperties"] is False
    assert "BEFORE saving" in save["properties"]["path"]["description"]
    assert "AFTER path extraction" in save["properties"]["type"]["description"]
    assert "without changing" in properties["path"]["description"]


def test_mcp_application_paths_and_python_child_load(tmp_path, monkeypatch):
    rows = [{"asking_price": 1000}, {"asking_price": 2000}]
    value = {"content": "DO_NOT_SEND_DATA", "data": {"rows": rows}, "success": True, "error": None}
    monkeypatch.setattr(mcp, "require_server", lambda *args: {})
    monkeypatch.setattr(
        mcp, "run_session", lambda *args, **kwargs: {"result": {"structuredContent": value}}
    )
    settings = settings_for(tmp_path)
    call = {
        "tool": "mcp",
        "arguments": {"action": "callTool", "server": "demo", "tool": "query", "arguments": {}},
    }

    [bad] = parse_openai_tool_calls(
        {
            "tool_calls": [
                tool_call(
                    "parameterSave",
                    {
                        "action": "call",
                        "call": call,
                        "save": [
                            {"name": "wrong", "path": "structuredContent.data.rows", "type": "json"}
                        ],
                    },
                )
            ]
        },
        settings,
    )
    error = execute_tool(bad, settings)
    assert error.startswith("toolError:")
    assert "top-level keys" in error
    assert "DO_NOT_SEND_DATA" not in error
    assert parameterSave.list_parameters(settings.conversation_id) == []

    result = run_parameter(
        settings,
        {
            "action": "call",
            "call": call,
            "save": [
                {"name": "whole"},
                {"name": "json_rows", "path": "data.rows", "type": "json"},
                {"name": "frame", "path": "data.rows", "type": "dataframe"},
            ],
        },
    )
    assert "DO_NOT_SEND_DATA" not in json.dumps(result)
    whole, json_rows, frame = result["saved"]
    with parameterSave.conversation_scope(settings.conversation_id):
        assert parameterSave.load_parameter(whole["ref"]) == value
        assert parameterSave.load_parameter(json_rows["ref"]) == rows
        pd.testing.assert_frame_equal(
            parameterSave.load_parameter(frame["ref"]), pd.DataFrame(rows)
        )
    [python_request] = parse_openai_tool_calls(
        {
            "tool_calls": [
                tool_call(
                    "python",
                    {
                        "code": f"rows = load_parameter({whole['ref']!r}, path='data.rows')\nprint(type(rows).__name__, len(rows))\nsave_parameter('count', len(rows))"
                    },
                )
            ]
        },
        settings,
    )
    output = execute_tool(python_request, settings)
    assert "list 2" in output
    assert "DO_NOT_SEND_DATA" not in output
    entries = parameterSave.list_parameters(settings.conversation_id)
    with parameterSave.conversation_scope(settings.conversation_id):
        assert parameterSave.load_parameter(entries[-1]["ref"]) == 2


def test_large_mcp_is_saved_before_truncation_and_references_are_expanded(tmp_path, monkeypatch):
    rows = [{"rent": index, "note": "DO_NOT_SEND_FULL_DATA"} for index in range(10000)]
    seen = []

    def fake_session(server_config, settings, *, method, params):
        seen.append(params)
        if params["name"] == "summary":
            return {"result": {"structuredContent": {"count": len(params["arguments"]["rows"])}}}
        return {"result": {"structuredContent": {"rows": rows}}}

    monkeypatch.setattr(mcp, "require_server", lambda *args: {})
    monkeypatch.setattr(mcp, "run_session", fake_session)
    settings = settings_for(tmp_path, mcp_mode="auto")
    result = run_parameter(
        settings,
        {
            "action": "call",
            "call": {
                "tool": "mcp",
                "arguments": {
                    "action": "callTool",
                    "server": "demo",
                    "tool": "query",
                    "arguments": {},
                },
            },
            "save": [{"name": "rent_rows", "path": "rows"}],
        },
    )

    assert "DO_NOT_SEND_FULL_DATA" not in json.dumps(result)
    item = result["saved"][0]
    assert item["rowCount"] == 10000
    with parameterSave.conversation_scope(settings.conversation_id):
        assert len(parameterSave.load_parameter(item["ref"])) == 10000
    summary = run_parameter(
        settings,
        {
            "action": "call",
            "call": {
                "tool": "mcp",
                "arguments": {
                    "action": "callTool",
                    "server": "demo",
                    "tool": "summary",
                    "arguments": {"rows": {"$ref": item["ref"]}},
                },
            },
            "save": [{"name": "count", "path": "count"}],
        },
    )
    assert len(seen[1]["arguments"]["rows"]) == 10000
    with parameterSave.conversation_scope(settings.conversation_id):
        assert parameterSave.load_parameter(summary["saved"][0]["ref"]) == 10000


def test_mcp_json_text_and_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp, "require_server", lambda *args: {})
    monkeypatch.setattr(
        mcp,
        "run_session",
        lambda *args, **kwargs: {
            "result": {
                "content": [{"type": "text", "text": '{"rows":[1,2,3]}'}],
            }
        },
    )
    settings = settings_for(tmp_path)
    arguments = {
        "action": "call",
        "call": {
            "tool": "mcp",
            "arguments": {
                "action": "callTool",
                "server": "demo",
                "tool": "query",
                "arguments": {},
            },
        },
        "save": [{"name": "rows", "path": "rows"}],
    }
    run_parameter(settings, arguments)
    monkeypatch.setattr(
        mcp,
        "run_session",
        lambda *args, **kwargs: {
            "result": {
                "isError": True,
                "content": [{"type": "text", "text": "Database unavailable"}],
            }
        },
    )
    [request] = parse_openai_tool_calls(
        {"tool_calls": [tool_call("parameterSave", arguments)]}, settings
    )
    output = execute_tool(request, settings)
    assert output.startswith("toolError:")
    assert "Database unavailable" in output
    assert len(parameterSave.list_parameters(settings.conversation_id)) == 1


def test_python_save_load_and_custom_tool_reference_without_raw_output(tmp_path):
    settings = settings_for(tmp_path)
    generated = run_parameter(
        settings,
        {
            "action": "call",
            "call": {
                "tool": "python",
                "arguments": {
                    "code": "import numpy as np\nvalues = np.arange(10000, dtype=np.int64)"
                },
            },
            "save": [{"name": "values", "path": "values"}],
        },
    )
    ref = generated["saved"][0]["ref"]
    assert generated["saved"][0]["shape"] == [10000]
    assert generated["files"] == []

    createTool.save_tool(
        "sum_values",
        "Sum stored values",
        {"type": "object"},
        "def run(arguments):\n    return int(arguments['values'].sum())",
    )
    summed = run_parameter(
        settings,
        {
            "action": "call",
            "call": {
                "tool": "custom__sum_values",
                "arguments": {"values": {"$ref": ref}},
            },
            "save": [{"name": "sum"}],
        },
    )
    with parameterSave.conversation_scope(settings.conversation_id):
        assert parameterSave.load_parameter(summed["saved"][0]["ref"]) == 49995000

    [request] = parse_openai_tool_calls(
        {
            "tool_calls": [
                tool_call(
                    "python",
                    {
                        "code": f"values = load_parameter({ref!r})\nsave_parameter('subset', values[:3])\nprint(len(values))"
                    },
                )
            ]
        },
        settings,
    )
    text = execute_tool(request, settings)
    assert "savedParameters:" in text
    assert "10000" in text
    assert "bindings.json" not in text
    assert "bridge.json" not in text
    assert len(parameterSave.list_parameters(settings.conversation_id)) == 3


@pytest.mark.parametrize("mode", ["ask", "plan"])
@pytest.mark.parametrize(
    "target,args",
    [
        ("python", {"code": "print(1)"}),
        ("mcp", {"action": "callTool", "server": "demo", "tool": "query"}),
        ("custom__unsafe", {}),
        ("fileEditor", {"action": "write", "path": "demo.py", "content": "1"}),
    ],
)
def test_wrapper_cannot_bypass_modes(mode, target, args):
    settings = make_tool_settings(conversation_mode=mode)
    with pytest.raises(ValueError, match="not available|cannot wrap"):
        parse_openai_tool_calls(
            {
                "tool_calls": [
                    tool_call(
                        "parameterSave",
                        {
                            "action": "call",
                            "call": {"tool": target, "arguments": args},
                            "save": [{"name": "result", "path": "result"}],
                        },
                    )
                ]
            },
            settings,
        )


def test_wrapper_preserves_disabled_tools_and_rejects_stateful_recursion():
    settings = make_tool_settings(mcp_mode="off")
    for target, arguments in [
        ("mcp", {"action": "listServers"}),
        ("parameterSave", {"action": "list"}),
        ("model", {"action": "switch", "model": "x"}),
    ]:
        with pytest.raises(ValueError):
            parse_openai_tool_calls(
                {
                    "tool_calls": [
                        tool_call(
                            "parameterSave",
                            {
                                "action": "call",
                                "call": {"tool": target, "arguments": arguments},
                            },
                        )
                    ]
                },
                settings,
            )


def test_wrapped_python_remains_high_risk_for_ai_review(tmp_path, monkeypatch):
    settings = settings_for(tmp_path, file_editor_approval="aiReview")
    [request] = parse_openai_tool_calls(
        {
            "tool_calls": [
                tool_call(
                    "parameterSave",
                    {
                        "action": "call",
                        "call": {"tool": "python", "arguments": {"code": "values = [1]"}},
                        "save": [{"name": "values", "path": "values"}],
                    },
                )
            ]
        },
        settings,
    )
    seen = []

    def reviewer(inner, **kwargs):
        seen.append(inner.name)
        return graph.ReviewDecision(False, "Denied in test")

    monkeypatch.setattr(graph, "review_tool_request", reviewer)
    state = {"settings": settings, "messages": [{"role": "assistant", "tool_calls": []}]}
    assert graph.is_high_risk_tool_request(request)
    assert not graph.maybe_review_tool_request(request, state).approved
    assert seen == ["python"]


def test_failed_python_does_not_publish_temporary_variables(tmp_path):
    settings = settings_for(tmp_path)
    [request] = parse_openai_tool_calls(
        {
            "tool_calls": [
                tool_call(
                    "python",
                    {
                        "code": "save_parameter('temporary', [1, 2, 3])\nraise ValueError('expected failure')",
                    },
                )
            ]
        },
        settings,
    )
    assert execute_tool(request, settings).startswith("toolError:")
    assert parameterSave.list_parameters(settings.conversation_id) == []


def test_unsupported_values_do_not_partially_publish_and_paths_are_scoped():
    with parameterSave.conversation_scope("valid"):
        with pytest.raises(ValueError, match="unsupported"):
            parameterSave.store_many([("good", [1], "auto"), ("bad", object(), "auto")])
        assert parameterSave.list_parameters() == []
        with pytest.raises(ValueError):
            parameterSave.load_parameter("../../other/index.json")
    with parameterSave.conversation_scope("../escape"):
        with pytest.raises(ValueError, match="conversation id"):
            parameterSave.list_parameters()


@pytest.mark.parametrize("mode", ["ask", "plan"])
def test_read_only_modes_allow_inspection_but_not_deletion(mode):
    settings = make_tool_settings(conversation_mode=mode)
    [request] = parse_openai_tool_calls(
        {"tool_calls": [tool_call("parameterSave", {"action": "list"})]}, settings
    )
    assert request.parameter_request.action == "list"
    with pytest.raises(ValueError, match="Agent mode"):
        parse_openai_tool_calls(
            {
                "tool_calls": [
                    tool_call("parameterSave", {"action": "delete", "ref": "param:" + "a" * 32})
                ]
            },
            settings,
        )


def test_chat_api_persists_refs_and_ignores_client_supplied_scope(tmp_path, monkeypatch):
    original_make = graph.make_tool_settings
    base = settings_for(tmp_path)
    monkeypatch.setattr(
        graph,
        "make_tool_settings",
        lambda **kwargs: replace(original_make(**kwargs), python=base.python),
    )
    calls = []

    def model(messages, *, model=None, tools=None):
        calls.append(messages)
        if len(calls) == 1:
            return {
                "role": "assistant",
                "tool_calls": [
                    tool_call(
                        "parameterSave",
                        {
                            "action": "call",
                            "call": {
                                "tool": "python",
                                "arguments": {"code": "values = list(range(5000))"},
                            },
                            "save": [{"name": "values", "path": "values"}],
                        },
                    )
                ],
            }
        if len(calls) == 3:
            return {
                "role": "assistant",
                "tool_calls": [tool_call("parameterSave", {"action": "list"})],
            }
        return {"role": "assistant", "content": "Stored without raw output"}

    monkeypatch.setattr(graph, "complete_chat_once", model)
    client = TestClient(server.app)
    first = client.post(
        "/api/chat/stream", json={"message": "store data", "conversation_id": "api-chat"}
    )
    frames = [json.loads(line[6:]) for line in first.text.splitlines() if line.startswith("data: ")]
    state = next(frame["state"] for frame in frames if frame["type"] == "assistant")
    stored = session_store.read_conversation("api-chat")
    assert len(stored["parameters"]) == 1
    ref = stored["parameters"][0]["ref"]
    state["conversation_id"] = "forged-scope"
    second = client.post(
        "/api/chat/stream",
        json={"message": "list my variables", "conversation_id": "api-chat", "state": state},
    )
    assert ref in second.text
    assert "forged-scope" not in second.text
    assert not parameterSave.scope_root("forged-scope").exists()
    assert client.get(f"/api/conversations/api-chat/parameters/{ref}").json()["length"] == 5000
    assert client.get(f"/api/conversations/other-chat/parameters/{ref}").status_code == 404
    assert "[0, 1, 2, 3" not in first.text


def test_branch_compression_deletion_and_preview_endpoints():
    with parameterSave.conversation_scope("source"):
        early = parameterSave.store_many([("rows", [{"value": n} for n in range(100)], "auto")])
        late = parameterSave.store_many([("later", "future data", "auto")])
    session_store.save_turn(
        "source",
        user_text="early",
        turn_events=[
            {"type": "parameters_changed", "parameters": early},
            {"type": "assistant", "text": "done"},
        ],
        state={"messages": [{"role": "system", "content": "rules"}]},
    )
    session_store.save_turn(
        "source",
        user_text="later",
        turn_events=[
            {"type": "parameters_changed", "parameters": early + late},
        ],
        state={},
    )
    branch = session_store.branch_conversation("source", 2)
    assert [item["ref"] for item in branch["parameters"]] == [early[0]["ref"]]
    assert branch["state"]["conversation_id"] == branch["id"]
    session_store.compress_conversation(branch["id"])
    client = TestClient(server.app)
    url = f"/api/conversations/{branch['id']}/parameters/{early[0]['ref']}"
    assert "preview" not in client.get(url).json()
    assert '"value": 3' in client.get(url, params={"offset": 3, "limit": 1}).json()["preview"]
    assert client.get(url, params={"limit": 21}).status_code == 422
    session_store.delete_conversation("source")
    with parameterSave.conversation_scope(branch["id"]):
        assert len(parameterSave.load_parameter(early[0]["ref"])) == 100
        with pytest.raises(ValueError):
            parameterSave.load_parameter(late[0]["ref"])
    assert client.delete(url).status_code == 200
    assert client.get(url).status_code == 404
    assert client.get(f"/api/conversations/{branch['id']}/parameters").json()["parameters"] == []


def test_graph_large_mcp_to_normal_fit_plot_keeps_data_out_of_context(tmp_path, monkeypatch):
    rows = [
        {"rent": 1000 + index % 500, "private": "RAW_DATA_MUST_STAY_BACKEND"}
        for index in range(6000)
    ]
    monkeypatch.setattr(mcp, "require_server", lambda *args: {})
    monkeypatch.setattr(
        mcp,
        "run_session",
        lambda *args, **kwargs: {"result": {"structuredContent": {"rows": rows}}},
    )
    base_settings = settings_for(tmp_path)
    original_make = graph.make_tool_settings
    monkeypatch.setattr(
        graph,
        "make_tool_settings",
        lambda **kwargs: replace(original_make(**kwargs), python=base_settings.python),
    )
    calls = []

    def model(messages, *, model=None, tools=None):
        context = json.dumps(messages)
        assert "RAW_DATA_MUST_STAY_BACKEND" not in context
        calls.append(context)
        if len(calls) == 1:
            return {
                "role": "assistant",
                "content": "Fetch the data without displaying it",
                "tool_calls": [
                    tool_call(
                        "parameterSave",
                        {
                            "action": "call",
                            "call": {
                                "tool": "mcp",
                                "arguments": {
                                    "action": "callTool",
                                    "server": "demo",
                                    "tool": "query",
                                    "arguments": {},
                                },
                            },
                            "save": [{"name": "rent_rows", "path": "rows"}],
                        },
                    )
                ],
            }
        if len(calls) == 2:
            saved = json.loads(messages[-1]["content"].split("\n", 1)[1])["saved"][0]
            code = (
                f"rows = load_parameter({saved['ref']!r})\n"
                "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom scipy.stats import norm\n"
                "values = pd.DataFrame(rows)['rent']\nmu, sigma = norm.fit(values)\n"
                "plt.hist(values, bins=30, density=True)\nx = np.linspace(values.min(), values.max(), 100)\n"
                "plt.plot(x, norm.pdf(x, mu, sigma))\nplt.savefig('distribution.png')\n"
                "save_parameter('fit', {'mean': float(mu), 'std': float(sigma)})\nprint('Fit complete')"
            )
            return {
                "role": "assistant",
                "content": "Fit and plot the saved data",
                "tool_calls": [tool_call("python", {"code": code}, "plot")],
            }
        assert "distribution.png" in messages[-1]["content"]
        return {"role": "assistant", "content": "Distribution generated"}

    monkeypatch.setattr(graph, "complete_chat_once", model)
    state = graph.new_chat_state(mcp_mode="auto", python_mode="auto")
    state["conversation_id"] = "graph-chat"
    events = list(graph.stream_turn(state, "Fetch rents and plot their distribution"))
    assert events[-1]["text"] == "Distribution generated"
    assert len(calls) == 3
    assert len(events[-1]["state"]["parameters"]) == 2
    assert len(list((tmp_path / "python_runs").rglob("distribution.png"))) == 1
    assert any(event["type"] == "parameters_changed" for event in events)
