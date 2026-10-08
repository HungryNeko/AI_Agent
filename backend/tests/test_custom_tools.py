from tools import createTool, plan
from tools.createTool import CreateToolRequest
from tools.plan import PlanRequest


def test_custom_tool_save_read_and_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(createTool, "CUSTOM_TOOL_ROOT", tmp_path / "custom_tools")
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }

    result = createTool.execute(
        CreateToolRequest(
            action="save",
            name="Double Value",
            description="Double an integer",
            parameters=parameters,
            code="def run(arguments):\n    return arguments['value'] * 2",
        )
    )
    saved = createTool.read_tool("double_value", include_code=True)
    schemas = createTool.openai_schemas()

    assert '"status": "saved"' in result
    assert saved["function_name"] == "custom__double_value"
    assert schemas[0]["function"]["name"] == "custom__double_value"
    assert schemas[0]["function"]["parameters"] == parameters


def test_plan_tool_uses_session_payload_and_exports_only_explicitly(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, "PLAN_ROOT", tmp_path / "plans")

    result = plan.execute(PlanRequest(action="finalize", name="Release Plan", content="# Release\n1. Test"))

    assert "# Release" in result
    assert '"status": "ready"' in result
    assert not (tmp_path / "plans").exists()

    saved = plan.save_copy("Release Plan", "# Release\n1. Test")

    assert saved["status"] == "saved"
    assert (tmp_path / "plans" / "release-plan.md").is_file()
