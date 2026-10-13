from tools import python as python_tool
from tools.executor import format_python_result
from tools.settings import PythonSettings


def settings(tmp_path):
    return PythonSettings(mode="auto", artifact_root=str(tmp_path / "runs"), timeout_seconds=60.0)


def test_workspace_persists_between_calls_while_artifact_dirs_stay_fresh(tmp_path):
    cfg = settings(tmp_path)
    first = python_tool.run(
        "import json, os\njson.dump({'step': 1}, open(os.path.join(WORKSPACE, 'state.json'), 'w'))\nprint('saved')",
        cfg,
        workspace_key="conv1",
    )
    second = python_tool.run(
        "import json, os\nstate = json.load(open(os.path.join(WORKSPACE, 'state.json')))\nstate['step'] += 1\n"
        "json.dump(state, open(os.path.join(WORKSPACE, 'state.json'), 'w'))\nprint(state)",
        cfg,
        workspace_key="conv1",
    )
    assert first["return_code"] == 0 and second["return_code"] == 0, second["stderr"]
    assert "'step': 2" in second["stdout"]
    assert first["artifact_dir"] != second["artifact_dir"]  # every call still gets its own artifact directory
    assert second["workspace_dir"] == first["workspace_dir"] and second["workspace_files"] == ["state.json"]


def test_workspaces_are_isolated_per_conversation_and_sub_agent(tmp_path):
    cfg = settings(tmp_path)
    python_tool.run("open(WORKSPACE + '/mine.txt', 'w').write('x')", cfg, workspace_key="conv1")
    other = python_tool.run("import os\nprint(os.listdir(WORKSPACE))", cfg, workspace_key="conv1-agent7")
    assert other["stdout"].strip() == "[]"
    assert python_tool.run("print(1)", cfg, workspace_key="../escape")["workspace_dir"].startswith(str(tmp_path))


def test_writes_elsewhere_are_still_blocked_and_no_key_falls_back_to_the_run_dir(tmp_path):
    cfg = settings(tmp_path)
    blocked = python_tool.run(f"open({str(tmp_path / 'outside.txt')!r}, 'w').write('x')", cfg, workspace_key="conv1")
    assert blocked["return_code"] != 0 and "WORKSPACE" in blocked["stderr"]
    assert not (tmp_path / "outside.txt").exists()
    plain = python_tool.run("print(WORKSPACE)", cfg)
    assert plain["stdout"].strip() == plain["artifact_dir"] and "workspace_dir" not in plain


def test_result_text_tells_the_model_where_persistent_files_are(tmp_path):
    cfg = settings(tmp_path)
    result = python_tool.run("open(WORKSPACE + '/bg_state.json', 'w').write('{}')", cfg, workspace_key="conv1")
    text = format_python_result(result)
    assert "WORKSPACE" in text and "workspaceFiles: bg_state.json" in text
    assert "workspaceFiles" not in format_python_result(python_tool.run("print(1)", cfg))


def test_deleting_a_conversation_removes_its_and_its_agents_workspaces(tmp_path):
    cfg = settings(tmp_path)
    for key in ("conv1", "conv1-agent7", "conv10"):
        python_tool.run("open(WORKSPACE + '/x.txt', 'w').write('x')", cfg, workspace_key=key)
    python_tool.delete_workspaces(tmp_path / "runs", "conv1")
    remaining = sorted(path.name for path in (tmp_path / "runs" / "_workspaces").iterdir())
    assert remaining == ["conv10"]  # a different conversation whose id merely starts the same survives
