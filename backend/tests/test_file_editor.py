import pytest

from tools import fileEditor
from tools.fileEditor import FileEditRequest
from tools.settings import FileEditorSettings


def settings(tmp_path, *, approval="auto"):
    return FileEditorSettings(mode="auto", root=str(tmp_path), approval=approval)


def test_file_editor_write_read_replace_and_insert(tmp_path):
    config = settings(tmp_path)

    write = fileEditor.execute(
        FileEditRequest(action="write", path="src/app.py", content="print('hello')\n"),
        config,
    )
    assert write["path"] == "src/app.py"
    assert write["applied"] is True

    read = fileEditor.execute(FileEditRequest(action="read", path="src/app.py"), config)
    assert read["content"] == "print('hello')\n"

    replaced = fileEditor.execute(
        FileEditRequest(
            action="replace",
            path="src/app.py",
            old_text="print('hello')",
            new_text="print('hi')",
        ),
        config,
    )
    assert replaced["replacements"] == 1
    assert replaced["applied"] is True

    inserted = fileEditor.execute(
        FileEditRequest(
            action="insertAfter",
            path="src/app.py",
            anchor="print('hi')",
            content="\nprint('done')",
        ),
        config,
    )
    assert inserted["insertions"] == 1
    assert inserted["applied"] is True
    assert (tmp_path / "src" / "app.py").read_text(encoding="utf-8") == "print('hi')\nprint('done')\n"


def test_file_editor_manual_approval_returns_diff_without_writing(tmp_path):
    config = settings(tmp_path, approval="manual")

    result = fileEditor.execute(
        FileEditRequest(action="write", path="proposal.txt", content="hello\n"),
        config,
    )

    assert result["approvalRequired"] is True
    assert result["applied"] is False
    assert "proposal.txt (after)" in result["diff"]
    assert not (tmp_path / "proposal.txt").exists()


def test_file_editor_manual_approval_validates_anchor_before_preview(tmp_path):
    config = settings(tmp_path, approval="manual")
    target = tmp_path / "dup.txt"
    target.write_text("x\nx\n", encoding="utf-8")

    with pytest.raises(ValueError, match="matched 2 times"):
        fileEditor.execute(
            FileEditRequest(action="replace", path="dup.txt", old_text="x", new_text="y"),
            config,
        )

    assert target.read_text(encoding="utf-8") == "x\nx\n"


def test_file_editor_readonly_approval_does_not_write(tmp_path):
    config = settings(tmp_path, approval="readOnly")

    result = fileEditor.execute(
        FileEditRequest(action="write", path="blocked.txt", content="x\n"),
        config,
    )

    assert result["approvalRequired"] is True
    assert result["applied"] is False
    assert "readOnly" in result["reason"]
    assert not (tmp_path / "blocked.txt").exists()


def test_file_editor_requires_unique_replace_anchor(tmp_path):
    config = settings(tmp_path)
    target = tmp_path / "dup.txt"
    target.write_text("x\nx\n", encoding="utf-8")

    with pytest.raises(ValueError, match="matched 2 times"):
        fileEditor.execute(
            FileEditRequest(action="replace", path="dup.txt", old_text="x", new_text="y"),
            config,
        )


def test_file_editor_list_hides_protected_files(tmp_path):
    config = settings(tmp_path)
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=x", encoding="utf-8")

    result = fileEditor.execute(FileEditRequest(action="list", path=".", pattern="*"), config)

    assert "a.py" in result["files"]
    assert ".env" not in result["files"]


def test_file_editor_blocks_outside_root(tmp_path):
    config = settings(tmp_path)

    with pytest.raises(ValueError, match="inside the configured file editor root"):
        fileEditor.execute(FileEditRequest(action="read", path="../outside.txt"), config)


def test_file_editor_blocks_protected_env_write(tmp_path):
    config = settings(tmp_path)

    with pytest.raises(ValueError, match="protected file"):
        fileEditor.execute(FileEditRequest(action="write", path=".env", content="x"), config)
