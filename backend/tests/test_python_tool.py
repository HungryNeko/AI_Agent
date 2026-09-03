import pytest

from tools import python
from tools.settings import PythonSettings


def test_python_run_executes_basic_analysis(tmp_path):
    result = python.run(
        "import statistics\nprint(statistics.mean([2, 4, 6]))",
        PythonSettings(mode="auto", artifact_root=str(tmp_path)),
    )

    assert result["return_code"] == 0
    assert result["stdout"].strip() == "4"
    assert result["artifact_dir"].startswith(str(tmp_path))


def test_python_run_can_create_plot_artifact(tmp_path):
    result = python.run(
        "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3], [1, 4, 9])\nplt.savefig('chart.png')\nprint('saved')",
        PythonSettings(mode="auto", artifact_root=str(tmp_path)),
    )

    assert result["return_code"] == 0
    assert result["stdout"].strip() == "saved"
    assert any(path.endswith("chart.png") for path in result["files"])


def test_python_run_allows_system_import(tmp_path):
    result = python.run(
        "import os\nprint(os.path.basename(os.getcwd()))",
        PythonSettings(mode="auto", artifact_root=str(tmp_path)),
    )

    assert result["return_code"] == 0
    assert result["stdout"].strip().startswith("run_")


def test_python_safety_blocks_delete_method(tmp_path):
    with pytest.raises(ValueError, match="destructive method `unlink`"):
        python.run("thing.unlink()", PythonSettings(mode="auto", artifact_root=str(tmp_path)))


def test_python_safety_allows_normal_library_version_metadata(tmp_path):
    result = python.run(
        "import numpy as np\nprint(bool(np.__version__))",
        PythonSettings(mode="auto", artifact_root=str(tmp_path)),
    )

    assert result["return_code"] == 0
    assert result["stdout"].strip() == "True"


def test_python_run_allows_dunder_introspection(tmp_path):
    result = python.run(
        "print((1).__class__.__name__)",
        PythonSettings(mode="auto", artifact_root=str(tmp_path)),
    )

    assert result["return_code"] == 0
    assert result["stdout"].strip() == "int"


def test_python_run_allows_local_file_read(tmp_path):
    outside_file = tmp_path / "outside.csv"
    outside_file.write_text("x\n1\n", encoding="utf-8")

    result = python.run(
        f"print(open({str(outside_file)!r}, encoding='utf-8').read().strip())",
        PythonSettings(mode="auto", artifact_root=str(tmp_path / "runs")),
    )

    assert result["return_code"] == 0
    assert result["stdout"].strip() == "x\n1"


def test_python_runtime_blocks_write_outside_artifact(tmp_path):
    outside_file = tmp_path / "outside.txt"

    result = python.run(
        f"open({str(outside_file)!r}, 'w', encoding='utf-8').write('x')",
        PythonSettings(mode="auto", artifact_root=str(tmp_path / "runs")),
    )

    assert result["return_code"] != 0
    assert "file writes are only allowed" in result["stderr"]
    assert not outside_file.exists()
