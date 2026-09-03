"""Run Python snippets for analysis and plotting."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tools.settings import PythonSettings

DESTRUCTIVE_METHODS = {"remove", "unlink", "rmdir", "rmtree", "removedirs"}
DESTRUCTIVE_SHELL_RE = re.compile(
    r"(^|[;&|\s])(rm|del|erase|rmdir|remove-item|rd)($|[;&|\s])",
    re.IGNORECASE,
)


class SafetyError(ValueError):
    pass


def run(code: str, settings: PythonSettings) -> dict[str, Any]:
    clean_code = code.strip()
    if not clean_code:
        raise ValueError("python code must not be empty.")
    check_code_safety(clean_code)

    run_dir = make_run_dir(settings)
    script_path = run_dir / "snippet.py"
    script_path.write_text(build_wrapped_code(clean_code, run_dir), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-I", str(script_path)],
        cwd=run_dir,
        env=build_child_env(run_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=settings.timeout_seconds,
        check=False,
    )

    return {
        "return_code": completed.returncode,
        "stdout": trim_output(completed.stdout, settings.max_output_chars),
        "stderr": trim_output(completed.stderr, settings.max_output_chars),
        "artifact_dir": str(run_dir),
        "files": list_artifacts(run_dir, settings),
    }


def check_code_safety(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SafetyError(f"syntax error on line {exc.lineno or 1}: {exc.msg}") from exc

    errors: list[str] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 1)
        if isinstance(node, ast.Attribute):
            if node.attr in DESTRUCTIVE_METHODS:
                errors.append(f"line {line}: destructive method `{node.attr}` is not allowed")
        elif isinstance(node, ast.Call):
            command = literal_command(node)
            if command and DESTRUCTIVE_SHELL_RE.search(command):
                errors.append(f"line {line}: destructive shell command is not allowed")
    if errors:
        raise SafetyError("python code blocked by safety check: " + "; ".join(errors))


def literal_command(node: ast.Call) -> str:
    if not node.args:
        return ""
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    if isinstance(first, (ast.List, ast.Tuple)):
        parts = [
            item.value
            for item in first.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        ]
        return " ".join(parts)
    return ""


def build_child_env(run_dir: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for key in ("SYSTEMROOT", "WINDIR", "PATH"):
        value = __import__("os").environ.get(key)
        if value:
            env[key] = value
    env["TEMP"] = str(run_dir)
    env["TMP"] = str(run_dir)
    env["MPLCONFIGDIR"] = str(run_dir / "matplotlib")
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"
    env["AI_AGENT_PYTHON_ARTIFACT_DIR"] = str(run_dir)
    return env


def build_wrapped_code(user_code: str, run_dir: Path) -> str:
    prelude = f"""
import os
import sys

_ARTIFACT_DIR = os.path.realpath({str(run_dir)!r})
os.environ["AI_AGENT_PYTHON_ARTIFACT_DIR"] = _ARTIFACT_DIR
os.environ.setdefault("MPLBACKEND", "Agg")


def _inside(path, parent):
    try:
        path = os.path.realpath(path)
        parent = os.path.realpath(parent)
        return path == parent or path.startswith(parent + os.sep)
    except Exception:
        return False


def _audit(event, args):
    if event in {{"os.remove", "os.rmdir", "shutil.rmtree"}}:
        target = args[0] if args else ""
        if not _inside(os.fspath(target), _ARTIFACT_DIR):
            raise RuntimeError(f"blocked runtime event outside artifact dir: {{event}}")
    if event == "open" and args:
        target = args[0]
        mode = str(args[1]) if len(args) > 1 else "r"
        if isinstance(target, int):
            return
        target_text = os.fspath(target)
        is_write = any(flag in mode for flag in ("w", "a", "+", "x"))
        if is_write and not _inside(target_text, _ARTIFACT_DIR):
            raise RuntimeError("file writes are only allowed inside AI_AGENT_PYTHON_ARTIFACT_DIR")


sys.addaudithook(_audit)
os.chdir(_ARTIFACT_DIR)
""".lstrip()
    return prelude + "\n" + user_code + "\n"


def make_run_dir(settings: PythonSettings) -> Path:
    base_dir = Path(settings.artifact_root).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000:06d}"
    run_dir = base_dir / f"run_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def trim_output(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def list_artifacts(run_dir: Path, settings: PythonSettings) -> list[str]:
    files: list[str] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "snippet.py":
            continue
        if "matplotlib" in path.relative_to(run_dir).parts:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size <= settings.max_artifact_bytes:
            files.append(str(path))
        if len(files) >= settings.max_artifacts:
            break
    return files
