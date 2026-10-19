"""Run Python snippets for analysis and plotting."""

from __future__ import annotations

import ast
import contextlib
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tools import parameterSave, procs
from tools.bridge import BRIDGE_HELPER_SOURCE
from tools.settings import PythonSettings

DESTRUCTIVE_METHODS = {"remove", "unlink", "rmdir", "rmtree", "removedirs"}
DESTRUCTIVE_SHELL_RE = re.compile(
    r"(^|[;&|\s])(rm|del|erase|rmdir|remove-item|rd)($|[;&|\s])",
    re.IGNORECASE,
)


class SafetyError(ValueError):
    pass


def run(
    code: str,
    settings: PythonSettings,
    *,
    bindings: Any = None,
    capture_result: bool = False,
    workspace_key: str = "",
    bridge: Any = None,
) -> dict[str, Any]:
    clean_code = code.strip()
    if not clean_code:
        raise ValueError("python code must not be empty.")
    check_code_safety(clean_code)

    run_dir = make_run_dir(settings)
    workspace = make_workspace(settings, workspace_key) if workspace_key else None
    write_helper_modules(run_dir)
    parameterSave.prepare_python_bridge(run_dir, bindings)
    script_path = run_dir / "snippet.py"
    script_path.write_text(build_wrapped_code(clean_code, run_dir, workspace), encoding="utf-8")

    # Tracked so that cancelling the owning sub-agent also kills this process tree. When a bridge serves
    # MCP/model calls for the code, the time spent waiting on them does not count against the timeout.
    bridge_dir = run_dir / "_bridge" if bridge is not None else None
    with bridge.serve(bridge_dir) if bridge is not None else contextlib.nullcontext():
        completed = procs.run_tracked(
            [sys.executable, "-I", str(script_path)],
            cwd=run_dir,
            env=build_child_env(run_dir, workspace, bridge_dir),
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=settings.timeout_seconds,
            grace=bridge.busy_seconds if bridge is not None else None,
        )

    result = {
        "return_code": completed.returncode,
        "stdout": trim_output(completed.stdout, settings.max_output_chars),
        "stderr": trim_output(completed.stderr, settings.max_output_chars),
        "artifact_dir": str(run_dir),
        "files": list_artifacts(run_dir, settings),
    }
    if workspace is not None:
        result["workspace_dir"] = str(workspace)
        result["workspace_files"] = list_workspace(workspace)
    if completed.returncode == 0:
        exports = parameterSave.collect_python_exports(run_dir)
        if exports:
            result["parameters"] = exports
        if capture_result:
            path = run_dir / "_parameters" / "result.json"
            if path.exists():
                import json

                result["result"] = parameterSave.decode(json.loads(path.read_text(encoding="utf-8")))
    return result


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


def workspace_name(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("._")[:80] or "default"


def delete_workspaces(artifact_root: Path | str, conversation_id: str) -> None:
    """Remove a conversation's workspace and those of its sub-agents (named `<conversation>-<agent>`)."""

    import shutil

    root = Path(artifact_root).resolve() / "_workspaces"
    name = workspace_name(conversation_id)
    if not root.is_dir() or name == "default":
        return
    for path in root.iterdir():
        if path.is_dir() and (path.name == name or path.name.startswith(name + "-")):
            shutil.rmtree(path, ignore_errors=True)


def make_workspace(settings: PythonSettings, key: str) -> Path:
    """Persistent scratch folder for one conversation (or one sub-agent): survives between python calls."""

    path = Path(settings.artifact_root).resolve() / "_workspaces" / workspace_name(key)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_workspace(workspace: Path, limit: int = 30) -> list[str]:
    files: list[str] = []
    for path in sorted(workspace.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            files.append(str(path.relative_to(workspace)).replace("\\", "/"))
        if len(files) >= limit:
            break
    return files


def build_child_env(run_dir: Path, workspace: Path | None = None, bridge_dir: Path | None = None) -> dict[str, str]:
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
    if workspace is not None:
        env["AI_AGENT_WORKSPACE"] = str(workspace)
    if bridge_dir is not None:
        env["AI_AGENT_BRIDGE_DIR"] = str(bridge_dir)
    env["AI_AGENT_PARAMETER_BRIDGE"] = str(run_dir / "_parameters" / "bridge.json")
    return env


def build_wrapped_code(user_code: str, run_dir: Path, workspace: Path | None = None) -> str:
    workspace_text = str(workspace) if workspace is not None else ''
    prelude = f"""
import os
import sys

_ARTIFACT_DIR = os.path.realpath({str(run_dir)!r})
_WS = {workspace_text!r}
_WORKSPACE = os.path.realpath(_WS) if _WS else ""
WORKSPACE = _WORKSPACE or _ARTIFACT_DIR  # persists across python calls when a workspace exists
os.environ["AI_AGENT_PYTHON_ARTIFACT_DIR"] = _ARTIFACT_DIR
os.environ.setdefault("MPLBACKEND", "Agg")
if _ARTIFACT_DIR not in sys.path:
    sys.path.insert(0, _ARTIFACT_DIR)


def _inside(path, parent):
    try:
        path = os.path.realpath(path)
        parent = os.path.realpath(parent)
        return path == parent or path.startswith(parent + os.sep)
    except Exception:
        return False


def _writable(path):
    return _inside(path, _ARTIFACT_DIR) or (bool(_WORKSPACE) and _inside(path, _WORKSPACE))


def _audit(event, args):
    if event in {{"os.remove", "os.rmdir", "shutil.rmtree"}}:
        target = args[0] if args else ""
        if not _writable(os.fspath(target)):
            raise RuntimeError(f"blocked runtime event outside artifact dir: {{event}}")
    if event == "open" and args:
        target = args[0]
        mode = str(args[1]) if len(args) > 1 else "r"
        if isinstance(target, int):
            return
        target_text = os.fspath(target)
        protected_names = {{".env", ".env.local", ".env.production", "api_configs.local.json", "servers.local.json"}}
        if os.path.basename(os.path.realpath(target_text)).lower() in protected_names:
            raise RuntimeError("protected secret file cannot be opened")
        is_write = any(flag in mode for flag in ("w", "a", "+", "x"))
        if is_write and not _writable(target_text):
            raise RuntimeError("file writes are only allowed inside AI_AGENT_PYTHON_ARTIFACT_DIR or WORKSPACE")


sys.addaudithook(_audit)
os.chdir(_ARTIFACT_DIR)
""".lstrip()
    return (
        prelude
        + "\nfrom ai_agent_parameters import load_parameter, save_parameter\n"
        + "from ai_agent_bridge import call_mcp, list_mcp_servers, list_mcp_tools, ask_llm, run_agent\n"
        + user_code
        + "\n"
    )


def make_run_dir(settings: PythonSettings) -> Path:
    base_dir = Path(settings.artifact_root).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000:06d}"
    run_dir = base_dir / f"run_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_helper_modules(run_dir: Path) -> None:
    (run_dir / "ai_agent_maps.py").write_text(MAP_HELPER_SOURCE, encoding="utf-8")
    (run_dir / "ai_agent_bridge.py").write_text(BRIDGE_HELPER_SOURCE, encoding="utf-8")


def trim_output(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def list_artifacts(run_dir: Path, settings: PythonSettings) -> list[str]:
    files: list[str] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in {"snippet.py", "ai_agent_maps.py", "ai_agent_parameters.py", "ai_agent_bridge.py"}:
            continue
        if "matplotlib" in path.relative_to(run_dir).parts:
            continue
        if any(part in path.relative_to(run_dir).parts for part in {"_parameters", "_bridge", "__pycache__"}):
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


MAP_HELPER_SOURCE = r'''
"""Small map helpers available inside the AI Agent Python artifact directory."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable, Mapping, Any


def write_osm_scatter(
    points: Iterable[Mapping[str, Any]],
    filename: str = "osm_scatter.html",
    *,
    title: str = "OpenStreetMap Scatter",
    zoom: int = 12,
) -> str:
    """Write a lightweight Leaflet/OpenStreetMap scatter map and return its path.

    Each point may contain lat/lon or latitude/longitude plus optional label,
    popup, color, radius, and opacity.
    """

    clean_points = []
    for item in points:
        lat = item.get("lat", item.get("latitude"))
        lon = item.get("lon", item.get("lng", item.get("longitude")))
        if lat is None or lon is None:
            continue
        clean_points.append(
            {
                "lat": float(lat),
                "lon": float(lon),
                "label": str(item.get("label", "")),
                "popup": str(item.get("popup", item.get("label", ""))),
                "color": str(item.get("color", "#2563eb")),
                "radius": float(item.get("radius", 7)),
                "opacity": float(item.get("opacity", 0.85)),
            }
        )
    if not clean_points:
        raise ValueError("write_osm_scatter requires at least one point with lat/lon.")

    center_lat = sum(point["lat"] for point in clean_points) / len(clean_points)
    center_lon = sum(point["lon"] for point in clean_points) / len(clean_points)
    safe_title = html.escape(title)
    data = json.dumps(clean_points, ensure_ascii=False)
    output = Path(filename)
    if output.suffix.lower() != ".html":
        output = output.with_suffix(".html")
    output.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body, #map {{ height: 100%; margin: 0; }}
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    #map {{ min-height: 520px; }}
    .title {{ position: absolute; z-index: 500; top: 12px; left: 12px; background: rgba(255,255,255,.92); padding: 8px 10px; border-radius: 6px; box-shadow: 0 1px 8px rgba(0,0,0,.16); font-weight: 700; }}
  </style>
</head>
<body>
  <div class="title">{safe_title}</div>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const points = {data};
    const map = L.map("map").setView([{center_lat:.6f}, {center_lon:.6f}], {int(zoom)});
    L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors"
    }}).addTo(map);
    const bounds = [];
    for (const point of points) {{
      bounds.push([point.lat, point.lon]);
      L.circleMarker([point.lat, point.lon], {{
        radius: point.radius,
        color: point.color,
        fillColor: point.color,
        fillOpacity: point.opacity,
        weight: 2
      }}).addTo(map).bindPopup(point.popup || point.label || `${{point.lat}}, ${{point.lon}}`);
    }}
    if (bounds.length > 1) map.fitBounds(bounds, {{ padding: [30, 30] }});
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return str(output)
'''
