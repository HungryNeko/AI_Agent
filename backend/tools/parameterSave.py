"""Conversation-scoped values and a tool-result wrapper. No pickle or executable codecs."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

PARAMETER_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "parameters"
_CONVERSATION: ContextVar[str] = ContextVar("parameter_conversation", default="")
_LOCK = RLock()
REF_RE = re.compile(r"param:([a-f0-9]{32})\Z")
DATA_TOOLS = {"mcp", "python", "webSearch", "rag", "curl", "fileReader", "fileEditor"}


@dataclass(frozen=True)
class ParameterRequest:
    action: str
    call: dict[str, Any] = field(default_factory=dict)
    save: tuple[dict[str, str], ...] = ()
    ref: str = ""
    path: str = ""
    offset: int = 0
    limit: int = 0


@contextmanager
def conversation_scope(conversation_id: str):
    token = _CONVERSATION.set(conversation_id)
    try:
        yield
    finally:
        _CONVERSATION.reset(token)


def scope_root(conversation_id: str | None = None) -> Path:
    name = _CONVERSATION.get() if conversation_id is None else conversation_id
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", name):
        raise ValueError("parameterSave requires a valid conversation id")
    root = PARAMETER_ROOT.resolve() / name
    if root.is_symlink():
        raise ValueError("parameter directory cannot be a symlink")
    return root


def read_index(conversation_id: str | None = None) -> dict[str, Any]:
    path = scope_root(conversation_id) / "index.json"
    return (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"revision": 0, "values": {}}
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def encode(value: Any) -> dict[str, Any]:
    if value is None or isinstance(value, (bool, int, str)):
        return {"kind": "json", "value": value}
    if isinstance(value, float):
        return (
            {"kind": "float", "value": str(value)}
            if not math.isfinite(value)
            else {"kind": "json", "value": value}
        )
    if isinstance(value, datetime):
        return {"kind": "datetime", "value": value.isoformat()}
    if isinstance(value, (list, tuple)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "value": [encode(item) for item in value],
        }
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("saved dictionaries require string keys")
        return {"kind": "dict", "value": {key: encode(item) for key, item in value.items()}}
    module = type(value).__module__.split(".")[0]
    if module == "numpy":
        import numpy as np

        if isinstance(value, np.ndarray):
            if value.dtype.hasobject or value.dtype.fields:
                raise ValueError("object/structured NumPy arrays are not supported; use a list")
            import base64

            return {
                "kind": "ndarray",
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "value": base64.b64encode(value.tobytes()).decode("ascii"),
            }
        if isinstance(value, np.generic):
            return encode(value.item())
    if module == "pandas":
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            return {"kind": "dataframe", "value": value.to_json(orient="table", date_format="iso")}
    raise ValueError(f"unsupported saved data type: {type(value).__name__}")


def decode(payload: dict[str, Any]) -> Any:
    kind, value = payload["kind"], payload["value"]
    if kind == "json":
        return value
    if kind == "float":
        return float(value)
    if kind == "datetime":
        return datetime.fromisoformat(value)
    if kind in {"list", "tuple"}:
        items = [decode(item) for item in value]
        return tuple(items) if kind == "tuple" else items
    if kind == "dict":
        return {key: decode(item) for key, item in value.items()}
    if kind == "ndarray":
        import base64
        import numpy as np

        dtype = np.dtype(payload["dtype"])
        if dtype.hasobject:
            raise ValueError("object-dtype NumPy arrays cannot be loaded")
        return (
            np.frombuffer(base64.b64decode(value, validate=True), dtype=dtype)
            .reshape(payload["shape"])
            .copy()
        )
    if kind == "dataframe":
        import pandas as pd

        return pd.read_json(StringIO(value), orient="table")
    raise ValueError(f"unsupported parameter codec: {kind}")


def describe(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"type": type(value).__name__, "pythonType": type(value).__name__}
    if isinstance(value, (list, tuple, dict, str)):
        result["length"] = len(value)
    if isinstance(value, dict):
        result["keys"] = [key[:120] for key in list(value)[:40]]
    rows = value if isinstance(value, list) else []
    if rows and isinstance(rows[0], dict):
        result.update(
            type="table", rowCount=len(rows), columns=[key[:120] for key in list(rows[0])[:40]]
        )
    if type(value).__module__.split(".")[0] == "numpy" and hasattr(value, "shape"):
        result.update(shape=list(value.shape), dtype=str(value.dtype))
    if type(value).__module__.split(".")[0] == "pandas" and hasattr(value, "columns"):
        result.update(
            rowCount=len(value),
            columns=[str(item)[:120] for item in value.columns[:40]],
            dtypes=[str(item) for item in value.dtypes[:40]],
        )
    return result


def normalize_type(value: Any, data_type: str) -> Any:
    if data_type == "auto":
        return value
    if data_type == "json":
        return to_wire(value)
    if data_type == "ndarray":
        import numpy as np

        return np.asarray(value)
    if data_type in {"table", "dataframe"}:
        import pandas as pd

        return value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    raise ValueError("type must be auto, json, ndarray, table, or dataframe")


def store_many(
    items: list[tuple[str, Any, str]], *, refs: list[str] | None = None
) -> list[dict[str, Any]]:
    prepared = []
    for index, (name, value, data_type) in enumerate(items):
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise ValueError("parameter name must contain 1 to 120 characters")
        value = normalize_type(value, data_type)
        ref = refs[index] if refs else f"param:{uuid4().hex}"
        prepared.append((ref, name.strip(), encode(value), describe(value)))
    with _LOCK:
        root, manifest = scope_root(), read_index()
        saved = []
        for ref, name, payload, metadata in prepared:
            key = ref_key(ref)
            if ref in manifest["values"]:
                raise ValueError("parameter references are immutable")
            versions = manifest.setdefault("versions", {})
            version = 1 + versions.get(name, 0)
            versions[name] = version
            write_json(root / f"{key}.json", payload)
            item = {"ref": ref, "name": name, "version": version, **metadata}
            manifest["values"][ref] = item
            saved.append(item)
        if saved:
            manifest["revision"] += 1
            write_json(root / "index.json", manifest)
        return saved


def ref_key(ref: str) -> str:
    match = REF_RE.fullmatch(ref)
    if not match:
        raise ValueError("invalid parameter reference")
    return match[1]


def list_parameters(conversation_id: str | None = None) -> list[dict[str, Any]]:
    return list(read_index(conversation_id)["values"].values())


def load_parameter(ref: str, path: str = "") -> Any:
    """Load the stored Python value, or a dot-path child, without converting its type."""
    if os.environ.get("AI_AGENT_PARAMETER_BRIDGE"):
        bridge = json.loads(
            Path(os.environ["AI_AGENT_PARAMETER_BRIDGE"]).read_text(encoding="utf-8")
        )
        source = bridge["inputs"].get(ref)
        if not source:
            raise ValueError("parameter not available in this conversation")
        value = decode(json.loads(Path(source).read_text(encoding="utf-8")))
    else:
        with _LOCK:
            if ref not in read_index()["values"]:
                raise ValueError("parameter not available in this conversation")
            value = decode(
                json.loads((scope_root() / f"{ref_key(ref)}.json").read_text(encoding="utf-8"))
            )
    return select_path(value, path)


def save_parameter(name: str, value: Any, data_type: str = "auto") -> str:
    bridge_path = os.environ.get("AI_AGENT_PARAMETER_BRIDGE")
    if not bridge_path:
        return store_many([(name, value, data_type)])[0]["ref"]
    value = normalize_type(value, data_type)
    ref = f"param:{uuid4().hex}"
    root = Path(bridge_path).parent
    path = root / f"{ref_key(ref)}.json"
    write_json(path, encode(value))
    bridge = json.loads(Path(bridge_path).read_text(encoding="utf-8"))
    bridge["inputs"][ref] = str(path)
    bridge["exports"].append({"ref": ref, "name": name})
    write_json(Path(bridge_path), bridge)
    return ref


def resolve_references(value: Any) -> Any:
    if isinstance(value, dict):
        if "$ref" in value:
            if set(value) - {"$ref", "path"}:
                raise ValueError("a parameter reference only accepts $ref and optional path")
            return load_parameter(value["$ref"], path=value.get("path", ""))
        return {key: resolve_references(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_references(item) for item in value]
    return value


def select_path(value: Any, path: str) -> Any:
    """Select dot-separated keys/list indexes; report structure, never values, on failure."""
    if not isinstance(path, str):
        raise ValueError("parameter path must be a string")
    if not path or path == "$":
        return value
    root = value
    clean = path.removeprefix("$.")
    traversed = []
    for key in clean.split("."):
        try:
            if isinstance(value, (list, tuple)):
                child = value[int(key)]
            else:
                child = value[key]
        except (KeyError, IndexError, TypeError, ValueError):
            details = []
            if isinstance(root, dict):
                details.append(
                    "top-level keys: " + json.dumps(describe(root)["keys"], ensure_ascii=False)
                )
            else:
                details.append("root structure: " + json.dumps(describe(root), ensure_ascii=False))
            location = ".".join(traversed) or "$"
            details.append(f"failed at {location[:240]!r}, segment {key[:120]!r}")
            details.append("current structure: " + json.dumps(describe(value), ensure_ascii=False))
            raise ValueError(f"unknown path {path[:240]!r}; " + "; ".join(details)) from None
        value = child
        traversed.append(key)
    return value


def inspect_parameter(
    ref: str, *, path: str = "", offset: int = 0, limit: int = 0
) -> dict[str, Any]:
    """Describe a selected child (inferred type/columns); never mutate or convert the stored value."""
    value = load_parameter(ref, path=path)
    result = {"ref": ref, **describe(value)}
    if path:
        result["path"] = path
    if limit:
        if hasattr(value, "iloc"):
            sample = value.iloc[offset : offset + limit].to_dict(orient="records")
        elif isinstance(value, dict):
            sample = dict(list(value.items())[offset : offset + limit])
        elif isinstance(value, (list, tuple, str)) or hasattr(value, "shape"):
            sample = value[offset : offset + limit]
        else:
            sample = value
        text = json.dumps(to_wire(sample), ensure_ascii=False, default=str)
        result["preview"] = text[:4000]
        result["previewTruncated"] = len(text) > 4000
    return result


def to_wire(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_wire(item) for item in value]
    if hasattr(value, "to_dict") and hasattr(value, "columns"):
        return to_wire(value.to_dict(orient="records"))
    if hasattr(value, "tolist"):
        return to_wire(value.tolist())
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def delete_parameter(ref: str) -> dict[str, str]:
    with _LOCK:
        manifest = read_index()
        if ref not in manifest["values"]:
            raise ValueError("parameter not available in this conversation")
        del manifest["values"][ref]
        manifest["revision"] += 1
        write_json(scope_root() / "index.json", manifest)
        (scope_root() / f"{ref_key(ref)}.json").unlink(missing_ok=True)
    return {"status": "deleted", "ref": ref}


def clone_parameters(source_id: str, target_id: str, entries: list[dict[str, Any]]) -> None:
    with _LOCK:
        current = read_index(source_id)["values"]
        target = scope_root(target_id)
        target.mkdir(parents=True, exist_ok=True)
        selected = {}
        for item in entries:
            ref = item["ref"]
            if ref in current:
                name = f"{ref_key(ref)}.json"
                shutil.copyfile(scope_root(source_id) / name, target / name)
                selected[ref] = current[ref]
        versions = {}
        for item in selected.values():
            versions[item["name"]] = max(versions.get(item["name"], 0), item["version"])
        write_json(target / "index.json", {"revision": 1, "values": selected, "versions": versions})


def delete_all(conversation_id: str) -> None:
    with _LOCK:
        root = scope_root(conversation_id)
        if root.exists():
            shutil.rmtree(root)


def prepare_python_bridge(run_dir: Path, bindings: Any = None) -> None:
    root = run_dir / "_parameters"
    inputs = {}
    if _CONVERSATION.get():
        inputs = {
            item["ref"]: str(scope_root() / f"{ref_key(item['ref'])}.json")
            for item in list_parameters()
        }
    write_json(root / "bridge.json", {"inputs": inputs, "exports": []})
    if bindings is not None:
        write_json(root / "bindings.json", encode(bindings))
    (run_dir / "ai_agent_parameters.py").write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )


def collect_python_exports(run_dir: Path) -> list[dict[str, Any]]:
    root = run_dir / "_parameters"
    bridge = json.loads((root / "bridge.json").read_text(encoding="utf-8"))
    entries = bridge["exports"]
    if not entries:
        return []
    items = [
        (
            item["name"],
            decode(json.loads((root / f"{ref_key(item['ref'])}.json").read_text(encoding="utf-8"))),
            "auto",
        )
        for item in entries
    ]
    return store_many(items, refs=[item["ref"] for item in entries])


def load_bindings() -> Any:
    root = Path(os.environ["AI_AGENT_PARAMETER_BRIDGE"]).parent
    path = root / "bindings.json"
    return decode(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else {}


def export_result(value: Any) -> None:
    root = Path(os.environ["AI_AGENT_PARAMETER_BRIDGE"]).parent
    write_json(root / "result.json", encode(value))
