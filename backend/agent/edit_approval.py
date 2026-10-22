"""Manual approval of file edits.

In manual mode a mutating fileEditor call is not applied: the turn pauses with an approval request that carries
the exact edits and their diffs. When the user approves, the server applies those stored edits itself (the model
does not redo them, so nothing different can be written) and tells the model what happened.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields, replace
from typing import Any

from tools.executor import execute_tool
from tools.fileEditor import FileEditRequest
from tools.request import ToolRequest
from tools.settings import make_tool_settings

PENDING_KEY = "pending_edits"
PAUSE_NOTE = (
    "\nThe user is now asked to approve this edit. If they approve, the system applies it exactly as shown and "
    "reports the result with the approvalResponse; do not repeat the edit."
)
EDIT_FIELDS = {field.name for field in fields(FileEditRequest)}


def pending_edit(request: FileEditRequest, result: str) -> dict[str, Any]:
    edit = asdict(request)
    edit["diff"] = result.split("\ndiff:\n", 1)[1] if "\ndiff:\n" in result else ""
    return edit


def approval_request(edits: list[dict[str, Any]]) -> dict[str, Any]:
    paths = ", ".join(dict.fromkeys(str(edit.get("path") or "") for edit in edits))
    count = len(edits)
    details = "\n\n".join(
        f"**{edit.get('path')}** ({edit.get('action')})\n\n```diff\n{str(edit.get('diff') or '').strip() or '(no changes)'}\n```"
        for edit in edits
    )
    return {
        "kind": "approval",
        "source": "fileEditor",
        "action": f"Apply {count} file edit{'s' if count != 1 else ''}: {paths}",
        "title": "",
        "details": details,
        "risk": "medium",
        PENDING_KEY: edits,
    }


def resolve_response(message: str, saved_state: dict[str, Any], conversation_id: str) -> str:
    """Apply the stored edits when this message approves them; returns the message the model should see."""

    if not message.startswith("approvalResponse:"):
        return message
    pending = saved_state.get("question_pending") or {}
    edits = pending.get(PENDING_KEY) if isinstance(pending, dict) else None
    if pending.get("kind") != "approval" or not isinstance(edits, list) or not edits:
        return message
    try:
        response = json.loads(message.split(":", 1)[1])
    except ValueError:
        return message
    if not isinstance(response, dict) or response.get("action") != pending.get("action"):
        return message  # an answer to some other request
    if response.get("status") != "approved":
        return f"{message}\n\nThe edits were not applied."
    settings = make_tool_settings(file_editor_mode="auto", file_editor_approval="auto")
    settings = replace(settings, conversation_id=conversation_id)
    results = []
    for index, edit in enumerate(edits):
        request = FileEditRequest(**{key: value for key, value in edit.items() if key in EDIT_FIELDS})
        results.append(execute_tool(ToolRequest(id=f"approved-edit-{index}", name="fileEditor", file_edit=request), settings))
    return f"{message}\n\nappliedEdits (the system applied the approved edits; do not repeat them):\n" + "\n\n".join(results)
