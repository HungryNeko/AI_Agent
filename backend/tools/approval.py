"""Manual-approval tool.

Like the question tool, this only describes the request. The graph emits it to the UI
and pauses the current turn; the user's decision (approve or reject, with optional
free text) returns through the normal chat loop as an ``approvalResponse`` message.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal

Risk = Literal["low", "medium", "high"]
RISK_LEVELS: tuple[str, ...] = ("low", "medium", "high")


@dataclass(frozen=True)
class ApprovalRequest:
    action: str
    title: str = ""
    details: str = ""
    risk: Risk = "medium"


def waiting_result(request: ApprovalRequest) -> str:
    """Return the tool message kept in model conversation history."""

    payload = asdict(request)
    payload["status"] = "waiting_for_user"
    return "approvalResult:\n" + json.dumps(payload, ensure_ascii=False)


def normalize_risk(value: object) -> Risk:
    text = str(value or "").strip().lower()
    return text if text in RISK_LEVELS else "medium"  # type: ignore[return-value]
