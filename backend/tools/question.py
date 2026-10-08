"""User-confirmation question tool.

The tool only describes a question. The graph emits it to the UI and pauses
the current turn; the user's response returns through the normal chat loop.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QuestionRequest:
    question: str
    options: tuple[str, ...] = ()
    multiple: bool = False
    title: str = ""
    placeholder: str = ""


def waiting_result(request: QuestionRequest) -> str:
    """Return the tool message kept in model conversation history."""

    payload = asdict(request)
    payload["options"] = list(request.options)
    payload["status"] = "waiting_for_user"
    return "questionResult:\n" + json.dumps(payload, ensure_ascii=False)
