"""Tool access to persistent app settings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from agent.app_settings import load_app_settings, patch_app_settings, save_app_settings

SettingsAction = Literal["read", "update", "replace"]


@dataclass(frozen=True)
class SettingsRequest:
    action: SettingsAction
    patch: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)


def execute(request: SettingsRequest) -> str:
    if request.action == "read":
        return json.dumps({"status": "ok", "settings": load_app_settings()}, ensure_ascii=False, indent=2)
    if request.action == "update":
        return json.dumps({"status": "saved", "settings": patch_app_settings(request.patch)}, ensure_ascii=False, indent=2)
    if request.action == "replace":
        return json.dumps({"status": "saved", "settings": save_app_settings(request.settings)}, ensure_ascii=False, indent=2)
    raise ValueError(f"unknown settings action: {request.action}")
