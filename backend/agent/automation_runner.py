"""Lightweight scheduler for saved automation JSON files."""

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from agent.config import PROJECT_ROOT
from agent.debug_log import log_event, log_exception, sanitize
from agent import session_store
from tools.automation import AutomationRequest, automation_context, execute as execute_automation_tool
from tools.settings import make_tool_settings

AUTOMATION_ROOT = PROJECT_ROOT / "backend" / "runtime" / "automations"
RUN_ROOT = PROJECT_ROOT / "backend" / "runtime" / "automation_runs"
DEFAULT_INTERVAL_SECONDS = 5
_RUNNER: AutomationRunner | None = None


class AutomationRunner:
    def __init__(
        self,
        *,
        automation_root: Path = AUTOMATION_ROOT,
        run_root: Path = RUN_ROOT,
        interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    ) -> None:
        self.automation_root = automation_root
        self.run_root = run_root
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="automation-runner", daemon=True)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            run_due_automations(self.automation_root, self.run_root)


def start_runner() -> None:
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = AutomationRunner()
        _RUNNER.start()


def stop_runner() -> None:
    global _RUNNER
    if _RUNNER is not None:
        _RUNNER.stop()
        _RUNNER = None


def run_due_automations(
    automation_root: Path = AUTOMATION_ROOT,
    run_root: Path = RUN_ROOT,
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    current_time = now or datetime.now(UTC)
    automation_root.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(automation_root.glob("*.json")):
        try:
            item = read_json_object(path)
            if not is_due(item, current_time):
                continue
            records.append(run_one(path, item, run_root, current_time))
        except Exception as exc:  # noqa: BLE001
            log_exception("automation_runner.error", exc, path=str(path))
    return records


def run_one(path: Path, item: dict[str, Any], run_root: Path, now: datetime) -> dict[str, Any]:
    original_schedule = json.dumps(item.get("schedule") or {}, ensure_ascii=False, sort_keys=True)
    started_at = now.isoformat(timespec="seconds").replace("+00:00", "Z")
    run_id = str(uuid.uuid4())
    record: dict[str, Any] = {
        "run_id": run_id,
        "automation_id": path.name,
        "title": str(item.get("title") or path.stem),
        "action": str(item.get("action") or "reminder"),
        "started_at": started_at,
    }
    log_event("automation.run.start", automation_id=path.name, title=record["title"], action=record["action"])
    try:
        result = execute_saved_automation(path, item)
        finished_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        record.update({"finished_at": finished_at, "status": "ok", "result": sanitize(result)})
    except Exception as exc:  # noqa: BLE001
        finished_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        record.update({"finished_at": finished_at, "status": "error", "error": str(exc)})
        log_exception("automation.run.error", exc, automation_id=path.name, title=record["title"])
    append_run_record(run_root, record)
    latest_item = read_json_object(path) if path.exists() else item
    latest_schedule = json.dumps(latest_item.get("schedule") or {}, ensure_ascii=False, sort_keys=True)
    update_after_run(path, latest_item, record, now, schedule_was_updated=latest_schedule != original_schedule)
    log_event("automation.run.finish", automation_id=path.name, status=record["status"])
    return record


def execute_saved_automation(path: Path, item: dict[str, Any]) -> Any:
    action = str(item.get("action") or "reminder")
    if action == "llm":
        return run_llm_automation_conversation(path, item)
    if action == "reminder":
        result = {"reminder": str(item.get("prompt") or item.get("title") or "automation")}
        result["conversation_id"] = append_automation_conversation_turn(path, item, str(result["reminder"]), result)
        return result
    request = AutomationRequest(
        action=action,  # type: ignore[arg-type]
        title=str(item.get("title") or ""),
        prompt=str(item.get("prompt") or ""),
        code=str(item.get("code") or ""),
        mcp_server=str(item.get("mcp_server") or ""),
        mcp_tool=str(item.get("mcp_tool") or ""),
        mcp_arguments=dict(item.get("mcp_arguments") or {}),
        mcp_config=dict(item.get("mcp_config") or {}),
        schedule=dict(item.get("schedule") or {}),
    )
    with automation_context(path):
        result = execute_automation_tool(
            request,
            make_tool_settings(
                automation_mode="auto",
                python_mode="auto",
                mcp_mode="auto",
                mcp_config_path="data/mcp/servers.json",
            ),
        )
    result["conversation_id"] = append_automation_conversation_turn(path, item, str(item.get("prompt") or action), result)
    return result


def run_llm_automation_conversation(path: Path, item: dict[str, Any]) -> dict[str, Any]:
    from agent.graph import new_chat_state, stream_turn

    prompt = str(item.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("llm automation requires prompt.")
    conversation_id = ensure_automation_conversation(path, item)
    path.write_text(json.dumps(item, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    conversation = session_store.read_conversation(conversation_id)
    state = dict(conversation.get("state") or {})
    if not state:
        state = new_chat_state(
            model=optional_string(item.get("model")),
            automation_mode="auto",
            history_mode="auto",
            rag_mode="auto",
            mcp_mode="auto",
            python_mode="auto",
            curl_mode="auto",
            max_tool_rounds=10,
        )
    state["automation_mode"] = "auto"
    state["history_mode"] = "auto"
    state["mcp_mode"] = state.get("mcp_mode") or "auto"
    state["python_mode"] = state.get("python_mode") or "auto"
    if optional_string(item.get("model")):
        state["model"] = str(item["model"])

    message = build_automation_user_message(path, item, prompt)
    events = []
    final_state = {}
    with automation_context(path):
        for event in stream_turn(state, message):
            events.append(event)
            if event.get("type") == "assistant" and isinstance(event.get("state"), dict):
                final_state = event["state"]
    conversation = session_store.save_turn(
        conversation_id,
        user_text=message,
        turn_events=events,
        state=final_state or state,
    )
    ensure_automation_conversation_title(conversation, item, path)
    return {
        "conversation_id": conversation_id,
        "events": [{key: value for key, value in event.items() if key != "state"} for event in events],
    }


def build_automation_user_message(path: Path, item: dict[str, Any], prompt: str) -> str:
    schedule = json.dumps(item.get("schedule") or {}, ensure_ascii=False, indent=2)
    return (
        f"自动化任务执行。\n"
        f"automationId: {path.name}\n"
        f"automationPath: {relative_to_project(path)}\n"
        f"title: {item.get('title') or path.stem}\n"
        f"当前 schedule:\n{schedule}\n\n"
        "规则：如果需要设置下一次运行，默认必须用 automation tool 更新当前 automationId；"
        "除非用户明确要求创建新的自动化，否则不要新建。"
        "斐波那契、自定义间隔等计划要更新同一个 schedule 的 previousRunAt/currentRunAt/fibIndex/nextRunAt。\n\n"
        f"任务内容：\n{prompt}"
    )


def append_automation_conversation_turn(
    path: Path,
    item: dict[str, Any],
    user_text: str,
    result: dict[str, Any],
) -> str:
    conversation_id = ensure_automation_conversation(path, item)
    conversation = session_store.save_turn(
        conversation_id,
        user_text=f"自动化任务执行：{user_text}",
        turn_events=[{"type": "assistant", "text": json.dumps(result, ensure_ascii=False, indent=2)}],
        state={},
    )
    ensure_automation_conversation_title(conversation, item, path)
    return conversation_id


def ensure_automation_conversation(path: Path, item: dict[str, Any]) -> str:
    existing = optional_string(item.get("conversation_id"))
    if existing:
        return existing
    conversation_id = f"automation-{safe_conversation_id(path.stem)}"
    item["conversation_id"] = conversation_id
    conversation = session_store.read_conversation(conversation_id)
    if not conversation.get("events"):
        conversation["title"] = f"自动化: {item.get('title') or path.stem}"
        conversation["summary"] = "Automation execution history"
        session_store.write_conversation(conversation)
    return conversation_id


def ensure_automation_conversation_title(conversation: dict[str, Any], item: dict[str, Any], path: Path) -> None:
    title = f"自动化: {item.get('title') or path.stem}"
    if conversation.get("title") == title:
        return
    conversation["title"] = title
    session_store.write_conversation(conversation)


def is_due(item: dict[str, Any], now: datetime) -> bool:
    if not item.get("enabled", True):
        return False
    schedule = item.get("schedule")
    if not isinstance(schedule, dict):
        return False
    next_run_at = parse_time(optional_string(schedule.get("nextRunAt")))
    return bool(next_run_at and next_run_at <= now)


def update_after_run(
    path: Path,
    item: dict[str, Any],
    record: dict[str, Any],
    now: datetime,
    *,
    schedule_was_updated: bool = False,
) -> None:
    schedule = dict(item.get("schedule") or {})
    previous_current = optional_string(schedule.get("currentRunAt"))
    schedule["previousRunAt"] = previous_current or optional_string(schedule.get("previousRunAt")) or ""
    schedule["currentRunAt"] = record["started_at"]
    if not schedule_was_updated:
        kind = str(schedule.get("kind") or "once")
        if kind == "interval":
            interval = max(1, int(schedule.get("intervalSeconds") or 0))
            schedule["nextRunAt"] = (now + timedelta(seconds=interval)).isoformat(timespec="seconds").replace("+00:00", "Z")
        elif kind == "once":
            schedule["nextRunAt"] = ""
            item["enabled"] = False
        else:
            schedule["nextRunAt"] = ""
            schedule["needsNextRunAt"] = True
    item["schedule"] = schedule
    item["last_run"] = {key: record.get(key) for key in ["run_id", "status", "started_at", "finished_at", "error"]}
    item["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    path.write_text(json.dumps(item, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def append_run_record(run_root: Path, record: dict[str, Any]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / f"runs-{datetime.now(UTC).strftime('%Y%m%d')}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def list_run_records(
    run_root: Path = RUN_ROOT,
    *,
    automation_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    if not run_root.exists():
        return []
    records = []
    for path in sorted(run_root.glob("runs-*.jsonl"), reverse=True):
        for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if automation_id and record.get("automation_id") != automation_id:
                continue
            record["path"] = str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
            records.append(record)
            if len(records) >= limit:
                return records
    return records


def relative_to_project(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved)


def read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return data


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def safe_conversation_id(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip())
    clean = "-".join(part for part in clean.split("-") if part)
    return clean[:80] or uuid.uuid4().hex[:8]
