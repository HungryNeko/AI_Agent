"""System-side tool settings.

The model should not see most of these values. They are backend controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RagMode = Literal["off", "on", "auto"]
WebSearchMode = Literal["off", "auto"]
CurlMode = Literal["off", "auto"]
PythonMode = Literal["off", "auto"]
FileEditorMode = Literal["off", "auto"]
FileEditorApproval = Literal["readOnly", "manual", "auto"]
McpMode = Literal["off", "auto"]
HistoryMode = Literal["off", "auto"]
AutomationMode = Literal["off", "auto"]
WebSearchProvider = Literal["duckduckgo", "searxng", "tavily"]


@dataclass(frozen=True)
class WebSearchSettings:
    mode: WebSearchMode = "auto"
    provider: WebSearchProvider = "duckduckgo"
    auto_switch: bool = False
    api_key_env: str = "TAVILY_API_KEY"
    base_url: str = ""
    max_results: int = 5
    timeout_seconds: float = 10.0
    search_depth: str = "basic"
    topic: str = "general"
    include_answer: bool = False

    @property
    def can_model_call(self) -> bool:
        return self.mode == "auto"


@dataclass(frozen=True)
class RagSettings:
    mode: RagMode = "auto"
    min_similarity: float = 0.05
    max_results: int = 5
    auto_include: bool = False
    include_knowledge: bool = True
    include_memory: bool = True
    include_skills: bool = True
    knowledge_root: str = "data/knowledge"
    memory_root: str = "data/memory"
    skills_root: str = "data/skills"
    user_knowledge_root: str = "backend/runtime/user_data/knowledge"
    user_memory_root: str = "backend/runtime/user_data/memory"
    user_skills_root: str = "backend/runtime/user_data/skills"
    max_file_bytes: int = 200_000
    max_chunk_chars: int = 2_000
    chunk_overlap_chars: int = 200
    index_path: str = "backend/runtime/rag_index/index.pkl"

    @property
    def can_model_call(self) -> bool:
        return self.mode in {"on", "auto"}

    @property
    def can_auto_include_context(self) -> bool:
        return self.mode == "on" or self.auto_include


@dataclass(frozen=True)
class CurlSettings:
    mode: CurlMode = "auto"
    timeout_seconds: float = 20.0
    max_bytes: int = 20_000

    @property
    def can_model_call(self) -> bool:
        return self.mode == "auto"


@dataclass(frozen=True)
class PythonSettings:
    mode: PythonMode = "auto"
    timeout_seconds: float = 10.0
    max_output_chars: int = 8_000
    artifact_root: str = "backend/runtime/python_runs"
    max_artifacts: int = 20
    max_artifact_bytes: int = 5_000_000

    @property
    def can_model_call(self) -> bool:
        return self.mode == "auto"


@dataclass(frozen=True)
class FileEditorSettings:
    mode: FileEditorMode = "auto"
    approval: FileEditorApproval = "auto"
    root: str = ""
    max_file_bytes: int = 400_000
    max_read_chars: int = 60_000
    max_write_bytes: int = 400_000
    max_list_results: int = 120

    @property
    def can_model_call(self) -> bool:
        return self.mode == "auto"


@dataclass(frozen=True)
class McpSettings:
    mode: McpMode = "auto"
    config_path: str = "data/mcp/servers.json"
    timeout_seconds: float = 20.0
    max_output_chars: int = 20_000

    @property
    def can_model_call(self) -> bool:
        return self.mode == "auto"


@dataclass(frozen=True)
class HistorySettings:
    mode: HistoryMode = "off"

    @property
    def can_model_call(self) -> bool:
        return self.mode == "auto"


@dataclass(frozen=True)
class AutomationSettings:
    mode: AutomationMode = "off"
    root: str = "backend/runtime/automations"

    @property
    def can_model_call(self) -> bool:
        return self.mode == "auto"


@dataclass(frozen=True)
class ToolSettings:
    web_search: WebSearchSettings = WebSearchSettings()
    rag: RagSettings = RagSettings()
    curl: CurlSettings = CurlSettings()
    python: PythonSettings = PythonSettings()
    file_editor: FileEditorSettings = FileEditorSettings()
    mcp: McpSettings = McpSettings()
    history: HistorySettings = HistorySettings()
    automation: AutomationSettings = AutomationSettings()

    def model_view(self) -> dict[str, list[str]]:
        """Return only what the model needs to know."""

        available = []
        if self.web_search.can_model_call:
            available.append("webSearch")
        if self.rag.can_model_call:
            available.append("rag")
        if self.curl.can_model_call:
            available.append("curl")
        if self.python.can_model_call:
            available.append("python")
        if self.file_editor.can_model_call:
            available.append("fileEditor")
        if self.mcp.can_model_call:
            available.append("mcp")
        if self.history.can_model_call:
            available.append("history")
        if self.automation.can_model_call:
            available.append("automation")
            available.append("settings")
        return {"available": available}


def make_tool_settings(
    *,
    web_search: bool | None = None,
    web_search_mode: str = "auto",
    rag_mode: str = "auto",
    curl: bool | None = None,
    curl_mode: str = "auto",
    python: bool | None = None,
    python_mode: str = "auto",
    file_editor: bool | None = None,
    file_editor_mode: str = "auto",
    file_editor_approval: str = "auto",
    mcp: bool | None = None,
    mcp_mode: str = "auto",
    history: bool | None = None,
    history_mode: str = "off",
    automation: bool | None = None,
    automation_mode: str = "off",
    web_search_provider: WebSearchProvider = "duckduckgo",
    web_search_auto_switch: bool = False,
    web_search_base_url: str | None = None,
    web_search_max_results: int = 5,
    web_search_timeout_seconds: float = 10.0,
    web_search_depth: str = "basic",
    web_search_topic: str = "general",
    web_search_include_answer: bool = False,
    rag_min_similarity: float = 0.05,
    rag_max_results: int = 5,
    rag_include_knowledge: bool = True,
    rag_include_memory: bool = True,
    rag_include_skills: bool = True,
    rag_knowledge_root: str = "data/knowledge",
    rag_memory_root: str = "data/memory",
    rag_skills_root: str = "data/skills",
    rag_user_knowledge_root: str = "backend/runtime/user_data/knowledge",
    rag_user_memory_root: str = "backend/runtime/user_data/memory",
    rag_user_skills_root: str = "backend/runtime/user_data/skills",
    rag_max_file_bytes: int = 200_000,
    rag_max_chunk_chars: int = 2_000,
    rag_chunk_overlap_chars: int = 200,
    rag_index_path: str = "backend/runtime/rag_index/index.pkl",
    curl_timeout_seconds: float = 20.0,
    curl_max_bytes: int = 20_000,
    python_timeout_seconds: float = 10.0,
    python_max_output_chars: int = 8_000,
    python_artifact_root: str = "backend/runtime/python_runs",
    python_max_artifacts: int = 20,
    python_max_artifact_bytes: int = 5_000_000,
    file_editor_root: str = "",
    file_editor_max_file_bytes: int = 400_000,
    file_editor_max_read_chars: int = 60_000,
    file_editor_max_write_bytes: int = 400_000,
    file_editor_max_list_results: int = 120,
    mcp_config_path: str = "data/mcp/servers.json",
    mcp_timeout_seconds: float = 20.0,
    mcp_max_output_chars: int = 20_000,
    automation_root: str = "backend/runtime/automations",
) -> ToolSettings:
    if web_search is not None:
        web_search_mode = "auto" if web_search else "off"
    if curl is not None:
        curl_mode = "auto" if curl else "off"
    if python is not None:
        python_mode = "auto" if python else "off"
    if file_editor is not None:
        file_editor_mode = "auto" if file_editor else "off"
    if mcp is not None:
        mcp_mode = "auto" if mcp else "off"
    if history is not None:
        history_mode = "auto" if history else "off"
    if automation is not None:
        automation_mode = "auto" if automation else "off"

    provider = normalize_web_search_provider(web_search_provider)
    return ToolSettings(
        web_search=WebSearchSettings(
            mode=normalize_web_search_mode(web_search_mode),
            provider=provider,
            auto_switch=bool(web_search_auto_switch),
            base_url=default_web_search_base_url(provider, web_search_base_url),
            max_results=web_search_max_results,
            timeout_seconds=web_search_timeout_seconds,
            search_depth=web_search_depth,
            topic=web_search_topic,
            include_answer=web_search_include_answer,
        ),
        rag=RagSettings(
            mode=normalize_rag_mode(rag_mode),
            min_similarity=rag_min_similarity,
            max_results=rag_max_results,
            include_knowledge=rag_include_knowledge,
            include_memory=rag_include_memory,
            include_skills=rag_include_skills,
            knowledge_root=rag_knowledge_root,
            memory_root=rag_memory_root,
            skills_root=rag_skills_root,
            user_knowledge_root=rag_user_knowledge_root,
            user_memory_root=rag_user_memory_root,
            user_skills_root=rag_user_skills_root,
            max_file_bytes=max(1_000, int(rag_max_file_bytes)),
            max_chunk_chars=max(500, int(rag_max_chunk_chars)),
            chunk_overlap_chars=max(0, int(rag_chunk_overlap_chars)),
            index_path=rag_index_path,
        ),
        curl=CurlSettings(
            mode=normalize_curl_mode(curl_mode),
            timeout_seconds=curl_timeout_seconds,
            max_bytes=max(1_000, int(curl_max_bytes)),
        ),
        python=PythonSettings(
            mode=normalize_python_mode(python_mode),
            timeout_seconds=python_timeout_seconds,
            max_output_chars=max(1_000, int(python_max_output_chars)),
            artifact_root=python_artifact_root,
            max_artifacts=max(1, int(python_max_artifacts)),
            max_artifact_bytes=max(1_000, int(python_max_artifact_bytes)),
        ),
        file_editor=FileEditorSettings(
            mode=normalize_file_editor_mode(file_editor_mode),
            approval=normalize_file_editor_approval(file_editor_approval),
            root=file_editor_root,
            max_file_bytes=max(1_000, int(file_editor_max_file_bytes)),
            max_read_chars=max(1_000, int(file_editor_max_read_chars)),
            max_write_bytes=max(1_000, int(file_editor_max_write_bytes)),
            max_list_results=max(1, int(file_editor_max_list_results)),
        ),
        mcp=McpSettings(
            mode=normalize_mcp_mode(mcp_mode),
            config_path=mcp_config_path,
            timeout_seconds=mcp_timeout_seconds,
            max_output_chars=max(1_000, int(mcp_max_output_chars)),
        ),
        history=HistorySettings(mode=normalize_history_mode(history_mode)),
        automation=AutomationSettings(
            mode=normalize_automation_mode(automation_mode),
            root=automation_root,
        ),
    )


def normalize_web_search_mode(web_search_mode: str) -> WebSearchMode:
    mode = web_search_mode.strip().lower()
    if mode not in {"off", "auto"}:
        raise ValueError("web_search_mode must be one of: off, auto.")
    return mode


def normalize_curl_mode(curl_mode: str) -> CurlMode:
    mode = curl_mode.strip().lower()
    if mode not in {"off", "auto"}:
        raise ValueError("curl_mode must be one of: off, auto.")
    return mode


def normalize_python_mode(python_mode: str) -> PythonMode:
    mode = python_mode.strip().lower()
    if mode not in {"off", "auto"}:
        raise ValueError("python_mode must be one of: off, auto.")
    return mode


def normalize_file_editor_mode(file_editor_mode: str) -> FileEditorMode:
    mode = file_editor_mode.strip().lower()
    if mode not in {"off", "auto"}:
        raise ValueError("file_editor_mode must be one of: off, auto.")
    return mode


def normalize_file_editor_approval(file_editor_approval: str) -> FileEditorApproval:
    value = file_editor_approval.strip().lower()
    aliases = {
        "readonly": "readOnly",
        "read_only": "readOnly",
        "read-only": "readOnly",
        "manual": "manual",
        "auto": "auto",
    }
    if value not in aliases:
        raise ValueError("file_editor_approval must be one of: readOnly, manual, auto.")
    return aliases[value]


def normalize_mcp_mode(mcp_mode: str) -> McpMode:
    mode = mcp_mode.strip().lower()
    if mode not in {"off", "auto"}:
        raise ValueError("mcp_mode must be one of: off, auto.")
    return mode


def normalize_history_mode(history_mode: str) -> HistoryMode:
    mode = history_mode.strip().lower()
    if mode not in {"off", "auto"}:
        raise ValueError("history_mode must be one of: off, auto.")
    return mode


def normalize_automation_mode(automation_mode: str) -> AutomationMode:
    mode = automation_mode.strip().lower()
    if mode not in {"off", "auto"}:
        raise ValueError("automation_mode must be one of: off, auto.")
    return mode


def normalize_web_search_provider(provider: str) -> WebSearchProvider:
    value = provider.strip().lower()
    if value not in {"duckduckgo", "searxng", "tavily"}:
        raise ValueError("web_search_provider must be one of: duckduckgo, searxng, tavily.")
    return value


def default_web_search_base_url(
    provider: WebSearchProvider,
    configured_base_url: str | None,
) -> str:
    if configured_base_url:
        return configured_base_url.rstrip("/")
    if provider == "searxng":
        return "http://localhost:8080"
    if provider == "tavily":
        return "https://api.tavily.com"
    return ""


def normalize_rag_mode(rag_mode: str) -> RagMode:
    mode = rag_mode.strip().lower()
    if mode not in {"off", "on", "auto"}:
        raise ValueError("rag_mode must be one of: off, on, auto.")
    return mode
