"""Validation of RAG options and the saved build options."""

from __future__ import annotations

from tools.rag.models import SEARCH_METHODS, SOURCE_TYPES, BuildOptions, SplitMode


def normalize_split_mode(value: str) -> SplitMode:
    mode = value.strip().lower()
    if mode not in {"simple", "llm"}:
        raise ValueError("split_mode must be one of: simple, llm.")
    return mode  # type: ignore[return-value]


def normalize_method(value: str | None) -> str:
    method = (value or "").strip().lower() or "hybrid"
    if method not in SEARCH_METHODS:
        raise ValueError(f"rag method must be one of: {', '.join(SEARCH_METHODS)}.")
    return method


SOURCE_ALIASES = {"skills": "skill"}


def normalize_sources(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    clean: list[str] = []
    for value in values or ():
        source = SOURCE_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())
        if source not in SOURCE_TYPES:
            raise ValueError(f"rag sources must be from: {', '.join(SOURCE_TYPES)}.")
        if source not in clean:
            clean.append(source)
    return tuple(clean)


def resolve_model(value: str) -> str:
    """An explicit `provider:model`, or the default model when none is chosen."""

    clean = value.strip()
    if clean:
        return clean
    from agent.config import get_model_config

    config = get_model_config()
    return f"{config.provider}:{config.model_id}"


def saved_build_options(split_mode: str | None = None, chunk_model: str = "") -> BuildOptions:
    """The build options from the saved app settings; a split mode or chunk model given here wins."""

    try:
        from agent.app_settings import load_app_settings

        saved = load_app_settings().get("rag_ingestion") or {}
    except Exception:  # noqa: BLE001 - unreadable settings fall back to the defaults
        saved = {}
    mode = normalize_split_mode(split_mode or str(saved.get("split_mode") or "simple"))
    return BuildOptions(
        split_mode=mode,
        chunk_model=chunk_model.strip() or str(saved.get("model") or "").strip(),
        contextual="llm" if saved.get("contextual") == "llm" else "off",
        context_model=str(saved.get("context_model") or "").strip(),
        embedding_model=str(saved.get("embedding_model") or "").strip(),
    )


def resolved_options(options: BuildOptions) -> BuildOptions:
    """Fill in the default model wherever an LLM step is switched on."""

    chunk_model = resolve_model(options.chunk_model) if options.split_mode == "llm" else options.chunk_model
    context_model = resolve_model(options.context_model) if options.contextual == "llm" else options.context_model
    return BuildOptions(
        split_mode=options.split_mode,
        chunk_model=chunk_model,
        contextual=options.contextual,
        context_model=context_model,
        embedding_model=options.embedding_model.strip(),
    )
