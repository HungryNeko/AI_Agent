"""RAG tool placeholder."""

from __future__ import annotations

from tools.settings import RagSettings


def auto_context(user_message: str, settings: RagSettings) -> str | None:
    """Return automatically injected RAG context when enabled.

    Real vector search is not implemented yet. Later this function should search
    embeddings with settings.min_similarity and settings.max_results.
    """

    if not settings.can_auto_include_context:
        return None
    return None


def search(query: str, settings: RagSettings) -> list[str]:
    """Search local/private knowledge.

    Placeholder for the future vector database lookup.
    """

    if not settings.can_model_call:
        raise ValueError("rag search is disabled for the model.")
    raise NotImplementedError(
        f"rag is configured but not implemented yet: query={query!r}, "
        f"min_similarity={settings.min_similarity}, max_results={settings.max_results}"
    )
