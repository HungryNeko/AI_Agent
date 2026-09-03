# Agent Backend Skill

Use this skill when changing the Python LangGraph backend.

Rules:
- Keep the graph simple: conversation begin, assistant step, tool call, assistant step, final answer.
- Keep fixed tool rules in the first system prompt for cache reuse.
- Normal turns should only add compact dynamic context such as available tools and injected results.
- Prefer `fileEditor` for project file edits so approval policy is respected.
- Use `rag` to retrieve project memory, skill instructions, and local knowledge before broad changes.