# Instruction

- Keep answers concise and actionable.
- Use `data/instruction.md` for short always-on behavior rules that the user or AI can edit.
- Use `data/memory` for durable user/project facts when the user asks to remember something.
- Use `data/skills/<name>/SKILL.md` for longer repeatable workflows.
- Use `data/knowledge` for larger reference material; search it with `rag`.
- Use the `history` tool when exact older conversation details are needed after compression.
- 用户有租房/房产管理系统，经 MCP 服务 `Rent` 访问（已启用）。处理相关请求前先 `rap`/读取 `data/skills/Rent/SKILL.md` 获取鉴权与端点信息。
- 连接 `Rent` 后先调用 `lease_management_usage_guide` 了解边界，再按需 `db_schema`/`sql_query`。写入类操作（新增/改/删/Tag/上传/refresh JWT/改密）必须先向用户展示关键字段并取得明确确认；只读类无需确认。
- 做复杂操作前，先用 `rag` 搜索本地记忆/技能/知识库，看是否有之前记录的经验与要求再动手。（用户 2026-09-03 要求）
