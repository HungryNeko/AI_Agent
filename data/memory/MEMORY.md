# Agent Memory

Use this folder for durable project notes that should be retrievable later.

Guidelines:
- Keep entries short and evidence-based.
- Prefer dated notes when behavior or project decisions may change.
- Use `fileEditor` to update memory so approval policy still applies.
## 用户信息
- 所在地：洛杉矶（Los Angeles）（记录于 2026-09-02）

## 租房系统（Lease Management System）
- 用户拥有一套本地租房/房产管理系统，通过 MCP 服务名 `Rent` 访问；连接已启用并验证可用（2026-09-03）。
- 详细集成指南见 skill：`data/skills/Rent/SKILL.md`（账号/鉴权、各模块 API、MCP SQL/PDF 规则、写法流程、安全确认要求均在文档内）。
- 可用 MCP tools：`lease_management_usage_guide`（连接后先调用）、`db_schema`、`sql_query`（只读）、`sql_execute`（写入须 `confirm_write=true`）、`generate_pdf`、`api_request`（写入须 `confirm_write=true`；鉴权与管理员接口不开放）。
- 数据写入（新增/改/删/Tag/上传/刷新JWT/改密码）前必须先向本用户展示关键字段并取得明确确认；只读查询/统计无需确认。
