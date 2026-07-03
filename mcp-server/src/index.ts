// Zero_MCP · MCP 服务层入口（骨架占位，本轮不实现任何逻辑）
//
// TODO（由工程师团队在首个真实功能时落地，见 .claude/rules/mcp-integration.md）：
//   - 用 @modelcontextprotocol/sdk 的 McpServer 注册工具 / 资源 / prompt
//   - 传输：Streamable HTTP（远程，首选）或 stdio（本地 / 子进程）
//   - 传输层只做协议 / 转发，业务逻辑在 Python 侧（src/*）
//   - 跨语言数据形状用 zod 在边界校验，与 Python 侧 pydantic 契约对齐
//   - endpoint / 模型 ID / token 走 .env，不硬编码

export {};
