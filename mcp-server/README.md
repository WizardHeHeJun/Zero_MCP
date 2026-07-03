# mcp-server（TS · MCP 服务层）

Zero_MCP 的 **MCP（Model Context Protocol）服务层**，用官方 [`@modelcontextprotocol/sdk`](https://github.com/modelcontextprotocol/typescript-sdk) 暴露/转发新 Agent 能力，是与原项目 **Zero** 及外部 host 的互操作边界。

> 当前为**空骨架**：只有 manifest 与入口占位，无任何业务逻辑。首个真实功能由**工程师团队**（`/eng-team`）落地。

## 约定（见 `.claude/rules/`）

- `typescript-code.md`：`tsc` strict、eslint/prettier、无 `any` 兜底。
- `mcp-integration.md`：官方 SDK；传输首选 Streamable HTTP、兼 stdio；**传输层不塞业务逻辑**（业务在 Python `src/*`）；跨语言数据形状用 `zod` 校验、与 Python 侧 `pydantic` 契约对齐；把 Zero 当外部服务，不直接 import Zero 代码库。

## 首次开发（待工程师团队执行）

```bash
cd mcp-server
npm install        # 锁定 @modelcontextprotocol/sdk / typescript / zod 版本（当前 package.json 为 "*" 占位）
npm run typecheck  # tsc --noEmit
```

依赖独立装在本目录（不污染 Python 侧）。endpoint/密钥走仓库根 `.env`（勿提交）。
