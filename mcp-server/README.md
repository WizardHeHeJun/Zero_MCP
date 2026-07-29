# mcp-server（TS · MCP 对外互操作层）

Zero_MCP 的 **TS 侧 MCP（Model Context Protocol）服务层**，用官方 [`@modelcontextprotocol/sdk`](https://github.com/modelcontextprotocol/typescript-sdk) 构建。**定位 = 对外互操作边界**：仅当需要向 Zero host 等外部方暴露**聚合**能力时启用。

> 本目录**仍为空骨架**（manifest 与入口占位），且这是**有意的**：按「MCP 层语言澄清」（CLAUDE.md 脚注 · 判据 **TS = 对外边界，Python = 内部封装**，依据见 [`notes/2026-07-10-screen-capability-blueprint.md`](../notes/2026-07-10-screen-capability-blueprint.md) §0 AD-1），内部能力封装的 MCP server 用官方 `mcp` python-sdk 直连自建，不为「MCP 必走 TS」加 child_process 桥。已落地的真实 MCP 实现均在 Python 侧：
>
> - `src/mcp/desktop/` — 桌面/屏幕感知与操控能力（server + client + perception/control tools）
> - `src/mcp/zero/` — 与原项目 Zero 的互操作层（client、感知注入、expression 消费、io_adapters 真采集）
>
> 本目录待「对外部 host 暴露聚合能力」的需求出现后再开工，由**工程师团队**（`/eng-team`）落地。

## 约定（见 `.claude/rules/`）

- `typescript-code.md`：`tsc` strict、eslint/prettier、无 `any` 兜底。
- `mcp-integration.md`：官方 SDK；传输首选 Streamable HTTP、兼 stdio；**传输层不塞业务逻辑**（业务在 Python `src/*`）；跨语言数据形状用 `zod` 校验、与 Python 侧 `pydantic` 契约对齐；把 Zero 当外部服务，不直接 import Zero 代码库。

## 首次开发（待触发条件出现后执行）

```bash
cd mcp-server
npm install        # 锁定 @modelcontextprotocol/sdk / typescript / zod 版本（当前 package.json 为 "*" 占位）
npm run typecheck  # tsc --noEmit
```

依赖独立装在本目录（不污染 Python 侧）。endpoint/密钥走仓库根 `.env`（勿提交）。
