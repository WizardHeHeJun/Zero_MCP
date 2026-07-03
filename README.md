# Zero_MCP

> **临时 README（WIP）**。本仓库当前为**空工程骨架**，尚无业务逻辑，仅提供目标架构与目录结构占位。

经 **MCP（[Model Context Protocol](https://modelcontextprotocol.io)）**，给情感引擎驱动的 AI 数字人项目 **Zero** 扩展新的 **Agent 能力模块**。Zero_MCP 与 Zero 相对独立：把 Zero 当作**外部服务**经 MCP 调用，不直接耦合其代码库。

## 架构

三层单向依赖，叠加 MCP 互操作边界（依赖只能自上而下）：

```
编排层  src/orchestration, src/agents   （LangGraph StateGraph / Supervisor）
   ↓
记忆层  src/memory                       （长期记忆 / 图谱读写封装）
   ↓
存储层  src/storage                      （Postgres / Neo4j / Redis）

MCP 互操作边界（横切）
  · src/mcp        Python 侧 MCP client：经 MCP 调 Zero（外部服务）
  · mcp-server/    TypeScript MCP 服务层：@modelcontextprotocol/sdk 暴露/转发能力
```

- **运行态**（LangGraph Checkpointer）与**长期记忆**分离存储。
- **MCP 传输层不塞业务逻辑**：协议/转发在 TS 侧，情感/agent 业务逻辑留在 Python `src/*`。

## 技术栈

| 侧 | 语言 | 用途 |
| --- | --- | --- |
| 主 | Python 3.12 | LLM agent 框架 + 编排/记忆/存储 + MCP client |
| MCP 服务层 | TypeScript | `mcp-server/`，`@modelcontextprotocol/sdk`（McpServer + 传输） |

## 目录结构（骨架）

```
src/
  orchestration/   Supervisor、StateGraph、Checkpointer 接线
  agents/          各 Worker Agent 角色
  memory/          记忆读写 API、抽取/去重策略
  storage/         Postgres / Neo4j / Redis 连接与 schema
  mcp/             Python 侧 MCP client（把 Zero 当外部服务调用）
mcp-server/        TS MCP 服务层（独立 package.json）
tests/             单测 + agent 行为/记忆回归
```

## 开发

```bash
# Python 侧（复用 conda 环境 affective-expression，见 environment.yml）
conda activate affective-expression
uv sync           # 依赖待 pyproject.toml 补全后可用
pytest
ruff check . && ruff format .

# MCP 服务层（TS）
cd mcp-server && npm install && npm run typecheck
```

配置与密钥走 `.env`（参考 `.env.example`），不入库。

## 状态

当前仅为工程骨架占位，各模块 `__init__.py` / 入口文件均为空实现。业务能力随后续实现逐步落地。

> 说明：本仓库仅收录工程代码骨架；项目自用的开发工具链与知识库文档未包含在内。
