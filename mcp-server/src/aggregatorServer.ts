// 聚合 MCP Server：低层 `Server`（非 `McpServer`）+ `ListToolsRequestSchema` /
// `CallToolRequestSchema` handler，纯转发 `BackendRegistry`。
//
// 这是「传输层零业务逻辑」的落点：本文件不 import 任何业务类型（VTS/桌面），
// inputSchema 与 tools/call 的参数、结果均原样透传给 registry，不做任何解释/改写。
//
// 实现自由度：用低层 `Server` 而非 `McpServer`——后者的 `registerTool` 要求把
// inputSchema 转成 zod 对象再由 SDK 反向序列化回 JSON Schema，这一来一回既多余
// （我们本就持有后端原始的 JSON Schema）又有精度损失风险（zod → JSON Schema
// 不总是无损往返）。低层 `Server` 直接在 handler 里回原始 `Tool[]`，是本聚合器
// 场景下更贴切的「纯转发」实现，SDK 文档也把 `Server` 标注为"advanced use cases"
// 而非弃用禁用。

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import type { BackendRegistry } from "./backend/registry.js";

/** 构造一个新的聚合 Server 实例（每个 HTTP 请求一个，见 httpTransport.ts 的 stateless 模式）。 */
export function createAggregatorServer(registry: BackendRegistry): Server {
  const server = new Server(
    { name: "zero-mcp-aggregator", version: "0.0.0" },
    { capabilities: { tools: {} } },
  );

  server.setRequestHandler(ListToolsRequestSchema, () => {
    return { tools: registry.listTools() };
  });

  server.setRequestHandler(CallToolRequestSchema, (request) => {
    const { name, arguments: args } = request.params;
    return registry.callTool(name, args);
  });

  return server;
}
