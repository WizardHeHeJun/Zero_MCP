// node:http server：每请求先 verifyBearer（未过 → 401 + JSON body，短路不触达 MCP
// 层）→ StreamableHTTPServerTransport（stateless：sessionIdGenerator: undefined）。
//
// 401 响应体形状镜像 Zero `src/mcp_server/auth.py::_send_401`（同为 JSON body +
// `WWW-Authenticate: Bearer`），跨仓一致，降低两侧鉴权失败时的排障心智负担。
//
// stateless 模式下每个 HTTP 请求各自 new 一个 Server + Transport（同 SDK
// `examples/server/simpleStatelessStreamableHttp` 的模式，只是不用 express——本聚合器
// 只需要单一 POST 端点，裸 node:http 更薄，不必为此多引一个依赖）；请求结束
// （res "close"）时收尾两者，避免每请求泄漏。

import {
  createServer,
  type IncomingMessage,
  type Server as HttpServer,
  type ServerResponse,
} from "node:http";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { createAggregatorServer } from "./aggregatorServer.js";
import { verifyBearer } from "./auth.js";
import type { BackendRegistry } from "./backend/registry.js";
import type { AggregatorConfig } from "./config.js";
import { logger } from "./logging.js";

/** Node http 头的 authorization 字段可能是 string | string[]（重复头）；取第一个。 */
function normalizeAuthHeader(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

function send401(res: ServerResponse): void {
  const body = JSON.stringify({
    error: "invalid_token",
    error_description: "缺少或无效的 Bearer token",
  });
  res.writeHead(401, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(body).toString(),
    "www-authenticate": 'Bearer error="invalid_token", error_description="authentication required"',
  });
  res.end(body);
}

function sendJsonRpcError(res: ServerResponse, status: number, message: string): void {
  const body = JSON.stringify({
    jsonrpc: "2.0",
    error: { code: -32000, message },
    id: null,
  });
  res.writeHead(status, { "content-type": "application/json" });
  res.end(body);
}

async function handleRequest(
  req: IncomingMessage,
  res: ServerResponse,
  config: AggregatorConfig,
  registry: BackendRegistry,
): Promise<void> {
  // 鉴权在最外层做，不受路径/方法影响——短路后完全不触达 MCP 层，镜像 Zero 侧
  // BearerAuthMiddleware 包住整个 ASGI app（而非只包某条路由）的做法。
  if (!verifyBearer(normalizeAuthHeader(req.headers.authorization), config.enforcedToken)) {
    send401(res);
    return;
  }

  const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);
  if (url.pathname !== config.path) {
    sendJsonRpcError(res, 404, "Not Found");
    return;
  }
  if (req.method !== "POST") {
    // stateless 聚合器（sessionIdGenerator: undefined）不支持 GET 独立 SSE 流 /
    // DELETE 会话终止——那些语义只在有状态模式下有意义。
    sendJsonRpcError(res, 405, "Method not allowed.");
    return;
  }

  const server = createAggregatorServer(registry);
  const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
  res.on("close", () => {
    transport.close().catch((exc: unknown) => {
      logger.warn(
        `关闭请求级 transport 时出错：${exc instanceof Error ? exc.message : String(exc)}`,
      );
    });
    server.close().catch((exc: unknown) => {
      logger.warn(`关闭请求级 server 时出错：${exc instanceof Error ? exc.message : String(exc)}`);
    });
  });
  await server.connect(transport);
  await transport.handleRequest(req, res);
}

/** 构造 HTTP server（未调用 listen；由 index.ts 决定何时起停）。 */
export function createHttpTransport(
  config: AggregatorConfig,
  registry: BackendRegistry,
): HttpServer {
  return createServer((req, res) => {
    handleRequest(req, res, config, registry).catch((exc: unknown) => {
      logger.error(`处理请求时未捕获异常：${exc instanceof Error ? exc.message : String(exc)}`);
      if (!res.headersSent) {
        sendJsonRpcError(res, 500, "Internal server error");
      }
    });
  });
}
