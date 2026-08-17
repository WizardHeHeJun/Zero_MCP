#!/usr/bin/env node
// 纯 JS 假 stdio MCP 后端（供 backendManager/registry/httpAggregator 测试使用）。
//
// 用真实 @modelcontextprotocol/sdk（从 mcp-server/node_modules 解析——本文件位于
// mcp-server/tests/fixtures/ 内，裸 specifier 沿目录树向上找得到）起一个最小 stdio
// MCP server，注册固定工具集：echo/add/slow/boom。行为受命令行 flag 控制，不受
// process.env 控制——每个测试用例经 BackendConfig.args 独立传参，避免并行测试互相
//污染全局 env（vitest 默认并发跑多个测试文件）。
//
// 支持的 flag（`--key=value`，均可选）：
//   --startup-delay-ms=N       连接 stdio transport 前先 sleep N ms（模拟慢启动，
//                               用于测试 BackendManager 的握手超时）。
//   --exit-after-connect-ms=N  连接成功后 N ms 强制 process.exit(1)（模拟后端崩溃，
//                               用于测试 BackendManager 的指数退避重启）。
//   --exit-immediately         不连接，进程启动后立即 process.exit(1)（模拟 spawn
//                               后瞬间崩溃，握手 Promise 永远不 resolve）。
//   --pollute-prefix=<prefix>  额外注册一个已带该前缀的工具名（如 vts__polluted），
//                               用于测试 registry 对「后端违规自带前缀」的跳过逻辑。

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

/** 解析 `--key=value` 形式的命令行 flag（未取到值的 flag 记为空字符串）。 */
function parseFlags(argv) {
  const flags = {};
  for (const arg of argv) {
    const match = /^--([^=]+)(?:=(.*))?$/.exec(arg);
    if (match) {
      flags[match[1]] = match[2] ?? "";
    }
  }
  return flags;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main() {
  const flags = parseFlags(process.argv.slice(2));

  if ("exit-immediately" in flags) {
    process.exit(1);
    return;
  }

  const startupDelayMs = Number(flags["startup-delay-ms"] ?? "0");
  const exitAfterConnectMs = flags["exit-after-connect-ms"];
  const polluteWithPrefix = flags["pollute-prefix"];

  const server = new McpServer({ name: "fake-backend", version: "0.0.0" });

  server.registerTool(
    "echo",
    {
      description: "原样返回传入参数",
      inputSchema: { value: z.unknown().optional() },
    },
    // 注意：故意不声明 outputSchema，SDK 不会对返回内容做结构化校验；
    // 用 text content 承载 JSON 字符串，方便测试断言原样透传。
    async (args) => ({
      content: [{ type: "text", text: JSON.stringify(args ?? {}) }],
    }),
  );

  server.registerTool(
    "add",
    {
      description: "两数相加",
      inputSchema: { a: z.number(), b: z.number() },
    },
    async ({ a, b }) => ({
      content: [{ type: "text", text: String(a + b) }],
    }),
  );

  server.registerTool(
    "slow",
    {
      description: "sleep 指定毫秒后返回（供未来的调用侧延迟测试使用）",
      inputSchema: { delayMs: z.number().optional() },
    },
    async ({ delayMs }) => {
      const effectiveDelay = delayMs ?? 0;
      await sleep(effectiveDelay);
      return { content: [{ type: "text", text: `slept ${effectiveDelay}ms` }] };
    },
  );

  server.registerTool(
    "env",
    {
      description:
        "回显子进程当前 process.env 中指定 key 的值（供测试间接验证 backendManager.ts " +
        "buildChildEnv 的过滤行为：不存在的 key 返回 null）",
      inputSchema: { keys: z.array(z.string()) },
    },
    async ({ keys }) => {
      const seen = {};
      for (const key of keys) {
        seen[key] = process.env[key] ?? null;
      }
      return { content: [{ type: "text", text: JSON.stringify(seen) }] };
    },
  );

  server.registerTool(
    "boom",
    {
      description: "恒抛错误，验证 ToolError 形态（SDK 自动转 isError CallToolResult）",
      inputSchema: {},
    },
    async () => {
      throw new Error("[fake:boom] simulated tool failure");
    },
  );

  if (polluteWithPrefix !== undefined) {
    server.registerTool(
      `${polluteWithPrefix}polluted`,
      {
        description: "违规自带前缀的工具名，用于测试 registry 跳过逻辑",
        inputSchema: {},
      },
      async () => ({ content: [{ type: "text", text: "should-be-skipped" }] }),
    );
  }

  if (startupDelayMs > 0) {
    await sleep(startupDelayMs);
  }

  const transport = new StdioServerTransport();
  await server.connect(transport);

  if (exitAfterConnectMs !== undefined) {
    const delay = Number(exitAfterConnectMs === "" ? "0" : exitAfterConnectMs);
    setTimeout(() => {
      process.exit(1);
    }, delay);
  }
}

main().catch((exc) => {
  process.stderr.write(
    `fake-backend-server 启动失败：${exc instanceof Error ? (exc.stack ?? exc.message) : String(exc)}\n`,
  );
  process.exit(1);
});
