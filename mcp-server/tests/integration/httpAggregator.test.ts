// 端到端集成测试：真实 loopback HTTP（随机端口）+ 真实 SDK Client
// （StreamableHTTPClientTransport）+ 两个真实 fake stdio 子进程后端。
//
// 覆盖：tools/list 并集、tools/call 经 HTTP 到后端往返、401（未设/错 token）、
// 单后端故障降级。收尾必须优雅关停（transport/server/registry 逐一 close），不
// 留悬挂子进程/端口。

import type { AddressInfo } from "node:net";
import type { Server as HttpServer } from "node:http";
import { afterEach, describe, expect, it } from "vitest";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { createHttpTransport } from "../../src/httpTransport.js";
import { BackendRegistry } from "../../src/backend/registry.js";
import type { AggregatorConfig } from "../../src/config.js";
import { buildFakeBackendConfig, type FakeBackendOptions } from "../support/fakeBackend.js";
import { waitUntil } from "../support/waitUntil.js";

interface Harness {
  httpServer: HttpServer;
  registry: BackendRegistry;
  config: AggregatorConfig;
  baseUrl: URL;
}

const harnesses: Harness[] = [];
const clients: Client[] = [];

function buildAggregatorConfig(overrides: Partial<AggregatorConfig> = {}): AggregatorConfig {
  return {
    enabled: true,
    host: "127.0.0.1",
    port: 0,
    path: "/mcp",
    enforcedToken: null,
    vts: { enabled: true, command: "python", args: [], cwd: "." },
    desktop: { enabled: true, command: "python", args: [], cwd: "." },
    backendStartTimeoutMs: 8000,
    backendRestartMax: 0,
    ...overrides,
  };
}

async function startHarness(
  vtsOptions: Partial<FakeBackendOptions> = {},
  desktopOptions: Partial<FakeBackendOptions> = {},
  configOverrides: Partial<AggregatorConfig> = {},
): Promise<Harness> {
  const registry = new BackendRegistry(
    buildFakeBackendConfig({ id: "vts", ...vtsOptions }),
    buildFakeBackendConfig({ id: "desktop", ...desktopOptions }),
  );
  await registry.start();

  const config = buildAggregatorConfig(configOverrides);
  const httpServer = createHttpTransport(config, registry);
  await new Promise<void>((resolve, reject) => {
    httpServer.once("error", reject);
    httpServer.listen(0, "127.0.0.1", () => {
      httpServer.removeListener("error", reject);
      resolve();
    });
  });
  const address = httpServer.address() as AddressInfo;
  const baseUrl = new URL(`http://127.0.0.1:${address.port}${config.path}`);

  const harness: Harness = { httpServer, registry, config, baseUrl };
  harnesses.push(harness);
  return harness;
}

function trackClient(client: Client): Client {
  clients.push(client);
  return client;
}

afterEach(async () => {
  await Promise.all(
    clients.splice(0).map(async (client) => {
      try {
        await client.close();
      } catch {
        // 已关闭/连接已失败，忽略。
      }
    }),
  );
  await Promise.all(
    harnesses.splice(0).map(async (harness) => {
      await new Promise<void>((resolve) => harness.httpServer.close(() => resolve()));
      await harness.registry.close();
    }),
  );
});

describe("httpAggregator 端到端：tools/list 与 tools/call", () => {
  it("tools/list 返回两后端工具并集（经真实 HTTP + SDK Client）", async () => {
    const harness = await startHarness();
    const client = trackClient(new Client({ name: "test-client", version: "0.0.0" }));
    const transport = new StreamableHTTPClientTransport(harness.baseUrl);
    await client.connect(transport);

    const { tools } = await client.listTools();
    expect(tools.map((tool) => tool.name).sort()).toEqual(
      [
        "vts__echo",
        "vts__add",
        "vts__slow",
        "vts__boom",
        "vts__env",
        "desk__echo",
        "desk__add",
        "desk__slow",
        "desk__boom",
        "desk__env",
      ].sort(),
    );
  });

  it("tools/call 经 HTTP 路由到假后端并原样往返", async () => {
    const harness = await startHarness();
    const client = trackClient(new Client({ name: "test-client", version: "0.0.0" }));
    await client.connect(new StreamableHTTPClientTransport(harness.baseUrl));

    const result = await client.callTool({ name: "desk__add", arguments: { a: 7, b: 8 } });
    expect(result.content).toEqual([{ type: "text", text: "15" }]);

    const echoResult = await client.callTool({
      name: "vts__echo",
      arguments: { value: { nested: true } },
    });
    expect(echoResult.content).toEqual([
      { type: "text", text: JSON.stringify({ value: { nested: true } }) },
    ]);
  });
});

describe("httpAggregator 401 鉴权短路", () => {
  it("设了 token 时：无 Authorization 头 → 401", async () => {
    const harness = await startHarness({}, {}, { enforcedToken: "secret-token" });
    const res = await fetch(harness.baseUrl, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{}",
    });
    expect(res.status).toBe(401);
    const body = (await res.json()) as { error: string };
    expect(body.error).toBe("invalid_token");
    expect(res.headers.get("www-authenticate")).toMatch(/Bearer/);
  });

  it("设了 token 时：错误 token → 401", async () => {
    const harness = await startHarness({}, {}, { enforcedToken: "secret-token" });
    const res = await fetch(harness.baseUrl, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: "Bearer wrong-token" },
      body: "{}",
    });
    expect(res.status).toBe(401);
  });

  it("设了 token 时：正确 token → 鉴权通过，可正常 tools/list", async () => {
    const harness = await startHarness({}, {}, { enforcedToken: "secret-token" });
    const client = trackClient(new Client({ name: "test-client", version: "0.0.0" }));
    const transport = new StreamableHTTPClientTransport(harness.baseUrl, {
      requestInit: { headers: { authorization: "Bearer secret-token" } },
    });
    await client.connect(transport);

    const { tools } = await client.listTools();
    expect(tools.length).toBeGreaterThan(0);
  });
});

describe("httpAggregator 单后端故障降级", () => {
  it("一个后端正常、一个立即崩溃 → 清单只剩健康后端的工具，故障后端调用返回 isError", async () => {
    const harness = await startHarness(
      {},
      { flags: ["--exit-immediately"], startTimeoutMs: 500 },
    );

    await waitUntil(
      () => harness.registry.listTools().every((tool) => tool.name.startsWith("vts__")),
      { timeoutMs: 3000, description: "等待 desktop 侧握手失败后从清单剔除" },
    );

    const client = trackClient(new Client({ name: "test-client", version: "0.0.0" }));
    await client.connect(new StreamableHTTPClientTransport(harness.baseUrl));

    const { tools } = await client.listTools();
    expect(tools.map((tool) => tool.name).sort()).toEqual(
      ["vts__echo", "vts__add", "vts__slow", "vts__boom", "vts__env"].sort(),
    );

    const result = await client.callTool({ name: "desk__echo", arguments: {} });
    expect(result.isError).toBe(true);
    expect((result.content as Array<{ text: string }>)[0]?.text).toMatch(
      /\[aggregator:backend-unavailable\]/,
    );
  });
});
