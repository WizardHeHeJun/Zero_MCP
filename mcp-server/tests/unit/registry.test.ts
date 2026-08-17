// BackendRegistry 测试：双假后端合并清单前缀正确、单后端 unhealthy 时清单剔除、
// tools/call 按前缀路由且参数/结果原样、未知前缀/后端不可用返回 isError +
// [aggregator:<code>]、前缀污染工具被跳过且不炸。

import { afterEach, describe, expect, it } from "vitest";
import { BackendRegistry } from "../../src/backend/registry.js";
import { buildFakeBackendConfig } from "../support/fakeBackend.js";
import { waitUntil } from "../support/waitUntil.js";

const registries: BackendRegistry[] = [];

function track(registry: BackendRegistry): BackendRegistry {
  registries.push(registry);
  return registry;
}

afterEach(async () => {
  await Promise.all(registries.splice(0).map((registry) => registry.close()));
});

describe("BackendRegistry 清单合并与前缀", () => {
  it("两个健康后端 → 工具清单是并集，各自加对应前缀", async () => {
    const registry = track(
      new BackendRegistry(
        buildFakeBackendConfig({ id: "vts" }),
        buildFakeBackendConfig({ id: "desktop" }),
      ),
    );
    await registry.start();

    const names = registry.listTools().map((tool) => tool.name).sort();
    expect(names).toEqual(
      [
        "desk__add",
        "desk__boom",
        "desk__echo",
        "desk__env",
        "desk__slow",
        "vts__add",
        "vts__boom",
        "vts__echo",
        "vts__env",
        "vts__slow",
      ].sort(),
    );
  });

  it("单后端 unhealthy（握手失败）→ 清单只剩另一个后端的工具", async () => {
    const registry = track(
      new BackendRegistry(
        buildFakeBackendConfig({ id: "vts", flags: ["--exit-immediately"], startTimeoutMs: 500 }),
        buildFakeBackendConfig({ id: "desktop" }),
      ),
    );
    await registry.start();

    await waitUntil(
      () => registry.listTools().every((tool) => tool.name.startsWith("desk__")),
      { timeoutMs: 3000, description: "等待 vts 侧握手失败后从清单剔除" },
    );

    const names = registry.listTools().map((tool) => tool.name).sort();
    expect(names).toEqual(["desk__add", "desk__boom", "desk__echo", "desk__env", "desk__slow"]);
  });

  it("前缀污染工具被跳过、不影响该后端其余工具正常暴露", async () => {
    const registry = track(
      new BackendRegistry(
        buildFakeBackendConfig({ id: "vts", flags: ["--pollute-prefix=vts__"] }),
        buildFakeBackendConfig({ id: "desktop", enabled: false }),
      ),
    );
    await registry.start();

    const names = registry.listTools().map((tool) => tool.name).sort();
    // 后端把自己的工具起名叫 "vts__polluted"（违规自带前缀）：加前缀后会变成
    // "vts__vts__polluted"——但 rebuildCache 应在"加前缀之前"发现原始名已带已知
    // 前缀，直接跳过，不应该出现在清单里（也不应该崩溃）。
    expect(names).toEqual(["vts__add", "vts__boom", "vts__echo", "vts__env", "vts__slow"]);
    expect(names).not.toContain("vts__polluted");
    expect(names.some((name) => name.includes("polluted"))).toBe(false);
  });
});

describe("BackendRegistry tools/call 路由", () => {
  it("按前缀路由到正确后端，参数与结果原样透传", async () => {
    const registry = track(
      new BackendRegistry(
        buildFakeBackendConfig({ id: "vts" }),
        buildFakeBackendConfig({ id: "desktop" }),
      ),
    );
    await registry.start();

    const echoResult = await registry.callTool("vts__echo", { value: [1, 2, 3] });
    expect(echoResult.isError).toBeFalsy();
    expect(echoResult.content).toEqual([{ type: "text", text: JSON.stringify({ value: [1, 2, 3] }) }]);

    const addResult = await registry.callTool("desk__add", { a: 10, b: 5 });
    expect(addResult.content).toEqual([{ type: "text", text: "15" }]);
  });

  it("后端工具内部抛错（boom）→ 原样转发为 isError CallToolResult（后端自身的错误文案）", async () => {
    const registry = track(new BackendRegistry(buildFakeBackendConfig({ id: "vts" }), buildFakeBackendConfig({ id: "desktop", enabled: false })));
    await registry.start();

    const result = await registry.callTool("vts__boom", {});
    expect(result.isError).toBe(true);
    expect(result.content).toEqual([{ type: "text", text: "[fake:boom] simulated tool failure" }]);
    // 不应被聚合器自己的 [aggregator:*] 令牌包裹——这是后端自身产生的错误，
    // 聚合器只做原样转发。
    expect((result.content[0] as { text: string }).text).not.toMatch(/\[aggregator:/);
  });

  it("未知前缀 → isError + [aggregator:unknown-tool]", async () => {
    const registry = track(
      new BackendRegistry(
        buildFakeBackendConfig({ id: "vts" }),
        buildFakeBackendConfig({ id: "desktop", enabled: false }),
      ),
    );
    await registry.start();

    const result = await registry.callTool("nope__echo", {});
    expect(result.isError).toBe(true);
    expect((result.content[0] as { text: string }).text).toMatch(/\[aggregator:unknown-tool\]/);
  });

  it("后端不可用（未启用）→ isError + [aggregator:backend-unavailable]", async () => {
    const registry = track(
      new BackendRegistry(
        buildFakeBackendConfig({ id: "vts", enabled: false }),
        buildFakeBackendConfig({ id: "desktop", enabled: false }),
      ),
    );
    await registry.start();

    const result = await registry.callTool("vts__echo", {});
    expect(result.isError).toBe(true);
    expect((result.content[0] as { text: string }).text).toMatch(/\[aggregator:backend-unavailable\]/);
  });

  it("后端启用但握手失败（unavailable）→ isError + [aggregator:backend-unavailable]", async () => {
    const registry = track(
      new BackendRegistry(
        buildFakeBackendConfig({ id: "vts", flags: ["--exit-immediately"], startTimeoutMs: 500 }),
        buildFakeBackendConfig({ id: "desktop", enabled: false }),
      ),
    );
    await registry.start();

    await waitUntil(() => registry.listTools().length === 0, {
      timeoutMs: 3000,
      description: "等待 vts 侧握手失败",
    });

    const result = await registry.callTool("vts__echo", {});
    expect(result.isError).toBe(true);
    expect((result.content[0] as { text: string }).text).toMatch(/\[aggregator:backend-unavailable\]/);
  });
});
