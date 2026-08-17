// BackendManager 生命周期测试：用真实假 stdio 子进程（tests/fixtures/fake-backend-server.mjs）
// 覆盖正常握手、握手超时转 unhealthy、崩溃触发有界指数退避重启、close() 优雅收尾
// （清掉待触发的重启 timer，不留悬挂句柄）。
//
// 退避基数经 BackendManagerOptions.restartBackoffBaseMs 注入（产品默认值不变，见
// src/backend/backendManager.ts 顶部注释），让崩溃重启用例能在秒级内跑完。

import { afterEach, describe, expect, it } from "vitest";
import { BackendManager } from "../../src/backend/backendManager.js";
import type { BackendState } from "../../src/backend/types.js";
import { buildFakeBackendConfig } from "../support/fakeBackend.js";
import { waitUntil } from "../support/waitUntil.js";

const managers: BackendManager[] = [];

function track(manager: BackendManager): BackendManager {
  managers.push(manager);
  return manager;
}

afterEach(async () => {
  await Promise.all(managers.splice(0).map((manager) => manager.close()));
});

describe("BackendManager 正常握手", () => {
  it("start() 成功后 healthy，工具清单来自后端 listTools，callTool 原样转发", async () => {
    const manager = track(new BackendManager(buildFakeBackendConfig({ id: "vts" })));
    await manager.start();

    expect(manager.isHealthy()).toBe(true);
    expect(manager.getState()).toMatchObject({ id: "vts", status: "healthy", restartCount: 0 });
    expect(manager.getTools().map((tool) => tool.name).sort()).toEqual([
      "add",
      "boom",
      "echo",
      "env",
      "slow",
    ]);

    const echoResult = await manager.callTool("echo", { value: 42 });
    expect(echoResult.isError).toBeFalsy();
    expect(echoResult.content).toEqual([{ type: "text", text: JSON.stringify({ value: 42 }) }]);

    const addResult = await manager.callTool("add", { a: 2, b: 3 });
    expect(addResult.content).toEqual([{ type: "text", text: "5" }]);
  });

  it("disabled 后端永不 spawn，start() 是 no-op", async () => {
    const manager = track(
      new BackendManager(buildFakeBackendConfig({ id: "desktop", enabled: false })),
    );
    expect(manager.getState().status).toBe("disabled");
    await manager.start();
    expect(manager.getState().status).toBe("disabled");
    expect(manager.getTools()).toEqual([]);
  });
});

describe("BackendManager 握手超时", () => {
  it("后端启动慢于 startTimeoutMs → 转为不健康（restartMax=0 时终态 unavailable）", async () => {
    const manager = track(
      new BackendManager(
        buildFakeBackendConfig({
          id: "vts",
          flags: ["--startup-delay-ms=1200"],
          startTimeoutMs: 250,
          restartMax: 0,
        }),
      ),
    );

    await manager.start();

    expect(manager.isHealthy()).toBe(false);
    expect(manager.getState().status).toBe("unavailable");
    expect(manager.getState().lastError).toMatch(/启动超时/);

    // 给后台"迟到"的 connectOnce（1200ms 延迟）留出时间自行跑完并自我关闭，
    // 避免子进程残留到下一个用例（见 backendManager.ts 对该竞态的处理注释）。
    await new Promise((resolve) => setTimeout(resolve, 1400));
  });
});

describe("BackendManager 崩溃触发有界指数退避重启", () => {
  it("崩溃后自动按退避重连并恢复 healthy；restartCount 在成功重连后归零（只计连续失败）", async () => {
    const observed: BackendState[] = [];
    const manager = track(
      new BackendManager(
        buildFakeBackendConfig({
          id: "desktop",
          // 非零延迟：exit-after-connect-ms=0 与「listTools 握手是否已完成」存在竞态
          // （子进程可能在 listTools 响应送达前就已退出），80ms 足够让首次握手稳定
          // 跑完（进入 healthy），再触发崩溃。
          flags: ["--exit-after-connect-ms=80"],
          startTimeoutMs: 5000,
          restartMax: 5,
        }),
        { restartBackoffBaseMs: 30 },
      ),
    );
    manager.onStateChange((state) => observed.push({ ...state }));

    await manager.start();
    expect(manager.isHealthy()).toBe(true);

    await waitUntil(() => manager.getState().status === "restarting", {
      timeoutMs: 5000,
      description: "等待首次崩溃后进入 restarting",
    });
    await waitUntil(() => manager.getState().status === "healthy", {
      timeoutMs: 5000,
      description: "等待退避后自动重连恢复 healthy",
    });

    // 成功重连即视为「不再连续失败」，restartCount 归零——这是 backendManager.ts
    // 现有的既定语义（restartMax 界定的是「连续失败次数」而非「总重启次数」），
    // 本用例把它钉死为回归断言。
    expect(manager.getState().restartCount).toBe(0);
    expect(observed.some((state) => state.status === "restarting")).toBe(true);
  });

  it("后端持续无法完成握手 → restartCount 累积（不重置），耗尽 restartMax 后进入 unavailable", async () => {
    const observed: BackendState[] = [];
    const manager = track(
      new BackendManager(
        buildFakeBackendConfig({
          id: "desktop",
          // 每次都在握手完成前退出，永不进入 healthy，restartCount 得以连续累积。
          flags: ["--exit-immediately"],
          startTimeoutMs: 5000,
          restartMax: 2,
        }),
        { restartBackoffBaseMs: 30 },
      ),
    );
    manager.onStateChange((state) => observed.push({ ...state }));

    await manager.start();

    await waitUntil(() => manager.getState().status === "unavailable", {
      timeoutMs: 10000,
      description: "等待重启耗尽后进入 unavailable",
    });

    expect(manager.getState().restartCount).toBe(2);
    expect(observed.every((state) => state.status !== "healthy")).toBe(true);
    expect(observed.some((state) => state.status === "restarting")).toBe(true);
  });
});

describe("BackendManager close() 优雅收尾", () => {
  it("close() 清掉待触发的重启 timer：等待期间关停后不会再次尝试重连", async () => {
    const observed: BackendState[] = [];
    const manager = new BackendManager(
      buildFakeBackendConfig({
        id: "vts",
        flags: ["--exit-after-connect-ms=0"],
        startTimeoutMs: 5000,
        restartMax: 10,
      }),
      { restartBackoffBaseMs: 400 },
    );
    manager.onStateChange((state) => observed.push({ ...state }));

    await manager.start();
    // 首次连接后子进程立即退出，manager 应很快进入 "restarting"（等待 400ms 退避）。
    await waitUntil(() => manager.getState().status === "restarting", {
      timeoutMs: 5000,
      description: "等待首次崩溃后进入 restarting",
    });
    const restartCountAtClose = manager.getState().restartCount;

    await manager.close();

    // 若 close() 没清掉 timer，~400ms 后会看到 status 变回 starting/healthy 或
    // restartCount 继续增长；等待超过退避时长后断言两者都没发生。
    await new Promise((resolve) => setTimeout(resolve, 700));
    expect(manager.getState().restartCount).toBe(restartCountAtClose);
    expect(observed.every((state) => state.status !== "healthy" || state.restartCount === 0)).toBe(
      true,
    );
  });

  it("close() 对已 healthy 的连接会关闭 transport，后续 callTool 抛错", async () => {
    const manager = new BackendManager(buildFakeBackendConfig({ id: "desktop" }));
    await manager.start();
    expect(manager.isHealthy()).toBe(true);

    await manager.close();

    await expect(manager.callTool("echo", {})).rejects.toThrow(/不可用/);
  });

  it("close() 可重复调用、无未捕获异常", async () => {
    const manager = new BackendManager(buildFakeBackendConfig({ id: "vts" }));
    await manager.start();
    await manager.close();
    await expect(manager.close()).resolves.toBeUndefined();
  });

  it("close() 发生在 connectWithTimeout 在途期间：不会事后\"复活\"为 healthy，迟到成功的 transport 被关闭", async () => {
    const observed: BackendState[] = [];
    const manager = track(
      new BackendManager(
        buildFakeBackendConfig({
          id: "vts",
          // 足够长的启动延迟，确保 close() 能稳稳落在 connectOnce 仍在途（fixture
          // 内部 sleep 尚未走完、client.connect()/listTools() 均未 resolve）的窗口
          // 内；startTimeoutMs 设得比它更长，保证是 close() 而不是握手超时打断了
          // 这次连接。
          flags: ["--startup-delay-ms=1500"],
          startTimeoutMs: 8000,
          restartMax: 0,
        }),
      ),
    );
    manager.onStateChange((state) => observed.push({ ...state }));

    const startPromise = manager.start();
    // 200ms 远小于 1500ms 的 fixture 启动延迟，此时 connectOnce 必然仍在途
    // （spawn 完成、fixture 还在 sleep，尚未接上 stdio transport）。
    await new Promise((resolve) => setTimeout(resolve, 200));
    expect(manager.getState().status).toBe("starting");
    expect(manager.isHealthy()).toBe(false);

    await manager.close();

    // close() 此刻 this.transport 仍是 null（从未被"当前连接"字段持有过），
    // 优雅关停应仅设 closed=true、不报错、不改写任何连接相关字段。
    expect(manager.isHealthy()).toBe(false);
    expect(manager.getTools()).toEqual([]);

    // 等待原本挂起的 start() 调用把"迟到成功"的 connectOnce 跑完——connectWithTimeout
    // 应检测到 this.closed 并主动关闭这个迟到的 transport，不写入任何实例字段、
    // 也不把状态机翻成 healthy（对称于 catch 分支对"迟到失败"连接的既有处理）。
    await startPromise;

    expect(manager.isHealthy()).toBe(false);
    expect(manager.getTools()).toEqual([]);
    expect(manager.getState().restartCount).toBe(0);
    expect(observed.every((state) => state.status !== "healthy")).toBe(true);

    // callTool 应恒抛错（client 从未被赋值），不应因为后台"迟到"的连接而悄悄变得可用。
    await expect(manager.callTool("echo", {})).rejects.toThrow(/不可用/);
  });
});

describe("BackendManager buildChildEnv 过滤（间接验证：经真实子进程 env 回显）", () => {
  it("聚合器控制面 env（ZERO_MCP_AGGREGATOR_*）不透传给子进程；业务 env 原样透传", async () => {
    // buildChildEnv 未导出（backendManager.ts 内部函数），经由 fixture 新增的
    // "env" 工具回显子进程实际看到的值来间接验证，比孤立单测一个纯函数更贴近
    // 真实行为（真的走了一次 spawn）。
    const originalToken = process.env.ZERO_MCP_AGGREGATOR_TOKEN;
    const originalVtsApiUrl = process.env.VTS_API_URL;
    process.env.ZERO_MCP_AGGREGATOR_TOKEN = "should-not-leak-to-child";
    process.env.VTS_API_URL = "http://example.test/vts";

    try {
      const manager = track(new BackendManager(buildFakeBackendConfig({ id: "vts" })));
      await manager.start();
      expect(manager.isHealthy()).toBe(true);

      const result = await manager.callTool("env", {
        keys: ["ZERO_MCP_AGGREGATOR_TOKEN", "VTS_API_URL"],
      });
      const text = (result.content as Array<{ text: string }>)[0]?.text ?? "{}";
      const seen = JSON.parse(text) as Record<string, string | null>;

      expect(seen.ZERO_MCP_AGGREGATOR_TOKEN).toBeNull();
      expect(seen.VTS_API_URL).toBe("http://example.test/vts");
    } finally {
      if (originalToken === undefined) {
        delete process.env.ZERO_MCP_AGGREGATOR_TOKEN;
      } else {
        process.env.ZERO_MCP_AGGREGATOR_TOKEN = originalToken;
      }
      if (originalVtsApiUrl === undefined) {
        delete process.env.VTS_API_URL;
      } else {
        process.env.VTS_API_URL = originalVtsApiUrl;
      }
    }
  });
});
