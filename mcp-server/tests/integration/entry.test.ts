// 入口级集成测试：真实 spawn `node dist/index.js`（不复用聚合器内部的 Server/Registry
// 单元，走的是编译后的产物 + 真实进程边界），覆盖 code-review 复核轮点名的 3 个场景：
//   1. ENABLED=false 默认关 → 零副作用（打印说明后 exit 0，不监听端口、不 spawn 后端）。
//   2. listen() 撞 EADDRINUSE（BLOCK-1 修复的入口级验证）→ 非零 exit，已 spawn 的两个
//      假后端子进程被一并收尾，不留孤儿。
//   3. HOST 非 loopback 且未设 TOKEN → loadConfig 阶段 fail-fast，非零 exit，stderr
//      带三态拒绝说明。
//
// 子进程识别用「marker 令牌」技巧：给假后端 fixture 追加一个 fixture 本身不认识、
// 会被安静忽略的 `--test-marker=<uuid>` flag（tests/fixtures/fake-backend-server.mjs
// 的 parseFlags 对未知 flag 只是塞进 flags 对象，不会因未知 flag 报错——因此这里
// **不需要改 fixture**），marker 只会出现在这批测试自己 spawn 的进程命令行里，用
// Windows `Get-CimInstance Win32_Process` 按 CommandLine 是否包含 marker 计数，
// 用来判定「这批子进程当前是否还活着」。选 CIM 而非 `tasklist`：`tasklist` 默认不
// 显示完整命令行，`/V` 也会截断，CIM 的 CommandLine 属性是完整原始值。

import { existsSync } from "node:fs";
import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { createServer, type AddressInfo, type Server as NetServer } from "node:net";
import { randomUUID } from "node:crypto";
import path from "node:path";
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { FAKE_BACKEND_SCRIPT, MCP_SERVER_ROOT } from "../support/paths.js";
import { waitUntil } from "../support/waitUntil.js";

const DIST_INDEX_PATH = path.join(MCP_SERVER_ROOT, "dist", "index.js");
const AGGREGATOR_ENV_PREFIX = "ZERO_MCP_AGGREGATOR_";

/**
 * 编译 src/ → dist/（`npm run build` 的等价物）。
 *
 * 不直接 `spawn("npm", ["run", "build"])`：Windows 上 `npm` 是 `npm.cmd`，要么得
 * `shell: true`、要么得自己找 `.cmd` 后缀，两者都比"直接用 node 跑 typescript 包
 * 自带的 JS 入口"更脆。`node_modules/typescript/bin/tsc` 是纯 JS 脚本，`node <path>`
 * 在任何平台行为一致，且与 package.json 的 `"build": "tsc -p tsconfig.build.json"`
 * 是同一件事。
 */
function buildDist(): void {
  const tscEntry = path.join(MCP_SERVER_ROOT, "node_modules", "typescript", "bin", "tsc");
  const buildConfig = path.join(MCP_SERVER_ROOT, "tsconfig.build.json");
  const result = spawnSync(process.execPath, [tscEntry, "-p", buildConfig], {
    cwd: MCP_SERVER_ROOT,
    encoding: "utf-8",
  });
  if (result.error !== undefined && result.error !== null) {
    throw new Error(`构建 dist/ 失败：无法启动 tsc 进程：${result.error.message}`);
  }
  if (result.status !== 0) {
    throw new Error(
      `构建 dist/ 失败（tsc -p tsconfig.build.json，exit ${String(result.status)}）：\n` +
        `${result.stdout}\n${result.stderr}`,
    );
  }
  if (!existsSync(DIST_INDEX_PATH)) {
    throw new Error(
      `tsc 报告成功但 ${DIST_INDEX_PATH} 不存在——请检查 tsconfig.build.json 的 ` +
        `rootDir/outDir/include 是否仍覆盖 src/index.ts`,
    );
  }
}

beforeAll(() => {
  buildDist();
}, 30000);

/** 生成一个只会出现在本次测试自己 spawn 的进程命令行里的唯一 marker（供后续按命令行识别/计数用）。 */
function randomMarker(): string {
  return `zeromcp_entrytest_${randomUUID().replace(/-/g, "")}`;
}

/**
 * 用 PowerShell `Get-CimInstance Win32_Process` 统计命令行包含 marker 的进程数。
 *
 * marker 恒为 `randomMarker()` 产出的纯字母数字/下划线字符串，不含引号/反引号等
 * PowerShell 字符串字面量特殊字符，故此处的直接字符串拼接是安全的（不接受外部
 * 输入）。
 */
function countProcessesWithMarker(marker: string): Promise<number> {
  return new Promise((resolve, reject) => {
    const psScript =
      `(Get-CimInstance Win32_Process | Where-Object { ` +
      `$_.CommandLine -ne $null -and $_.CommandLine.Contains('${marker}') }).Count`;
    const proc = spawn("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", psScript], {
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    proc.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf-8");
    });
    proc.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf-8");
    });
    proc.on("error", reject);
    proc.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`powershell 查询进程列表失败（exit ${String(code)}）：${stderr}`));
        return;
      }
      const trimmed = stdout.trim();
      resolve(trimmed === "" ? 0 : Number(trimmed));
    });
  });
}

/** 构造指向假后端 fixture、带 marker 的 `*_ARGS` JSON 字符串。 */
function buildFakeBackendArgsJson(marker: string, extraFlags: string[] = []): string {
  return JSON.stringify([FAKE_BACKEND_SCRIPT, ...extraFlags, `--test-marker=${marker}`]);
}

interface AggregatorOutcome {
  code: number | null;
  signal: NodeJS.Signals | null;
  stdout: string;
  stderr: string;
}

/**
 * 构造子进程 env：只读 `process.env`（不写回、不 mutate），过滤掉宿主 shell 里可能
 * 残留的 `ZERO_MCP_AGGREGATOR_*`（避免开发机 .env/shell profile 污染场景判定），
 * 再叠加本次场景显式给的 overrides。测试进程自身的 `process.env` 全程不受影响。
 */
function buildEntryEnv(overrides: Record<string, string>): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (value !== undefined && !key.startsWith(AGGREGATOR_ENV_PREFIX)) {
      env[key] = value;
    }
  }
  return { ...env, ...overrides };
}

const trackedChildren: ChildProcess[] = [];

function trackChild(child: ChildProcess): ChildProcess {
  trackedChildren.push(child);
  return child;
}

afterEach(() => {
  for (const child of trackedChildren.splice(0)) {
    if (child.exitCode === null && child.signalCode === null) {
      child.kill();
    }
  }
});

/** spawn 真实 `node dist/index.js`，收集 stdout/stderr，返回子进程句柄 + 退出结果的 Promise。 */
function spawnAggregator(envOverrides: Record<string, string>): {
  child: ChildProcess;
  result: Promise<AggregatorOutcome>;
} {
  const child = spawn(process.execPath, [DIST_INDEX_PATH], {
    cwd: MCP_SERVER_ROOT,
    env: buildEntryEnv(envOverrides),
    stdio: ["ignore", "pipe", "pipe"],
  });
  let stdout = "";
  let stderr = "";
  child.stdout?.on("data", (chunk: Buffer) => {
    stdout += chunk.toString("utf-8");
  });
  child.stderr?.on("data", (chunk: Buffer) => {
    stderr += chunk.toString("utf-8");
  });
  const result = new Promise<AggregatorOutcome>((resolve) => {
    child.on("close", (code, signal) => {
      resolve({ code, signal, stdout, stderr });
    });
  });
  return { child, result };
}

/** 占住一个 127.0.0.1 上的随机端口，返回句柄与实际分配到的端口号。 */
async function occupyPort(): Promise<{ server: NetServer; port: number }> {
  const server = createServer();
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.removeListener("error", reject);
      resolve();
    });
  });
  const address = server.address() as AddressInfo;
  return { server, port: address.port };
}

describe("入口 main()：ENABLED=false 默认关，零副作用", () => {
  it("打印说明后 exit 0，不监听端口、不 spawn 任何后端子进程", async () => {
    const marker = randomMarker();
    const { child, result } = spawnAggregator({
      ZERO_MCP_AGGREGATOR_ENABLED: "false",
      // 即便两个后端子域自身 enabled=true，顶层 ENABLED=false 应该在 main() 的
      // 最前面短路返回——用带 marker 的假后端配置来证明「真的从未尝试 spawn」，
      // 而不是只靠读源码推断。
      ZERO_MCP_AGGREGATOR_VTS_BACKEND_ENABLED: "true",
      ZERO_MCP_AGGREGATOR_VTS_COMMAND: process.execPath,
      ZERO_MCP_AGGREGATOR_VTS_ARGS: buildFakeBackendArgsJson(marker),
      ZERO_MCP_AGGREGATOR_VTS_CWD: MCP_SERVER_ROOT,
      ZERO_MCP_AGGREGATOR_DESKTOP_BACKEND_ENABLED: "true",
      ZERO_MCP_AGGREGATOR_DESKTOP_COMMAND: process.execPath,
      ZERO_MCP_AGGREGATOR_DESKTOP_ARGS: buildFakeBackendArgsJson(marker),
      ZERO_MCP_AGGREGATOR_DESKTOP_CWD: MCP_SERVER_ROOT,
    });
    trackChild(child);

    const outcome = await result;

    expect(outcome.code).toBe(0);
    expect(outcome.stderr).toMatch(/ZERO_MCP_AGGREGATOR_ENABLED=false/);
    expect(outcome.stderr).toMatch(/聚合器未启用/);

    const lingering = await countProcessesWithMarker(marker);
    expect(lingering).toBe(0);
  });
});

describe("入口 main()：listen() 撞 EADDRINUSE 不留孤儿子进程", () => {
  it(
    "端口已被占用 → 非零 exit code，已 spawn 的两个假后端子进程被一并收尾",
    async () => {
      const { server, port } = await occupyPort();
      const marker = randomMarker();
      try {
        const { child, result } = spawnAggregator({
          ZERO_MCP_AGGREGATOR_ENABLED: "true",
          ZERO_MCP_AGGREGATOR_HOST: "127.0.0.1",
          ZERO_MCP_AGGREGATOR_PORT: String(port),
          ZERO_MCP_AGGREGATOR_VTS_BACKEND_ENABLED: "true",
          ZERO_MCP_AGGREGATOR_VTS_COMMAND: process.execPath,
          // startup-delay-ms：让假后端进程在完成握手前有一段稳定存活窗口，方便
          // 下面的「正对照」轮询稳稳捕捉到它们已经被 spawn（而不是快到测不到）。
          ZERO_MCP_AGGREGATOR_VTS_ARGS: buildFakeBackendArgsJson(marker, ["--startup-delay-ms=1200"]),
          ZERO_MCP_AGGREGATOR_VTS_CWD: MCP_SERVER_ROOT,
          ZERO_MCP_AGGREGATOR_DESKTOP_BACKEND_ENABLED: "true",
          ZERO_MCP_AGGREGATOR_DESKTOP_COMMAND: process.execPath,
          ZERO_MCP_AGGREGATOR_DESKTOP_ARGS: buildFakeBackendArgsJson(marker, [
            "--startup-delay-ms=1200",
          ]),
          ZERO_MCP_AGGREGATOR_DESKTOP_CWD: MCP_SERVER_ROOT,
          ZERO_MCP_AGGREGATOR_BACKEND_START_TIMEOUT_MS: "8000",
          ZERO_MCP_AGGREGATOR_BACKEND_RESTART_MAX: "0",
        });
        trackChild(child);

        // 正对照：先证明两个假后端确实被 spawn 出来了，排除「本来就没起来所以
        // 计数恰好是 0」这种会把测试变成永远绿的假阴性。registry.start() 内部
        // 并发 spawn 两个后端，各自 sleep(1200ms) 才完成握手，这段时间内两个
        // 进程必然存活、可被数到。
        await waitUntil(async () => (await countProcessesWithMarker(marker)) >= 2, {
          timeoutMs: 5000,
          intervalMs: 50,
          description: "等待两个假后端子进程出现（正对照）",
        });

        const outcome = await result;
        expect(outcome.code).not.toBe(0);

        // 主断言：聚合器进程退出后，两个假后端子进程应已被 registry.close()
        // 一并收尾——用轮询而非单次检查，容忍 Windows 上进程真正消失的收尾延迟。
        await waitUntil(async () => (await countProcessesWithMarker(marker)) === 0, {
          timeoutMs: 6000,
          intervalMs: 100,
          description: "等待两个假后端子进程在聚合器退出后被清理干净",
        });
        expect(await countProcessesWithMarker(marker)).toBe(0);
      } finally {
        await new Promise<void>((resolve) => server.close(() => resolve()));
      }
    },
    30000,
  );
});

describe("入口 main()：非 loopback 无 token → fail-fast", () => {
  it("HOST=0.0.0.0 且未设 TOKEN → 非零 exit，stderr 含三态拒绝说明", async () => {
    const { child, result } = spawnAggregator({
      ZERO_MCP_AGGREGATOR_ENABLED: "true",
      ZERO_MCP_AGGREGATOR_HOST: "0.0.0.0",
    });
    trackChild(child);

    const outcome = await result;

    expect(outcome.code).not.toBe(0);
    expect(outcome.stderr).toMatch(/非 loopback/);
    expect(outcome.stderr).toMatch(/未设/);
    expect(outcome.stderr).toMatch(/ZERO_MCP_AGGREGATOR_TOKEN/);
  });
});
