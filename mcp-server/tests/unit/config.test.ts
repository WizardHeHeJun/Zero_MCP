// config.ts 测试：默认值全景 + fail-fast 校验分支（端口越界/ARGS 非法 JSON/token
// 非 ASCII/ENABLED=false 时校验仍生效）。loadConfig 直接读 process.env，测试隔离
// 靠每个用例前后备份/恢复 process.env（含清空本次用例可能残留的 ZERO_MCP_AGGREGATOR_*
// 键），避免用例间串味、也避免真实开发机上的 .env 泄进用例。

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { loadConfig } from "../../src/config.js";

const ENV_PREFIX = "ZERO_MCP_AGGREGATOR_";
let envSnapshot: NodeJS.ProcessEnv;

function clearAggregatorEnv(): void {
  for (const key of Object.keys(process.env)) {
    if (key.startsWith(ENV_PREFIX)) {
      delete process.env[key];
    }
  }
}

beforeEach(() => {
  envSnapshot = { ...process.env };
  clearAggregatorEnv();
});

afterEach(() => {
  for (const key of Object.keys(process.env)) {
    if (!(key in envSnapshot)) {
      delete process.env[key];
    }
  }
  Object.assign(process.env, envSnapshot);
});

describe("loadConfig 默认值", () => {
  it("无任何 ZERO_MCP_AGGREGATOR_* env 时返回完整默认值", () => {
    const config = loadConfig();
    expect(config).toEqual({
      enabled: false,
      host: "127.0.0.1",
      port: 8850,
      path: "/mcp",
      enforcedToken: null,
      vts: {
        enabled: true,
        command: "python",
        args: ["-m", "src.mcp.vts_behavior_mcp_server"],
        cwd: "D:\\Zero_MCP",
      },
      desktop: {
        enabled: true,
        command: "python",
        args: ["-m", "src.mcp.desktop_mcp_server"],
        cwd: "D:\\Zero_MCP",
      },
      backendStartTimeoutMs: 15000,
      backendRestartMax: 3,
    });
  });
});

describe("loadConfig 无条件校验（不受 enabled 影响）", () => {
  it("ENABLED=false 时校验仍生效：非 loopback 无 token 照样 fail-fast", () => {
    process.env.ZERO_MCP_AGGREGATOR_ENABLED = "false";
    process.env.ZERO_MCP_AGGREGATOR_HOST = "0.0.0.0";
    expect(() => loadConfig()).toThrow(/非 loopback/);
  });

  it("host 非 loopback 但设了 token → 正常通过", () => {
    process.env.ZERO_MCP_AGGREGATOR_HOST = "0.0.0.0";
    process.env.ZERO_MCP_AGGREGATOR_TOKEN = "abc123";
    const config = loadConfig();
    expect(config.enforcedToken).toBe("abc123");
    expect(config.host).toBe("0.0.0.0");
  });
});

describe("loadConfig 端口越界", () => {
  it("PORT=0 → 抛错", () => {
    process.env.ZERO_MCP_AGGREGATOR_PORT = "0";
    expect(() => loadConfig()).toThrow(/校验失败/);
  });

  it("PORT=65536 → 抛错", () => {
    process.env.ZERO_MCP_AGGREGATOR_PORT = "65536";
    expect(() => loadConfig()).toThrow(/校验失败/);
  });

  it("PORT 非整数（8850.5）→ 抛错（parseIntEnv 阶段即拒）", () => {
    process.env.ZERO_MCP_AGGREGATOR_PORT = "8850.5";
    expect(() => loadConfig()).toThrow(/不是合法整数/);
  });
});

describe("loadConfig ARGS 非法 JSON", () => {
  it("VTS_ARGS 非法 JSON → 抛错", () => {
    process.env.ZERO_MCP_AGGREGATOR_VTS_ARGS = "not-json{";
    expect(() => loadConfig()).toThrow(/不是合法 JSON/);
  });

  it("DESKTOP_ARGS 合法 JSON 但非字符串数组 → 抛错", () => {
    process.env.ZERO_MCP_AGGREGATOR_DESKTOP_ARGS = "[1,2,3]";
    expect(() => loadConfig()).toThrow(/须解析为字符串数组/);
  });

  it("VTS_ARGS 合法字符串数组 JSON → 正常覆盖默认值", () => {
    process.env.ZERO_MCP_AGGREGATOR_VTS_ARGS = '["--foo", "bar"]';
    const config = loadConfig();
    expect(config.vts.args).toEqual(["--foo", "bar"]);
  });
});

describe("loadConfig TOKEN 非 ASCII", () => {
  it("TOKEN 非 ASCII → 抛错（resolveEnforcedToken 先于 zod 抛出）", () => {
    process.env.ZERO_MCP_AGGREGATOR_TOKEN = "秘密令牌";
    expect(() => loadConfig()).toThrow(/纯 ASCII/);
  });
});

describe("loadConfig 其余字段校验", () => {
  it("ENABLED 非法布尔值 → 抛错", () => {
    process.env.ZERO_MCP_AGGREGATOR_ENABLED = "maybe";
    expect(() => loadConfig()).toThrow(/不是合法布尔值/);
  });

  it("PATH 不以 / 开头 → 抛错", () => {
    process.env.ZERO_MCP_AGGREGATOR_PATH = "mcp";
    expect(() => loadConfig()).toThrow(/校验失败/);
  });

  it("BACKEND_RESTART_MAX 为负数 → 抛错", () => {
    process.env.ZERO_MCP_AGGREGATOR_BACKEND_RESTART_MAX = "-1";
    expect(() => loadConfig()).toThrow(/校验失败/);
  });

  it("BACKEND_START_TIMEOUT_MS 为 0 → 抛错（须为正数）", () => {
    process.env.ZERO_MCP_AGGREGATOR_BACKEND_START_TIMEOUT_MS = "0";
    expect(() => loadConfig()).toThrow(/校验失败/);
  });
});
