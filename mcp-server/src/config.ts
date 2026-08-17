// Zero_MCP · TS 聚合器配置装载：读 process.env → 校验 → AggregatorConfig。
//
// env 域全部收在 ZERO_MCP_AGGREGATOR_* 前缀下（含两个后端子域 *_VTS_* / *_DESKTOP_*），
// 避免与 Python 后端进程自身要读的 env（如 VTS_API_URL、SCREEN_CAPABILITY_ENABLED）
// 同名——后端子进程按「实现自由度」调整点：spawn 时透传聚合器进程的完整 process.env
// （backendManager.ts），若聚合器再用裸 VTS_COMMAND 之类的名字，一旦 Python 侧也读到
// 同名变量会产生隐性耦合，故聚合器自身配置键统一加前缀区分「聚合器控制面」与
// 「后端业务面」两个命名空间。
//
// 校验失败（端口越界 / ARGS 不是字符串数组 JSON / token 非 ASCII / host 非 loopback
// 却未设 token）→ 抛出携带可读中文说明的 Error，供 index.ts 在启动期 fail-fast。

import { z } from "zod";
import { resolveEnforcedToken } from "./auth.js";

/** 单个 stdio 子进程后端的启动参数。 */
export interface BackendSpawnConfig {
  enabled: boolean;
  command: string;
  args: string[];
  cwd: string;
}

/** 聚合器整体配置（loadConfig 的返回类型，已做过全部校验/归一化）。 */
export interface AggregatorConfig {
  enabled: boolean;
  host: string;
  port: number;
  path: string;
  /** resolveEnforcedToken 的结果；null=该 host 免鉴权，非 null=每请求须携带的 Bearer token。 */
  enforcedToken: string | null;
  vts: BackendSpawnConfig;
  desktop: BackendSpawnConfig;
  backendStartTimeoutMs: number;
  backendRestartMax: number;
}

const AggregatorConfigSchema = z.object({
  enabled: z.boolean(),
  host: z.string().min(1),
  port: z.number().int().min(1).max(65535),
  path: z.string().min(1).startsWith("/"),
  enforcedToken: z.string().min(1).nullable(),
  vts: z.object({
    enabled: z.boolean(),
    command: z.string().min(1),
    args: z.array(z.string()),
    cwd: z.string().min(1),
  }),
  desktop: z.object({
    enabled: z.boolean(),
    command: z.string().min(1),
    args: z.array(z.string()),
    cwd: z.string().min(1),
  }),
  backendStartTimeoutMs: z.number().int().positive(),
  backendRestartMax: z.number().int().nonnegative(),
});

/** 读取一个环境变量，缺失返回默认值。 */
function readEnv(key: string, fallback: string): string {
  const raw = process.env[key];
  return raw === undefined || raw === "" ? fallback : raw;
}

/** 解析布尔型环境变量（大小写不敏感，true/1/yes 或 false/0/no）；非法值 fail-fast。 */
function parseBoolEnv(key: string, fallback: boolean): boolean {
  const raw = process.env[key];
  if (raw === undefined || raw === "") {
    return fallback;
  }
  const normalized = raw.trim().toLowerCase();
  if (["true", "1", "yes"].includes(normalized)) {
    return true;
  }
  if (["false", "0", "no"].includes(normalized)) {
    return false;
  }
  throw new Error(`${key}=${raw} 不是合法布尔值（true/false/1/0/yes/no）`);
}

/** 解析整型环境变量；非法值 fail-fast。 */
function parseIntEnv(key: string, fallback: number): number {
  const raw = process.env[key];
  if (raw === undefined || raw === "") {
    return fallback;
  }
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed)) {
    throw new Error(`${key}=${raw} 不是合法整数`);
  }
  return parsed;
}

/** 解析 JSON 数组形式的 stdio 子进程参数列表；非法 JSON 或非字符串数组 fail-fast。 */
function parseArgsEnv(key: string, fallback: string[]): string[] {
  const raw = process.env[key];
  if (raw === undefined || raw === "") {
    return fallback;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (exc) {
    throw new Error(`${key}=${raw} 不是合法 JSON：${String(exc)}`);
  }
  if (!Array.isArray(parsed) || !parsed.every((item): item is string => typeof item === "string")) {
    throw new Error(`${key}=${raw} 须解析为字符串数组`);
  }
  return parsed;
}

/**
 * 装载并校验聚合器配置。
 *
 * 无条件校验（不受 `enabled` 影响）：配置本身的内部一致性（端口范围/ARGS 形状/
 * 鉴权三态）与功能是否启用是两件事——功能关着也不该允许一份自相矛盾的配置静默
 * 通过，等哪天打开 flag 才在生产环境暴雷。
 *
 * @throws {Error} 端口越界 / ARGS 非字符串数组 JSON / token 非 ASCII / host 非
 *   loopback 却未设 token。
 */
export function loadConfig(): AggregatorConfig {
  const host = readEnv("ZERO_MCP_AGGREGATOR_HOST", "127.0.0.1");
  const token = readEnv("ZERO_MCP_AGGREGATOR_TOKEN", "");
  const enforcedToken = resolveEnforcedToken(host, token);

  const raw = {
    enabled: parseBoolEnv("ZERO_MCP_AGGREGATOR_ENABLED", false),
    host,
    port: parseIntEnv("ZERO_MCP_AGGREGATOR_PORT", 8850),
    path: readEnv("ZERO_MCP_AGGREGATOR_PATH", "/mcp"),
    enforcedToken,
    vts: {
      enabled: parseBoolEnv("ZERO_MCP_AGGREGATOR_VTS_BACKEND_ENABLED", true),
      command: readEnv("ZERO_MCP_AGGREGATOR_VTS_COMMAND", "python"),
      args: parseArgsEnv("ZERO_MCP_AGGREGATOR_VTS_ARGS", ["-m", "src.mcp.vts_behavior_mcp_server"]),
      cwd: readEnv("ZERO_MCP_AGGREGATOR_VTS_CWD", "D:\\Zero_MCP"),
    },
    desktop: {
      enabled: parseBoolEnv("ZERO_MCP_AGGREGATOR_DESKTOP_BACKEND_ENABLED", true),
      command: readEnv("ZERO_MCP_AGGREGATOR_DESKTOP_COMMAND", "python"),
      args: parseArgsEnv("ZERO_MCP_AGGREGATOR_DESKTOP_ARGS", ["-m", "src.mcp.desktop_mcp_server"]),
      cwd: readEnv("ZERO_MCP_AGGREGATOR_DESKTOP_CWD", "D:\\Zero_MCP"),
    },
    backendStartTimeoutMs: parseIntEnv("ZERO_MCP_AGGREGATOR_BACKEND_START_TIMEOUT_MS", 15000),
    backendRestartMax: parseIntEnv("ZERO_MCP_AGGREGATOR_BACKEND_RESTART_MAX", 3),
  };

  const result = AggregatorConfigSchema.safeParse(raw);
  if (!result.success) {
    throw new Error(`聚合器配置校验失败：${result.error.message}`);
  }
  return result.data;
}
