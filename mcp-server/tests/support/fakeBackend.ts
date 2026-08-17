// 构造指向 tests/fixtures/fake-backend-server.mjs 的 BackendConfig，供
// backendManager/registry/httpAggregator 测试复用。行为经命令行 flag 控制（见
// fixture 文件头注释），不用 process.env——每个 BackendConfig 独立传参，避免并行
// 测试互相污染全局 env。

import process from "node:process";
import type { BackendConfig, BackendId } from "../../src/backend/types.js";
import { FAKE_BACKEND_SCRIPT, MCP_SERVER_ROOT } from "./paths.js";

export interface FakeBackendOptions {
  id: BackendId;
  /** 追加给 fake-backend-server.mjs 的 flag（如 "--startup-delay-ms=500"）。 */
  flags?: string[];
  enabled?: boolean;
  startTimeoutMs?: number;
  restartMax?: number;
}

/** 构造一个指向假后端 fixture 的 BackendConfig（默认 enabled=true）。 */
export function buildFakeBackendConfig(options: FakeBackendOptions): BackendConfig {
  return {
    id: options.id,
    command: process.execPath,
    args: [FAKE_BACKEND_SCRIPT, ...(options.flags ?? [])],
    cwd: MCP_SERVER_ROOT,
    enabled: options.enabled ?? true,
    startTimeoutMs: options.startTimeoutMs ?? 8000,
    restartMax: options.restartMax ?? 0,
  };
}
