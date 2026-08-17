// 两个 stdio 子进程后端（vts / desktop）共享的类型定义。
//
// 传输层零业务逻辑：这里只描述「后端是谁、状态如何、工具怎么加前缀」这类协议/编排
// 层面的结构，不出现任何 VTS/桌面业务语义（工具参数与结果的语义解释留给各自后端）。

import type { Tool } from "@modelcontextprotocol/sdk/types.js";

export type BackendId = "vts" | "desktop";

/** 工具名前缀（拓扑：工具对外 = 两后端并集加前缀）。 */
export const BACKEND_PREFIXES: Record<BackendId, string> = {
  vts: "vts__",
  desktop: "desk__",
};

/** 单个后端的启动/重启参数（由 config.ts 的 BackendSpawnConfig 映射而来）。 */
export interface BackendConfig {
  id: BackendId;
  command: string;
  args: string[];
  cwd: string;
  enabled: boolean;
  /** 首次 spawn + listTools 握手的超时（毫秒）。 */
  startTimeoutMs: number;
  /** 指数退避重启的次数上限；耗尽后进入 "unavailable" 且不再重试。 */
  restartMax: number;
}

export type BackendStatus =
  // 未启用（对应 config 里 enabled=false），永不 spawn。
  | "disabled"
  // 正在 spawn + 握手（listTools 尚未成功）。
  | "starting"
  // 已连接且 listTools 成功，可路由 tools/call。
  | "healthy"
  // transport 关闭/出错后正在指数退避重启。
  | "restarting"
  // 重启次数耗尽，永久不可用（除非进程重启聚合器）。
  | "unavailable";

/** 后端当前状态快照（backendManager 持有，经回调通知 registry）。 */
export interface BackendState {
  id: BackendId;
  status: BackendStatus;
  lastError: string | undefined;
  restartCount: number;
}

/** 一个已加前缀、可对外暴露的工具：保留原始后端归属，供 registry 路由 tools/call。 */
export interface NamespacedTool {
  backendId: BackendId;
  originalName: string;
  /** 对外暴露的 Tool（name 字段已替换为 `<prefix><originalName>`，其余字段原样透传）。 */
  tool: Tool;
}

/** 后端状态变化回调（backendManager → registry，用于触发清单重建）。 */
export type BackendStateListener = (state: BackendState) => void;
