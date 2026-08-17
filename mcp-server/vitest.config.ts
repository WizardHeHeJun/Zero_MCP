// vitest 配置：本仓测试大量 spawn 真实 node 子进程（tests/fixtures/fake-backend-server.mjs
// 起的假 stdio MCP 后端），子进程冷启动（node 解释器 + SDK import）本身有几百毫秒开销，
// 叠加握手超时/指数退避重启等用例的显式 sleep，单测默认 5s 超时偏紧——统一放宽。
//
// fileParallelism 关闭：多个测试文件并发时会同时 spawn/kill 多个子进程，在 Windows 上
// 观测到端口/进程句柄的资源竞争会偶发拖慢或抖动 CI（这批测试量不大，串行跑总时长仍可
// 接受，用确定性换速度是划算的）。

import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    testTimeout: 20000,
    hookTimeout: 20000,
    fileParallelism: false,
  },
});
