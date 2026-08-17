// Zero_MCP · TS 聚合器入口。
//
// 拓扑：外部 host ─(Streamable HTTP, stateless, Bearer 三态)→ 本聚合器
// ─(stdio 子进程 ×2)→ vts_behavior_mcp_server / desktop_mcp_server。
//
// 流程：loadConfig（含 fail-fast 校验）→ ENABLED=false 打印说明后正常退出（exit 0，
// 零副作用）→ ENABLED=true 并发启动已启用后端（任一失败仅 warn，不阻塞其余）→
// 起 HTTP → 装 SIGINT/SIGTERM 优雅关停（先停 HTTP 收新连接，再逐个
// backendManager.close()，再退出）。

import type { Server as HttpServer } from "node:http";
import { BackendRegistry } from "./backend/registry.js";
import type { BackendConfig } from "./backend/types.js";
import { loadConfig } from "./config.js";
import { createHttpTransport } from "./httpTransport.js";
import { logger } from "./logging.js";

function describeError(exc: unknown): string {
  return exc instanceof Error ? exc.message : String(exc);
}

function installShutdownHandlers(httpServer: HttpServer, registry: BackendRegistry): void {
  let shuttingDown = false;
  const shutdown = (signal: NodeJS.Signals): void => {
    if (shuttingDown) {
      return;
    }
    shuttingDown = true;
    logger.info(`收到 ${signal}，开始优雅关停……`);
    void (async () => {
      // 先停 HTTP（不再接受新连接），再收两个后端子进程，最后退出。
      await new Promise<void>((resolve) => httpServer.close(() => resolve()));
      await registry.close();
      logger.info("聚合器已关停。");
      process.exit(0);
    })();
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

async function main(): Promise<void> {
  const config = loadConfig();

  if (!config.enabled) {
    logger.info("ZERO_MCP_AGGREGATOR_ENABLED=false：聚合器未启用，直接退出（零副作用）。");
    return;
  }

  // 提到函数作用域顶层、用 let 声明：registry 一旦 new 出来就可能已经 spawn 了
  // 两个 Python 子进程（BackendRegistry 构造函数虽不 spawn，但紧接着的
  // registry.start() 会）——BLOCK-1 修复：任何后续失败路径（listen 失败/其他
  // 异常）都必须能拿到这个引用去 close()，不能让子进程在 catch 里变成孤儿。
  let registry: BackendRegistry | undefined;
  try {
    const vtsConfig: BackendConfig = {
      id: "vts",
      enabled: config.vts.enabled,
      command: config.vts.command,
      args: config.vts.args,
      cwd: config.vts.cwd,
      startTimeoutMs: config.backendStartTimeoutMs,
      restartMax: config.backendRestartMax,
    };
    const desktopConfig: BackendConfig = {
      id: "desktop",
      enabled: config.desktop.enabled,
      command: config.desktop.command,
      args: config.desktop.args,
      cwd: config.desktop.cwd,
      startTimeoutMs: config.backendStartTimeoutMs,
      restartMax: config.backendRestartMax,
    };
    registry = new BackendRegistry(vtsConfig, desktopConfig);

    // WARN-2 修复：装 SIGINT/SIGTERM 提前到 registry 创建后立即安装，而不是等
    // HTTP listen 成功——registry.start()（最长 BACKEND_START_TIMEOUT_MS，默认
    // 15s）到 listen 之间若收到信号，此前完全没装 handler，会走 Node 默认终止、
    // 不清理子进程。此时 httpServer 尚未起，shutdown 里 `httpServer.close()`
    // 对一个未 listen 的 server 调用是安全的 no-op（不会挂起，直接触发 callback）。
    const httpServer = createHttpTransport(config, registry);
    installShutdownHandlers(httpServer, registry);

    // registry.start() 内部已对两个后端 Promise.allSettled + warn-only，任一失败
    // 不阻塞另一个，也不阻塞后续起 HTTP（拓扑要求：后端不健康时工具清单自动剔除，
    // 不代表聚合器本身不可用）。
    await registry.start();

    await new Promise<void>((resolve, reject) => {
      httpServer.once("error", reject);
      httpServer.listen(config.port, config.host, () => {
        httpServer.removeListener("error", reject);
        resolve();
      });
    });
    logger.info(`聚合器已启动：http://${config.host}:${config.port}${config.path}`);
  } catch (exc) {
    // BLOCK-1：registry.start() 已成功但后续 listen() 失败（如端口被占）这类
    // 路径，两个后端子进程已经 spawn，必须在退出前收尾，不能指望顶层 catch
    // 直接 exit(1) ——那样子进程会变孤儿。
    await registry?.close();
    throw exc;
  }
}

main().catch((exc: unknown) => {
  logger.error(`聚合器启动失败：${describeError(exc)}`);
  process.exit(1);
});
