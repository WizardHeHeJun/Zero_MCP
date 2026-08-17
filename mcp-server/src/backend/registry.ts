// 持两个 BackendManager（vts / desktop）：
//   - 合并清单：加前缀缓存，后端状态变化（healthy/unhealthy 切换）触发重建；
//     不健康的后端从清单剔除（拓扑：tools/list 用连接期/重连期缓存）。
//   - tools/call：剥前缀路由到对应后端，原样转发参数与结果。
//
// 传输层零业务逻辑：本文件只做「前缀 ⇄ 后端」的路由与清单合并，不解释任何工具
// 参数/结果的业务语义。

import type { CallToolResult, Tool } from "@modelcontextprotocol/sdk/types.js";
import { logger } from "../logging.js";
import { BackendManager } from "./backendManager.js";
import { BACKEND_PREFIXES } from "./types.js";
import type { BackendConfig, BackendId, NamespacedTool } from "./types.js";

/** 把可能非 Error 的异常统一转成可读字符串（同 backendManager.ts 的小工具，故意不共享
 * 一个跨文件的私有 helper——两处都很小，拆出去反而多一层间接）。 */
function describeError(exc: unknown): string {
  return exc instanceof Error ? exc.message : String(exc);
}

/**
 * 构造 isError=true 的 CallToolResult。
 *
 * 实现自由度：未知前缀 / 后端不可用 / 后端调用失败三类聚合器自身产生的错误，
 * 统一走 **isError CallToolResult**，不走协议级 McpError——理由：这些是「某次
 * 具体 tools/call 执行失败」，与 Python 后端把 ToolError 报成 isError 内容
 * （见 desktop_mcp_server.py 的 `[desk:*]` / vts 侧 `[vtsb:*]` 令牌约定）同一
 * 语义层级，让调用方（LLM）能看到并推理错误内容；协议级错误留给"请求本身不合法"
 * 这类场景（如 tools/call 的 name 字段缺失，SDK 已在更底层处理）。错误文案统一带
 * `[aggregator:<code>]` 机读令牌，呼应仓内既有 `[desk:*]`/`[vtsb:*]`/`[zero:*]`
 * 令牌约定，供未来的消费方 `re.search`/`.includes` 定位，不依赖前缀匹配。
 */
function errorResult(token: string, message: string): CallToolResult {
  return {
    content: [{ type: "text", text: `[aggregator:${token}] ${message}` }],
    isError: true,
  };
}

interface ResolvedTool {
  backendId: BackendId;
  originalName: string;
}

export class BackendRegistry {
  private readonly managers: Record<BackendId, BackendManager>;
  private cachedTools: NamespacedTool[] = [];

  constructor(vtsConfig: BackendConfig, desktopConfig: BackendConfig) {
    this.managers = {
      vts: new BackendManager(vtsConfig),
      desktop: new BackendManager(desktopConfig),
    };
    for (const manager of Object.values(this.managers)) {
      manager.onStateChange(() => this.rebuildCache());
    }
  }

  /** 并发启动已启用的后端；任一失败仅 warn，不阻塞其余后端（拓扑要求）。 */
  async start(): Promise<void> {
    const entries = Object.entries(this.managers) as [BackendId, BackendManager][];
    const results = await Promise.allSettled(entries.map(([, manager]) => manager.start()));
    results.forEach((result, index) => {
      if (result.status === "rejected") {
        const [id] = entries[index] as [BackendId, BackendManager];
        logger.warn(`后端 ${id} 启动失败（不阻塞其余后端）：${describeError(result.reason)}`);
      }
    });
    this.rebuildCache();
  }

  /** 优雅关停两个后端（逐个 close，任一失败不阻塞另一个）。 */
  async close(): Promise<void> {
    await Promise.allSettled(Object.values(this.managers).map((manager) => manager.close()));
  }

  /** 对外暴露的工具清单（连接期/重连期缓存，不健康后端已剔除）。 */
  listTools(): Tool[] {
    return this.cachedTools.map((entry) => entry.tool);
  }

  /** 按前缀路由并转发一次 tools/call；错误一律走 isError CallToolResult（见上方注释）。 */
  async callTool(namespacedName: string, args: unknown): Promise<CallToolResult> {
    const resolved = this.resolve(namespacedName);
    if (resolved === null) {
      return errorResult(
        "unknown-tool",
        `未知工具：${namespacedName}（须以已注册后端前缀开头，如 vts__/desk__）`,
      );
    }
    const manager = this.managers[resolved.backendId];
    if (!manager.isHealthy()) {
      return errorResult("backend-unavailable", `后端 ${resolved.backendId} 当前不可用`);
    }
    try {
      return await manager.callTool(resolved.originalName, args);
    } catch (exc) {
      // healthy 判定与真正转发之间的罕见竞态（如判定后紧接着 transport 关闭）；
      // 同样按 isError 上报，不让聚合器自身的编排失败升级成协议级错误。
      return errorResult(
        "backend-call-failed",
        `后端 ${resolved.backendId} 调用失败：${describeError(exc)}`,
      );
    }
  }

  private resolve(namespacedName: string): ResolvedTool | null {
    for (const [id, prefix] of Object.entries(BACKEND_PREFIXES) as [BackendId, string][]) {
      if (namespacedName.startsWith(prefix)) {
        return { backendId: id, originalName: namespacedName.slice(prefix.length) };
      }
    }
    return null;
  }

  private rebuildCache(): void {
    const merged: NamespacedTool[] = [];
    for (const [id, manager] of Object.entries(this.managers) as [BackendId, BackendManager][]) {
      if (!manager.isHealthy()) {
        continue; // 拓扑要求：后端不健康即从清单剔除
      }
      const prefix = BACKEND_PREFIXES[id];
      for (const tool of manager.getTools()) {
        if (this.hasKnownPrefix(tool.name)) {
          // 启动时断言：原始工具名不应已带 vts__/desk__ 前缀，否则加前缀后会与
          // 路由方案本身冲突（resolve() 无法区分"后端 A 的原始名恰好长这样"和
          // "这是后端 B 加过前缀的名字"）。命中即视为后端实现违规，跳过该工具、
          // 大声记日志，而不是静默暴露一个会被错误路由的工具名。
          logger.error(`后端 ${id} 的工具名 ${tool.name} 已带已知前缀，违反命名约定，跳过暴露`);
          continue;
        }
        merged.push({
          backendId: id,
          originalName: tool.name,
          tool: { ...tool, name: `${prefix}${tool.name}` },
        });
      }
    }
    this.cachedTools = merged;
  }

  private hasKnownPrefix(name: string): boolean {
    return Object.values(BACKEND_PREFIXES).some((prefix) => name.startsWith(prefix));
  }
}
