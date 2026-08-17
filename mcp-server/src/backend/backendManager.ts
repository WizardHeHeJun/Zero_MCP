// 单后端生命周期管理：StdioClientTransport spawn + Client.connect + listTools 握手
// → health 状态机 → transport close/error 时指数退避重启（1s/2s/4s，上限
// BACKEND_RESTART_MAX，耗尽后永久 unavailable）→ close() 优雅收尾。
//
// 传输层零业务逻辑：本文件只做进程/连接生命周期编排，callTool 的参数与结果原样
// 转发（不解释、不改写），listTools 的结果原样缓存供 registry 加前缀。

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import type { CallToolResult, Tool } from "@modelcontextprotocol/sdk/types.js";
import { logger } from "../logging.js";
import type { BackendConfig, BackendState, BackendStateListener, BackendStatus } from "./types.js";

const RESTART_BACKOFF_BASE_MS = 1000;

interface ConnectResult {
  client: Client;
  transport: StdioClientTransport;
  tools: Tool[];
}

/**
 * 可选构造参数（生产路径不传，走默认值——仅为测试注入用）。
 *
 * ⚠ 实现自由度调整：指数退避的基数原是模块级常量，测试若要在秒级内验证多轮重启/
 * 耗尽 restartMax 的行为，1s/2s/4s 的真实退避会把用例拖到分钟级。加一个可选构造
 * 参数覆盖基数，产品默认值（`RESTART_BACKOFF_BASE_MS` = 1000ms）不变、`index.ts`
 * 等生产调用点不受影响。
 */
export interface BackendManagerOptions {
  restartBackoffBaseMs?: number;
}

/** 把可能非 Error 的异常统一转成可读字符串（用于 lastError 记录与日志）。 */
function describeError(exc: unknown): string {
  return exc instanceof Error ? exc.message : String(exc);
}

/** 聚合器自己的控制面 env 前缀——见 buildChildEnv 的过滤理由。 */
const AGGREGATOR_CONTROL_PLANE_ENV_PREFIX = "ZERO_MCP_AGGREGATOR_";

/**
 * 构造子进程环境变量：透传聚合器进程的业务 env，过滤掉聚合器自己的控制面 env。
 *
 * ⚠ 实现自由度调整：不用 StdioClientTransport 的默认行为——SDK 未显式传 `env` 时走
 * `getDefaultEnvironment()`，那是一份出于安全考虑的**受限安全清单**（不含
 * SCREEN_CAPABILITY_ENABLED / VTS_API_URL / ANTHROPIC_API_KEY 等业务 env），会让
 * Python 后端在子进程里读不到自己的配置。聚合器与两个后端本就同属一个受信部署
 * （本机 stdio 子进程，非跨信任边界），故显式透传聚合器进程的业务 env。
 *
 * WARN-1 修复：`ZERO_MCP_AGGREGATOR_*`（含 `ZERO_MCP_AGGREGATOR_TOKEN` 等控制面
 * 变量）**不**透传——Python 后端不消费这些键，白白带过去只是扩大暴露面（子进程
 * 一旦被利用可读到聚合器自己的 Bearer token，即便同属受信部署也不该"顺手"给）。
 * 业务 env（`VTS_API_URL`/`SCREEN_CAPABILITY_ENABLED`/`ANTHROPIC_API_KEY` 等，
 * 均不带这个前缀）透传不变。
 */
function buildChildEnv(): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (value !== undefined && !key.startsWith(AGGREGATOR_CONTROL_PLANE_ENV_PREFIX)) {
      env[key] = value;
    }
  }
  return env;
}

export class BackendManager {
  private readonly config: BackendConfig;
  private readonly restartBackoffBaseMs: number;
  private readonly listeners: Set<BackendStateListener> = new Set();
  private client: Client | null = null;
  private transport: StdioClientTransport | null = null;
  private tools: Tool[] = [];
  private status: BackendStatus;
  private restartCount = 0;
  private lastError: string | undefined;
  private restartTimer: NodeJS.Timeout | null = null;
  private closed = false;

  constructor(config: BackendConfig, options: BackendManagerOptions = {}) {
    this.config = config;
    this.restartBackoffBaseMs = options.restartBackoffBaseMs ?? RESTART_BACKOFF_BASE_MS;
    this.status = config.enabled ? "starting" : "disabled";
  }

  /** 注册状态变化回调（registry 用它来触发清单重建）。 */
  onStateChange(listener: BackendStateListener): void {
    this.listeners.add(listener);
  }

  getState(): BackendState {
    return {
      id: this.config.id,
      status: this.status,
      lastError: this.lastError,
      restartCount: this.restartCount,
    };
  }

  /** 当前缓存的工具清单（未加前缀，registry 负责加前缀）；不健康时为空数组。 */
  getTools(): readonly Tool[] {
    return this.tools;
  }

  isHealthy(): boolean {
    return this.status === "healthy";
  }

  /** 启动后端：未启用直接跳过（永不 spawn）；启用则发起首次连接。 */
  async start(): Promise<void> {
    if (!this.config.enabled) {
      logger.info(`后端 ${this.config.id} 未启用（enabled=false），跳过启动`);
      return;
    }
    await this.connectWithTimeout();
  }

  /** 转发一次 tools/call 到本后端；后端不健康时抛错（由 registry 转成结构化 MCP 错误）。 */
  async callTool(name: string, args: unknown): Promise<CallToolResult> {
    if (this.client === null || this.status !== "healthy") {
      throw new Error(`后端 ${this.config.id} 当前不可用（status=${this.status}）`);
    }
    const result = await this.client.callTool({
      name,
      arguments: args as Record<string, unknown> | undefined,
    });
    return result as CallToolResult;
  }

  /** 优雅关停：取消待触发的重启定时器，关闭 transport（终止子进程）。 */
  async close(): Promise<void> {
    this.closed = true;
    if (this.restartTimer !== null) {
      clearTimeout(this.restartTimer);
      this.restartTimer = null;
    }
    const transport = this.transport;
    this.client = null;
    this.transport = null;
    this.tools = [];
    if (transport !== null) {
      try {
        await transport.close();
      } catch (exc) {
        logger.warn(`后端 ${this.config.id} 关闭 transport 时出错：${describeError(exc)}`);
      }
    }
  }

  private async connectOnce(): Promise<ConnectResult> {
    const transport = new StdioClientTransport({
      command: this.config.command,
      args: this.config.args,
      cwd: this.config.cwd,
      env: buildChildEnv(),
    });
    const client = new Client({ name: `zero-mcp-aggregator-${this.config.id}`, version: "0.0.0" });
    // 用闭包捕获的 transport 做身份核对：握手超时后台跑完 / 已被 close() 收尾的
    // transport，其 onclose/onerror 不应再驱动"当前"状态机（见 connectWithTimeout
    // 与 close() 的处理）。
    transport.onclose = () => {
      if (this.transport === transport) {
        this.handleTransportClosed();
      }
    };
    transport.onerror = (error: Error) => {
      if (this.transport === transport) {
        this.handleTransportError(error);
      }
    };
    await client.connect(transport);
    const listed = await client.listTools();
    return { client, transport, tools: listed.tools };
  }

  private async connectWithTimeout(): Promise<void> {
    if (this.closed) {
      return;
    }
    this.setStatus("starting");
    const timeoutMs = this.config.startTimeoutMs;
    const connectPromise = this.connectOnce();
    let timer: NodeJS.Timeout | undefined;
    const timeoutPromise = new Promise<never>((_resolve, reject) => {
      timer = setTimeout(() => {
        reject(new Error(`后端 ${this.config.id} 启动超时（${timeoutMs}ms）`));
      }, timeoutMs);
    });
    try {
      const result = await Promise.race([connectPromise, timeoutPromise]);
      if (this.closed) {
        // BLOCK-2：close() 发生在本次连接在途时（当时 this.transport 仍是
        // null，close() 拿不到句柄、直接返回），随后 connectOnce 才成功——不能
        // 把已经关停的 manager"复活"成 healthy，否则这个刚建立的 transport/
        // 子进程永远不会被 close() 收尾。对称于下面 catch 分支对"迟到成功"的
        // 处理：直接关掉这次连接，不写入任何实例字段。
        await result.transport.close().catch(() => undefined);
        return;
      }
      this.client = result.client;
      this.transport = result.transport;
      this.tools = result.tools;
      this.restartCount = 0;
      this.setStatus("healthy");
    } catch (exc) {
      this.lastError = describeError(exc);
      logger.warn(`后端 ${this.config.id} 连接失败：${this.lastError}`);
      // 若败因是握手超时，connectOnce 可能仍在后台跑完并悄悄建立连接——一旦建立
      // 就主动关闭，避免半开的 transport/子进程泄漏；此刻状态机已经走 restarting/
      // unavailable 分支，不把这次迟到的连接当作"当前"连接。
      void connectPromise.then((result) => result.transport.close()).catch(() => undefined);
      this.scheduleRestart();
    } finally {
      if (timer !== undefined) {
        clearTimeout(timer);
      }
    }
  }

  private handleTransportClosed(): void {
    if (this.closed) {
      return;
    }
    logger.warn(`后端 ${this.config.id} 连接关闭，准备重连`);
    this.client = null;
    this.transport = null;
    this.tools = [];
    this.scheduleRestart();
  }

  private handleTransportError(error: Error): void {
    if (this.closed) {
      return;
    }
    this.lastError = error.message;
    logger.error(`后端 ${this.config.id} 传输错误：${error.message}`);
  }

  private scheduleRestart(): void {
    if (this.closed) {
      return;
    }
    if (this.restartCount >= this.config.restartMax) {
      this.setStatus("unavailable");
      logger.error(
        `后端 ${this.config.id} 重启 ${this.restartCount} 次后仍失败，标记为永久不可用（须重启聚合器进程）`,
      );
      return;
    }
    this.setStatus("restarting");
    const delayMs = this.restartBackoffBaseMs * 2 ** this.restartCount;
    this.restartCount += 1;
    this.restartTimer = setTimeout(() => {
      this.restartTimer = null;
      void this.connectWithTimeout();
    }, delayMs);
  }

  private setStatus(status: BackendStatus): void {
    this.status = status;
    const state = this.getState();
    for (const listener of this.listeners) {
      listener(state);
    }
  }
}
