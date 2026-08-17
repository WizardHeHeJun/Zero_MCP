// 轮询等待某个条件成立（用于等待异步状态机——如 BackendManager 的 status
// 变化——收敛，避免用固定 sleep 时长赌时序）。

export interface WaitUntilOptions {
  timeoutMs?: number;
  intervalMs?: number;
  /** 超时时附加到错误信息里的说明，便于失败时定位是在等什么。 */
  description?: string;
}

export async function waitUntil(
  predicate: () => boolean | Promise<boolean>,
  options: WaitUntilOptions = {},
): Promise<void> {
  const timeoutMs = options.timeoutMs ?? 8000;
  const intervalMs = options.intervalMs ?? 25;
  const deadline = Date.now() + timeoutMs;
  while (!(await predicate())) {
    if (Date.now() >= deadline) {
      const suffix = options.description !== undefined ? `：${options.description}` : "";
      throw new Error(`waitUntil 超时（${timeoutMs}ms）${suffix}`);
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
