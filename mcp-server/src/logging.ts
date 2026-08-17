// 极薄 stderr-only logger（对齐 Python 侧 src/logging_config.py 的约束：console 恒
// stderr——stdio 子进程的 stdout 是 JSON-RPC 线路，本聚合器自身走 HTTP，但仍统一
// 不用 stdout，避免未来任何一层不小心接上 stdio 传输时被日志污染）。
//
// 禁 console.log（会走 stdout）；只用 console.error 写 stderr。不引入第三方日志库
// ——聚合器体量小，标准库足够，避免给互操作边界层加不必要的依赖面。

export type LogLevel = "info" | "warn" | "error";

function format(level: LogLevel, message: string): string {
  const timestamp = new Date().toISOString();
  return `${timestamp} [aggregator] [${level}] ${message}`;
}

export const logger = {
  info(message: string, ...args: unknown[]): void {
    console.error(format("info", message), ...args);
  },
  warn(message: string, ...args: unknown[]): void {
    console.error(format("warn", message), ...args);
  },
  error(message: string, ...args: unknown[]): void {
    console.error(format("error", message), ...args);
  },
};
