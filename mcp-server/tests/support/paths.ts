// 测试专用路径解析：假后端 fixture 脚本、mcp-server 根目录。
// 不是测试文件本身（不带 .test. 后缀），vitest 默认 include glob 不会把它当用例跑。

import { fileURLToPath } from "node:url";

/** mcp-server/ 根目录（各 BackendConfig.cwd 默认指向这里）。 */
export const MCP_SERVER_ROOT = fileURLToPath(new URL("../../", import.meta.url));

/** 假 stdio MCP 后端 fixture 脚本的绝对路径（node 可直接执行，见文件内注释的 flag 约定）。 */
export const FAKE_BACKEND_SCRIPT = fileURLToPath(
  new URL("../fixtures/fake-backend-server.mjs", import.meta.url),
);
