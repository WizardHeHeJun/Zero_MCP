# mcp-server（TS · MCP 对外聚合层）

Zero_MCP 的 **TS 侧 MCP（Model Context Protocol）服务层**，用官方 [`@modelcontextprotocol/sdk`](https://github.com/modelcontextprotocol/typescript-sdk) 构建。**定位 = 对外互操作边界**：仅当需要向 Zero host 等外部方**聚合**暴露本仓已落地的 Python MCP 能力时启用；不承载任何情感/agent 业务逻辑。

> **v1 已落地**：一个无状态 Streamable HTTP 聚合器，把两个 Python stdio 子进程后端（`vts_behavior_mcp_server` / `desktop_mcp_server`）的工具清单**并集**加前缀后原样转发。默认关（`ZERO_MCP_AGGREGATOR_ENABLED=false`），开发/测试环境需要对外聚合暴露时才开。

## 语言判据（TS = 对外边界，Python = 内部封装）

按「MCP 层语言澄清」（`CLAUDE.md` 脚注 · 判据见 [`notes/2026-07-10-screen-capability-blueprint.md`](../notes/2026-07-10-screen-capability-blueprint.md) §0 AD-1）：内部能力封装的 MCP server 用官方 `mcp` python-sdk 直连自建，不为「MCP 必走 TS」加 child_process 桥。已落地的**真实业务** MCP 实现均在 Python 侧：

- `src/mcp/vts_behavior_mcp_server.py` — VTube Studio Live2D 渲染终端 + 离散行为层（10 工具）
- `src/mcp/desktop_mcp_server.py` — 桌面/屏幕感知与操控能力（10 工具）
- `src/mcp/zero/` — 与原项目 Zero 的互操作层（client、感知注入、expression 消费）

本目录（`mcp-server/`）**只做协议层的聚合与转发**：spawn 两个 Python 后端子进程、把两侧工具清单加前缀合并、按前缀路由 `tools/call`、经 Streamable HTTP 对外暴露。不解释、不改写任何工具的参数/结果语义（见下方「传输层零业务逻辑」）。

## 拓扑

```text
外部 host（Zero 等）
   │  Streamable HTTP（stateless，POST /mcp，Bearer 三态鉴权）
   ▼
TS 聚合器（本目录，node dist/index.js）
   │                                   │
   │ stdio 子进程                       │ stdio 子进程
   ▼                                   ▼
vts_behavior_mcp_server.py         desktop_mcp_server.py
（Python，conda env                （Python，同上）
 affective-expression）
```

- 聚合器进程持有两个 `BackendManager`（各自一个 `StdioClientTransport` + SDK `Client`），启动期并发 `connect()` + `listTools()` 握手；任一后端失败仅 `warn`，不阻塞另一个、也不阻塞 HTTP 起服（后端不健康时其工具从清单自动剔除，聚合器本身仍可用）。
- 对外每个 HTTP 请求各自 `new` 一个聚合 `Server` + `StreamableHTTPServerTransport`（`sessionIdGenerator: undefined`，即**无状态**），请求结束即收尾，不维护跨请求会话。
- 后端子进程 transport 关闭/出错 → 指数退避重连（`1s/2s/4s…`，见下方 env 表）。
- 收到 `SIGINT`/`SIGTERM` → 先停 HTTP（不再接新连接）→ 逐个 `BackendManager.close()`（关闭 transport，终止子进程）→ 退出。⚠ **Windows 平台已知边界**见下方「已知边界」。

## 20 工具命名空间

工具对外 = 两后端工具清单**并集**，各自加前缀区分归属，原始 `inputSchema`/`description`/结果内容**原样透传**（不做 zod 反序列化往返，见 `src/aggregatorServer.ts` 顶部注释：用低层 `Server` 而非 `McpServer` 正是为了避免这一来一回的精度损失风险）：

| 前缀 | 后端 | 工具数 | 工具名 |
| --- | --- | --- | --- |
| `vts__` | `vts_behavior_mcp_server.py` | 10 | `vts__behavior_list` `vts__behavior_trigger` `vts__behavior_interrupt` `vts__behavior_status` `vts__params_list` `vts__params_animate` `vts__params_clear` `vts__speech_play` `vts__vts_connect` `vts__vts_disconnect` |
| `desk__` | `desktop_mcp_server.py` | 10 | `desk__screen_snapshot` `desk__get_uia_tree` `desk__ocr_region` `desk__window_list` `desk__get_capability_flags` `desk__click_element` `desk__type_text` `desk__send_key` `desk__focus_window` `desk__close_window` |

`tools/call` 按前缀剥离路由到对应后端（`desk__get_capability_flags` → 后端收到的是 `get_capability_flags`），调用参数与返回结果原样转发。未知前缀 / 后端不健康 / 转发异常三类**聚合器自身**产生的错误，统一走 `isError=true` 的 `CallToolResult`，错误文案带 `[aggregator:<code>]` 机读令牌（呼应仓内既有 `[desk:*]`/`[vtsb:*]`/`[zero:*]` 令牌约定）；**后端自身**报的错误（如 feature flag 未开）原样透传，不套聚合器的令牌前缀——真机验证确认过：`vts__vts_connect` 在 `VTS_BEHAVIOR_ENABLED=false` 时返回的错误文本是 FastMCP 加壳后的 Python 原文（含 `[vtsb:disabled]`），未被聚合器改写。

## env 变量（与 `.env.example` 一致）

以下变量只被 `mcp-server/` 读取，本仓 Python 侧不读；刻意与后端业务 env（`VTS_API_URL`、`SCREEN_CAPABILITY_ENABLED` 等）区分命名空间——除了避免隐性同名耦合，也是子进程 env 过滤的判据（见下方 spawn 说明：`ZERO_MCP_AGGREGATOR_*` 前缀本身就是"不透传给后端"的边界）。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ZERO_MCP_AGGREGATOR_ENABLED` | `false` | `false`=禁用（零副作用，打印说明后 `exit 0`）；`true` 才起 HTTP + spawn 两后端 |
| `ZERO_MCP_AGGREGATOR_HOST` | `127.0.0.1` | HTTP 监听 host；鉴权三态见下 |
| `ZERO_MCP_AGGREGATOR_PORT` | `8850` | HTTP 监听端口 |
| `ZERO_MCP_AGGREGATOR_PATH` | `/mcp` | HTTP 路径；仅 `POST` 该路径有效（stateless 模式不支持 `GET` SSE 流 / `DELETE` 会话终止） |
| `ZERO_MCP_AGGREGATOR_TOKEN` | 未设 | Bearer token；鉴权三态见下 |
| `ZERO_MCP_AGGREGATOR_VTS_BACKEND_ENABLED` | `true` | 是否 spawn VTS 后端 |
| `ZERO_MCP_AGGREGATOR_VTS_COMMAND` | `python` | VTS 后端解释器命令（真机建议填 conda env 绝对路径，见下方「运行方式」） |
| `ZERO_MCP_AGGREGATOR_VTS_ARGS` | `["-m","src.mcp.vts_behavior_mcp_server"]` | JSON 数组形式的启动参数；非法 JSON / 非字符串数组 fail-fast |
| `ZERO_MCP_AGGREGATOR_VTS_CWD` | `D:\Zero_MCP` | VTS 后端子进程工作目录 |
| `ZERO_MCP_AGGREGATOR_DESKTOP_BACKEND_ENABLED` | `true` | 是否 spawn 桌面后端 |
| `ZERO_MCP_AGGREGATOR_DESKTOP_COMMAND` | `python` | 桌面后端解释器命令 |
| `ZERO_MCP_AGGREGATOR_DESKTOP_ARGS` | `["-m","src.mcp.desktop_mcp_server"]` | 同上 |
| `ZERO_MCP_AGGREGATOR_DESKTOP_CWD` | `D:\Zero_MCP` | 桌面后端子进程工作目录 |
| `ZERO_MCP_AGGREGATOR_BACKEND_START_TIMEOUT_MS` | `15000` | 单后端 spawn + `listTools` 握手超时（毫秒） |
| `ZERO_MCP_AGGREGATOR_BACKEND_RESTART_MAX` | `3` | 指数退避重启上限，见下方语义说明 |

**`BACKEND_RESTART_MAX` 语义（测试角色点名的易错点）**：这是**连续失败次数**的上限，不是总重启次数——**成功重连一次即归零**。例如 `restartMax=3`：连续失败 3 次后进入永久 `unavailable`（须重启聚合器进程）；但若失败 2 次后第 3 次重连成功（转 `healthy`），计数器清零，后续再失败仍从头计满 3 次才耗尽。源码见 `src/backend/backendManager.ts::connectWithTimeout`（成功分支 `this.restartCount = 0`）与 `tests/unit/backendManager.test.ts`（用例名直接写明「restartCount 在成功重连后归零（只计连续失败）」）。

两后端子进程 spawn 时透传聚合器进程的 `process.env`（非 SDK 默认的受限安全清单），因为聚合器与两个后端子进程同属一次本机部署（非跨信任边界）；这意味着后端业务 flag（`VTS_BEHAVIOR_ENABLED`、`VTS_SPEECH_ENABLED`、`SCREEN_CAPABILITY_ENABLED` 等）在**同一个 `.env`/shell 环境**里对聚合器和直连后端同时生效，聚合器不重新声明这些开关。⚠ **不透传聚合器自己的控制面变量**：`ZERO_MCP_AGGREGATOR_*` 前缀（含 `ZERO_MCP_AGGREGATOR_TOKEN`）会被 `buildChildEnv` 过滤掉，不出现在子进程环境里——Python 后端不消费这些键，透传只会白白扩大暴露面（子进程一旦被利用可读到聚合器自己的 Bearer token），故最小化暴露面（`src/backend/backendManager.ts::buildChildEnv`）。

## 鉴权三态

`ZERO_MCP_AGGREGATOR_HOST` + `ZERO_MCP_AGGREGATOR_TOKEN` 的组合决定是否强制鉴权（`src/auth.ts::resolveEnforcedToken`），**口径与 Zero 侧先例一致**（见根 `README.md`「zero-link · Zero MCP Client」一节回指的 Zero `src/mcp_server/auth.py::resolve_enforced_token`，本仓 `src/mcp/zero/` 消费 Zero server 时走的是同一套三态）：

1. **设了 token** → 恒强制鉴权，即便 `host` 是 loopback；token 须纯 ASCII，否则启动期 fail-fast（非 ASCII 密钥经 HTTP 头编码歧义会恒 401，宁可拒绝启动也不留一个永远 401 的死配置）。
2. **未设 token 且 `host` 为 loopback**（`127.0.0.0/8`、`::1`、`localhost`）→ 免鉴权（本机零回归）。「`localhost` 归入 loopback」是相对 Zero 侧 Python 先例的显式扩展，判据同名集合，不是新发明。
3. **未设 token 且 `host` 非 loopback**（如 `0.0.0.0`）→ 启动期 fail-fast，拒绝对外开无鉴权裸端口。

鉴权在 HTTP handler 最外层做（`src/httpTransport.ts`），短路后完全不触达 MCP 层——镜像 Zero 侧 `BearerAuthMiddleware` 包住整个 ASGI app（而非只包某条路由）的做法；401 响应体形状（JSON body + `WWW-Authenticate: Bearer`）也与 Zero `_send_401` 对齐，降低两侧鉴权失败时的排障心智负担。

## 运行方式

```bash
cd mcp-server
npm install         # 首次；锁定依赖版本（package-lock.json 已提交）
npm run build        # tsc -p tsconfig.build.json → dist/
node dist/index.js    # 读 .env（或 shell 已导出的同名变量）
```

Windows 真机验证建议 `*_COMMAND` 指向 conda env 的解释器**绝对路径**（而非裸 `python`，避免落到 `PATH` 上错误的解释器）：

```env
ZERO_MCP_AGGREGATOR_VTS_COMMAND=E:/anaconda/envs/affective-expression/python.exe
ZERO_MCP_AGGREGATOR_DESKTOP_COMMAND=E:/anaconda/envs/affective-expression/python.exe
```

若该路径不存在，退化为 `"E:/anaconda/Scripts/conda.exe" run -n affective-expression python`（`COMMAND` 填 `conda.exe` 绝对路径，`ARGS` 前缀 `["run","-n","affective-expression","python","-m",...]`）。

Python 后端子进程的 `stdout` 是 JSON-RPC 线路，日志走 `stderr`（`src/logging_config.py` 的既有约束）；聚合器自身日志也**恒走 `stderr`**（`src/logging.ts` 禁 `console.log`），即便 HTTP 传输不需要保留 stdout 干净，仍统一不用 stdout，避免未来任何一层不小心接上 stdio 传输时被日志污染。

**开发/构建脚本**：`npm run typecheck`（`tsc --noEmit`）、`npm run lint`（eslint）、`npm run format`（prettier --write）。

## 测试方式

```bash
npm run test    # vitest run
```

- `tests/unit/`：`auth.ts`（鉴权三态纯函数）、`config.ts`（默认值全景 + fail-fast 校验分支）、`backendManager.ts`（握手/超时/指数退避重启/`close()` 优雅收尾）、`registry.ts`（前缀路由/清单合并）。
- `tests/integration/httpAggregator.test.ts`：真实 loopback HTTP（随机端口）+ 真实 SDK `Client`（`StreamableHTTPClientTransport`）+ 两个 fake stdio 子进程后端，覆盖 `tools/list` 并集、`tools/call` 往返、401（未设/错 token）、单后端故障降级；收尾逐一 `close()`，不留悬挂子进程/端口。
- 单测走 `tests/support/fakeBackend.ts` 的 fake stdio 后端（快、无外部依赖）；本 README「已知边界」以外的真实 Python 后端联调结果见集成收尾记录（不入库，由执行任务时的验证脚本产出）。

## 已知边界

- **v1 无 TLS**：HTTP 明文，仅面向本机/受信内网部署（loopback 或显式 token）；跨公网需在更外层加反向代理终止 TLS，聚合器自身不实现。
- **两后端单例共享**：`BackendRegistry` 持有的两个 `BackendManager` 是进程级单例，所有 HTTP 请求共享同一对后端连接（stateless 只体现在 MCP session 层，不代表每请求独立后端连接）；高并发 `tools/call` 会在同一 stdio 连接上串行等待 SDK `Client` 的请求-响应配对，未做后端连接池。
- **`close()` 后 `status` 字段不重置（观察记录）**：`BackendManager.close()` 会清空 `client`/`transport`/`tools` 并终止子进程，但**不**调用 `setStatus(...)`——若在 `close()` 之后直接读 `getState().status`，会看到关停前的最后状态（如 `"healthy"`），而非语义上更准确的「已关停」。功能上不受影响（`callTool` 判定走 `this.client === null` 短路，`close()` 后必然抛「不可用」），但若未来有代码路径直接消费 `status` 字段做展示/告警，需注意这一点。
- **Windows 下外部 `SIGTERM` 可能不触发优雅关停路径**：`index.ts` 注册了 `process.on("SIGTERM", ...)`，但 Node.js 在 Windows 上对 `SIGTERM`/`SIGHUP` 等信号**不做真实的 POSIX 信号语义模拟**（仅 `SIGINT`/`SIGBREAK` 经控制台事件有支持）；真机验证中，从另一 Node 进程 `process.kill(pid, "SIGTERM")` 能终止聚合器进程且两个子进程也随之退出（未见孤儿进程），但日志里**没有**出现 `installShutdownHandlers` 里那行同步的"收到 SIGTERM，开始优雅关停……"，说明该次终止更接近无条件终止而非我们代码里的「先停 HTTP、再逐个 `close()`」路径生效。实操结论：Windows 部署下**进程终止本身可靠**（不会留孤儿子进程），但不应依赖 `SIGTERM` 处理器里的顺序收尾逻辑真的跑到；用 `taskkill /F` 或任务管理器结束整个进程树同样安全。
- **鉴权只做 Bearer 常量比较，无速率限制**：`verifyBearer` 用 `timingSafeEqual` 避免计时旁路，但没有失败重试限流；对外暴露仍建议配合更外层网关做限流。
- **`close()` 不等待在途握手（code-review 复核轮 WARN 留痕）**：`BackendManager.close()` 只关"已确立"的连接；若关停信号精确落在某后端 spawn+握手窗口内（默认最长 15s），`close()` 因 `transport` 尚未赋值而无事可关，随后 `process.exit(0)` 会先于"迟到成功自清理"分支执行，该后端子进程可能成孤儿（可手动 `taskkill` 清理，无数据损坏/持续泄漏）。后续收口方向：`start()` 记录 `connectingPromise`、`close()` 时 `await` 其结算。
- **启动期子进程孤儿问题已在 v1 内收口（不是遗留边界，记录修复口径供排障参考）**：`registry.start()` 成功 spawn 两个后端子进程后，若紧接着 `httpServer.listen()` 失败（如端口被占），`index.ts::main()` 会在 `catch` 里 `await registry.close()` 再重新抛出，确保两个 Python 子进程被收尾，不留孤儿；`SIGINT`/`SIGTERM` handler 也提前到 `registry` 创建后立即安装（而非等 HTTP `listen` 成功），覆盖 `registry.start()`（最长 `BACKEND_START_TIMEOUT_MS`，默认 15s）期间收到信号的窗口。与之对称，`BackendManager.connectWithTimeout()` 的成功分支也会复核 `this.closed`——`close()` 发生在一次连接"在途"（此时 `this.transport` 仍是 `null`，`close()` 拿不到句柄）之后，若该连接后台才成功，会被立即关闭而不是把已关停的 manager 复活成 `healthy`。

## 约定（见 `.claude/rules/`）

- `typescript-code.md`：`tsc` strict、eslint/prettier、无 `any` 兜底。
- `mcp-integration.md`：官方 SDK；传输首选 Streamable HTTP、兼 stdio；**传输层不塞业务逻辑**（业务在 Python `src/*`，本目录全部文件均遵守——`aggregatorServer.ts`/`registry.ts`/`backendManager.ts` 顶部注释都明确写了这条）；跨语言数据形状原样透传（不经 zod 反序列化改写 Python 侧的 `inputSchema`）；把 Zero/后端当外部服务，不直接 import 其代码库。

依赖独立装在本目录（不污染 Python 侧）。endpoint/密钥/后端启动参数全走仓库根 `.env`（勿提交），见 `.env.example` 对应小节。
