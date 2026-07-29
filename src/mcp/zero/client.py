"""Zero MCP Client（Python 侧）。

生命周期：async context manager（AsyncExitStack 嵌套 transport + ClientSession）。
工具面：open_session / step / close_session / graceful_step 四个 async 方法。

设计约束（蓝图 Task 1-3）：
- 不 import Zero 代码库；经 call_tool 字符串工具名调用（AD-2）。
- 传输层零业务逻辑：异常封装 + 工具转发，情感/agent 逻辑在 Python src/* 层。
- 跨语言契约：从 src/agents/models/zero_affect（共享契约层）import 数据形状。
- ZERO_LINK_ENABLED=false 时拒绝连接（双侧 flag 检查）；新能力默认关。
- env 传递：StdioServerParameters.env 注入完整 os.environ 副本（同 desktop 侧），
  确保子进程能解析项目 src 包（PYTHONPATH / conda PATH 等由 conda 配置）。
- session_id 不由 client 持有（无状态句柄，可服务多会话）。
- Windows ProactorEventLoop（Python 3.12 默认）：anyio 在 Windows 上用 asyncio backend，
  stdio_client 内部的 create_windows_process 已处理兼容性。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import secrets
import sys
import types
from typing import Any

import httpx
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError
from mcp.types import TextContent
from pydantic import ValidationError

from src.agents.models.zero_affect import AffectStimulus, ExpressionBundle, ModalityPrior
from src.mcp.zero.external_priors import build_external_priors_override

logger = logging.getLogger(__name__)

# ── Zero 机读错误码（zero-link 跨仓契约·2026-07-29 换代）───────────────────────
# Zero server 的 ToolError 文案带**位置不敏感**令牌 `[zero:<code>]`（ASCII kebab-case，
# 全文恰出现一次，位置不限），由 `src/mcp_server/server.py::_tool_error(code, message)` 构造。
#
# 🛑 为什么必须位置无关、不能用位置 0 的裸前缀（旧实现的致命缺陷，2026-07-29 两侧实证）：
#   FastMCP 在**工具层**统一加壳——`mcp/server/fastmcp/tools/base.py::Tool.run` 的
#   `except Exception as e: raise ToolError(f"Error executing tool {self.name}: {e}")`
#   （ToolError 继承 Exception，自己也被这一支重新包一层）。⇒ wire 上的真实文本是
#     "Error executing tool zero.step: <Zero 原文>"
#   本仓 stdio 直连 D:\Zero `src.mcp_server` 实测（mcp SDK 见 environment）：
#     text = "Error executing tool zero.step: [zero:unknown-session] 未知 session_id='bogus-…'；…"
#     text.lstrip().startswith("unknown-session")      -> False   ← 旧判据恒 False
#     re.search(r"\[zero:([a-z][a-z0-9-]*)\]", text)   -> "unknown-session"
#   故旧判定（`startswith(_UNKNOWN_SESSION_MARKER)`）对真 server **恒不命中**，
#   T6·④ 的 resume 重试通路曾是**生产死码**；两侧旧单测都喂**未加壳**夹具，故长期假绿。
#   → 本仓夹具一律改用**真 wire 形态**（带 "Error executing tool <name>: " 外壳），
#     见 tests/mcp/test_zero_client.py::_wire 的注释。
#
# 码值按**符号名**与 Zero `src/mcp_server/server.py` 的 `ZERO_ERROR_CODE_*` 对齐；本仓仍持有
# 自己的期望值与全表（不是「对方有什么就认什么」），跨仓漂移由
# `tests/mcp/test_zero_contract_crosscheck.py::TestZeroErrorCodeCrosscheck` 拦截。
ZERO_ERROR_CODE_UNKNOWN_SESSION = "unknown-session"
ZERO_ERROR_CODE_CONFIG_INCOMPATIBLE = "config-incompatible"
ZERO_ERROR_CODE_EXTERNAL_PRIOR_INVALID = "external-prior-invalid"
ZERO_ERROR_CODE_PAYLOAD_INVALID = "payload-invalid"
ZERO_ERROR_CODE_CONFIG_INVALID = "config-invalid"
ZERO_ERROR_CODE_DEPLOY_ENV_INVALID = "deploy-env-invalid"
# ── 超时是**两个码不是一个**（本仓第二轮回件 §2.1 建议、Zero 2026-07-29 采纳落地）：
# 二者可否原样重试**相反**，单码会把判别推回人读文案。语义见各自异常类 docstring。
ZERO_ERROR_CODE_TIMEOUT_LOCK = "timeout-lock"
ZERO_ERROR_CODE_TIMEOUT_STEP = "timeout-step"

ZERO_ERROR_CODES: frozenset[str] = frozenset(
    {
        ZERO_ERROR_CODE_UNKNOWN_SESSION,
        ZERO_ERROR_CODE_CONFIG_INCOMPATIBLE,
        ZERO_ERROR_CODE_EXTERNAL_PRIOR_INVALID,
        ZERO_ERROR_CODE_PAYLOAD_INVALID,
        ZERO_ERROR_CODE_CONFIG_INVALID,
        ZERO_ERROR_CODE_DEPLOY_ENV_INVALID,
        ZERO_ERROR_CODE_TIMEOUT_LOCK,
        ZERO_ERROR_CODE_TIMEOUT_STEP,
    }
)

# 消费方提取正则——**Zero 指定口径**，位置无关（`search` 非 `match`/`startswith`）。
_ZERO_ERROR_TOKEN_RE = re.compile(r"\[zero:([a-z][a-z0-9-]*)\]")

# 兼容别名：旧名保留、值不变（Zero 侧亦保留同名别名）。仅供跨仓守卫与历史调用点引用，
# **产品判定不再用它做前缀匹配**——前缀匹配正是上面那条死码的成因。
_UNKNOWN_SESSION_MARKER = ZERO_ERROR_CODE_UNKNOWN_SESSION

# 🕒 **过渡兼容**：老部署（Zero < 2026-07-29 令牌换代）发的是**裸前缀**
# `f"unknown-session: 未知 session_id=…"`，经 FastMCP 加壳后落在文案中部。无令牌时退回本正则：
# 要求 `unknown-session:` 出现在**行首或空白之后**（加壳恰好留一个空格），比裸子串判别性强
# ——"error: unknown-session happened" 这类无冒号的偶然子串不命中。
# ⏳ **何时可撤**：确认所连 Zero 部署全部 ≥ 令牌换代提交（Zero `_tool_error` 上线，
# 本仓 crosscheck 守卫已 pin 其全表）后，删本正则与 `classify_zero_error` 里的回退分支即可；
# 届时 `test_legacy_bare_prefix_still_recognized` 一并删（它是本兼容层的**唯一**理由）。
_LEGACY_UNKNOWN_SESSION_RE = re.compile(r"(?:^|\s)unknown-session:")


# ── 自定义异常 ─────────────────────────────────────────────────────────────────


class ZeroLinkDisabledError(RuntimeError):
    """ZERO_LINK_ENABLED=false 时尝试连接/调用抛出。

    客户端侧 flag 检查（双侧检查策略：client 禁用即拒绝连接，不等到 server 报错）。
    """


class ZeroLinkConnectionError(OSError):
    """transport 连接失败或 ClientSession.initialize() 失败。

    附带 stderr 输出供诊断（stdio 模式下尤其有用）。
    """

    def __init__(self, message: str, stderr: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr


class ZeroLinkCallError(RuntimeError):
    """工具调用失败（isError=True 或 McpError）。"""

    def __init__(self, tool: str, message: str) -> None:
        super().__init__(f"[{tool}] {message}")
        self.tool = tool


class ZeroLinkUnknownSessionError(ZeroLinkCallError):
    """step 命中 Zero 侧**未知/过期 session_id**（server 重启或会话已 close）。

    是 `ZeroLinkCallError` 子类（zero-link T6·②）：
    - `graceful_step` **可自愈**：用同 id `open_session(session_id=…)` 重开续会话后重试一次，
      仍失败才降级 `None`（配合 Zero resume-by-id，T6·④）；
    - 直接调 `step()` 的编排层可 **catch 本子类**走同样的 resume 逻辑，区别于连接失败/畸形响应
      （不可 resume）。

    判定走机读令牌 `[zero:unknown-session]`（位置无关），抗 Zero 侧文案漂移与 FastMCP 加壳。
    """


class ZeroLinkLockTimeoutError(ZeroLinkCallError):
    """step 等待 Zero 会话锁超时（`[zero:timeout-lock]`）——**可退避后原样重试**。

    Zero 只超时「获取锁」不超时「执行」：本轮**未进入内核、运行态未改动**（其
    `_acquire_with_timeout` 明言），归责为并发/前一轮挂起。故属**可降级**族：
    `graceful_step` 兜住降级 `None`（非关键路径丢一帧无所谓）；关键路径直调 `step()`
    的编排层可 catch 本子类做退避重试——与不可原样重试的 `ZeroLinkStepTimeoutError`
    重试语义**相反**，这正是两码不合并的理由（本仓第二轮回件 §2.1）。
    ⚠ Zero 侧 default-off：`ZERO_MCP_STEP_LOCK_TIMEOUT` 未设时无限等锁、本码不产出。
    """


class ZeroLinkStepTimeoutError(ZeroLinkCallError):
    """Zero 内核执行超时（`[zero:timeout-step]`）——**不可原样重试**。

    取消 ainvoke 会在 checkpointer 留**半截运行态**：原样重试会让已跑完的节点重跑、
    reducer 通道双重累加（机制两仓联合实证，见 notes/2026-07-29-mcp-reply-round2.md
    §2.4；危害面待 Zero 核 LastValue 标量通道前按**最坏情况**处置）。`graceful_step`
    按本仓 §2.5 承诺执行三件套：**不重试、日志 ERROR、降级 None**——仍是可降级族
    （非每轮必复现的配置/部署错），但 ERROR 级日志保证「内核慢」有人看见。
    ⚠ Zero 当前**只登记不产出**（执行超时尚未实现）；先落消费侧是让分类表一次到位。
    """


class ZeroLinkNonDegradableError(ZeroLinkCallError):
    """**不可静默降级**的一类调用错误——`graceful_step` 遇到它一律**上抛**而非返回 `None`。

    分界线：错误是否**每轮必复现且 client 无法自愈**。
    - 可降级（返回 `None`）：连接抖动、偶发协议错误、未分类的 server 错误——重试有意义，
      非关键路径丢一帧无所谓。
    - 不可降级（本类）：配置/传参/部署问题——静默 `None` 会让**每一轮**都悄悄丢一次 step，
      且与「偶发抖动」在观测上不可区分（看板只见帧率下降，不见根因），排障成本极高。

    子类见 `ZeroLinkConfigIncompatibleError` / `ZeroLinkCallerFaultError` /
    `ZeroLinkDeployEnvError`。仍是 `ZeroLinkCallError` 子类 → 既有
    `except ZeroLinkCallError` 的调用点行为不变（零回归）。
    """


class ZeroLinkConfigIncompatibleError(ZeroLinkNonDegradableError):
    """Zero 内核执行失败，且**活跃会话的 config 不可变** → 必须**以新配置重开会话**。

    对应 Zero `[zero:config-incompatible]`（其 step 的 `except ValueError` 分支）：
    多为会话级配置组合不兼容，表现为 **open 成功、每 step 崩**——改传参无效、重试无效，
    只有换 config 重开会话能好。故 `graceful_step` 不吞它（Zero §4.4-9 明确要求）。
    """


class ZeroLinkCallerFaultError(ZeroLinkNonDegradableError):
    """**调用方**传参/配置不合法——改传参就能好，属本仓自己的 bug。

    对应 Zero `[zero:payload-invalid]` / `[zero:external-prior-invalid]` /
    `[zero:config-invalid]`。与既有「M3/M6 `ValueError` 不 graceful、须透传」同口径：
    `build_external_priors_override` 的本地预校验与 Zero 侧判定若出现分歧
    （本地放行、Zero 拒），那是**跨仓契约漂移**，必须炸出来而不是每轮静默丢帧。
    """


class ZeroLinkDeployEnvError(ZeroLinkNonDegradableError):
    """**部署端** env 值不合法（Zero `[zero:deploy-env-invalid]`）——改 client 传参永远改不好。

    Zero 刻意把它与 client-config 错误分码，正是为了不让 client 照着 config 瞎改。
    ⚠ stdio 传输下 server 进程环境**就是**本进程环境（`_build_subprocess_env` 全量拷贝
    `os.environ`），所以「部署端」很可能就是本机 `.env` —— 须抛给人看，不可静默降级。
    """


# 码 → 异常类。未登记的新码（Zero 先行加码、本仓未跟）落到 `None` → 退回基类
# `ZeroLinkCallError` + 一条 warning 日志，**不炸**（跨仓单边升级零回归）；
# 表本身的漂移由 crosscheck 守卫判红。
_CODE_TO_EXCEPTION: dict[str, type[ZeroLinkCallError]] = {
    ZERO_ERROR_CODE_UNKNOWN_SESSION: ZeroLinkUnknownSessionError,
    ZERO_ERROR_CODE_CONFIG_INCOMPATIBLE: ZeroLinkConfigIncompatibleError,
    ZERO_ERROR_CODE_EXTERNAL_PRIOR_INVALID: ZeroLinkCallerFaultError,
    ZERO_ERROR_CODE_PAYLOAD_INVALID: ZeroLinkCallerFaultError,
    ZERO_ERROR_CODE_CONFIG_INVALID: ZeroLinkCallerFaultError,
    ZERO_ERROR_CODE_DEPLOY_ENV_INVALID: ZeroLinkDeployEnvError,
    ZERO_ERROR_CODE_TIMEOUT_LOCK: ZeroLinkLockTimeoutError,
    ZERO_ERROR_CODE_TIMEOUT_STEP: ZeroLinkStepTimeoutError,
}


# ── 内部辅助 ───────────────────────────────────────────────────────────────────


def _is_enabled() -> bool:
    """检查客户端侧 ZERO_LINK_ENABLED feature flag。

    宽松真值判定（与 SCREEN_CAPABILITY_ENABLED 解析风格一致）：
    "1" / "true" / "yes"（大小写不敏感）均视为 True，其余视为 False。
    """
    return os.environ.get("ZERO_LINK_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
    }


def _build_subprocess_env() -> dict[str, str]:
    """构造子进程环境变量。

    StdioServerParameters.env 若非 None，SDK 内部会做 {**get_default_environment(), **env}，
    即只继承有限白名单 env（Windows: APPDATA/PATH/TEMP 等）再叠加我们传入的 key。
    为确保子进程能正确解析项目 src 包（依赖 PYTHONPATH / conda PATH 等），
    直接传递完整 os.environ 的副本（字符串化），再显式透传 ZERO_LINK_ENABLED。
    """
    env: dict[str, str] = {k: v for k, v in os.environ.items() if isinstance(v, str)}
    # 显式透传 ZERO_LINK_ENABLED，保证子进程能看到（parent 进程中可能已设）
    env["ZERO_LINK_ENABLED"] = os.environ.get("ZERO_LINK_ENABLED", "false")
    return env


def _build_transport_params() -> tuple[str, Any]:
    """根据 ZERO_LINK_TRANSPORT 构造传输参数。

    Returns:
        ("stdio", StdioServerParameters) 或 ("http", (endpoint_url, token))。

    默认值均为 Zero server 未建前的**临时占位**，可通过 .env 覆盖：
    - ZERO_SERVER_COMMAND：stdio 模式 server 命令（默认 sys.executable）。
    - ZERO_SERVER_ARGS：stdio 模式 server 参数 JSON 列表（默认 ["-m","src.mcp_server"]）。
    - ZERO_SERVER_CWD：stdio 模式 server 工作目录（默认 D:\\Zero）。
    - ZERO_HTTP_ENDPOINT：http 模式 endpoint URL。
    - ZERO_HTTP_TOKEN：http 模式 Bearer token。
    """
    transport = os.getenv("ZERO_LINK_TRANSPORT", "stdio").lower()

    if transport == "stdio":
        command = os.getenv("ZERO_SERVER_COMMAND", sys.executable)
        args_raw = os.getenv("ZERO_SERVER_ARGS", '["-m","src.mcp_server"]')
        args: list[str] = json.loads(args_raw)
        cwd = os.getenv("ZERO_SERVER_CWD", r"D:\Zero")
        params = StdioServerParameters(
            command=command,
            args=args,
            cwd=cwd,
            env=_build_subprocess_env(),
        )
        return ("stdio", params)

    # http 传输
    endpoint = os.getenv("ZERO_HTTP_ENDPOINT", "")
    token = os.getenv("ZERO_HTTP_TOKEN", "")
    return ("http", (endpoint, token))


def _build_http_client(token: str) -> httpx.AsyncClient | None:
    """http 传输的鉴权客户端：有 token 则预置 ``Authorization: Bearer <token>`` 头的
    ``httpx.AsyncClient``（新 SDK ``streamable_http_client`` 不直收 headers，须经 ``http_client``
    注入）；无 token 返回 ``None``（不鉴权——默认 127.0.0.1 本地场景零回归）。

    ⚠ Bearer 是标准方案（RFC 6750 ``Authorization: Bearer <token>``），与 Zero server 侧
    对齐的只是**共享 token 值**（两侧 .env），格式无歧义。抽成独立函数以便单测 header 构造
    （连接路径难在单测里跑，构造逻辑可）。
    """
    if not token:
        return None
    return httpx.AsyncClient(headers={"Authorization": f"Bearer {token}"})


def classify_zero_error(text: str) -> str | None:
    """从 Zero 错误文案中提取**机读错误码**；无从判定返回 ``None``。

    判定顺序（**位置无关**，故对 FastMCP 加壳后的 wire 文本同样成立）：
    1. 令牌 ``[zero:<code>]``——`re.search` 非 `startswith`，这是 Zero 指定的消费口径；
       Zero 保证「全文恰出现一次」（其 `_tool_error` 会把人读文案里回显的同形字面量
       `[zero:` 净化成 `(zero:`），故取首个匹配即可。
    2. **过渡兼容**：无令牌时退回旧裸前缀 `unknown-session:`（老部署，见
       `_LEGACY_UNKNOWN_SESSION_RE` 的撤除条件）。

    返回的码**不保证**在 `ZERO_ERROR_CODES` 内——Zero 可能先行加码；调用方按 `_CODE_TO_EXCEPTION`
    查表，查不到即按基类处理（不炸）。
    """
    match = _ZERO_ERROR_TOKEN_RE.search(text)
    if match is not None:
        return match.group(1)
    if _LEGACY_UNKNOWN_SESSION_RE.search(text):
        return ZERO_ERROR_CODE_UNKNOWN_SESSION
    return None


def _exception_for_error_text(tool_name: str, text: str, message: str) -> ZeroLinkCallError:
    """按机读码把 Zero 错误文案映射成对应异常实例（查不到码 → 基类）。

    ⚠ `unknown-session` 语义**只对 `zero.step` 成立**（会话不存在 → 可用同 id resume）：
    open/close_session 即便文案带该码也不升级为子类，保「子类 ⇒ resume 通路可走」严格成立
    （code-review W1 结论沿用）。其余码与工具无关（如 payload-invalid 两个工具都会出）。
    """
    code = classify_zero_error(text)
    if code is None:
        return ZeroLinkCallError(tool_name, message)
    if code == ZERO_ERROR_CODE_UNKNOWN_SESSION and tool_name != "zero.step":
        return ZeroLinkCallError(tool_name, message)
    exc_type = _CODE_TO_EXCEPTION.get(code)
    if exc_type is None:
        logger.warning(
            "Zero 返回本仓未登记的机读错误码 %r（tool=%s）——按通用调用错误处理；"
            "请同步 client._CODE_TO_EXCEPTION 与跨仓守卫。",
            code,
            tool_name,
        )
        return ZeroLinkCallError(tool_name, message)
    return exc_type(tool_name, message)


def _extract_text(result: Any, tool_name: str) -> str:
    """从 CallToolResult 中提取文本内容。

    result.isError=True 时按 Zero 机读令牌 `[zero:<code>]` 分类抛出对应
    `ZeroLinkCallError` 子类（unknown-session / config-incompatible / caller-fault /
    deploy-env / timeout-lock / timeout-step）；无码或未登记码 → 基类。
    content 为空或首元素无 text 属性时也抛错。
    """
    if result.isError:
        err_text = ""
        if result.content and isinstance(result.content[0], TextContent):
            err_text = result.content[0].text
        message = err_text or "server 返回 isError=True"
        raise _exception_for_error_text(tool_name, err_text, message)

    if not result.content:
        raise ZeroLinkCallError(tool_name, "server 返回空 content")

    first = result.content[0]
    if not isinstance(first, TextContent):
        raise ZeroLinkCallError(
            tool_name,
            f"期望 TextContent，得到 {type(first).__name__}",
        )
    return first.text


def generate_session_id() -> str:
    """生成**不可猜**的会话 id（zero-link T6·④），供调用方传给 `open_session(session_id=…)`。

    session_id 既是 resume 键、也是**运行态访问凭据**（回执信任模型）——多用户/对外场景须配 T5
    Bearer 鉴权，且 id **不可枚举**。用 `secrets.token_hex(16)`（128-bit CSPRNG，等价 uuid4 熵、
    十六进制无歧义）而非序号/时间戳。单机单用户可直接用 Zero 默认 uuid4（不传 session_id）。
    """
    return secrets.token_hex(16)


# ── 主类 ───────────────────────────────────────────────────────────────────────


class ZeroLinkClient:
    """Zero MCP Client，async context manager。

    把 Zero 当外部服务经 MCP call_tool 调用（不 import Zero 代码库，AD-2）。
    session_id 不由 client 持有，单实例可服务多 Zero 会话（无状态句柄）。

    用法::

        async with ZeroLinkClient() as client:
            sid = await client.open_session(persona="default")
            bundle = await client.step(sid, AffectStimulus(valence=0.3, arousal=0.5))
            await client.close_session(sid)

    生命周期：
        - __aenter__：flag 检查 → transport 连接 → ClientSession.initialize()。
        - __aexit__：session=None，关 AsyncExitStack（清理 transport + session）。
        - 连接失败统一包装为 ZeroLinkConnectionError（stdio 尽量带 stderr 诊断）。
    """

    def __init__(self) -> None:
        """初始化 ZeroLinkClient。

        无参构造：传输参数（stdio 命令/cwd 或 http endpoint/token）全部由 .env 提供
        （见 _build_transport_params）。session_id 不由 client 持有，单实例可服务多
        Zero 会话（无状态句柄）。
        """
        self.exit_stack: contextlib.AsyncExitStack | None = None
        self.session: ClientSession | None = None

    # ── context manager ───────────────────────────────────────────────────────

    async def __aenter__(self) -> ZeroLinkClient:
        # 1. 客户端侧 feature flag 检查
        if not _is_enabled():
            raise ZeroLinkDisabledError(
                "Zero Link 未启用（ZERO_LINK_ENABLED=false）。"
                "请设置 ZERO_LINK_ENABLED=true 后重试。"
            )

        # 2. 选择传输
        transport_kind, transport_params = _build_transport_params()

        # 3. AsyncExitStack 嵌套管理 transport + ClientSession
        stack = contextlib.AsyncExitStack()
        try:
            await stack.__aenter__()

            if transport_kind == "stdio":
                # stdio_client yield (read, write)
                read_stream, write_stream = await stack.enter_async_context(
                    stdio_client(transport_params)
                )
            else:
                # http 传输：streamable_http_client(url, *, http_client) yield 三元组。
                # 新 API 不直接收 headers——Bearer token 经预置 httpx.AsyncClient 注入。
                endpoint, token = transport_params
                http_client = _build_http_client(token)
                if http_client is not None:
                    await stack.enter_async_context(http_client)
                read_stream, write_stream, _get_session_id = await stack.enter_async_context(
                    streamable_http_client(endpoint, http_client=http_client)
                )

            # 建立 ClientSession
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            try:
                await session.initialize()
            except Exception as exc:
                raise ZeroLinkConnectionError(
                    f"ClientSession 初始化失败：{exc}",
                    stderr="",
                ) from exc

            self.exit_stack = stack
            self.session = session
            logger.info(
                "ZeroLinkClient 连接成功（transport=%s）",
                transport_kind,
            )

        except ZeroLinkConnectionError:
            if self.exit_stack is None:
                try:
                    await stack.aclose()
                except Exception:
                    pass
            raise
        except Exception as exc:
            if self.exit_stack is None:
                try:
                    await stack.aclose()
                except Exception:
                    pass
            raise ZeroLinkConnectionError(
                f"Zero Link 连接失败（transport={transport_kind}）：{exc}",
                stderr="",
            ) from exc
        except asyncio.CancelledError as exc:
            # streamable-http 传输在 HTTP 层被拒（如 T5 Bearer 401 鉴权失败）时，其内部 anyio task
            # group 取消，向上抛 CancelledError——它是 **BaseException 非 Exception**，故上面
            # `except Exception` 接不住会穿透。用 Task.cancelling() 区分：>0=本任务被**外部**取消
            # （尊重取消语义、原样重抛）；==0=传输内部因连接被拒而取消 → 归**连接失败**
            # （ZeroLinkConnectionError·连接层，符合回执「401 走连接层不走 graceful_step」）。
            # aclose 在取消态可能再抛（含 anyio「exit cancel scope in different task」），尽力吞。
            if self.exit_stack is None:
                try:
                    await stack.aclose()
                except BaseException:  # noqa: BLE001 - 取消态清理尽力而为，二次异常不掩盖首因
                    pass
            task = asyncio.current_task()
            if task is not None and task.cancelling() > 0:
                raise
            # from exc 保留原始 CancelledError 作 __cause__（供诊断非鉴权类的内部取消根因，
            # code-review W2）——CancelledError 出现在 ZeroLinkConnectionError 链里是正常异常链。
            raise ZeroLinkConnectionError(
                f"Zero Link 连接被传输层取消（transport={transport_kind}）——"
                "HTTP 可能为 401 鉴权失败或连接被拒；请核对 ZERO_HTTP_TOKEN 与 Zero "
                "ZERO_MCP_HTTP_TOKEN 是否同值。",
                stderr="",
            ) from exc

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.session = None
        if self.exit_stack is not None:
            await self.exit_stack.__aexit__(exc_type, exc_val, exc_tb)
            self.exit_stack = None

    # ── 内部工具调用 ──────────────────────────────────────────────────────────

    def _require_session(self) -> ClientSession:
        """断言 session 存在（在 context 内调用时始终满足）。

        Raises:
            ZeroLinkConnectionError: 在 async with 块外调用时。
        """
        if self.session is None:
            raise ZeroLinkConnectionError("ZeroLinkClient 尚未初始化，请在 async with 块内使用。")
        return self.session

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """调用 MCP 工具并返回 TextContent.text。

        Args:
            tool_name: 工具名（与 Zero server 注册名一致，如 "zero.open_session"）。
            arguments: 工具调用参数字典。

        Returns:
            server 返回的 JSON 字符串（调用方负责解析）。

        Raises:
            ZeroLinkCallError: server 返回 isError=True 或 McpError。
        """
        session = self._require_session()
        try:
            result = await session.call_tool(tool_name, arguments)
        except McpError as exc:
            raise ZeroLinkCallError(tool_name, str(exc)) from exc
        return _extract_text(result, tool_name)

    # ── 公开工具方法 ──────────────────────────────────────────────────────────

    async def open_session(
        self,
        *,
        persona: str | None = None,
        config: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> str:
        """在 Zero 侧创建或 **resume** 一个会话，返回 session_id。

        生命周期失败须明确报错（不 graceful），上层须显式处理异常。

        Args:
            persona: 可选人格标识（传给 Zero zero.open_session 工具）。
            config:  可选配置字典（传给 Zero zero.open_session 工具）。
            session_id: 可选会话 id（zero-link T6·④ resume-by-id）：传了 → Zero 以此 id 重开
                （已活跃则幂等返回同 id；否则新建绑该 thread_id，运行态是否真续取决于 Zero
                `ZERO_CHECKPOINT_BACKEND=sqlite`——memory 后端重开=全新会话、不报错）。不传 →
                Zero 新铸 uuid4。⚠ SessionConfig 不进 checkpoint，resume 须**再供同一 config**。
                ⚠ 信任模型：session_id = 运行态访问凭据；多用户须配 T5 鉴权 + 用
                `generate_session_id()` 生成不可猜 id（勿用可枚举序号）。

        Returns:
            Zero 侧的 session_id 字符串（resume 时 == 传入的 session_id）。

        Raises:
            ZeroLinkCallError:      工具调用失败（server 返回 isError=True 或协议错误）。
            ZeroLinkConnectionError: 未在 async with 内调用。
        """
        args: dict[str, Any] = {}
        if persona is not None:
            args["persona"] = persona
        if config is not None:
            args["config"] = config
        if session_id is not None:
            args["session_id"] = session_id
        text = await self._call_tool("zero.open_session", args)
        # 响应解析防御：畸形 JSON / 缺 session_id 键统一封装为 ZeroLinkCallError，
        # 不让原始 JSONDecodeError/KeyError 穿透异常封装边界（调用方只预期 ZeroLink* 异常）。
        try:
            data: dict[str, Any] = json.loads(text)
            returned_id: str = data["session_id"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ZeroLinkCallError("zero.open_session", f"响应格式非预期：{exc}") from exc
        return returned_id

    async def step(
        self,
        session_id: str,
        stimulus: AffectStimulus,
        priors: list[ModalityPrior] | None = None,
    ) -> ExpressionBundle:
        """向 Zero 发送单步情感刺激，返回表达包。

        Args:
            session_id: 由 open_session() 获得的 Zero 会话 ID。
            stimulus:   情感刺激（valence/arousal/coping_potential）。
            priors:     可选多模态先验列表（非空时构造 external_priors 载荷注入）。

        Returns:
            ExpressionBundle 解析结果。

        Raises:
            ValueError:              priors 不满足 M3/M6 约束（由 build_external_priors_override
                                     抛出，不 catch——调用方参数错误须透传，fail-fast）。
            ZeroLinkCallError:       工具调用失败。
            ZeroLinkConnectionError: 未在 async with 内调用。
        """
        # exclude_none=True：coping_potential 为 None 时略去该键（最小合法载荷），
        # 避免线上 stim 带 "coping_potential": null 依赖 Zero 侧对 null 可选字段的宽容性。
        stim_dict: dict[str, Any] = stimulus.model_dump(exclude_none=True)
        arguments: dict[str, Any] = {"session_id": session_id, "stim": stim_dict}

        if priors:
            override = build_external_priors_override(priors)
            # tuple→list 显式转换（可见可测，不靠 json 隐式处理）
            arguments["external_priors"] = [
                [name, list(mu), list(precision)]
                for name, mu, precision in override["external_priors"]
            ]

        text = await self._call_tool("zero.step", arguments)
        # 响应解析防御：畸形 JSON / expression 结构不合契约统一封装为 ZeroLinkCallError，
        # 使 graceful_step 能兜住畸形响应降级为 None（ValidationError 是 ValueError 子类，
        # 但此处是 server 响应问题而非调用方 M3/M6 参数错误——后者在 _call_tool 之前已抛）。
        try:
            return ExpressionBundle.from_step_output(json.loads(text))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ZeroLinkCallError("zero.step", f"响应解析失败：{exc}") from exc

    async def close_session(self, session_id: str) -> None:
        """关闭 Zero 侧的会话。

        生命周期失败须明确报错（不 graceful），上层须显式处理异常。

        Args:
            session_id: 要关闭的 Zero 会话 ID。

        Raises:
            ZeroLinkCallError:       工具调用失败。
            ZeroLinkConnectionError: 未在 async with 内调用。
        """
        await self._call_tool("zero.close_session", {"session_id": session_id})

    async def graceful_step(
        self,
        session_id: str | None,
        stimulus: AffectStimulus,
        priors: list[ModalityPrior] | None = None,
        *,
        resume_config: dict[str, Any] | None = None,
    ) -> ExpressionBundle | None:
        """容错版单步调用，供编排层在「非关键路径」降级使用。

        **降级 vs 上抛的分界线 = 错误是否每轮必复现且 client 无法自愈**（Zero §4.4-9）：

        静默返回 None（可降级）：
        - ZERO_LINK_ENABLED=false（未启用）。
        - session_id 为 None（会话未建立）。
        - 未分类的 ZeroLinkCallError / ZeroLinkConnectionError / McpError
          （连接抖动、偶发协议错误）。
        - `[zero:timeout-lock]` → `ZeroLinkLockTimeoutError`：等锁超时，未进内核、
          运行态未改动，可退避后原样重试——非关键路径直接降级即可，关键路径的重试
          由直调 `step()` 的编排层自己做。
        - `[zero:timeout-step]` → `ZeroLinkStepTimeoutError`：内核执行超时，
          **不重试**（半截运行态，原样重试会节点重跑/reducer 双重累加），
          **ERROR** 级日志后降级（§2.5 承诺三件套；Zero 当前只登记不产出该码）。

        **上抛不吞**（`ZeroLinkNonDegradableError` 及其子类；连同既有的 `ValueError`）：
        - `[zero:config-incompatible]` → `ZeroLinkConfigIncompatibleError`：活跃会话 config
          **不可变**，须**以新配置重开会话**。静默 None 会让每一轮都悄悄丢一次 step，且与偶发抖动
          在观测上不可区分（看板只见帧率下降、不见根因）——故必须炸给调用方去换 config 重开。
        - `[zero:payload-invalid]` / `[zero:external-prior-invalid]` / `[zero:config-invalid]`
          → `ZeroLinkCallerFaultError`：调用方 bug，改传参就能好，与既有 M3/M6 fail-fast 同口径。
        - `[zero:deploy-env-invalid]` → `ZeroLinkDeployEnvError`：部署端 env 问题，client 改不好，
          须抛给人。

        **unknown-session resume 重试（zero-link T6·④）**：step 命中 Zero 侧未知/过期 session
        （`ZeroLinkUnknownSessionError`，server 重启 / 会话已 close）时，用**同一 session_id 重开
        （+再供 `resume_config`）后重试一次 step**——Zero `ZERO_CHECKPOINT_BACKEND=sqlite` 时按
        thread_id 自动续运行态，memory 后端则重开=全新会话（不报错）。重开或重试再失败 → 降级 None
        （只重试一次、不递归；但重试路径上的**不可降级错误同样上抛**）。⚠ SessionConfig 不进
        checkpoint，未供 `resume_config` 则 resume 会话走 Zero env 默认门控（非原会话 config）；
        须续原门控时调用方应传原 config。

        Args:
            session_id:    Zero 会话 ID（None 时立即返回 None）。
            stimulus:      情感刺激。
            priors:        可选多模态先验列表。
            resume_config: unknown-session resume 重开时**再供的会话 config**（应与原 open_session
                           一致）；None → resume 会话走 Zero env 默认门控。

        Returns:
            ExpressionBundle 或 None（降级时）。

        Raises:
            ValueError:                    priors 不满足 M3/M6 约束（透传，不 graceful）。
            ZeroLinkNonDegradableError:    config-incompatible / 调用方传参错 / 部署端 env 错
                                           （见上「上抛不吞」；均为 ZeroLinkCallError 子类）。
        """
        if not _is_enabled():
            logger.debug("graceful_step: ZERO_LINK_ENABLED=false，跳过")
            return None
        if session_id is None:
            logger.debug("graceful_step: session_id=None，跳过")
            return None
        try:
            return await self.step(session_id, stimulus, priors)
        except ZeroLinkUnknownSessionError:
            # 机读令牌命中 Zero 侧未知/过期 session（server 重启 / 会话已 close）：据 Zero 回执
            # （T6·④）用**同一 session_id 重开(+再供 config)后重试一次** step。重试的 step 若再抛
            # ZeroLinkUnknownSessionError（是 ZeroLinkCallError 子类）会被**内层**
            # except (ZeroLinkCallError, …) 兜住 → None，故不递归、至多重试一次。
            logger.warning(
                "graceful_step: session=%s 未知/过期（Zero unknown-session）；"
                "用同 id resume 重开续会话并重试一次。",
                session_id,
            )
            try:
                await self.open_session(session_id=session_id, config=resume_config)
                return await self.step(session_id, stimulus, priors)
            except ZeroLinkNonDegradableError:
                # resume 路径上同样不吞不可降级错误（如重开时 resume_config 不合法 →
                # caller-fault）。写在通用分支**之前**：它是 ZeroLinkCallError 子类，
                # 顺序颠倒会被通用分支先兜住 → 又变成静默 None。
                raise
            except ZeroLinkStepTimeoutError as exc:
                # resume 重试的 step 内核执行超时：同外层，ERROR 级日志 + 降级（不再重试）。
                logger.error(
                    "graceful_step: session=%s resume 重试遇内核执行超时（timeout-step）"
                    "——不可原样重试，降级 None：%s",
                    session_id,
                    exc,
                )
                return None
            except (ZeroLinkCallError, ZeroLinkConnectionError, McpError) as exc:
                logger.warning(
                    "graceful_step: session=%s resume 重试仍失败（exc=%s）：%s；降级 None。",
                    session_id,
                    type(exc).__name__,
                    exc,
                )
                return None
        except ZeroLinkNonDegradableError:
            # 必须排在下面通用分支之前（子类先于基类），否则被静默吞成 None。
            raise
        except ZeroLinkStepTimeoutError as exc:
            # §2.5 承诺三件套：**不重试**（半截运行态，原样重试会节点重跑/reducer 双重
            # 累加）、**ERROR** 级日志（与偶发抖动的 warning 在观测上分开——「内核慢」
            # 须有人看见）、降级 None。同样须排在通用分支之前，否则退化成 warning。
            logger.error(
                "graceful_step: session=%s Zero 内核执行超时（timeout-step）"
                "——不可原样重试，降级 None：%s",
                session_id,
                exc,
            )
            return None
        except (ZeroLinkCallError, ZeroLinkConnectionError, McpError) as exc:
            logger.warning(
                "graceful_step 降级（session=%s, exc=%s）：%s",
                session_id,
                type(exc).__name__,
                exc,
            )
            return None
