"""Zero MCP Client（Python 侧）。

生命周期：async context manager（AsyncExitStack 嵌套 transport + ClientSession）。
工具面：open_session / step / close_session / purge_session / graceful_step 五个 async 方法。
⚠ `purge_session` 是**不可逆破坏性**动作（删该会话全部持久运行态），永不自动调用，
见 `graceful_step(purge_on_interrupted=…)` 的默认值论证。

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
from dataclasses import dataclass
from enum import StrEnum
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


# ── zero.open_session 响应形状（Zero 2026-07-29 换代：返回体「只增不改」）───────────
# Zero `zero.open_session` 现返回 `{session_id, resumed}`；**且仅当** resume 时探测到上一轮被
# 中途取消，才另带 `{interrupted_at: [待执行节点名]}`（Zero server.py 的
# `out = {"session_id": sid, "resumed": resuming}` + 条件分支 `out["interrupted_at"] = …`）。
#
# 🛑 **缺键即回落**（对现网 Zero 零回归·硬约束）：老部署只回 `{session_id}`，两个新键都读不到
# 时行为必须与换代前逐字相同——`open_session` 照常返回 session_id，不抛、不打额外日志。
# 这也是 Zero 侧「返回体只增不改」承诺的消费侧对价，守卫见
# `tests/mcp/test_zero_client.py::test_open_session_without_new_keys_is_zero_regression`。
#
# 形状防御：两个键都按「读不到就当没有」处理（记一条 warning 后回落 None），**任何形状异常都
# 不得让 open_session 炸** —— 会话生命周期不能因为一条观测量的类型不对就打不开。
#
# 🛑 `interrupted_at` **缺席有四义**（Zero `daecce1` 现场核验，2026-07-29 20:0x）——不得一律
#    当成「未中断，可安全续跑」。Zero `open_session` 里**四条**路径都会让该键缺席
#    （上一版标题写「三义」而正文枚举四条，本轮订正；计数词随认知修订而漂移，
#     故守卫**一律不得**锚在它上面，见下方 `LOG_MARKER_*`）：
#      ① **未探测·新建会话**：`resuming` 为假时整段探测被跳过（返回体 `resumed: false`）；
#      ② **未探测·活跃幂等重开**：`registry.get(sid) is not None` 分支**提前 return**
#         `{"session_id", "resumed": True}`，根本走不到探测（Zero 源码注释未列此义）；
#      ③ **探测失败**：`try: interrupted = await session.interrupted_at()` 外面是
#         `except Exception: logger.exception(...)`，异常被吞、`interrupted` 留在 None；
#      ④ **探测成功且干净**：`nxt or None` 返回 None。
#    ⚠ 其中 ③ 与我方要防的半截态是**故障相关**的：探测读的 (`graph.aget_state`) 正是那份
#      可能半写的 checkpoint。把缺席一律读成「安全」= 止血在最该生效时静默失效。
#    ⇒ 解析层必须把「键缺席」「键在但为空」「键在且非空」「键在但形状坏」四态**分开表达**
#      （见 `ZeroInterruptProbe`），由消费点自己决定每一格怎么处置。
_OPEN_SESSION_KEY_RESUMED = "resumed"
_OPEN_SESSION_KEY_INTERRUPTED_AT = "interrupted_at"


# ── 日志锚点（守卫的**稳定标识**）─────────────────────────────────────────────────
# 🛑 为什么必须有（2026-07-29 复审实证）：上一版的三条守卫用中文计数词 `"三义"` 做锚点，
#    其中**两条是否定式**（`assert not any("三义" in m …)`）。一旦文案按认知修订改成「四义」，
#    肯定式那条会红、两条否定式却**静默变成空真**（vacuous）——守卫看着还是绿的，判别力已归零。
#    ⇒ 凡是被守卫钉住的日志分支，一律带一个下列 marker，断言只锚 marker；中文文案可自由修订。
# 形制：ASCII、分支唯一、`[zl:…]` 前缀（**刻意避开** Zero 错误码令牌 `[zero:<code>]` 的形状，
#    见 `_ZERO_ERROR_TOKEN_RE`——两者若同形，日志文案会被误当成机读错误码）。
LOG_MARKER_INTERRUPTED_REFUSED = "[zl:interrupted-refused]"
"""半截运行态 → 本帧拒绝续跑（`purge_on_interrupted=False`，脏态保留）。"""

LOG_MARKER_INTERRUPTED_REFUSED_PURGING = "[zl:interrupted-refused-purging]"
"""半截运行态 → 本帧拒绝续跑，且**已请求** purge（`purge_on_interrupted=True`）。"""

LOG_MARKER_PROBE_MALFORMED = "[zl:interrupt-probe-malformed]"
"""`interrupted_at` 形状非法（跨仓契约漂移）——照常续跑但必须有人看见。"""

LOG_MARKER_PROBE_UNDECIDABLE = "[zl:interrupt-probe-undecidable]"
"""`resumed=True` 但 `interrupted_at` 缺席——我方不可判，照常续跑 + 可区分告警。"""

LOG_MARKER_INTERRUPTED_ON_OPEN = "[zl:interrupted-at-open]"
"""`open_session` 返回体直接带非空 `interrupted_at`（**任何**调用路径，含无守卫的常规 resume）。"""


class ZeroInterruptProbe(StrEnum):
    """`open_session` 返回体里 `interrupted_at` 这一位的**判读四态**。

    一个 ``tuple | None`` 承担不了四件事：今天 `MALFORMED` 与 `ABSENT` 都塌缩成 ``None``，
    消费点无从区分「对方没说」与「对方说了但契约漂移了」。本枚举把该位显式化。

    Attributes:
        ABSENT:      键缺席。**我方不可判**——见上方四义（未探测·新建 / 未探测·活跃幂等重开 /
                     探测失败 / 探测成功且干净）。
        CLEAN:       键在且为空序列 ⇒ 对方**明确**探测过且无待执行节点。Zero 今天
                     `nxt or None` 不发空表，故该态在现网不出现；但契约未禁止发，
                     而「对方明确说干净」比「对方没说」强得多，值得留一格。
        INTERRUPTED: 键在且是非空 ``list[str]`` ⇒ **确定**半截：运行态停在 super-step
                     边界，续跑从待执行节点继续而非重跑整轮（Zero 自己的契约表述）。
        MALFORMED:   键在但形状非 ``list[str]`` ⇒ 跨语言契约已漂移，同样**不可判**。
    """

    ABSENT = "absent"
    CLEAN = "clean"
    INTERRUPTED = "interrupted"
    MALFORMED = "malformed"


@dataclass(frozen=True, slots=True)
class ZeroOpenSessionInfo:
    """`zero.open_session` 的**完整**返回体（session_id + 两条运行态观测量）。

    Attributes:
        session_id:     Zero 侧会话 id（resume 时 == 传入的 id）。
        resumed:        Zero 是否按 resume 语义重开该会话；键缺席或形状非 bool → ``None``
                        （「读不到」与「读到 False」语义不同，故不塌缩成 False）。
                        🔑 该键还是**新老部署的判别位**：新 Zero **无条件**回它（两条
                        return 路径都带），老部署根本不发 ⇒ ``resumed is None`` ⇔
                        「对方不是会发中断观测量的那一代」，此时缺 `interrupted_at`
                        是正常态、不是信号。零回归判定就挂在这一位上。
                        ⏳ **这一位是临时判据**：它靠「新 Zero 无条件回 `resumed`」这条
                        **间接推断**（对方的实现细节，非契约），所以本仓才要向 Zero 索要
                        「承诺 `resumed` 不被条件化」。Zero `daecce1` 已落
                        `zero.describe_config(session_id?)`，带 `describe_config_version`
                        （增删任何键都 bump）+ `error_codes` 全量表 —— 那才是**代际判别的
                        正解**（直接问对方「你是哪一代」，不再从返回体形状反推）。本轮**有意
                        不接**（论证见 `graceful_step` docstring 末「为何仍用 `resumed` 判别」）；
                        接上后本行与下方 ABSENT 分支的零回归条件都应改挂 describe_config。
        interrupted_at: 待执行节点名；仅 ``interrupt_probe is INTERRUPTED`` 时非空。
                        ``ABSENT``/``MALFORMED`` → ``None``，``CLEAN`` → ``()``。
                        ⚠ **不要**只看这一个字段做判定：``None`` 同时覆盖「对方没说」与
                        「形状坏」两义，判定请读 `interrupt_probe`。
        interrupt_probe: 上一字段的四态判读（见 `ZeroInterruptProbe`）。
    """

    session_id: str
    resumed: bool | None = None
    interrupted_at: tuple[str, ...] | None = None
    interrupt_probe: ZeroInterruptProbe = ZeroInterruptProbe.ABSENT


# ── 内部辅助 ───────────────────────────────────────────────────────────────────


def _parse_open_session_resumed(data: dict[str, Any]) -> bool | None:
    """从 open_session 响应体取 ``resumed``；缺键或形状非 bool → ``None``（不炸）。

    只认真正的 ``bool``：``isinstance(1, bool) is False``，故 JSON 里写成 ``1``/``"true"``
    的伪真值一律按「读不到」处置 —— 宁可少一条观测量，也不把 truthy 字符串当成 True 用。
    """
    if _OPEN_SESSION_KEY_RESUMED not in data:
        return None
    value = data[_OPEN_SESSION_KEY_RESUMED]
    if isinstance(value, bool):
        return value
    logger.warning(
        "zero.open_session 返回的 %r 形状非预期（期望 bool，实得 %s=%r）——按读不到处置。",
        _OPEN_SESSION_KEY_RESUMED,
        type(value).__name__,
        value,
    )
    return None


def _parse_open_session_interrupted_at(
    data: dict[str, Any],
) -> tuple[ZeroInterruptProbe, tuple[str, ...] | None]:
    """从 open_session 响应体取 ``interrupted_at``，判读成**四态 + 节点名**。

    🛑 **「键缺席」与「键在但为空」必须分开**（本轮修的信息损失）：前者是「对方没说」
    （四义，见模块上方注释），后者是「对方明确说探测过且干净」——两者的证据强度不同，
    塌缩成同一个 ``None`` 会让消费点无法对前者施加额外保守处置。
    Zero 今天 `nxt or None` 理论上不发空表，但**我方不能靠对方的实现细节吃饭**：
    这是它随时可以改而不算破坏「返回体只增不改」承诺的自由度。

    形状坏（`MALFORMED`）整条丢弃而非逐元素过滤：节点名列表是**半截运行态**的证据，混入非 str
    说明契约已漂移，此时「部分读到」比「读不到」更危险（会让调用方以为自己拿到了完整的待执行
    节点集）。``str`` 本身可迭代但**不是** list/tuple，故 ``"abc"`` 走 MALFORMED 而非被拆成
    三个字符。任何形状异常都只记 warning、不抛 —— 会话生命周期不能因一条观测量的类型不对就断。
    """
    if _OPEN_SESSION_KEY_INTERRUPTED_AT not in data:
        return ZeroInterruptProbe.ABSENT, None
    value = data[_OPEN_SESSION_KEY_INTERRUPTED_AT]
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        nodes = tuple(value)
        probe = ZeroInterruptProbe.INTERRUPTED if nodes else ZeroInterruptProbe.CLEAN
        return probe, nodes
    logger.warning(
        "zero.open_session 返回的 %r 形状非预期（期望 list[str]，实得 %s=%r）——判为契约漂移"
        "（MALFORMED），**不**等同于「未中断」。",
        _OPEN_SESSION_KEY_INTERRUPTED_AT,
        type(value).__name__,
        value,
    )
    return ZeroInterruptProbe.MALFORMED, None


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

        `last_open_session` 是**只读观测量**（最近一次 `open_session` 收到的完整返回体，
        含 `resumed` / `interrupted_at`），供调用方读；它**不**参与内部判定——`graceful_step`
        用 `_open_session_info` 的**返回值**判定，故多会话并发下不会互相串味。
        """
        self.exit_stack: contextlib.AsyncExitStack | None = None
        self.session: ClientSession | None = None
        self.last_open_session: ZeroOpenSessionInfo | None = None

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

    async def _open_session_info(
        self,
        *,
        persona: str | None = None,
        config: dict[str, Any] | None = None,
        session_id: str | None = None,
        downstream_guard: bool = False,
    ) -> ZeroOpenSessionInfo:
        """`open_session` 的**全量返回体**版本（内部用）。

        与 `open_session` 唯一区别是返回 `ZeroOpenSessionInfo` 而非裸 session_id，
        使 `graceful_step` 能就地读到 `interrupted_at` 而**不经** `self.last_open_session`
        —— 后者是共享可变态，多会话并发下会串味（本 client 明示可服务多会话）。

        参数/异常与 `open_session` 完全一致，见其 docstring。

        `downstream_guard`：调用方在本次返回后**是否还有拦截**。
        `graceful_step` 的 unknown-session 自愈分支传 True（其后紧跟拒绝续跑的 ERROR，
        可能还有 purge）；公开 `open_session()` 走默认 False（其后确无守卫）。
        它**只影响 `interrupted_at` 那条 WARNING 的措辞**，不改变任何行为——
        分支无关的文案断言分支相关的结论，正是 2026-07-29 终审判为缺陷的那一类。
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

        # 🛑 新键解析在 session_id 之后、且**只用 .get/in**：缺键 → 两者皆 None，行为逐字回落
        # 到换代前（现网老 Zero 零回归）。形状异常只 warning 不抛（见两个 _parse_* 的 docstring）。
        probe, nodes = _parse_open_session_interrupted_at(data)
        info = ZeroOpenSessionInfo(
            session_id=returned_id,
            resumed=_parse_open_session_resumed(data),
            interrupted_at=nodes,
            interrupt_probe=probe,
        )
        self.last_open_session = info
        if info.resumed is not None:
            logger.info(
                "zero.open_session: session=%s resumed=%s",
                returned_id,
                info.resumed,
            )
        if info.interrupted_at:
            # WARNING 而非 INFO：非空 interrupted_at ⇒ 该会话运行态**停在 super-step 边界**
            # （上一轮被中途取消，已跑完节点的写入已落盘且 sqlite 后端跨重启保留）。
            # 续跑会从待执行节点继续、而非重跑整轮 —— 这是「拿到的下一帧不可全信」的信号。
            #
            # ⚠ 尾句**按调用方分支出**：`downstream_guard=False`（公开 `open_session()`，
            # 即常规 resume 路径）之后确无守卫——step 照常发、bundle 照常回、连一条 ERROR 都不会有
            # （Zero 的 `zero.step` 亦不做任何中断检查，daecce1 核验），缺口由
            # `test_normal_resume_path_has_no_interrupt_guard` 特征化钉住；
            # `downstream_guard=True`（`graceful_step` 自愈分支）之后紧跟拒绝续跑的 ERROR、
            # 可能还有 purge，此时**不得**再说「无守卫」「须自行决定调 purge_session」
            # ——那会劝调用方去做刚刚已经做完的事（2026-07-29 终审判为 blocking 的原话）。
            tail = (
                "后续由调用方拦截（紧随其后的 ERROR 给出处置）。"
                if downstream_guard
                else (
                    "⚠ 常规 resume 路径**无守卫**：除本条外不会再有任何日志或拦截，"
                    "调用方须自行决定轮换 session_id 或调 purge_session。"
                )
            )
            logger.warning(
                "%s zero.open_session: session=%s 上一轮被中途取消，运行态停在 super-step 边界，"
                "待执行节点=%s；在其上续跑 = 新刺激叠加到半截运行态。%s",
                LOG_MARKER_INTERRUPTED_ON_OPEN,
                returned_id,
                list(info.interrupted_at),
                tail,
            )
        return info

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
            Zero 换代后随响应一并返回的 `resumed` / `interrupted_at` 见
            `self.last_open_session`（`ZeroOpenSessionInfo`）——**缺键即回落**，老部署上
            两者皆 `None`，本方法的返回值与日志行为与换代前逐字一致（零回归）。

        Raises:
            ZeroLinkCallError:      工具调用失败（server 返回 isError=True 或协议错误）。
            ZeroLinkConnectionError: 未在 async with 内调用。
        """
        info = await self._open_session_info(
            persona=persona,
            config=config,
            session_id=session_id,
        )
        return info.session_id

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

    async def purge_session(self, session_id: str) -> bool:
        """删除 Zero 侧该会话的**全部持久运行态**（按 thread_id 清 checkpoint）。⚠ **不可逆**。

        与 `close_session` 语义不同（Zero `daecce1` 现场核验）：close 只摘牌 + 关连接，
        数据仍在；purge 内部**先跑一遍 close_session**（幂等）再删 checkpoint。

        🛑 破坏面比「清掉那份半截 checkpoint」大得多：它删的是该 thread_id 的**全部**
        checkpoint 历史，包括那些干净的、本可回滚过去的祖先版本。Zero 今天不暴露
        「回滚一格」这种更弱的补救，故本方法是**过度杀伤**的唯一可用替代品，
        绝不由本 client 自动触发（见 `graceful_step(purge_on_interrupted=…)`）。

        Args:
            session_id: 要清除运行态的 Zero 会话 ID（未知 id 在 Zero 侧幂等）。

        Returns:
            Zero 报告的 `purged`；键缺席/形状非 bool → ``False``（不猜）。
            ⚠ 该值语义是「**删除通路可用**」而非「确有数据被删」：Zero 的 `purged` 只在
            checkpointer 没有 `adelete_thread` 时才为 False，而对不存在的 thread 调
            `adelete_thread` 是 no-op 也照样回 True。⇒ **不得**用它反推「原先存在脏运行态」。

        Raises:
            ZeroLinkCallError:       工具调用失败（含老部署未注册该工具 → 未知工具错误）。
            ZeroLinkConnectionError: 未在 async with 内调用。
        """
        text = await self._call_tool("zero.purge_session", {"session_id": session_id})
        try:
            data: Any = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ZeroLinkCallError("zero.purge_session", f"响应格式非预期：{exc}") from exc
        if not isinstance(data, dict):
            raise ZeroLinkCallError(
                "zero.purge_session",
                f"响应格式非预期：期望 JSON object，实得 {type(data).__name__}",
            )
        purged = data.get("purged")
        return purged if isinstance(purged, bool) else False

    async def _purge_after_interrupted(self, session_id: str) -> bool:
        """`graceful_step` 拒绝续跑后的**可选**善后：清掉该会话运行态。best-effort。

        **不上抛的边界，如实写**（2026-07-29 复审订正——上一版写「永不上抛」，措辞过强）：
        吞的是 `Exception` 全族，**`BaseException` 仍原样穿透**（`asyncio.CancelledError` /
        `KeyboardInterrupt` / `SystemExit`）—— 取消语义必须尊重，善后不是压住取消的理由。

        为什么把 except 从「三个具体类型」放宽到 `Exception`（本轮修）：原实现只捕
        `(ZeroLinkCallError, ZeroLinkConnectionError, McpError)`，而 purge 恰好发生在**刚撞过
        unknown-session**（多半是对端重启）之后，正是传输最易碎的时刻——stdio 管道断时
        anyio 抛的是 `BrokenResourceError` / `ClosedResourceError`，二者继承 `Exception` 而
        **不属** OSError 家族（`ZeroLinkConnectionError` 是 OSError 子类，接不住），会原样穿透
        本方法、再穿透 `graceful_step` 内层 except 组（同样只列 ZeroLink*/McpError），
        把一条已判定「降级 None」的调用变成 raise —— 正是本方法要避免的那种失败模式升级。

        放宽**不制造静默面**：失败分支用 `logger.exception` 打**完整 traceback**（ERROR 级），
        故连「我方自己的 AttributeError/TypeError」这类真 bug 也是响亮的，不是被吞没。
        为什么这里连 `ZeroLinkNonDegradableError` 也吞：本方法只在 `graceful_step` 已决定
        「本帧降级 None」之后被调用，是**善后**而非主路径。让一次清理失败把一条本已判定为
        可降级的调用变成 raise，等于用更坏的失败模式替换较轻的那个。失败只记 ERROR
        （调用方仍会看到上一条「拒绝续跑」的 ERROR，两条一起构成完整现场）。
        """
        try:
            purged = await self.purge_session(session_id)
        except Exception as exc:  # noqa: BLE001 - 善后路径刻意宽；BaseException 仍穿透，见 docstring
            logger.exception(
                "graceful_step: session=%s 已按 purge_on_interrupted=True 请求清除运行态，"
                "但 zero.purge_session 失败（exc=%s）：%s；**脏运行态仍在**，"
                "下一帧仍会在其上续跑。",
                session_id,
                type(exc).__name__,
                exc,
            )
            return False
        logger.warning(
            "graceful_step: session=%s 已按 purge_on_interrupted=True 清除该会话**全部**持久"
            "运行态（purged=%s，不可逆）。下一帧将以同 id 重开出一个空运行态的新会话。",
            session_id,
            purged,
        )
        return purged

    async def graceful_step(
        self,
        session_id: str | None,
        stimulus: AffectStimulus,
        priors: list[ModalityPrior] | None = None,
        *,
        resume_config: dict[str, Any] | None = None,
        purge_on_interrupted: bool = False,
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
        （+再供 `resume_config`）后**先看重开的返回体**，再决定是否重试一次 step**——Zero
        `ZERO_CHECKPOINT_BACKEND=sqlite` 时按 thread_id 自动续运行态，memory 后端则重开=全新会话
        （不报错）。重开或重试再失败 → 降级 None（只重试一次、不递归；但重试路径上的**不可降级
        错误同样上抛**）。⚠ SessionConfig 不进 checkpoint，未供 `resume_config` 则 resume 会话走
        Zero env 默认门控（非原会话 config）；须续原门控时调用方应传原 config。

        🛑 **半截运行态本帧不续跑**（2026-07-29）：若重开的返回体带非空 `interrupted_at`
        （上一轮被中途取消，运行态停在 super-step 边界），**不重试 step**，而是 **ERROR 级日志
        （带待执行节点名）+ 降级 None**。

        ⚠ **本机制的真实收益，如实表述**（上一版在此处过度宣称，2026-07-29 跨仓复核订正）：
        它买到的是「把一次**静默**续跑换成一条**响亮的 ERROR** + 本帧拒绝」，**不是**「避免污染」。
        污染并未被避免，只被推迟一帧 —— 因果链（Zero `daecce1` 现场核验）：
          · 止血判定只能在**重开之后**做（`interrupted_at` 来自 `open_session` 返回体），
            而重开这一步已经让该会话在 Zero registry 里**变活跃**；
          · 下一帧 `graceful_step` 因此不再报 unknown-session ⇒ 走正常 `step` 路径，
            而 Zero 的 `zero.step` **完全不做中断检查** ⇒ 照样在带 pending `next` 的线程上续跑；
          · 且 Zero 的 `interrupted_at()` 只在「resume 且**不活跃**」时探测（活跃分支提前
            return），⇒ 这条 ERROR 对同一 session **一生只出现一次**，此后污染不可观测。
        ⇒ 「有界一次性 vs 无界累积不可逆」的旧论证在实际控制流下**不成立**，已撤回。
        本帧拒绝要真正变成止血，须调用方**拿这条 ERROR 去做事**（轮换 session_id，或开
        `purge_on_interrupted`）；否则它只是一条更早、更响的告警。特征化守卫见
        `tests/mcp/test_zero_client.py::test_next_frame_after_interrupted_refusal_runs_normal_step`。
        ⚖ **措辞订正（2026-07-29 复审）**：上面这条因果链只证明**第一帧**躲不掉（判定必须在
        重开之后做），**不能**推出「后续帧 client 侧无解」。可实现的单侧缓解是存在的：client
        记一笔脏 session（`session_id → INTERRUPTED`），下一帧开头即拒绝或再报一次 ERROR。
        本轮**有意不做**，代价与理由：① 本 client 明示是**无状态句柄**（可服务多会话、
        `last_open_session` 已刻意不参与判定），加会话级记账等于反转该设计；② 多实例/多进程
        下各记各的，覆盖率天然不完整，会给出「已止血」的错觉；③ **清账时机无契约可依**——
        什么时候认为这个 session 又干净了？Zero 今天不提供任何「已回滚/已清理」的正向信号，
        只能靠 purge 成功反推，而 purge 是破坏性动作、默认关。⇒ 不是无解，是**权衡后不做**；
        真要做时上述特征化用例会变红，那是预期的。

        缺 `interrupted_at` 键（老部署）→ 行为与本次改动前逐字一致（零回归）。
        新 Zero 上「`resumed` 为真但 `interrupted_at` 缺席」这一格我方**不可判**（缺席四义：
        未探测·新建 / 未探测·活跃幂等重开 / 探测失败 / 探测成功且干净；其中第一义与
        `resumed=True` 互斥，故本格实际面对后三义）——处置是**照常续跑 + 一条可区分的
        WARNING**，理由见分支内注释（保守拒绝会误伤 100% 的健康 resume，等于废掉整个自愈能力）。

        🕳 **常规 resume 路径无守卫**（2026-07-29 复审揭出，如实披露）：上面这一整套探测/
        拒绝/purge **只挂在 unknown-session 自愈分支上**。调用方若自己 `open_session(
        session_id=…)` 拿到 `{resumed: true, interrupted_at: [...]}` 后再调 `step()` 或走本
        方法的**正常**路径（首次 step 就成功，压根不进 except），则全程只有 `open_session`
        里的那条 WARNING，**连 ERROR 都没有**，bundle 照常返回。
        本轮**有意不在正常路径加拦截**，理由：
          ① 该路径上的 `interrupted_at` 与自愈路径同源，但**处置权不在本方法**——正常路径的
             会话是调用方自己开的，它有 `last_open_session` 可读、有 `purge_session()` 可调，
             我方替它拒绝等于把「非关键路径丢一帧」的降级契约扩张成「替调用方否决其会话」；
          ② 零回归代价不对称：拒绝健康 resume 的误伤面在这里**比自愈路径大得多**——自愈路径
             一帧内已知 session 出过问题，正常路径则是每一次正常业务调用；
          ③ 真要收口，正确形态是**给调用方一个显式 API**（如 `open_session(...,
             refuse_if_interrupted=True)`）而非在降级路径里偷偷改语义 —— 那是另一次改动。
        缺口由 `test_normal_resume_path_has_no_interrupt_guard` 特征化钉住。

        🕒 **为何本轮仍用 `resumed` 做新老部署判别位**（临时性，如实标注）：判别本该问
        `zero.describe_config`（Zero `daecce1` 已落，带 `describe_config_version` +
        `error_codes` 全量表），而不是从 `resumed` 键在不在**反推**代际。本轮不接的理由：
          ① 它要在自愈分支里插一次**额外 round-trip**，而这条路径正是对端刚出过问题的时刻；
          ② 老部署没注册该工具 ⇒ 调用即 isError，得再写一层「工具不存在=老部署」的回退，
             判别链反而更长；
          ③ 本轮判定所需的位（`resumed` / `interrupted_at`）**就在已拿到的 open_session
             返回体里**，零额外调用。
        ⇒ 接上 `describe_config` 后，本方法与 `ZeroOpenSessionInfo.resumed` 上的代际判别应
        一并改挂 `describe_config_version`，届时可撤回向 Zero 索要的「承诺 `resumed`
        不被条件化」那条契约请求。

        Args:
            session_id:    Zero 会话 ID（None 时立即返回 None）。
            stimulus:      情感刺激。
            priors:        可选多模态先验列表。
            resume_config: unknown-session resume 重开时**再供的会话 config**（应与原 open_session
                           一致）；None → resume 会话走 Zero env 默认门控。
            purge_on_interrupted: 检出半截运行态时，是否额外调 `zero.purge_session` 清掉它。
                           **默认 False**，理由（这是破坏性动作，默认值须论证）：
                           ① 不可逆且**过度杀伤** —— Zero 的 purge 删该 thread 的全部
                              checkpoint 历史（含干净祖先），而真正需要的是「回滚一格」，
                              Zero 今天不暴露这种更弱的补救；
                           ② **判据有良性同形** —— 我方的证据只是 LangGraph 的 `next` 非空，
                              而 `next` 非空同样是「图停在 `interrupt()`/断点等待人工输入」
                              的正常表现。Zero 今天的图没有 interrupt 节点，但那是**对方的
                              实现细节**，不是契约；靠它来决定删不删数据不可接受；
                           ③ **层级错配** —— `graceful_step` 的契约就是「非关键路径，丢一帧
                              无所谓」，让全系统最可降级的那条路径去做全系统最不可逆的动作，
                              方向是反的；
                           ④ 「保留期多久、哪一侧是数据控制方」两侧都还没拍板（Zero 的
                              `purge_session` docstring 自己也这么写）。
                           开着它才是真正的止血：purge 后下一帧会重开出空运行态的新会话，
                           不再有「下一帧续跑」的残留缺口。代价是丢掉该会话的全部运行态历史。

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
                info = await self._open_session_info(
                    session_id=session_id,
                    config=resume_config,
                    downstream_guard=True,
                )
                # ── `interrupted_at` 四态决策表（每一格的处置都单独论证）──────────────
                # INTERRUPTED  确定半截    → 本帧拒绝 + ERROR（+ 可选 purge）
                # MALFORMED    契约漂移    → 照常续跑 + ERROR（不可判，但必须有人看见）
                # ABSENT       四义不可判  → 照常续跑；resumed 为真时补一条可区分 WARNING
                # CLEAN        明确干净    → 照常续跑，不打日志
                if info.interrupt_probe is ZeroInterruptProbe.INTERRUPTED:
                    # 🛑 重开的返回体带非空 interrupted_at ⇒ 上一轮被中途取消、运行态停在
                    # super-step 边界（已跑完节点的写入已落盘，sqlite 后端跨重启保留）。
                    #
                    # ⚠ **本帧拒绝买到的是什么，如实写**（2026-07-29 跨仓复核撤回上一版论证）：
                    # 买到的是「一次**静默**续跑 → 一条**响亮的 ERROR** + 本帧拒绝」。
                    # **没有**买到「避免污染」——污染只被推迟一帧：
                    #   (a) 止血判定只能在**重开之后**做，而重开已让该会话在 Zero registry
                    #       变活跃 ⇒ 下一帧不再报 unknown-session ⇒ 走正常 step 路径，
                    #       而 Zero 的 `zero.step` **不做任何中断检查**（现场核验 daecce1）
                    #       ⇒ 照样在带 pending `next` 的线程上续跑；
                    #   (b) Zero 的 `interrupted_at()` 只在「resume 且不活跃」时探测
                    #       （活跃分支提前 return）⇒ 这条 ERROR 对同一 session 一生只出现
                    #       一次，之后污染彻底不可观测。
                    # ⇒ 旧注释里「有界一次性 vs 无界累积不可逆」的对称性论证**不成立**，已删。
                    # 残留缺口有特征化守卫钉住：
                    # `test_next_frame_after_interrupted_refusal_runs_normal_step`。
                    #
                    # 那为什么仍然不重试？两条**站得住**的理由：
                    #   ① 本帧重试必然产出一个混合值并当作正常返回值交出去（Zero 自己的契约：
                    #      「续跑会从此处继续而非重跑整轮」）；拒绝则至少这一帧不发错值。
                    #   ② 这条 ERROR 是该污染**唯一一次**可观测的机会，它必须存在且醒目 ——
                    #      而只要还重试，日志就会被一个「成功返回」的表象冲淡。
                    # 真正的止血只有两条，都在调用方手里：轮换 session_id，或开
                    # `purge_on_interrupted`（破坏性，默认关，理由见本方法 docstring）。
                    #
                    # 🛑 **文案必须按 `purge_on_interrupted` 分支出**（2026-07-29 复审修）：
                    # 上一版无条件打同一条 ERROR，于是开着开关时会输出「脏运行态仍在、下一帧
                    # 将在其上续跑……请以 purge_on_interrupted=True 调用」，而紧随其后的
                    # WARNING 却是「已清除该会话**全部**持久运行态」——既与事实相反，又在劝
                    # 调用方去开一个**已经开着**的开关。两条文案的守卫分别锚
                    # `LOG_MARKER_INTERRUPTED_REFUSED{,_PURGING}`（不锚中文，见 marker 段注释）。
                    nodes = list(info.interrupted_at or ())
                    if purge_on_interrupted:
                        # 注意用词：此刻 purge **尚未执行**（下一行才发），且它可能失败
                        # （`_purge_after_interrupted` 失败时另出一条 ERROR）。故只说「已请求」
                        # 与「结果见随后日志」，不预先断言任何一种结局。
                        logger.error(
                            "%s graceful_step: session=%s resume 重开后发现上一轮被中途取消"
                            "——运行态停在 super-step 边界，待执行节点=%s；本帧拒绝续跑并降级 "
                            "None。已按 purge_on_interrupted=True 请求清除该会话**全部**持久"
                            "运行态（破坏性、不可逆），成败见随后日志：成功则下一帧以同 id "
                            "重开出空运行态的新会话、污染就此终止；失败则脏运行态仍在，"
                            "下一帧将走正常 step 路径在其上续跑。",
                            LOG_MARKER_INTERRUPTED_REFUSED_PURGING,
                            session_id,
                            nodes,
                        )
                        await self._purge_after_interrupted(session_id)
                    else:
                        logger.error(
                            "%s graceful_step: session=%s resume 重开后发现上一轮被中途取消"
                            "——运行态停在 super-step 边界，待执行节点=%s；本帧拒绝续跑并降级 "
                            "None。⚠ 这**不是**避免了污染：会话已被重开，脏运行态仍在，"
                            "**下一帧**将走正常 step 路径在其上续跑，且 Zero 只在 resume-不活跃 "
                            "时探测中断 ⇒ 本条 ERROR 对该 session 只会出现这一次。要真正止血请"
                            "轮换 session_id，或以 purge_on_interrupted=True 调用"
                            "（破坏性，不可逆）。",
                            LOG_MARKER_INTERRUPTED_REFUSED,
                            session_id,
                            nodes,
                        )
                    return None
                if info.interrupt_probe is ZeroInterruptProbe.MALFORMED:
                    # 契约漂移：对方发了这个键但形状不对。**不可判**，故不拒绝（拒绝会因对方
                    # 一个类型笔误就永久废掉自愈通路）；但必须 ERROR ——它是跨语言契约破裂的
                    # 直接证据，比任何一帧的数据都重要。
                    logger.error(
                        "%s graceful_step: session=%s resume 重开的返回体里 %r 形状非法"
                        "（跨仓契约漂移）——本帧**无法判定**是否半截，按续跑处置；"
                        "请核对 Zero 侧 open_session 返回体形状。",
                        LOG_MARKER_PROBE_MALFORMED,
                        session_id,
                        _OPEN_SESSION_KEY_INTERRUPTED_AT,
                    )
                elif info.interrupt_probe is ZeroInterruptProbe.ABSENT and info.resumed is True:
                    # 🛑 「resumed 为真但 interrupted_at 缺席」——**我方不可判**的一格。
                    # 缺席四义（Zero daecce1 现场核验，见模块头注释）：①未探测·新建会话
                    # （`resuming` 假）②未探测·活跃幂等重开（`registry.get(sid)` 非空 → 提前
                    # return）③探测抛异常被宽 except 吞掉 ④探测成功且干净。
                    # 本分支挂在 `resumed is True` 上 ⇒ ① 被排除，实际面对的是 ②③④ 三义。
                    # 其中 ③ 与我方要防的半截态**故障相关**（探测读的正是那份可能半写的
                    # checkpoint），所以这不是一格无害的不确定。
                    #
                    # 处置 = **照常续跑 + 一条可区分的 WARNING**，不保守拒绝。论证：
                    #   · 保守拒绝的代价是**结构性**的：③ 是每一次健康 resume 的正常形态
                    #     （Zero 探测干净就不发这个键），拒绝它 = 100% 的健康自愈都丢一帧，
                    #     而且永远无法好转（该态不会因为重试而变得可判）⇒ 等于废掉 T6·④
                    #     整个自愈能力，换来的只是对 ② 这一小概率子集的防护。
                    #   · 照常续跑的**边际**代价是有界的：它逐字等于本次改动之前的行为，
                    #     且如上所述，即便判出半截我方也只能推迟一帧 —— 亦即这一格的
                    #     "误放行" 与已知残留缺口同源，没有引入新的失败模式。
                    #   · WARNING 必须**可区分**（自己的文案 + resumed=True 标记），使运维能与
                    #     Zero 侧的 `zero.open_session 中断探测失败` ERROR 做跨仓时间对齐 ——
                    #     今天这是区分 ② 与 ③ 的**唯一**手段。
                    # 零回归：本分支挂在 `resumed is True` 上。`resumed` 是新老部署的判别位
                    # （新 Zero 无条件回，老部署根本不发）⇒ 老部署走 `resumed is None`，
                    # 不进本分支、不打这条 WARNING，行为与换代前逐字一致。
                    # ⏳ 该判别位是**间接推断**（靠对方实现细节，非契约）。Zero `daecce1` 已落
                    # `zero.describe_config`（带 `describe_config_version`）——那才是代际判别的
                    # 正解；本轮有意不接，论证见本方法 docstring 末段，接上后本行条件应改挂它。
                    logger.warning(
                        "%s graceful_step: session=%s resume 重开回了 resumed=True 但**未带** %r"
                        "——该键缺席有四义（未探测·新建 / 未探测·活跃幂等重开 / 探测失败 / "
                        "探测干净），其中首义与 resumed=True 互斥、余下三者在返回体上同形，"
                        "我方无法判定"
                        "上一轮是否被中途取消，本帧按续跑处置。若 Zero 侧同时刻有"
                        "「中断探测失败」记录，则本帧很可能续跑在半截运行态上。",
                        LOG_MARKER_PROBE_UNDECIDABLE,
                        session_id,
                        _OPEN_SESSION_KEY_INTERRUPTED_AT,
                    )
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
