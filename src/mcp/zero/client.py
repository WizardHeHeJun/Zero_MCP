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
import dataclasses
import json
import logging
import math
import os
import re
import secrets
import sys
import types
from collections.abc import Iterable, Mapping
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
from src.mcp.zero.external_priors import (
    EXTERNAL_PRIOR_SCHEMA_VERSION,
    _resolve_max_streams,
    _resolve_precision_cap,
    build_external_priors_override,
)

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
# ── 动作通道总开关未开（Zero 2026-08-11 回件 §3.1 确认：**新失效模式，不接管任何旧码支线**）。
# 现场核验（AST，剥 docstring/注释后按 Name 节点数）：对方全仓该常量恰 3 处——定义、
# describe_config 的 error_codes 集合、以及 `motion` 工具体内的唯一 raise 点。⇒ 只有调
# `zero.motion` 才可能遇到；本仓当前**不调该工具**，此处为**预登记**（守卫要求每个码留下
# 书面族归属判断，不得靠一条警告挂账）。
ZERO_ERROR_CODE_MOTION_DISABLED = "motion-disabled"

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
        ZERO_ERROR_CODE_MOTION_DISABLED,
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

    🛑 **上述归责与重试语义并非无条件成立，依赖对方 ≥ 其 PR #51（main `75e8a36`，2026-07-30）**：
    在此之前 Zero 对该 env **静默接受非正/非有限值**，而 `0` 与负数会让**锁空闲时也无条件超时**
    ——此时其文案里的「上一轮 step 仍在执行」与「可原样重试」**两句都是假的**（对方实证，
    撤掉校验的 9 格变异逐格跑过；`nan`/`inf` 则形同未设、step 照常成功）。
    对方现已在**读 env 那一步**拒收这两类值、改报 `[zero:deploy-env-invalid]`，语义恢复成立。
    ⇒ 若将来连的是**旧版 Zero**，收到本码时不可直接采信「未进内核」这一归责；
    我方无法自行验证对方的锁是否真的忙，这条只能靠对方的 env 校验保证——
    这是**归责语义**的依赖，不是安全守卫的依赖（安全面由 M8/M9 等出网守卫单边兜住，不依赖对方）。
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


class ZeroLinkMotionDisabledError(ZeroLinkNonDegradableError):
    """所连部署的**动作通道总开关未开**（Zero `[zero:motion-disabled]`，`ZERO_MOTION_ENABLED`
    默认关）——只有部署端开 env 并**重启 server** 才能好。

    🛑 **为什么不复用 `ZeroLinkDeployEnvError`**（两者都是「改 client 传参永远改不好」）：
    那一条的语义是部署端 env 值**不合法**＝配置坏了，该报警；本条是一个**合法的、有意
    的默认关闭态**——部署方没打算开这个能力。二者的正确处置不同：前者要惊动人去修，
    后者调用方应当**认命并停止再调 `zero.motion`**（记一次 INFO 即可，不该每轮报警）。
    混进同一个类会让「能力没开」和「部署坏了」在观测上不可区分，正是本仓分码的初衷。

    ⚠ 仍归 `ZeroLinkNonDegradableError`（Zero 2026-08-11 回件 §3.1 的建议，本仓采纳）：
    它**每轮必复现且 client 无法自愈**——归可降级会让 `graceful_step` 每轮静默
    `return None`，与「偶发抖动」在看板上不可区分，而它一次也不会自愈。

    ⚠ 触发面：仅在调用 `zero.motion` 时可能遇到。**本仓当前不调该工具**，故这是预登记；
    将来接入动作通道时，调用点应当把本异常当「能力未开」的正常分支处理，而非故障。
    """


class ZeroLinkSchemaIncompatibleError(ZeroLinkNonDegradableError):
    """**所连部署**的 `external_prior_schema_version` 与本仓不一致 —— 跨语言契约不兼容。

    由 `ZeroLinkClient.preflight_external_priors()` 在发流**之前**主动抛出（不是 Zero 回的错误
    码；`tool` 字段记为 `zero.describe_config`，即读出该结论的那个回读面）。

    🛑 **为什么这一条 raise、而错误码表漂移只 warn**（两者处置有意不同）：
      · 版本不一致 ⇒ 我方按 v{本仓} 构造的 `external_priors` 三元组，对方可能按另一套形状
        **解释成功**。被拒是响亮失败（`[zero:external-prior-invalid]` → CallerFault → 上抛），
        **没被拒却被误解**才是灾难 —— 后者不可观测，且污染的是内核后验。不能用「试试看会不会
        被拒」的方式发现它。
      · 错误码表只影响**归责与重试语义**，且每一次错分类都伴随一条 warning 日志（可观测），
        故 warn 足矣。

    ⚠ 它是 `ZeroLinkNonDegradableError` 子类：既有 `except ZeroLinkCallError` 的调用点能接住，
    而 `graceful_step` 的降级分支**不会**吞它（NonDegradable 一律上抛）。
    ⚠ 判据仅在 `describe_config_version` 属**本仓认识的版本**时才升级为 raise；版本不认识时
    我方对该键的读法本身就不可信，**在不可信的观测量上 raise 是错的** → 降级为 warn。
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
    ZERO_ERROR_CODE_MOTION_DISABLED: ZeroLinkMotionDisabledError,
}


# ── zero.open_session 响应形状（Zero 2026-07-29 换代：返回体「只增不改」）───────────
# Zero `zero.open_session` 现返回 `{session_id, resumed}`；resume 且探测到上一轮被中途取消时
# 另带 `{interrupted_at: [待执行节点名]}`。
#
# 🛑 **缺键即回落**（对现网 Zero 零回归·硬约束）：老部署只回 `{session_id}`，新键都读不到
# 时行为必须与换代前逐字相同——`open_session` 照常返回 session_id，不抛、不打额外日志。
# 这也是 Zero 侧「返回体只增不改」承诺的消费侧对价，守卫见
# `tests/mcp/test_zero_client.py::test_open_session_without_new_keys_is_zero_regression`。
#
# 形状防御：每个键都按「读不到就当没有」处理（记一条 warning 后回落），**任何形状异常都
# 不得让 open_session 炸** —— 会话生命周期不能因为一条观测量的类型不对就打不开。
#
# ══ 中断态判读走**双轨**（2026-07-30 起）══════════════════════════════════════════
# 同一件事（上一轮是否被中途取消）现在有**两条**来源，取哪条取决于所连部署的代际：
#
#   · **新轨（权威·优先）**：返回体带一个**恒存在**的 `interrupt_probe` 键，取值是 Zero 显式
#     声明的四态 `not_probed / clean / interrupted / probe_failed`（见下方 token 表）。
#     ⇒ **该键在就一律以它为判据**，不再看 `interrupted_at` 是否缺席。这是我方当初向对方
#     索要的东西：其中 `probe_failed` 那一格（对方探测**自己抛了**）此前在返回体上与
#     「探测干净」完全同形，正是最危险的一格。
#   · **老轨（仅老部署·保留不删）**：没有 `interrupt_probe` 键时，只能从 `interrupted_at`
#     这个键**在不在**去反推 —— 那条推断链就是下面的「缺席四义」。它**只对老部署成立**，
#     新部署上已被显式态取代（不要再把它当无条件事实读）。
#
# 🛑 **版本依赖必须标注**（`ZeroLinkLockTimeoutError` docstring 立的分界：安全守卫一律不依赖
#    对方状态，**能力探测 / 归责语义**可以依赖但须标注版本）。本面属**后者**：
#      · `interrupt_probe` 上线于 Zero **`667e923`**（其**未合并**工作树分支
#        `fix/stage60-purge-correctness`；main @ `75e8a36` 上**还没有**这个键），同批把
#        `DESCRIBE_CONFIG_VERSION` 1→2（见 `KNOWN_DESCRIBE_CONFIG_VERSIONS`）；
#      · 我方 stdio 默认 `ZERO_SERVER_CWD=D:\Zero` ⇒ 今天真连的正是对方**工作树**（有该键），
#        而任何 main 部署（或更早）都**不发**该键 ⇒ 老轨必须留着、且不得因此报异常；
#      · 依赖的内容说清：`probe_failed` 是「**对方**探测抛了」这一事实的唯一来源 —— 那份可能
#        半写的 checkpoint 只有 Zero 读得到，我方**无法独立验证**，故这一位只能靠对方给。
#        安全面（M8 自点燃上界 / M9 physio μv 契约等出网守卫）不在此列、一律单边兜住。
#      · ⚠ 对方 bump 纪律明写「② 某键的**值域/取值集合**变化（如 interrupt_probe 加一个新态）」
#        也要 bump ⇒ **将来可能有第五态**。故我方不认识的取值必须有兜底（见 `UNRECOGNIZED`），
#        跨仓取值集合的漂移由 `tests/mcp/test_zero_contract_crosscheck.py::
#        TestInterruptProbeCrosscheck` 提醒（日常 warn + STRICT 判红）。
#
# 🛑 `interrupted_at` **缺席有四义**（Zero `daecce1` 现场核验，2026-07-29 20:0x）——**这段只对
#    老部署（无 `interrupt_probe` 键）成立**，是老轨的判读依据，故保留不删；新部署上四义已被
#    对方逐格显式化（映射见 `_ZERO_PROBE_TOKEN_TO_STATE` 的注释）。在老轨上不得把缺席一律
#    当成「未中断，可安全续跑」——Zero `open_session` 里**四条**路径都会让该键缺席
#    （上一版标题写「三义」而正文枚举四条，2026-07-29 订正；计数词随认知修订而漂移，
#     故守卫**一律不得**锚在它上面，见下方 `LOG_MARKER_*`）：
#      ① **未探测·新建会话**：`resuming` 为假时整段探测被跳过（返回体 `resumed: false`）；
#      ② **未探测·活跃幂等重开**：`registry.get(sid) is not None` 分支**提前 return**
#         `{"session_id", "resumed": True}`，根本走不到探测（Zero 源码注释未列此义）；
#      ③ **探测失败**：`try: interrupted = await session.interrupted_at()` 外面是
#         `except Exception: logger.exception(...)`，异常被吞、`interrupted` 留在 None；
#      ④ **探测成功且干净**：`nxt or None` 返回 None。
#    ⚠ 其中 ③ 与我方要防的半截态是**故障相关**的：探测读的 (`graph.aget_state`) 正是那份
#      可能半写的 checkpoint。把缺席一律读成「安全」= 止血在最该生效时静默失效。
#    ⇒ 老轨的解析层必须把「键缺席」「键在但为空」「键在且非空」「键在但形状坏」四态**分开
#      表达**（见 `ZeroInterruptProbe`），由消费点自己决定每一格怎么处置。
_OPEN_SESSION_KEY_RESUMED = "resumed"
_OPEN_SESSION_KEY_INTERRUPTED_AT = "interrupted_at"
_OPEN_SESSION_KEY_INTERRUPT_PROBE = "interrupt_probe"


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
"""**老轨**：`resumed=True` 但 `interrupted_at` 缺席——缺席四义不可判，照常续跑 + 可区分告警。"""

LOG_MARKER_INTERRUPTED_ON_OPEN = "[zl:interrupted-at-open]"
"""`open_session` 判出 `INTERRUPTED`（**任何**调用路径，含无守卫的常规 resume）。"""

# ── 新轨（Zero `667e923` 起的显式 `interrupt_probe`）专属锚点 ──────────────────────
# 🛑 **open 面与决策面必须用不同 marker**：`graceful_step` 内部调 `_open_session_info`，两处日志
#    落在**同一个** caplog 里 ⇒ 若共用一个 marker，一条「决策点确实拒绝了」的断言会被 open 面
#    那条日志喂饱而恒真（pitfalls ⑥ 同类）。故 `*_ON_OPEN`（只报告）与 `*_REFUSED`（真拒绝）分列。
LOG_MARKER_PROBE_FAILED_ON_OPEN = "[zl:interrupt-probe-failed-at-open]"
"""对方显式回 `probe_failed`（它自己的探测抛了）——open 面只报告，不处置。"""

LOG_MARKER_PROBE_FAILED_REFUSED = "[zl:interrupt-probe-failed-refused]"
"""`probe_failed` → `graceful_step` 按**最坏情况**拒绝本帧（**不** purge，理由见分支注释）。"""

LOG_MARKER_PROBE_UNRECOGNIZED_ON_OPEN = "[zl:interrupt-probe-unrecognized-at-open]"
"""对方回了一个我方**不认识**的态（第五态 / 值非 str）——open 面只报告，不处置。"""

LOG_MARKER_PROBE_UNRECOGNIZED_REFUSED = "[zl:interrupt-probe-unrecognized-refused]"
"""未知态 → `graceful_step` 按最坏情况拒绝本帧（**不** purge），并提示去核对方新增了什么态。"""

LOG_MARKER_PROBE_NOT_PROBED_UNDECIDABLE = "[zl:interrupt-probe-not-probed-undecidable]"
"""**新轨**：对方显式回 `not_probed` 且 `resumed=True`（活跃幂等重开）——它没看，故仍不可判。"""

LOG_MARKER_PROBE_STATE_MISMATCH = "[zl:interrupt-probe-state-mismatch]"
"""`interrupt_probe` 与 `interrupted_at` 载荷**自相矛盾**（跨仓契约漂移）——按保守方向取态。"""

LOG_MARKER_DESCRIBE_NOT_REGISTERED = "[zl:describe-config-not-registered]"
"""所连部署**未注册** `zero.describe_config`（老部署）——经 `list_tools` **确证**，非猜测。"""

LOG_MARKER_DESCRIBE_CALL_FAILED = "[zl:describe-config-call-failed]"
"""`zero.describe_config` 调用失败，但工具**在册**（或连能力面都问不到）——不确定，不缓存。"""

LOG_MARKER_DESCRIBE_VERSION_UNKNOWN = "[zl:describe-config-version-unknown]"
"""`describe_config_version` 不在本仓认识的集合里 —— 降级为「只报告不强制」，不炸。"""

LOG_MARKER_DESCRIBE_FIELDS_DRIFT = "[zl:describe-config-fields-drift]"
"""返回体键集与本仓期望不符（少键 / 多键）—— 字段级判读按缺席处置，不炸。"""

LOG_MARKER_ERROR_CODE_TABLE_DRIFT = "[zl:error-code-table-drift]"
"""运行期错误码表与本仓手抄镜像不一致（对方多 / 本仓多）—— warn，绝不 raise。"""

LOG_MARKER_EXTERNAL_PRIOR_PREFLIGHT = "[zl:external-prior-preflight]"
"""发流前自检发现与所连部署的阈值/契约版本不一致（处置见各字段，schema 版本另抛异常）。"""


class ZeroInterruptProbe(StrEnum):
    """「上一轮是否被中途取消」的**我方判读**（双轨合流后共七态）。

    🛑 **这是我方的判读，不是对方的线上令牌**：对方的四态字符串（`not_probed`/`clean`/
    `interrupted`/`probe_failed`）经 `_ZERO_PROBE_TOKEN_TO_STATE` 映射进来，其余三态
    （`ABSENT`/`MALFORMED`/`UNRECOGNIZED`）在对方那边**没有对应物**——它们描述的是
    「对方没说」「对方说的读不懂」这类只有消费侧才有的处境。故请勿拿 `ZeroInterruptProbe(raw)`
    直接构造：新增令牌时那样写会抛 ValueError，而正确行为是落到 `UNRECOGNIZED`。

    为什么要一个枚举而不是 ``tuple | None``：`MALFORMED`/`ABSENT`/`CLEAN` 在节点名字段上会
    **全部塌缩成 ``None``**，消费点无从区分「对方没说」「对方说了但契约漂移」「对方明确说干净」。

    Attributes:
        ABSENT:       **老轨专属**：`interrupt_probe` 与 `interrupted_at` 两个键都缺席。
                      **我方不可判**——见上方缺席四义（未探测·新建 / 未探测·活跃幂等重开 /
                      探测失败 / 探测成功且干净）。⇒ 该态今天**等价于「所连部署是老代」**
                      （新部署恒发 `interrupt_probe`），归因日志就靠它与下面几态分开。
        CLEAN:        对方**明确**说探测过且无待执行节点。两轨都可产出：新轨 =
                      `interrupt_probe == "clean"`；老轨 = `interrupted_at` 在且为空序列
                      （Zero 今天 `nxt or None` 不发空表，故老轨这一格现网不出现，但契约
                      未禁止发，而「明确说干净」比「没说」强得多，值得留一格）。
        INTERRUPTED:  **确定**半截：运行态停在 super-step 边界，续跑从待执行节点继续而非重跑
                      整轮（Zero 自己的契约表述）。新轨 = `interrupt_probe == "interrupted"`；
                      老轨 = `interrupted_at` 是非空 ``list[str]``。
        MALFORMED:    **老轨专属**：`interrupted_at` 在但形状非 ``list[str]``，且没有
                      `interrupt_probe` 可依 ⇒ 跨语言契约已漂移，**不可判**。
        NOT_PROBED:   **新轨**：对方明确说「压根没探测」（新建会话，或活跃幂等重开时提前
                      return）。⚠ 这**不是**「干净」——它只是「对方没看」。配合 `resumed`
                      才能定性：`resumed is False` ⇒ 新建，没有旧运行态可污染，真安全；
                      `resumed is True` ⇒ 活跃幂等重开（缺席四义的第②义），**仍不可判**。
        PROBE_FAILED: **新轨**：对方的探测**自己抛了**。🛑 **不可判，且必须按最坏情况处置、
                      绝不可当 CLEAN** —— 探测读的正是那份可能半写的 checkpoint，故这一格与
                      要防的半截态**故障相关**：越是真出事的时候越可能落到这里。
                      这一格正是我方当初向对方索要显式化的目标（此前它与 CLEAN 在返回体上同形）。
        UNRECOGNIZED: **新轨兜底**：`interrupt_probe` 在，但取值不是我方认识的四个令牌之一
                      （对方新增了**第五态**），或其值根本不是 ``str``（形状漂移）。
                      同样按最坏情况处置 + 告警，理由见 `_parse_open_session_interrupt_state`。
    """

    ABSENT = "absent"
    CLEAN = "clean"
    INTERRUPTED = "interrupted"
    MALFORMED = "malformed"
    NOT_PROBED = "not-probed"
    PROBE_FAILED = "probe-failed"
    UNRECOGNIZED = "unrecognized"


# Zero 的**线上令牌** → 我方判读态。逐格即「缺席四义」被对方显式化后的落点：
#   "not_probed"   ← 四义之 ①（未探测·新建）与 ②（未探测·活跃幂等重开）**合并**成一个令牌
#                    ⇒ 单看它仍分不开这两义，须配 `resumed`（①=False / ②=True）才能定性。
#   "probe_failed" ← 四义之 ③（探测失败）。**本轮的主要收益**：从「与干净同形」变成可判可归因。
#   "clean"        ← 四义之 ④（探测成功且干净）。
#   "interrupted"  ← 原本就不靠缺席表达（该键在且非空），新轨只是把它也纳入同一位。
# 🛑 令牌是**对方的**字面量，与本仓枚举值刻意不同形（下划线 vs 连字符）：混用会让人以为
#    可以 `ZeroInterruptProbe(raw)` 直接转，而那条路在第五态上会抛异常而不是优雅降级。
_ZERO_PROBE_TOKEN_TO_STATE: dict[str, ZeroInterruptProbe] = {
    "not_probed": ZeroInterruptProbe.NOT_PROBED,
    "clean": ZeroInterruptProbe.CLEAN,
    "interrupted": ZeroInterruptProbe.INTERRUPTED,
    "probe_failed": ZeroInterruptProbe.PROBE_FAILED,
}

KNOWN_ZERO_INTERRUPT_PROBE_VALUES: frozenset[str] = frozenset(_ZERO_PROBE_TOKEN_TO_STATE)
"""本仓**认识**的 `interrupt_probe` 取值集合（现场核自 Zero `667e923` 的 `open_session`）。

跨仓漂移守卫：`tests/mcp/test_zero_contract_crosscheck.py::TestInterruptProbeCrosscheck`
把本集合与**对方源码里实际会赋给 `probe` / 写进返回体的字面量**逐值比对——对方按其 bump
纪律 ② 加第五态时，该守卫日常 warn、STRICT 判红，逼人现场核一遍再登记。
运行期遇到不在本集合里的取值 → `ZeroInterruptProbe.UNRECOGNIZED`（最坏情况处置 + 告警），
**不炸**：跨仓单边升级不该让会话打不开。
"""


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
                        ✅ **2026-07-30 订正（`describe_config` 已接入，但这一位仍然不撤）**：
                        上一版在此处写「这是临时判据，接上 `describe_config` 后应改挂
                        `describe_config_version`」——**该结论经核验不成立，已撤回**。
                        `resumed` 在此承担的是**两件事**，回读面只覆盖其中一件：
                          (a) **代际**（对方是不是会发中断观测量的那一代）—— `describe_config`
                              确实覆盖，且更直接（`available` 这一位甚至与版本号取值无关）；
                          (b) **本次 open 到底是不是 resume** —— 回读面**结构上给不出**：它是
                              部署级/会话级的**配置**面，回答不了「你刚才那一次调用走的是新建
                              还是续会话」。而下方 ABSENT 分支正是靠 (b) 排除掉「未探测·新建
                              会话」这一义（缺席四义里的第①义）。
                        ⇒ 拿版本号替换 `resumed` 会**丢掉 (b)**，把已排除的第①义放回不可判集合，
                        判别力不升反降。故 `resumed` 保留；向 Zero 索要的「承诺 `resumed` 不被
                        条件化」那条契约请求**同样不撤**，只是理由从「我方拿它当代际位」改成
                        「我方拿它判本次 open 的 resume 语义」（代际那半的依据可以撤）。
        interrupted_at: 待执行节点名（**载荷**，不是判据）；仅 ``interrupt_probe is
                        INTERRUPTED`` 时可能非空，其余态一律 ``None``/``()``。
                        ⚠ **不要**只看这一个字段做判定：``None`` 同时覆盖「对方没说」「形状坏」
                        「对方明确说干净」多义，判定一律读 `interrupt_probe`。
                        ⚠ 新轨下 ``CLEAN`` 对应 ``None``（对方干净时压根不发该键），老轨下
                        ``CLEAN`` 对应 ``()``（该键在且为空）——故**不可**拿 ``() vs None``
                        反推轨道，要判轨道读 `interrupt_probe_raw`。
        interrupt_probe: 中断态的**唯一判据**（七态，见 `ZeroInterruptProbe`）。
        interrupt_probe_raw: 对方 `interrupt_probe` 键的**原始令牌**，用于**归因**（不参与判定）：
                        · ``None`` ⇒ 该键缺席 = **老部署**（走缺席推断的老轨），**或** 该键在
                          但值非 ``str``（形状漂移，此时 `interrupt_probe` 恒为 ``UNRECOGNIZED``
                          ⇒ 两者仍可分：`UNRECOGNIZED` + raw ``None`` = 形状坏，
                          非 ``UNRECOGNIZED`` + raw ``None`` = 老部署）；
                        · 非 ``None`` ⇒ **新部署**（≥ Zero `667e923`）显式说的那个词，原样保留
                          **包括我方不认识的第五态**——日志要能把它打出来给人看，故不做归一化。
    """

    session_id: str
    resumed: bool | None = None
    interrupted_at: tuple[str, ...] | None = None
    interrupt_probe: ZeroInterruptProbe = ZeroInterruptProbe.ABSENT
    interrupt_probe_raw: str | None = None


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
    """**老轨**：从 open_session 响应体取 ``interrupted_at``，判读成**四态 + 节点名**。

    ⚠ **调用者注意**：本函数只看 `interrupted_at` 这一个键，其「缺席 ⇒ ABSENT ⇒ 四义不可判」
    的结论**只在没有 `interrupt_probe` 键时（老部署）才是判据**。新部署上判据是显式态，
    本函数退化为**只负责取节点名载荷**（返回的态被 `_parse_open_session_interrupt_state`
    丢弃，只留形状告警与「非空即正证据」这一位）。入口一律走
    `_parse_open_session_interrupt_state`，不要直接调本函数做判定。

    🛑 **「键缺席」与「键在但为空」必须分开**（2026-07-29 修的信息损失）：前者是「对方没说」
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


def _parse_open_session_interrupt_state(
    data: dict[str, Any],
) -> tuple[ZeroInterruptProbe, tuple[str, ...] | None, str | None]:
    """中断态判读的**唯一入口**：新部署读显式 `interrupt_probe`，老部署回落缺席推断。

    返回 ``(态, 节点名载荷, 原始令牌)``；原始令牌只用于**归因日志**，不参与判定
    （语义见 `ZeroOpenSessionInfo.interrupt_probe_raw`）。

    ── 双轨分派 ──
    · `interrupt_probe` **键缺席** ⇒ 老部署，整段回落到 `_parse_open_session_interrupted_at`
      的四态推断，且 raw 回 ``None``。🛑 **零回归靠这条**：老部署的返回体里没有这个键，
      本函数于是逐字等价于改动前的实现（连告警都不多一条）。
    · 键在 ⇒ 新部署，**以它为判据**；`interrupted_at` 降级为「节点名载荷 + 一位正证据」。

    ── 不认识的取值一律 `UNRECOGNIZED`，并按最坏情况处置（**不**当 CLEAN）──
    对方的 bump 纪律明写「某键的**值域/取值集合**变化（如 interrupt_probe 加一个新态）」也要
    bump ⇒ **第五态是被明确预告的事**。此时三条路里只能选一条：
      ① 当 CLEAN（乐观）—— 直接违反本字段存在的理由：新态很可能正是又一种「不可判/出事了」，
         乐观解释会让止血在最该生效时静默失效，与我方当初索要显式化的动机完全相反；
      ② 抛异常 —— 让对方单边加一个态就能让我方**会话打不开**，跨仓单边升级零回归的红线不许；
      ③ **按最坏情况 + 告警**（本实现）—— 行为等同 `PROBE_FAILED`（不可判 ⇒ 拒绝本帧，
         见 `graceful_step`），代价上界是「在可降级路径上丢一帧并留一条响亮日志」，
         而收益是「对方任何新态都不会被我方**静默**误读成安全」。
    ⚠ 代价如实写：若对方新增的是一个**良性**态（如「本部署按配置关掉了探测」），本实现会在
    该部署上让自愈路径每帧丢一帧 —— 这是**有意选的方向**（宁可吵不可静默），且有两道提前
    预警把它挡在真部署之前：跨仓取值集合守卫（STRICT 判红）与 `describe_config_version`
    bump 守卫。收到告警就该现场核一遍再把新令牌登记进 `_ZERO_PROBE_TOKEN_TO_STATE`。

    ── 自洽性：**正证据优先**（两侧都发但互相矛盾时）──
    对方今天的实现里 ``probe == "interrupted"`` 与 ``interrupted_at`` 非空是**等价**的
    （`interrupted is not None` 是同一个条件的两处使用），故下面两格今天都不该出现；但那是
    **对方的实现细节**，我方不靠它吃饭：
      · 载荷是**well-shaped 非空** list[str] 而态不是 `INTERRUPTED` ⇒ **取 INTERRUPTED**。
        理由：非空待执行节点名是半截态的**正证据**，而令牌只是一句声明；两者冲突时按证据
        取保守方向（若反过来信令牌，"clean" + 一串待执行节点 = 我方明知有节点仍放行）。
      · 态是 `INTERRUPTED` 而载荷缺席/为空/形状坏 ⇒ **仍取 INTERRUPTED**（令牌是判据，
        载荷只是细节），节点名按空表报，日志里说明拿不到节点名。
    两格都打 `LOG_MARKER_PROBE_STATE_MISMATCH`（WARNING）：它是跨仓契约漂移的直接证据，
    但不改变「保守」这一方向，故不在解析层升到 ERROR（真拒绝时决策点会另出 ERROR）。
    """
    # 载荷先解析（形状告警照旧从这里出），新轨只用它的「非空正证据」这一位。
    payload_probe, nodes = _parse_open_session_interrupted_at(data)
    if _OPEN_SESSION_KEY_INTERRUPT_PROBE not in data:
        # 老轨：逐字回落到改动前的判读（老部署零回归）。
        return payload_probe, nodes, None

    raw = data[_OPEN_SESSION_KEY_INTERRUPT_PROBE]
    if not isinstance(raw, str):
        logger.warning(
            "%s zero.open_session 返回的 %r 形状非预期（期望 str，实得 %s=%r）——判为契约漂移，"
            "按**最坏情况**处置（等同 probe_failed），**不**等同于「未中断」。",
            LOG_MARKER_PROBE_UNRECOGNIZED_ON_OPEN,
            _OPEN_SESSION_KEY_INTERRUPT_PROBE,
            type(raw).__name__,
            raw,
        )
        return ZeroInterruptProbe.UNRECOGNIZED, nodes, None

    state = _ZERO_PROBE_TOKEN_TO_STATE.get(raw)
    if state is None:
        logger.warning(
            "%s zero.open_session 返回的 %r=%r 不在本仓认识的取值集合 %s 里——对方很可能按其 "
            "bump 纪律②新增了一个态。按**最坏情况**处置（等同 probe_failed，不可判 ⇒ 拒绝本帧），"
            "**绝不**当 clean；请现场核对方 open_session 的新态语义，确认后登记进 client 的 "
            "_ZERO_PROBE_TOKEN_TO_STATE。",
            LOG_MARKER_PROBE_UNRECOGNIZED_ON_OPEN,
            _OPEN_SESSION_KEY_INTERRUPT_PROBE,
            raw,
            sorted(KNOWN_ZERO_INTERRUPT_PROBE_VALUES),
        )
        return ZeroInterruptProbe.UNRECOGNIZED, nodes, raw

    # ── 自洽性核对（正证据优先，见 docstring）──
    if (
        payload_probe is ZeroInterruptProbe.INTERRUPTED
        and state is not ZeroInterruptProbe.INTERRUPTED
    ):
        logger.warning(
            "%s zero.open_session 自相矛盾：%r=%r（非 interrupted）却带**非空** %r=%r。"
            "按**正证据**取 INTERRUPTED——非空待执行节点名是半截态的直接证据，令牌只是声明；"
            "请核对方两处是否已不同源。",
            LOG_MARKER_PROBE_STATE_MISMATCH,
            _OPEN_SESSION_KEY_INTERRUPT_PROBE,
            raw,
            _OPEN_SESSION_KEY_INTERRUPTED_AT,
            list(nodes or ()),
        )
        return ZeroInterruptProbe.INTERRUPTED, nodes, raw
    if state is ZeroInterruptProbe.INTERRUPTED and not nodes:
        logger.warning(
            "%s zero.open_session 自相矛盾：%r=%r 但 %r 缺席/为空/形状坏 ——**仍按 INTERRUPTED "
            "处置**（令牌是判据），只是本帧拿不到待执行节点名，诊断信息少一份。",
            LOG_MARKER_PROBE_STATE_MISMATCH,
            _OPEN_SESSION_KEY_INTERRUPT_PROBE,
            raw,
            _OPEN_SESSION_KEY_INTERRUPTED_AT,
        )
        return ZeroInterruptProbe.INTERRUPTED, nodes, raw
    return state, nodes, raw


# ── zero.describe_config：**运行期**回读所连部署真正生效的门控（Zero main `75e8a36` 上线）──
#
# 为什么必须接（本仓 2026-07-29 向 Zero 提、对方落地）：此前我方**无手段确认**所连部署到底开了
# 哪些门 —— open/step/close 三个工具都不回显配置，而 HTTP 传输下两进程**不共享 env**，
# 「两仓同名 env 对齐」这条机制在结构上就不成立，跨仓 env 对照表只能当文档、不能当校验。
#
# 🛑 **它属「能力探测 / 归责语义」，不属「安全守卫」**（`ZeroLinkLockTimeoutError` docstring 立的
#    分界）：安全面（M8 自点燃上界 / M9 physio μv 契约）一律**单边**兜住、不依赖对方状态；
#    本回读面依赖对方，故**必须标注版本**、且不可用时必须优雅回退。任何一条基于它的判定，
#    在「对方没这个工具」时都只能降级、不能变成硬失败。
_DESCRIBE_CONFIG_TOOL = "zero.describe_config"
_DESCRIBE_CONFIG_KEY_VERSION = "describe_config_version"
_DESCRIBE_CONFIG_KEY_RESOLVED = "resolved_for_session"
_DESCRIBE_CONFIG_KEY_ERROR_CODES = "error_codes"
_DESCRIBE_CONFIG_KEY_PRECISION_CAP = "external_prior_precision_cap"
_DESCRIBE_CONFIG_KEY_MAX_STREAMS = "max_external_streams"
_DESCRIBE_CONFIG_KEY_SCHEMA_VERSION = "external_prior_schema_version"

DESCRIBE_CONFIG_EXPECTED_KEYS: frozenset[str] = frozenset(
    {
        _DESCRIBE_CONFIG_KEY_VERSION,
        "session_id",
        _DESCRIBE_CONFIG_KEY_RESOLVED,
        "workspace_enabled",
        "gate_fusion",
        "exclude_physio_fusion",
        "precision_commensurable",
        "ignition_beta",
        "coping_potential_enabled",
        "text_coping_enabled",
        "fear_domain_enabled",
        "canonical_physiology",
        "facs_extended",
        _DESCRIBE_CONFIG_KEY_PRECISION_CAP,
        _DESCRIBE_CONFIG_KEY_MAX_STREAMS,
        _DESCRIBE_CONFIG_KEY_SCHEMA_VERSION,
        "governance_gated_flags",
        _DESCRIBE_CONFIG_KEY_ERROR_CODES,
        "sample_sigma_cap",
        "affect_readout",
        "weights_version",
    }
)
"""本仓**独立持有**的**必需**键集（21 键，v1 起就有，现场核自 Zero
`src/mcp_server/server.py::describe_config`）。缺任一 ⇒ 我方那一位读不到，判读静默降级。

不是「对方回什么就认什么」：跨仓漂移由 `tests/mcp/test_zero_contract_crosscheck.py::
TestDescribeConfigCrosscheck` 静态判红，运行期不符只降级+warn（见 `ZeroDeployConfig.describe`）。

⚠ 与 `DESCRIBE_CONFIG_OPTIONAL_KEYS` 的分工见后者 docstring——**不要把新版本才有的键加到这里**，
那会让我方对旧部署误报缺键。
"""

DESCRIBE_CONFIG_OPTIONAL_KEYS: frozenset[str] = frozenset(
    {
        "transport",
        "stateless_http",
        # v4 起（Zero `0effea7`，即 bump 3→4 那一次提交本身）新增的后端回读三键。
        "checkpointer_impl",
        "memory_store_impl",
        "semantic_store_impl",
    }
)
"""**已登记但按版本可缺**的键：对方 v3 起新增，v1/v2 部署上不存在。

为什么需要这一层（2026-07-30 实测踩到）：字段集守卫的判据是「我方期望集与对方返回体**逐键相等**」，
`extra`（对方有、我方未登记）判红是**有意**的——它提醒「我方漏读了对方的新能力」。
但若把这类新键直接并进 `DESCRIBE_CONFIG_EXPECTED_KEYS`，对**旧部署**（21 键）就会反过来报缺 2 键
⇒ 把「对方版本旧」误报成「契约漂移」。两个方向都要不误报，就必须分层：
**必需键缺 ⇒ 真问题；可选键缺 ⇒ 只说明对方版本旧。**

⚠ 分层**不削弱守卫**：对方若再加第 24 个键，它仍落进 `extra` 并判红（判别力已实证）。
分层只承认「我方已知道这两个键存在」，**不等于已消费**——见下面「尚未消费」。

**尚未消费（下一轮待办，与 `interrupt_probe` 同型的「信息扔掉」风险）**：
- `transport` = 对方**实际起的**传输（不是我方以为的那个）⇒ 可用于校验我方配置与对方实况一致；
- `stateless_http` ⇒ 若对方是 stateless，**每次请求独立、session 语义完全不同**，
  我方的 resume / session 管理前提可能整体不成立。这条影响面比 `transport` 大得多，须专门评估。
- **后端回读三键**（下详）⇒ 可用于判「对方到底跑在什么后端上」（内存态 vs 落盘态直接影响
  我方对 resume 的预期）。⚠ 未消费**不等于信息被扔掉**：三键的值经 `ZeroDeployConfig.fields`
  原样透传，随时可读；`OPTIONAL` 分层只影响 `missing_keys`/`absent_optional_keys` 的记账。

━━ 后端回读三键（v4 起）：`checkpointer_impl` / `memory_store_impl` / `semantic_store_impl` ━━

🛑 **为什么登记进「可选」而不是「必需」——尽管 Zero 明确建议了 EXPECTED**（2026-08-11 回件
§3.2）：两侧的「EXPECTED」不是同一件事，对方的建议在**它自己的轴上完全成立**，只是与本集合
的语义不同轴：
  · 对方说的是**门控轴**：「键恒出现、不随 feature flag 增删」——这一点现场核过，属实
    （返回体是字面量三键恒在，值走 `... if session else None`）。
  · 本集合说的是**版本轴**：「我方支持的**每一个代际**上都有」。而这三键是 v4 才新增的
    （Zero `0effea7` 同一次提交引入三键并 bump 3→4；`git log -S` 逐个核过），v1/v2/v3 上
    **不存在**。放进必需集会让我方对旧部署反向误报「缺 3 键」——把「对方版本旧」误报成
    「契约漂移」，正是本可选层被造出来要避免的那个方向。
⇒ 结论：**门控恒在 ≠ 各代际恒在**。已就此回复对方（同分支回件 §2）。分层不削弱守卫：
对方若再加第 27 个键，它仍落进 `extra` 判红。

**值的语义（现场核自 Zero `aa531b2`，消费前必读——三者都不是「有/没有」两态）**：
  · 传了 `session_id` ⇒ 值是**该会话实际构造出的类名**；
  · 不传 sid（部署端默认面）⇒ 值恒为 `null`，含义是「**不可知**」而**不是**「没有」。
    对方有意为之：那一面没有会话实例可读，现构造一次会有副作用（sqlite 建目录开连接 /
    graphiti 连 Neo4j），故按「不可知项显式回 null」处置，不拿 env 字面量充数。
  · ⚠ `semantic_store_impl` 是**三态**：`null`（无 sid ⇒ 不可知）/ `"disabled"`（已解析且
    语义后端关闭，这是默认）/ 实际类名（开启）。**把「关闭」与「不可知」都读成 null 就丢了
    判别力**，消费时必须分开。
  · ⚠ 回的是**实际构造出的类名、不是 env 字面量**：两个后端工厂在依赖缺失时会**静默回退**
    （`neo4j` 缺驱动 → InMemory；`sqlite_vec` 缺依赖 → None）⇒ 拿这三键判「对方到底跑在
    什么后端上」可靠，拿 env 猜不可靠。这正是对方 `0effea7` 的提交题意（「env 名证明不了
    『全内存』」）。
"""

KNOWN_DESCRIBE_CONFIG_VERSIONS: frozenset[int] = frozenset({1, 2, 3, 4})
"""本仓**已逐键核验过**的 `describe_config_version`。不在此集合 ⇒ 只报告不强制（见下）。

现场核验（2026-07-30，只读 D:\\Zero；两版逐键比对经 AST 取 `describe_config` 的 return 字面量）：
- **v1** = Zero `origin/main` @ `75e8a36`：上表 21 键（`DESCRIBE_CONFIG_VERSION = 1`）。
- **v2** = Zero **未合并**的工作树分支 `fix/stage60-purge-correctness` @ `667e923`
  （`DESCRIBE_CONFIG_VERSION = 2`；其父提交 `218771a` 上仍是 1，bump 就发生在 `667e923`）：
  `describe_config` 返回体与 v1 **逐键相同**（21 键，连顺序都一样）；bump 的真实动因**不在
  describe_config 自身**，而在 `zero.open_session` ——该提交把「`interrupted_at` 缺席」拆成显式
  四态 `interrupt_probe`（not_probed / clean / interrupted / probe_failed），属新契约故 1→2；
  对方同时把该常量的措辞从「字段集版本」改成「**契约**版本」，并把 bump 纪律扩到
  ①增删键 ②某键值域变化 ③某键语义变化。⇒ v1/v2 对**我方读的这 21 键**等价，故同列为「认识」。
- **v3** = 对方**未提交工作树**（`DESCRIBE_CONFIG_VERSION = 3`；其 HEAD `e9dc79c` 上是 2、
  `origin/main` 上仍是 1 —— **同一时刻三态并存**）：返回体从 21 键增到 **23 键**，
  新增 `transport`（对方实际起的传输）与 `stateless_http`，动因是把 `__main__.main` 与
  `describe_config` 的传输解析**收敛到同一符号**（对方注释：「describe_config 报传输的全部价值
  就在于回的值与真正起传输的那段代码同源」）。属**增删键** ⇒ 按其纪律 ① 必须 bump。
  ⇒ v3 与 v1/v2 **不是逐键相同**，故那两键登记进 `DESCRIBE_CONFIG_OPTIONAL_KEYS` 而非必需集。

- **v4** = Zero **main** `0effea7`（`DESCRIBE_CONFIG_VERSION = 4`；现场核于其 HEAD `aa531b2`、
  工作树 clean，2026-08-11）：返回体从 23 键增到 **26 键**，新增 `checkpointer_impl` /
  `memory_store_impl` / `semantic_store_impl`（后端回读三键，登记进
  `DESCRIBE_CONFIG_OPTIONAL_KEYS`，值语义见该处）。三键与 bump 由**同一次提交**引入
  （`git log -S` 逐键核过）⇒ 符合对方 bump 纪律 ①（增删键）。对 v1–v3 的 23/21 键**逐键
  向后兼容**（只增不改），我方读的那 21 必需键语义未动。
  **核验方法**（写明是为了下一个审查者不必重做这趟）：`git show 0effea7 -- src/mcp_server/
  server.py` 对 `describe_config` 函数体只有两处改动——版本号常量与其注释、返回字典**末尾
  追加**三键；既有 21 键对应的代码行**一行未动**。⇒ 对方 bump 纪律 ③（某键语义变化）在本
  次未被触发。⚠ 这是**代码行未动**的举证，不等于「语义在别处未被间接改变」（如某键的值
  来源函数被改），后者只能靠对方遵守自己的 bump 纪律。
  ✅ **这一版首次是从对方 `main` 且工作树 clean 的状态读到的**——与 v2/v3 都读自未提交
  工作树不同（见下方那条「同一个坑的第二次」）。此处记一笔，是为了将来能看出哪几版当时
  只活在对方工作树里、哪几版是主干固化的。

  ⚠ 我方 stdio 传输默认 `ZERO_SERVER_CWD=D:\\Zero`，即真正连的是对方**工作树**（今天 = v4），
  不是 main（v1）。三版都得认，否则今天的 live 调用会全程降级。
  ⚠ **这是同一个坑的第二次**：v2 与 v3 都是从对方**未提交工作树**读到的。我方每次「跟随」
  都在把守卫从「提醒」变成「追认」。缓解不是不跟随（不跟随则 live 全程降级），而是
  **把三态差异显式记在这里**——将来看到我方认识 {1,2,3} 而对方 main 只有 1 时，
  能立刻知道中间两版从未被对方主干固化过。

🛑 **这条「认识」有保质期**（写明是为了将来能查出它何时失准）：v2 此刻只活在对方一条未合并
分支上，其语义**尚未被 main 固化**。若对方在合入前**重写 v2 的内容**（同一个数字 2 换一套
含义），我方这里的「已核验 = 等价」就直接失准，而运行期看不出来——版本号一样，读法照旧，
结论悄悄变错。两道守卫是缓冲、不是保险：**字段集守卫**（`TestDescribeConfigCrosscheck::
test_field_set_matches_client_expectation`，硬红）在增删键时当场红；**版本守卫**
（同类 `test_version_is_known_to_client`，日常 warn + STRICT 判红）在版本号再次 bump 时逼人
现场核。二者都盖不住「版本号不变而某键**语义**变了」——那一格只能靠对方遵守自己写下的
bump 纪律 ③，故若发现对方改了 v2 的内容却没再 bump，应视为跨仓契约事故上报，而非本地绕过。

🛑 **不认识的版本一律降级、不炸**，因为按对方 v2 起的 bump 纪律，「某键**语义**变了」也会
bump —— 那正是「同名同类型但含义变了、消费方看不出来」的一类，我方**照旧解析会得出错误结论**。
故：不认识 ⇒ ① 仍解析、仍报告（信息不丢）；② 但任何「强制」动作降级为 warn
（`preflight_external_priors` 的 raise 是唯一的强制动作，见其 `strict` 参数）；③ 打一条
`LOG_MARKER_DESCRIBE_VERSION_UNKNOWN`。

⚠ **例外：代际判别不受版本约束**。「返回体里有没有 `describe_config_version` 这个键」这件事
本身与它的取值无关 ⇒ `ZeroDeployConfig.available` 对任意整数版本都成立，这是本回读面上唯一
**版本无关**的判据。
"""


class ZeroConfigProbe(StrEnum):
    """`describe_config()` 这一次探测的**结局**（四态，缺一就分不清「没有」与「没问到」）。

    Attributes:
        OK:             调通且返回体是 JSON object。
        NOT_REGISTERED: 对方**未注册**该工具（老部署）——经 `list_tools` 确证，**不是**从错误
                        文案猜的（文案是脆弱锚点，pitfalls ⑦）。这一态**可负缓存**：工具不在册
                        与 session_id 无关，短路后续所有探测，避免每次自检都白付 2 个 RTT。
        CALL_FAILED:    调用失败，但工具**在册**、或连 `list_tools` 都问不到 ⇒ **不确定**。
                        与 NOT_REGISTERED 分开的理由：把一次网络抖动缓存成「老部署」，会让本
                        连接的剩余生命期里所有自检永久降级，且**看不出降级是错的**。本态**不缓存**。
        MALFORMED:      调通了但返回体不是合法 JSON object（跨仓契约漂移）。同样不缓存。
    """

    OK = "ok"
    NOT_REGISTERED = "not-registered"
    CALL_FAILED = "call-failed"
    MALFORMED = "malformed"


@dataclass(frozen=True, slots=True)
class ZeroDeployConfig:
    """`zero.describe_config` 的一次判读结果（不可用时也是一个合法值，**不是 None/异常**）。

    Attributes:
        probe:                本次探测结局（见 `ZeroConfigProbe`）。
        detail:               人读原因串（`probe is OK` 时为 ""）——判据返回**原因**而非 bool，
                              便于测试逐格断言「红在正确的原因上」。
        version:              `describe_config_version`；缺键/形状非 int → ``None``。
        resolved_for_session: 对方是否按**该会话真实生效**的值回答（未知 sid 视同不传 ⇒ False）。
        fields:               原始返回体只读视图（`probe is OK` 之外恒为空）。
    """

    probe: ZeroConfigProbe
    detail: str = ""
    version: int | None = None
    resolved_for_session: bool = False
    fields: Mapping[str, Any] = types.MappingProxyType({})

    @property
    def available(self) -> bool:
        """对方**有**这个回读面且本次调通 —— 唯一版本无关的判据（代际判别即用这一位）。"""
        return self.probe is ZeroConfigProbe.OK

    @property
    def version_known(self) -> bool:
        """版本在 `KNOWN_DESCRIBE_CONFIG_VERSIONS` 内（``None`` 不在任何集合内，自然为 False）。"""
        return self.version in KNOWN_DESCRIBE_CONFIG_VERSIONS

    @property
    def enforceable(self) -> bool:
        """可用**且**版本认识 ⇒ 允许把不一致升级成硬失败；否则一律只报告。"""
        return self.available and self.version_known

    @property
    def missing_keys(self) -> tuple[str, ...]:
        """**必需**键里对方没回的（排序稳定，便于断言）。

        只算 `DESCRIBE_CONFIG_EXPECTED_KEYS`：`DESCRIBE_CONFIG_OPTIONAL_KEYS`（对方 v3 起才有）
        缺席**不算缺**——那只说明对方版本旧，把它算进来会让我方对旧部署误报契约漂移。
        """
        return tuple(sorted(DESCRIBE_CONFIG_EXPECTED_KEYS - set(self.fields)))

    @property
    def absent_optional_keys(self) -> tuple[str, ...]:
        """已登记的**可选**键里对方没回的 —— 只表示「对方版本旧」，**不是**问题。

        与 `missing_keys` 刻意分开：混在一起就无法区分「契约漂移」与「对方还没升到那一版」。
        """
        return tuple(sorted(DESCRIBE_CONFIG_OPTIONAL_KEYS - set(self.fields)))

    @property
    def unexpected_keys(self) -> tuple[str, ...]:
        """对方回了、本仓**两个集合都没登记**的键 —— 通常是**正常演进**（新增能力），不是错误。

        ⚠ 已登记的可选键**不算** unexpected：否则连了 v3 部署就会每次都报「多了两键」。
        真正落进这里的是我方**尚不知道**的键 ⇒ 提示我方漏读了对方的新能力。
        """
        known = DESCRIBE_CONFIG_EXPECTED_KEYS | DESCRIBE_CONFIG_OPTIONAL_KEYS
        return tuple(sorted(set(self.fields) - known))

    def describe(self) -> str:
        """一行人读判读串（供日志与测试断言；**不要**用它做程序判定，判定读属性）。"""
        if not self.available:
            return f"describe_config 不可用（{self.probe.value}）：{self.detail}"
        bits = [
            f"version={self.version}",
            "版本已核验" if self.version_known else "⚠ 版本不认识（只报告不强制）",
            f"resolved_for_session={self.resolved_for_session}",
        ]
        if self.missing_keys:
            bits.append(f"⚠ 缺键={list(self.missing_keys)}")
        if self.unexpected_keys:
            bits.append(f"新增键={list(self.unexpected_keys)}")
        return "；".join(bits)


_DESCRIBE_CONFIG_NOT_PROBED = ZeroDeployConfig(
    probe=ZeroConfigProbe.CALL_FAILED,
    detail="尚未探测（本对象由调用方直接构造，用于纯判读函数的单测）",
)


def _read_int_field(fields: Mapping[str, Any], key: str) -> int | None:
    """取 int 字段；缺键/形状不符 → ``None``（**不猜**）。

    ⚠ 显式排除 ``bool``：``isinstance(True, int)`` 为真，不排除会把 `gate_fusion` 这类布尔门
    误读成 0/1 的数值旋钮 —— 正是 Zero 要求「按字段名显式取值、不得用类型过滤器」要避开的坑。
    """
    value = fields.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _read_float_field(fields: Mapping[str, Any], key: str) -> float | None:
    """取 float 字段（int 亦收，JSON 里 `1` 与 `1.0` 同形）；缺键/形状不符/非有限 → ``None``。"""
    value = fields.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _read_str_tuple_field(fields: Mapping[str, Any], key: str) -> tuple[str, ...] | None:
    """取 `list[str]` 字段；缺键或**任一**元素非 str → ``None``（整条丢弃，不逐元素过滤）。

    与 `_parse_open_session_interrupted_at` 同口径：混入非 str 说明契约已漂移，此时
    「部分读到」比「读不到」更危险 —— 会让调用方以为自己拿到了完整集合。
    """
    value = fields.get(key)
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        return None
    return tuple(value)


@dataclass(frozen=True, slots=True)
class ZeroErrorCodeDiff:
    """本仓 `ZERO_ERROR_CODES` 手抄镜像 vs **所连部署**运行期回的 `error_codes` 全表。

    ⚠ **覆盖面要如实说**：对方回的是它**登记**的码表（`sorted(ZERO_ERROR_CODES)`），
    **不是**「它会产出哪些码」。所以本比对能发现「登记面漂移」，发现不了 m8 那类
    「把某个失效模式重切到另一个已登记码」——那一格运行期原理上不可见（我方看得见码、
    看不见产出点语义），只能靠 `test_zero_contract_crosscheck.py` 的源码级守卫，
    且已定性为**要向 Zero 索取的契约**而非我方能单边补的守卫。

    Attributes:
        checked:     是否真比对了（回读面不可用 / 缺 `error_codes` 键 → False）。
        reason:      未比对的原因；`checked` 为真时是判读结论串。
        zero_only:   对方有、本仓无。
        client_only: 本仓有、对方无。
    """

    checked: bool
    reason: str
    zero_only: tuple[str, ...] = ()
    client_only: tuple[str, ...] = ()

    @property
    def in_sync(self) -> bool:
        """两侧登记面完全一致（未比对时为 False —— 「没比」不等于「一致」）。"""
        return self.checked and not self.zero_only and not self.client_only

    @property
    def rename_suspected(self) -> bool:
        """两侧同时非空 ⇒ 很可能是**改名**（一删一增），而不是各自独立地增/删。

        运行期只拿得到码**值**、拿不到符号名，故「值不同」这一情形只能以此联合信号呈现。
        """
        return bool(self.zero_only) and bool(self.client_only)


def diff_error_codes(
    cfg: ZeroDeployConfig,
    *,
    client_codes: Iterable[str] = ZERO_ERROR_CODES,
) -> ZeroErrorCodeDiff:
    """比对错误码表（**纯函数**，无 I/O、永不抛 —— 可脱离连接逐格单测）。

    🛑 **三种不一致的处置有意不同**（都不 raise，但严重度与文案不同）：

    · **对方多（`zero_only`）= 正常演进**。我方 `_CODE_TO_EXCEPTION` 查不到即退回基类
      `ZeroLinkCallError` + 一条 warning，跨仓单边升级零回归。⚠ 但**不能就此宣称无害**：
      若对方是把某个既有失效模式**切分**到新码（如 payload-invalid 的一支切成 stim-invalid），
      我方就不是「多一条没归类的新错误」，而是**既有归类被掏空**——CallerFault（不可降级、
      上抛）退化成基类（可降级）→ 被 `graceful_step` 吞成每轮静默 None。运行期分不出这两种
      情形（见 `ZeroErrorCodeDiff` 的覆盖面说明），故文案必须点名让人去查。
    · **本仓多（`client_only`）= 我方拿着过期认知**，比上一种重：我方表里那条**永远不会命中**，
      而对方那个失效模式现在换了别的码回来 ⇒ 落未登记码 → 回基类；若原码属**不可降级族**，
      归责就从「上抛」退化成「静默降级」。
    · **两侧同时非空 = 疑似改名**（`rename_suspected`），优先按改名查，别当成一增一删两件事。

    🛑 **为什么全部只 warn 不 raise**（本条不是安全守卫）：
      ① 一个**只读**的能力探测面不该有权炸掉整条业务通路；
      ② 后果本身可观测 —— 每一次落到未登记码都会打一条 warning，不存在「静默」；
      ③ 我方无法区分「对方真删了这个码」与「这个部署只是版本旧」，在不可区分的观测上
         硬失败会把跨仓单边升级变成互相锁死。
      要硬拦的是**源码级**漂移，那已由 `TestZeroErrorCodeCrosscheck`（含 STRICT 转 fail）负责。

    ⚠ 版本不认识时**仍然比对**：本函数的最强动作就是 warn，而「不比对」等于主动丢掉唯一信号；
    结论串里会带上版本存疑的标注，由调用方 `ZeroLinkClient.check_error_codes` 打进日志。
    """
    expected = frozenset(client_codes)
    if not cfg.available:
        return ZeroErrorCodeDiff(checked=False, reason=cfg.describe())
    zero_codes = _read_str_tuple_field(cfg.fields, _DESCRIBE_CONFIG_KEY_ERROR_CODES)
    if zero_codes is None:
        return ZeroErrorCodeDiff(
            checked=False,
            reason=(
                f"未比对：返回体缺 {_DESCRIBE_CONFIG_KEY_ERROR_CODES} 键或形状非 list[str]"
                f"（实得 {type(cfg.fields.get(_DESCRIBE_CONFIG_KEY_ERROR_CODES)).__name__}）"
            ),
        )
    zero_set = frozenset(zero_codes)
    zero_only = tuple(sorted(zero_set - expected))
    client_only = tuple(sorted(expected - zero_set))
    if not zero_only and not client_only:
        reason = f"两侧登记面一致（{len(zero_set)} 个码）"
    else:
        parts = [f"错误码表不一致（对方 {len(zero_set)} 个 / 本仓 {len(expected)} 个）"]
        if zero_only and client_only:
            parts.append("两侧同时非空 ⇒ **疑似改名**，请优先按改名核，别当成一增一删")
        if zero_only:
            parts.append(
                f"对方多={list(zero_only)}（正常演进；但若是把既有失效模式**切分**到新码，"
                f"我方既有归类会被掏空 → 不可降级族退化成静默降级，请核对方产出点）"
            )
        if client_only:
            parts.append(
                f"本仓多={list(client_only)}（**本仓拿着过期认知**：这几条永不命中，"
                f"对方那个失效模式若换码回来会落未登记码 → 归责降级，请同步 "
                f"client._CODE_TO_EXCEPTION 与跨仓守卫）"
            )
        reason = "；".join(parts)
    if not cfg.version_known:
        reason = f"{reason}；⚠ describe_config_version={cfg.version} 本仓不认识，结论仅供参考"
    return ZeroErrorCodeDiff(
        checked=True,
        reason=reason,
        zero_only=zero_only,
        client_only=client_only,
    )


@dataclass(frozen=True, slots=True)
class ZeroExternalPriorPreflight:
    """发流**前**自检：本仓 external_priors 契约/阈值 vs **所连部署**真正生效的值。

    🛑 **与既有 M5 静态守卫不重复**（覆盖面不相交，别当成重复造轮子）：
      · `test_zero_contract_crosscheck.py::TestExternalPriorSchemaVersion` /
        `TestExternalPriorValidationDefaults` 读的是**本机 `D:\\Zero` 源码树**的常量与
        `AffectState` 字段默认 —— 它答的是「我们两个仓的代码对不对得上」。
      · 本类读的是**真正连上的那个部署**在**运行期**的生效值 —— 它答的是「我现在要发流的
        这个 server，此刻的门是什么」。二者在 HTTP 远端、或本机 env 覆盖了默认值时**必然分叉**
        （env 一改，源码常量纹丝不动），而分叉时权威的是运行期这一份。

    Attributes:
        checked:               是否真做了比对。
        reason:               人读结论串（未比对时是原因）。
        zero_*/client_*:      两侧的 schema 版本 / 精度上界 / 流数上界（对方侧不可读 → ``None``；
                              `client_*` 为 ``None`` 表示**本机 env 坏了没读出来**，见
                              `local_env_error`——不是「本机没有默认值」）。
        rejection:            传了 `priors` 且按**对方阈值**重跑本仓 M3/M6/M7/M8/M9 校验被拒时，
                              这里是拒绝原因；空串 = 不会被拒（或没传 priors）。
        local_env_error:      **本机**阈值 env（`ZERO_EXTERNAL_PRIOR_PRECISION_CAP` /
                              `ZERO_MAX_EXTERNAL_STREAMS`）解析失败的原因；
                              ``None`` = 本机 env 正常。
        version_known:        `describe_config_version` 是否属本仓已核验版本。

    🛑 `rejection` 与 `local_env_error` **必须分列两格**（2026-07-30 审查订正，原实现混用
    `rejection` 一格）：前者是「**对方**会拒这批 priors」，后者是「**我方**部署的 env 写坏了」。
    混用的直接后果是 —— 本机 env 一坏，**一条 priors 都没传**时 `would_be_rejected` 也变 True，
    与本文档串「空串 = 不会被拒（或没传 priors）」自相矛盾；调用方据此放弃发流，等于本机的
    配置笔误把自己关在门外。两者对调用方的处置也不同：前者改 priors、后者改本机 env。
    """

    checked: bool
    reason: str
    zero_schema_version: int | None = None
    client_schema_version: int = EXTERNAL_PRIOR_SCHEMA_VERSION
    zero_precision_cap: float | None = None
    client_precision_cap: float | None = None
    zero_max_streams: int | None = None
    client_max_streams: int | None = None
    rejection: str = ""
    local_env_error: str | None = None
    version_known: bool = False

    @property
    def schema_mismatch(self) -> bool:
        """对方 schema 版本可读**且**与本仓不等 —— 契约不兼容（唯一会被升级成 raise 的一格）。

        对方不可读（``None``）时为 False：「读不到」不是「不一致」，在读不到的位上硬失败
        等于把老部署一律判死。
        """
        return (
            self.zero_schema_version is not None
            and self.zero_schema_version != self.client_schema_version
        )

    @property
    def limits_differ(self) -> bool:
        """精度上界 / 流数上界与本仓本地默认不同（**不是错误**，只是我方该按对方的来）。"""
        cap_differs = (
            self.zero_precision_cap is not None
            and self.client_precision_cap is not None
            and self.zero_precision_cap != self.client_precision_cap
        )
        max_differs = (
            self.zero_max_streams is not None
            and self.client_max_streams is not None
            and self.zero_max_streams != self.client_max_streams
        )
        return cap_differs or max_differs

    @property
    def would_be_rejected(self) -> bool:
        """这批 priors 按对方阈值**必被拒** —— 提前知道，省一次 step 往返与一次内核 ToolError。

        只读 `rejection` 一格：它**只**由「拿对方阈值干跑一遍本仓校验」置位，与本机 env 是否
        坏掉无关（后者进 `local_env_error`，理由见类 docstring）。
        """
        return bool(self.rejection)


def check_external_prior_limits(
    cfg: ZeroDeployConfig,
    priors: list[ModalityPrior] | None = None,
) -> ZeroExternalPriorPreflight:
    """按回读面判读 external_priors 契约与阈值（**纯函数**，无 I/O、永不抛）。

    `priors` 非空时，用**对方的**阈值重跑一遍本仓 `build_external_priors_override`
    ——不重写一套校验逻辑（重写必然与 M3/M6/M7/M8/M9 的执行序和合并语义漂移），
    直接拿本仓那份唯一真相跑一次干跑，被拒即把它的 ValueError 文案原样带出来。

    「对方会不会拒这批 priors」（`rejection`）与「本机阈值 env 写坏了」（`local_env_error`）
    是**两件互不依赖的事**，各占一格、各自独立判定，理由见 `ZeroExternalPriorPreflight`。
    """
    if not cfg.available:
        return ZeroExternalPriorPreflight(checked=False, reason=cfg.describe())
    # 本机阈值 env 只服务**一件事**：把 client_cap/client_max 摆出来与对方值对比展示。
    # 它坏掉不进 `rejection`（那格只表示「对方会拒这批 priors」），单列 local_env_error。
    # 也不在这里抛：自检面的职责是如实报告，真正发流时 build_* 会照抛不误。
    local_env_error: str | None = None
    try:
        client_cap: float | None = _resolve_precision_cap(None)
        client_max: int | None = _resolve_max_streams(None)
    except ValueError as exc:
        # 两个 env 共用这一次解析 ⇒ 任一坏掉，两侧对比都标记为不可得（`None`）。
        # 这是**如实标注**而非伪造：此时我方本地默认到底是多少，本身就已不可信。
        client_cap = None
        client_max = None
        local_env_error = f"本机 env 不合法，无法与对方阈值比对：{exc}"
    zero_schema = _read_int_field(cfg.fields, _DESCRIBE_CONFIG_KEY_SCHEMA_VERSION)
    zero_cap = _read_float_field(cfg.fields, _DESCRIBE_CONFIG_KEY_PRECISION_CAP)
    zero_max = _read_int_field(cfg.fields, _DESCRIBE_CONFIG_KEY_MAX_STREAMS)

    # ⚠ 这条干跑**刻意不受 local_env_error 阻断**：它显式传 zero_cap/zero_max，而
    # `_resolve_precision_cap`/`_resolve_max_streams` 在拿到显式值时直接返回、根本不读 env
    # （见 external_priors 两函数首行）⇒ 本机 env 坏不坏与这条判读无关。若在此短路，
    # 一个本机配置笔误就会把「对方会不会拒」这条真信号一并丢掉，恰是最需要它的时候。
    rejection_reason = ""
    if priors and zero_cap is not None and zero_max is not None:
        try:
            build_external_priors_override(priors, max_streams=zero_max, precision_cap=zero_cap)
        except ValueError as exc:
            rejection_reason = str(exc)

    report = ZeroExternalPriorPreflight(
        checked=True,
        reason="",
        zero_schema_version=zero_schema,
        client_schema_version=EXTERNAL_PRIOR_SCHEMA_VERSION,
        zero_precision_cap=zero_cap,
        client_precision_cap=client_cap,
        zero_max_streams=zero_max,
        client_max_streams=client_max,
        rejection=rejection_reason,
        local_env_error=local_env_error,
        version_known=cfg.version_known,
    )
    parts: list[str] = []
    if report.schema_mismatch:
        parts.append(
            f"🛑 external_prior_schema_version 不一致：对方={zero_schema}、"
            f"本仓={EXTERNAL_PRIOR_SCHEMA_VERSION} —— **契约不兼容**，"
            f"我方构造的三元组可能被对方按另一套形状解释成功（被拒是响亮失败，"
            f"没被拒却被误解才是灾难）"
        )
    if zero_schema is None:
        parts.append(f"对方 {_DESCRIBE_CONFIG_KEY_SCHEMA_VERSION} 不可读（缺键/形状不符）")
    if report.limits_differ:
        parts.append(
            f"阈值与本机默认不同：precision_cap 对方={zero_cap}/本仓={client_cap}、"
            f"max_streams 对方={zero_max}/本仓={client_max}（不是错误，但发流须按对方的来）"
        )
    if report.would_be_rejected:
        parts.append(f"这批 priors 按对方阈值**会被拒**：{rejection_reason}")
    if local_env_error is not None:
        parts.append(
            f"⚠ {local_env_error} ⇒ 只给得出对方侧数值，本机默认这一侧不可读"
            f"（**与上面这批 priors 会不会被拒无关**：那条只按对方阈值判。"
            f"发流时若同样走 env（未显式传 cap/max），build_external_priors_override "
            f"会因同一个 env 照抛不误，请先修 env）"
        )
    if not parts:
        parts.append(
            f"自检通过（schema v{zero_schema}、precision_cap={zero_cap}、max_streams={zero_max}）"
        )
    if not cfg.version_known:
        parts.append(
            f"⚠ describe_config_version={cfg.version} 本仓不认识 ⇒ 上述读法本身不可信，"
            f"schema 不一致**降级为告警不上抛**"
        )
    return dataclasses.replace(report, reason="；".join(parts))


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

        `describe_config_cache` / `describe_config_absent` 是 `describe_config()` 的实例级缓存，
        生命周期 = **一次连接**（`__aexit__` 清空）。它**按 session_id 分键**，故与
        `last_open_session` 那种「单个共享可变字段」不同，多会话并发下不会串味；失效边界与
        「为什么敢缓存」的完整论证见 `describe_config()` docstring。
        """
        self.exit_stack: contextlib.AsyncExitStack | None = None
        self.session: ClientSession | None = None
        self.last_open_session: ZeroOpenSessionInfo | None = None
        self.describe_config_cache: dict[str | None, ZeroDeployConfig] = {}
        self.describe_config_absent: ZeroDeployConfig | None = None

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
        # 🛑 回读面缓存的作用域 = **一次连接**：下一次 `async with` 很可能是另一个 server 进程
        # （stdio 每次重起子进程；HTTP 也可能已重启换了 env），沿用旧缓存 = 拿上一个部署的门控
        # 回答这一个部署的问题。清空的代价只是重付一次 RTT。
        self.describe_config_cache.clear()
        self.describe_config_absent = None
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

        # 🛑 新键解析在 session_id 之后、且**只用 .get/in**：缺键 → 一律回落，行为逐字等于
        # 换代前（现网老 Zero 零回归）。形状异常只 warning 不抛（见各 _parse_* 的 docstring）。
        # 中断态走**双轨唯一入口**：新部署读显式 `interrupt_probe`，老部署回落缺席推断。
        probe, nodes, probe_raw = _parse_open_session_interrupt_state(data)
        info = ZeroOpenSessionInfo(
            session_id=returned_id,
            resumed=_parse_open_session_resumed(data),
            interrupted_at=nodes,
            interrupt_probe=probe,
            interrupt_probe_raw=probe_raw,
        )
        self.last_open_session = info
        # 🛑 回读面缓存失效点之一：开/resume 都会让 Zero **重建**该会话的 config。
        # 尤其 resume —— SessionConfig 不进 checkpoint，未再供 config 时会**回落到 env 默认**
        # （我方 R11 提过的那条），此时同一个 sid 的会话级门控与上一轮可以完全不同。
        # 请求 id 与返回 id 都丢（不传 sid 时 Zero 新铸 uuid4，丢它是无害的 no-op）。
        self._forget_describe_config(session_id)
        self._forget_describe_config(returned_id)
        if info.resumed is not None:
            logger.info(
                "zero.open_session: session=%s resumed=%s",
                returned_id,
                info.resumed,
            )
        # ⚠ 尾句**按调用方分支出**：`downstream_guard=False`（公开 `open_session()`，
        # 即常规 resume 路径）之后确无守卫——step 照常发、bundle 照常回、连一条 ERROR 都不会有，
        # 缺口由 `test_normal_resume_path_has_no_interrupt_guard` 特征化钉住；
        # `downstream_guard=True`（`graceful_step` 自愈分支）之后紧跟拒绝续跑的 ERROR、
        # 可能还有 purge，此时**不得**再说「无守卫」「须自行决定调 purge_session」
        # ——那会劝调用方去做刚刚已经做完的事（2026-07-29 终审判为 blocking 的原话）。
        # 🛑 本层**只报告、不处置**（处置权在调用方，见 `graceful_step` docstring 的论证）：
        # 三条 WARNING 各自锚一个 `*_ON_OPEN` marker，与决策层的 `*_REFUSED` **刻意不同名**
        # ——两层日志落在同一个 caplog 里，同名会让「决策层真拒绝了」的断言被本层喂成恒真。
        tail = (
            "后续由调用方拦截（紧随其后的 ERROR 给出处置）。"
            if downstream_guard
            else (
                "⚠ 常规 resume 路径**无守卫**：除本条外不会再有任何日志或拦截，"
                "调用方须自行决定轮换 session_id 或调 purge_session。"
            )
        )
        if info.interrupt_probe is ZeroInterruptProbe.INTERRUPTED:
            # WARNING 而非 INFO：确定半截 ⇒ 该会话运行态**停在 super-step 边界**
            # （上一轮被中途取消，已跑完节点的写入已落盘且 sqlite 后端跨重启保留）。
            # 续跑会从待执行节点继续、而非重跑整轮 —— 这是「拿到的下一帧不可全信」的信号。
            # ⚠ 判据从「`interrupted_at` 非空」换成「态是 INTERRUPTED」（2026-07-30 双轨化）：
            # 新轨下令牌是判据、节点名只是载荷，可能出现「令牌说 interrupted 但载荷缺席」
            # （自洽性告警已在解析层发过）——那一格也必须落这条日志，否则最该响的时候没声。
            logger.warning(
                "%s zero.open_session: session=%s 上一轮被中途取消，运行态停在 super-step 边界，"
                "待执行节点=%s；在其上续跑 = 新刺激叠加到半截运行态。%s",
                LOG_MARKER_INTERRUPTED_ON_OPEN,
                returned_id,
                list(info.interrupted_at or ()),
                tail,
            )
        elif info.interrupt_probe is ZeroInterruptProbe.PROBE_FAILED:
            # 🛑 对方**显式**说它自己的探测抛了（Zero `667e923` 起才有这一位）。
            # 这一格与半截态**故障相关**：探测读的正是那份可能半写的 checkpoint。
            # 归因写清「是对方探测失败」，而不是「我方读不到」——后者是老部署的样子，
            # 两者行为可能相同（都不可判）但归因完全不同，日志必须能分开。
            logger.warning(
                "%s zero.open_session: session=%s 对方回 %r=%r：**它自己的中断探测抛了**，"
                "本次**无法判定**上一轮是否被中途取消。⚠ 探测读的正是那份可能半写的 "
                "checkpoint ⇒ 该格与半截态故障相关，**不得视同干净**。%s",
                LOG_MARKER_PROBE_FAILED_ON_OPEN,
                returned_id,
                _OPEN_SESSION_KEY_INTERRUPT_PROBE,
                info.interrupt_probe_raw,
                tail,
            )
        elif info.interrupt_probe is ZeroInterruptProbe.UNRECOGNIZED:
            # 第五态 / 值形状坏：解析层已发过一条带原始取值的告警（含已知集合），此处只补
            # 「这一帧后续怎么处置」的上下文，不重复取值细节。
            logger.warning(
                "%s zero.open_session: session=%s 对方的 %r 取值我方**不认识**（raw=%r），"
                "按最坏情况处置、不当干净。%s",
                LOG_MARKER_PROBE_UNRECOGNIZED_ON_OPEN,
                returned_id,
                _OPEN_SESSION_KEY_INTERRUPT_PROBE,
                info.interrupt_probe_raw,
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
        # 会话没了 ⇒ 会话级回读面缓存失效（同 id 再开会是**新**会话、新 config）。
        self._forget_describe_config(session_id)

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
        # purge 内部先跑一遍 close（Zero 侧 `registry.close`）⇒ 该 sid 的会话级门控已不存在，
        # 缓存必须失效；下一次同 id 重开会以**当时的** env 默认重建 config（R11 那条回落）。
        self._forget_describe_config(session_id)
        return purged if isinstance(purged, bool) else False

    # ── 只读回读面：zero.describe_config ──────────────────────────────────────

    def _forget_describe_config(self, session_id: str | None) -> None:
        """丢弃某个 session_id 的回读面缓存（不动部署端默认那一条，也不动负缓存）。

        ⚠ **只丢会话级那一条**：部署端默认（键 ``None``）由 server 进程的 env 决定，不因某个
        会话开/关/purge 而变；负缓存（`describe_config_absent`）是「对方根本没这个工具」，
        更与会话无关。把它们一起清掉只会白付 RTT。
        """
        self.describe_config_cache.pop(session_id, None)

    async def _tool_registered(self, tool_name: str) -> bool | None:
        """问 MCP 协议自己的能力面：这个工具**在不在册**。``None`` = 连能力面都问不到。

        🛑 **为什么不从错误文案判**：FastMCP 未注册工具回的是 `Unknown tool: <name>` 这类
        人读文案，按它做判定是脆弱锚点（pitfalls ⑦：对方一次措辞修订就静默失效，且失效方向是
        「误判成老部署 → 永久降级」）。`list_tools` 是协议**定义**的能力面，稳定得多。

        任何异常都吞成 ``None``（而非 False）：问不到 ≠ 不在册。返回 False 是**下定论**，
        它会触发负缓存、短路本连接剩余生命期里的所有探测，故只在真拿到工具清单时才敢给。
        """
        session = self._require_session()
        try:
            result = await session.list_tools()
        except Exception as exc:  # noqa: BLE001 - 能力探测降级路径，问不到即返回「不确定」
            logger.warning(
                "list_tools 失败（%s: %s）——无法确认 %s 是否在册，按「不确定」处置。",
                type(exc).__name__,
                exc,
                tool_name,
            )
            return None
        tools = getattr(result, "tools", None)
        if not isinstance(tools, list):
            logger.warning("list_tools 返回体形状非预期（%r）——按「不确定」处置。", type(tools))
            return None
        return any(getattr(tool, "name", None) == tool_name for tool in tools)

    async def _describe_config_after_failure(self, exc: Exception) -> ZeroDeployConfig:
        """`describe_config` 调用失败后的**归因**：老部署没这工具，还是这一次没调通。"""
        registered = await self._tool_registered(_DESCRIBE_CONFIG_TOOL)
        if registered is False:
            cfg = ZeroDeployConfig(
                probe=ZeroConfigProbe.NOT_REGISTERED,
                detail=(
                    f"{_DESCRIBE_CONFIG_TOOL} 不在对方工具清单里（list_tools 确证），"
                    f"判为**老部署**：调用错误={type(exc).__name__}: {exc}"
                ),
            )
            self.describe_config_absent = cfg
            logger.warning(
                "%s 所连 Zero 未注册 %s（老部署）——依赖该回读面的能力全部降级："
                "错误码表运行期核对、发流前自检均不执行（各自返回 checked=False），"
                "行为回落到接入本工具之前（撞了 ExternalPriorError 才知道）。",
                LOG_MARKER_DESCRIBE_NOT_REGISTERED,
                _DESCRIBE_CONFIG_TOOL,
            )
            return cfg
        logger.warning(
            "%s %s 调用失败（%s: %s），但工具在册=%s ⇒ **不下「老部署」的定论、不缓存**，"
            "下次调用会重新探测。",
            LOG_MARKER_DESCRIBE_CALL_FAILED,
            _DESCRIBE_CONFIG_TOOL,
            type(exc).__name__,
            exc,
            registered,
        )
        return ZeroDeployConfig(
            probe=ZeroConfigProbe.CALL_FAILED,
            detail=(
                f"调用失败但工具在册={registered}（None=连 list_tools 都问不到）："
                f"{type(exc).__name__}: {exc}"
            ),
        )

    async def describe_config(
        self,
        session_id: str | None = None,
        *,
        force_refresh: bool = False,
    ) -> ZeroDeployConfig:
        """回读**所连部署**真正生效的门控（Zero `zero.describe_config`）。

        **永不因对方缺这工具而抛。**

        不传 `session_id` → **部署端默认**（env + caps + versions），供在 `open_session`
        **之前**决定要不要发某类流；传 → **该会话真实生效**的值（未知 id 视同不传，此时
        `resolved_for_session=False`）。

        ── **何时调**（本方法刻意不自动调用）──────────────────────────────────────
        懒加载、由调用方在需要时显式调。**绝不挂进 `step()`**：那是每帧一次的热路径，为一份
        「会话活跃期不会变」的配置每帧付一个 RTT 是纯浪费。也不挂进 `__aenter__`：连接建立
        不该为一个**可选**的只读面多付一次往返，更不该让连接的失败模式受它牵连。
        典型调用点：进程起来后一次（部署端默认）+ 每次 `open_session` 之后一次（会话级）。

        ── **缓存与失效边界**（这是本方法最需要论证的地方）────────────────────────
        实例级缓存，**按 session_id 分键**，生命周期 = 一次连接（`__aexit__` 清空，理由见那里）。
        · **键 ``None``（部署端默认）敢缓存**：它由 server 进程的 env 决定，env 要变必须重启进程，
          而 stdio 下进程重启 = 我方连接断，HTTP 下 MCP 会话失效 —— 两条路都会走到 `__aexit__`
          或重连。
        · **会话级（键 = sid）敢缓存，但只在 `resolved_for_session is True` 时**：
          Zero 明言活跃会话的门控**构造时固定**（其 describe_config 从 `session.config` 取而非
          现算），故活跃期内不可变。
          🛑 而 `resolved_for_session is False` 意味着**对方不认识这个 id、回的是部署端默认**
          —— 把它缓存到 sid 键下是**主动制造一颗定时炸弹**：等会话真的开出来，同一个 sid 的
          正确答案已经变了，我方却还在服务那份「默认值伪装成会话值」的旧答案。故不缓存。
        · **失效点（三处，覆盖本 client 能观测到的全部状态变更）**：`_open_session_info`
          （开/resume 都会重建 config —— ⚠ **跨 resume 会回落 env 默认**，这正是我方 R11 提过的
          「SessionConfig 不进 checkpoint」，是会话级缓存唯一真正的过期成因）、`close_session`、
          `purge_session`。
        · **残留缺口，如实写**：另一个进程/实例用**同一个 sid** 重开出不同 config 时，本实例的
          缓存不会失效（我方看不见对方的动作）。后果止于「自检结论过期」，不影响任何安全判定
          （安全面 M8/M9 单边兜住、不依赖本回读面）。需要绝对新鲜时传 `force_refresh=True`。
          ⚠ 跨进程共用一个 sid 本身已违反「session_id = 运行态访问凭据」的信任模型。

        ── **老部署（没注册这个工具）**────────────────────────────────────────────
        返回 `probe=NOT_REGISTERED` 的 `ZeroDeployConfig`，**不抛**。归因走 `list_tools`
        协议能力面而非错误文案（理由见 `_tool_registered`）。该结论**负缓存**并短路后续所有
        探测（工具不在册与 sid 无关），避免每次自检白付 2 个 RTT。
        调用方怎么知道降级了：`cfg.available is False`，且 `diff_error_codes` /
        `check_external_prior_limits` 都会返回 `checked=False` + 原因串（**不是**「检查通过」）。
        调用失败但工具在册 → `probe=CALL_FAILED`，**不缓存**（不确定的事不下定论）。

        ── **`describe_config_version` 演进**──────────────────────────────────────
        认识的版本正常用；不认识 → **降级但不炸**：仍解析、仍报告，但唯一的强制动作
        （`preflight_external_priors` 的 raise）降级为 warn，并打一条
        `LOG_MARKER_DESCRIBE_VERSION_UNKNOWN`。理由见 `KNOWN_DESCRIBE_CONFIG_VERSIONS`
        （对方的 bump 纪律覆盖「某键**语义**变了」，那是照旧解析会得出错误结论的一类）。

        Args:
            session_id:    None → 部署端默认；非 None → 该会话真实生效的值。
            force_refresh: 跳过并清掉本键缓存与负缓存，强制重新探测。

        Returns:
            `ZeroDeployConfig`（不可用时也是合法值，读 `.probe` / `.available` 判定）。

        Raises:
            ZeroLinkConnectionError: 未在 async with 内调用（编程错误，照旧透传）。
        """
        if force_refresh:
            self._forget_describe_config(session_id)
            self.describe_config_absent = None
        elif self.describe_config_absent is not None:
            return self.describe_config_absent
        else:
            cached = self.describe_config_cache.get(session_id)
            if cached is not None:
                return cached

        args: dict[str, Any] = {} if session_id is None else {"session_id": session_id}
        try:
            text = await self._call_tool(_DESCRIBE_CONFIG_TOOL, args)
        except (ZeroLinkCallError, McpError) as exc:
            # ⚠ 连 `ZeroLinkNonDegradableError` 子类也在这里被判读成「探测失败」而非上抛：
            # 一个**只读的可选**探测面不该有权炸掉调用方；同一个部署问题会在真正的
            # open/step 路径上原样抛出（那里才是它该被看见的地方）。
            return await self._describe_config_after_failure(exc)

        try:
            data: Any = json.loads(text)
        except json.JSONDecodeError as exc:
            return self._describe_config_malformed(f"响应非合法 JSON：{exc}")
        if not isinstance(data, dict):
            return self._describe_config_malformed(
                f"响应不是 JSON object，实得 {type(data).__name__}"
            )

        cfg = ZeroDeployConfig(
            probe=ZeroConfigProbe.OK,
            version=_read_int_field(data, _DESCRIBE_CONFIG_KEY_VERSION),
            # `is True` 而非 truthy：只认真正的 bool，与 `_parse_open_session_resumed` 同口径。
            resolved_for_session=data.get(_DESCRIBE_CONFIG_KEY_RESOLVED) is True,
            fields=types.MappingProxyType(dict(data)),
        )
        self._log_describe_config_shape(cfg)
        if session_id is None or cfg.resolved_for_session:
            self.describe_config_cache[session_id] = cfg
        else:
            logger.debug(
                "describe_config(session_id=%r) 回了 resolved_for_session=False（对方不认识该 id，"
                "回的是部署端默认）——**不缓存**，否则会话真开出来后仍在服务这份旧答案。",
                session_id,
            )
        return cfg

    def _describe_config_malformed(self, detail: str) -> ZeroDeployConfig:
        """返回体畸形：记 warning、判 MALFORMED、**不缓存**（畸形不是稳定事实）。"""
        logger.warning("%s %s：%s", LOG_MARKER_DESCRIBE_FIELDS_DRIFT, _DESCRIBE_CONFIG_TOOL, detail)
        return ZeroDeployConfig(probe=ZeroConfigProbe.MALFORMED, detail=detail)

    def _log_describe_config_shape(self, cfg: ZeroDeployConfig) -> None:
        """版本/键集的形状告警（每条缓存只会打一次，因为命中缓存的调用根本不到这里）。"""
        if not cfg.version_known:
            logger.warning(
                "%s describe_config_version=%r 不在本仓已核验集合 %s —— 依赖本回读面的判定"
                "**降级为只报告不强制**（schema 版本不一致由 raise 降为 warn）。"
                "请现场核对方 describe_config 返回体后再把新版本号收进 "
                "KNOWN_DESCRIBE_CONFIG_VERSIONS。",
                LOG_MARKER_DESCRIBE_VERSION_UNKNOWN,
                cfg.version,
                sorted(KNOWN_DESCRIBE_CONFIG_VERSIONS),
            )
        if cfg.missing_keys or cfg.unexpected_keys:
            logger.warning(
                "%s describe_config 键集与本仓期望不符：缺键=%s、新增键=%s。"
                "缺键按「该位不可读」处置（不猜默认值）；新增键是对方正常演进，本仓忽略。",
                LOG_MARKER_DESCRIBE_FIELDS_DRIFT,
                list(cfg.missing_keys),
                list(cfg.unexpected_keys),
            )

    async def check_error_codes(self, *, session_id: str | None = None) -> ZeroErrorCodeDiff:
        """① **运行期**核对本仓 `ZERO_ERROR_CODES` 手抄镜像 vs 所连部署的 `error_codes` 全表。

        这条替代了本仓原本要向 Zero 索要的「码字符集自校验」（该请求已在跨仓件里撤回）——
        对方现在直接回全表，比让对方替我方校验更直接、且看的是**真正连上的那个部署**。

        处置一律 **warn 不 raise**，三种不一致的严重度与文案不同，逐条论证见 `diff_error_codes`。
        回读面不可用 → `checked=False` + 原因串（**不是**「一致」）。
        """
        cfg = await self.describe_config(session_id=session_id)
        diff = diff_error_codes(cfg)
        if not diff.checked:
            logger.info("错误码表运行期核对未执行：%s", diff.reason)
        elif diff.in_sync:
            logger.debug("错误码表运行期核对：%s", diff.reason)
        else:
            logger.warning("%s %s", LOG_MARKER_ERROR_CODE_TABLE_DRIFT, diff.reason)
        return diff

    async def preflight_external_priors(
        self,
        priors: list[ModalityPrior] | None = None,
        *,
        session_id: str | None = None,
        strict: bool = True,
    ) -> ZeroExternalPriorPreflight:
        """② **发流前**自检：契约版本 + 精度/流数上界 vs 所连部署的真实生效值。

        没有这条时我方只能「撞了 `ExternalPriorError` 才知道」；有了回读面就能在发流之前发现。
        与既有 M5 静态守卫**覆盖面不相交**（源码树 vs 真实部署），论证见
        `ZeroExternalPriorPreflight` 类 docstring。

        Args:
            priors:     可选。给了就用**对方的**阈值把本仓 `build_external_priors_override`
                        干跑一遍，提前拿到确切的拒绝原因。
            session_id: None → 按部署端默认自检（`open_session` 之前就能做）。
            strict:     schema 版本不一致时是否上抛。默认 True。

        Returns:
            `ZeroExternalPriorPreflight`（阈值不同 / 这批 priors 会被拒 / 本机阈值 env 坏了
            → 一律只报告 + WARNING，不抛）。⚠ `would_be_rejected` **只**回答「对方会不会拒
            这批 priors」；本机 env 坏掉走 `local_env_error`，不会把「没传 priors」染成会被拒。

        Raises:
            ZeroLinkSchemaIncompatibleError: `strict` 且回读面**版本可信**且
                `external_prior_schema_version` 与本仓不一致。⚠ 版本不认识时**不抛**——
                在不可信的观测量上 raise 是错的，降级为 warn。
        """
        cfg = await self.describe_config(session_id=session_id)
        report = check_external_prior_limits(cfg, priors)
        if not report.checked:
            logger.info("发流前自检未执行：%s", report.reason)
            return report
        if report.schema_mismatch and strict and cfg.version_known:
            logger.error("%s %s", LOG_MARKER_EXTERNAL_PRIOR_PREFLIGHT, report.reason)
            raise ZeroLinkSchemaIncompatibleError(_DESCRIBE_CONFIG_TOOL, report.reason)
        if (
            report.schema_mismatch
            or report.limits_differ
            or report.would_be_rejected
            or report.local_env_error is not None
        ):
            # `local_env_error` 单列后仍须留在告警面：它此前混在 `rejection` 里，是能见的；
            # 拆格若不补这一项，本机 env 写坏就从 WARNING 掉进 DEBUG（发流时才炸，且不知为何）。
            logger.warning("%s %s", LOG_MARKER_EXTERNAL_PRIOR_PREFLIGHT, report.reason)
        else:
            logger.debug("发流前自检：%s", report.reason)
        return report

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

        🛑 **半截运行态本帧不续跑**（2026-07-29）：若重开的返回体判出 `INTERRUPTED`
        （上一轮被中途取消，运行态停在 super-step 边界），**不重试 step**，而是 **ERROR 级日志
        （带待执行节点名）+ 降级 None**。

        🛑 **不可判且与故障相关的两格同样拒绝**（2026-07-30，接对方显式 `interrupt_probe`）：
        `PROBE_FAILED`（对方自述「我的探测抛了」）与 `UNRECOGNIZED`（对方回了我方不认识的第五态
        或该键形状坏）**一律按最坏情况拒绝本帧 + ERROR，但不 purge**。
        为什么这与下面「不可判就照常续跑」的老口径不矛盾：老口径的成本论证建立在「探测干净与
        探测失败在返回体上同形」之上，拒绝其一等于拒绝**全部**健康 resume；新部署把两者拆开后
        该前提消失 —— 健康 resume 回 `clean` 走续跑，拒绝只落在对方真出事的那几帧。
        不 purge 的理由见分支注释（判据是「不可判」而非「确定半截」，不可逆动作不赌未知状态）。

        ⚠ **本机制的真实收益，如实表述**（上一版在此处过度宣称，2026-07-29 跨仓复核订正）：
        它买到的是「把一次**静默**续跑换成一条**响亮的 ERROR** + 本帧拒绝」，**不是**「避免污染」。
        污染并未被避免，只被推迟一帧 —— 因果链（Zero `daecce1` 现场核验）：
          · 止血判定只能在**重开之后**做（中断态来自 `open_session` 返回体），
            而重开这一步已经让该会话在 Zero registry 里**变活跃**；
          · 下一帧 `graceful_step` 因此不再报 unknown-session ⇒ 走正常 `step` 路径，
            照样在带 pending `next` 的线程上续跑；
          · 且 Zero 的 `interrupted_at()` 只在「resume 且**不活跃**」时探测（活跃分支提前
            return），⇒ **我方**能看到的这条 ERROR 对同一 session **一生只出现一次**。
        ⇒ 「有界一次性 vs 无界累积不可逆」的旧论证在实际控制流下**不成立**，已撤回。
        ✅ **2026-07-30 事实订正（上一版把「不可观测」写过头了）**：旧措辞说「Zero 的 `zero.step`
        **完全不做**中断检查 ⇒ 此后污染不可观测」——前半句在 `daecce1` 那代为真，但对方已在
        `667e923` 给 `zero.step` 加了**每轮事后检查**（跑完看一眼 `next`，非空即一条 WARNING；
        只记日志、**不改返回体、不拒绝本帧**，探测失败只吞进 debug）。⇒ 准确表述是：
        污染在**新部署的 Zero 侧每帧可观测**（其日志），在**我方侧仍不可观测**（返回体没这一位，
        client 拿不到）。差别不只是措辞：跨仓排障时该去对方日志按 sid 对齐，而非断定「没人看得见」。
        ⇒ 若将来要让我方也每帧可见，正确做法是向对方索要「把这一位放进 step 返回体」，
        不是我方在 client 侧记账（那条的取舍见下）。
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

        **老部署零回归 + 双轨归因**（2026-07-30 订正，此前这段写成了无条件成立的推断链）：
        · 连**老部署**（返回体无 `interrupt_probe` 键，≤ Zero main `75e8a36`）时，判读整段回落
          到「`interrupted_at` 缺席推断」老轨，行为与接显式态之前**逐字一致**（零回归）；
          其中「`resumed` 为真但 `interrupted_at` 缺席」这一格我方**不可判**（缺席四义：
          未探测·新建 / 未探测·活跃幂等重开 / 探测失败 / 探测成功且干净；第一义与 `resumed=True`
          互斥，故实际面对后三义）——处置是**照常续跑 + 一条可区分的 WARNING**，理由见分支内
          注释（在那一代上保守拒绝会误伤 100% 的健康 resume，等于废掉整个自愈能力）。
        · 连**新部署**（≥ Zero `667e923`）时判据是对方的显式令牌，上述四义**逐义**落到
          `NOT_PROBED`(①②) / `PROBE_FAILED`(③) / `CLEAN`(④)，不再靠缺席反推。
          ⇒ 剩下的不可判只有两格：`NOT_PROBED`+`resumed=True`（②活跃幂等重开，对方明确没看，
          **续跑**+告警，因为它不是故障相关的）与 `PROBE_FAILED`（③，故障相关，**拒绝**）。
        · **两者行为可能相同但归因必须分得开**：老部署不可判 → `LOG_MARKER_PROBE_UNDECIDABLE`；
          新部署对方探测失败 → `LOG_MARKER_PROBE_FAILED_{ON_OPEN,REFUSED}`。
          代际判别位是「有没有 `interrupt_probe` 键」（`ZeroOpenSessionInfo.interrupt_probe_raw`
          与「态是否为老轨专属的 ABSENT/MALFORMED」两处都能读出来），**不是** `resumed`。

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

        ✅ **`resumed` 判别位：`describe_config` 已接入，仍然不撤**（2026-07-30 复核，撤回旧计划）。
        上一版在此写「接上 `describe_config` 后本方法与 `ZeroOpenSessionInfo.resumed` 上的代际
        判别应一并改挂 `describe_config_version`」——**核验后判定该计划错误**：
          · 下方 ABSENT 分支的条件 `info.resumed is True` 同时干两件事：**(a) 代际**（老部署
            不发 `resumed` ⇒ 不进分支 ⇒ 零回归）与 **(b) 本次 open 是不是 resume**（排除缺席
            四义里的第①义「未探测·新建会话」）。
          · `describe_config` 只能覆盖 (a)：它是**配置**回读面，答不了「你刚才那一次调用走的
            是新建还是续会话」——(b) 是**每次调用**的事实，不是部署/会话的属性。
          ⇒ 改挂版本号 = 用一个覆盖不全的判据换掉覆盖全的那个，第①义会被放回不可判集合，
            这条 WARNING 会在**每一次健康的新建会话**上误发。**判别力不升反降，故不改。**
        另外三条原始理由今天依然成立、且都指向同一个结论：
          ① 在自愈分支里插一次**额外 round-trip**，而这条路径正是对端刚出过问题的时刻；
          ② 老部署没注册该工具 ⇒ 得再写一层「工具不存在=老部署」的回退，判别链更长
             （该回退现已实现于 `describe_config`，但它的 RTT 成本没消失）；
          ③ 判定所需的位（`resumed`/`interrupted_at`）**就在已拿到的 open_session 返回体里**，
             零额外调用。
        ⇒ 向 Zero 索要的「承诺 `resumed` 不被条件化」**不撤回**，但**理由要换**：不再是
        「我方拿它当代际位」（这半确已被 `describe_config_version` 顶替），而是
        「我方拿它判**本次 open** 的 resume 语义」。`describe_config()` 作为**独立**能力面提供，
        不接进本方法的热路径。

        Args:
            session_id:    Zero 会话 ID（None 时立即返回 None）。
            stimulus:      情感刺激。
            priors:        可选多模态先验列表。
            resume_config: unknown-session resume 重开时**再供的会话 config**（应与原 open_session
                           一致）；None → resume 会话走 Zero env 默认门控。
            purge_on_interrupted: 检出**确定**半截运行态（`INTERRUPTED`）时，是否额外调
                           `zero.purge_session` 清掉它。
                           🛑 **只覆盖 `INTERRUPTED` 这一格**（2026-07-30 明确）：`PROBE_FAILED`
                           / `UNRECOGNIZED` 那两格即便本参数为 True 也**不 purge** —— 参数名
                           承诺的是「检出半截就清」，而那两格是「读不出来」；把它悄悄扩张成
                           「不可判也清」，等于让调用方在不知情的情况下对一个可能完全健康的
                           会话执行不可逆删除。要那种语义须另开一个显式参数。
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
                # ── 中断态决策表（**双轨七态**，每一格的处置都单独论证）─────────────
                # 判据一律是 `info.interrupt_probe`（新部署=对方显式态，老部署=缺席推断）：
                #   INTERRUPTED   确定半截      → 本帧拒绝 + ERROR（+ 可选 purge）
                #   PROBE_FAILED  对方探测抛了  → 本帧拒绝 + ERROR，**不** purge   〔新轨〕
                #   UNRECOGNIZED  第五态/值坏   → 本帧拒绝 + ERROR，**不** purge   〔新轨〕
                #   MALFORMED     载荷形状坏    → 照常续跑 + ERROR（不可判，但必须有人看见）
                #   NOT_PROBED    对方没探测    → 照常续跑；resumed 为真时补 WARNING 〔新轨〕
                #   ABSENT        缺席四义      → 照常续跑；resumed 为真时补 WARNING 〔老轨〕
                #   CLEAN         明确干净      → 照常续跑，不打日志
                #
                # 🛑 **「拒绝」这一档为何现在敢用在不可判的格上**（本轮的核心判断）：
                # 上一版对不可判一律选「续跑 + 告警」，理由是「保守拒绝会误伤 100% 的健康
                # resume」——那条论证**只在老轨上成立**：老部署里「探测干净」与「探测失败」
                # 在返回体上同形，拒绝其中一个就等于拒绝全部健康 resume。
                # 新轨把这两者拆开了（clean vs probe_failed）⇒ 拒绝 `probe_failed` **不再
                # 误伤任何健康 resume**（健康的那条回 clean，走续跑）。代价换算彻底反转：
                # 收益（对方唯一一次「我探测不了」的自述被当真）不变，成本从 100% 误伤降到
                # 「只在对方真出事那几帧丢一帧」。⇒ 这正是我方当初索要显式四态的目的，
                # 拿到了就必须用上；继续按老口径「不可判 ⇒ 续跑」等于把要来的信息扔掉。
                if info.interrupt_probe is ZeroInterruptProbe.INTERRUPTED:
                    # 🛑 判出确定半截 ⇒ 上一轮被中途取消、运行态停在 super-step 边界
                    # （已跑完节点的写入已落盘，sqlite 后端跨重启保留）。
                    #
                    # ⚠ **本帧拒绝买到的是什么，如实写**（2026-07-29 跨仓复核撤回上一版论证）：
                    # 买到的是「一次**静默**续跑 → 一条**响亮的 ERROR** + 本帧拒绝」。
                    # **没有**买到「避免污染」——污染只被推迟一帧：
                    #   (a) 止血判定只能在**重开之后**做，而重开已让该会话在 Zero registry
                    #       变活跃 ⇒ 下一帧不再报 unknown-session ⇒ 走正常 step 路径，
                    #       照样在带 pending `next` 的线程上续跑；
                    #   (b) Zero 的 `interrupted_at()` 只在「resume 且不活跃」时探测
                    #       （活跃分支提前 return）⇒ **我方**能看到的这条 ERROR 对同一 session
                    #       一生只出现一次。
                    # ⇒ 旧注释里「有界一次性 vs 无界累积不可逆」的对称性论证**不成立**，已删。
                    # ✅ 2026-07-30 订正：旧注释在 (a) 里断言「Zero 的 `zero.step` 不做任何中断
                    # 检查（daecce1）」并据此说「之后污染彻底不可观测」——对方已在 `667e923` 给
                    # `zero.step` 加了每轮事后检查（非空 `next` → 一条 WARNING，不改返回体、
                    # 不拒帧）⇒ 新部署上污染在**对方日志里每帧可见**，只是**我方**读不到
                    # （那一位没进 step 返回体）。「不可观测」须限定为「我方侧」，排障要去对方日志。
                    # 残留缺口有特征化守卫钉住：
                    # `test_next_frame_after_interrupted_refusal_runs_normal_step`。
                    #
                    # 那为什么仍然不重试？两条**站得住**的理由：
                    #   ① 本帧重试必然产出一个混合值并当作正常返回值交出去（Zero 自己的契约：
                    #      「续跑会从此处继续而非重跑整轮」）；拒绝则至少这一帧不发错值。
                    #   ② 这条 ERROR 是该污染在**我方侧**唯一一次可观测的机会，必须存在且醒目
                    #      —— 而只要还重试，日志就会被一个「成功返回」的表象冲淡。
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
                if info.interrupt_probe is ZeroInterruptProbe.PROBE_FAILED:
                    # 🛑 **本任务的核心格**：对方显式说「我的探测抛了」（Zero `667e923` 起）。
                    # 处置 = **按最坏情况**：本帧拒绝 + ERROR，**绝不当 clean**。
                    #
                    # 为什么必须最坏而不是「续跑 + 告警」：这一格与要防的半截态是**故障相关**
                    # 的 —— 探测读的正是那份可能半写的 checkpoint ⇒ 越是真出事的时候越可能
                    # 落到这里。把它当干净 = 止血在最该生效的那一帧静默失效，且此前正因为
                    # 它与 clean 同形而**完全不可见**，这才是我方向对方索要显式化的那一格。
                    #
                    # 🛑 **拒绝但不 purge**（与 INTERRUPTED 的关键差别，`purge_on_interrupted`
                    # 开着也不 purge）：purge 删该 thread **全部** checkpoint 历史、不可逆，
                    # 而这里的判据是「**不可判**」，不是「确定半截」。一次探测抛异常完全可能
                    # 出在一个运行态干净的会话上（如后端瞬时报错），据此删掉它的全部历史
                    # 是拿不可逆动作赌一个我方明知读不出来的状态。调用方 opt-in 的语义是
                    # 「检出半截就清」（参数名即 `purge_on_interrupted`），把它悄悄扩张成
                    # 「读不出来也清」是偷改语义 ⇒ 本分支永不 purge，且日志明说这一点。
                    logger.error(
                        "%s graceful_step: session=%s resume 重开后对方回 %r=%r"
                        "——**它自己的中断探测抛了**，我方无法判定上一轮是否被中途取消。"
                        "按最坏情况本帧拒绝续跑并降级 None（**不**当干净）。"
                        "⚠ 未执行 purge：判据是「不可判」而非「确定半截」，据此删该会话全部"
                        "运行态历史是不可逆的过度杀伤；purge_on_interrupted 只管确定半截那一格。"
                        "⚠ 与 INTERRUPTED 同理，这**不是**避免了污染：会话已被重开，"
                        "**下一帧**将走正常 step 路径。要真正止血请轮换 session_id，"
                        "或在 Zero 侧同时刻的「中断探测失败」日志上定位根因。",
                        LOG_MARKER_PROBE_FAILED_REFUSED,
                        session_id,
                        _OPEN_SESSION_KEY_INTERRUPT_PROBE,
                        info.interrupt_probe_raw,
                    )
                    return None
                if info.interrupt_probe is ZeroInterruptProbe.UNRECOGNIZED:
                    # 未知第五态（或该键的值非 str）：对方的 bump 纪律②明说取值集合会变。
                    # 处置同 `PROBE_FAILED`——**最坏情况 + 拒绝 + 不 purge**，理由见
                    # `_parse_open_session_interrupt_state` docstring 的三选一论证：
                    # 乐观当 clean 违反本字段存在的理由，抛异常违反跨仓单边升级零回归，
                    # 故取「宁可吵不可静默」。marker 与 PROBE_FAILED 分开：两者都拒绝本帧，
                    # 但归因不同（「对方探测失败」vs「对方说了我方读不懂」），运维要分得开。
                    logger.error(
                        "%s graceful_step: session=%s resume 重开后对方回 %r=%r，"
                        "**不在**本仓认识的取值集合 %s 里——本帧按最坏情况拒绝续跑并降级 None"
                        "（**不**当干净），未执行 purge（判据是不可判）。"
                        "请现场核对方 open_session 新态的语义并登记进 client 的"
                        " _ZERO_PROBE_TOKEN_TO_STATE；跨仓取值集合守卫（STRICT 判红）本应"
                        "在部署前就提醒到这一步。",
                        LOG_MARKER_PROBE_UNRECOGNIZED_REFUSED,
                        session_id,
                        _OPEN_SESSION_KEY_INTERRUPT_PROBE,
                        info.interrupt_probe_raw,
                        sorted(KNOWN_ZERO_INTERRUPT_PROBE_VALUES),
                    )
                    return None
                if info.interrupt_probe is ZeroInterruptProbe.MALFORMED:
                    # **老轨专属**：没有 `interrupt_probe` 可依，而 `interrupted_at` 形状不对。
                    # **不可判**，故不拒绝（拒绝会因对方一个类型笔误就永久废掉自愈通路）；
                    # 但必须 ERROR ——它是跨语言契约破裂的直接证据，比任何一帧的数据都重要。
                    # ⚠ 与上面两格的差别：那两格里对方**说了**「我不可判 / 我有新态」，是可
                    # 采信的自述；这里对方什么都没自述，只是一个我方读不懂的载荷，且新轨一旦
                    # 上线该格就不再出现（有令牌兜着）⇒ 维持既有的「续跑 + ERROR」不变，零回归。
                    logger.error(
                        "%s graceful_step: session=%s resume 重开的返回体里 %r 形状非法"
                        "（跨仓契约漂移）——本帧**无法判定**是否半截，按续跑处置；"
                        "请核对 Zero 侧 open_session 返回体形状。",
                        LOG_MARKER_PROBE_MALFORMED,
                        session_id,
                        _OPEN_SESSION_KEY_INTERRUPTED_AT,
                    )
                elif info.interrupt_probe is ZeroInterruptProbe.NOT_PROBED and info.resumed is True:
                    # 〔新轨〕对方**明确**说「压根没探测」，且 `resumed=True`
                    # ⇒ 落到的是缺席四义里的**第②义：活跃幂等重开**（Zero 的
                    # `registry.get(sid) is not None` 分支提前 return，走不到探测）。
                    #
                    # 🛑 这一格**仍然不可判**，但不可判的**原因变了**：不再是「四义同形分不清」，
                    # 而是「已确知对方没看」。⇒ 归因从「三义之一」收窄成**唯一一义**，这是本轮
                    # 拿到的判别力增益：日志可以直接说「它没探测，因为会话还活跃」。
                    # ⚠ 为什么活跃重开也不安全：会话仍在 registry 里**不等于**运行态干净——
                    # 上一轮的 step 若被取消，session 对象仍留在 registry、pending `next` 也
                    # 仍在，只是这条路径压根不去看。故不能当 clean。
                    #
                    # 处置 = **照常续跑 + 一条可区分 WARNING**，**不**拒绝。与 `PROBE_FAILED`
                    # 的取舍差别（同为不可判、处置不同，必须论证）：
                    #   · 本格**不是**故障相关的：它是「对方按设计跳过探测」，触发条件是并发/
                    #     幂等重开这类正常控制流，与 checkpoint 是否半写**无关**；
                    #     `probe_failed` 则是探测**读那份 checkpoint 时炸了**，与半截态同因。
                    #   · 本格在自愈路径上还意味着一件事：我方刚收到 unknown-session、重开却
                    #     发现它活跃 ⇒ 另有一方正在用同一个 session_id。此时拒绝本帧既拦不住
                    #     那一方，又把并发场景下的正常业务打成丢帧。
                    # ⇒ 保守只用在故障相关的格上，正常控制流的不可判仍走「续跑 + 可区分告警」。
                    logger.warning(
                        "%s graceful_step: session=%s resume 重开后对方回 %r=%r 且 resumed=True"
                        "——**活跃幂等重开**（会话仍在对方 registry 里、提前 return，压根没探测），"
                        "故仍**无法判定**上一轮是否被中途取消，本帧按续跑处置。"
                        "⚠ 会话活跃≠运行态干净：上一轮 step 若被取消，pending 节点仍在，"
                        "只是这条路径不去看它。另有一方正在用同一 session_id 的可能请一并排查。",
                        LOG_MARKER_PROBE_NOT_PROBED_UNDECIDABLE,
                        session_id,
                        _OPEN_SESSION_KEY_INTERRUPT_PROBE,
                        info.interrupt_probe_raw,
                    )
                elif info.interrupt_probe is ZeroInterruptProbe.ABSENT and info.resumed is True:
                    # 🛑 **老轨专属**（新部署恒发 `interrupt_probe` ⇒ 不会落到 ABSENT）：
                    # 「resumed 为真但 interrupted_at 缺席」——**我方不可判**的一格。
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
                    #     在**老部署上**这是区分 ② 与 ③ 的**唯一**手段。
                    # ✅ 2026-07-30 事实更新：上面这套论证的前提（②③④ 在返回体上同形）**只在
                    # 老部署成立**。新部署（Zero `667e923`+）把三者拆成了显式令牌 ⇒ ③ 走
                    # `PROBE_FAILED`（已改为**拒绝**，因为拆开后拒绝不再误伤健康 resume）、
                    # ② 走 `NOT_PROBED`+resumed、④ 走 `CLEAN`。本分支因此**只服务老部署**，
                    # 内容保留不动（老部署仍在跑，它对那一代仍然逐字正确）。
                    # 零回归：本分支挂在 `resumed is True` 上。`resumed` 是新老部署的判别位
                    # （新 Zero 无条件回，老部署根本不发）⇒ 老部署走 `resumed is None`，
                    # 不进本分支、不打这条 WARNING，行为与换代前逐字一致。
                    # ✅ 2026-07-30：`describe_config` 已接入（见 `ZeroLinkClient` 同名方法），
                    # 但**本行条件不改挂它**——`resumed is True` 在这里同时排除了缺席四义的第①义
                    # （未探测·新建会话），而回读面答不了「本次 open 是不是 resume」。改挂 = 丢掉
                    # 该排除、在每次健康新建会话上误发本 WARNING。论证见本方法 docstring。
                    logger.warning(
                        "%s graceful_step: session=%s resume 重开回了 resumed=True 但**未带** %r"
                        "，也没有 %r（**老部署**：该代 Zero 不发显式中断态）"
                        "——此时该键缺席有四义（未探测·新建 / 未探测·活跃幂等重开 / 探测失败 / "
                        "探测干净），其中首义与 resumed=True 互斥、余下三者在返回体上同形，"
                        "我方无法判定"
                        "上一轮是否被中途取消，本帧按续跑处置。若 Zero 侧同时刻有"
                        "「中断探测失败」记录，则本帧很可能续跑在半截运行态上。"
                        "⚠ 归因提示：这条是**老部署不可判**，与新部署的 probe_failed"
                        "（对方自述探测失败 → 本帧会被拒绝）是不同的事，勿混读。",
                        LOG_MARKER_PROBE_UNDECIDABLE,
                        session_id,
                        _OPEN_SESSION_KEY_INTERRUPTED_AT,
                        _OPEN_SESSION_KEY_INTERRUPT_PROBE,
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
