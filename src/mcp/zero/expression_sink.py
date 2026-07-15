"""表达消费通路骨架（Zero → MCP 方向）——T4。

HeadPolicy：双通路取舍策略（AD-6）。
Mapper Protocols：FACS/韵律/生理 → 具体引擎参数的映射接口（本阶段只定接口）。
ExpressionSink：表达渲染终端 Protocol（如 Live2D / TTS 引擎）。
ExpressionRouter：解析 step_out → ExpressionBundle → 按 HeadPolicy 分发各 sink。

持有 asyncio.gather 的 task 引用，单 sink 失败不拖垮其他（gather return_exceptions=True）。
"""

from __future__ import annotations

import asyncio
import logging
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from src.agents.models.zero_affect import ExpressionBundle, ExpressionHead

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HeadPolicy
# ---------------------------------------------------------------------------


class HeadPolicy(StrEnum):
    """双通路表达头取舍策略（AD-6）。

    Zero 每轮 step() 输出 spontaneous（真情）与 voluntary（掩饰）两个头：
    - VOLUNTARY_ONLY（默认）：只向 sink 送 voluntary 头——社交面，掩饰面；
    - SPONTANEOUS_ONLY：只送 spontaneous 头——真情泄漏；
    - DUAL：主表情送 voluntary，微表情泄漏送 spontaneous（HANDOFF 建议形态）。

    DUAL 时 ExpressionRouter 以 voluntary 为主头、spontaneous 为微表情泄漏，
    渲染层自行决定如何混合（如 Live2D 用 voluntary 驱动主 blendshape，
    用 spontaneous 细微调制眼周 AU）。
    """

    VOLUNTARY_ONLY = "voluntary_only"
    SPONTANEOUS_ONLY = "spontaneous_only"
    DUAL = "dual"


# ---------------------------------------------------------------------------
# Mapper Protocols（本阶段只定接口，不实现具体引擎）
# ---------------------------------------------------------------------------


@runtime_checkable
class FacsMapper(Protocol):
    """将 ExpressionHead.facs_au（13 AU 子集）→ 引擎 blendshape 系数。

    返回 `dict[str, float]`：引擎 blendshape/参数名 → 系数（值域 [0,1]，
    只含被驱动的键，未驱动项由消费方默认静息 0）。
    参考实现：`src.mcp.zero.mappers.facs.ArkitFacsMapper`（AU → ARKit 52 blendshape）。
    """

    async def map(self, channel: ExpressionHead) -> dict[str, float]:
        """async：将 FACS 通道映射到引擎 blendshape 系数 dict[str, float]。"""
        ...


@runtime_checkable
class ProsodyMapper(Protocol):
    """将 ExpressionHead.prosody → 具体 TTS/声学引擎参数。

    量纲双方言（AD-5）由 mapper 实现方自行处理。
    """

    async def map(self, channel: ExpressionHead) -> Any:
        """async：将韵律通道映射到引擎参数 dict。"""
        ...


@runtime_checkable
class PhysiologyMapper(Protocol):
    """将 ExpressionHead.physiology → 具体引擎参数（如心率驱动呼吸动画）。"""

    async def map(self, channel: ExpressionHead) -> Any:
        """async：将生理通道映射到引擎参数 dict。"""
        ...


# ---------------------------------------------------------------------------
# ExpressionSink Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ExpressionSink(Protocol):
    """表达渲染终端（如 Live2D 引擎、TTS 引擎、日志 sink）。

    render() 接收完整 ExpressionBundle 与 HeadPolicy；
    由 ExpressionRouter 分发，单 sink 失败不拖垮其他。
    """

    async def render(self, bundle: ExpressionBundle, *, policy: HeadPolicy) -> None:
        """async：按 policy 渲染 bundle 中的表达头。"""
        ...


# ---------------------------------------------------------------------------
# ExpressionRouter
# ---------------------------------------------------------------------------


class ExpressionRouter:
    """解析 Zero step_out → ExpressionBundle，按 HeadPolicy 分发各 sink。

    设计要点：
    - 持有 asyncio.gather 的 coroutine 结果（return_exceptions=True），
      确保单 sink 失败 logger.warning 后不拖垮其他 sink。
    - route() 返回已解析的 ExpressionBundle 供调用方复用（如记录 metrics、
      传入后续编排节点），不需要调用方重新解析。
    - gather 结果引用被持有（赋值给局部变量 results），不丢弃 Promise。

    ⚠ sink 生命周期由**调用方**管理：有状态 sink（如 RenderingExpressionSink 的
    frames 只增不减）长期复用同一实例时须配合其 clear()，或每轮构造新实例，防无界增长。
    """

    def __init__(
        self,
        sinks: list[ExpressionSink],
        *,
        policy: HeadPolicy = HeadPolicy.VOLUNTARY_ONLY,
    ) -> None:
        self.sinks = sinks
        self.policy = policy

    async def route(self, step_out: dict[str, Any]) -> ExpressionBundle:
        """async：解析 step_out → ExpressionBundle，并发分发各 sink，返回 bundle。

        step_out 可为 Zero session.step() 完整返回 dict（含 "expression" 键）
        或直接 expression 子 dict——由 ExpressionBundle.from_step_output 兼容处理。

        HeadPolicy 说明（分发到 sink.render 时由 sink 按 policy 取用对应头）：
        - VOLUNTARY_ONLY：sink 应仅驱动 voluntary 头。
        - SPONTANEOUS_ONLY：sink 应仅驱动 spontaneous 头。
        - DUAL：sink 以 voluntary 为主头，spontaneous 为微表情泄漏，
          渲染层自行决定混合方式。

        单 sink 失败 → logger.warning，不影响其他 sink；
        所有 sink 完成后（无论成败）返回 bundle。
        """
        bundle = ExpressionBundle.from_step_output(step_out)

        if not self.sinks:
            return bundle

        coros = [sink.render(bundle, policy=self.policy) for sink in self.sinks]
        # 持有 gather 结果引用，不丢弃（python-code.md：不裸 asyncio.create_task 丢句柄）
        results: list[None | BaseException] = await asyncio.gather(*coros, return_exceptions=True)

        for idx, result in enumerate(results):
            if isinstance(result, BaseException):
                sink_name = type(self.sinks[idx]).__name__
                logger.warning(
                    "ExpressionSink[%d] %r render() 失败，已跳过: %s",
                    idx,
                    sink_name,
                    result,
                )

        return bundle
