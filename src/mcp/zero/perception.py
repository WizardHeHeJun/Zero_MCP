"""感知输入通路骨架（MCP → Zero 方向）——T3。

PerceptionChannel：单模态感知源 Protocol。
PerceptionHub：汇聚多路 PerceptionChannel，产出独立先验列表，
对齐 Zero affect_core streams 形状（D:\\Zero\\src\\affect_core.py:77-95）。

AD-3 硬约束：**禁止均值融合**——各先验独立保留，竞争融合是 Zero 内核的事。
单通道失败（异常或返回 None）→ 降级跳过，不拖垮整体。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol, runtime_checkable

from src.agents.models.zero_affect import ModalityPrior

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PerceptionChannel Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PerceptionChannel(Protocol):
    """单模态感知源。

    实现方须：
    - 提供 name: str 属性（标识模态，如 "vision" / "audio" / "physiology"）。
    - 实现 async sense() -> ModalityPrior | None：
        - 返回 ModalityPrior：本轮有证据，含 (μ_v,μ_a) + (Π_v,Π_a)。
        - 返回 None：本轮无证据（如采集失败、置信度过低），PerceptionHub 跳过。
    """

    name: str

    async def sense(self) -> ModalityPrior | None:
        """async：感知一次并返回模态先验；无证据时返回 None。"""
        ...


# ---------------------------------------------------------------------------
# PerceptionHub
# ---------------------------------------------------------------------------


class PerceptionHub:
    """多模态感知汇聚器。

    汇聚 N 个 PerceptionChannel，产出**独立先验列表**（不做均值融合，AD-3）。
    单通道失败不拖垮其他通道（asyncio.gather return_exceptions=True）。

    典型用法::

        hub = PerceptionHub([VisionChannel(), AudioChannel()])
        priors = await hub.collect()
        streams = PerceptionHub.as_zero_streams(priors)   # → Zero streams 形状
        overrides = PerceptionHub.as_state_overrides(priors)  # → state_overrides 形态
    """

    def __init__(self, channels: list[PerceptionChannel]) -> None:
        self.channels = channels

    async def collect(self) -> list[ModalityPrior]:
        """async：并发收集各通道先验，单通道失败/无证据降级跳过。

        实现：asyncio.gather(return_exceptions=True) 收集所有结果；
        异常 → logger.warning 记录通道名与错误，跳过；
        None → 本轮无证据，跳过；
        ModalityPrior → 保留（不做任何融合，顺序与 channels 一致）。
        """
        tasks = [ch.sense() for ch in self.channels]
        results: list[ModalityPrior | None | BaseException] = await asyncio.gather(
            *tasks, return_exceptions=True
        )
        priors: list[ModalityPrior] = []
        for ch, result in zip(self.channels, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "PerceptionChannel %r sense() 异常，本轮跳过: %s",
                    ch.name,
                    result,
                )
            elif result is None:
                logger.warning(
                    "PerceptionChannel %r 本轮无证据（返回 None），跳过",
                    ch.name,
                )
            else:
                priors.append(result)
        return priors

    @staticmethod
    def as_zero_streams(
        priors: list[ModalityPrior],
    ) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
        """将先验列表转为 Zero affect_core streams 形状。

        对齐 D:\\Zero\\src\\affect_core.py:77-95::

            streams = [(name, (μ_v, μ_a), (Π_v, Π_a)), ...]

        每条先验调用 ModalityPrior.as_stream()，**独立保留**，不做均值（AD-3）。
        """
        return [p.as_stream() for p in priors]

    @staticmethod
    def as_state_overrides(priors: list[ModalityPrior]) -> dict[str, Any]:
        """将先验转为 Zero session.step(stim, state_overrides={...}) 形态。

        当前实现：透传首条先验的 mu 作为 text_affect（优先级最高的模态先验）。
        Zero 的 text_affect 先验以 text_affect_precision（默认 0.3）加权注入
        （affect_core.py:77-95 / chat_driver.py:313-320）。

        多条先验时只透传第一条，其余独立先验应经 as_zero_streams() 走多流接口。
        TODO(Q3)：待 Zero 在 AffectCore streams 开放正式多流注入口后，
        将所有先验以完整 streams 形式注入，废弃此单条 text_affect 兜底。
        """
        if not priors:
            return {}
        first = priors[0]
        return {"text_affect": first.mu}
