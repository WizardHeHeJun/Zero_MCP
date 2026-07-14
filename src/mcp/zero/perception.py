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
from typing import Protocol, runtime_checkable

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
        streams = PerceptionHub.as_zero_streams(priors)   # → 独立先验流（待接 external_priors）
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
        """将先验列表转为 Zero 独立先验流形状（不均值，AD-3）。

        每条先验调用 ModalityPrior.as_stream() → (name, (μ_v,μ_a), (Π_v,Π_a))，
        对齐 Zero 内部 streams 形状（D:\\Zero\\src\\affect_core.py:77-95）。

        **Q3 已定（Zero 回传 2026-07-14）**：正式多流注入口 = Zero 将新增的专用字段
        ``external_priors: list[(name, (μ_v,μ_a), precision: float)]``（**需 Zero 走
        PRP + 科学家议会门后落地**；默认空 = 零回归），AffectCore 把每条 append 进 streams。
        届时本方法产物接进该字段。⚠ external_priors 的 precision 是**标量 float**
        （AffectCore 广播为 (p,p)），本方法当前返回 tuple 精度——待 external_priors
        落地时按最终签名收敛精度形状。

        ⛔ Zero 明确否决了借 ``text_affect``（PerceptionAgent 每轮覆写）或
        ``interlocutor_affect``（ToM 共情偏置）挪用作过渡——故本模块**不提供**
        state_overrides 过渡路径；external_priors 落地前不发多模态注入路径。
        """
        return [p.as_stream() for p in priors]
