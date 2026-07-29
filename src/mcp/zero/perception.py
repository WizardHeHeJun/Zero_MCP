"""感知输入通路骨架（MCP → Zero 方向）——T3。

PerceptionChannel：单模态感知源 Protocol。
PerceptionHub：汇聚多路 PerceptionChannel，产出独立先验列表，对齐 Zero agents 层
affect_core 模块（现为 `src/agents/affect_core.py`，该路径仍在迁移中）的 external streams
组装段——即 `expand_external_priors` 调用点前后那个 `streams` 列表，
**形状 `(name, (μv, μa), (Πv, Πa))`**（形状写死在此处，比指向对方任何坐标都稳）。

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
        # 重依赖延迟 import 是否已串行预热（见 prepare_all：防并发首次 import 的半成品模块竞态）
        self.prepared = False

    def reset_all(self) -> None:
        """被试切换时的统一入口：对**有状态**通道（实现了 reset()）调用其 reset()。

        有状态感知通道（**维护窗间基线历史**的 EdaChannel / HrvChannel）在被试切换时须清空
        历史，否则新被试沿用旧被试的自适应基线（跨被试污染）。无 reset() 的无状态通道跳过。
        鸭子类型检测——不强制 PerceptionChannel Protocol 带 reset()（多数通道无状态）。
        """
        for ch in self.channels:
            reset = getattr(ch, "reset", None)
            if callable(reset):
                reset()

    async def prepare_all(self) -> None:
        """**串行**预热各通道的重依赖延迟 import（并发派发前调用，幂等）。

        为什么必须串行、且必须在 gather 之前（真实缺陷，非预防性优化）：各通道的 ``_process``
        经 ``asyncio.to_thread`` 落到线程池并发执行，而重依赖是**在工作线程里首次 import** 的。
        实测竞态（完整 traceback 见 pitfalls「并发首次 import torch 撞 scipy array-API 探测」）：

            AudioChannel 线程   : import torch  →  torch 已进 sys.modules 但**尚未初始化完**
            HrvChannel  线程   : nk.hrv_time → scipy.stats.iqr → scipy 的 array-API 分发
                                 → array_api_compat `_issubclass_fast`
                                 → getattr(sys.modules["torch"], "Tensor")
            → AttributeError: partially initialized module 'torch' has no attribute 'Tensor'

        即**并非 physio 通道自己去 import torch**，而是 SciPy 会**探测** ``sys.modules["torch"]``
        以判断入参是否 torch 张量；恰好撞上另一线程的半成品 torch 就抛。后果是 physio 先验被
        ``collect()`` 当作「通道异常」**静默跳过**——冷启动下间歇丢流，且日志之外无任何征兆。

        预热把这些 import 挪到**并发之前逐个完成**，此后 ``sys.modules`` 里恒是完整模块，
        探测不再有半成品窗口。``prepare()`` 是**可选**协议（鸭子类型，同 ``reset()``）：
        无重依赖的通道不必实现。单通道预热失败**不阻断**其余通道——该通道会在 ``sense()``
        时按既有约定优雅回退（缺库→warning+None）。
        """
        for ch in self.channels:
            prepare = getattr(ch, "prepare", None)
            if not callable(prepare):
                continue
            try:
                await prepare()
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                # 预热失败不致命：该通道 sense() 会走既有优雅回退路径
                logger.warning(
                    "PerceptionChannel %r 预热失败（不阻断，sense() 时将优雅回退）: %s",
                    getattr(ch, "name", ch),
                    exc,
                )
        self.prepared = True

    async def collect(self) -> list[ModalityPrior]:
        """async：并发收集各通道先验，单通道失败/无证据降级跳过。

        实现：asyncio.gather(return_exceptions=True) 收集所有结果；
        异常 → logger.warning 记录通道名与错误，跳过；
        None → 本轮无证据，跳过；
        ModalityPrior → 保留（不做任何融合，顺序与 channels 一致）。

        首次调用前先 ``prepare_all()`` 串行预热重依赖 import——避免并发首次 import 的
        半成品模块竞态导致通道被静默跳过（见 ``prepare_all`` docstring 的实测 traceback）。
        """
        if not self.prepared:
            await self.prepare_all()
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
        对齐 Zero agents 层 affect_core 模块的 external streams 组装段形状
        （`expand_external_priors` 调用点前后的 `streams` 列表；跨仓只锚符号不锚行号）。

        **Q3 已交付（Zero 2026-07-15，commit 143ac72）**：正式多流注入口 = Zero 专用字段
        ``external_priors: list[(name, (μ_v,μ_a), (Π_v,Π_a))]``（默认空 = 零回归），
        AffectCore 展开后把每条 append 进 streams 竞争融合。⚠ **M1 议会裁决：精度为
        逐维 tuple (Π_v,Π_a)，非标量 float**（三席强收敛：面部 valence 强/语音 arousal
        强/生理 valence 盲，逐维信噪比不对称）——本方法返回的 tuple 精度即最终契约形状，
        无需收敛。产物经 external_priors.build_external_priors_override() 构造为
        session.step(state_overrides=...) 载荷（M3/M6 客户端 fail-fast，阈值对齐 Zero）。

        ⛔ Zero 明确否决了借 ``text_affect``（PerceptionAgent 每轮覆写）或
        ``interlocutor_affect``（ToM 共情偏置）挪用作过渡——external_priors 是唯一正式口，
        本方法产物经 build_external_priors_override 接入，不走任何 state_overrides 过渡路径。
        """
        return [p.as_stream() for p in priors]
