"""感知半程适配器——将任意 async callable 包装为 PerceptionChannel。

CallablePerceptionChannel 让「async 函数 → ModalityPrior|None」的感知源
无需显式继承即可接入 PerceptionHub（结构化子类型满足 PerceptionChannel Protocol）。

典型用法::

    async def my_sensor() -> ModalityPrior | None:
        return ModalityPrior(modality="audio", mu=(0.1, 0.3), precision=(0.5, 0.5))

    ch = CallablePerceptionChannel(name="audio", sense_fn=my_sensor)
    hub = PerceptionHub([ch])
    priors = await hub.collect()
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from src.agents.models.zero_affect import ModalityPrior


class CallablePerceptionChannel:
    """将任意「async () → ModalityPrior|None」包装为 PerceptionChannel 结构化满足者。

    结构上满足 PerceptionChannel Protocol（无需继承）：
    - name: str 属性（runtime_checkable isinstance 检查数据属性）。
    - async sense() -> ModalityPrior | None（委托给 sense_fn）。

    Args:
        name: 模态标识符，如 ``"vision"`` / ``"audio"`` / ``"physiology"``。
        sense_fn: 无参 async callable，返回 ModalityPrior 或 None（无证据时）。
    """

    def __init__(
        self,
        name: str,
        sense_fn: Callable[[], Awaitable[ModalityPrior | None]],
    ) -> None:
        self.name = name
        self.sense_fn = sense_fn

    async def sense(self) -> ModalityPrior | None:
        """async：委托 sense_fn 执行感知，返回模态先验或 None。"""
        return await self.sense_fn()
