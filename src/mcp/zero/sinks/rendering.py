"""表达半程渲染终端——RenderingExpressionSink 与 RenderFrame。

RenderingExpressionSink 结构上满足 ExpressionSink Protocol（无需继承），
按 HeadPolicy 从 ExpressionBundle 中取头，经可选 ProsodyMapper / FacsMapper /
PhysiologyMapper 映射后，将渲染指令追加到 self.frames（供检视 / 测试 / 后续 sink 链消费）。

透传说明：
- facs_au          : 原样透传（供其他 mapper / 调试）。
- facs_mapped      : FacsMapper 输出（有 facs_mapper 时为 ARKit blendshape 系数 dict，否则 None）。
- physiology       : 原样透传 PhysiologyChannel.model_dump()（供调试及其他消费方）。
- physiology_mapped: PhysiologyMapper 输出（有 mapper 时为 PhysiologyParams，否则 None）。
- prosody          : 已经 ProsodyMapper 映射（有 prosody_mapper 时为 ProsodyParams，否则 None）。

典型用法::

    from src.mcp.zero.mappers.facs import ArkitFacsMapper
    from src.mcp.zero.mappers.physiology import LinearPhysiologyMapper
    from src.mcp.zero.mappers.prosody import LinearProsodyMapper

    sink = RenderingExpressionSink(
        prosody_mapper=LinearProsodyMapper(),
        facs_mapper=ArkitFacsMapper(),
        physiology_mapper=LinearPhysiologyMapper(),
    )
    router = ExpressionRouter([sink], policy=HeadPolicy.DUAL)
    bundle = await router.route(step_out)
    frames = sink.frames   # list[RenderFrame]，可检视 / 断言
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.agents.models.zero_affect import ExpressionBundle, ExpressionHead
from src.mcp.zero.expression_sink import FacsMapper, HeadPolicy, PhysiologyMapper, ProsodyMapper
from src.mcp.zero.mappers.physiology import PhysiologyParams
from src.mcp.zero.mappers.prosody import ProsodyParams

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RenderFrame
# ---------------------------------------------------------------------------


class RenderFrame(BaseModel):
    """一条渲染指令——ExpressionHead 展开后的扁平化帧。

    head             : 来源头标识（"spontaneous" 或 "voluntary"）。
    is_micro         : DUAL 策略下 spontaneous 微表情泄漏帧标为 True，主帧为 False。
    text_label       : 直接来自 ExpressionHead.text_label。
    facs_au          : 原样透传 ExpressionHead.facs_au（13 AU 子集），供调试及其他 mapper 消费。
    facs_mapped      : FacsMapper（如 ArkitFacsMapper）输出的 ARKit blendshape 系数
                       dict[str, float]；有 facs_mapper 时由 ArkitFacsMapper.map() 填充，
                       否则为 None。只含被驱动的 blendshape，未驱动项由消费方默认静息 0。
    physiology       : 原样透传 ExpressionHead.physiology.model_dump()，供调试及其他消费方。
    physiology_mapped: PhysiologyMapper（如 LinearPhysiologyMapper）输出的引擎无关生理
                       动画参数 PhysiologyParams；有 physiology_mapper 时填充，否则为 None。
    prosody          : 有 prosody_mapper 时为 ProsodyMapper.map() 映射结果，否则 None。
    """

    model_config = ConfigDict(extra="forbid")

    head: Literal["spontaneous", "voluntary"]
    is_micro: bool = False
    text_label: str
    facs_au: dict[str, float]
    facs_mapped: dict[str, float] | None = None
    # physiology 原样透传 model_dump()——canonical=WESAD 后 temperature_c/pupil_mm 可选（可 None），
    # 故值类型放宽为 float | None（透传保真，不吞 None）。
    physiology: dict[str, float | None]
    physiology_mapped: PhysiologyParams | None = None
    prosody: ProsodyParams | None


# ---------------------------------------------------------------------------
# RenderingExpressionSink
# ---------------------------------------------------------------------------


class RenderingExpressionSink:
    """渲染终端——结构上满足 ExpressionSink Protocol（无需继承）。

    收集所有渲染帧到 self.frames，供测试断言、调试检视与后续 sink 链消费。

    facs_au 原样透传；facs_mapped 经 facs_mapper 映射（有时为 ARKit blendshape 系数，
    否则 None）。physiology 原样透传；physiology_mapped 经 physiology_mapper 映射
    （有时为 PhysiologyParams，否则 None）。prosody 已经 prosody_mapper 映射
    （有 mapper 时输出 ProsodyParams，否则 None）。

    零回归保证：facs_mapper / physiology_mapper 默认 None → 对应 mapped 字段 = None，
    行为与旧版完全一致。

    ⚠ 生命周期：self.frames **只增不减**——长期运行 pipeline 中同一实例被多轮
    route()/render() 复用时会无界增长。调用方须二选一：每轮消费完帧后调 clear()，
    或每轮构造新 sink 实例。

    Args:
        prosody_mapper   : 可选 ProsodyMapper 实现；为 None 时 RenderFrame.prosody = None。
        facs_mapper      : 可选 FacsMapper 实现（如 ArkitFacsMapper）；
                           为 None 时 RenderFrame.facs_mapped = None（默认，零回归）。
        physiology_mapper: 可选 PhysiologyMapper 实现（如 LinearPhysiologyMapper）；
                           为 None 时 RenderFrame.physiology_mapped = None（默认，零回归）。
    """

    def __init__(
        self,
        *,
        prosody_mapper: ProsodyMapper | None = None,
        facs_mapper: FacsMapper | None = None,
        physiology_mapper: PhysiologyMapper | None = None,
    ) -> None:
        self.prosody_mapper = prosody_mapper
        self.facs_mapper = facs_mapper
        self.physiology_mapper = physiology_mapper
        self.frames: list[RenderFrame] = []

    def clear(self) -> None:
        """清空已收集的渲染帧（生命周期管理，见类 docstring）。

        长期运行 pipeline 中多轮复用同一实例时，调用方应在每轮消费完帧后调此方法，
        防止 self.frames 无界增长。
        """
        self.frames.clear()

    async def render(
        self,
        bundle: ExpressionBundle,
        *,
        policy: HeadPolicy,
    ) -> None:
        """async：按 policy 取头、构造 RenderFrame 并追加到 self.frames。

        策略分支：
        - VOLUNTARY_ONLY：1 帧，voluntary 头，is_micro=False。
        - SPONTANEOUS_ONLY：1 帧，spontaneous 头，is_micro=False。
        - DUAL：2 帧——主帧 voluntary(is_micro=False) +
          微表情泄漏帧 spontaneous(is_micro=True)。

        帧追加顺序：主帧在前，泄漏帧在后（DUAL 时）。
        """
        if policy is HeadPolicy.VOLUNTARY_ONLY:
            frame = await self._build_frame(bundle.voluntary, head_name="voluntary", is_micro=False)
            self.frames.append(frame)
        elif policy is HeadPolicy.SPONTANEOUS_ONLY:
            frame = await self._build_frame(
                bundle.spontaneous, head_name="spontaneous", is_micro=False
            )
            self.frames.append(frame)
        else:
            # DUAL：主帧 voluntary + 微表情泄漏帧 spontaneous
            main_frame = await self._build_frame(
                bundle.voluntary, head_name="voluntary", is_micro=False
            )
            micro_frame = await self._build_frame(
                bundle.spontaneous, head_name="spontaneous", is_micro=True
            )
            self.frames.append(main_frame)
            self.frames.append(micro_frame)

    async def _build_frame(
        self,
        head: ExpressionHead,
        *,
        head_name: Literal["spontaneous", "voluntary"],
        is_micro: bool,
    ) -> RenderFrame:
        """async：将单个 ExpressionHead 构造为 RenderFrame。

        prosody 映射     ：有 prosody_mapper 时 await map(head)，否则 None。
        facs_mapped      ：有 facs_mapper 时 await map(head)，否则 None。
        physiology_mapped：有 physiology_mapper 时 await map(head)，否则 None。
        facs_au / physiology 原样透传（保留原始值供调试及其他消费方）。
        """
        prosody: ProsodyParams | None
        if self.prosody_mapper is not None:
            prosody = await self.prosody_mapper.map(head)
        else:
            prosody = None

        facs_mapped: dict[str, float] | None
        if self.facs_mapper is not None:
            facs_mapped = await self.facs_mapper.map(head)
        else:
            facs_mapped = None

        physiology_mapped: PhysiologyParams | None
        if self.physiology_mapper is not None:
            physiology_mapped = await self.physiology_mapper.map(head)
        else:
            physiology_mapped = None

        return RenderFrame(
            head=head_name,
            is_micro=is_micro,
            text_label=head.text_label,
            facs_au=dict(head.facs_au),
            facs_mapped=facs_mapped,
            physiology=head.physiology.model_dump(),
            physiology_mapped=physiology_mapped,
            prosody=prosody,
        )
