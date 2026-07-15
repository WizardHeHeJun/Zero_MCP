"""生理通道映射器（PhysiologyMapper 实现，蓝图第二阶段 P1）。

LinearPhysiologyMapper 将 ExpressionHead 中 PhysiologyChannel 的三值
（heart_rate_bpm / skin_conductance / pupil_mm）映射成引擎无关的
生理驱动动画参数 PhysiologyParams，满足 PhysiologyMapper Protocol。

映射语义（引擎无关，供 Live2D / 3D avatar 等消费方驱动动画）：
- heart_rate_bpm → 呼吸动画速率 breath_rate_bpm（心肺耦合近似），并原样透传 HR 供直接驱动脉搏；
- pupil_mm       → 归一化瞳孔扩张系数 pupil_dilation [0,1]（驱动眼球瞳孔缩放）；
- skin_conductance → 唤醒/出汗强度 skin_conductance_level [0,1]（驱动脸红/出汗着色）。

async 为对齐 PhysiologyMapper Protocol 预留，当前实现纯标量计算无实际 I/O 等待
（便于未来接入真实生理/渲染 SDK 时无缝替换，照 prosody/facs 同款风格）。
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

from src.agents.models.zero_affect import ExpressionHead

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    """将 x 截断到 [0.0, 1.0]。"""
    return max(0.0, min(1.0, x))


# ---------------------------------------------------------------------------
# PhysiologyParams
# ---------------------------------------------------------------------------


class PhysiologyParams(BaseModel):
    """引擎无关的生理驱动动画参数。

    heart_rate_bpm        : 心率（bpm）非负透传，供引擎直接驱动脉搏/心跳动画
                            （负值经 mapper clamp≥0，与 breath_rate_bpm 取向一致）。
    breath_rate_bpm       : 呼吸动画速率（每分钟呼吸次数），由心率经心肺耦合比推导；
                            驱动胸廓/肩部起伏循环动画。
    pupil_dilation        : 瞳孔扩张系数 [0,1]，由 pupil_mm 在生理范围内归一；
                            驱动眼球瞳孔缩放（0=最小、1=最大扩张）。
    skin_conductance_level: 皮肤电导水平 [0,1]，唤醒/出汗强度代理；
                            驱动脸红/出汗着色（0=平静、1=高唤醒）。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    heart_rate_bpm: float = Field(..., ge=0.0, description="心率 bpm，非负透传")
    breath_rate_bpm: float = Field(..., ge=0.0, description="呼吸动画速率，每分钟呼吸次数")
    pupil_dilation: float = Field(..., ge=0.0, le=1.0, description="瞳孔扩张系数 [0,1]")
    skin_conductance_level: float = Field(
        ..., ge=0.0, le=1.0, description="皮肤电导/唤醒水平 [0,1]"
    )


# ---------------------------------------------------------------------------
# LinearPhysiologyMapper
# ---------------------------------------------------------------------------


class LinearPhysiologyMapper:
    """线性生理映射器——满足 PhysiologyMapper Protocol（结构化，不显式继承）。

    将 ExpressionHead.physiology（PhysiologyChannel）映射为 PhysiologyParams：
    - breath_rate_bpm = clamp≥0(heart_rate_bpm / cardio_respiratory_ratio)；
    - pupil_dilation  = clamp01((pupil_mm - lo) / (hi - lo))，(lo,hi)=pupil_mm_range；
    - skin_conductance_level = clamp01(skin_conductance)；
    - heart_rate_bpm = clamp≥0(heart_rate_bpm)（非负透传，防负值/NaN 传入下游引擎）。

    async：对齐 PhysiologyMapper Protocol（Protocol.map 是 async def），当前纯标量计算
    无真正 I/O 等待，async 为预留——便于未来接入真实生理/渲染 SDK 时无缝替换。

    ⚠ **工程假设**：心肺耦合比 cardio_respiratory_ratio 与瞳孔归一范围 pupil_mm_range
    是引擎映射启发式（非文献推导，静息 HR:呼吸 ≈ 4:1、瞳孔常态 3–5mm 的工程经验取值），
    消费方可按目标模型/绑定覆盖。（遵 agent-framework-rules：无据选择显式标工程假设。）

    Args:
        cardio_respiratory_ratio: 心率÷呼吸速率比（>0），默认 4.0。
        pupil_mm_range          : (min_mm, max_mm) 瞳孔归一范围，默认 (3.0, 5.0)。
    """

    def __init__(
        self,
        *,
        cardio_respiratory_ratio: float = 4.0,
        pupil_mm_range: tuple[float, float] = (3.0, 5.0),
    ) -> None:
        if cardio_respiratory_ratio <= 0.0:
            raise ValueError(
                f"cardio_respiratory_ratio 须 >0，实际 {cardio_respiratory_ratio}"
                "（作除数推导呼吸速率）"
            )
        self.cardio_respiratory_ratio = cardio_respiratory_ratio
        self.pupil_mm_range = pupil_mm_range

    async def map(self, channel: ExpressionHead) -> PhysiologyParams:
        """async：将 ExpressionHead 生理通道映射为 PhysiologyParams。

        async 为对齐 PhysiologyMapper Protocol 预留，当前纯标量计算，无实际 await。

        Args:
            channel: ExpressionHead，其 physiology 为 PhysiologyChannel。

        Returns:
            PhysiologyParams——引擎无关的生理驱动动画参数。
        """
        physiology = channel.physiology

        heart_rate_bpm = max(0.0, physiology.heart_rate_bpm)
        breath_rate_bpm = max(0.0, physiology.heart_rate_bpm / self.cardio_respiratory_ratio)
        pupil_dilation = self._normalize_pupil(physiology.pupil_mm)
        skin_conductance_level = _clamp01(physiology.skin_conductance)

        logger.debug(
            "LinearPhysiologyMapper.map: HR=%.1f → breath=%.2f, pupil=%.2fmm → dilation=%.3f, "
            "sc=%.3f",
            physiology.heart_rate_bpm,
            breath_rate_bpm,
            physiology.pupil_mm,
            pupil_dilation,
            skin_conductance_level,
        )
        return PhysiologyParams(
            heart_rate_bpm=heart_rate_bpm,
            breath_rate_bpm=breath_rate_bpm,
            pupil_dilation=pupil_dilation,
            skin_conductance_level=skin_conductance_level,
        )

    def _normalize_pupil(self, pupil_mm: float) -> float:
        """将 pupil_mm 在 pupil_mm_range 内归一到 [0,1]（clamp）。

        span ≤ 0（范围退化）时回退 0.0 并 warning，避免除零。
        """
        lo, hi = self.pupil_mm_range
        span = hi - lo
        if span <= 0.0:
            logger.warning(
                "pupil_mm_range=%r 退化（span≤0），瞳孔扩张回退 0.0", self.pupil_mm_range
            )
            return 0.0
        return _clamp01((pupil_mm - lo) / span)
