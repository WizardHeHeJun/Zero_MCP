"""生理通道映射器（PhysiologyMapper 实现，蓝图第二阶段 P1）。

LinearPhysiologyMapper 将 ExpressionHead 中 PhysiologyChannel 的 **WESAD canonical** 三值
（heart_rate_bpm / skin_conductance μS / temperature_c）映射成引擎无关的生理驱动动画参数
PhysiologyParams，满足 PhysiologyMapper Protocol。

映射语义（引擎无关，供 Live2D / 3D avatar 等消费方驱动动画）：
- heart_rate_bpm → 呼吸速率 breath_rate_bpm（心肺耦合近似），并原样透传 HR 供直接驱动脉搏；
- skin_conductance（μS）→ 唤醒/出汗强度 skin_conductance_level [0,1]（除 skin_conductance_max_us
  归一，驱动脸红/出汗着色）；
- temperature_c → 皮肤温度水平 skin_temperature_level [0,1]（在 temperature_range 内归一，驱动
  面部温度/潮红着色）；temperature_c 缺省（过渡期占位无 temp）时该项为 None；
- pupil_mm → 瞳孔扩张系数 pupil_dilation [0,1]（**过渡期兼容**旧 avatar 契约；缺省时 None）。

canonical=WESAD 真 physiology_decoder 口径（2026-07-23 拍板，见 notes 2026-07-23 physiology 简报）：
skin_conductance 是 μS 物理单位（非 [0,1]）；temperature_c/pupil_mm 皆可选（跨仓迁移不原子），故
skin_temperature_level/pupil_dilation 对应可空——源字段缺省则该动画维无数据（None，消费方不驱动）。

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
    """引擎无关的生理驱动动画参数（canonical=WESAD）。

    heart_rate_bpm        : 心率（bpm）非负透传，供引擎直接驱动脉搏/心跳动画
                            （负值经 mapper clamp≥0，与 breath_rate_bpm 取向一致）。
    breath_rate_bpm       : 呼吸动画速率（每分钟呼吸次数），由心率经心肺耦合比推导；
                            驱动胸廓/肩部起伏循环动画。
    skin_conductance_level: 皮肤电导水平 [0,1]，由 skin_conductance（μS）除归一上界得出；
                            唤醒/出汗强度代理，驱动脸红/出汗着色（0=平静、1=高唤醒）。
    skin_temperature_level: 皮肤温度水平 [0,1]，由 temperature_c 在生理范围内归一；驱动面部
                            温度/潮红着色。**None** = 源无 temperature_c（过渡期占位），不驱动该维。
    pupil_dilation        : 瞳孔扩张系数 [0,1]，由 pupil_mm 在生理范围内归一（**过渡期兼容**旧
                            avatar 契约）。**None** = 源无 pupil_mm（WESAD 无此字段），不驱动。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    heart_rate_bpm: float = Field(..., ge=0.0, description="心率 bpm，非负透传")
    breath_rate_bpm: float = Field(..., ge=0.0, description="呼吸动画速率，每分钟呼吸次数")
    skin_conductance_level: float = Field(
        ..., ge=0.0, le=1.0, description="皮肤电导/唤醒水平 [0,1]（μS 归一）"
    )
    skin_temperature_level: float | None = Field(
        None, ge=0.0, le=1.0, description="皮肤温度水平 [0,1]；None=源无 temperature_c"
    )
    pupil_dilation: float | None = Field(
        None, ge=0.0, le=1.0, description="瞳孔扩张系数 [0,1]（过渡期）；None=源无 pupil_mm"
    )


# ---------------------------------------------------------------------------
# LinearPhysiologyMapper
# ---------------------------------------------------------------------------


class LinearPhysiologyMapper:
    """线性生理映射器——满足 PhysiologyMapper Protocol（结构化，不显式继承）。

    将 ExpressionHead.physiology（PhysiologyChannel，canonical=WESAD）映射为 PhysiologyParams：
    - breath_rate_bpm = clamp≥0(heart_rate_bpm / cardio_respiratory_ratio)；
    - heart_rate_bpm = clamp≥0(heart_rate_bpm)（非负透传，防负值/NaN 传入下游引擎）；
    - skin_conductance_level = clamp01(skin_conductance / skin_conductance_max_us)（μS→[0,1]）；
    - skin_temperature_level = clamp01((temperature_c - lo) / (hi - lo))，(lo,hi)=temperature_range;
      temperature_c 为 None（过渡期占位无 temp）→ None（不驱动该维）；
    - pupil_dilation = clamp01((pupil_mm - lo) / (hi - lo))，(lo,hi)=pupil_mm_range；
      pupil_mm 为 None（canonical WESAD 无此字段）→ None（不驱动该维）。

    async：对齐 PhysiologyMapper Protocol（Protocol.map 是 async def），当前纯标量计算
    无真正 I/O 等待，async 为预留——便于未来接入真实生理/渲染 SDK 时无缝替换。

    ⚠ **工程假设**：心肺耦合比 cardio_respiratory_ratio、皮电归一上界 skin_conductance_max_us、
    体温归一范围 temperature_range、瞳孔归一范围 pupil_mm_range 是引擎映射启发式（非文献推导：
    静息 HR:呼吸 ≈ 4:1、WESAD EDA 上界 ~20μS、皮肤温 30–40°C、瞳孔常态 3–5mm 的工程经验取值），
    消费方可按目标模型/绑定覆盖。（遵 agent-framework-rules：无据选择显式标工程假设。）

    ⚠ **保护策略有意不对称**（code-review W3）：作**除数**的 cardio_respiratory_ratio/
    skin_conductance_max_us 在 `__init__` 即 fail-fast（≤0 → ValueError，防 ZeroDivisionError）；作
    **归一范围** temperature_range/pupil_mm_range 退化（span≤0）**不构造期 raise**，而 map() 优雅
    回退对应维 None（"无有效范围→不驱动该维"，比 raise 更适合可选动画维；退化范围测试锁定）。

    Args:
        cardio_respiratory_ratio: 心率÷呼吸速率比（>0，≤0 构造期 raise），默认 4.0。
        skin_conductance_max_us : 皮电归一上界（μS，>0，≤0 构造期 raise），默认 20.0（WESAD 量级）。
        temperature_range       : (min_c, max_c) 温度归一范围，默认 (30,40)；退化→map() 回 None。
        pupil_mm_range          : (min_mm, max_mm) 瞳孔归一范围，默认 (3,5)；退化→map() 回退 None。
    """

    def __init__(
        self,
        *,
        cardio_respiratory_ratio: float = 4.0,
        skin_conductance_max_us: float = 20.0,
        temperature_range: tuple[float, float] = (30.0, 40.0),
        pupil_mm_range: tuple[float, float] = (3.0, 5.0),
    ) -> None:
        if cardio_respiratory_ratio <= 0.0:
            raise ValueError(
                f"cardio_respiratory_ratio 须 >0，实际 {cardio_respiratory_ratio}"
                "（作除数推导呼吸速率）"
            )
        if skin_conductance_max_us <= 0.0:
            raise ValueError(
                f"skin_conductance_max_us 须 >0，实际 {skin_conductance_max_us}（作除数归一皮电）"
            )
        self.cardio_respiratory_ratio = cardio_respiratory_ratio
        self.skin_conductance_max_us = skin_conductance_max_us
        self.temperature_range = temperature_range
        self.pupil_mm_range = pupil_mm_range

    async def map(self, channel: ExpressionHead) -> PhysiologyParams:
        """async：将 ExpressionHead 生理通道映射为 PhysiologyParams。

        async 为对齐 PhysiologyMapper Protocol 预留，当前纯标量计算，无实际 await。

        Args:
            channel: ExpressionHead，其 physiology 为 PhysiologyChannel（canonical=WESAD）。

        Returns:
            PhysiologyParams——引擎无关的生理驱动动画参数（temp/pupil 源缺省时对应维为 None）。
        """
        physiology = channel.physiology

        heart_rate_bpm = max(0.0, physiology.heart_rate_bpm)
        breath_rate_bpm = max(0.0, physiology.heart_rate_bpm / self.cardio_respiratory_ratio)
        skin_conductance_level = _clamp01(
            physiology.skin_conductance / self.skin_conductance_max_us
        )
        skin_temperature_level = self._normalize_temperature(physiology.temperature_c)
        pupil_dilation = self._normalize_pupil(physiology.pupil_mm)

        logger.debug(
            "LinearPhysiologyMapper.map: HR=%.1f → breath=%.2f, sc=%.2fμS → level=%.3f, "
            "temp=%s → tlevel=%s, pupil=%s → dilation=%s",
            physiology.heart_rate_bpm,
            breath_rate_bpm,
            physiology.skin_conductance,
            skin_conductance_level,
            physiology.temperature_c,
            skin_temperature_level,
            physiology.pupil_mm,
            pupil_dilation,
        )
        return PhysiologyParams(
            heart_rate_bpm=heart_rate_bpm,
            breath_rate_bpm=breath_rate_bpm,
            skin_conductance_level=skin_conductance_level,
            skin_temperature_level=skin_temperature_level,
            pupil_dilation=pupil_dilation,
        )

    def _normalize_temperature(self, temperature_c: float | None) -> float | None:
        """将 temperature_c 在 temperature_range 内归一到 [0,1]（clamp）。

        temperature_c 为 None（过渡期占位无 temp）→ None（不驱动该维）；
        span ≤ 0（范围退化）→ 回退 None 并 warning，避免除零。
        """
        if temperature_c is None:
            return None
        lo, hi = self.temperature_range
        span = hi - lo
        if span <= 0.0:
            logger.warning(
                "temperature_range=%r 退化（span≤0），皮肤温度水平回退 None", self.temperature_range
            )
            return None
        return _clamp01((temperature_c - lo) / span)

    def _normalize_pupil(self, pupil_mm: float | None) -> float | None:
        """将 pupil_mm 在 pupil_mm_range 内归一到 [0,1]（clamp）。

        pupil_mm 为 None（canonical WESAD 无此字段）→ None（不驱动该维）；
        span ≤ 0（范围退化）→ 回退 None 并 warning，避免除零。
        """
        if pupil_mm is None:
            return None
        lo, hi = self.pupil_mm_range
        span = hi - lo
        if span <= 0.0:
            logger.warning(
                "pupil_mm_range=%r 退化（span≤0），瞳孔扩张回退 None", self.pupil_mm_range
            )
            return None
        return _clamp01((pupil_mm - lo) / span)
