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

from src.agents.models.zero_affect import ExpressionHead, PhysiologyChannel

logger = logging.getLogger(__name__)

_SCALE_MISMATCH_MARKER: str = "physiology 口径失配（W6）"
"""口径失配 warning 的稳定前缀——与本模块「归一范围退化」等其它 warning 区分。

测试按此串筛选记录：若只断言「有 warning」，退化范围那条会让用例**红在错误的原因上**。
"""


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

    ⚠⚠ **本 mapper 按 canonical μS 标定——legacy 口径须显式配 `skin_conductance_max_us=1.0`**
    （code-review W6）：Zero 门开/真模型出 canonical `skin_conductance`=μS[0,20]，默认上界 20 归一
    正确；但 Zero **门关无真模型**出的 legacy sc 已是 [0,1]（`clamp(|arousal|)`），默认 μS 归一会再
    除 20 → **系统性欠标度 ~20×**（legacy sc=1.0→level≈0.05），且**不报错**（超集只保证解析、不保
    证消费标度）。physiology 通道**无量纲兄弟键**（不同 prosody `prosody_scale`），mapper 无法自识
    别口径。消费方接线**须知所连 Zero 口径**：canonical→默认；legacy→配 `max_us=1.0`。
    判据：`temperature_c` 出现≈canonical(μS)、`pupil_mm` 出现且无 temp≈legacy([0,1])。行为差异见
    test_zero_physiology_mapper.py::TestLegacyScaleConsumptionGap。

    **失配观测（2026-07-29 补，Zero 12:25 回执 D-5）**：上述判据已从纯 docstring 升级为
    `_detect_scale_mismatch()` + `map()` 里一条 **warning**（实例级去重）。它**只观测、不改任何
    数值**——三条数值锁（TestLegacyScaleConsumptionGap）原样成立，即本改动的零回归看门狗。
    ⚠ **绝不可据此宣称「20× 欠标度已解决」**：判据挂在 `temperature_c`/`pupil_mm` **键的有无**上，
    Zero 一旦增删这两个键（其 §4.4-4 正要动键集）本启发式即**静默失效**——既不再报，也不误报，
    调用方无从察觉。真解仍是 Zero 允诺的 `physiology_scale` 量纲兄弟键（对齐 `prosody_scale`），
    落地后本判据须改按兄弟键判定、并把形状启发式降级为兜底。

    ⚠ **保护策略有意不对称**（code-review W3）：作**除数**的 cardio_respiratory_ratio/
    skin_conductance_max_us 在 `__init__` 即 fail-fast（≤0 → ValueError，防 ZeroDivisionError）；作
    **归一范围** temperature_range/pupil_mm_range 退化（span≤0）**不构造期 raise**，而 map() 优雅
    回退对应维 None（"无有效范围→不驱动该维"，比 raise 更适合可选动画维；退化范围测试锁定）。

    Args:
        cardio_respiratory_ratio: 心率÷呼吸速率比（>0，≤0 构造期 raise），默认 4.0。
        skin_conductance_max_us : 皮电归一上界（μS，>0，≤0 构造期 raise），默认 20.0（WESAD 量级）。
        temperature_range       : (min_c, max_c) 温度归一范围，默认 (30,40)；退化→map() 回 None。
        pupil_mm_range          : (min_mm, max_mm) 瞳孔归一范围，默认 (3,5)；退化→map() 回退 None。

    Attributes:
        scale_mismatch_warned: 口径失配 warning 的**实例级**去重标志（首次失配后置 True，避免逐帧
            刷屏）。有意不做模块级去重——那会让测试顺序相关（先跑的用例吃掉 warning，后跑的假绿）。
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
        self.scale_mismatch_warned = False

    async def map(self, channel: ExpressionHead) -> PhysiologyParams:
        """async：将 ExpressionHead 生理通道映射为 PhysiologyParams。

        async 为对齐 PhysiologyMapper Protocol 预留，当前纯标量计算，无实际 await。

        Args:
            channel: ExpressionHead，其 physiology 为 PhysiologyChannel（canonical=WESAD）。

        Returns:
            PhysiologyParams——引擎无关的生理驱动动画参数（temp/pupil 源缺省时对应维为 None）。
        """
        physiology = channel.physiology

        # 口径失配观测（W6·零回归）：**只发 warning，不改任何数值**——数值分支在下面原样保留。
        if not self.scale_mismatch_warned:
            mismatch_reason = self._detect_scale_mismatch(physiology)
            if mismatch_reason is not None:
                self.scale_mismatch_warned = True
                logger.warning("%s：%s", _SCALE_MISMATCH_MARKER, mismatch_reason)

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

    def _detect_scale_mismatch(self, physiology: PhysiologyChannel) -> str | None:
        """按形状启发式判定「来源口径 ⟂ 本 mapper 的 sc 标度设定」是否失配。

        返回**原因串**（而非 bool）：调用方直接把它写进日志，测试也据此断言「红在正确的原因上」
        ——只判 True/False 会让「因另一个原因发的 warning」被当成通过。无失配返回 None。

        判据（与类 docstring 的 W6 判据同源）：
        - `temperature_c` 出现 ⇒ 疑似 canonical（`skin_conductance` 是 μS）；
        - `pupil_mm` 出现且无 `temperature_c` ⇒ 疑似 legacy（`skin_conductance` 已是 [0,1]）；
        - 两键皆无 ⇒ **判不出**，静默放行（不猜、不误报）。

        ⚠ 这是**观测**不是**保证**：判据挂在键的有无上，Zero 增删 `temperature_c`/`pupil_mm`
        会让它静默失效。真解是 Zero 允诺的 `physiology_scale` 兄弟键，见类 docstring。

        Args:
            physiology: 待消费的生理通道（canonical 或 legacy 形状）。

        Returns:
            失配原因串（可直接进日志）；无失配或形状判不出 → None。
        """
        looks_canonical = physiology.temperature_c is not None
        looks_legacy = physiology.temperature_c is None and physiology.pupil_mm is not None

        if looks_legacy and self.skin_conductance_max_us != 1.0:
            return (
                "来源疑似 legacy 口径（无 temperature_c、有 pupil_mm ⇒ skin_conductance 已是 "
                "[0,1]），但本 mapper 按 μS 归一（skin_conductance_max_us="
                f"{self.skin_conductance_max_us}）⇒ 系统性欠标度 "
                f"~{self.skin_conductance_max_us:g}×。"
                "接线须显式配 skin_conductance_max_us=1.0（W6）"
            )
        if looks_canonical and self.skin_conductance_max_us == 1.0:
            return (
                "来源疑似 canonical 口径（有 temperature_c ⇒ skin_conductance 为 μS），但本 mapper "
                "配了 skin_conductance_max_us=1.0 ⇒ 过标度"
                "（μS 量级读数被当 [0,1] 消费，将常态饱和）"
            )
        return None

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
