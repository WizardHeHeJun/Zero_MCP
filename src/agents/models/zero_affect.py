"""Zero↔MCP 边界契约数据模型（蓝图 §4 T1）。

契约的**唯一真相**——MCP 层 `src/mcp/zero/` 与 Agent 层共同 import 此模块，
不得在其他地方重复定义这些数据形状。

数据来源均经现场核验（2026-07-14 首核 / 2026-07-29 R7 复核）。口径：**Zero 仓内相对路径
+ `::` + 符号名**，跨仓行号一律不写（行号一次编辑即失效且腐烂不驱红；符号名是契约、
路径只是提示，Zero agents 层仍在迁移）：
- FACS 常量：Zero `src/agents/models/facs_decoder.py::{FACS_KEYS, FACS_KEYS_EXT}`
- coping AU 子集：Zero `src/agents/models/composite.py::_COPING_DRIVEN_AUS`
- text_label 枚举：Zero `src/agents/affect_math.py::text_label`
- expression 形状：Zero `src/orchestration/runner.py::_state_to_entry` 的 "expression" 键
  （即 `ConversationSession.step` 返回体的该子树）
- appraise_text 签名：Zero `src/agents/language.py::ConversationModel.appraise_text`
- streams 形状：Zero `src/agents/affect_core.py` 的 external streams 组装段
"""

from __future__ import annotations

import logging
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 规范常量（来自 Zero `src/agents/models/facs_decoder.py` 与 `.../composite.py` 的同名符号）
# ---------------------------------------------------------------------------

FACS_KEYS: list[str] = ["AU04", "AU06", "AU12", "AU15", "intensity"]
"""旧 5 维 FACS 键集（Zero `src/agents/models/facs_decoder.py::FACS_KEYS`）。"""

FACS_KEYS_EXT: list[str] = [
    "AU01",
    "AU02",
    "AU04",
    "AU05",
    "AU06",
    "AU07",
    "AU12",
    "AU15",
    "AU17",
    "AU20",
    "AU23",
    "AU26",
    "intensity",
]
"""扩展 13 维 FACS 键集（Zero `src/agents/models/facs_decoder.py::FACS_KEYS_EXT`，
议会任务 D 起含 AU17/AU26 通用 AU）。"""

COPING_DRIVEN_AUS: tuple[str, ...] = ("AU23", "AU01", "AU02", "AU20")
"""coping_potential 驱动的 AU 子集（Zero `src/agents/models/composite.py::_COPING_DRIVEN_AUS`）。

⚠ 归属订正（2026-07-29 跨仓现场核验）：此前注为 facs_decoder，实际该元组定义在 composite.py，
由 `CompositeChannelDecoder.predict_channels_coping` 的 C2 residual 叠加消费。
"""

TEXT_LABELS: frozenset[str] = frozenset({"excited", "content", "angry", "sad"})
"""text_label 的合法枚举集（Zero `src/agents/affect_math.py::text_label` 的四象限返回值）。

⚠ 归属订正（2026-07-29 跨仓现场核验）：此前注为「expression.py 推断 + composite.py」，
但真正产 label 的是 affect_math 的 `text_label`（v≥0 → excited/content，v<0 → angry/sad，
分界 a≥0.33）；composite 只是转调它。
"""

# 用于 facs_au 键校验的全集（FACS_KEYS ∪ FACS_KEYS_EXT）
_FACS_VALID_KEYS: frozenset[str] = frozenset(FACS_KEYS) | frozenset(FACS_KEYS_EXT)


# ---------------------------------------------------------------------------
# 感知输入方向（MCP → Zero）
# ---------------------------------------------------------------------------


class AffectStimulus(BaseModel):
    """喂给 Zero 的单条情感刺激（MCP → Zero 方向）。

    对应 ConversationModel.appraise_text() 产出 + coping 独立通道。
    - valence/arousal: 各维 [-1,1]，由 Zero
      `src/agents/language.py::ConversationModel.appraise_text` 返回。
    - coping_potential: 独立通道，默认 None；非 None 时同样 [-1,1]。
      **Q4 已定（Zero 回传 2026-07-14）**：正式入口 = Zero
      `src/orchestration/state.py::Stimulus.control_appraisal`（Smith & Ellsworth 1985
      control 维，与 goal_congruence 正交）——
      本字段接线时**映射到 `Stimulus.control_appraisal`**，**不要**走
      `state_overrides={"coping_potential_state":...}`（enabled 时被 Zero
      `src/agents/appraisal.py::AppraisalAgent` 每轮从 stim.control_appraisal 覆写）。需 Zero 侧开
      `coping_potential_enabled` 门控。coping 决定 (-v,+a) 象限愤怒↔恐惧判别性 AU。
    """

    model_config = ConfigDict(extra="forbid")

    valence: float = Field(..., ge=-1.0, le=1.0)
    arousal: float = Field(..., ge=-1.0, le=1.0)
    coping_potential: float | None = Field(default=None, ge=-1.0, le=1.0)


class ModalityPrior(BaseModel):
    """单模态的低精度先验（MCP → Zero 内核 streams 注入一条）。

    对应 Zero affect_core.py 里 streams 变量的形状（按符号名锚定：其行号已多次漂移）：
    streams = list[(name: str, (μ_v, μ_a), (Π_v, Π_a))]

    精度 (Π_v, Π_a) 必须 > 0（高斯精度，精度=0 无意义）。
    注意：PerceptionHub 禁止均值融合——各先验独立保留由内核竞争融合（AD-3）。

    as_stream() 输出对齐 Zero affect_core streams 形状（三元组）——
    ⚠ `coping` **不在**该三元组内（external_prior schema v1 不传输，见该字段 description）。
    """

    # frozen：构造后不可变。同仓兄弟模型（mappers/prosody.py::ProsodyParams、
    # mappers/physiology.py）本就是 frozen，本类此前是唯一漏锁的——实测 `prior.mu = (5.0, -9.0)`
    # 静默生效并一路穿过 build_external_priors_override 进入发往 Zero 的载荷。
    # ⚠ frozen 只堵「构造后赋值」，堵不住 model_construct/model_copy/鸭子类型伪造，
    # 故**不能替代**出境侧的 M7 守卫（external_priors.py）。两道都要。
    model_config = ConfigDict(extra="forbid", frozen=True)

    modality: str
    mu: tuple[float, float] = Field(
        ...,
        description="(μ_v, μ_a)，各维 [-1,1]",
    )
    precision: tuple[float, float] = Field(
        ...,
        description="(Π_v, Π_a)，各维 > 0 且有限",
    )
    coping: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description=(
            "可选 coping 分量。⚠ **external_prior schema v1 不传输本字段**：as_stream() 只返 "
            "(name, μ, Π) 三元组、Zero 侧 ExternalPrior 无对应槽位 ⇒ 写进来会被**静默丢弃**。"
            "保留字段是为兼容 build_recommended_prior(coping=...) 这一既有公开入口；真要送达"
            "须先与 Zero 商定 schema v2 的三元组扩展形状（版本号由跨仓协议 "
            "EXTERNAL_PRIOR_SCHEMA_VERSION 承载，本仓不单方面承诺 bump 时点）。"
            "丢弃面由 test_as_stream_drops_exactly_coping 锁定。"
        ),
    )

    @model_validator(mode="after")
    def _validate_ranges(self) -> ModalityPrior:
        v, a = self.mu
        if not (-1.0 <= v <= 1.0 and -1.0 <= a <= 1.0):
            raise ValueError("mu 各维必须在 [-1, 1] 内")
        pv, pa = self.precision
        # ⚠ 必须用 isfinite 显式判 NaN：`pv <= 0.0` 对 NaN **恒 False**，NaN 精度会静默通过。
        # 本条是 MCP 侧**单边兜底**，**不依赖 Zero 是否自带有限性校验**：其 M3 的
        # `pi_v <= 0.0` / `pi_v > cap` 两条判据同为比较式（对 NaN 恒 False），M7 又只守 μ
        # 不守 Π；Zero 是否在 M3 前置 isfinite 是**会随其提交/回退变动的运行时事实**，
        # 把本仓判别力挂在对方未落地的编辑上是跨仓硬教训（cec7fe3 同族）。
        # 对比：越界 μ 至少会被 Zero M7 响亮 raise；NaN 精度则更隐蔽——一旦漏过就直接进
        # Zero 融合数学产出 NaN 后验。故这一条无论对侧如何都由我方守住。
        if not (math.isfinite(pv) and math.isfinite(pa)):
            raise ValueError("precision 各维必须为有限值（NaN/inf 会静默污染 Zero 融合后验）")
        if pv <= 0.0 or pa <= 0.0:
            raise ValueError("precision 各维必须 > 0")
        return self

    def as_stream(self) -> tuple[str, tuple[float, float], tuple[float, float]]:
        """转为 Zero affect_core streams 单条形状：(name, (μ_v,μ_a), (Π_v,Π_a))。

        **只携带 modality / mu / precision 三项**；`coping` 在此丢弃（schema v1 无槽位）。
        丢弃面是被守卫锁定的既定事实，见 tests/agents/test_zero_affect_models.py::
        test_as_stream_drops_exactly_coping。
        """
        return (self.modality, self.mu, self.precision)


# ---------------------------------------------------------------------------
# 表达输出方向（Zero → MCP）的通道子模型
# ---------------------------------------------------------------------------


class ProsodyChannel(BaseModel):
    """韵律通道（Zero → MCP 方向），保持纯 3 值（Zero 侧刻意不塞 prosody_scale）。

    量纲双方言（Q1 已定 2026-07-14，canonical=normalized [0,1]）由**兄弟键**
    `prosody_scale` 标注（在 ExpressionHead / ExpressionBundle 上，非本通道内）：
    - "ratio"：占位路径，speech_rate∈[0.5,1.5]/pitch∈[0.7,1.3] 倍率（基线 1.0）；
    - "normalized"：真模型，三值归一 [0,1]。
    本通道只做 ≥0 sanity，不硬卡上界（收窄由 ExpressionHead 依 prosody_scale 承担）。
    """

    model_config = ConfigDict(extra="forbid")

    speech_rate: float = Field(..., ge=0.0)
    pitch: float = Field(..., ge=0.0)
    energy: float = Field(..., ge=0.0)


class PhysiologyChannel(BaseModel):
    """生理通道（Zero → MCP 方向）。

    **canonical = WESAD 真 physiology_decoder 输出口径**（zero-link physiology 对称接线，
    2026-07-23 拍板采 WESAD 真信号，见 notes/2026-07-23-zero-link-physiology-*）：
    - heart_rate_bpm:   心率 bpm，参考 [50, 120]（decoder 反归一化）。
    - skin_conductance: 皮肤电导 **μS**（微西门子物理单位，非 [0,1]），参考 [0, 20]（EDA）。
    - temperature_c:    皮肤温度 °C，参考 [30, 40]（decoder 反归一化）。

    ⚠ **保超集，不收窄（2026-07-23 定论，非过渡态）**：`temperature_c` 与 `pupil_mm` 均**可选**
    （默认 None），两种形状都解析。Zero 侧迁移**已落地**（真 decoder + `ZERO_PHYSIOLOGY_CANONICAL_
    PLACEHOLDER` 门），但 canonical 的充要条件是「真模型 **或** gate on」，而两仓独立部署、该门默认
    关 → 本仓无法保证所连 Zero 实例满足条件，收到 legacy `{hr, sc, pupil_mm}` 仍须不报错。
    **故「收窄 temperature_c 为必填 + 删 pupil_mm」不是待办**，仅在部署侧能保证充要条件时才可议；
    依据见 notes/2026-07-23-zero-link-physiology-consume-gate-landed.md §1。
    契约不硬卡数值范围，消费方（mapper）按实际引擎范围归一。
    """

    model_config = ConfigDict(extra="forbid")

    heart_rate_bpm: float
    skin_conductance: float
    temperature_c: float | None = None
    pupil_mm: float | None = None


class ExpressionHead(BaseModel):
    """表达头（spontaneous 或 voluntary 一条，Zero step() expression 的子结构）。

    facs_au 键校验：键 ⊆ FACS_KEYS_EXT 全集（FACS_KEYS ∪ FACS_KEYS_EXT），
    值 ∈ [0, 1]。AD-4：不要求全集——占位路径只出象限相关子集（3/9 键）。

    prosody_scale（Q1，Zero 回传 2026-07-14）：韵律量纲**兄弟键**，与 prosody 同级
    （Zero 刻意不塞进 prosody 子 dict，见 `src/agents/affect_math.py::decode_channels`
    输出的 "prosody_scale" 兄弟键）——"ratio"=倍率占位、
    "normalized"=归一真模型；缺省 None（decoder 未标注量纲，如 mock，additive 零回归）。
    当 "normalized" 时校验 prosody 三值收窄到 [0,1]。
    """

    model_config = ConfigDict(extra="forbid")

    facs_au: dict[str, float]
    text_label: str
    physiology: PhysiologyChannel
    prosody: ProsodyChannel
    prosody_scale: Literal["ratio", "normalized"] | None = None

    @model_validator(mode="after")
    def _validate(self) -> ExpressionHead:
        unknown = set(self.facs_au) - _FACS_VALID_KEYS
        if unknown:
            raise ValueError(f"facs_au 包含未知 AU 键: {unknown}")
        out_of_range = {k: v for k, v in self.facs_au.items() if not (0.0 <= v <= 1.0)}
        if out_of_range:
            raise ValueError(f"facs_au 值超出 [0,1]: {out_of_range}")
        if self.text_label not in TEXT_LABELS:
            raise ValueError(f"text_label {self.text_label!r} 不在 TEXT_LABELS 内")
        if self.prosody_scale == "normalized":
            for name, val in (
                ("speech_rate", self.prosody.speech_rate),
                ("pitch", self.prosody.pitch),
                ("energy", self.prosody.energy),
            ):
                if not (0.0 <= val <= 1.0):
                    raise ValueError(f"prosody_scale=normalized 下 prosody.{name}={val} 超出 [0,1]")
        return self


class LanguageOutput(BaseModel):
    """语言层输出（仅在语言层开启时出现，ExpressionBundle.language 可为 None）。

    对应 Zero `src/agents/language.py::LanguageDraft`。
    affect: (v, a) 二元组，JSON 化后为 list，pydantic 自动强转（AD 兼容性）。

    ⚠ 内容订正（2026-07-29 跨仓现场核验）：Zero 现行 `LanguageDraft` **只有 `text` 与
    `affect`**，`iters` / `consistency` 已从其 dataclass 上移除。本模型**宽松保留**这两个
    带默认值的可选字段——保超集不收窄：删属收窄，若某条路径/旧版 Zero 仍发这两个键，
    `extra="forbid"` 会让整个 `ExpressionBundle.from_step_output` 解析炸掉。
    故它们是**本仓兼容字段，不构成 Zero 侧契约要求**。
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    affect: tuple[float, float] | None = None
    iters: int = 0
    consistency: float | None = None


# ---------------------------------------------------------------------------
# 顶层 Bundle
# ---------------------------------------------------------------------------


class ExpressionBundle(BaseModel):
    """Zero session.step() 返回 expression 的完整结构化表示。

    来自 Zero `src/orchestration/runner.py::_state_to_entry` 的 "expression" 键
    （即 `ConversationSession.step` 返回体的该子树）。

    valence_arousal / ExpressionHead 内部的 tuple 字段均兼容 JSON 化后的 list 输入
    （Zero 内部是 tuple，过 MCP/JSON 边界变 list）——pydantic v2 默认支持 list→tuple 强转。

    prosody_scale（Q1）：Zero 在 expression 顶层**提升**一份 prosody 量纲标记
    （`expression["prosody_scale"] = spontaneous["prosody_scale"]`，见 Zero
    `src/agents/expression.py::ExpressionAgent`——注意路径已从 `src/expression.py` 迁到
    `src/agents/expression.py`），
    供 MCP TTS mapper 单点读；两头共用同一无状态 decoder 故量纲一致。缺省 None
    （decoder 未标注量纲时不挂键，additive 零回归）。各头内也各带一份同名兄弟键。
    """

    model_config = ConfigDict(extra="forbid")

    valence_arousal: tuple[float, float]
    spontaneous: ExpressionHead
    voluntary: ExpressionHead
    prosody_scale: Literal["ratio", "normalized"] | None = None
    language: LanguageOutput | None = None

    @classmethod
    def from_step_output(cls, step_out: dict[str, Any]) -> ExpressionBundle:
        """从 Zero session.step() 返回 dict 解析 ExpressionBundle。

        step_out 可以是：
        1. step() 完整返回 dict（含 "expression" 键，见 Zero
           `src/orchestration/runner.py::_state_to_entry`）→ 取 expression 子树。
        2. 直接传入 expression 子 dict（含 "valence_arousal"/"spontaneous"/"voluntary"）。

        两种形态都兼容；extra 键（如外层 trace 等）在挑取 expression 子树时自然隔离，
        不触发 extra="forbid" 校验。
        """
        if "expression" in step_out:
            data: dict[str, Any] = step_out["expression"]
        else:
            data = step_out
        return cls.model_validate(data)
