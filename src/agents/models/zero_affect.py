"""Zero↔MCP 边界契约数据模型（蓝图 §4 T1）。

契约的**唯一真相**——MCP 层 `src/mcp/zero/` 与 Agent 层共同 import 此模块，
不得在其他地方重复定义这些数据形状。

数据来源均经现场核验（2026-07-14 对 D:\\Zero 源码的双视角对抗核验）：
- 常量：D:\\Zero\\src\\models\\facs_decoder.py:16-42 · composite.py:17
- expression 形状：D:\\Zero\\src\\orchestration\\runner.py:174,491-516
- appraise_text 签名：D:\\Zero\\src\\orchestration\\chat_driver.py:184,313-320
- streams 形状：D:\\Zero\\src\\affect_core.py:77-95
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 规范常量（来自 D:\Zero\src\models\facs_decoder.py:16-42 · composite.py:17）
# ---------------------------------------------------------------------------

FACS_KEYS: list[str] = ["AU04", "AU06", "AU12", "AU15", "intensity"]
"""旧 5 维 FACS 键集（facs_decoder.py:16-20）。"""

FACS_KEYS_EXT: list[str] = [
    "AU01",
    "AU02",
    "AU04",
    "AU05",
    "AU06",
    "AU07",
    "AU12",
    "AU15",
    "AU20",
    "AU23",
    "intensity",
]
"""扩展 11 维 FACS 键集（facs_decoder.py:26-42）。"""

COPING_DRIVEN_AUS: tuple[str, ...] = ("AU23", "AU01", "AU02", "AU20")
"""coping_potential 驱动的 AU 子集（facs_decoder.py:45）。"""

TEXT_LABELS: frozenset[str] = frozenset({"excited", "content", "angry", "sad"})
"""text_label 的合法枚举集（expression.py 推断 + composite.py:17）。"""

# 用于 facs_au 键校验的全集（FACS_KEYS ∪ FACS_KEYS_EXT）
_FACS_VALID_KEYS: frozenset[str] = frozenset(FACS_KEYS) | frozenset(FACS_KEYS_EXT)


# ---------------------------------------------------------------------------
# 感知输入方向（MCP → Zero）
# ---------------------------------------------------------------------------


class AffectStimulus(BaseModel):
    """喂给 Zero 的单条情感刺激（MCP → Zero 方向）。

    对应 ConversationModel.appraise_text() 产出 + coping 独立通道。
    - valence/arousal: 各维 [-1,1]，由 appraise_text 返回（language.py:85）。
    - coping_potential: 独立通道，默认 None（门控 coping_potential_enabled=False，
      orchestration/state.py:63）；非 None 时同样 [-1,1]。
    """

    model_config = ConfigDict(extra="forbid")

    valence: float = Field(..., ge=-1.0, le=1.0)
    arousal: float = Field(..., ge=-1.0, le=1.0)
    coping_potential: float | None = Field(default=None, ge=-1.0, le=1.0)


class ModalityPrior(BaseModel):
    """单模态的低精度先验（MCP → Zero 内核 streams 注入一条）。

    对应 affect_core.py:77-95 的 streams 形状：
    streams = list[(name: str, (μ_v, μ_a), (Π_v, Π_a))]

    精度 (Π_v, Π_a) 必须 > 0（高斯精度，精度=0 无意义）。
    注意：PerceptionHub 禁止均值融合——各先验独立保留由内核竞争融合（AD-3）。

    as_stream() 输出对齐 Zero affect_core streams 形状（三元组）。
    """

    model_config = ConfigDict(extra="forbid")

    modality: str
    mu: tuple[float, float] = Field(
        ...,
        description="(μ_v, μ_a)，各维 [-1,1]",
    )
    precision: tuple[float, float] = Field(
        ...,
        description="(Π_v, Π_a)，各维 > 0",
    )
    coping: float | None = Field(default=None, ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def _validate_ranges(self) -> ModalityPrior:
        v, a = self.mu
        if not (-1.0 <= v <= 1.0 and -1.0 <= a <= 1.0):
            raise ValueError("mu 各维必须在 [-1, 1] 内")
        pv, pa = self.precision
        if pv <= 0.0 or pa <= 0.0:
            raise ValueError("precision 各维必须 > 0")
        return self

    def as_stream(self) -> tuple[str, tuple[float, float], tuple[float, float]]:
        """转为 Zero affect_core streams 单条形状：(name, (μ_v,μ_a), (Π_v,Π_a))。"""
        return (self.modality, self.mu, self.precision)


# ---------------------------------------------------------------------------
# 表达输出方向（Zero → MCP）的通道子模型
# ---------------------------------------------------------------------------


class ProsodyChannel(BaseModel):
    """韵律通道（Zero → MCP 方向的占位/真实口径双方言）。

    AD-5 量纲双方言说明：
    - 占位路径（placeholder decoder）：speech_rate ∈ [0.5, 1.5]（基线 1.0），
      pitch ∈ [0.7, 1.3]（基线 1.0），为倍率口径。
    - 真实语音模型：三值归一 [0, 1]。
    待对齐项：与 Zero 窗口 Q1 确认统一量纲后收窄校验上界。
    当前契约只做 sanity ≥ 0，不硬卡上界，兼容两种方言。
    """

    model_config = ConfigDict(extra="forbid")

    speech_rate: float = Field(..., ge=0.0)
    pitch: float = Field(..., ge=0.0)
    energy: float = Field(..., ge=0.0)


class PhysiologyChannel(BaseModel):
    """生理通道（Zero → MCP 方向）。

    参考值范围（expression 形状 §2）：
    - heart_rate_bpm: [70, 110]
    - skin_conductance: [0, 1]
    - pupil_mm: [3, 5]
    当前契约不硬卡，消费方按实际引擎范围自行处理。
    """

    model_config = ConfigDict(extra="forbid")

    heart_rate_bpm: float
    skin_conductance: float
    pupil_mm: float


class ExpressionHead(BaseModel):
    """表达头（spontaneous 或 voluntary 一条，Zero step() expression 的子结构）。

    facs_au 键校验：键 ⊆ FACS_KEYS_EXT 全集（FACS_KEYS ∪ FACS_KEYS_EXT），
    值 ∈ [0, 1]。AD-4：不要求全集——占位路径只出象限相关子集（3/9 键）。
    """

    model_config = ConfigDict(extra="forbid")

    facs_au: dict[str, float]
    text_label: str
    physiology: PhysiologyChannel
    prosody: ProsodyChannel

    @model_validator(mode="after")
    def _validate_facs(self) -> ExpressionHead:
        unknown = set(self.facs_au) - _FACS_VALID_KEYS
        if unknown:
            raise ValueError(f"facs_au 包含未知 AU 键: {unknown}")
        out_of_range = {k: v for k, v in self.facs_au.items() if not (0.0 <= v <= 1.0)}
        if out_of_range:
            raise ValueError(f"facs_au 值超出 [0,1]: {out_of_range}")
        if self.text_label not in TEXT_LABELS:
            raise ValueError(f"text_label {self.text_label!r} 不在 TEXT_LABELS 内")
        return self


class LanguageOutput(BaseModel):
    """语言层输出（仅在语言层开启时出现，ExpressionBundle.language 可为 None）。

    来自 D:\\Zero\\src\\agents\\language.py:58-85。
    affect: (v, a) 二元组，JSON 化后为 list，pydantic 自动强转（AD 兼容性）。
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

    来自 D:\\Zero\\src\\orchestration\\runner.py:174,491-516。

    valence_arousal / ExpressionHead 内部的 tuple 字段均兼容 JSON 化后的 list 输入
    （Zero 内部是 tuple，过 MCP/JSON 边界变 list）——pydantic v2 默认支持 list→tuple 强转。
    """

    model_config = ConfigDict(extra="forbid")

    valence_arousal: tuple[float, float]
    spontaneous: ExpressionHead
    voluntary: ExpressionHead
    language: LanguageOutput | None = None

    @classmethod
    def from_step_output(cls, step_out: dict[str, Any]) -> ExpressionBundle:
        """从 Zero session.step() 返回 dict 解析 ExpressionBundle。

        step_out 可以是：
        1. step() 完整返回 dict（含 "expression" 键，runner.py:174）→ 取 expression 子树。
        2. 直接传入 expression 子 dict（含 "valence_arousal"/"spontaneous"/"voluntary"）。

        两种形态都兼容；extra 键（如外层 trace 等）在挑取 expression 子树时自然隔离，
        不触发 extra="forbid" 校验。
        """
        if "expression" in step_out:
            data: dict[str, Any] = step_out["expression"]
        else:
            data = step_out
        return cls.model_validate(data)
