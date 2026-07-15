"""ARKit blendshape FACS 映射器（FacsMapper 实现，蓝图第二阶段 F1）。

ArkitFacsMapper 将 ExpressionHead.facs_au（13 AU 子集，值域 [0,1]）
映射为 ARKit 52 blendshape 系数 dict[str, float]，满足 FacsMapper Protocol。

设计要点：
- AU_TO_ARKIT 为工程默认对照表，可通过子类或调参覆盖；
- "intensity" 是全局强度标量，不在表里——作乘子 gain 处理（apply_intensity=True 时）；
- 输出只含被驱动的 blendshape（未驱动的由消费方默认静息 0），便于稀疏传输；
- async 为对齐 FacsMapper Protocol 预留，当前实现纯标量计算无实际 I/O 等待。
"""

from __future__ import annotations

import logging

from src.agents.models.zero_affect import ExpressionHead

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AU → ARKit blendshape 对照表（工程默认，可按目标模型调）
# ---------------------------------------------------------------------------

AU_TO_ARKIT: dict[str, tuple[str, ...]] = {
    # AU01 – 内眉提升（Inner Brow Raise）
    # 对应 ARKit browInnerUp（单一对称控制）
    "AU01": ("browInnerUp",),
    # AU02 – 外眉提升（Outer Brow Raise）
    # 对应 ARKit browOuterUpLeft / browOuterUpRight（左右各一）
    "AU02": ("browOuterUpLeft", "browOuterUpRight"),
    # AU04 – 皱眉（Brow Lowerer）
    # 对应 ARKit browDownLeft / browDownRight（左右各一）
    "AU04": ("browDownLeft", "browDownRight"),
    # AU05 – 上眼睑提升（Upper Lid Raiser）
    # 对应 ARKit eyeWideLeft / eyeWideRight（左右各一）
    "AU05": ("eyeWideLeft", "eyeWideRight"),
    # AU06 – 颧肌提升 / 微笑眯眼（Cheek Raiser）
    # 对应 ARKit cheekSquintLeft / cheekSquintRight（左右各一）
    "AU06": ("cheekSquintLeft", "cheekSquintRight"),
    # AU07 – 眼睑收紧（Lid Tightener）
    # 对应 ARKit eyeSquintLeft / eyeSquintRight（左右各一）
    "AU07": ("eyeSquintLeft", "eyeSquintRight"),
    # AU12 – 嘴角上拉（Lip Corner Puller，微笑）
    # 对应 ARKit mouthSmileLeft / mouthSmileRight（左右各一）
    "AU12": ("mouthSmileLeft", "mouthSmileRight"),
    # AU15 – 嘴角下压（Lip Corner Depressor）
    # 对应 ARKit mouthFrownLeft / mouthFrownRight（左右各一）
    "AU15": ("mouthFrownLeft", "mouthFrownRight"),
    # AU17 – 下唇上推（Chin Raiser）
    # 对应 ARKit mouthShrugLower（单一对称控制）
    "AU17": ("mouthShrugLower",),
    # AU20 – 嘴唇横向伸展（Lip Stretcher，恐惧唇型）
    # 对应 ARKit mouthStretchLeft / mouthStretchRight（左右各一）
    "AU20": ("mouthStretchLeft", "mouthStretchRight"),
    # AU23 – 嘴唇收紧（Lip Tightener，压力/愤怒）
    # 对应 ARKit mouthPressLeft / mouthPressRight（左右各一）
    "AU23": ("mouthPressLeft", "mouthPressRight"),
    # AU26 – 下颌下落（Jaw Drop，开口度）
    # 对应 ARKit jawOpen（单一对称控制）
    "AU26": ("jawOpen",),
}
"""AU → ARKit blendshape 名对照表（值域 [0,1]，可按目标模型调）。

⚠ **工程假设（engineering assumption）**：AU→ARKit 对照基于工程经验（FACS AU 语义
↔ ARKit `ARFaceAnchor.BlendShapeLocation` 标准 blendshape 名），**非文献推导**；
消费方可按目标模型/绑定覆盖本表。（遵 agent-framework-rules：无据选择显式标工程假设。）

每个 AU 分发到 1–2 个对称 L/R blendshape（或单控制点）。
"intensity" 不在表里——它是 ExpressionHead.facs_au 中的全局强度标量，
由 ArkitFacsMapper 单独作乘子处理（apply_intensity=True 时）。

ARKit blendshape 名遵循苹果 ARKit 52 标准 camelCase 命名（ARFaceAnchor.BlendShapeLocation）。
"""


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    """将 x 截断到 [0.0, 1.0]。"""
    return max(0.0, min(1.0, x))


# ---------------------------------------------------------------------------
# ArkitFacsMapper
# ---------------------------------------------------------------------------


class ArkitFacsMapper:
    """ARKit blendshape FACS 映射器——满足 FacsMapper Protocol（结构化，不显式继承）。

    将 ExpressionHead.facs_au（13 AU 子集，值域 [0,1]）映射为
    ARKit 52 blendshape 系数 dict[str, float]。

    映射规则：
    1. 若 apply_intensity=True，取 facs_au.get("intensity", 1.0) 作全局增益 gain。
    2. 遍历 AU_TO_ARKIT：对每个在 facs_au 中出现的 AU，
       coeff = clamp([0,1], facs_au[au] * gain)，
       写入该 AU 对应的每个 blendshape 名。
    3. 输出只含被驱动的 blendshape——未驱动项由消费方默认静息 0。

    async 为对齐 FacsMapper Protocol 预留，当前实现纯标量计算无实际 I/O 等待。

    Args:
        apply_intensity: 是否用 facs_au["intensity"] 作全局增益乘子，默认 True。
    """

    def __init__(self, *, apply_intensity: bool = True) -> None:
        self.apply_intensity = apply_intensity

    async def map(self, channel: ExpressionHead) -> dict[str, float]:
        """async：将 FACS 通道映射为 ARKit blendshape 系数 dict。

        async 为对齐 FacsMapper Protocol 预留，当前纯标量计算，无实际 await。

        输出只含被驱动的 blendshape（未驱动的 blendshape 由消费方默认静息 0），
        便于稀疏传输与增量驱动。

        Args:
            channel: ExpressionHead，其 facs_au 键 ⊆ FACS_KEYS_EXT（13 AU + intensity）。

        Returns:
            dict[str, float]——ARKit blendshape 名 → 系数（值域 [0,1]）。
        """
        facs_au = channel.facs_au
        gain: float = facs_au.get("intensity", 1.0) if self.apply_intensity else 1.0

        result: dict[str, float] = {}
        for au, blendshapes in AU_TO_ARKIT.items():
            if au not in facs_au:
                # 占位路径只出象限子集，缺的 AU 跳过，不驱动对应 blendshape
                continue
            coeff = _clamp01(facs_au[au] * gain)
            for bs in blendshapes:
                result[bs] = coeff

        logger.debug(
            "ArkitFacsMapper.map: %d AU → %d blendshape(s), gain=%.3f",
            len([au for au in AU_TO_ARKIT if au in facs_au]),
            len(result),
            gain,
        )
        return result
