"""情感 TTS 韵律映射器（ProsodyMapper 实现，蓝图第二阶段 T1）。

LinearProsodyMapper 按 prosody_scale 分支，将 ExpressionHead 中
ProsodyChannel 的三值（speech_rate/pitch/energy）映射成引擎无关的
TTS 控制参数 ProsodyParams。

量纲双方言处理（Q1 已定，canonical=normalized）：
- "normalized"：三值均 [0,1]，线性映射到各自目标范围。
- "ratio" 或 None：speech_rate/pitch 是倍率（基线 1.0），
  energy 仍为 [0,1]（两口径共用）。
"""

from __future__ import annotations

import logging
import math

from pydantic import BaseModel, ConfigDict, Field

from src.agents.models.zero_affect import ExpressionHead, ProsodyChannel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _lerp(bounds: tuple[float, float], t: float) -> float:
    """线性插值：lo + (hi-lo)*clip(t,0,1)。

    t 期望在 [0,1]（量纲契约保证）；**防御性 clamp 到 [0,1]**——Zero 回执建议（对未来非 sigmoid
    注入的 normalized 模型稳健；Zero 现役 ProsodyDecoder 末端 sigmoid 本已 ∈(0,1)、其侧同步在打
    tag 处加 bounds 断言，双保险）。见 notes/2026-07-22-zero-link-t4t5t6-*.md。
    """
    lo, hi = bounds
    t_clamped = min(max(t, 0.0), 1.0)
    return lo + (hi - lo) * t_clamped


# ---------------------------------------------------------------------------
# ProsodyParams
# ---------------------------------------------------------------------------


class ProsodyParams(BaseModel):
    """引擎无关的 TTS 韵律控制参数。

    rate_ratio     : 语速倍率，1.0 = 基线正常语速。
    pitch_semitones: 基频偏移（半音），0 = 基线；正值升调，负值降调。
    gain_db        : 音量增益（dB），0 = 基线；正值增大，负值减小。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rate_ratio: float = Field(..., description="语速倍率，1.0=基线正常语速")
    pitch_semitones: float = Field(..., description="基频偏移，半音；0=基线")
    gain_db: float = Field(..., description="音量增益，dB；0=基线")

    def to_ssml_prosody_attrs(self) -> str:
        """生成 SSML <prosody> 属性串。

        格式::

            rate="{rate_ratio*100:.0f}%" pitch="{pitch_semitones:+.1f}st" volume="{gain_db:+.1f}dB"

        示例（rate_ratio=1.0, pitch_semitones=0.0, gain_db=0.0）::

            'rate="100%" pitch="+0.0st" volume="+0.0dB"'

        ⚠ 引擎兼容：rate `%`、volume `dB`、pitch **半音 `st`** 为 Azure Speech /
        W3C SSML 1.1 合法值。**Amazon Polly 不支持 pitch 的 `st` 单位**（Polly 只接受
        x-low/…/x-high 或 `±N%`）——接 Polly 时需另出 `%` 口径（后续按引擎补 mapper）。
        """
        rate_pct = self.rate_ratio * 100.0
        return (
            f'rate="{rate_pct:.0f}%"'
            f' pitch="{self.pitch_semitones:+.1f}st"'
            f' volume="{self.gain_db:+.1f}dB"'
        )


# ---------------------------------------------------------------------------
# LinearProsodyMapper
# ---------------------------------------------------------------------------


class LinearProsodyMapper:
    """线性韵律映射器——满足 ProsodyMapper Protocol（结构化，不显式继承）。

    按 ExpressionHead.prosody_scale 分支处理量纲双方言（Q1 已定）：
    - "normalized"：三值均 [0,1]，各自线性映射到目标范围。
    - "ratio" 或 None（Zero 当前占位默认出 "ratio"；未标注为 None）：
      speech_rate/pitch 是倍率，energy 仍视为 [0,1]。

    async：对齐 ProsodyMapper Protocol（Protocol.map 是 async def），
    当前实现纯标量计算无真正的 I/O 等待，async 为预留——
    便于未来接入真实 TTS SDK（如 Azure Speech SDK async 调用）时无缝替换。
    """

    def __init__(
        self,
        *,
        rate_range: tuple[float, float] = (0.5, 1.5),
        pitch_semitone_range: float = 4.0,
        gain_db_range: tuple[float, float] = (-6.0, 6.0),
    ) -> None:
        """初始化线性映射参数。

        rate_range          : (min, max) 语速倍率范围，默认 (0.5, 1.5)。
        pitch_semitone_range: 基频偏移对称幅度（半音），默认 ±4.0 st；
                              normalized pitch=0.5 → 0 半音，0→-range，1→+range。
        gain_db_range       : (min, max) 音量增益 dB 范围，默认 (-6.0, 6.0)。
        """
        self.rate_range = rate_range
        self.pitch_semitone_range = pitch_semitone_range
        self.gain_db_range = gain_db_range

    async def map(self, channel: ExpressionHead) -> ProsodyParams:
        """async：将 ExpressionHead 韵律通道映射为 ProsodyParams。

        async 为对齐 ProsodyMapper Protocol 预留，当前纯标量计算，无实际 await。

        分支逻辑（按 channel.prosody_scale，Q1）：
        - "normalized"：三值均 [0,1]，各自线性映射到目标范围。
        - "ratio" / None（Zero 当前占位默认）/ 未知值（Literal 已限定，防御性回退）：
          speech_rate/pitch 是倍率、energy 仍 [0,1]——统一走 `_map_ratio`。
        """
        prosody = channel.prosody
        scale = channel.prosody_scale

        if scale == "normalized":
            rate_ratio = _lerp(self.rate_range, prosody.speech_rate)
            pitch_semitones = _lerp(
                (-self.pitch_semitone_range, self.pitch_semitone_range),
                prosody.pitch,
            )
            gain_db = _lerp(self.gain_db_range, prosody.energy)
        else:
            if scale not in ("ratio", None):
                # Literal["ratio","normalized"]|None 不应出现其他值，防御性回退
                logger.warning("未知 prosody_scale=%r，回退到 ratio 分支处理", scale)
            rate_ratio, pitch_semitones, gain_db = self._map_ratio(prosody)

        return ProsodyParams(
            rate_ratio=rate_ratio,
            pitch_semitones=pitch_semitones,
            gain_db=gain_db,
        )

    def _map_ratio(self, prosody: ProsodyChannel) -> tuple[float, float, float]:
        """ratio / 未标注口径映射：speech_rate/pitch 是倍率（基线 1.0）、energy [0,1]。

        返回 (rate_ratio, pitch_semitones, gain_db)。ratio 分支与未知回退共用此逻辑，
        避免两处并联维护而分叉（review WARN2）。pitch≤0 时取 1e-6 兜底避免 log2 域错误并 warning。
        """
        rate_ratio = prosody.speech_rate
        raw_pitch = prosody.pitch
        if raw_pitch <= 0.0:
            logger.warning("prosody.pitch=%r 不可取 log2，兜底取 1e-6", raw_pitch)
            raw_pitch = 1e-6
        pitch_semitones = 12.0 * math.log2(raw_pitch)
        gain_db = _lerp(self.gain_db_range, prosody.energy)
        return rate_ratio, pitch_semitones, gain_db
