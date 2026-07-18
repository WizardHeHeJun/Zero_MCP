"""生理感知通道（真接入）—— EDA/SC 与 HRV/RMSSD → ModalityPrior。

EDA/HRV 对 valence 盲，仅产 arousal 分量（μv 恒 0.0，Πv=MIN_PRECISION）；
唤醒分量 μa 由对应生理指标归一化得到。两通道各自独立，各 sense() 产一条先验（AD-3）。

设计依据（文献门纪要 notes/2026-07-16-zero-link-perception-litreview.md）：
- [Kreibig 2010 ANS in emotion (DOI:10.1016/j.biopsycho.2010.03.010)]
  EDA/HRV 主编码 arousal，对 valence 区分几无独立贡献 → μv=0.0，Πv=MIN_PRECISION（M2）。
- [NeuroKit2 (DOI:10.3758/s13428-020-01516-y)]
  cvxEDA（>4Hz）/ highpass（≤4Hz）分解 SCR；ecg_process + hrv_time 提取 RMSSD。
- gap-6：EDA 采样率差异影响分解质量，低采样率用 highpass 更稳，按硬件采样率适配。
- gap-3：SCR 量级度量采 abs().mean()（零均值信号正确度量），ref=0.6（合成信号校准初值，工程假设）。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from src.agents.models.zero_affect import ModalityPrior
from src.mcp.zero.external_priors import ModalityKind, build_recommended_prior

logger = logging.getLogger(__name__)

# EDA 采样率阈值：高于此值使用 cvxEDA，否则使用 highpass（gap-6）
_EDA_CVXEDA_RATE_HZ: int = 4

# SCR 量级度量参考值（工程假设，gap-3）。
# 度量：phasic_df["EDA_Phasic"].abs().mean()——对 nk.standardize 后零均值信号捕获
# SCR 量级最合适：有符号 .mean() ≈ 0（退化 bug），abs 均值随 SCR 数单调递增。
# ref=0.6：据 nk.eda_simulate(rate=4,highpass) 合成信号 abs_mean 量级校准：
#   scr=0 → abs_mean≈0.002 → μa≈-0.99；scr=4 → ≈0.33 → μa≈0.10；
#   scr=9 → ≈0.53 → μa≈0.77（单调，Δμa≥0.9，远超 0.3 要求）。
# 工程假设初值，集成测试接真硬件后再校。
_SCR_REF_AMPLITUDE: float = 0.6

# RMSSD 参考上界（ms，工程假设；健康成人休息 RMSSD 通常 20-80ms，取 100ms 保守上界；
# 集成测试接真硬件后再校）。RMSSD>100ms（极度放松/迷走神经张力极高）时 inverted 钉底
# → μa=-1，方向正确（高 RMSSD = 低 arousal = 副交感优势），信息损失可接受。
_RMSSD_REF_MS: float = 100.0


def _linear_normalize(value: float, ref: float) -> float:
    """线性归一化到 [-1, 1]，clip 到 [0, ref] 后映射。

    公式：clip(value / ref, 0, 1) * 2 - 1
    结果 ∈ [-1, 1]；value=0 → -1.0，value≥ref → 1.0。
    """
    ratio = min(max(value / ref, 0.0), 1.0)
    return ratio * 2.0 - 1.0


# ---------------------------------------------------------------------------
# EdaChannel
# ---------------------------------------------------------------------------


class EdaChannel:
    """EDA/SC（皮肤电）感知通道。

    从注入的 EDA 信号提取 SCR 幅度，归一化为 arousal 分量。
    valence 恒 0.0（EDA 对 valence 盲，Kreibig 2010）。

    **Protocol 兼容**：结构上满足 PerceptionChannel Protocol（无需继承）：
    - name: str 属性（"eda/sc"）。
    - async sense(signal=None) -> ModalityPrior | None（signal 有默认值=无参可调）。

    Args:
        sampling_rate:   默认采样率 Hz（构造时传入；信号 dict 可覆盖）。默认 4Hz。
        normalization:   归一化策略：``"linear"``（当前实现）或 ``"percentile"``（预留接口）。
        signal_source:   async callable → dict | None；PerceptionHub 无参调 sense()
                         时由此获取信号；测试可直接向 sense(signal=...) 传 dict 绕过。
    """

    name: str = "eda/sc"

    def __init__(
        self,
        sampling_rate: int = 4,
        normalization: Literal["linear", "percentile"] = "linear",
        signal_source: Any | None = None,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.normalization = normalization
        self.signal_source = signal_source

    async def sense(
        self,
        signal: dict[str, Any] | None = None,
    ) -> ModalityPrior | None:
        """async：从 EDA 信号提取 SCR 幅度，产出一条 ModalityPrior；无证据则返回 None。

        Args:
            signal: dict 形状 ``{'eda': ndarray, 'ecg_or_ppg': ndarray|None,
                    'sampling_rate': int|None}``。None 时使用构造注入的 signal_source。

        Returns:
            ModalityPrior(modality="eda/sc", mu=(0.0, μa), precision=(MIN,0.18)) 或 None。

        Raises:
            不抛：I/O 异常（OSError/TimeoutError/RuntimeError/ValueError/ImportError）
            均 warning+None 回退；编程错误（TypeError 等）上抛供 PerceptionHub 兜。
        """
        # 运行时读 env——感知构造后 env 变更即时生效（不在 __init__ 缓存）
        if os.getenv("ZERO_PHYSIO_CHANNEL_ENABLED", "false").lower() != "true":
            return None

        # 取信号：优先直接传入，否则走 signal_source（async callable）
        raw: dict[str, Any] | None = signal
        if raw is None and self.signal_source is not None:
            try:
                raw = await self.signal_source()
            except (OSError, TimeoutError, RuntimeError, ValueError) as exc:
                logger.warning("EdaChannel signal_source 调用失败，本轮跳过: %s", exc)
                return None

        if raw is None:
            logger.warning("EdaChannel 无可用信号（signal=None 且 signal_source=None），跳过")
            return None

        try:
            return self._process(raw)
        except ImportError as exc:
            logger.warning("EdaChannel: neurokit2 不可用，本轮跳过: %s", exc)
            return None
        except (ValueError, RuntimeError) as exc:
            logger.warning("EdaChannel 信号处理失败，本轮跳过: %s", exc)
            return None

    def _process(self, raw: dict[str, Any]) -> ModalityPrior | None:
        """延迟 import neurokit2 并执行 EDA 处理。

        neurokit2 仅在此处 import，避免模块加载时因缺包崩溃。
        采样率 >4Hz 用 cvxEDA，≤4Hz 用 highpass（gap-6，低采样率 highpass 更稳）。

        参考：[NeuroKit2 DOI:10.3758/s13428-020-01516-y]
        """
        import neurokit2 as nk  # 延迟 import（ImportError 由 sense() 捕获）

        eda: Any = raw.get("eda")
        if eda is None:
            raise ValueError("signal dict 缺少 'eda' 键")

        rate: int = int(raw.get("sampling_rate") or self.sampling_rate)

        method = "cvxEDA" if rate > _EDA_CVXEDA_RATE_HZ else "highpass"
        phasic_df = nk.eda_phasic(nk.standardize(eda), sampling_rate=rate, method=method)

        # NeuroKit2 phasic 输出列名：EDA_Phasic
        if "EDA_Phasic" not in phasic_df.columns:
            raise RuntimeError(
                f"eda_phasic 输出缺少 'EDA_Phasic' 列，实际列: {list(phasic_df.columns)}"
            )

        # abs().mean() 作 SCR 量级度量：nk.standardize 后 EDA_Phasic 零均值，
        # 有符号 .mean() ≈ 0（退化），abs 均值随 SCR 活动单调递增，是零均值信号的正确量级度量。
        # gap-3 工程假设：ref=0.6（据合成信号 abs_mean 量级校准的初值，集成测试再校）。
        scr_amplitude = float(phasic_df["EDA_Phasic"].abs().mean())

        if self.normalization == "linear":
            mu_a = _linear_normalize(scr_amplitude, _SCR_REF_AMPLITUDE)
        else:
            # percentile 归一化接口预留，当前 fallback 到 linear（gap-3）
            logger.warning("EdaChannel: normalization='percentile' 尚未实现，fallback 到 linear")
            mu_a = _linear_normalize(scr_amplitude, _SCR_REF_AMPLITUDE)

        # valence 恒 0.0：EDA 对 valence 盲（Kreibig 2010，Zero M2 也会覆写 Πv=MIN）
        mu_v = 0.0
        return build_recommended_prior(
            modality=self.name,
            mu=(mu_v, mu_a),
            kind=ModalityKind.PHYSIO,
        )


# ---------------------------------------------------------------------------
# HrvChannel
# ---------------------------------------------------------------------------


class HrvChannel:
    """HRV/RMSSD（心率变异性）感知通道。

    从注入的 ECG/PPG 信号提取 RMSSD，归一化为 arousal 分量。
    valence 恒 0.0（HRV 对 valence 盲，Kreibig 2010）。

    **Protocol 兼容**：结构上满足 PerceptionChannel Protocol（无需继承）：
    - name: str 属性（"hrv/rmssd"）。
    - async sense(signal=None) -> ModalityPrior | None（signal 有默认值=无参可调）。

    Args:
        sampling_rate:   默认采样率 Hz（构造时传入；信号 dict 可覆盖）。默认 256Hz（ECG 常见）。
        signal_source:   async callable → dict | None；PerceptionHub 无参调 sense()
                         时由此获取信号；测试可直接向 sense(signal=...) 传 dict 绕过。
    """

    name: str = "hrv/rmssd"

    def __init__(
        self,
        sampling_rate: int = 256,
        signal_source: Any | None = None,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.signal_source = signal_source

    async def sense(
        self,
        signal: dict[str, Any] | None = None,
    ) -> ModalityPrior | None:
        """async：从 ECG/PPG 信号提取 RMSSD，产出一条 ModalityPrior；无证据则返回 None。

        Args:
            signal: dict 形状 ``{'ecg_or_ppg': ndarray, 'sampling_rate': int|None}``。
                    None 时使用构造注入的 signal_source。

        Returns:
            ModalityPrior(modality="hrv/rmssd", mu=(0.0, μa), precision=(MIN,0.18)) 或 None。

        Raises:
            不抛：I/O 异常（OSError/TimeoutError/RuntimeError/ValueError/ImportError）
            均 warning+None 回退；编程错误（TypeError 等）上抛供 PerceptionHub 兜。
        """
        # 运行时读 env——感知构造后 env 变更即时生效（不在 __init__ 缓存）
        if os.getenv("ZERO_PHYSIO_CHANNEL_ENABLED", "false").lower() != "true":
            return None

        raw: dict[str, Any] | None = signal
        if raw is None and self.signal_source is not None:
            try:
                raw = await self.signal_source()
            except (OSError, TimeoutError, RuntimeError, ValueError) as exc:
                logger.warning("HrvChannel signal_source 调用失败，本轮跳过: %s", exc)
                return None

        if raw is None:
            logger.warning("HrvChannel 无可用信号（signal=None 且 signal_source=None），跳过")
            return None

        try:
            return self._process(raw)
        except ImportError as exc:
            logger.warning("HrvChannel: neurokit2 不可用，本轮跳过: %s", exc)
            return None
        except (ValueError, RuntimeError) as exc:
            logger.warning("HrvChannel 信号处理失败，本轮跳过: %s", exc)
            return None

    def _process(self, raw: dict[str, Any]) -> ModalityPrior | None:
        """延迟 import neurokit2 并执行 ECG/HRV 处理。

        pipeline：nk.ecg_process → nk.hrv_time → RMSSD → 线性归一化。
        RMSSD 越低 = 交感激活越强 = arousal 越高 → 取反后归一化。

        参考：[NeuroKit2 DOI:10.3758/s13428-020-01516-y] ·
              [Kreibig 2010 DOI:10.1016/j.biopsycho.2010.03.010]
        """
        import neurokit2 as nk  # 延迟 import

        ecg: Any = raw.get("ecg_or_ppg")
        if ecg is None:
            raise ValueError("signal dict 缺少 'ecg_or_ppg' 键")

        rate: int = int(raw.get("sampling_rate") or self.sampling_rate)

        signals, _info = nk.ecg_process(ecg, sampling_rate=rate)
        hrv_df = nk.hrv_time(signals, sampling_rate=rate, show=False)

        if "HRV_RMSSD" not in hrv_df.columns:
            raise RuntimeError(f"hrv_time 输出缺少 'HRV_RMSSD' 列，实际列: {list(hrv_df.columns)}")

        rmssd_ms = float(hrv_df["HRV_RMSSD"].iloc[0])

        # RMSSD 高 = 副交感优势 = arousal 低；取反后归一化到 [-1,1]。
        # RMSSD>_RMSSD_REF_MS（极度放松）时 inverted 钉底 → μa=-1，方向正确：
        # 高 RMSSD = 低 arousal，方向与生理语义一致（_RMSSD_REF_MS 工程假设，集成测试再校）。
        inverted = max(0.0, _RMSSD_REF_MS - rmssd_ms)
        mu_a = _linear_normalize(inverted, _RMSSD_REF_MS)

        mu_v = 0.0
        return build_recommended_prior(
            modality=self.name,
            mu=(mu_v, mu_a),
            kind=ModalityKind.PHYSIO,
        )
