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

percentile 归一化选型依据（文献门）：
- [Mõttus 2024 (DOI:10.5772/intechopen.1007760)]
  跨被试 EDA 幅度差 10–100×，固定阈值（如 ref）鲁棒性差；
  被试自身近期幅度的滚动分位做归一化才跨被试可比（Lykken range correction 思想）。
- [Matesanz 2024 (DOI:10.3390/math12020202)]
  在线自适应归一化在生理多模态信号中优于固定尺度归一化，支撑滚动历史分位方案。
"""

from __future__ import annotations

import asyncio
import collections
import logging
import os
import threading
from typing import Any, Literal

import numpy as np

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
# 工程假设初值，集成测试接真硬件后再校。**仅用于 highpass 分支**（rate≤4Hz）。
_SCR_REF_AMPLITUDE: float = 0.6

# cvxEDA 分支（rate>4Hz）专属 ref（gap-3 续）：cvxEDA 反卷积出的 phasic 量级比 highpass
# 大 ~55×，沿用 highpass 的 0.6 会令 μa 恒饱和到 +1（丢失分级分辨力）。据 cvxEDA(rate=8)
# 合成信号 abs_mean 现场校准（standardize 后）：
#   scr=0→0.0→μa=-1；scr=2→22.7→+0.13；scr=4→28.7→+0.43；scr=6→31.7→+0.59；scr=9→37.8→+0.89
#   （ref=40 单调非饱和，除 scr=0 正确钉底 -1）。工程假设初值，接真硬件后再校。
# ⚠ 装 cvxopt 后现场实测暴露：该 cvxEDA 路径此前因环境缺 cvxopt **未被真测**，旧版对其
#   复用 0.6 会饱和；真判别 eval（test_zero_physio_real.py）现锁定其分级判别力。
_SCR_REF_AMPLITUDE_CVXEDA: float = 40.0

# RMSSD 参考上界（ms，工程假设；健康成人休息 RMSSD 通常 20-80ms，取 100ms 保守上界；
# 集成测试接真硬件后再校）。RMSSD>100ms（极度放松/迷走神经张力极高）时 inverted 钉底
# → μa=-1，方向正确（高 RMSSD = 低 arousal = 副交感优势），信息损失可接受。
_RMSSD_REF_MS: float = 100.0

# percentile 归一化：退化窗口守卫阈值（p_high - p_low < eps 时回退 linear，避免除零）
_PERCENTILE_EPS: float = 1e-6


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

    **percentile 归一化（有状态在线自适应）**：
    normalization="percentile" 时，维护逐实例的滚动 SCR 幅度历史（按 method 分桶，
    cvxEDA 与 highpass 量级差 55× 不可混）。将当前幅度定位到被试自身近期 [p_low, p_high]
    区间，实现跨被试可比的自适应归一化（Lykken range correction 思想）。

    **并发/异步**：SCR 提取（neurokit2，cvxEDA/ecg 计算重且非 GIL 释放型）在 ``sense()``
    里经 ``asyncio.to_thread`` 调度到线程池，不阻塞事件循环（对齐 audio/vision）。滚动历史的
    读改写由 ``history_lock`` 保护——同实例并发 ``collect()`` 时线程安全（append + 快照落临界区）。

    文献依据：
    - [Mõttus 2024 (DOI:10.5772/intechopen.1007760)] 跨被试 EDA 幅度差 10–100×，
      被试自身滚动分位归一化才跨被试可比。
    - [Matesanz 2024 (DOI:10.3390/math12020202)] 在线自适应归一化优于固定尺度归一化。

    Args:
        sampling_rate:        默认采样率 Hz（构造时传入；信号 dict 可覆盖）。默认 4Hz。
        normalization:        归一化策略：``"linear"``（默认）或 ``"percentile"``（有状态自适应）。
        signal_source:        async callable → dict | None；PerceptionHub 无参调 sense()
                              时由此获取信号；测试可直接向 sense(signal=...) 传 dict 绕过。
        percentile_window:    滚动历史最大长度（工程假设，接真被试数据再校）。默认 60。
        percentile_cold_start: 未达此样本数时走冷启动回退（工程假设，接真被试数据再校）。
                              默认 20（4Hz 下约需累积 20 帧感知才暖机；每帧越短暖机越慢）。
        percentile_range:     分位范围 (q_low, q_high)（工程假设，接真被试数据再校）。
                              默认 (5, 95)。
    """

    name: str = "eda/sc"

    def __init__(
        self,
        sampling_rate: int = 4,
        normalization: Literal["linear", "percentile"] = "linear",
        signal_source: Any | None = None,
        percentile_window: int = 60,
        percentile_cold_start: int = 20,
        percentile_range: tuple[int, int] = (5, 95),
    ) -> None:
        self.sampling_rate = sampling_rate
        self.normalization = normalization
        self.signal_source = signal_source
        self.percentile_window = percentile_window
        self.percentile_cold_start = percentile_cold_start
        self.percentile_range = percentile_range
        # 按 method 分桶的滚动幅度历史（逐实例状态，非全局）
        # key: "cvxEDA" | "highpass"；value: 最近 percentile_window 个 scr_amplitude 值
        self._amplitude_history: dict[str, collections.deque[float]] = {
            "cvxEDA": collections.deque(maxlen=percentile_window),
            "highpass": collections.deque(maxlen=percentile_window),
        }
        # 保护 _amplitude_history 的读改写：_process 现经 asyncio.to_thread 在线程池执行
        # （对齐 audio/vision，不阻塞事件循环），同实例并发 collect 时多线程会并发读改同一
        # deque（共享可变状态）——把 append + 快照收进临界区取一致视图（W3；详见 _process 内
        # 注释的现场核验与自由线程前瞻）。与 audio/vision 的 threading.Lock 同族（那两处防并发
        # 双载模型，此处防并发改历史）。
        self.history_lock = threading.Lock()

    def reset(self) -> None:
        """清空滚动幅度历史（**被试切换时调用方必须调用**）。

        调用后 percentile 模式回到冷启动状态，linear 模式不受影响。
        ⚠ 调用方（编排/接入层）有责任在被试切换时调用本方法或 ``PerceptionHub.reset_all()``；
        **未调用将导致新被试沿用旧被试的历史分位区间（跨被试污染）**——有状态通道的固有契约，
        Protocol 无法在编译期强制（W2，见 pitfalls「有状态感知通道被试切换须 reset」）。
        """
        with self.history_lock:
            for key in self._amplitude_history:
                self._amplitude_history[key].clear()

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

        # 阻塞处理走线程池，不堵事件循环（cvxEDA 是 cvxopt 线性规划、ecg_process 也重，
        # 均非 GIL 释放型；对齐 audio/vision 的 asyncio.to_thread，W3）。异常在工作线程内抛出，
        # 经 await 传回当前协程，仍由下方 except 兜底优雅回退。
        try:
            return await asyncio.to_thread(self._process, raw)
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

        percentile 分支：维护逐实例滚动历史，按 method 分桶（cvxEDA/highpass 量级差
        55× 不可混）。冷启动（< percentile_cold_start 样本）回退 linear，保证暖机前
        行为等价于 linear（零回归过渡）。退化窗口（p_high - p_low < eps）同样回退
        linear（避免除零/单调窗口塌缩）。

        参考：[NeuroKit2 DOI:10.3758/s13428-020-01516-y]
              [Mõttus 2024 DOI:10.5772/intechopen.1007760]
              [Matesanz 2024 DOI:10.3390/math12020202]
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
        # gap-3 工程假设：ref 随分解方法（cvxEDA phasic 量级远大于 highpass，各用各的校准 ref，
        # 否则 cvxEDA 会饱和丢分级）。均为工程假设初值，集成测试再校。
        scr_amplitude = float(phasic_df["EDA_Phasic"].abs().mean())
        ref = _SCR_REF_AMPLITUDE_CVXEDA if method == "cvxEDA" else _SCR_REF_AMPLITUDE

        if self.normalization == "linear":
            mu_a = _linear_normalize(scr_amplitude, ref)
        else:
            # percentile 有状态在线自适应归一化
            # 按 method 分桶——cvxEDA 与 highpass 量级差 55×，混桶会令分位区间跨度失控。
            # 线程安全（W3）：_process 现经 asyncio.to_thread 在线程池执行，同实例并发 collect
            # 时多线程会并发读改同一 deque（共享可变状态）。history_lock 把 append + 快照收进
            # 临界区，保证快照是一致视图；percentile 计算落锁外（快照已是独立 ndarray）。
            # ⚠ 现场核验（scratch mech_probe，CPython 3.12）：np.asarray(deque)/list(deque) 是
            # GIL 下 C 级原子拷贝，与并发 append 不互撕、也**不**触发「deque mutated during
            # iteration」（该异常仅经显式 Python 迭代 iter(deque) 触发，本处不用）——即无锁在当前
            # 解释器下也不崩。锁在此是**不赖此 GIL 偶发原子性**的显式保证：跨线程访问共享可变
            # 状态不依赖未文档化的原子性，前瞻自由线程(no-GIL)Python 与「日后改成显式迭代」。
            # 取桶引用可在锁外：_amplitude_history 的 dict 键在 __init__ 后固定、reset() 用
            # .clear() 原地清（deque 身份不变），此处只读引用不写 dict；真正的 append + 快照落锁内。
            hist = self._amplitude_history[method]
            with self.history_lock:
                hist.append(scr_amplitude)
                # deque→ndarray 一次转换（快照）：避免两次 list 分配 +「两次转换间 deque 不变」的
                # 隐式假设（W4），并把对 deque 的一致性拷贝收进锁内。
                snapshot = np.asarray(hist, dtype=float)

            if snapshot.size < self.percentile_cold_start:
                # 冷启动：样本不足，回退 linear（暖机前行为等价于 linear，零回归过渡）
                mu_a = _linear_normalize(scr_amplitude, ref)
            else:
                q_low, q_high = self.percentile_range
                # 一次 np.percentile 取双分位（用锁内取的独立快照，锁外计算）
                pcts = np.percentile(snapshot, [q_low, q_high])
                p_low, p_high = float(pcts[0]), float(pcts[1])

                if p_high - p_low < _PERCENTILE_EPS:
                    # 退化窗口守卫：p_high ≈ p_low（单调信号窗口塌缩）→ 回退 linear 避免除零
                    mu_a = _linear_normalize(scr_amplitude, ref)
                else:
                    # 自适应归一化：把当前幅度定位到被试自身近期 [p_low, p_high] 区间
                    # → 跨被试可比（Lykken range correction 思想）
                    clipped = min(max((scr_amplitude - p_low) / (p_high - p_low), 0.0), 1.0)
                    mu_a = clipped * 2.0 - 1.0

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

        # 阻塞处理走线程池，不堵事件循环（ecg_process + hrv_time 计算重；对齐 audio/vision 与
        # EdaChannel，W3）。异常在工作线程内抛出，经 await 传回，仍由下方 except 兜底回退。
        try:
            return await asyncio.to_thread(self._process, raw)
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
