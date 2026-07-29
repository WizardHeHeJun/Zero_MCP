"""生理感知通道（真接入）—— EDA/SC 与 HRV/RMSSD → ModalityPrior。

EDA/HRV 对 valence 盲，仅产 arousal 分量（μv 恒 0.0，Πv=MIN_PRECISION）；
唤醒分量 μa 由对应生理指标归一化得到。两通道各自独立，各 sense() 产一条先验（AD-3）。

设计依据（文献门纪要 notes/2026-07-16-zero-link-perception-litreview.md）：
- [Kreibig 2010 ANS in emotion (DOI:10.1016/j.biopsycho.2010.03.010)]
  EDA/HRV 主编码 arousal，对 valence 区分几无独立贡献 → μv=0.0，Πv=MIN_PRECISION（M2）。
- [NeuroKit2 (DOI:10.3758/s13428-020-01516-y)] HrvChannel 用其 ecg_process + hrv_time
  提取 RMSSD。**EdaChannel 不依赖 neurokit2**（不做 phasic 分解，理由见其 `_process`）。
- [Mõttus 2024 (DOI:10.5772/intechopen.1007760)] 跨被试 EDA 绝对水平差 10–100×，固定阈值
  鲁棒性差 → EdaChannel 以**被试自身近期基线**为参照（Lykken range correction 思想）。
- [Matesanz 2024 (DOI:10.3390/math12020202)] 在线自适应归一化在生理多模态信号中优于固定
  尺度归一化，支撑滚动基线方案。

⚠ 两通道的成熟度不对称：EdaChannel 的度量经 WESAD 真被试全会话回放验收（见其类 docstring）；
HrvChannel 仍是「单一固定 ref、零跨被试自适应」的初版，审计结论见
notes/2026-07-29-hrv-rmssd-metric-audit.md（判别力正常但抗漂移不及格，待独立立项改造）。
"""

from __future__ import annotations

import asyncio
import collections
import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from src.agents.models.zero_affect import ModalityPrior
from src.mcp.zero.external_priors import ModalityKind, build_recommended_prior

logger = logging.getLogger(__name__)

# RMSSD 参考上界（ms，工程假设；健康成人休息 RMSSD 通常 20-80ms，取 100ms 保守上界；
# 集成测试接真硬件后再校）。RMSSD>100ms（极度放松/迷走神经张力极高）时 inverted 钉底
# → μa=-1，方向正确（高 RMSSD = 低 arousal = 副交感优势），信息损失可接受。
# ⚠ HrvChannel 专用。该「单一固定 ref、零跨被试自适应」模式与已退役的 EDA v1 同族，
#   其审计结论见 notes/2026-07-29-hrv-rmssd-metric-audit.md（HRV 判别力 4/5 正常、无反号，
#   但漂移/信号比中位数 0.72 不及格 → 待独立立项按 EdaChannel 的基线相减结构改造）。
_RMSSD_REF_MS: float = 100.0


# ── EDA 唤醒度量常量：均由 WESAD 真被试 P0–P3 探针选定 ────────────────────────
# 依据 notes/2026-07-28-eda-v2-probe-p0-p2-results.md §10——四判据全面胜过「裸 SCL 无修正」对照臂：
# 判别 10/10 vs 9/10 · 抗漂移 0.050 vs 0.324 · 持续比 +1.010 vs +0.966 · 跨被试极差 0.548 vs 1.745
_SCL_BASELINE_HORIZON_SECONDS: float = 1800.0
"""窗间基线回溯时长（秒）。30 分钟——须**舒适超过典型唤醒事件时长**。

⚠ 已知限制：持续比在 horizon ≥ ~1.4× 事件时长时才饱和到 ~1.0；本参数在 WESAD 上只验证到
「1800s 覆盖 645s 应激事件」，**45 分钟以上的持续唤醒是否击穿该值，本数据集无法验证**。
过大同样有害：horizon 逼近会话总长时基线历史开始纳入唤醒期本身（实测 3600s 下判别力塌到 2/10）。
"""

_SCL_DELTA_REF_US: float = 1.0
"""Δ→μa 的对称归一化参考（μS）。判别/持续/跨被试三项对该值不敏感（0.5–1.5 同表现）。"""

_SCL_BASELINE_MIN_OBSERVATIONS: int = 2
"""出首个读数所需的最少历史窗数（与覆盖率双门，取更严者）。"""

_SCL_BASELINE_MIN_COVERAGE: float = 0.15
"""出首个读数所需的历史**时间跨度**占 horizon 的比例。

0.15 → 首读约 4.5 分钟、None 占比 4.2%（对比 0.5：首读 15 分钟、None 14%）。
⚠ 实测降低该值**反而改善抗漂移**（漂移比 0.156→0.050），且持续比/跨被试可比完全不受影响
（二者量的是稳态读数）。未取更激进的 0.05，因其早期基线样本极少、对异常首窗的鲁棒性未验证。
"""


def _linear_normalize(value: float, ref: float) -> float:
    """线性归一化到 [-1, 1]，clip 到 [0, ref] 后映射。

    公式：clip(value / ref, 0, 1) * 2 - 1
    结果 ∈ [-1, 1]；value=0 → -1.0，value≥ref → 1.0。
    """
    ratio = min(max(value / ref, 0.0), 1.0)
    return ratio * 2.0 - 1.0


def _symmetric_normalize(value: float, ref: float) -> float:
    """**对称**归一化到 [-1, 1]：``clip(value / ref, -1, 1)``。

    与 `_linear_normalize` 的区别（**不可互换**）：后者假设输入非负、把 [0, ref] 映到 [-1, 1]
    （value=0 → -1.0）；本函数的输入 Δ 可正可负，零输入必须映到 **0.0**（中性）而非 -1.0。
    EdaChannel 的 Δ（当前窗 SCL − 窗间基线）用本函数；HrvChannel 的 RMSSD 非负，用前者。
    """
    return min(max(value / ref, -1.0), 1.0)


# ---------------------------------------------------------------------------
# EdaChannel
# ---------------------------------------------------------------------------


class EdaChannel:
    """EDA/SC（皮肤电）感知通道：窗内 SCL 均值 − 窗间中位数基线 → arousal 分量。

    valence 恒 0.0（EDA 对 valence 盲，Kreibig 2010）。

    **度量**（`scl_baseline_delta`）：取窗内原始 SCL 均值（μS），减去近
    ``baseline_horizon_seconds`` 内各窗 SCL 的**中位数**基线，再对称归一化到 [-1, 1]。
    **不做 phasic 分解、不依赖 neurokit2**——实测 15/30/60s 窗内 ``mean(eda)`` 与
    ``EDA_Tonic.mean()`` 相对差 <0.1%（胸带腕带均成立），分解在 SCL 层近乎零增量信息；
    跳过它连带消除了 cvxEDA/highpass 双分支这个跨采样率失败成因。

    **为什么减基线**：跨被试 EDA 绝对水平差一个数量级以上，且**会话内漂移量级与信号相当**
    （WESAD 实测 baseline 段自身漂移达 stress−baseline 差值的 72%~98%）。减去被试自身近期
    基线同时吸收这两个轴。

    **冷启动返回 None 而非回退固定 ref**：无基线证据时给出的读数必然基于臆断零点，比不给更有害。
    None 会使 ``merge_physio_priors`` 降级为裸 ``hrv/rmssd``——这是**有意的诚实降级**。

    验收（`notes/2026-07-28-eda-v2-probe-p0-p2-results.md` §11，对接真实例的全会话流式回放）：
    判别 10/10 vs「裸 SCL 无修正」对照臂 9/10 · 抗漂移 0.042（阈值 0.5）· 持续比 +1.010
    （无欠检测）· 跨采样率极差 0.0002 · 胸带腕带各 5/5 · 冷启动首读 270s、None 占比 4.2%。

    ⚠ **已知限制**：``baseline_horizon_seconds`` 须**舒适超过典型唤醒事件时长**——持续比在
    horizon ≥ ~1.4× 事件时长时才饱和到 ~1.0。默认 1800s 在 WESAD 上只验证到「覆盖 645s 应激
    事件」，**45 分钟以上的持续唤醒是否击穿该值，本数据集无法验证**。

    **Protocol 兼容**：结构上满足 PerceptionChannel Protocol（无需继承）：
    - name: str 属性（"eda/sc"）。
    - async sense(signal=None) -> ModalityPrior | None（signal 有默认值=无参可调）。

    **并发/异步**：处理经 ``asyncio.to_thread`` 调度到线程池（对齐 audio/vision），
    基线历史的读改写由 ``history_lock`` 保护——同实例并发 ``collect()`` 时线程安全
    （裁剪 + 快照 + append 全在临界区内）。

    Args:
        sampling_rate: 默认采样率 Hz（构造时传入；信号 dict 可覆盖）。默认 4Hz。
                       ⚠ 本度量对采样率不敏感（窗内取均值），该参数仅作日志/透传。
        signal_source: async callable → dict | None；PerceptionHub 无参调 sense() 时由此
                       获取信号；测试可直接向 sense(signal=...) 传 dict 绕过。
        clock:         时钟注入（默认 ``time.monotonic``）。历史裁剪与冷启动判定按**真实秒数**，
                       注入可测时钟即可让离线回放/单测完全确定（不依赖墙钟）。
        baseline_horizon_seconds: 窗间基线回溯时长（秒）。默认 1800.0，取值依据与已知限制见
                       ``_SCL_BASELINE_HORIZON_SECONDS``。
        delta_ref_us:  Δ→μa 对称归一化参考（μS）。默认 1.0；实测判别/持续/跨被试三项对该值
                       不敏感（0.5–1.5 同表现）。
        baseline_min_observations: 出首个读数所需的最少历史窗数。默认 2（与覆盖率**双门**，
                       取更严者）。
        baseline_min_coverage_fraction: 出首个读数所需的历史时间跨度占 horizon 的比例。
                       默认 0.15（首读约 4.5 分钟、None 占比 4.2%）。
    """

    name: str = "eda/sc"

    def __init__(
        self,
        sampling_rate: int = 4,
        signal_source: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
        baseline_horizon_seconds: float = _SCL_BASELINE_HORIZON_SECONDS,
        delta_ref_us: float = _SCL_DELTA_REF_US,
        baseline_min_observations: int = _SCL_BASELINE_MIN_OBSERVATIONS,
        baseline_min_coverage_fraction: float = _SCL_BASELINE_MIN_COVERAGE,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.signal_source = signal_source
        self.clock = clock
        self.baseline_horizon_seconds = baseline_horizon_seconds
        self.delta_ref_us = delta_ref_us
        self.baseline_min_observations = baseline_min_observations
        self.baseline_min_coverage_fraction = baseline_min_coverage_fraction
        # 窗间基线历史：(时钟秒, 窗内 SCL 均值)。**无 maxlen**——按真实秒数手动裁剪。
        # ⚠ 刻意不用「样本数」表达窗长：deque 里存的是**每次 sense() 产出的一个标量**，
        # 不是原始 EDA 采样点，任何 `秒 × sampling_rate` 式的换算都会单位错配（历史教训，
        # 见 `notes/2026-07-28-eda-metric-redesign-blueprint.md` §4）。
        self.baseline_history: collections.deque[tuple[float, float]] = collections.deque()
        # 保护 baseline_history 的读改写：_process 经 asyncio.to_thread 在线程池执行
        # （对齐 audio/vision，不阻塞事件循环），同实例并发 collect 时多线程会并发读改同一
        # deque（共享可变状态）——把裁剪 + 快照 + append 收进临界区取一致视图（W3）。
        self.history_lock = threading.Lock()

    def reset(self) -> None:
        """清空基线历史（**被试切换时调用方必须调用**）。

        调用后回到冷启动状态：下若干轮返回 None，直到重新攒够基线证据。
        ⚠ 调用方（编排/接入层）有责任在被试切换时调用本方法或 ``PerceptionHub.reset_all()``；
        **未调用将导致新被试沿用旧被试的基线（跨被试污染）**——有状态通道的固有契约，
        Protocol 无法在编译期强制（W2，见 pitfalls「有状态感知通道被试切换须 reset」）。
        污染很直接：旧被试基线直接决定新被试 Δ 的零点。
        """
        with self.history_lock:
            self.baseline_history.clear()

    async def sense(
        self,
        signal: dict[str, Any] | None = None,
    ) -> ModalityPrior | None:
        """async：从 EDA 信号算出 SCL 相对基线的偏移，产出一条 ModalityPrior；无证据则 None。

        Args:
            signal: dict 形状 ``{'eda': ndarray, 'ecg_or_ppg': ndarray|None,
                    'sampling_rate': int|None}``。None 时使用构造注入的 signal_source。

        Returns:
            ModalityPrior(modality="eda/sc", mu=(0.0, μa), precision=(MIN,0.18)) 或 None
            （通道关闭 / 无信号 / 冷启动未攒够基线 / 信号退化）。

        Raises:
            不抛：I/O 异常（OSError/TimeoutError/RuntimeError/ValueError）均 warning+None
            回退；编程错误（TypeError 等）上抛供 PerceptionHub 兜。
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

        # 处理走线程池，与 audio/vision 保持同一调度形状（W3）：本度量本身很轻，但
        # `to_thread` 让本通道与其余通道在 `PerceptionHub.collect()` 的并发语义下一致，
        # 也保留了「日后换重算法不必再改调度」的余量。异常在工作线程内抛出，经 await
        # 传回当前协程，仍由下方 except 兜底优雅回退。
        try:
            return await asyncio.to_thread(self._process, raw)
        except (ValueError, RuntimeError) as exc:
            logger.warning("EdaChannel 信号处理失败，本轮跳过: %s", exc)
            return None

    def _process(self, raw: dict[str, Any]) -> ModalityPrior | None:
        """窗内 SCL 均值 − 窗间中位数基线 → 对称归一化（**不做 phasic 分解**）。

        设计与选参依据：`notes/2026-07-28-eda-metric-redesign-blueprint.md` §2.1 +
        `notes/2026-07-28-eda-v2-probe-p0-p2-results.md` §10/§11（WESAD 真被试四判据全面
        胜过「裸 SCL 无修正」对照臂）。

        为什么不做 ``nk.eda_phasic``：实测 15/30/60s 窗内 ``mean(eda)`` 与 ``EDA_Tonic.mean()``
        相对差中位数 <0.1%（litreview §10.1），分解在 SCL 层近乎零增量信息。跳过它连带消除
        cvxEDA/highpass 双分支——那是旧度量「方法边界断崖」这个跨采样率失败成因本身。
        故本通道**不需要 neurokit2**，也不受采样率驱动的算法分支影响。

        为什么冷启动返回 None 而非回退某个固定 ref：无基线证据时给出的读数必然是「按某个
        臆断零点算出的 Δ」，比不给更有害（旧度量的「回退 linear」在评审实测中导致冷启动期
        读数钉死 −1.0）。None 会使 ``merge_physio_priors`` 降级为裸 ``hrv/rmssd``——这是
        **有意的诚实降级**。
        """
        eda: Any = raw.get("eda")
        if eda is None:
            raise ValueError("signal dict 缺少 'eda' 键")

        scl_raw = float(np.mean(np.asarray(eda, dtype=np.float64)))
        # NaN 守卫**早于任何状态写入**：坏值进了 baseline_history 会污染后续所有窗的基线中位数。
        # 判别力已实证：把本守卫下移到 append 之后，TestV2NaNGuard 立即变红（历史混入 (t, nan)）。
        if not np.isfinite(scl_raw):
            logger.warning("EdaChannel v2: 窗内 SCL 非有限值（信号退化），本轮无证据")
            return None

        now = self.clock()
        with self.history_lock:
            # 按**真实秒数**裁剪（非样本数/调用数）——v1 单位错配不继承
            while (
                self.baseline_history
                and now - self.baseline_history[0][0] > self.baseline_horizon_seconds
            ):
                self.baseline_history.popleft()
            prior = list(self.baseline_history)  # 快照**不含**本次样本
            self.baseline_history.append((now, scl_raw))

        span = (now - prior[0][0]) if prior else 0.0
        if (
            len(prior) < self.baseline_min_observations
            or span < self.baseline_horizon_seconds * self.baseline_min_coverage_fraction
        ):
            return None  # 冷启动：无基线证据 → 无证据

        # 中位数而非均值：对偶发 SCR 尖峰更稳健
        baseline = float(np.median([value for _, value in prior]))
        delta = scl_raw - baseline
        mu_a = _symmetric_normalize(delta, self.delta_ref_us)
        mu_v = 0.0  # EDA 对 valence 盲（Kreibig 2010），与 v1 一致

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
        # NaN 守卫：R 峰不足/坏 ECG → RMSSD 可能 NaN，会穿透归一化产出 NaN 先验。
        # NaN=无有效证据 → 返回 None（对齐通道「本轮无证据则跳过」契约）。
        if not np.isfinite(rmssd_ms):
            logger.warning("HrvChannel: RMSSD 非有限值（NaN/inf，R 峰不足/坏信号），本轮跳过")
            return None

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
