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

⚠ 精度限定：`_process` 逐次 `nk.standardize`（z-score）在进 eda_phasic 前已消除全局幅度差
与基线偏移（现场核验 amp(base·k)==amp(base)，Δ≈1e-16，k=2/5/10 全等）；故 percentile 在
标准化后适配的是「SCR 事件密度」的分布，非原始幅度。Mõttus 的「跨被试幅度差 10–100×」属
standardize 前的原始信号层；幅度轴的完整 Lykken range-correction 意义需真被试数据验证
（见重校协议 notes/2026-07-21-eda-percentile-recalibration-protocol.md）。
"""

from __future__ import annotations

import asyncio
import collections
import logging
import os
import threading
import time
from collections.abc import Callable
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


# ── v2（scl_baseline_delta）常量：均由 WESAD 真被试 P0–P3 探针选定 ─────────────
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
    v2 的 Δ（当前窗 SCL − 窗间基线）用本函数。
    """
    return min(max(value / ref, -1.0), 1.0)


# ---------------------------------------------------------------------------
# EdaChannel
# ---------------------------------------------------------------------------


class EdaChannel:
    """EDA/SC（皮肤电）感知通道。

    从注入的 EDA 信号提取 SCR 幅度，归一化为 arousal 分量。
    valence 恒 0.0（EDA 对 valence 盲，Kreibig 2010）。

    🔴 **v1（`arousal_metric="scr_amplitude_v1"`，当前默认）已知失效**——2026-07-28 WESAD
    真被试实测，其 arousal 读数**在真数据上与唤醒系统性反相关**，勿作真实唤醒指标消费。
    ✅ **修法已落地为 v2**（`arousal_metric="scl_baseline_delta_v2"`，见下），**默认仍是 v1**；
    翻默认值是独立的第二个 PR（蓝图 `notes/2026-07-28-eda-metric-redesign-blueprint.md` §2.6）。

    实测（`notes/2026-07-28-wesad-eda-metric-invalidation.md`，15 被试中取 5 例）：
    「stress > baseline」排序正确率**v1 1/5、经典 SCL 4/5**；跨采样率极差中位数 0.68
    （协议 G1 阈值 0.15）、12 组中 8 组饱和到 ±1.0。
    根因不在 `_SCR_REF_AMPLITUDE*` 两个常量，而在 `_process_v1_scr_amplitude` 的 **per-window
    `nk.standardize`**：z-score 抹掉绝对水平后，`EDA_Phasic.abs().mean()` 实际度量的是
    「相位成分占窗内总方差的比例」；而真实唤醒的主载体是**缓慢的紧张性上升**（stress 段
    SCL 斜率 +0.35~+1.78 μS/窗），它在 z-score 后越大越主导方差 → 相位占比反而越小。
    合成 `nk.eda_simulate` eval 长期全绿，是因为合成信号紧张性平坦、只变 SCR 数——
    恰好落在该度量唯一成立的区域。

    ✅ **v2 = `_process_v2_scl_baseline_delta`**：窗内 SCL 均值 − 窗间中位数基线 → 对称归一化，
    **不做 phasic 分解**（连带消除 cvxEDA/highpass 双分支这个跨采样率失败成因）。
    常量由 WESAD P0–P3 探针选定，四判据全面胜过「裸 SCL 无修正」对照臂
    （`notes/2026-07-28-eda-v2-probe-p0-p2-results.md` §10）。
    ⚠ v2 已知限制：`baseline_horizon_seconds` 须舒适超过典型唤醒事件时长，
    **45 分钟以上的持续唤醒是否击穿默认 1800s，WESAD 数据集无法验证**。

    当前无生产影响：physio 先验在 Zero 点燃门下恒不可点燃（对内核零贡献）；但 Zero 正在
    修点燃门，**一旦 physio 变为可点燃，v1 的反号读数会真的污染内核**——两者是同一条时间线，
    翻默认值到 v2 的时点应与之协调（蓝图 §2.6）。

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
    ⚠ 精度限定：per-call standardize 已消除全局幅度差；percentile 适配的是「SCR 事件密度」
    分布而非原始幅度（详见模块 docstring 精度限定段）。

    Args:
        sampling_rate:        默认采样率 Hz（构造时传入；信号 dict 可覆盖）。默认 4Hz。
        normalization:        归一化策略：``"linear"``（默认）或 ``"percentile"``（有状态自适应）。
        signal_source:        async callable → dict | None；PerceptionHub 无参调 sense()
                              时由此获取信号；测试可直接向 sense(signal=...) 传 dict 绕过。
        window_seconds:       滚动历史时间跨度（秒，**采样率无关的主参数**）。默认 15.0（落文献
                              5–20s 区间）。样本数经 ``round(window_seconds × sampling_rate)``
                              推导：4Hz→60、8Hz→120、256Hz→3840——同一时间跨度在任意采样率下
                              覆盖等长历史（修 sample-count 硬编码 footgun：固定 60 在 4Hz=15s、
                              256Hz 仅 0.23s「崩」）。⚠ 按构造期 sampling_rate 解析；per-call 覆盖
                              sampling_rate 只改 SCR 提取，不改 deque maxlen（窗固定于构造）。
        cold_start_seconds:   冷启动暖机时长（秒，采样率无关）。默认 10.0（4Hz→40 样本，落 Matesanz
                              5–20s 区间；折中安全下界）。样本数经 ``round(cold_start_seconds ×
                              构造期 sampling_rate)`` 推导，未达前回退 linear（零回归过渡）。
                              依据：Lahlou 2022 [PMC9197539] N<20 时 P5/P95 尾分位不可靠
                              （90%CI 需≥175）；Oliveira 2019 [PMC6294150] 偏态数据尾分位
                              建议≥120–300 样本；秒数与阈值仍工程假设，真被试数据再校。
                              ⚠ percentile 适配的是 SCR 事件密度历史（非原始幅度，standardize 已消
                              幅度差）。⚠ 解析后 cold_start > window 时 deque(maxlen=window) 永达
                              不到暖机阈值、percentile 恒退化 linear——构造时**告警**（不再静默）；
                              cold_start==window 为满窗后激活的边界（功能仍可用、不告警）。
        percentile_window:    **样本数显式覆盖**（int，None=由 window_seconds 推导）。默认 None。
                              传非 None 时直接作 deque maxlen、优先于 window_seconds——保精确控制与
                              既有调用零回归。解析后 max(1,…) 兜底：0/负 → 1，不产死 deque。
        percentile_cold_start: **样本数显式覆盖**（int，None=由 cold_start_seconds 推导）。默认
                              None。传非 None 时优先于 cold_start_seconds。
        percentile_range:     分位范围 (q_low, q_high)（工程假设，接真被试数据再校）。
                              默认 (5, 95)。P5/P95 Winsorization 有文献背书、优于 min/max
                              （Lykken 原典 min/max 对伪迹不鲁棒）；样本量不足时可退 (10,90)，
                              与 cold_start 联动调整。

        ── 以下为 v2（scl_baseline_delta）专属；v1 路径完全不读 ──
        arousal_metric:       度量选择。None（默认）= 走 env ``ZERO_EDA_AROUSAL_METRIC``，
                              env 缺省/非法则 ``"scr_amplitude_v1"``（零回归）。显式入参优先于 env。
                              ⚠ **构造期一次性解析**——v2 有状态（baseline_history），运行中切换
                              会让基线跨语义污染，故不在 sense() 内逐次读 env。
        clock:                时钟注入（默认 ``time.monotonic``）。v2 的历史裁剪与冷启动判定按
                              **真实秒数**，注入可测时钟即可让离线回放/单测完全确定（不依赖墙钟）。
        baseline_horizon_seconds: v2 窗间基线回溯时长（秒）。默认 1800.0，见
                              ``_SCL_BASELINE_HORIZON_SECONDS`` 的取值依据与已知限制。
        delta_ref_us:         v2 的 Δ→μa 对称归一化参考（μS）。默认 1.0；实测判别/持续/跨被试
                              三项对该值不敏感（0.5–1.5 同表现）。
        baseline_min_observations: v2 出首个读数所需的最少历史窗数。默认 2（与覆盖率**双门**，
                              取更严者）。
        baseline_min_coverage_fraction: v2 出首个读数所需的历史时间跨度占 horizon 的比例。
                              默认 0.15（首读约 4.5 分钟、None 占比 4.2%）。
    """

    name: str = "eda/sc"

    def __init__(
        self,
        sampling_rate: int = 4,
        normalization: Literal["linear", "percentile"] = "linear",
        signal_source: Any | None = None,
        window_seconds: float = 15.0,
        cold_start_seconds: float = 10.0,
        percentile_window: int | None = None,
        percentile_cold_start: int | None = None,
        percentile_range: tuple[int, int] = (5, 95),
        arousal_metric: Literal["scr_amplitude_v1", "scl_baseline_delta_v2"] | None = None,
        clock: Callable[[], float] = time.monotonic,
        baseline_horizon_seconds: float = _SCL_BASELINE_HORIZON_SECONDS,
        delta_ref_us: float = _SCL_DELTA_REF_US,
        baseline_min_observations: int = _SCL_BASELINE_MIN_OBSERVATIONS,
        baseline_min_coverage_fraction: float = _SCL_BASELINE_MIN_COVERAGE,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.normalization = normalization
        self.signal_source = signal_source
        self.window_seconds = window_seconds
        self.cold_start_seconds = cold_start_seconds
        self.percentile_range = percentile_range

        # 度量选择：显式入参 > env > 默认 v1（零回归）。**构造期一次性解析**，不在 sense() 内
        # 逐次读——v2 是有状态的（baseline_history），运行中切换会让基线跨语义污染。
        self.arousal_metric = arousal_metric or self._resolve_arousal_metric_env()
        self.clock = clock
        self.baseline_horizon_seconds = baseline_horizon_seconds
        self.delta_ref_us = delta_ref_us
        self.baseline_min_observations = baseline_min_observations
        self.baseline_min_coverage_fraction = baseline_min_coverage_fraction
        # v2 的窗间基线历史：(时钟秒, 窗内 SCL 均值)。**无 maxlen**——按真实秒数手动裁剪，
        # 不复用 v1 那套 `round(秒 × sampling_rate)` 的样本数推导（该推导与「每次调用产出一个
        # 标量」的实际单位错配，见蓝图 §4；v2 不继承它）。
        self.baseline_history: collections.deque[tuple[float, float]] = collections.deque()

        # 秒制化解析：窗/冷启动 = round(秒 × 构造期 sampling_rate)，采样率无关（修固定 60 在
        # 256Hz=0.23s「崩」的 footgun）。显式样本数覆盖（非 None）优先——保精确控制与既有调用零回归。
        # max(1,…) 对推导+覆盖两路兜底：window_seconds≤0 或误传 percentile_window=0 均不产死 deque。
        resolved_window = (
            percentile_window
            if percentile_window is not None
            else round(window_seconds * sampling_rate)
        )
        resolved_cold_start = (
            percentile_cold_start
            if percentile_cold_start is not None
            else round(cold_start_seconds * sampling_rate)
        )
        self.percentile_window = max(1, resolved_window)
        self.percentile_cold_start = max(1, resolved_cold_start)
        # cold_start > window 时 deque(maxlen=window) 永达不到暖机阈值 → percentile 恒退化 linear
        # （原静默 footgun）。== 边界满窗后仍激活、功能可用，不在此列（门控是 size<cold_start）。
        # 不 raise（保优雅回退），仅 percentile 模式告警。
        if (
            self.normalization == "percentile"
            and self.percentile_cold_start > self.percentile_window
        ):
            logger.warning(
                "EdaChannel percentile：cold_start(%d) > window(%d)，deque 永达不到暖机阈值，"
                "percentile 恒退化为 linear。请增大 window_seconds 或减小 cold_start_seconds。",
                self.percentile_cold_start,
                self.percentile_window,
            )

        # 按 method 分桶的滚动幅度历史（逐实例状态，非全局）
        # key: "cvxEDA" | "highpass"；value: 最近 percentile_window 个 scr_amplitude 值
        self._amplitude_history: dict[str, collections.deque[float]] = {
            "cvxEDA": collections.deque(maxlen=self.percentile_window),
            "highpass": collections.deque(maxlen=self.percentile_window),
        }
        # 保护 _amplitude_history 的读改写：_process 现经 asyncio.to_thread 在线程池执行
        # （对齐 audio/vision，不阻塞事件循环），同实例并发 collect 时多线程会并发读改同一
        # deque（共享可变状态）——把 append + 快照收进临界区取一致视图（W3；详见 _process 内
        # 注释的现场核验与自由线程前瞻）。与 audio/vision 的 threading.Lock 同族（那两处防并发
        # 双载模型，此处防并发改历史）。
        self.history_lock = threading.Lock()

    @staticmethod
    def _resolve_arousal_metric_env() -> Literal["scr_amplitude_v1", "scl_baseline_delta_v2"]:
        """解析 ``ZERO_EDA_AROUSAL_METRIC``；非法值告警后回退 v1（保优雅回退，不 raise）。"""
        raw = os.getenv("ZERO_EDA_AROUSAL_METRIC")
        if raw is None:
            return "scr_amplitude_v1"
        value = raw.strip()
        if value in ("scr_amplitude_v1", "scl_baseline_delta_v2"):
            return value  # type: ignore[return-value]
        logger.warning(
            "ZERO_EDA_AROUSAL_METRIC=%r 非法（合法值：scr_amplitude_v1 / scl_baseline_delta_v2），"
            "回退 scr_amplitude_v1",
            raw,
        )
        return "scr_amplitude_v1"

    def reset(self) -> None:
        """清空滚动历史（**被试切换时调用方必须调用**）——v1 幅度历史与 v2 基线历史**都清**。

        调用后 percentile 模式回到冷启动状态，linear 模式不受影响；v2 回到冷启动（下若干轮返回
        None 直到重新攒够基线）。
        ⚠ 调用方（编排/接入层）有责任在被试切换时调用本方法或 ``PerceptionHub.reset_all()``；
        **未调用将导致新被试沿用旧被试的历史分位区间/基线（跨被试污染）**——有状态通道的固有契约，
        Protocol 无法在编译期强制（W2，见 pitfalls「有状态感知通道被试切换须 reset」）。
        v2 的污染更直接：旧被试基线直接决定新被试 Δ 的零点。
        """
        with self.history_lock:
            for key in self._amplitude_history:
                self._amplitude_history[key].clear()
            self.baseline_history.clear()

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
        """按 ``arousal_metric`` 分派到 v1 / v2 实现（构造期已定，运行期不再读 env）。"""
        if self.arousal_metric == "scl_baseline_delta_v2":
            return self._process_v2_scl_baseline_delta(raw)
        return self._process_v1_scr_amplitude(raw)

    def _process_v2_scl_baseline_delta(self, raw: dict[str, Any]) -> ModalityPrior | None:
        """v2：窗内 SCL 均值 − 窗间中位数基线 → 对称归一化（**不做 phasic 分解**）。

        设计与选参依据：`notes/2026-07-28-eda-metric-redesign-blueprint.md` §2.1 +
        `notes/2026-07-28-eda-v2-probe-p0-p2-results.md` §10（WESAD 真被试四判据全面胜过
        「裸 SCL 无修正」对照臂）。

        为什么不做 ``nk.eda_phasic``：实测 15/30/60s 窗内 ``mean(eda)`` 与 ``EDA_Tonic.mean()``
        相对差中位数 <0.1%（litreview §10.1），分解在 SCL 层近乎零增量信息。跳过它连带消除
        cvxEDA/highpass 双分支——即 v1 「方法边界断崖」这个跨采样率失败成因本身。
        故 v2 **不需要 neurokit2**，也不受采样率驱动的算法分支影响。

        为什么冷启动返回 None 而非回退固定 ref：无基线证据时给出的读数必然是「按某个臆断零点
        算出的 Δ」，比不给更有害（v1 的「回退 linear」在评审实测中导致冷启动期读数钉死 −1.0）。
        None 会使 ``merge_physio_priors`` 降级为裸 ``hrv/rmssd``——这是**有意的诚实降级**。
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

    def _process_v1_scr_amplitude(self, raw: dict[str, Any]) -> ModalityPrior | None:
        """v1（🔴 已知失效，见类 docstring）：延迟 import neurokit2 并执行 EDA 处理。

        **本方法自 v2 引入起逻辑零改动**（仅重命名），以保 ``arousal_metric="scr_amplitude_v1"``
        默认路径的零回归。

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
        # NaN 守卫（置于 linear/percentile 分支之前，两分支共用）：phasic 全 NaN（退化/坏信号）→
        # abs().mean()=NaN，会穿透 min/max 钳制产出 NaN 先验；percentile 的退化守卫
        # `p_high-p_low<eps` 也因 NaN 比较恒 False 而被跳过，且 NaN 会污染滚动历史。NaN=**无有效
        # 证据** → 返回 None（对齐通道「本轮无证据则跳过」契约，非 -1 假称低唤醒；此处 return 亦阻止
        # NaN 进 _amplitude_history）。
        if not np.isfinite(scr_amplitude):
            logger.warning("EdaChannel: SCR 幅度非有限值（NaN/inf，退化信号），本轮跳过")
            return None
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
