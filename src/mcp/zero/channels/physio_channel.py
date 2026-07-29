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

两通道**同构不同参**（2026-07-29 起）：都是「窗间中位数基线 + 对称归一化 + 按秒裁剪 +
冷启动返回 None」，都经 WESAD 真被试全会话流式回放标定与验收（各自类 docstring 有实测数字）。
⚠ **同构只到结构为止**：两组常量刻意分家（`_SCL_*` / `_RMSSD_*`，量纲 μS vs ms 不可比），
且**喂入 Δ 的符号方向相反**（SCL↑=唤醒↑，RMSSD↑=唤醒↓）——照抄另一侧的减法方向即复刻
EDA v1 的系统性反号病（那次判别力 1/5，合成信号上长期全绿、直到真被试才暴露）。
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


# ── HRV 唤醒度量常量：由 WESAD S2–S6 胸带 ECG(700Hz) 全会话流式回放独立标定 ────
# 标定脚本 `evals/wesad_hrv_baseline_delta_calibration.py`（两阶段：逐窗抽 RMSSD → replay
# 参数组合），验收 `evals/wesad_hrv_v2_acceptance_gates.py`（对接**本通道真实例**）。
# ⚠ **一个都不是从 `_SCL_*` 抄来的**：量纲（ms vs μS）与时间常数各自独立推导；两组常量刻意
#   分家，任何一侧的重标不得静默改动另一通道（多路径优先级误配面的同族风险）。
_RMSSD_BASELINE_HORIZON_SECONDS: float = 1800.0
"""窗间基线回溯时长（秒）。

选参目标函数是**持续比拐点**——刻意**不**取三门中的任何一门：G-Drift 与跨采样率极差对
「恒输出 0 的退化度量」都给满分，用 G-Drift 最小选参会选出 120s 那档（判别力仅 2/5、
持续比 −0.432，读数在应激段内反向）。实测持续比随 horizon 上升并饱和于 0.899，1800s 是
达到饱和 95% 的最小候选（实测 0.858；1200s 只有 0.538）；取最小达标值即最省冷启动
（冷启动 = horizon × min_coverage，随 horizon 线性增长）。1800s ≈ 2.7× WESAD 应激事件
时长（600–660s）。候选覆盖 120–7200s（60×）。

⚠ **与 `_SCL_BASELINE_HORIZON_SECONDS` 同为 1800.0 纯属独立推导后的巧合**：本值完全没有
套用 EDA 的取值，目标函数是 HRV 自己的持续比拐点；同族的只是「horizon 须舒适超过事件
时长」这条理由，而 WESAD 的应激块对两通道恰是同一段。**这不构成「HRV 可照抄 EDA」的
证据**——`_RMSSD_DELTA_REF_MS` 与 EDA 侧完全不同（100ms vs 1.0μS，量纲都不可比）。

⚠ 已知限制：分离度在整个可测区间内随 horizon **单调上升、无内部极值**——WESAD 应激块仅
600–660s，EDA 那个「horizon 逼近会话总长 → 基线开始纳入唤醒期本身」的失效点**根本没进
可测范围**（会话仅 6100–6500s，≥5400s 的档按秒裁剪几乎不触发，等同「全历史基线」退化
对照）。故本值的**上侧边界未被实测确定**。另有结构性上界：冷启动门要求
``span ≥ horizon × min_coverage``，即 ``horizon ≤ 会话长度 / 0.15``，超过则通道永不出读数。
"""

_RMSSD_DELTA_REF_MS: float = 100.0
"""Δ→μa 的对称归一化参考（**毫秒**；与 EDA 的 μS 量纲不可比，勿跨通道搬运）。

取「**最差被试** stress 段撞 |μa|=1 边界 <10% 的最小候选」：ref=100 时 S3 9.1%、其余四人
0.0%（ref=60 时 S3 已 36.4%）。该值 ≈ baseline+stress 窗 |Δ| 的 p99（92.6ms）。
判据刻意用最差被试而非池化率——池化率在 ref≥40 时就已 <11%，但那 100% 全由 S3 一人贡献，
「多数人没事」不构成放行理由（EDA 那次池化率掩盖单人问题的教训）。

⚠ **这是 n=5 下由单个被试决定的脆弱结论**：S3 的 RMSSD 中位数 125ms 是 S5（27ms）的 4.6×；
把 S3 压到 9.1% 的代价是所有人读数幅度砍掉约 40%（分离度 +0.184 → +0.110）。换一批被试
大概率要重标。
⚠ 更根本的已知缺陷（本轮**未**修）：|Δ| 幅度与被试自身 RMSSD 水平成比例（各被试
|Δ|p90 / RMSSD中位 收敛在 0.35–0.49，而 |Δ|p90 本身跨被试差 6.1×）→ **任何单一 ms 常数
必然对一部分被试失配**。基线相减修掉了跨被试的**水平**，没修掉**尺度**。相对 Δ 口径
（``(baseline − now) / baseline``，ref≈0.7）实测在同等饱和水平下分离度约 1.8×，属另开
一项（偏离审计明令的「结构复用、只换喂入量」），本轮未采纳。
"""

_RMSSD_BASELINE_MIN_OBSERVATIONS: int = 2
"""出首个读数所需的最少历史窗数（与覆盖率双门，取更严者）。

⚠ **沿用 EDA v2 选定值，本轮未 sweep**（标定授权只含 horizon / delta_ref / 窗长三项）。
1800s horizon 下冷启动由覆盖率门主导（270s ≫ 2 窗 × 60s），故本值当前不是瓶颈；
若日后把 horizon 调小到 `min_obs × 窗长` 量级，两道门的相对严格度会互换，须重标。
"""

_RMSSD_BASELINE_MIN_COVERAGE: float = 0.15
"""出首个读数所需的历史**时间跨度**占 horizon 的比例。

⚠ **沿用 EDA v2 选定值，本轮未 sweep**。60s 窗实测首读 300s、None 占比 4.7%，与算式
``1800 × 0.15 = 270s`` 吻合（数据时钟回放，非墙钟——墙钟下按秒裁剪与冷启动门都不会触发）。
⚠ EDA 那条「降低 coverage **反而**改善抗漂移」的实测经验**未在 HRV 上验证**，不得外推。
"""


def _symmetric_normalize(value: float, ref: float) -> float:
    """**对称**归一化到 [-1, 1]：``clip(value / ref, -1, 1)``。

    EdaChannel 与 HrvChannel **共用**：两者喂入的都是「当前窗指标与窗间基线之差」，可正可负，
    零输入必须映到 **0.0**（中性）。这正是它不可被「非负输入 → [0, ref] 线性映到 [-1, 1]」
    那类单侧公式替代的原因——后者把 Δ=0 读成 −1.0，且实践中总要配一个 ``max(0.0, ·)`` 地板
    把整个低唤醒半边压成常数（HrvChannel v1 的实测缺陷：S3 静息 89.5% 的窗钉在 μa=−1.0，
    全体撞界 10.5%；改用本函数后降到 0.7%）。

    ⚠ 两通道喂入的 Δ **符号方向相反**（EDA：当前 − 基线；HRV：基线 − 当前），方向由各自
    ``_process`` 负责，本函数只做无符号偏好的钳制缩放，不含任何方向语义。
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
    """HRV/RMSSD（心率变异性）感知通道：**窗间中位数基线 − 窗内 RMSSD** → arousal 分量。

    valence 恒 0.0（HRV 对 valence 盲，Kreibig 2010）。

    **度量**（`rmssd_baseline_delta`）：从 ECG/PPG 提取窗内 RMSSD（ms），用近
    ``baseline_horizon_seconds`` 内各窗 RMSSD 的**中位数**基线**减去**它，再对称归一化到
    [-1, 1]。⚠ **减法方向与 EdaChannel 相反**：RMSSD↑ = 迷走张力↑ = 唤醒**↓**，故
    Δ = 基线 − 当前（EDA 是 当前 − 基线）。这是本通道唯一的高危处，见 ``_process`` 就近注释。

    **为什么减基线**（改造的正当理由，2026-07-29）：**跨被试可比性**。固定 ref 的旧实现下，
    同样是「坐着休息」的静息态，S3 读 −0.994（89.5% 的静息窗钉在 −1.0）、S10 读 +0.693，
    跨被试静息读数极差 **1.687**——而 Zero 消费的是 [-1,1] 上的**绝对值**（μa 直接进精度
    加权融合），这个偏置会变成对被试的系统性**错误情绪归因**，不只是精度问题。基线相减
    买到的正是「μa=0 对每个被试同义」：极差降到 **0.293（−83%）**，撞 |μa|=1 从 10.5%
    降到 0.7%（−93%），残差 σ 2.00 → 1.39（Πa 0.25 → 0.52）。
    ⚠ **原立项理由已被推翻，勿再引用**：审计说的「按 EDA v2 的 G-Drift 门不及格（0.72）」
    用的是**单窗首/末**估计量；换成 v2 门自己的**早/晚半均值**估计量，现状实现是 0.154
    （PASS）。同数据同实现、两种估计量给出跨越合格线的相反判定。

    **诚实附注（改造并非全面更优）**：判别力两版都是 4/5（S6 是三处独立分析里一致的非
    响应者，其 stress 段 RMSSD 反而高于 baseline）；G-Drift 0.164 vs 0.154（同门内略输）；
    **组内分离度反而更低**（+0.110 vs +0.303 中位）。本改造的目标函数是跨被试可比，不是
    判别力/抗漂移——后两者只作「不许劣化」的约束。且按 Zero 自己的判据（σ > 值域半宽 1.0
    即噪声占主导），本通道**仍不合格**（σ=1.6）；改造把它从 2.0 改善到 1.6，不改变结论。

    **冷启动返回 None 而非回退固定 ref**：无基线证据时给出的读数必然基于臆断零点，比不给
    更有害（v1 的固定 ref 正是 S3 静息钉死 −1.0 的成因）。⚠ 与 EdaChannel 的冷启动窗口
    **重合**（两侧 horizon×coverage 均为 270s），故会话开始后约 270–300s **两条 physio 流
    同时为 None** → ``merge_physio_priors`` 零命中 → 载荷里**完全没有 physio 家族流**
    （不是流名降级，是流不存在）。这是有意的诚实降级，已作 R5.1-bis 跨仓报备。

    **窗长恒定性（EDA 无此约束）**：RMSSD 随窗内 R-R 间期数变化，窗长一变 ``rmssd_now``
    与历史中位数即**不同口径**、Δ 失去意义 → 调用方须在一个会话内保持投喂窗长恒定，
    变更窗长等同换被试，须先 ``reset()``。取「文档约定」而非运行期检测，依据是 G-WindowLen
    实测敏感度小：{15,30,60}s 下同一 stress 段 μa 的逐被试极差中位仅 **0.0156**（最大
    0.0970），暖机后 None 率三档**均为 0.00%**（15s 也能稳定出 RMSSD）。
    主用 60s、建议下界 30s，15s 可行但不推荐作默认——代价不是「算不出来」而是噪声：
    标定实测窗间 CV 涨 2.1×（0.151→0.312）、RMSSD>150ms 的伪迹窗从 4.1% 涨到 6.0%，
    且**伪迹是有限值、NaN 守卫拦不住**，会直接进基线历史（中位数基线是唯一的缓冲）。

    验收（`evals/wesad_hrv_v2_acceptance_gates.py`，**对接本类真实例**的 WESAD 全会话流式
    回放，60s 窗 / 700Hz 胸带 ECG / S2–S6）：判别力 4/5（对照臂 v1 同为 4/5）PASS ·
    G-Drift 中位 0.164 / 最大 0.343（门 <0.5）PASS · G1' 跨采样率（700/350/175Hz，有状态）
    极差中位 0.0057 / 最大 0.0125（门 <0.15）PASS · G-Sign 池化 stress +0.183 > 0 >
    baseline −0.096 PASS · 冷启动 300s（≈ horizon×coverage=270s）· None 4.7% ·
    **跨被试静息 μa 极差 0.227（对照臂 1.400）** ← 本改造的目标函数。
    另由标定/分布脚本给出（`wesad_hrv_baseline_delta_calibration.py` /
    `wesad_mu_a_distribution.py` 同族）：stress 撞 |μa|=1 最差被试 9.1%、其余四人 0.0%
    （对照臂 S3 静息 89.5%）；n=15 全体撞界 0.7% vs 对照臂 10.5%。

    ⚠ **已知限制（覆盖面很小，报告时勿放大）**：n=5 标定 / n=15 分布，单数据集（WESAD）、
    单设备通路（RespiBAN 胸带 ECG，坐姿实验室）。**腕带 BVP/PPG 完全未测**（PPG 峰定位
    精度远低于 ECG，RMSSD 对此高度敏感，不可假设可迁移）；amusement 段实测几乎无响应
    （p50 −0.017），σ 是在 baseline vs TSST 这一对上标定的；60s 窗本身已是对短时 HRV
    经典 5 分钟建议的激进外推，未做与金标准的一致性比对。

    **Protocol 兼容**：结构上满足 PerceptionChannel Protocol（无需继承）：
    - name: str 属性（"hrv/rmssd"）。
    - async sense(signal=None) -> ModalityPrior | None（signal 有默认值=无参可调）。
    - 可选协议 ``reset()`` / ``prepare()``（鸭子类型，由 PerceptionHub 自动接管）。

    **并发/异步**：处理经 ``asyncio.to_thread`` 调度到线程池（ecg_process + hrv_time 计算重），
    基线历史的读改写由 ``history_lock`` 保护——同实例并发 ``collect()`` 时线程安全
    （裁剪 + 快照 + append 全在临界区内）。

    Args:
        sampling_rate:   默认采样率 Hz（构造时传入；信号 dict 可覆盖）。默认 256Hz（ECG 常见）。
                         ⚠ 与 EdaChannel 相反，**本通道真的会用它**：R 峰定位对采样率敏感
                         （审计在 175Hz 已见劣化），rate 会传给 ``ecg_process``/``hrv_time``。
        signal_source:   async callable → dict | None；PerceptionHub 无参调 sense()
                         时由此获取信号；测试可直接向 sense(signal=...) 传 dict 绕过。
        clock:           时钟注入（默认 ``time.monotonic``）。历史裁剪与冷启动判定按**真实
                         秒数**，注入可测时钟即可让离线回放/单测完全确定（不依赖墙钟）。
        baseline_horizon_seconds: 窗间基线回溯时长（秒）。默认 1800.0，取值依据与已知限制见
                         ``_RMSSD_BASELINE_HORIZON_SECONDS``（⚠ 与 EDA 同值属巧合，非照抄）。
        delta_ref_ms:    Δ→μa 对称归一化参考（**毫秒**）。默认 100.0，依据见
                         ``_RMSSD_DELTA_REF_MS``（由单个被试 S3 决定的 n=5 脆弱结论）。
        baseline_min_observations: 出首个读数所需的最少历史窗数。默认 2（与覆盖率**双门**，
                         取更严者）；**沿用 EDA v2 未 sweep**。
        baseline_min_coverage_fraction: 出首个读数所需的历史时间跨度占 horizon 的比例。
                         默认 0.15（首读约 300s）；**沿用 EDA v2 未 sweep**。
    """

    name: str = "hrv/rmssd"

    def __init__(
        self,
        sampling_rate: int = 256,
        signal_source: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
        baseline_horizon_seconds: float = _RMSSD_BASELINE_HORIZON_SECONDS,
        delta_ref_ms: float = _RMSSD_DELTA_REF_MS,
        baseline_min_observations: int = _RMSSD_BASELINE_MIN_OBSERVATIONS,
        baseline_min_coverage_fraction: float = _RMSSD_BASELINE_MIN_COVERAGE,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.signal_source = signal_source
        self.clock = clock
        self.baseline_horizon_seconds = baseline_horizon_seconds
        self.delta_ref_ms = delta_ref_ms
        self.baseline_min_observations = baseline_min_observations
        self.baseline_min_coverage_fraction = baseline_min_coverage_fraction
        # 窗间基线历史：(时钟秒, 窗内 RMSSD ms)。**无 maxlen**——按真实秒数手动裁剪。
        # ⚠ 刻意不用「样本数」表达窗长：deque 里存的是**每次 sense() 产出的一个标量**，
        # 不是原始 ECG 采样点，任何 `秒 × sampling_rate` 式的换算都会单位错配（同 EdaChannel
        # 的历史教训）。
        self.baseline_history: collections.deque[tuple[float, float]] = collections.deque()
        # 保护 baseline_history 的读改写：_process 经 asyncio.to_thread 在线程池执行，
        # 同实例并发 collect 时多线程会并发读改同一 deque——把裁剪 + 快照 + append 收进
        # 临界区取一致视图（同 EdaChannel W3）。
        self.history_lock = threading.Lock()

    def reset(self) -> None:
        """清空基线历史（**被试切换、或投喂窗长变更时，调用方必须调用**）。

        调用后回到冷启动状态：下若干轮返回 None，直到重新攒够基线证据。
        ⚠ 调用方（编排/接入层）有责任在被试切换时调用本方法或 ``PerceptionHub.reset_all()``；
        **未调用将导致新被试沿用旧被试的基线（跨被试污染）**——有状态通道的固有契约，
        Protocol 无法在编译期强制（见 pitfalls「有状态感知通道被试切换须 reset」）。
        污染很直接：旧被试基线直接决定新被试 Δ 的零点，而 RMSSD 的跨被试水平差达 4.6×
        （S3 中位 125ms vs S5 27ms），足以把新被试整段读数推到饱和。
        """
        with self.history_lock:
            self.baseline_history.clear()

    async def prepare(self) -> None:
        """async：预热 neurokit2 的延迟 import（PerceptionHub 并发派发前串行调用）。

        **这不是性能优化，是修一个实测竞态**：``PerceptionHub.prepare_all`` docstring 里那条
        traceback 的受害者正是本通道——AudioChannel 线程首次 import torch 时，本通道线程的
        ``nk.hrv_time → scipy.stats.iqr`` 会探测 ``sys.modules["torch"]`` 撞上半成品模块并抛
        AttributeError，先验被 ``collect()`` 当「通道异常」静默跳过。

        改造后代价比改造前更高：一次被静默跳过 = **少一条基线历史** = 冷启动更慢
        （以前只是丢一轮读数，现在还会推迟首读时刻）。

        幂等（模块已在 sys.modules 时为一次字典查找）；缺库时 warning 后返回——``sense()``
        仍按既有约定优雅回退（不在预热阶段把缺库抬升为错误）。
        """

        def _warm() -> None:
            import neurokit2  # noqa: F401  # 关键：让 nk 及其 scipy 依赖在并发之前初始化完

        try:
            await asyncio.to_thread(_warm)
        except ImportError as exc:
            logger.warning(
                "HrvChannel 预热 neurokit2 失败（不阻断，sense() 时将优雅回退）: %s", exc
            )

    async def sense(
        self,
        signal: dict[str, Any] | None = None,
    ) -> ModalityPrior | None:
        """async：从 ECG/PPG 信号提取 RMSSD，产出一条 ModalityPrior；无证据则返回 None。

        Args:
            signal: dict 形状 ``{'ecg_or_ppg': ndarray, 'sampling_rate': int|None}``。
                    None 时使用构造注入的 signal_source。

        Returns:
            ModalityPrior(modality="hrv/rmssd", mu=(0.0, μa), precision=(MIN,0.18)) 或 None
            （通道关闭 / 无信号 / 冷启动未攒够基线 / RMSSD 非有限值）。

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
        """延迟 import neurokit2 → 窗内 RMSSD → **窗间中位数基线 − 当前** → 对称归一化。

        pipeline：``nk.ecg_process`` → ``nk.hrv_time`` → RMSSD(ms) → 基线相减 → clip 到 [-1,1]。

        为什么减基线而不是打固定 ref：固定 ref 版本下静息态读数的跨被试极差达 1.687
        （S3 −0.994 / S10 +0.693，同为「坐着休息」却给出相反的情绪归因），而消费方按
        [-1,1] 的**绝对值**做精度加权融合。减去被试自身近期基线使「μa=0 对每个被试同义」，
        极差降至 0.293。标定与验收见类 docstring 与 `evals/wesad_hrv_*` 三件。

        为什么冷启动返回 None 而非回退某个固定 ref：无基线证据时给出的读数必然基于臆断
        零点。旧实现正因此把 S3 静息 89.5% 的窗钉在 −1.0。None 是**有意的诚实降级**
        （与 EdaChannel 冷启动窗口重合时，该轮完全不出 physio 家族流，已跨仓报备）。

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
        # NaN 守卫（R 峰不足/坏 ECG）**必须早于任何状态写入**：坏值一旦进 baseline_history，
        # 就会污染后续所有窗的基线中位数，而不只是丢掉本轮读数。
        # 判别力已实证：把本守卫下移到 append 之后，TestHrvNaNGuard 立即变红。
        if not np.isfinite(rmssd_ms):
            logger.warning("HrvChannel: RMSSD 非有限值（NaN/inf，R 峰不足/坏信号），本轮跳过")
            return None

        now = self.clock()
        with self.history_lock:
            # 按**真实秒数**裁剪（非样本数/调用数——deque 存的是每窗一个 RMSSD 标量）
            while (
                self.baseline_history
                and now - self.baseline_history[0][0] > self.baseline_horizon_seconds
            ):
                self.baseline_history.popleft()
            history = list(self.baseline_history)  # 快照**不含**本次样本
            self.baseline_history.append((now, rmssd_ms))

        span = (now - history[0][0]) if history else 0.0
        if (
            len(history) < self.baseline_min_observations
            or span < self.baseline_horizon_seconds * self.baseline_min_coverage_fraction
        ):
            return None  # 冷启动：无基线证据 → 无证据

        # 中位数而非均值：异位搏动/运动伪迹会让单窗 RMSSD 爆表（实测最大 395ms，60s 窗下
        # 4.1% 的窗 >150ms），且这些是**有限值**、NaN 守卫拦不住 —— 中位数对偶发尖峰稳健。
        baseline = float(np.median([value for _, value in history]))
        # ⚠⚠ 本行是本次改造唯一的高危处：**减法方向与 EdaChannel 相反，不可照抄**。
        #     EDA：SCL↑ = 交感激活↑ = 唤醒↑     → Δ = 当前 − 基线
        #     HRV：RMSSD↑ = 迷走张力↑ = 唤醒**↓** → Δ = 基线 − 当前（本行）
        # 写成 `rmssd_ms - baseline` 即逐字复刻 EDA v1 的系统性反号病：那次判别力 1/5、
        # 在合成信号上长期全绿，直到 WESAD 真被试回放才暴露，整条度量随后被删除。
        # 方向依据：[Kreibig 2010]（应激下迷走撤退 → RMSSD 降、唤醒升）+ WESAD 实测
        # （stress 段 μa 中位 +0.103 vs baseline −0.004；15 被试 11 人方向正确，
        # Wilcoxon p=0.013）。三重守卫：本注释 · 单测 TestHrvSignDirection（反号必红）·
        # `evals/wesad_hrv_v2_acceptance_gates.py` 的 G-Sign 门（真被试同实例内方向性）。
        delta = baseline - rmssd_ms
        mu_a = _symmetric_normalize(delta, self.delta_ref_ms)
        mu_v = 0.0  # HRV 对 valence 盲（Kreibig 2010）

        return build_recommended_prior(
            modality=self.name,
            mu=(mu_v, mu_a),
            kind=ModalityKind.PHYSIO,
        )
