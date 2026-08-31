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
⚠ **同构只到结构为止**：两组常量刻意分家（`_SCL_*` / `_RMSSD_*`，量纲 μS vs nats/ms 不可比），
且**喂入 Δ 的符号方向相反**（SCL↑=唤醒↑，RMSSD↑=唤醒↓）——照抄另一侧的减法方向即复刻
EDA v1 的系统性反号病（那次判别力 1/5，合成信号上长期全绿、直到真被试才暴露）。

**HrvChannel v3（2026-08-31 起，lnRMSSD 差分）**：文献门 `notes/2026-08-31-hrv-v3-litgate.md`
核验 [Shaffer & Ginsberg 2017 (DOI:10.3389/fpubh.2017.00258)] 的 log 变换惯例后，把 v2 的
**原始 ms 差**换成**自然对数差**，用 WESAD 真被试重标定并验收（σ 改善、方向性守恒、
None/撞界不劣化，见 `HrvChannel` 类 docstring）——只换 HRV 一侧的喂入量与归一化参考，
EdaChannel 与其余结构完全不受影响。
"""

from __future__ import annotations

import asyncio
import collections
import logging
import math
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
证据**——`_RMSSD_LN_DELTA_REF` 与 EDA 侧完全不同（v3 起 3.0nats vs 1.0μS，量纲都不可比；
v2 时代是 100ms vs 1.0μS，结论同样成立）。

⚠ 已知限制：分离度在整个可测区间内随 horizon **单调上升、无内部极值**——WESAD 应激块仅
600–660s，EDA 那个「horizon 逼近会话总长 → 基线开始纳入唤醒期本身」的失效点**根本没进
可测范围**（会话仅 6100–6500s，≥5400s 的档按秒裁剪几乎不触发，等同「全历史基线」退化
对照）。故本值的**上侧边界未被实测确定**。另有结构性上界：冷启动门要求
``span ≥ horizon × min_coverage``，即 ``horizon ≤ 会话长度 / 0.15``，超过则通道永不出读数。
"""

_RMSSD_LN_DELTA_REF: float = 3.0
"""Δ→μa 的对称归一化参考（**nats**，自然对数差；替代 v2 的 `_RMSSD_DELTA_REF_MS`，
两者量纲不可比、不可互换、不可跨版本搬运）。

标定 `evals/wesad_hrv_v3_ln_calibration.py`（WESAD S2–S6，与 v2 同被试集合）：
取「判别力 ≥4/5 且最差被试（S3）stress 撞 |μa|=1 边界 ≤5%」约束内**分离度最大**的候选
（判据换向依据见该脚本 `_pick_ln_delta_ref` docstring——v2 用「约束<10%、取最小 ref 保
分辨率」，v3 按任务蓝图明令改为「约束更严 ≤5%、取最大分离度」）。ref<3.0 时 S3 仍撞界
（ref=2.0 时 9.1%），ref=3.0 起最差被试与池化撞界均降到 0.0%，且判别力不掉。

**v3 修掉了 v2 明文自承的「更根本的已知缺陷」**：v2 的 |Δ| 幅度与被试自身 RMSSD 水平
成比例（各被试 |Δ|p90/RMSSD中位 收敛在 0.35–0.49，任何单一 ms 常数必然对一部分被试
失配）。ln 差分对此**按构造免疫**——`ln(a)−ln(b) = ln(a/b)` 只依赖**比值**，与绝对水平
无关。实测跨被试静息 μa 极差从 v2 的 0.293 降到 **0.075**（−74%），验证了这条设计预期
（v2 文献门当轮记录的「相对 Δ 实测 1.8× 分离度」正是这个效应的早期观测，本轮把它
落到 ln 口径下转正，见 `notes/2026-08-31-hrv-v3-litgate.md` 五问结论 #1）。

⚠ **仍是 n=5 下由单个被试（S3）决定的脆弱结论**——与 v2 同款局限未消除，只是脆弱点从
「幅度失配」换成了「撞界约束仍由 S3 一人卡住选参下界」。换一批被试大概率要重标。
"""

_RMSSD_EPSILON_MS: float = 1.0
"""RMSSD / 基线中位数的数值防御下限（ms，**工程假设**，未见文献量级依据）。

ln 差分要求两个操作数都严格为正：`ln(≤0)` 数学上未定义。且成人静息 RMSSD 典型
20–150ms，<1ms 已是提取退化（R 峰检测崩溃/信号全零）而非真实生理值。触发时该窗按
「无证据」处理，返回 None 且**不进基线历史**——早于任何状态写入拦截，原则与 NaN 守卫
相同（坏值一旦进历史即污染后续所有窗的中位数基线）。
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
    """HRV/RMSSD（心率变异性）感知通道：**窗间中位数基线 − 窗内 RMSSD 的 ln 差** → arousal 分量。

    valence 恒 0.0（HRV 对 valence 盲，Kreibig 2010）。

    **度量 v3**（`rmssd_baseline_ln_delta`，2026-08-31 起）：从 ECG/PPG 提取窗内 RMSSD（ms），
    用近 ``baseline_horizon_seconds`` 内各窗 RMSSD 的**中位数**基线，与当前窗做**自然对数差**
    ``Δ = ln(baseline_median) − ln(rmssd_now)``，再对称归一化到 [-1, 1]。⚠ **减法方向与
    EdaChannel 相反**：RMSSD↑ = 迷走张力↑ = 唤醒**↓**，方向与 v2 一致（未变），这是本通道
    唯一的高危处，见 ``_process`` 就近注释。数值防御：``rmssd_now`` 或 ``baseline_median``
    ≤ ``epsilon_ms``（默认 1.0ms，工程假设）时该窗判定为提取退化，返回 None 且不进历史——
    ``ln(≤0)`` 数学上未定义，早于任何状态写入拦截，原则同 NaN 守卫。

    **为什么换成 ln 差（v3 相对 v2 的改造理由，`notes/2026-08-31-hrv-v3-litgate.md`）**：
    v2 明文自承「更根本的已知缺陷」——|Δ| 幅度与被试自身 RMSSD 水平成比例（各被试
    |Δ|p90/RMSSD中位 收敛在 0.35–0.49），任何单一 **ms** 常数必然对一部分被试失配。
    ln 差分对此按构造免疫：``ln(a)−ln(b)=ln(a/b)`` 只依赖**比值**，与绝对水平无关。
    实测（`evals/wesad_hrv_v3_ln_calibration.py`，S2–S6）跨被试静息 μa 极差从 v2 的
    **0.293 降到 0.075**（−74%），验证了这条设计预期；标定同时把最差被试（S3）stress
    撞 |μa|=1 从 v2 的 9.1% 压到 **0.0%**（新约束 ≤5%，比 v2 的 <10% 更严）。

    **σ 交付（`evals/wesad_hrv_v3_residual_sigma.py` + `wesad_hrv_v3_sigma_delivery_checks.py`，
    双锚——自评锚/设计锚，跨会话交付纪律要求同时给出）**：

    | 子集 | σ_D 自评锚 v2→v3 | σ_D 设计锚 v2→v3 |
    | --- | --- | --- |
    | 样本内 S2–S6（n=5，参数标定集） | 0.845 → **0.686** | 0.759 → 0.765（≈打平，n=5 噪声内） |
    | 样本外 S7–S17（n=10） | 1.831 → **1.304** | 1.607 → **0.986**（跌破 1.0） |
    | 全部 S2–S17（n=15） | 1.390 → **1.078** | 1.139 → **0.935**（跌破 1.0） |

    12 组对比里 11 组改善、1 组（样本内设计锚）在 n=5 噪声范围内打平，无一组劣化。
    ⚠ **诚实附注**：自评锚在样本外留一（S7–S17 内留一，n=10）仍**全部 >1.0**（v2 1.310–2.417，
    v3 1.105–1.619）——按 Zero 的 σ≤1.0 判据，"精度而非噪声"这条结论对自评锚**仍不成立**；
    设计锚留一有部分复本跌破 1.0（v2 从未跌破，v3 部分跌破），是本轮**唯一**让判据翻转的
    子集。方向性守恒且更显著：n=15 判别 11/15（与 v2 同），Wilcoxon p=0.0062（v2 0.0128，
    效应量中位 0.146 vs v2 0.110）。None 占比中位 4.7%（与 v2 打平，未劣化）。

    **冷启动返回 None 而非回退固定 ref**：无基线证据时给出的读数必然基于臆断零点，比不给
    更有害（v1 的固定 ref 正是 S3 静息钉死 −1.0 的成因）。⚠ 与 EdaChannel 的冷启动窗口
    **重合**（两侧 horizon×coverage 均为 270s），故会话开始后约 270–300s **两条 physio 流
    同时为 None** → ``merge_physio_priors`` 零命中 → 载荷里**完全没有 physio 家族流**
    （不是流名降级，是流不存在）。这是有意的诚实降级，已作 R5.1-bis 跨仓报备。

    **窗长恒定性（EDA 无此约束）**：RMSSD 随窗内 R-R 间期数变化，窗长一变 ``rmssd_now``
    与历史中位数即**不同口径**、Δ 失去意义 → 调用方须在一个会话内保持投喂窗长恒定，
    变更窗长等同换被试，须先 ``reset()``。取「文档约定」而非运行期检测，依据沿用 v2
    G-WindowLen 实测（窗长/horizon 本轮**未**重新 sweep，见 `_RMSSD_BASELINE_HORIZON_SECONDS`
    与文献门第 3 条——窗长敏感度是 v2 在 ms 口径下测的，ln 差分只改变归一化分母，窗长本身
    对 RMSSD 提取质量的影响不受 v3 改动波及，结论沿用）：{15,30,60}s 下同一 stress 段
    μa 的逐被试极差中位 **0.0156**，暖机后 None 率三档均为 0.00%。主用 60s、建议下界 30s。

    验收（`evals/wesad_hrv_v3_acceptance_gates.py`，**对接本类真实例**的 WESAD 全会话流式
    回放，60s 窗 / 700Hz 胸带 ECG / S2–S6；数字与 `wesad_hrv_v3_ln_calibration.py` 的离线
    参考实现逐位一致，交叉确认两者未分叉）：判别力 4/5（v2 同为 4/5）PASS · G-Drift 中位
    0.106（门 <0.5）PASS · G1' 跨采样率极差中位 0.0064（门 <0.15）PASS · G-Sign 方向性
    中位 Δ=+0.146>0（单侧 Wilcoxon p=0.0625，n=5 功效低仅作报告项）PASS · 冷启动 300s ·
    None 4.7%（不劣于 v2）PASS · 撞界%stress 中位 0.0%（约束 ≤5%）PASS ·
    **跨被试静息 μa 极差 0.075（v2 0.293）** ← 本改造的目标函数 · stress 撞 |μa|=1 全被试
    0.0%（v2 最差被试 9.1%）。

    ⚠ **已知限制（覆盖面很小，报告时勿放大）**：n=5 标定 / n=15 σ 交叉核验，单数据集
    （WESAD）、单设备通路（RespiBAN 胸带 ECG，坐姿实验室）。**腕带 BVP/PPG 完全未测**
    （PPG 峰定位精度远低于 ECG，RMSSD 对此高度敏感，不可假设可迁移）；60s 窗本身已是
    对短时 HRV 经典 5 分钟建议的激进外推，未做与金标准的一致性比对；``LN_DELTA_REF=3.0``
    仍是 n=5 下由单个被试（S3）的撞界约束决定的脆弱结论，换一批被试大概率要重标。

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
                         ``_RMSSD_BASELINE_HORIZON_SECONDS``（⚠ 与 EDA 同值属巧合，非照抄；
                         v3 本轮未重新 sweep，沿用 v2 选定值）。
        ln_delta_ref:    Δ→μa 对称归一化参考（**nats**，自然对数差；不可与 v2 的 ms 常数
                         混用）。默认 3.0，依据见 ``_RMSSD_LN_DELTA_REF``（同样是由单个被试
                         S3 决定的 n=5 脆弱结论，只是脆弱点从「幅度失配」换成了「撞界约束」）。
        epsilon_ms:      RMSSD / 基线中位数的数值防御下限（ms）。默认 1.0，依据见
                         ``_RMSSD_EPSILON_MS``（工程假设：ln(≤0) 未定义 + <1ms 生理不可信）。
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
        ln_delta_ref: float = _RMSSD_LN_DELTA_REF,
        epsilon_ms: float = _RMSSD_EPSILON_MS,
        baseline_min_observations: int = _RMSSD_BASELINE_MIN_OBSERVATIONS,
        baseline_min_coverage_fraction: float = _RMSSD_BASELINE_MIN_COVERAGE,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.signal_source = signal_source
        self.clock = clock
        self.baseline_horizon_seconds = baseline_horizon_seconds
        self.ln_delta_ref = ln_delta_ref
        self.epsilon_ms = epsilon_ms
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
        """延迟 import neurokit2 → 窗内 RMSSD → **ln(窗间中位数基线) − ln(当前)** → 对称归一化。

        pipeline：``nk.ecg_process`` → ``nk.hrv_time`` → RMSSD(ms) → 数值防御（epsilon）→
        ln 基线相减 → clip 到 [-1,1]。

        为什么用 ln 差而不是 v2 的原始 ms 差：v2 明文自承|Δ|幅度与被试自身 RMSSD 水平
        成比例，任何单一 ms 常数必然对一部分被试失配。``ln(a)−ln(b)=ln(a/b)`` 只依赖比值，
        按构造消除这个失配——实测跨被试静息 μa 极差从 v2 的 0.293 降到 0.075。
        标定与验收见类 docstring 与 `evals/wesad_hrv_v3_*` 三件。

        为什么先做数值防御：RMSSD 或基线中位数 ≤ ``epsilon_ms`` 时 ``ln`` 未定义/不可信，
        必须**早于状态写入**拦截（同 NaN 守卫的道理——坏值一旦进历史即污染后续所有窗的
        中位数基线，不只是丢本轮读数）。

        为什么冷启动返回 None 而非回退某个固定 ref：无基线证据时给出的读数必然基于臆断
        零点。旧实现正因此把 S3 静息 89.5% 的窗钉在 −1.0。None 是**有意的诚实降级**
        （与 EdaChannel 冷启动窗口重合时，该轮完全不出 physio 家族流，已跨仓报备）。

        参考：[NeuroKit2 DOI:10.3758/s13428-020-01516-y] ·
              [Kreibig 2010 DOI:10.1016/j.biopsycho.2010.03.010] ·
              [Shaffer & Ginsberg 2017 DOI:10.3389/fpubh.2017.00258]（RMSSD 的 ln 变换惯例）
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
        # 数值防御（v3 新增，早于状态写入同 NaN 守卫）：ln(≤0) 未定义，且 <epsilon_ms 的
        # RMSSD 生理上不可信（提取退化）。同样不进历史——否则污染后续所有窗的基线中位数。
        if rmssd_ms <= self.epsilon_ms:
            logger.warning(
                "HrvChannel: RMSSD 退化（%.3fms ≤ epsilon %.3fms），本轮跳过",
                rmssd_ms,
                self.epsilon_ms,
            )
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
        # 基线数值防御（同上）：理论上历史里的每个值构造时都已过 epsilon 门，中位数不可能
        # ≤epsilon，但仍显式检查——防御性编程不依赖"不可能发生"的隐含前提。
        if baseline <= self.epsilon_ms:
            logger.warning(
                "HrvChannel: 基线退化（中位数 %.3fms ≤ epsilon %.3fms），本轮跳过",
                baseline,
                self.epsilon_ms,
            )
            return None
        # ⚠⚠ 本行是本次改造唯一的高危处：**减法方向与 EdaChannel 相反，不可照抄**。
        #     EDA：SCL↑ = 交感激活↑ = 唤醒↑     → Δ = 当前 − 基线
        #     HRV：RMSSD↑ = 迷走张力↑ = 唤醒**↓** → Δ = ln(基线) − ln(当前)（本行，v3）
        # 写成 `ln(rmssd_ms) - ln(baseline)` 即逐字复刻 EDA v1 的系统性反号病：那次判别力
        # 1/5、在合成信号上长期全绿，直到 WESAD 真被试回放才暴露，整条度量随后被删除。
        # 方向依据：[Kreibig 2010]（应激下迷走撤退 → RMSSD 降、唤醒升）+ [Kim et al. 2018
        # meta-analysis DOI:10.30773/pi.2017.08.17]（37 篇独立佐证）+ WESAD 实测（v3：
        # n=15 判别 11/15，Wilcoxon p=0.0062，效应量中位 +0.146）。三重守卫：本注释 ·
        # 单测 TestHrvSignDirection（反号必红）· `evals/wesad_hrv_v3_acceptance_gates.py`
        # 的 G-Sign 门（真被试同实例内方向性）。
        delta = math.log(baseline) - math.log(rmssd_ms)
        mu_a = _symmetric_normalize(delta, self.ln_delta_ref)
        mu_v = 0.0  # HRV 对 valence 盲（Kreibig 2010）

        return build_recommended_prior(
            modality=self.name,
            mu=(mu_v, mu_a),
            kind=ModalityKind.PHYSIO,
        )
