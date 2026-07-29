"""HrvChannel 唤醒度量（rmssd_baseline_delta）单测——HRV 基线相减改造 T14。

覆盖：冷启动双门 · Δ 语义与跨被试可比 · **符号方向（与 EdaChannel 相反）** · 单侧地板
已删除 · horizon 裁剪 · NaN 早于状态写入 · reset · clock 注入确定性 · 并发下
baseline_history 线程安全 · 两流同时冷启动的载荷形状 · 常量与标定选值一致。

⚠ 通道级契约（先验形状、signal_source、Hub 集成、默认关）在
`tests/mcp/test_zero_physio_channel.py`；真 NeuroKit2 路径在
`tests/mcp/test_zero_physio_real.py`；本文件只测**度量语义**，故用假 nk 把「窗内 RMSSD」
变成可直接指定的量——RMSSD 的**提取**质量不在本文件范围内。

⚠ **任何常量标定、任何「度量有效」的结论都不在本文件**：合成/构造信号上全绿不构成证据
（EDA v1 正是合成全绿、接真数据后反号）。度量有效性见
`evals/wesad_hrv_v2_acceptance_gates.py`（WESAD 真被试全会话流式回放，对接本通道真实例）。

设计与选参依据：`evals/wesad_hrv_baseline_delta_calibration.py`（horizon 持续比拐点 →
1800s、delta_ref 最差被试撞界 <10% → 100ms、窗长下界 → 30s 建议/60s 主用）。
改造的正当理由是**跨被试可比**（静息读数极差 1.687 → 0.293），不是判别力/抗漂移
（二者都没赢过现状）——见 `HrvChannel` 类 docstring。
"""

from __future__ import annotations

import asyncio
import itertools
import threading
import time
from collections import deque
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.agents.models.zero_affect import ModalityPrior
from src.mcp.zero.channels.physio_channel import (
    _RMSSD_BASELINE_HORIZON_SECONDS,
    _RMSSD_BASELINE_MIN_COVERAGE,
    _RMSSD_BASELINE_MIN_OBSERVATIONS,
    _RMSSD_DELTA_REF_MS,
    EdaChannel,
    HrvChannel,
)
from src.mcp.zero.external_priors import build_external_priors_override, is_physio_stream


class FakeClock:
    """可注入时钟：手动推进，使时间语义在测试里完全确定（不用墙钟）。"""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeNeurokit:
    """假 neurokit2：把「窗内 RMSSD(ms)」变成测试可直接指定的量。

    本文件测的是**基线相减度量本身**（Δ 语义、冷启动、裁剪、符号方向）。用假 nk 把 RMSSD
    作为直接输入，断言就不必依赖 nk 版本间的数值 offset，也不必为构造某个 RMSSD 去反解
    合成 ECG 的参数。真 nk 路径由 `test_zero_physio_real.py` 覆盖。

    ⚠ 它替换的只是「从 ECG 得到一个 RMSSD 标量」这一步；`HrvChannel._process` 的其余部分
    （NaN 守卫、裁剪、快照、冷启动双门、中位数、减法方向、归一化、出线）全是真产品码。
    """

    def __init__(self, rmssd_ms: float) -> None:
        self.rmssd_ms = rmssd_ms

    def ecg_process(
        self, signal: Any, sampling_rate: int | None = None
    ) -> tuple[Any, dict[str, Any]]:
        return pd.DataFrame({"ECG_R_Peaks": [0] * 5}), {}

    def hrv_time(
        self, signals: Any, sampling_rate: int | None = None, show: bool = False
    ) -> pd.DataFrame:
        return pd.DataFrame({"HRV_RMSSD": [self.rmssd_ms]})


_ECG_SIGNAL: dict[str, Any] = {
    "ecg_or_ppg": np.zeros(256 * 10, dtype=np.float64),
    "sampling_rate": 256,
}
"""喂给通道的信号占位——内容不参与运算（RMSSD 由 `_FakeNeurokit` 指定），但键名与形状真实。"""


def _hrv_channel(clock: FakeClock, **kwargs: object) -> HrvChannel:
    """构造测试用通道：horizon 缩到 1000s 让覆盖率门（150s）在少数几步内可跨过。"""
    params: dict[str, object] = {
        "clock": clock,
        "baseline_horizon_seconds": 1000.0,
        "delta_ref_ms": 100.0,
        "baseline_min_observations": 2,
        "baseline_min_coverage_fraction": 0.15,
    }
    params.update(kwargs)
    return HrvChannel(**params)  # type: ignore[arg-type]


def _sense(channel: HrvChannel, rmssd_ms: float) -> ModalityPrior | None:
    """跑一次 sense，本窗的 RMSSD 由入参指定。"""
    with patch.dict("sys.modules", {"neurokit2": _FakeNeurokit(rmssd_ms)}):
        return asyncio.run(channel.sense(_ECG_SIGNAL))


def _warm(channel: HrvChannel, clock: FakeClock, rmssd_ms: float, rounds: int = 4) -> None:
    """喂若干窗把通道推过冷启动双门（基线 = 该 RMSSD 水平）。"""
    for _ in range(rounds):
        _sense(channel, rmssd_ms)
        clock.advance(100.0)


@pytest.fixture(autouse=True)
def _enable_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    """通道 flag 默认关；本文件统一开启。"""
    monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")


class TestHrvSinglePathNoMetricSwitch:
    """单路径直落（A 案）：不留任何 A/B 开关。

    HRV 改造**没有**走 EDA v2 那条「枚举 env + 构造期读取 → 翻默认 → 退役」的三段路：
    v1 只有两行公式，对照实验完全可以在 evals 里用纯函数做（EDA 那边的 `control_arm()`
    也从未依赖 src 开关）。保留双路径的收益为负——「默认值 / env / 显式入参」三者的
    优先级会成为长期误配面（EDA 任务 9 退役 v1 时立的档）。本类防的是它被加回来。
    """

    def test_default_construction_allocates_baseline_state(self) -> None:
        """默认构造即分配基线状态且为空（冷启动起点）。"""
        channel = HrvChannel(sampling_rate=256)
        assert len(channel.baseline_history) == 0

    def test_no_metric_switch_parameter(self) -> None:
        """构造器不接受度量选择入参（单一路径，无 A/B 分支）。"""
        with pytest.raises(TypeError):
            HrvChannel(arousal_metric="fixed_ref_v1")  # type: ignore[call-arg]

    def test_no_metric_env_is_consulted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """凭空设一个度量开关 env 不得改变行为（HRV 从未有过该 env，本例防它被加回来）。"""
        monkeypatch.setenv("ZERO_HRV_AROUSAL_METRIC", "fixed_ref_v1")
        clock = FakeClock()
        channel = _hrv_channel(clock)
        _warm(channel, clock, 60.0)
        reading = _sense(channel, 40.0)
        assert reading is not None, "冒牌 env 不应把通道切成别的度量"
        assert reading.mu[1] == pytest.approx(0.2, abs=1e-9), "读数应是 Δ 语义（60−40=+20ms）"

    def test_default_clock_is_monotonic(self) -> None:
        """默认时钟是 `time.monotonic`（墙钟不可回退，且与 EdaChannel 同源）。"""
        assert HrvChannel().clock is time.monotonic


class TestHrvColdStart:
    """冷启动双门：观测数不足**或**时间跨度不足，均返回 None（无基线证据→无证据）。"""

    def test_first_window_returns_none(self) -> None:
        clock = FakeClock()
        channel = _hrv_channel(clock)
        assert _sense(channel, 40.0) is None

    def test_cold_start_still_records_history(self) -> None:
        """返回 None 的窗**仍要进历史**——否则通道永远暖不起来（顺序不变式）。"""
        clock = FakeClock()
        channel = _hrv_channel(clock)
        _sense(channel, 40.0)
        assert len(channel.baseline_history) == 1

    def test_min_observations_gate(self) -> None:
        """跨度已够但观测数不足 → 仍 None；攒够第 5 条历史后才出读数。

        计数语义：第 n 次调用看到的快照**不含本次**，故需 5 次调用才攒下 5 条，
        第 6 次才满足 min_observations=5。
        """
        clock = FakeClock()
        channel = _hrv_channel(clock, baseline_min_observations=5)
        for _ in range(5):
            assert _sense(channel, 40.0) is None
            clock.advance(200.0)  # 跨度远超 1000×0.15=150s，只有观测数这道门在起作用
        assert len(channel.baseline_history) == 5
        assert _sense(channel, 40.0) is not None

    def test_coverage_fraction_gate(self) -> None:
        """观测数已够但时间跨度不足 → 仍 None。"""
        clock = FakeClock()
        channel = _hrv_channel(clock)  # 需跨度 ≥ 1000×0.15 = 150s
        for _ in range(5):
            assert _sense(channel, 40.0) is None
            clock.advance(10.0)  # 累计仅 50s
        clock.advance(200.0)
        assert _sense(channel, 40.0) is not None


class TestHrvDeltaSemantics:
    """Δ = 基线 − 当前 的取值语义（符号方向另见 `TestHrvSignDirection`）。"""

    def test_flat_signal_reads_neutral(self) -> None:
        """RMSSD 不变 → Δ=0 → μa=0.0（中性），**不是 −1.0**。

        v1 公式 `_linear_normalize(max(0, 100−rmssd), 100)` 在 RMSSD=100ms 时才给 0，
        在 RMSSD≥100ms 时一律给 −1.0——「没有变化」被读成「极度低唤醒」正是它的病。
        """
        clock = FakeClock()
        channel = _hrv_channel(clock)
        _warm(channel, clock, 60.0)
        prior = _sense(channel, 60.0)
        assert prior is not None
        assert prior.mu[1] == pytest.approx(0.0, abs=1e-9)

    def test_rmssd_drop_is_positive_arousal(self) -> None:
        """RMSSD **下降**（迷走撤退）→ μa **正**：60→30ms，Δ=+30，ref=100 → +0.3。"""
        clock = FakeClock()
        channel = _hrv_channel(clock)
        _warm(channel, clock, 60.0)
        prior = _sense(channel, 30.0)
        assert prior is not None
        assert prior.mu[1] == pytest.approx(0.3, abs=1e-9)

    def test_rmssd_rise_is_negative_arousal(self) -> None:
        """RMSSD **上升**（迷走增强）→ μa **负**：60→90ms，Δ=−30 → −0.3。"""
        clock = FakeClock()
        channel = _hrv_channel(clock)
        _warm(channel, clock, 60.0)
        prior = _sense(channel, 90.0)
        assert prior is not None
        assert prior.mu[1] == pytest.approx(-0.3, abs=1e-9)

    def test_saturates_symmetrically_not_floored(self) -> None:
        """两端都钳到 ±1（对称），不是一端被地板压平。"""
        clock = FakeClock()
        channel = _hrv_channel(clock)
        _warm(channel, clock, 200.0)
        assert (up := _sense(channel, 20.0)) is not None
        clock.advance(100.0)
        assert (down := _sense(channel, 400.0)) is not None
        assert up.mu[1] == pytest.approx(1.0)
        assert down.mu[1] == pytest.approx(-1.0)

    def test_absolute_level_does_not_matter(self) -> None:
        """判别性：跨被试可比的核心——同样的 Δ 在不同 RMSSD 绝对水平上读数**相同**。

        直接覆盖 v1 的饱和缺陷：固定 ref=100ms 下，RMSSD 中位 125ms 的 S3 有 89.5% 的
        静息窗钉在 μa=−1.0，而 RMSSD 中位 27ms 的 S5 静息读 +0.41——同为「坐着休息」，
        两人读数差 1.4。基线相减买到的就是「μa=0 对每个被试同义」。
        """
        readings = []
        for level in (30.0, 60.0, 150.0):
            clock = FakeClock()
            channel = _hrv_channel(clock)
            _warm(channel, clock, level)
            prior = _sense(channel, level - 10.0)  # 同样 Δ=+10ms
            assert prior is not None
            readings.append(prior.mu[1])
        assert max(readings) - min(readings) == pytest.approx(0.0, abs=1e-9)
        assert readings[0] == pytest.approx(0.1, abs=1e-9)

    def test_valence_blind(self) -> None:
        """μv 恒 0.0（HRV 对 valence 盲）——跨仓承诺「physio 流 μv≡0」的通道侧。"""
        clock = FakeClock()
        channel = _hrv_channel(clock)
        _warm(channel, clock, 60.0)
        prior = _sense(channel, 30.0)
        assert prior is not None
        assert prior.mu[0] == 0.0

    def test_baseline_is_median_not_mean(self) -> None:
        """基线取**中位数**：单个伪迹窗（RMSSD 爆表）不得整体抬走基线。

        判别性：把产品码的 `np.median` 换成 `np.mean`，本例读数从 +0.1 变成 +0.725（基线被
        单个 310ms 伪迹窗从 60 抬到 122.5），必红——已逐变异实证。
        伪迹是真实存在的——实测 60s 窗有 4.1% 的窗 RMSSD>150ms（最大 395ms），且它们是
        **有限值**，NaN 守卫拦不住，一定会进基线历史。
        """
        clock = FakeClock()
        channel = _hrv_channel(clock)
        for value in (60.0, 60.0, 60.0, 310.0):  # 末窗为伪迹；均值 122.5 vs 中位 60
            _sense(channel, value)
            clock.advance(100.0)
        prior = _sense(channel, 50.0)
        assert prior is not None
        assert prior.mu[1] == pytest.approx(0.1, abs=1e-9)

    def test_horizon_prunes_old_baseline(self) -> None:
        """超出 horizon 的历史被按**真实秒数**裁掉（非样本数/调用数）。"""
        clock = FakeClock()
        channel = _hrv_channel(clock, baseline_horizon_seconds=300.0)
        for _ in range(3):
            _sense(channel, 60.0)
            clock.advance(50.0)
        assert len(channel.baseline_history) == 3
        clock.advance(500.0)  # 全部超龄
        _sense(channel, 40.0)
        assert len(channel.baseline_history) == 1  # 仅剩本次

    def test_pruned_history_changes_the_reading(self) -> None:
        """裁剪不只是清内存——它改变基线本身（否则「裁掉了」这件事无从观测）。

        正控：同样一串输入，horizon 大到不裁剪时基线含旧值、读数不同。
        """
        readings = []
        for horizon in (300.0, 2000.0):
            clock = FakeClock()
            channel = _hrv_channel(clock, baseline_horizon_seconds=horizon)
            for _ in range(3):
                _sense(channel, 200.0)  # 旧基线：高 RMSSD
                clock.advance(50.0)
            clock.advance(400.0)  # horizon=300 时旧值全超龄；horizon=5000 时全保留
            for _ in range(2):
                _sense(channel, 60.0)  # 新基线：低 RMSSD
                clock.advance(50.0)
            prior = _sense(channel, 50.0)
            assert prior is not None
            readings.append(prior.mu[1])
        assert readings[0] != pytest.approx(readings[1]), "裁剪与不裁剪读数相同 → 裁剪未生效"
        assert readings[0] == pytest.approx(0.1, abs=1e-9)  # 基线=60（旧值已裁）


class TestHrvSignDirection:
    """⚠ **本次改造唯一的高危项专项**：Δ 的减法方向与 EdaChannel **相反**。

    EDA：SCL↑ = 交感激活↑ = 唤醒↑     → Δ = 当前 − 基线
    HRV：RMSSD↑ = 迷走张力↑ = 唤醒**↓** → Δ = 基线 − 当前

    照抄 EdaChannel 那一行即逐字复刻 EDA v1 的系统性反号病（判别力 1/5、合成信号上长期
    全绿、直到 WESAD 真被试回放才暴露，整条度量随后被删）。

    判别力已逐变异实证：把产品码 `baseline - rmssd_ms` 改成 `rmssd_ms - baseline`，
    本类**四条全红**（含跨通道那条）。
    """

    def test_higher_metric_means_lower_arousal(self) -> None:
        """指标本身升高 → μa 下降（与 EDA 的「指标升高 → μa 上升」相反）。"""
        clock = FakeClock()
        channel = _hrv_channel(clock)
        _warm(channel, clock, 60.0)
        low = _sense(channel, 90.0)  # RMSSD 高
        clock.advance(100.0)
        high = _sense(channel, 30.0)  # RMSSD 低
        assert low is not None and high is not None
        assert high.mu[1] > low.mu[1]
        assert high.mu[1] > 0.0 > low.mu[1], "高/低 RMSSD 未落在中性零点两侧"

    def test_monotone_decreasing_in_rmssd(self) -> None:
        """μa 对 RMSSD **严格单调递减**（不是「大体上」——同一基线下逐点比较）。"""
        clock = FakeClock()
        channel = _hrv_channel(clock)
        _warm(channel, clock, 80.0)
        readings = []
        for rmssd in (20.0, 40.0, 60.0, 80.0, 100.0, 120.0):
            prior = _sense(channel, rmssd)
            assert prior is not None
            readings.append(prior.mu[1])
            clock.advance(100.0)
            # 每步都把基线拉回 80ms，使各点可比（否则基线被前一次读数带走）
            _sense(channel, 80.0)
            clock.advance(100.0)
        assert all(a > b for a, b in itertools.pairwise(readings)), (
            f"μa 对 RMSSD 非严格递减：{readings}"
        )

    def test_opposite_sign_to_eda_channel_on_same_numbers(self) -> None:
        """**跨通道反号守卫**：同一组数字喂两条通道，μa 必须**符号相反**。

        这是「照抄 EdaChannel 那一行」这一具体错误的直接探针——两通道结构完全同构、
        只有减法方向不同，故同样的 (基线, 当前) 必须给出互为相反数的读数。
        任一侧的方向被改成与另一侧一致，本例立刻红。
        （量纲刻意对齐：EDA 用 ref=1.0μS 喂 Δ=+0.3；HRV 用 ref=100ms 喂 Δ=−30ms。）
        """
        eda_clock, hrv_clock = FakeClock(), FakeClock()
        eda = EdaChannel(
            clock=eda_clock,
            baseline_horizon_seconds=1000.0,
            delta_ref_us=1.0,
            baseline_min_observations=2,
            baseline_min_coverage_fraction=0.15,
        )
        hrv = _hrv_channel(hrv_clock)

        for _ in range(4):
            asyncio.run(eda.sense({"eda": np.full(60, 5.0), "sampling_rate": 4}))
            eda_clock.advance(100.0)
        _warm(hrv, hrv_clock, 60.0)

        # 两侧都让「当前 = 基线 + 0.3×ref」：EDA 5.0→5.3μS；HRV 60→90ms
        eda_prior = asyncio.run(eda.sense({"eda": np.full(60, 5.3), "sampling_rate": 4}))
        hrv_prior = _sense(hrv, 90.0)
        assert eda_prior is not None and hrv_prior is not None
        assert eda_prior.mu[1] == pytest.approx(0.3, abs=1e-9)
        assert hrv_prior.mu[1] == pytest.approx(-0.3, abs=1e-9)
        assert eda_prior.mu[1] * hrv_prior.mu[1] < 0, (
            f"两通道对「指标高于基线」给出同号读数（EDA {eda_prior.mu[1]}, "
            f"HRV {hrv_prior.mu[1]}）——HRV 很可能照抄了 EDA 的减法方向"
        )

    def test_no_single_sided_floor_remains(self) -> None:
        """v1 的 `max(0.0, ref − rmssd)` 单侧地板必须已删干净。

        地板的可观测后果：**所有**高于基线的 RMSSD 被压成同一个读数（v1 里是 −1.0）。
        故本例喂两个不同的「高于基线」水平，要求读数**互不相同**且都严格 >−1（未饱和）。
        判别力已实证：在产品码里恢复 `delta = max(0.0, delta)` 或对 μa 取
        `max(-1.0, ...)` 之类的单侧处理，本例即红。
        """
        clock = FakeClock()
        channel = _hrv_channel(clock)
        _warm(channel, clock, 60.0)
        mild = _sense(channel, 80.0)
        clock.advance(100.0)
        strong = _sense(channel, 100.0)
        assert mild is not None and strong is not None
        assert mild.mu[1] == pytest.approx(-0.2, abs=1e-9)
        assert strong.mu[1] == pytest.approx(-0.4, abs=1e-9)
        assert mild.mu[1] != strong.mu[1], "两个不同的低唤醒水平读出同一个数 → 地板还在"
        assert -1.0 < strong.mu[1] < mild.mu[1] < 0.0


class TestHrvNaNGuard:
    def test_nan_returns_none_and_does_not_pollute_history(self) -> None:
        """NaN 守卫必须**早于**任何状态写入——否则坏值进历史，`np.median` 直接产 NaN。"""
        clock = FakeClock()
        channel = _hrv_channel(clock)
        _warm(channel, clock, 60.0)
        size_before = len(channel.baseline_history)
        assert size_before > 0, "预热未生效，下面的『长度未变』断言会退化为恒真"

        assert _sense(channel, float("nan")) is None
        assert len(channel.baseline_history) == size_before  # 未写入
        assert all(np.isfinite(v) for _, v in channel.baseline_history)

        clock.advance(100.0)
        prior = _sense(channel, 30.0)
        assert prior is not None
        assert prior.mu[1] == pytest.approx(0.3, abs=1e-9)  # 基线未被 NaN 破坏

    def test_inf_is_also_rejected(self) -> None:
        """inf 同样是「非有限」→ 无证据（`np.isfinite` 而非 `not np.isnan`）。"""
        clock = FakeClock()
        channel = _hrv_channel(clock)
        _warm(channel, clock, 60.0)
        size_before = len(channel.baseline_history)
        assert _sense(channel, float("inf")) is None
        assert len(channel.baseline_history) == size_before


class TestHrvReset:
    def test_reset_clears_baseline_history(self) -> None:
        clock = FakeClock()
        channel = _hrv_channel(clock)
        _warm(channel, clock, 60.0)
        assert len(channel.baseline_history) > 0

        channel.reset()
        assert len(channel.baseline_history) == 0
        assert _sense(channel, 60.0) is None  # 回到冷启动

    def test_missing_reset_would_carry_baseline_across_subjects(self) -> None:
        """判别性正控：**不** reset 就换被试，新被试的读数由旧被试基线决定（污染实证）。

        本例把「reset 契约的必要性」变成可观测量：同样喂 30ms 的新被试窗，
        沿用旧被试 200ms 基线时读到 +1.0（饱和），reset 后则回到冷启动 None。
        删掉 `HrvChannel.reset` 会让 `PerceptionHub.reset_all()` 对 HRV **静默无效**——
        无异常、无日志，只有读数悄悄错，故必须有守卫直接盯住它。
        """
        clock = FakeClock()
        channel = _hrv_channel(clock)
        _warm(channel, clock, 200.0)  # 旧被试：高 RMSSD 水平

        polluted = _sense(channel, 30.0)  # 新被试第一窗，未 reset
        assert polluted is not None
        assert polluted.mu[1] == pytest.approx(1.0), "污染态应把新被试读数推到饱和"

        channel.reset()
        clock.advance(100.0)
        assert _sense(channel, 30.0) is None, "reset 后应回到冷启动（不给臆断读数）"

    def test_baseline_history_is_the_only_mutable_state(self) -> None:
        """基线历史是通道**唯一**的跨调用可变状态——reset 清空它即回到全新实例等价态。

        判别性：若日后新增第二份历史（如滑窗缓存/伪迹计数）而 reset 漏清，本例会捕到差异。
        """
        clock = FakeClock()
        channel = _hrv_channel(clock)
        _warm(channel, clock, 60.0)
        channel.reset()

        mutable = {
            key: value
            for key, value in vars(channel).items()
            if isinstance(value, (list, dict, set, deque))
        }
        assert set(mutable) == {"baseline_history"}, f"出现未被 reset 覆盖的可变状态：{mutable}"
        assert len(channel.baseline_history) == 0


class TestHrvClockInjection:
    def test_readings_are_deterministic_under_injected_clock(self) -> None:
        """同一投喂序列 + 同一注入时钟 → 逐位相同的读数（回放/单测不依赖墙钟）。"""
        sequence = [60.0, 55.0, 70.0, 40.0, 90.0, 30.0]

        def run() -> list[float | None]:
            clock = FakeClock()
            channel = _hrv_channel(clock)
            out: list[float | None] = []
            for value in sequence:
                prior = _sense(channel, value)
                out.append(None if prior is None else prior.mu[1])
                clock.advance(100.0)
            return out

        first, second = run(), run()
        assert first == second
        assert any(v is not None for v in first), "全 None 会让上面的相等断言恒真"


class TestHrvConcurrencySafety:
    """并发不变式守卫（诚实定位：非「锁必要性」判别，同 EdaChannel 侧）。

    `_process` 经 `asyncio.to_thread` 在线程池执行，同实例并发 collect 会并发读改
    `baseline_history`；本例验证并发投喂后历史不撕裂、条目数守恒。
    """

    def test_concurrent_sense_keeps_history_consistent(self) -> None:
        clock = FakeClock()
        channel = _hrv_channel(clock, baseline_horizon_seconds=1e9)
        errors: list[BaseException] = []

        with patch.dict("sys.modules", {"neurokit2": _FakeNeurokit(60.0)}):

            def worker() -> None:
                try:
                    for _ in range(20):
                        asyncio.run(channel.sense(_ECG_SIGNAL))
                except BaseException as exc:  # noqa: BLE001 - 线程内异常须带回主线程
                    errors.append(exc)

            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert not errors
        assert len(channel.baseline_history) == 80  # 4×20，无丢失无重复
        assert all(np.isfinite(v) for _, v in channel.baseline_history)


class TestBothPhysioStreamsColdStart:
    """R5.1-bis 的本仓侧实证：两条 physio 流同时冷启动 → 载荷里**完全没有** physio 流。

    这不是故障，是「无基线证据不臆造读数」在两条流上同时发生的诚实降级（会话开头约
    270–300s）。改造前只有「EDA None → 降级为裸 hrv/rmssd」一态；改造后 HRV 也会 None，
    于是多出「两条同时 None」这一全新形状——已作跨仓件报备，本例把它钉成可回归的事实。
    """

    def test_no_physio_stream_when_both_cold(self) -> None:
        eda_clock, hrv_clock = FakeClock(), FakeClock()
        eda = EdaChannel(clock=eda_clock)
        hrv = HrvChannel(clock=hrv_clock)

        eda_prior = asyncio.run(eda.sense({"eda": np.full(60, 5.0), "sampling_rate": 4}))
        hrv_prior = _sense(hrv, 60.0)
        # 载荷由**通道实际产出**驱动（不是手写 []），否则断言退化成「我断言我刚传进去的空表」。
        # ⚠ 过滤必须写在下面那条 assert **之前**：assert 会把两个变量在类型层面收窄成
        # None，之后再写这个推导式，mypy 视角下它恒为空列表（静态死代码），
        # 与「由实际产出驱动」的立意相悖。
        collected = [p for p in (eda_prior, hrv_prior) if p is not None]
        assert eda_prior is None and hrv_prior is None, "冷启动应两条都 None"

        payload = build_external_priors_override(collected)["external_priors"]
        assert not [s for s in payload if is_physio_stream(s[0])]
        assert payload == []

    def test_single_warm_stream_still_ships(self) -> None:
        """正控：只要有一条暖了，它就照常出流（否则上面的「空」断言不可解读）。"""
        hrv_clock = FakeClock()
        hrv = _hrv_channel(hrv_clock)
        _warm(hrv, hrv_clock, 60.0)
        prior = _sense(hrv, 30.0)
        assert prior is not None

        payload = build_external_priors_override([prior])["external_priors"]
        physio = [s for s in payload if is_physio_stream(s[0])]
        assert [s[0] for s in physio] == ["hrv/rmssd"]
        assert physio[0][1][0] == 0.0, "physio 流出线 μv 必须恒 0"


class TestHrvConstantsMatchCalibration:
    """**四个** HRV 常量须与 WESAD 标定选定值一致——改动即须重跑标定（防静默漂移）。

    同时断言**模块常量**与**构造后实例属性**：只 pin 模块常量会漏掉「默认值绑错常量」
    这一类改动（EDA 那次 code-review WARN #4 的教训——类名承诺多少就 pin 多少）。

    ⚠ horizon 与 EDA 的 `_SCL_BASELINE_HORIZON_SECONDS` 同为 1800.0 是**独立推导后的
    巧合**（HRV 目标函数是持续比拐点），不是照抄——两组常量刻意分家，改一处不得静默
    影响另一通道。`delta_ref` 就完全不同（100ms vs 1.0μS，量纲不可比）。
    """

    def test_horizon_pinned(self) -> None:
        assert _RMSSD_BASELINE_HORIZON_SECONDS == 1800.0
        assert HrvChannel().baseline_horizon_seconds == 1800.0

    def test_delta_ref_pinned(self) -> None:
        assert _RMSSD_DELTA_REF_MS == 100.0
        assert HrvChannel().delta_ref_ms == 100.0

    def test_min_observations_pinned(self) -> None:
        assert _RMSSD_BASELINE_MIN_OBSERVATIONS == 2
        assert HrvChannel().baseline_min_observations == 2

    def test_min_coverage_pinned(self) -> None:
        assert _RMSSD_BASELINE_MIN_COVERAGE == 0.15
        assert HrvChannel().baseline_min_coverage_fraction == 0.15

    def test_constants_are_not_shared_with_eda(self) -> None:
        """两组常量必须是各自独立的对象/取值——`delta_ref` 同值即说明有人共用了常量。"""
        from src.mcp.zero.channels.physio_channel import _SCL_DELTA_REF_US

        assert _RMSSD_DELTA_REF_MS != _SCL_DELTA_REF_US
        assert HrvChannel().delta_ref_ms != EdaChannel().delta_ref_us
