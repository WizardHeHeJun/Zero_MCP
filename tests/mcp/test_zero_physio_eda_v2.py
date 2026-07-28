"""EdaChannel v2（scl_baseline_delta）单测——蓝图任务 6。

覆盖：默认零回归 · env 解析 · NaN 早于状态写入 · reset 清双份状态 · 冷启动双门 ·
对称归一化边界 · clock 注入确定性 · 并发下 baseline_history 线程安全 · 不依赖 neurokit2。

设计与选参依据：`notes/2026-07-28-eda-metric-redesign-blueprint.md` §2.1、
`notes/2026-07-28-eda-v2-probe-p0-p2-results.md` §10。
"""

from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest

from src.mcp.zero.channels.physio_channel import (
    _SCL_BASELINE_HORIZON_SECONDS,
    _SCL_BASELINE_MIN_COVERAGE,
    _SCL_BASELINE_MIN_OBSERVATIONS,
    _SCL_DELTA_REF_US,
    EdaChannel,
    _symmetric_normalize,
)


class FakeClock:
    """可注入时钟：手动推进，使 v2 的时间语义在测试里完全确定（不用墙钟）。"""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _flat_signal(level: float, size: int = 60) -> dict[str, object]:
    return {"eda": np.full(size, level, dtype=np.float64), "sampling_rate": 4}


def _v2_channel(clock: FakeClock, **kwargs: object) -> EdaChannel:
    params: dict[str, object] = {
        "arousal_metric": "scl_baseline_delta_v2",
        "clock": clock,
        "baseline_horizon_seconds": 1000.0,
        "delta_ref_us": 1.0,
        "baseline_min_observations": 2,
        "baseline_min_coverage_fraction": 0.15,
    }
    params.update(kwargs)
    return EdaChannel(**params)  # type: ignore[arg-type]


def _sense(channel: EdaChannel, signal: dict[str, object]):
    return asyncio.run(channel.sense(signal))  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _enable_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    """通道 flag 默认关；v2 测试统一开启，并清掉度量 env 避免宿主污染。"""
    monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
    monkeypatch.delenv("ZERO_EDA_AROUSAL_METRIC", raising=False)


class TestSymmetricNormalize:
    """对称归一化：与 `_linear_normalize` **不可互换**（零输入必须映到 0.0 而非 -1.0）。"""

    def test_zero_maps_to_neutral(self) -> None:
        assert _symmetric_normalize(0.0, 1.0) == 0.0

    def test_sign_preserved_and_clipped(self) -> None:
        assert _symmetric_normalize(0.5, 1.0) == pytest.approx(0.5)
        assert _symmetric_normalize(-0.5, 1.0) == pytest.approx(-0.5)
        assert _symmetric_normalize(10.0, 1.0) == 1.0
        assert _symmetric_normalize(-10.0, 1.0) == -1.0

    def test_ref_scales(self) -> None:
        assert _symmetric_normalize(1.0, 2.0) == pytest.approx(0.5)


class TestDefaultIsNowV2:
    """⚠ **蓝图任务 8 已翻默认值**：默认从 v1 改为 v2。

    翻转依据见 `_DEFAULT_AROUSAL_METRIC` docstring（验收门全 PASS + 当前爆炸半径为零）。
    v1 代码路径完整保留，把该常量改回 `"scr_amplitude_v1"` 即一键回滚。
    """

    def test_default_metric_is_v2(self) -> None:
        assert EdaChannel().arousal_metric == "scl_baseline_delta_v2"

    def test_v1_still_selectable(self) -> None:
        """v1 未被删除（蓝图任务 9 未执行），显式指定仍可用——回滚路径存在。"""
        channel = EdaChannel(arousal_metric="scr_amplitude_v1")
        assert channel.arousal_metric == "scr_amplitude_v1"

    def test_default_construction_allocates_v2_state(self) -> None:
        """默认构造即分配 v2 状态且为空（冷启动起点）。"""
        channel = EdaChannel(sampling_rate=4)
        assert channel.arousal_metric == "scl_baseline_delta_v2"
        assert len(channel.baseline_history) == 0


class TestArousalMetricEnvResolution:
    def test_env_selects_v1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """env 可把默认（现为 v2）覆盖回 v1——运维层的回滚开关。"""
        monkeypatch.setenv("ZERO_EDA_AROUSAL_METRIC", "scr_amplitude_v1")
        assert EdaChannel().arousal_metric == "scr_amplitude_v1"

    def test_explicit_argument_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ZERO_EDA_AROUSAL_METRIC", "scr_amplitude_v1")
        assert (
            EdaChannel(arousal_metric="scl_baseline_delta_v2").arousal_metric
            == "scl_baseline_delta_v2"
        )

    def test_illegal_env_falls_back_to_default_with_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """非法 env 回退**当前默认**（现为 v2），并告警——不 raise，保优雅回退。"""
        monkeypatch.setenv("ZERO_EDA_AROUSAL_METRIC", "scl_baseline_delta_v3")
        with caplog.at_level("WARNING"):
            assert EdaChannel().arousal_metric == "scl_baseline_delta_v2"
        assert "ZERO_EDA_AROUSAL_METRIC" in caplog.text

    def test_env_read_once_at_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """构造期一次性解析：构造后改 env 不得影响已有实例（有状态量不可运行中切换）。"""
        monkeypatch.setenv("ZERO_EDA_AROUSAL_METRIC", "scr_amplitude_v1")
        channel = EdaChannel()
        assert channel.arousal_metric == "scr_amplitude_v1"
        monkeypatch.setenv("ZERO_EDA_AROUSAL_METRIC", "scl_baseline_delta_v2")
        assert channel.arousal_metric == "scr_amplitude_v1"  # 不受运行中变更影响


class TestV2ColdStart:
    """冷启动双门：观测数不足**或**时间跨度不足，均返回 None（无基线证据→无证据）。"""

    def test_first_window_returns_none(self) -> None:
        clock = FakeClock()
        channel = _v2_channel(clock)
        assert _sense(channel, _flat_signal(1.0)) is None

    def test_min_observations_gate(self) -> None:
        """跨度已够但观测数不足 → 仍 None；攒够第 5 条历史后才出读数。

        计数语义：第 n 次调用看到的 prior **不含本次**，故需 5 次调用才攒下 5 条，
        第 6 次才满足 min_observations=5。
        """
        clock = FakeClock()
        channel = _v2_channel(clock, baseline_min_observations=5)
        for _ in range(5):
            assert _sense(channel, _flat_signal(1.0)) is None
            clock.advance(200.0)  # 跨度远超 1000×0.15=150s，故只有观测数这道门在起作用
        assert len(channel.baseline_history) == 5
        assert _sense(channel, _flat_signal(1.0)) is not None

    def test_coverage_fraction_gate(self) -> None:
        """观测数已够但时间跨度不足 → 仍 None。"""
        clock = FakeClock()
        channel = _v2_channel(clock)  # 需跨度 ≥ 1000×0.15 = 150s
        for _ in range(5):
            assert _sense(channel, _flat_signal(1.0)) is None
            clock.advance(10.0)  # 累计仅 50s
        clock.advance(200.0)
        assert _sense(channel, _flat_signal(1.0)) is not None


class TestV2DeltaSemantics:
    def _warm(self, channel: EdaChannel, clock: FakeClock, level: float) -> None:
        for _ in range(4):
            _sense(channel, _flat_signal(level))
            clock.advance(100.0)

    def test_flat_signal_reads_neutral(self) -> None:
        """信号恒定 → Δ=0 → μa=0.0（中性），**不是 -1.0**（这正是 v1 公式不可复用的原因）。"""
        clock = FakeClock()
        channel = _v2_channel(clock)
        self._warm(channel, clock, 5.0)
        prior = _sense(channel, _flat_signal(5.0))
        assert prior is not None
        assert prior.mu[1] == pytest.approx(0.0, abs=1e-9)

    def test_rise_above_baseline_is_positive(self) -> None:
        clock = FakeClock()
        channel = _v2_channel(clock)
        self._warm(channel, clock, 5.0)
        prior = _sense(channel, _flat_signal(5.5))
        assert prior is not None
        assert prior.mu[1] == pytest.approx(0.5, abs=1e-9)

    def test_drop_below_baseline_is_negative(self) -> None:
        clock = FakeClock()
        channel = _v2_channel(clock)
        self._warm(channel, clock, 5.0)
        prior = _sense(channel, _flat_signal(4.5))
        assert prior is not None
        assert prior.mu[1] == pytest.approx(-0.5, abs=1e-9)

    def test_absolute_level_does_not_matter(self) -> None:
        """判别性：跨被试可比的核心——同样的 Δ 在不同绝对水平下读数相同。

        这正是 v1 失败、对照臂「裸 SCL 固定 ref」也失败的那条轴（跨被试极差 1.745）。
        """
        readings = []
        for level in (0.2, 5.0, 15.0):
            clock = FakeClock()
            channel = _v2_channel(clock)
            for _ in range(4):
                _sense(channel, _flat_signal(level))
                clock.advance(100.0)
            prior = _sense(channel, _flat_signal(level + 0.5))
            assert prior is not None
            readings.append(prior.mu[1])
        assert max(readings) - min(readings) == pytest.approx(0.0, abs=1e-9)

    def test_valence_blind(self) -> None:
        clock = FakeClock()
        channel = _v2_channel(clock)
        self._warm(channel, clock, 5.0)
        prior = _sense(channel, _flat_signal(6.0))
        assert prior is not None
        assert prior.mu[0] == 0.0

    def test_horizon_prunes_old_baseline(self) -> None:
        """超出 horizon 的历史被按**真实秒数**裁掉（非样本数/调用数）。"""
        clock = FakeClock()
        channel = _v2_channel(clock, baseline_horizon_seconds=300.0)
        for _ in range(3):
            _sense(channel, _flat_signal(1.0))
            clock.advance(50.0)
        assert len(channel.baseline_history) == 3
        clock.advance(500.0)  # 全部超龄
        _sense(channel, _flat_signal(9.0))
        assert len(channel.baseline_history) == 1  # 仅剩本次


class TestV2NaNGuard:
    def test_nan_returns_none_and_does_not_pollute_history(self) -> None:
        """NaN 守卫必须**早于**任何状态写入——否则坏值进历史，污染后续所有窗的基线中位数。"""
        clock = FakeClock()
        channel = _v2_channel(clock)
        for _ in range(4):
            _sense(channel, _flat_signal(5.0))
            clock.advance(100.0)
        size_before = len(channel.baseline_history)

        bad = {"eda": np.array([np.nan, np.nan, np.nan], dtype=np.float64), "sampling_rate": 4}
        assert _sense(channel, bad) is None  # type: ignore[arg-type]
        assert len(channel.baseline_history) == size_before  # 未写入

        clock.advance(100.0)
        prior = _sense(channel, _flat_signal(5.5))
        assert prior is not None
        assert prior.mu[1] == pytest.approx(0.5, abs=1e-9)  # 基线未被 NaN 破坏


class TestV2Reset:
    def test_reset_clears_baseline_history(self) -> None:
        clock = FakeClock()
        channel = _v2_channel(clock)
        for _ in range(4):
            _sense(channel, _flat_signal(5.0))
            clock.advance(100.0)
        assert len(channel.baseline_history) > 0

        channel.reset()
        assert len(channel.baseline_history) == 0
        assert _sense(channel, _flat_signal(5.0)) is None  # 回到冷启动

    def test_reset_clears_both_v1_and_v2_state(self) -> None:
        """reset 须同时清 v1 幅度历史与 v2 基线历史（漏清任一都会造成跨被试污染）。"""
        channel = EdaChannel(arousal_metric="scl_baseline_delta_v2", clock=FakeClock())
        channel.baseline_history.append((0.0, 1.0))
        channel._amplitude_history["highpass"].append(0.5)
        channel.reset()
        assert len(channel.baseline_history) == 0
        assert len(channel._amplitude_history["highpass"]) == 0


class TestV2DoesNotRequireNeurokit:
    def test_no_phasic_decomposition(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """v2 不做 phasic 分解 → 即使 neurokit2 不可 import 也应正常出读数。

        判别性：若实现里残留 `import neurokit2`，本例会因 ImportError 被 sense() 兜成 None 而失败。
        """
        import builtins

        real_import = builtins.__import__

        def blocked(name: str, *args: object, **kwargs: object) -> object:
            if name == "neurokit2":
                raise ImportError("blocked by test")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", blocked)

        clock = FakeClock()
        channel = _v2_channel(clock)
        for _ in range(4):
            _sense(channel, _flat_signal(5.0))
            clock.advance(100.0)
        prior = _sense(channel, _flat_signal(5.5))
        assert prior is not None
        assert prior.mu[1] == pytest.approx(0.5, abs=1e-9)


class TestV2ConcurrencySafety:
    """并发不变式守卫（诚实定位：非「锁必要性」判别，同 TestEdaChannelConcurrencySafety）。

    `_process` 经 `asyncio.to_thread` 在线程池执行，同实例并发 collect 会并发读改
    `baseline_history`；本例验证并发投喂后历史不撕裂、条目数守恒。
    """

    def test_concurrent_sense_keeps_history_consistent(self) -> None:
        clock = FakeClock()
        channel = _v2_channel(clock, baseline_horizon_seconds=1e9)
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                for _ in range(20):
                    _sense(channel, _flat_signal(5.0))
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


class TestV2ConstantsMatchProbeSelection:
    """**四个** v2 常量须与 P0–P3 探针选定值一致——改动即须重跑探针（防静默漂移）。

    ⚠ 本类曾只 pin 了 4 个中的 1 个（code-review WARN #4：类名承诺的保障范围大于实际交付，
    属 pitfalls「绿灯必须先证明它能红」同族）。现四个全 pin，且同时断言
    **模块常量**与**构造后实例属性**——只 pin 模块常量会漏掉「默认值绑错常量」这一类改动。
    """

    def test_horizon_pinned(self) -> None:
        assert _SCL_BASELINE_HORIZON_SECONDS == 1800.0
        assert EdaChannel().baseline_horizon_seconds == 1800.0

    def test_delta_ref_pinned(self) -> None:
        assert _SCL_DELTA_REF_US == 1.0
        assert EdaChannel().delta_ref_us == 1.0

    def test_min_observations_pinned(self) -> None:
        assert _SCL_BASELINE_MIN_OBSERVATIONS == 2
        assert EdaChannel().baseline_min_observations == 2

    def test_min_coverage_pinned(self) -> None:
        assert _SCL_BASELINE_MIN_COVERAGE == 0.15
        assert EdaChannel().baseline_min_coverage_fraction == 0.15
