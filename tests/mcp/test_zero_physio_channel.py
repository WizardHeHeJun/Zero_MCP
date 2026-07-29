"""EdaChannel / HrvChannel 单测（Task H · E1-E13）—— mock neurokit2，脱真库依赖。

本文件覆盖**两通道共有的通道级契约**（先验形状、signal_source、优雅回退、Protocol
符合性、Hub 集成、默认关、NaN 守卫）与 **HrvChannel 的度量路径**。

⚠ **EdaChannel 的度量语义**（SCL−基线 Δ、冷启动、基线历史裁剪、跨被试可比）不在本文件，
在 `tests/mcp/test_zero_physio_eda_v2.py`。本文件对 EDA 只断言「通道级」行为，故用注入时钟
预热到能出读数即可，不关心读数取值。

覆盖蓝图 E1-E13：
  E1.  EDA 非空 ndarray → sense 返回 ModalityPrior 非 None（已过冷启动）。
  E2.  modality 是生理流前缀（is_physio_stream True）。
  E3.  precision[0] == MIN_PRECISION（效价精度零容差）。
  E4.  mu[1] ∈ [-1, 1]（唤醒 μa 合法）。
  E5.  mu[0] == 0.0（valence 恒零）。
  E6.  signal=None 且无 signal_source → None。
  E7.  signal_source(async) 注入 → 取到信号并正常处理。
  E8.  EDA 信号 dict 缺 'eda' 键 → ValueError → None + warning。
  E9.  HRV mock 处理失败 → None + warning。
  E10. EdaChannel / HrvChannel isinstance(PerceptionChannel) True。
  E11. 注册 PerceptionHub，含抛异常通道时该通道先验仍保留、异常通道跳过（AD-3）。
  E12. HrvChannel 独立产 modality="hrv/rmssd"、Πv=MIN_PRECISION。
  E13. ZERO_PHYSIO_CHANNEL_ENABLED=false（或未设）→ sense 返回 None。
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.models.zero_affect import ModalityPrior
from src.mcp.zero.channels.physio_channel import EdaChannel, HrvChannel
from src.mcp.zero.external_priors import (
    MIN_PRECISION,
    build_external_priors_override,
    is_physio_stream,
)
from src.mcp.zero.perception import PerceptionChannel, PerceptionHub

# ---------------------------------------------------------------------------
# 辅助：mock neurokit2（**仅 HrvChannel 需要**；EdaChannel 不依赖 neurokit2）
# ---------------------------------------------------------------------------


def _make_fake_hrv_df(rmssd_ms: float = 40.0) -> pd.DataFrame:
    """构造 neurokit2 hrv_time 返回的假 DataFrame。"""
    return pd.DataFrame({"HRV_RMSSD": [rmssd_ms]})


def _make_nk_mock(rmssd_ms: float = 40.0) -> MagicMock:
    """构造 HrvChannel 所需的 neurokit2 mock（ecg_process / hrv_time）。"""
    nk = MagicMock()
    # ecg_process 返回 (signals_df, info_dict)
    signals_df = pd.DataFrame({"ECG_R_Peaks": [0] * 5})
    nk.ecg_process.return_value = (signals_df, {})
    nk.hrv_time.return_value = _make_fake_hrv_df(rmssd_ms)
    return nk


def _make_eda_signal(rate: int = 8, level: float = 5.0) -> dict[str, Any]:
    """构造合法 EDA 信号 dict（恒定 SCL，取值本身在本文件无语义）。"""
    return {"eda": np.full(rate * 10, level, dtype=np.float64), "sampling_rate": rate}


def _make_ecg_signal(rate: int = 256) -> dict[str, Any]:
    """构造合法 ECG 信号 dict。"""
    return {"ecg_or_ppg": np.random.default_rng(42).random(rate * 10), "sampling_rate": rate}


class _StepClock:
    """可注入的确定性时钟：每 tick 前进固定秒数（不依赖墙钟）。"""

    def __init__(self, step_seconds: float = 300.0) -> None:
        self.now = 0.0
        self.step_seconds = step_seconds

    def __call__(self) -> float:
        return self.now

    def advance(self) -> None:
        self.now += self.step_seconds


def _make_warm_eda_channel(**kwargs: Any) -> tuple[EdaChannel, _StepClock]:
    """构造一个**已过冷启动**的 EdaChannel（喂满基线历史，跨过覆盖率与观测数双门）。

    EdaChannel 有状态：无基线证据时按契约返回 None。本文件测的是通道级行为（先验形状、
    Hub 集成…），不是「冷启动该不该返 None」——那条在 v2 度量测试文件里。
    """
    clock = _StepClock()
    channel = EdaChannel(clock=clock, **kwargs)
    # 直接注入基线历史：等价于此前若干窗的观测，避免在本文件里重复 v2 的预热语义
    for _ in range(3):
        channel.baseline_history.append((clock.now, 5.0))
        clock.advance()
    return channel, clock


# ---------------------------------------------------------------------------
# E1-E5：EdaChannel 正常路径（已过冷启动）
# ---------------------------------------------------------------------------


class TestEdaChannelNormalPath:
    """EdaChannel 正常路径：注入合法信号 → 产合法 ModalityPrior。"""

    async def test_returns_modality_prior_not_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """E1：非空 ndarray + 已有基线 → sense 返回 ModalityPrior 非 None。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch, _clock = _make_warm_eda_channel(sampling_rate=8)
        result = await ch.sense(signal=_make_eda_signal(rate=8))
        assert result is not None
        assert isinstance(result, ModalityPrior)

    def test_modality_is_physio_stream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """E2：modality="eda/sc" 满足 is_physio_stream 前缀（M2 命名约定）。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch = EdaChannel()
        assert is_physio_stream(ch.name)

    async def test_precision_v_equals_min_precision(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """E3：precision[0]（效价精度 Πv）== MIN_PRECISION（零容差）。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch, _clock = _make_warm_eda_channel(sampling_rate=8)
        result = await ch.sense(signal=_make_eda_signal(rate=8))
        assert result is not None
        assert result.precision[0] == pytest.approx(MIN_PRECISION)

    async def test_mu_a_in_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """E4：mu[1]（μa）∈ [-1, 1]——含远超 ref 的 Δ 也不越界（对称归一化钳制）。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch, _clock = _make_warm_eda_channel(sampling_rate=8)
        result = await ch.sense(signal=_make_eda_signal(rate=8, level=500.0))
        assert result is not None
        assert -1.0 <= result.mu[1] <= 1.0

    async def test_mu_v_is_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """E5：mu[0]（μv）== 0.0（valence 盲）。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch, _clock = _make_warm_eda_channel(sampling_rate=8)
        result = await ch.sense(signal=_make_eda_signal(rate=8))
        assert result is not None
        assert result.mu[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# E6-E7：无信号 / signal_source 分支
# ---------------------------------------------------------------------------


class TestEdaChannelSignalSource:
    """EdaChannel signal=None 与 signal_source 分支。"""

    async def test_none_signal_no_source_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, caplog: Any
    ) -> None:
        """E6：signal=None 且无 signal_source → None（有 warning）。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch = EdaChannel()
        with caplog.at_level(logging.WARNING, logger="src.mcp.zero.channels.physio_channel"):
            result = await ch.sense(signal=None)
        assert result is None

    async def test_signal_source_async_provides_signal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """E7：signal_source(async callable) 注入 → 取到信号并正常处理。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        source = AsyncMock(return_value=_make_eda_signal(rate=8))
        ch, _clock = _make_warm_eda_channel(sampling_rate=8, signal_source=source)
        result = await ch.sense()  # 无参，走 signal_source
        assert result is not None
        source.assert_awaited_once()

    async def test_signal_source_exception_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, caplog: Any
    ) -> None:
        """signal_source 调用失败（抛异常）→ None + warning（边界防御）。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        source = AsyncMock(side_effect=RuntimeError("设备超时"))
        ch = EdaChannel(signal_source=source)
        with caplog.at_level(logging.WARNING, logger="src.mcp.zero.channels.physio_channel"):
            result = await ch.sense()
        assert result is None
        assert "signal_source" in caplog.text


# ---------------------------------------------------------------------------
# E8-E9：优雅回退（处理期异常 → None + warning，不上抛）
# ---------------------------------------------------------------------------


class TestPhysioChannelGracefulFallback:
    """处理失败 → None + warning，不把异常抛给 PerceptionHub。"""

    async def test_eda_malformed_signal_returns_none_with_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: Any
    ) -> None:
        """E8：EDA 信号 dict 缺 'eda' 键 → _process 抛 ValueError → None + warning。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch, _clock = _make_warm_eda_channel(sampling_rate=8)
        with caplog.at_level(logging.WARNING, logger="src.mcp.zero.channels.physio_channel"):
            result = await ch.sense(signal={"sampling_rate": 8})
        assert result is None
        assert len(caplog.records) > 0

    async def test_hrv_import_error_returns_none_with_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: Any
    ) -> None:
        """HrvChannel 缺 neurokit2 → None + warning（缺库不崩溃）。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch = HrvChannel(sampling_rate=256)

        import sys

        original = sys.modules.get("neurokit2", None)
        sys.modules["neurokit2"] = None  # type: ignore[assignment]
        try:
            with caplog.at_level(logging.WARNING, logger="src.mcp.zero.channels.physio_channel"):
                result = await ch.sense(signal=_make_ecg_signal(rate=256))
        finally:
            if original is None:
                sys.modules.pop("neurokit2", None)
            else:
                sys.modules["neurokit2"] = original

        assert result is None
        assert "neurokit2" in caplog.text.lower() or "import" in caplog.text.lower()

    async def test_hrv_missing_column_returns_none_with_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: Any
    ) -> None:
        """E9：hrv_time 输出缺 HRV_RMSSD 列 → RuntimeError → None + warning。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch = HrvChannel(sampling_rate=256)
        nk = _make_nk_mock()
        nk.hrv_time.return_value = pd.DataFrame({"OTHER_COL": [0.1]})
        with patch.dict("sys.modules", {"neurokit2": nk}):
            with caplog.at_level(logging.WARNING, logger="src.mcp.zero.channels.physio_channel"):
                result = await ch.sense(signal=_make_ecg_signal(rate=256))
        assert result is None
        assert len(caplog.records) > 0


# ---------------------------------------------------------------------------
# E10：PerceptionChannel Protocol 符合性
# ---------------------------------------------------------------------------


class TestPhysioChannelProtocolCompliance:
    """EdaChannel / HrvChannel isinstance(PerceptionChannel) True（runtime_checkable）。"""

    def test_eda_channel_satisfies_perception_channel(self) -> None:
        """E10a：EdaChannel isinstance(PerceptionChannel) True。"""
        ch = EdaChannel()
        assert isinstance(ch, PerceptionChannel)

    def test_hrv_channel_satisfies_perception_channel(self) -> None:
        """E10b：HrvChannel isinstance(PerceptionChannel) True。"""
        ch = HrvChannel()
        assert isinstance(ch, PerceptionChannel)


# ---------------------------------------------------------------------------
# E11：PerceptionHub 集成 —— 含抛异常通道时生理先验保留（AD-3）
# ---------------------------------------------------------------------------


class TestPhysioChannelInPerceptionHub:
    """注册 PerceptionHub：异常通道跳过，生理通道先验正常保留（AD-3）。"""

    async def test_physio_prior_preserved_when_other_channel_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """E11：含一个抛异常 mock 通道 + EdaChannel，EdaChannel 先验仍保留。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")

        # 用 signal_source 注入信号，让 hub 无参调 sense()
        eda_ch, _clock = _make_warm_eda_channel(
            sampling_rate=8, signal_source=AsyncMock(return_value=_make_eda_signal(rate=8))
        )

        # 构造一个抛异常的 mock 通道
        # spec 限定：裸 MagicMock 会自动伪造 prepare/reset 等可选协议方法，令 Hub 的鸭子类型
        # 检测误判（并 await 一个不可等待的 mock）。只暴露 Protocol 真实成员。
        bad_ch: Any = MagicMock(spec=["name", "sense"])
        bad_ch.name = "bad_channel"
        bad_ch.sense = AsyncMock(side_effect=RuntimeError("设备故障"))

        hub = PerceptionHub([bad_ch, eda_ch])
        priors = await hub.collect()

        # bad_ch 跳过，eda_ch 先验保留
        assert len(priors) == 1
        assert priors[0].modality == "eda/sc"


# ---------------------------------------------------------------------------
# E12：HrvChannel 独立路径
# ---------------------------------------------------------------------------


class TestHrvChannelPath:
    """HrvChannel 独立产生 modality="hrv/rmssd"、Πv=MIN_PRECISION。"""

    async def test_hrv_channel_returns_hrv_rmssd_modality(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """E12a：HrvChannel sense → modality == "hrv/rmssd"。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch = HrvChannel(sampling_rate=256)
        nk = _make_nk_mock(rmssd_ms=40.0)
        with patch.dict("sys.modules", {"neurokit2": nk}):
            result = await ch.sense(signal=_make_ecg_signal(rate=256))
        assert result is not None
        assert result.modality == "hrv/rmssd"

    async def test_hrv_precision_v_is_min(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """E12b：HrvChannel Πv == MIN_PRECISION（效价精度）。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch = HrvChannel(sampling_rate=256)
        nk = _make_nk_mock(rmssd_ms=40.0)
        with patch.dict("sys.modules", {"neurokit2": nk}):
            result = await ch.sense(signal=_make_ecg_signal(rate=256))
        assert result is not None
        assert result.precision[0] == pytest.approx(MIN_PRECISION)

    async def test_hrv_mu_v_is_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """E12c：HrvChannel mu[0] == 0.0（valence 盲）。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch = HrvChannel(sampling_rate=256)
        nk = _make_nk_mock(rmssd_ms=40.0)
        with patch.dict("sys.modules", {"neurokit2": nk}):
            result = await ch.sense(signal=_make_ecg_signal(rate=256))
        assert result is not None
        assert result.mu[0] == pytest.approx(0.0)

    async def test_hrv_mu_a_in_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """E12d：HrvChannel mu[1]（μa）∈ [-1, 1]。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch = HrvChannel(sampling_rate=256)
        nk = _make_nk_mock(rmssd_ms=40.0)
        with patch.dict("sys.modules", {"neurokit2": nk}):
            result = await ch.sense(signal=_make_ecg_signal(rate=256))
        assert result is not None
        assert -1.0 <= result.mu[1] <= 1.0

    async def test_hrv_is_physio_stream(self) -> None:
        """E12e：HrvChannel.name 满足 is_physio_stream（M2 命名约定）。"""
        ch = HrvChannel()
        assert is_physio_stream(ch.name)


# ---------------------------------------------------------------------------
# E13：ZERO_PHYSIO_CHANNEL_ENABLED 关闭时 → None
# ---------------------------------------------------------------------------


class TestPhysioChannelDisabled:
    """ZERO_PHYSIO_CHANNEL_ENABLED 未设或 false → sense 返回 None。"""

    async def test_eda_disabled_by_default_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """E13a：EdaChannel enabled=false（默认）→ None，且不写基线历史。"""
        monkeypatch.delenv("ZERO_PHYSIO_CHANNEL_ENABLED", raising=False)
        ch = EdaChannel()
        result = await ch.sense(signal=_make_eda_signal())
        assert result is None
        assert len(ch.baseline_history) == 0, "关闭状态不应产生任何状态写入"

    async def test_eda_explicitly_disabled_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """E13b：ZERO_PHYSIO_CHANNEL_ENABLED=false → None。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "false")
        ch = EdaChannel()
        result = await ch.sense(signal=_make_eda_signal())
        assert result is None

    async def test_hrv_disabled_by_default_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """E13c：HrvChannel enabled=false（默认）→ None。"""
        monkeypatch.delenv("ZERO_PHYSIO_CHANNEL_ENABLED", raising=False)
        ch = HrvChannel()
        result = await ch.sense(signal=_make_ecg_signal())
        assert result is None


# ---------------------------------------------------------------------------
# NaN 守卫 —— 度量非有限值 → sense() 返回 None（无有效证据）
# ---------------------------------------------------------------------------


class TestPhysioChannelNaNGuard:
    """退化/坏信号致度量为 NaN 时通道返回 None（对齐「无证据本轮跳过」契约）。

    EdaChannel 侧的 NaN 守卫（含「不污染基线历史」的顺序断言）在
    `tests/mcp/test_zero_physio_eda_v2.py::TestV2NaNGuard`；此处只覆盖 HrvChannel。
    """

    async def test_hrv_nan_rmssd_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """HRV_RMSSD=NaN（R 峰不足）→ sense() 返回 None。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch = HrvChannel(sampling_rate=256)
        nk = _make_nk_mock()
        nk.hrv_time.return_value = pd.DataFrame({"HRV_RMSSD": [float("nan")]})
        with patch.dict("sys.modules", {"neurokit2": nk}):
            result = await ch.sense(signal=_make_ecg_signal(rate=256))
        assert result is None


# ---------------------------------------------------------------------------
# PerceptionHub.reset_all —— 有状态通道被清、无状态通道跳过
# ---------------------------------------------------------------------------


class TestPhysioChannelReset:
    """有状态通道（EdaChannel）的 reset 契约（W2：被试切换必须调用）。"""

    def test_hub_reset_all_clears_stateful_skips_stateless(self) -> None:
        """PerceptionHub.reset_all() 清空 EdaChannel 基线历史；无 reset 的通道安全跳过。"""
        eda, _clock = _make_warm_eda_channel()
        assert len(eda.baseline_history) > 0

        stateless: Any = MagicMock(spec=["name", "sense"])
        stateless.name = "stateless"

        hub = PerceptionHub([eda, stateless])
        hub.reset_all()  # 不应因 stateless 无 reset 而抛

        assert len(eda.baseline_history) == 0, "reset_all 应清空 EdaChannel 基线历史"


# ---------------------------------------------------------------------------
# D-6：通道 → wire 端到端 —— 真通道产出的 physio 先验落到载荷上仍须 μv≡0
# ---------------------------------------------------------------------------


class TestChannelToWireMuVZero:
    """`PHYSIO_PRECISION_A_SELF_IGNITE_BOUND=0.359` 的前提须在**出网口**成立，而非只在通道内。

    既有 E5 / E12c 只断言单通道 `ModalityPrior.mu[0] == 0.0`；本类补的是**单条 physio 流
    不触发 EDA/HRV 预合并、原样透传到 `build_external_priors_override` 载荷**这条路径——
    该路径上没有任何产品码会去归零 μv，故通道侧一旦写出非零 μv 就会**直接上 wire**，
    而按 μv≡0 闭式复算的 0.359 上界守卫不会报错（真实上界应收紧到 ≈0.2536）。

    ⚠ 判别力设计：μv 由**真 EdaChannel 计算**（非手写常量喂进去），故「通道改成带出非零
    μv」这一变异会真正驱红；若改用 `_make_prior(mu=(0.0, ...))` 手工构造，断言就退化成
    「我断言我自己刚写进去的 0」——恒真式（pitfalls⑥）。
    """

    async def test_channel_priors_reach_wire_with_zero_mu_v(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """真 EdaChannel 产出的单条先验直达载荷（不合并），其 wire 上的 μv 仍为 0。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch, _clock = _make_warm_eda_channel(sampling_rate=8)
        prior = await ch.sense(signal=_make_eda_signal(rate=8, level=500.0))
        assert prior is not None, "通道未产出先验，后续全称断言会恒真"

        payload = build_external_priors_override([prior])["external_priors"]
        physio_streams = [s for s in payload if is_physio_stream(s[0])]
        # 正控①：载荷里必须真有 physio 流被观测到（空集会让下面的 for 循环恒真）。
        assert physio_streams, f"载荷里没有 physio 流：{[s[0] for s in payload]}"
        # 正控②：确认走的是**不合并**路径（单条 EDA，无 HRV），即本用例覆盖的透传面。
        assert len(payload) == 1 and payload[0][0] == prior.modality

        for name, mu, _precision in physio_streams:
            assert mu[0] == 0.0, (
                f"physio 流 {name!r} 的出线 μv={mu[0]} ≠ 0 —— 自点燃上界 0.359 的前提已破；"
                "须把上界收紧到 ≈0.2536 并同步跨仓守卫的现算式"
            )
        # μa 由真信号算出、非平凡：证明本用例观测的确是「μv 那一维被钉 0」，
        # 而不是「整条先验退化成零向量」。
        assert physio_streams[0][1][1] != 0.0
