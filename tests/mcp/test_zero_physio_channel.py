"""EdaChannel / HrvChannel 单测（Task H · E1-E11）—— mock neurokit2，脱真库依赖。

覆盖蓝图 E1-E11：
  E1.  EDA 非空 ndarray(rate>4) → sense 返回 ModalityPrior 非 None。
  E2.  modality 是生理流前缀（is_physio_stream True）。
  E3.  precision[0] == MIN_PRECISION（效价精度零容差）。
  E4.  mu[1] ∈ [-1, 1]（唤醒 μa 合法）。
  E5.  mu[0] == 0.0（valence 恒零）。
  E6.  signal=None 且无 signal_source → None。
  E7.  signal_source(async) 注入 → 取到信号并正常处理。
  E8.  mock neurokit2 ImportError → None + warning。
  E9.  mock 处理抛 ValueError → None + warning。
  E10. EdaChannel / HrvChannel isinstance(PerceptionChannel) True。
  E11. 注册 PerceptionHub，含抛异常通道时该通道先验仍保留、异常通道跳过（AD-3）。
  E12. HrvChannel 独立产 modality="hrv/rmssd"、Πv=MIN_PRECISION。
  E13. ZERO_PHYSIO_CHANNEL_ENABLED=false（或未设）→ sense 返回 None。
  E14. 采样率≤4Hz 用 method="highpass"；>4Hz 用 method="cvxEDA"。
"""

from __future__ import annotations

import asyncio
import logging
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.models.zero_affect import ModalityPrior
from src.mcp.zero.channels.physio_channel import EdaChannel, HrvChannel
from src.mcp.zero.external_priors import MIN_PRECISION, is_physio_stream
from src.mcp.zero.perception import PerceptionChannel, PerceptionHub

# ---------------------------------------------------------------------------
# 辅助：构造 mock neurokit2 模块
# ---------------------------------------------------------------------------

_FAKE_PHASIC_VALUE = 0.5  # mock EDA_Phasic 列均值（> 0 产正 μa）


def _make_fake_eda_phasic_df(phasic_value: float = _FAKE_PHASIC_VALUE) -> pd.DataFrame:
    """构造 neurokit2 eda_phasic 返回的假 DataFrame。"""
    return pd.DataFrame({"EDA_Phasic": [phasic_value] * 10})


def _make_fake_hrv_df(rmssd_ms: float = 40.0) -> pd.DataFrame:
    """构造 neurokit2 hrv_time 返回的假 DataFrame。"""
    return pd.DataFrame({"HRV_RMSSD": [rmssd_ms]})


def _make_nk_mock(
    phasic_value: float = _FAKE_PHASIC_VALUE,
    rmssd_ms: float = 40.0,
) -> MagicMock:
    """构造完整 neurokit2 mock 对象（eda_phasic / standardize / ecg_process / hrv_time）。"""
    nk = MagicMock()
    nk.standardize.return_value = np.zeros(100)
    nk.eda_phasic.return_value = _make_fake_eda_phasic_df(phasic_value)
    # ecg_process 返回 (signals_df, info_dict)
    signals_df = pd.DataFrame({"ECG_R_Peaks": [0] * 5})
    nk.ecg_process.return_value = (signals_df, {})
    nk.hrv_time.return_value = _make_fake_hrv_df(rmssd_ms)
    return nk


def _make_eda_signal(rate: int = 8) -> dict[str, Any]:
    """构造合法 EDA 信号 dict。"""
    return {"eda": np.random.default_rng(42).random(rate * 10), "sampling_rate": rate}


def _make_ecg_signal(rate: int = 256) -> dict[str, Any]:
    """构造合法 ECG 信号 dict。"""
    return {"ecg_or_ppg": np.random.default_rng(42).random(rate * 10), "sampling_rate": rate}


# ---------------------------------------------------------------------------
# E1-E5：EdaChannel 正常路径（mock neurokit2，rate>4）
# ---------------------------------------------------------------------------


class TestEdaChannelNormalPath:
    """EdaChannel 正常路径：注入合法信号 → 产合法 ModalityPrior。"""

    async def test_returns_modality_prior_not_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """E1：非空 ndarray + rate>4 → sense 返回 ModalityPrior 非 None。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch = EdaChannel(sampling_rate=8)
        nk = _make_nk_mock()
        with patch.dict("sys.modules", {"neurokit2": nk}):
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
        ch = EdaChannel(sampling_rate=8)
        nk = _make_nk_mock()
        with patch.dict("sys.modules", {"neurokit2": nk}):
            result = await ch.sense(signal=_make_eda_signal(rate=8))
        assert result is not None
        assert result.precision[0] == pytest.approx(MIN_PRECISION)

    async def test_mu_a_in_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """E4：mu[1]（μa）∈ [-1, 1]。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch = EdaChannel(sampling_rate=8)
        nk = _make_nk_mock()
        with patch.dict("sys.modules", {"neurokit2": nk}):
            result = await ch.sense(signal=_make_eda_signal(rate=8))
        assert result is not None
        assert -1.0 <= result.mu[1] <= 1.0

    async def test_mu_v_is_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """E5：mu[0]（μv）== 0.0（valence 盲）。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch = EdaChannel(sampling_rate=8)
        nk = _make_nk_mock()
        with patch.dict("sys.modules", {"neurokit2": nk}):
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
        sig = _make_eda_signal(rate=8)
        source = AsyncMock(return_value=sig)
        ch = EdaChannel(sampling_rate=8, signal_source=source)
        nk = _make_nk_mock()
        with patch.dict("sys.modules", {"neurokit2": nk}):
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
# E8-E9：优雅回退（ImportError / ValueError）
# ---------------------------------------------------------------------------


class TestEdaChannelGracefulFallback:
    """EdaChannel 优雅回退：缺库 / 处理失败 → None + warning。"""

    async def test_import_error_returns_none_with_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: Any
    ) -> None:
        """E8：neurokit2 ImportError → None + warning（缺库不崩溃）。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch = EdaChannel(sampling_rate=8)

        # _process 内的 `import neurokit2 as nk` 抛 ImportError
        import sys

        original = sys.modules.get("neurokit2", None)
        sys.modules["neurokit2"] = None  # type: ignore[assignment]
        try:
            with caplog.at_level(logging.WARNING, logger="src.mcp.zero.channels.physio_channel"):
                result = await ch.sense(signal=_make_eda_signal(rate=8))
        finally:
            if original is None:
                sys.modules.pop("neurokit2", None)
            else:
                sys.modules["neurokit2"] = original

        assert result is None
        # warning 应提及 neurokit2 或 ImportError
        assert "neurokit2" in caplog.text.lower() or "import" in caplog.text.lower()

    async def test_value_error_in_processing_returns_none_with_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: Any
    ) -> None:
        """E9：_process 抛 ValueError → None + warning。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch = EdaChannel(sampling_rate=8)
        nk = _make_nk_mock()
        # eda_phasic 返回缺少 EDA_Phasic 列的 DataFrame → RuntimeError 路径
        nk.eda_phasic.return_value = pd.DataFrame({"OTHER_COL": [0.1] * 10})
        with patch.dict("sys.modules", {"neurokit2": nk}):
            with caplog.at_level(logging.WARNING, logger="src.mcp.zero.channels.physio_channel"):
                result = await ch.sense(signal=_make_eda_signal(rate=8))
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

        eda_ch = EdaChannel(sampling_rate=8)
        nk = _make_nk_mock()

        # 用 signal_source 注入信号，让 hub 无参调 sense()
        eda_ch.signal_source = AsyncMock(return_value=_make_eda_signal(rate=8))

        # 构造一个抛异常的 mock 通道
        # spec 限定：裸 MagicMock 会自动伪造 prepare/reset 等可选协议方法，令 Hub 的鸭子类型
        # 检测误判（并 await 一个不可等待的 mock）。只暴露 Protocol 真实成员。
        bad_ch: Any = MagicMock(spec=["name", "sense"])
        bad_ch.name = "bad_channel"
        bad_ch.sense = AsyncMock(side_effect=RuntimeError("设备故障"))

        hub = PerceptionHub([bad_ch, eda_ch])
        with patch.dict("sys.modules", {"neurokit2": nk}):
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
        """E13a：EdaChannel enabled=false（默认）→ None，不调用 neurokit2。"""
        monkeypatch.delenv("ZERO_PHYSIO_CHANNEL_ENABLED", raising=False)
        ch = EdaChannel()
        result = await ch.sense(signal=_make_eda_signal())
        assert result is None

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
# E14：采样率分支 — cvxEDA(>4Hz) / highpass(≤4Hz)
# ---------------------------------------------------------------------------


class TestEdaChannelSamplingRateBranch:
    """采样率决定 eda_phasic method 选择（gap-6）。"""

    async def test_high_rate_uses_cvxeda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """E14a：rate=8（>4）→ eda_phasic 以 method="cvxEDA" 调用。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch = EdaChannel(sampling_rate=8)
        nk = _make_nk_mock()
        with patch.dict("sys.modules", {"neurokit2": nk}):
            await ch.sense(signal=_make_eda_signal(rate=8))
        # 断言 eda_phasic 被以 method="cvxEDA" 调用
        call_kwargs = nk.eda_phasic.call_args
        assert call_kwargs is not None
        _, kwargs = call_kwargs
        assert kwargs.get("method") == "cvxEDA"

    async def test_low_rate_uses_highpass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """E14b：rate=4（≤4）→ eda_phasic 以 method="highpass" 调用。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch = EdaChannel(sampling_rate=4)
        nk = _make_nk_mock()
        with patch.dict("sys.modules", {"neurokit2": nk}):
            await ch.sense(signal=_make_eda_signal(rate=4))
        call_kwargs = nk.eda_phasic.call_args
        assert call_kwargs is not None
        _, kwargs = call_kwargs
        assert kwargs.get("method") == "highpass"

    async def test_signal_dict_rate_overrides_constructor_rate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """E14c：信号 dict 中的 sampling_rate 覆盖构造器默认值。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        # 构造器给 rate=4（低），但信号 dict 给 rate=8（高）→ 应用 cvxEDA
        ch = EdaChannel(sampling_rate=4)
        nk = _make_nk_mock()
        sig = _make_eda_signal(rate=8)
        sig["sampling_rate"] = 8
        with patch.dict("sys.modules", {"neurokit2": nk}):
            await ch.sense(signal=sig)
        call_kwargs = nk.eda_phasic.call_args
        assert call_kwargs is not None
        _, kwargs = call_kwargs
        assert kwargs.get("method") == "cvxEDA"


# ---------------------------------------------------------------------------
# P1-P5：EdaChannel percentile 归一化（mock neurokit2）
# ---------------------------------------------------------------------------


def _make_nk_mock_with_amplitude(phasic_abs_mean: float) -> MagicMock:
    """构造 mock neurokit2，eda_phasic 返回的 EDA_Phasic 均值 abs == phasic_abs_mean。

    用正值填充列（abs().mean() == phasic_abs_mean），绕开零均值陷阱。
    """
    nk = _make_nk_mock(phasic_value=phasic_abs_mean)
    # _make_nk_mock 已设 phasic_value 作 DataFrame 值，abs().mean() == phasic_abs_mean（均正值）
    return nk


class TestEdaChannelPercentileNormalization:
    """percentile 归一化逻辑（mock neurokit2，P1-P5）。

    覆盖：
      P1. 冷启动期间 percentile 输出 == 同输入 linear 输出（零回归过渡）。
      P2. reset() 清空历史 → reset 后又走冷启动。
      P3. normalization 默认 "linear" → 历史 deque 保持空。
      P4. 退化窗口（喂常量幅度）→ 回退 linear 不崩。
      P5. 分桶：rate>4(cvxEDA) 与 rate≤4(highpass) 历史互不污染。
    """

    async def test_p1_cold_start_equals_linear(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """P1：前 percentile_cold_start-1 次 percentile 输出 == 对应 linear 输出（冷启动零回归）。

        在暖机阈值（默认 20）前，percentile 模式每次输出应与 linear 模式相同输入的输出一致。
        测试取 cold_start=5（缩短以节省时间），喂 4 次（< cold_start），断言每次相等。
        """
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        cold_start = 5
        phasic_values = [0.1, 0.3, 0.5, 0.2]  # 4次 < cold_start=5

        for phasic_val in phasic_values:
            # percentile 通道
            ch_pct = EdaChannel(
                sampling_rate=4,
                normalization="percentile",
                percentile_cold_start=cold_start,
            )
            # linear 通道（基准）
            ch_lin = EdaChannel(sampling_rate=4, normalization="linear")

            nk_pct = _make_nk_mock(phasic_value=phasic_val)
            nk_lin = _make_nk_mock(phasic_value=phasic_val)

            with patch.dict("sys.modules", {"neurokit2": nk_pct}):
                result_pct = await ch_pct.sense(signal=_make_eda_signal(rate=4))
            with patch.dict("sys.modules", {"neurokit2": nk_lin}):
                result_lin = await ch_lin.sense(signal=_make_eda_signal(rate=4))

            assert result_pct is not None
            assert result_lin is not None
            assert result_pct.mu[1] == pytest.approx(result_lin.mu[1], abs=1e-9), (
                f"冷启动 phasic={phasic_val}：percentile μa={result_pct.mu[1]:.6f} "
                f"!= linear μa={result_lin.mu[1]:.6f}，冷启动零回归违反"
            )

    async def test_p1_cold_start_stateful_accumulation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P1b：同一实例连续调用 cold_start-1 次，每次均应等于对应 linear 输出。

        验证逐实例历史正确积累（deque 按顺序追加），且追加后仍在冷启动区间时回退 linear。
        """
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        cold_start = 5
        ch_pct = EdaChannel(
            sampling_rate=4,
            normalization="percentile",
            percentile_cold_start=cold_start,
        )
        phasic_values = [0.1, 0.2, 0.3, 0.4]  # 4 次 < cold_start=5

        for i, phasic_val in enumerate(phasic_values):
            nk_mock = _make_nk_mock(phasic_value=phasic_val)
            nk_lin_mock = _make_nk_mock(phasic_value=phasic_val)

            with patch.dict("sys.modules", {"neurokit2": nk_mock}):
                result_pct = await ch_pct.sense(signal=_make_eda_signal(rate=4))

            ch_lin = EdaChannel(sampling_rate=4, normalization="linear")
            with patch.dict("sys.modules", {"neurokit2": nk_lin_mock}):
                result_lin = await ch_lin.sense(signal=_make_eda_signal(rate=4))

            assert result_pct is not None and result_lin is not None
            assert result_pct.mu[1] == pytest.approx(result_lin.mu[1], abs=1e-9), (
                f"第{i + 1}次调用（phasic={phasic_val}）：percentile μa != linear μa，"
                f"冷启动积累期零回归违反（hist长度={len(ch_pct._amplitude_history['highpass'])}）"
            )

        # 验证历史已积累 cold_start-1 个值
        assert len(ch_pct._amplitude_history["highpass"]) == len(phasic_values)

    async def test_p2_reset_clears_history_and_restores_cold_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P2：reset() 清空历史；reset 后再调用应再次走冷启动（输出 == linear）。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        cold_start = 3
        ch = EdaChannel(
            sampling_rate=4,
            normalization="percentile",
            percentile_cold_start=cold_start,
            percentile_window=20,
        )

        # 喂满暖机（>= cold_start 次）
        for _ in range(cold_start):
            nk_mock = _make_nk_mock(phasic_value=0.3)
            with patch.dict("sys.modules", {"neurokit2": nk_mock}):
                await ch.sense(signal=_make_eda_signal(rate=4))

        assert len(ch._amplitude_history["highpass"]) >= cold_start, "暖机后历史应 >= cold_start"

        # reset
        ch.reset()
        assert len(ch._amplitude_history["highpass"]) == 0, "reset 后 highpass 桶应为空"
        assert len(ch._amplitude_history["cvxEDA"]) == 0, "reset 后 cvxEDA 桶应为空"

        # reset 后首次调用 → 冷启动，输出应 == linear
        phasic_val = 0.25
        nk_after = _make_nk_mock(phasic_value=phasic_val)
        nk_lin = _make_nk_mock(phasic_value=phasic_val)
        with patch.dict("sys.modules", {"neurokit2": nk_after}):
            result_pct = await ch.sense(signal=_make_eda_signal(rate=4))
        ch_lin = EdaChannel(sampling_rate=4, normalization="linear")
        with patch.dict("sys.modules", {"neurokit2": nk_lin}):
            result_lin = await ch_lin.sense(signal=_make_eda_signal(rate=4))

        assert result_pct is not None and result_lin is not None
        assert result_pct.mu[1] == pytest.approx(result_lin.mu[1], abs=1e-9), (
            f"reset 后冷启动：percentile μa={result_pct.mu[1]:.6f} "
            f"!= linear μa={result_lin.mu[1]:.6f}"
        )

    async def test_p3_default_linear_does_not_touch_history(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P3：normalization="linear"（默认）时，历史 deque 保持空，不累积任何幅度值。

        linear 路径完全不动，历史 deque 始终不被写入。
        """
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch = EdaChannel(sampling_rate=4, normalization="linear")

        for _ in range(5):
            nk_mock = _make_nk_mock(phasic_value=0.4)
            with patch.dict("sys.modules", {"neurokit2": nk_mock}):
                await ch.sense(signal=_make_eda_signal(rate=4))

        assert len(ch._amplitude_history["highpass"]) == 0, "linear 模式不应写入 highpass 历史桶"
        assert len(ch._amplitude_history["cvxEDA"]) == 0, "linear 模式不应写入 cvxEDA 历史桶"

    async def test_p4_degenerate_window_falls_back_to_linear(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P4：喂常量幅度（p_high - p_low < eps）→ 退化窗口守卫回退 linear，不崩溃。

        当所有历史样本幅度相同时，p5 == p95，分位区间塌缩为零宽，
        应回退 linear 归一化而非除零崩溃。
        """
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        cold_start = 3
        constant_amplitude = 0.3  # 所有样本相同幅度 → 退化窗口
        ch = EdaChannel(
            sampling_rate=4,
            normalization="percentile",
            percentile_cold_start=cold_start,
            percentile_window=20,
        )

        # 喂超过 cold_start 次的常量幅度，进入暖机后路径
        for _ in range(cold_start + 2):
            nk_mock = _make_nk_mock(phasic_value=constant_amplitude)
            with patch.dict("sys.modules", {"neurokit2": nk_mock}):
                result = await ch.sense(signal=_make_eda_signal(rate=4))
            assert result is not None, "退化窗口守卫后应仍产出 ModalityPrior（非 None）"
            assert -1.0 <= result.mu[1] <= 1.0, f"μa={result.mu[1]} 超出 [-1,1]"

        # 最后一次输出应等于 linear（退化窗口回退）
        nk_final = _make_nk_mock(phasic_value=constant_amplitude)
        nk_lin = _make_nk_mock(phasic_value=constant_amplitude)
        ch_lin = EdaChannel(sampling_rate=4, normalization="linear")

        # 对退化后的 ch，再喂一次（历史已全常量，必触发退化守卫）
        with patch.dict("sys.modules", {"neurokit2": nk_final}):
            result_pct = await ch.sense(signal=_make_eda_signal(rate=4))
        with patch.dict("sys.modules", {"neurokit2": nk_lin}):
            result_lin = await ch_lin.sense(signal=_make_eda_signal(rate=4))

        assert result_pct is not None and result_lin is not None
        assert result_pct.mu[1] == pytest.approx(result_lin.mu[1], abs=1e-9), (
            f"退化窗口应回退 linear：percentile μa={result_pct.mu[1]:.6f} "
            f"!= linear μa={result_lin.mu[1]:.6f}"
        )

    async def test_p5_buckets_are_independent_by_method(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P5：cvxEDA(rate>4) 与 highpass(rate≤4) 历史桶互不污染。

        给同一实例分别喂高采样率（rate=8, cvxEDA）和低采样率（rate=4, highpass）信号，
        断言两个桶各自独立积累，且各自的桶长度正确。
        """
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch = EdaChannel(
            sampling_rate=4,
            normalization="percentile",
            percentile_window=20,
            percentile_cold_start=10,
        )

        # 喂 3 次 highpass（rate=4），幅度 = 0.2
        for _ in range(3):
            nk_mock = _make_nk_mock(phasic_value=0.2)
            sig = _make_eda_signal(rate=4)
            with patch.dict("sys.modules", {"neurokit2": nk_mock}):
                await ch.sense(signal=sig)

        # 喂 5 次 cvxEDA（rate=8），幅度 = 25.0（cvxEDA 量级）
        for _ in range(5):
            nk_mock = _make_nk_mock(phasic_value=25.0)
            sig = _make_eda_signal(rate=8)
            sig["sampling_rate"] = 8
            with patch.dict("sys.modules", {"neurokit2": nk_mock}):
                await ch.sense(signal=sig)

        # 断言两桶独立：highpass 桶 3 个，cvxEDA 桶 5 个
        assert len(ch._amplitude_history["highpass"]) == 3, (
            f"highpass 桶应有 3 个样本，实际 {len(ch._amplitude_history['highpass'])}"
        )
        assert len(ch._amplitude_history["cvxEDA"]) == 5, (
            f"cvxEDA 桶应有 5 个样本，实际 {len(ch._amplitude_history['cvxEDA'])}"
        )

        # 验证桶内容互不污染：highpass 桶全为 0.2，cvxEDA 桶全为 25.0
        assert all(v == pytest.approx(0.2) for v in ch._amplitude_history["highpass"]), (
            "highpass 桶被 cvxEDA 数据污染"
        )
        assert all(v == pytest.approx(25.0) for v in ch._amplitude_history["cvxEDA"]), (
            "cvxEDA 桶被 highpass 数据污染"
        )

    def test_p6_hub_reset_all_clears_stateful_skips_stateless(self) -> None:
        """P6：PerceptionHub.reset_all() 清空有状态通道（EdaChannel）历史，无 reset() 的
        通道（HrvChannel）安全跳过（W2 修复回归）。"""
        eda = EdaChannel(normalization="percentile")
        eda._amplitude_history["highpass"].extend([1.0, 2.0, 3.0])
        hrv = HrvChannel()  # 无 reset() 方法（鸭子类型应被跳过）
        assert not hasattr(hrv, "reset")

        hub = PerceptionHub([eda, hrv])
        hub.reset_all()  # 不应因 hrv 无 reset 而崩

        assert len(eda._amplitude_history["highpass"]) == 0, "reset_all 应清空 EdaChannel 历史"


# ---------------------------------------------------------------------------
# W3：并发线程安全 —— _process 经 asyncio.to_thread 后同实例并发 sense() 不损坏历史
# ---------------------------------------------------------------------------


def _fake_standardize(x: Any) -> Any:
    """纯函数 neurokit2.standardize 桩（无状态、无调用记录 → 线程安全）。"""
    return np.asarray(x, dtype=float)


def _fake_eda_phasic(x: Any, sampling_rate: int, method: str) -> pd.DataFrame:
    """纯函数 neurokit2.eda_phasic 桩：EDA_Phasic 均值 = 输入 abs 均值（随信号变化）。

    用输入自身派生幅度（无共享状态）：不同信号 → 不同幅度 → 暖机后走非退化 percentile
    分支，同时对并发临界区施压。返回全新 DataFrame，无跨线程共享可变对象。
    """
    val = float(np.abs(np.asarray(x, dtype=float)).mean())
    return pd.DataFrame({"EDA_Phasic": [val] * 10})


# 纯函数桩：**刻意不用 MagicMock**——MagicMock 的调用记录（call_args_list 等）本身线程
# 不安全，并发下会伪造/掩盖竞争，使本测试失去意义。SimpleNamespace + 纯函数则完全无状态。
_THREAD_SAFE_FAKE_NK = types.SimpleNamespace(
    standardize=_fake_standardize,
    eda_phasic=_fake_eda_phasic,
)


class TestEdaChannelConcurrencySafety:
    """W3：_process 经 asyncio.to_thread 到线程池后，同实例并发 sense() 的**不变式守卫**。

    本测试是**并发正确性/回归守卫**，不是「锁必要性」判别测试——现场核验（scratch mech_probe，
    CPython 3.12）表明 np.asarray(deque)/list(deque) 是 GIL 下 C 级原子拷贝，并发 append 既不
    互撕也**不**触发「deque mutated during iteration」（该异常仅经显式 Python 迭代 iter(deque)
    才有）；故当前解释器下**有锁/无锁本测试都全绿**（已 scratch 验证 None==0）。history_lock 是
    「不赖此 GIL 偶发原子性」的显式保证（前瞻 no-GIL Python 与日后迭代式改写），不由本测试判别。

    本测试实际守护的是：to_thread 化后同实例高并发 sense() 仍产出**合法结果、无崩溃/死锁、
    deque maxlen 不变式不被写坏**——若日后 _process 引入观察得到的并发损坏（如临界区被拆开
    且改成显式迭代），此守卫可捕获。压测：window=8 令 deque 常满、200 次并发 sense()。
    """

    async def test_concurrent_sense_same_instance_thread_safe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """同实例并发 200 次 sense()：全部产合法先验、无 None/崩溃、历史长度受 maxlen 约束。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        window = 8
        ch = EdaChannel(
            sampling_rate=4,
            normalization="percentile",
            percentile_cold_start=3,
            percentile_window=window,
        )
        n_calls = 200
        # 每次不同 seed → 幅度各异 → 暖机后走非退化 percentile 分支（同时压并发临界区）
        signals: list[dict[str, Any]] = [
            {"eda": np.random.default_rng(i).random(40), "sampling_rate": 4} for i in range(n_calls)
        ]

        with patch.dict("sys.modules", {"neurokit2": _THREAD_SAFE_FAKE_NK}):
            results = await asyncio.gather(*[ch.sense(signal=s) for s in signals])

        # 并发下每次都应产出合法先验（None 会暴露被吞的线程内异常/观察得到的并发损坏）
        assert all(r is not None for r in results), (
            "并发下出现 sense()==None：疑同实例并发对滚动历史的读改写产生了观察得到的损坏"
        )
        assert all(-1.0 <= r.mu[1] <= 1.0 for r in results if r is not None), "并发下产出 μa 越界"
        # 200 次 append 后 deque 受 maxlen 约束，长度恒为 window（maxlen 不变式未被并发写坏）
        assert len(ch._amplitude_history["highpass"]) == window, (
            f"并发 append 后 highpass 桶长={len(ch._amplitude_history['highpass'])}，应 == {window}"
        )


# ---------------------------------------------------------------------------
# NaN 守卫 —— scr_amplitude/rmssd 非有限值 → sense() 返回 None（无有效证据）
# ---------------------------------------------------------------------------


class TestPhysioChannelNaNGuard:
    """退化/坏信号致度量为 NaN 时通道返回 None（对齐「无证据本轮跳过」契约）——不产出 NaN
    先验、不污染 percentile 历史。守卫置于 linear/percentile 分支之前，两分支共用。"""

    async def test_eda_nan_amplitude_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EDA_Phasic 全 NaN → abs().mean()=NaN → sense() 返回 None（linear 分支）。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch = EdaChannel(sampling_rate=8)
        nk = _make_nk_mock()
        nk.eda_phasic.return_value = pd.DataFrame({"EDA_Phasic": [float("nan")] * 10})
        with patch.dict("sys.modules", {"neurokit2": nk}):
            result = await ch.sense(signal=_make_eda_signal(rate=8))
        assert result is None

    async def test_eda_nan_amplitude_percentile_returns_none_and_history_clean(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """percentile 分支下 NaN 幅度 → None，且不进滚动历史（守卫在 append 之前 return）。"""
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch = EdaChannel(sampling_rate=8, normalization="percentile", percentile_cold_start=3)
        nk = _make_nk_mock()
        nk.eda_phasic.return_value = pd.DataFrame({"EDA_Phasic": [float("nan")] * 10})
        with patch.dict("sys.modules", {"neurokit2": nk}):
            result = await ch.sense(signal=_make_eda_signal(rate=8))
        assert result is None
        assert len(ch._amplitude_history["cvxEDA"]) == 0, "NaN 不应进 cvxEDA 历史桶"
        assert len(ch._amplitude_history["highpass"]) == 0, "NaN 不应进 highpass 历史桶"

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
# 秒制化参数（window_seconds / cold_start_seconds）—— 采样率无关的窗口推导
# ---------------------------------------------------------------------------


class TestEdaChannelSecondsParameterization:
    """percentile 滚动窗从「样本数硬编码」改为「秒 × 构造期采样率」推导（采样率无关）。

    修 footgun：固定 window=60 在 4Hz=15s、但 256Hz 仅 0.23s（窗口塌缩「崩」）。秒制化后同一
    window_seconds 在任意采样率下都覆盖等长时间历史。显式样本数覆盖
    （percentile_window/percentile_cold_start 非 None）优先——保既有调用零回归。
    """

    def test_default_seconds_derive_legacy_sample_defaults(self) -> None:
        """默认 window_seconds=15 / cold_start_seconds=10 @ 4Hz → 60 / 40（与旧样本默认等价）。"""
        ch = EdaChannel(sampling_rate=4, normalization="percentile")
        assert ch.percentile_window == 60
        assert ch.percentile_cold_start == 40
        assert ch._amplitude_history["highpass"].maxlen == 60
        assert ch._amplitude_history["cvxEDA"].maxlen == 60

    def test_window_seconds_is_sampling_rate_invariant(self) -> None:
        """⚠ **单位错配修复后**（蓝图任务 10）：同一 window_seconds 在任意采样率下 maxlen **恒定**。

        修复前该断言写的是 4Hz→60 / 8Hz→120 / 256Hz→3840——即 maxlen **随采样率成比例**，
        与本方法名（"sampling_rate_invariant"）**正好相反**。根因：deque 存的是每次
        `_process()` 调用产出的**一个标量**，不是采样点，故除数应是**分析窗长**而非采样率。
        """
        for rate in (4, 8, 256):
            channel = EdaChannel(sampling_rate=rate, normalization="percentile")
            assert channel.percentile_window == 60, f"{rate}Hz 下 maxlen 应恒为 60"

    def test_cold_start_seconds_is_sampling_rate_invariant(self) -> None:
        """cold_start 同理：任意采样率下恒为 40（修复前 4Hz→40 / 8Hz→80）。"""
        for rate in (4, 8, 256):
            channel = EdaChannel(sampling_rate=rate, normalization="percentile")
            assert channel.percentile_cold_start == 40, f"{rate}Hz 下 cold_start 应恒为 40"

    def test_custom_window_seconds_derives_maxlen_by_analysis_window(self) -> None:
        """自定义秒数经 round(window_seconds ÷ analysis_window_seconds) 推导（含小数商 round）。"""
        ch = EdaChannel(
            sampling_rate=8,
            normalization="percentile",
            window_seconds=600.0,
            analysis_window_seconds=15.0,
        )
        assert ch.percentile_window == 40  # 600/15
        assert ch._amplitude_history["highpass"].maxlen == 40
        # 小数商 round：100s ÷ 6s = 16.67 → 17
        ch2 = EdaChannel(
            sampling_rate=7,
            normalization="percentile",
            window_seconds=100.0,
            analysis_window_seconds=6.0,
        )
        assert ch2.percentile_window == 17

    def test_analysis_window_seconds_drives_maxlen_not_sampling_rate(self) -> None:
        """判别性：**改分析窗长**才动 maxlen，**改采样率**不动——这是本次修复的核心断言。"""
        base = EdaChannel(sampling_rate=4, normalization="percentile")
        rate_changed = EdaChannel(sampling_rate=256, normalization="percentile")
        window_changed = EdaChannel(
            sampling_rate=4, normalization="percentile", analysis_window_seconds=60.0
        )
        assert rate_changed.percentile_window == base.percentile_window  # 采样率无关
        assert window_changed.percentile_window == 30  # 1800/60，分析窗长翻倍 → maxlen 减半
        assert window_changed.percentile_window != base.percentile_window

    def test_explicit_sample_override_wins_over_seconds(self) -> None:
        """显式样本数覆盖（percentile_window/percentile_cold_start）优先于秒制参数。"""
        ch = EdaChannel(
            sampling_rate=8,
            normalization="percentile",
            window_seconds=999.0,  # 若被采用 → 7992；断言其被样本数覆盖
            cold_start_seconds=999.0,
            percentile_window=20,
            percentile_cold_start=5,
        )
        assert ch.percentile_window == 20
        assert ch.percentile_cold_start == 5
        assert ch._amplitude_history["highpass"].maxlen == 20

    def test_legacy_style_construction_unchanged(self) -> None:
        """旧式构造（只传样本数）解析值与旧默认完全一致——零回归。"""
        ch = EdaChannel(
            sampling_rate=8,
            normalization="percentile",
            percentile_window=60,
            percentile_cold_start=40,
        )
        assert ch.percentile_window == 60
        assert ch.percentile_cold_start == 40

    def test_nonpositive_window_seconds_floored_to_one(self) -> None:
        """window_seconds≤0 退化输入被 max(1,…) 兜底，不产 maxlen=0 死 deque。"""
        ch = EdaChannel(sampling_rate=4, normalization="percentile", window_seconds=0.0)
        assert ch.percentile_window == 1
        assert ch._amplitude_history["highpass"].maxlen == 1

    def test_nonpositive_cold_start_seconds_floored_to_one(self) -> None:
        """cold_start_seconds≤0 退化输入被 max(1,…) 兜底（与 window 侧对称）。"""
        ch = EdaChannel(sampling_rate=4, normalization="percentile", cold_start_seconds=0.0)
        assert ch.percentile_cold_start == 1

    def test_explicit_window_zero_floored_to_one(self) -> None:
        """显式 percentile_window=0/cold_start=0（笔误）被 max(1,…) 兜底，不产死 deque。"""
        ch = EdaChannel(
            sampling_rate=4,
            normalization="percentile",
            percentile_window=0,
            percentile_cold_start=0,
        )
        assert ch.percentile_window == 1
        assert ch.percentile_cold_start == 1
        assert ch._amplitude_history["highpass"].maxlen == 1

    def test_cold_start_gt_window_warns_in_percentile(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """percentile 模式下 cold_start > window → 构造期告警（恒退化 linear 的真条件）。"""
        with caplog.at_level(logging.WARNING):
            EdaChannel(
                sampling_rate=4,
                normalization="percentile",
                percentile_window=10,
                percentile_cold_start=11,
            )
        assert any("恒退化为 linear" in r.getMessage() for r in caplog.records)

    def test_cold_start_seconds_gt_window_seconds_warns_via_derivation(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """秒制推导路径 footgun：cold_start_seconds > window_seconds → 推导 cold>window → 告警。

        window_seconds=1800 / cold_start_seconds=2400 @ 分析窗长 30s
        → window=60、cold_start=80（80>60）。
        守卫此路径——重构新增暴露面，勿只经显式样本数覆盖路径触发告警。
        """
        with caplog.at_level(logging.WARNING):
            ch = EdaChannel(
                sampling_rate=4,
                normalization="percentile",
                window_seconds=1800.0,
                cold_start_seconds=2400.0,
            )
        assert ch.percentile_window == 60
        assert ch.percentile_cold_start == 80
        assert any("恒退化为 linear" in r.getMessage() for r in caplog.records)

    def test_cold_start_equals_window_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """cold_start == window 是满窗后激活的边界（功能仍可用）→ 严格 > 条件下**不**告警。"""
        with caplog.at_level(logging.WARNING):
            EdaChannel(
                sampling_rate=4,
                normalization="percentile",
                percentile_window=10,
                percentile_cold_start=10,
            )
        assert not any("恒退化为 linear" in r.getMessage() for r in caplog.records)

    def test_cold_start_gt_window_silent_in_linear(self, caplog: pytest.LogCaptureFixture) -> None:
        """linear 模式（percentile 参数无关）即便 cold_start > window 也不告警。"""
        with caplog.at_level(logging.WARNING):
            EdaChannel(
                sampling_rate=4,
                normalization="linear",
                percentile_window=10,
                percentile_cold_start=11,
            )
        assert not any("恒退化为 linear" in r.getMessage() for r in caplog.records)

    async def test_seconds_derived_cold_start_gates_warmup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """经 cold_start_seconds 推导的暖机阈值真正门控冷启动：未达阈值时 == linear。

        cold_start_seconds=150 ÷ 分析窗长 30s → 5；喂 4 次（<5）应恒等 linear。
        """
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        ch_pct = EdaChannel(
            sampling_rate=4,
            normalization="percentile",
            cold_start_seconds=150.0,  # ÷30 → cold_start=5
        )
        assert ch_pct.percentile_cold_start == 5
        phasic_values = [0.1, 0.3, 0.5, 0.2]  # 4 次 < 5，全程冷启动
        for phasic_val in phasic_values:
            ch_lin = EdaChannel(sampling_rate=4, normalization="linear")
            nk_pct = _make_nk_mock(phasic_value=phasic_val)
            nk_lin = _make_nk_mock(phasic_value=phasic_val)
            with patch.dict("sys.modules", {"neurokit2": nk_pct}):
                result_pct = await ch_pct.sense(signal=_make_eda_signal(rate=4))
            with patch.dict("sys.modules", {"neurokit2": nk_lin}):
                result_lin = await ch_lin.sense(signal=_make_eda_signal(rate=4))
            assert result_pct is not None and result_lin is not None
            assert result_pct.mu[1] == pytest.approx(result_lin.mu[1], abs=1e-9)
