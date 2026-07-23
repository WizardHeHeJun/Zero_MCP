"""CallablePerceptionChannel 单测（zero-link T1）。

覆盖：
  1. isinstance(CallablePerceptionChannel("vision", fn), PerceptionChannel) 为 True。
  2. sense() 委托 sense_fn：AsyncMock 返回 ModalityPrior → sense() 返回同一对象。
  3. sense() 委托 sense_fn：AsyncMock 返回 None → sense() 返回 None。
  4. 与 PerceptionHub 集成：多通道 collect() 独立保留各先验，不均值。
  5. 单通道 sense_fn 抛异常时 PerceptionHub 降级跳过，其余先验不受影响。
  6. name 属性按构造参数正确赋值。
  7. sense_fn 按 callable 协议只被调用一次（每次 sense() 调用一次）。
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.agents.models.zero_affect import ModalityPrior
from src.mcp.zero.channels import CallablePerceptionChannel
from src.mcp.zero.perception import PerceptionChannel, PerceptionHub

# ---------------------------------------------------------------------------
# 辅助构造函数
# ---------------------------------------------------------------------------


def _make_prior(
    modality: str = "vision",
    mu: tuple[float, float] = (0.5, 0.3),
    precision: tuple[float, float] = (0.5, 0.5),
) -> ModalityPrior:
    """构造合法 ModalityPrior。"""
    return ModalityPrior(modality=modality, mu=mu, precision=precision)


def _make_channel(
    name: str,
    return_value: ModalityPrior | None,
) -> CallablePerceptionChannel:
    """用 AsyncMock 构造 CallablePerceptionChannel。"""
    sense_fn: AsyncMock = AsyncMock(return_value=return_value)
    return CallablePerceptionChannel(name=name, sense_fn=sense_fn)


def _make_failing_channel(name: str, exc: BaseException) -> CallablePerceptionChannel:
    """构造 sense_fn 抛异常的 CallablePerceptionChannel。"""
    sense_fn: AsyncMock = AsyncMock(side_effect=exc)
    return CallablePerceptionChannel(name=name, sense_fn=sense_fn)


# ---------------------------------------------------------------------------
# 1. PerceptionChannel 协议符合性
# ---------------------------------------------------------------------------


class TestCallablePerceptionChannelProtocol:
    """CallablePerceptionChannel 满足 PerceptionChannel Protocol（runtime_checkable）。"""

    def test_isinstance_perception_channel(self) -> None:
        """CallablePerceptionChannel 实例的 isinstance(PerceptionChannel) 为 True。"""
        fn: AsyncMock = AsyncMock(return_value=None)
        ch = CallablePerceptionChannel(name="vision", sense_fn=fn)
        assert isinstance(ch, PerceptionChannel)

    def test_name_attribute_set_correctly(self) -> None:
        """name 属性按构造参数正确赋值。"""
        fn: AsyncMock = AsyncMock(return_value=None)
        ch = CallablePerceptionChannel(name="audio", sense_fn=fn)
        assert ch.name == "audio"

    def test_sense_fn_attribute_stored(self) -> None:
        """sense_fn 属性按构造参数正确存储。"""
        fn: AsyncMock = AsyncMock(return_value=None)
        ch = CallablePerceptionChannel(name="physio", sense_fn=fn)
        assert ch.sense_fn is fn

    def test_has_sense_method(self) -> None:
        """CallablePerceptionChannel 实例具有 sense 方法。"""
        fn: AsyncMock = AsyncMock(return_value=None)
        ch = CallablePerceptionChannel(name="vision", sense_fn=fn)
        assert callable(ch.sense)


# ---------------------------------------------------------------------------
# 2. sense() 委托行为
# ---------------------------------------------------------------------------


class TestCallablePerceptionChannelSense:
    """sense() 正确委托 sense_fn，返回值透传。"""

    async def test_sense_returns_modality_prior(self) -> None:
        """sense_fn 返回 ModalityPrior 时，sense() 返回同一对象。"""
        prior = _make_prior("vision", mu=(0.8, 0.2))
        fn: AsyncMock = AsyncMock(return_value=prior)
        ch = CallablePerceptionChannel(name="vision", sense_fn=fn)

        result = await ch.sense()

        assert result is prior

    async def test_sense_returns_none_when_fn_returns_none(self) -> None:
        """sense_fn 返回 None 时，sense() 返回 None。"""
        fn: AsyncMock = AsyncMock(return_value=None)
        ch = CallablePerceptionChannel(name="audio", sense_fn=fn)

        result = await ch.sense()

        assert result is None

    async def test_sense_calls_fn_exactly_once(self) -> None:
        """每次调用 sense() 恰好调用 sense_fn 一次。"""
        prior = _make_prior("physio")
        fn: AsyncMock = AsyncMock(return_value=prior)
        ch = CallablePerceptionChannel(name="physio", sense_fn=fn)

        await ch.sense()

        fn.assert_awaited_once()

    async def test_sense_calls_fn_multiple_times(self) -> None:
        """多次调用 sense()，sense_fn 被调用相同次数。"""
        prior = _make_prior("text")
        fn: AsyncMock = AsyncMock(return_value=prior)
        ch = CallablePerceptionChannel(name="text", sense_fn=fn)

        await ch.sense()
        await ch.sense()
        await ch.sense()

        assert fn.await_count == 3

    async def test_sense_propagates_exception_from_fn(self) -> None:
        """sense_fn 抛异常时，sense() 也应传播该异常（PerceptionHub 捕获处理）。"""
        fn: AsyncMock = AsyncMock(side_effect=RuntimeError("传感器故障"))
        ch = CallablePerceptionChannel(name="fail_ch", sense_fn=fn)

        with pytest.raises(RuntimeError, match="传感器故障"):
            await ch.sense()


# ---------------------------------------------------------------------------
# 3. 与 PerceptionHub 集成
# ---------------------------------------------------------------------------


class TestCallableChannelWithPerceptionHub:
    """CallablePerceptionChannel 与 PerceptionHub 集成行为。"""

    async def test_hub_collects_from_callable_channels(self) -> None:
        """PerceptionHub([CallablePerceptionChannel(...)]).collect() 收集各通道先验。"""
        prior_a = _make_prior("vision", mu=(0.8, 0.2), precision=(0.6, 0.6))
        prior_b = _make_prior("audio", mu=(-0.6, 0.5), precision=(0.4, 0.4))

        hub = PerceptionHub(
            [
                _make_channel("vision", prior_a),
                _make_channel("audio", prior_b),
            ]
        )
        priors = await hub.collect()

        assert len(priors) == 2

    async def test_hub_preserves_each_prior_independently(self) -> None:
        """collect 后各先验独立保留，不做均值融合（AD-3）。"""
        prior_a = _make_prior("vision", mu=(0.8, 0.2), precision=(0.6, 0.6))
        prior_b = _make_prior("audio", mu=(-0.6, 0.5), precision=(0.4, 0.4))

        hub = PerceptionHub(
            [
                _make_channel("vision", prior_a),
                _make_channel("audio", prior_b),
            ]
        )
        priors = await hub.collect()

        mus = [p.mu for p in priors]
        # 两条 mu 都应在结果中
        assert (pytest.approx(0.8), pytest.approx(0.2)) in mus
        assert (pytest.approx(-0.6), pytest.approx(0.5)) in mus
        # 不应有均值 (0.1, 0.35)
        averaged_v = (0.8 + -0.6) / 2
        averaged_a = (0.2 + 0.5) / 2
        for mu in mus:
            is_average = mu[0] == pytest.approx(averaged_v) and mu[1] == pytest.approx(averaged_a)
            assert not is_average, f"collect 返回了均值 mu={mu}，违反 AD-3"

    async def test_hub_degrades_on_exception(self, caplog: Any) -> None:
        """单通道 sense_fn 抛异常时 PerceptionHub 降级跳过，其余先验不受影响。"""
        good_prior = _make_prior("audio", mu=(0.3, 0.4), precision=(0.5, 0.5))

        hub = PerceptionHub(
            [
                _make_failing_channel("vision", RuntimeError("传感器超时")),
                _make_channel("audio", good_prior),
            ]
        )

        with caplog.at_level(logging.WARNING, logger="src.mcp.zero.perception"):
            priors = await hub.collect()

        assert len(priors) == 1
        assert priors[0].modality == "audio"
        # 有包含通道名的 warning 日志
        assert "vision" in caplog.text

    async def test_hub_skips_none_returns(self) -> None:
        """单通道返回 None 时被跳过，其余先验不受影响。"""
        good_prior = _make_prior("physio", mu=(0.1, 0.2), precision=(0.6, 0.6))

        hub = PerceptionHub(
            [
                _make_channel("vision", None),
                _make_channel("physio", good_prior),
            ]
        )
        priors = await hub.collect()

        assert len(priors) == 1
        assert priors[0].modality == "physio"

    async def test_as_zero_streams_with_callable_channels(self) -> None:
        """CallablePerceptionChannel + PerceptionHub.as_zero_streams 输出形状正确。"""
        prior_a = _make_prior("vision", mu=(0.8, 0.2), precision=(0.6, 0.6))
        prior_b = _make_prior("audio", mu=(-0.6, 0.5), precision=(0.4, 0.4))

        hub = PerceptionHub(
            [
                _make_channel("vision", prior_a),
                _make_channel("audio", prior_b),
            ]
        )
        priors = await hub.collect()
        streams = PerceptionHub.as_zero_streams(priors)

        # 两条独立流，不均值
        assert len(streams) == 2
        for stream in streams:
            name, mu, prec = stream
            assert isinstance(name, str)
            assert isinstance(mu, tuple) and len(mu) == 2
            assert isinstance(prec, tuple) and len(prec) == 2

        # 验证具体值
        stream_map = {s[0]: s for s in streams}
        assert "vision" in stream_map
        assert "audio" in stream_map
        assert stream_map["vision"][1] == (pytest.approx(0.8), pytest.approx(0.2))
        assert stream_map["audio"][1] == (pytest.approx(-0.6), pytest.approx(0.5))


# ---------------------------------------------------------------------------
# 4. 顶层包导出验证
# ---------------------------------------------------------------------------


class TestTopLevelExport:
    """CallablePerceptionChannel 通过 src.mcp.zero 顶层包可访问。"""

    def test_exported_from_top_level_package(self) -> None:
        """从 src.mcp.zero 顶层导入 CallablePerceptionChannel 成功。"""
        from src.mcp.zero import CallablePerceptionChannel as CPC  # noqa: PLC0415

        fn: AsyncMock = AsyncMock(return_value=None)
        ch = CPC(name="test", sense_fn=fn)
        assert isinstance(ch, PerceptionChannel)
