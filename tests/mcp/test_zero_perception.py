"""PerceptionHub 多流独立性与 PerceptionChannel 协议结构符合性单测（T7）。

覆盖：
  1. 多通道 collect：每条先验独立保留，数量 == 通道数，不做均值（AD-3 硬约束）。
  2. 两条 mu 分别 (0.8, 0.2) / (-0.6, 0.5)，断言两条都在、没有被平均成一条。
  3. 单通道 sense() 抛异常 → 被跳过（warning），其余先验不受影响。
  4. 单通道 sense() 返回 None → 被跳过（warning），其余先验不受影响。
  5. as_zero_streams 形状 = list[(name, (v,a), (Πv,Πa))]。
  6. Q3：state_overrides 过渡路径已撤下（Zero 回传 2026-07-14 否决 text_affect 挪用）。
  7. isinstance(channel, PerceptionChannel) True（runtime_checkable 协议符合性）。
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.models.zero_affect import ModalityPrior
from src.mcp.zero.perception import PerceptionChannel, PerceptionHub

# ---------------------------------------------------------------------------
# 辅助构造函数
# ---------------------------------------------------------------------------


def _make_prior(
    modality: str = "vision",
    mu: tuple[float, float] = (0.0, 0.0),
    precision: tuple[float, float] = (0.5, 0.5),
) -> ModalityPrior:
    """构造合法 ModalityPrior。"""
    return ModalityPrior(modality=modality, mu=mu, precision=precision)


def _make_channel(
    name: str,
    sense_return: ModalityPrior | None | Exception,
) -> Any:
    """构造满足 PerceptionChannel Protocol 结构的假通道对象。

    sense_return:
    - ModalityPrior  → sense() 返回该先验。
    - None           → sense() 返回 None（无证据）。
    - Exception 实例 → sense() 抛出该异常。
    """
    channel = MagicMock()
    channel.name = name
    if isinstance(sense_return, BaseException):
        channel.sense = AsyncMock(side_effect=sense_return)
    else:
        channel.sense = AsyncMock(return_value=sense_return)
    return channel


# ---------------------------------------------------------------------------
# 1. 多通道 collect：独立保留，不均值
# ---------------------------------------------------------------------------


class TestPerceptionHubCollect:
    """PerceptionHub.collect 多流独立性——核心 AD-3 断言。"""

    async def test_two_channels_both_preserved(self) -> None:
        """两条 mu 不同的先验，collect 后两条都在，数量 == 2。"""
        prior_a = _make_prior("vision", mu=(0.8, 0.2), precision=(0.6, 0.6))
        prior_b = _make_prior("audio", mu=(-0.6, 0.5), precision=(0.4, 0.4))

        hub = PerceptionHub(
            [
                _make_channel("vision", prior_a),
                _make_channel("audio", prior_b),
            ]
        )
        priors = await hub.collect()

        assert len(priors) == 2, f"应保留 2 条先验，实际 {len(priors)}"
        mus = [p.mu for p in priors]
        assert (pytest.approx(0.8), pytest.approx(0.2)) in mus, "视觉先验 mu 应在结果中"
        assert (pytest.approx(-0.6), pytest.approx(0.5)) in mus, "音频先验 mu 应在结果中"

    async def test_no_averaging_of_mu(self) -> None:
        """关键 AD-3 断言：两条 mu 不应被平均成一条中间值。

        (0.8, 0.2) 与 (-0.6, 0.5) 的均值为 (0.1, 0.35)——若结果中出现该值说明做了融合（错误）。
        """
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
        averaged_v = (0.8 + -0.6) / 2  # = 0.1
        averaged_a = (0.2 + 0.5) / 2  # = 0.35
        for mu in mus:
            assert mu[0] != pytest.approx(averaged_v) or mu[1] != pytest.approx(averaged_a), (
                f"collect 返回了均值 mu={mu}，违反 AD-3 独立保留原则"
            )

    async def test_three_channels_all_preserved(self) -> None:
        """三通道全部成功，collect 结果数量为 3、顺序与通道一致。"""
        priors = [
            _make_prior("ch_0", mu=(0.5, 0.1), precision=(0.3, 0.3)),
            _make_prior("ch_1", mu=(-0.3, 0.8), precision=(0.5, 0.5)),
            _make_prior("ch_2", mu=(0.0, -0.5), precision=(0.2, 0.2)),
        ]
        channels = [_make_channel(f"ch_{i}", p) for i, p in enumerate(priors)]
        hub = PerceptionHub(channels)
        result = await hub.collect()

        assert len(result) == 3
        assert [p.modality for p in result] == ["ch_0", "ch_1", "ch_2"]

    async def test_empty_channels_returns_empty(self) -> None:
        """无通道时 collect 返回空列表。"""
        hub = PerceptionHub([])
        result = await hub.collect()
        assert result == []


# ---------------------------------------------------------------------------
# 2. 单通道异常 / None 降级跳过
# ---------------------------------------------------------------------------


class TestPerceptionHubDegradation:
    """单通道失败不拖垮其他通道。"""

    async def test_exception_in_one_channel_skipped_with_warning(self, caplog: Any) -> None:
        """单通道 sense() 抛异常 → 被跳过，其余先验不受影响，有 warning 日志。"""
        good_prior = _make_prior("audio", mu=(0.3, 0.4), precision=(0.5, 0.5))
        hub = PerceptionHub(
            [
                _make_channel("vision", RuntimeError("感知设备超时")),
                _make_channel("audio", good_prior),
            ]
        )

        with caplog.at_level(logging.WARNING, logger="src.mcp.zero.perception"):
            priors = await hub.collect()

        assert len(priors) == 1
        assert priors[0].modality == "audio"
        assert "vision" in caplog.text  # warning 包含通道名

    async def test_none_from_channel_skipped_with_warning(self, caplog: Any) -> None:
        """单通道 sense() 返回 None → 被跳过，其余先验不受影响，有 warning 日志。"""
        good_prior = _make_prior("physio", mu=(0.1, 0.2), precision=(0.6, 0.6))
        hub = PerceptionHub(
            [
                _make_channel("vision", None),
                _make_channel("physio", good_prior),
            ]
        )

        with caplog.at_level(logging.WARNING, logger="src.mcp.zero.perception"):
            priors = await hub.collect()

        assert len(priors) == 1
        assert priors[0].modality == "physio"
        assert "vision" in caplog.text

    async def test_all_channels_fail_returns_empty(self) -> None:
        """所有通道都抛异常时，collect 返回空列表，不 raise。"""
        hub = PerceptionHub(
            [
                _make_channel("ch0", RuntimeError("err0")),
                _make_channel("ch1", RuntimeError("err1")),
            ]
        )
        priors = await hub.collect()
        assert priors == []

    async def test_exception_and_none_and_success_mixed(self) -> None:
        """混合异常、None、成功三种情况，只保留成功的先验。"""
        good = _make_prior("text", mu=(0.0, 0.5), precision=(0.3, 0.3))
        hub = PerceptionHub(
            [
                _make_channel("fail", RuntimeError("bad")),
                _make_channel("none_ch", None),
                _make_channel("text", good),
            ]
        )
        priors = await hub.collect()
        assert len(priors) == 1
        assert priors[0].modality == "text"


# ---------------------------------------------------------------------------
# 3. as_zero_streams 形状
# ---------------------------------------------------------------------------


class TestAsZeroStreams:
    """as_zero_streams 静态方法：形状 = list[(name, (v,a), (Πv,Πa))]。"""

    def test_shape_matches_zero_affect_core_format(self) -> None:
        """两条先验 → as_zero_streams 返回长度为 2 的列表，每条为三元组。"""
        priors = [
            _make_prior("vision", mu=(0.8, 0.2), precision=(0.6, 0.6)),
            _make_prior("audio", mu=(-0.3, 0.5), precision=(0.4, 0.4)),
        ]
        streams = PerceptionHub.as_zero_streams(priors)

        assert len(streams) == 2
        for stream in streams:
            assert len(stream) == 3
            name, mu, prec = stream
            assert isinstance(name, str)
            assert isinstance(mu, tuple) and len(mu) == 2
            assert isinstance(prec, tuple) and len(prec) == 2

    def test_values_match_priors(self) -> None:
        """as_zero_streams 各条值与原先验一致。"""
        prior = _make_prior("vision", mu=(0.7, -0.1), precision=(0.5, 0.8))
        streams = PerceptionHub.as_zero_streams([prior])

        name, mu, prec = streams[0]
        assert name == "vision"
        assert mu == (pytest.approx(0.7), pytest.approx(-0.1))
        assert prec == (pytest.approx(0.5), pytest.approx(0.8))

    def test_empty_priors_returns_empty_list(self) -> None:
        """空先验列表 → 空 streams 列表。"""
        assert PerceptionHub.as_zero_streams([]) == []

    def test_order_preserved(self) -> None:
        """streams 顺序与先验列表顺序一致。"""
        priors = [
            _make_prior(f"ch{i}", mu=(float(i) * 0.1, 0.0), precision=(0.5, 0.5)) for i in range(5)
        ]
        streams = PerceptionHub.as_zero_streams(priors)
        assert [s[0] for s in streams] == [f"ch{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# 4. Q3：state_overrides 过渡路径已撤下（Zero 回传 2026-07-14）
# ---------------------------------------------------------------------------


class TestNoStateOverridesTransitional:
    """Q3 决议：借 text_affect 的 state_overrides 过渡路径会被 PerceptionAgent
    每轮覆盖、不生效，Zero 明确否决。正式入口是 Zero 将新增的 external_priors 字段
    （需 Zero 走 PRP+议会门）。锁定 PerceptionHub 不提供 as_state_overrides，
    避免误发无效过渡路径。
    """

    def test_hub_does_not_expose_state_overrides(self) -> None:
        """PerceptionHub 不应提供 as_state_overrides（Q3 已撤下）。"""
        assert not hasattr(PerceptionHub, "as_state_overrides")

    def test_as_zero_streams_still_available(self) -> None:
        """as_zero_streams 仍在（正式多流形状，待接 external_priors）。"""
        prior = _make_prior("vision", mu=(0.5, -0.3), precision=(0.6, 0.6))
        streams = PerceptionHub.as_zero_streams([prior])
        assert streams == [("vision", (pytest.approx(0.5), pytest.approx(-0.3)), (0.6, 0.6))]


# ---------------------------------------------------------------------------
# 5. isinstance 协议符合性（runtime_checkable）
# ---------------------------------------------------------------------------


class TestPerceptionChannelProtocol:
    """PerceptionChannel 为 runtime_checkable Protocol，结构符合即通过 isinstance。"""

    def test_mock_channel_satisfies_protocol(self) -> None:
        """构造满足 PerceptionChannel 结构的 Mock 对象，isinstance 为 True。"""
        prior = _make_prior("test", mu=(0.1, 0.2), precision=(0.5, 0.5))
        channel = _make_channel("test", prior)
        # runtime_checkable 仅检查方法存在性，不检查 async
        assert isinstance(channel, PerceptionChannel)

    def test_object_without_sense_fails_protocol(self) -> None:
        """缺少 sense 方法的对象 isinstance 为 False。"""

        class BadChannel:
            name = "no_sense"

        assert not isinstance(BadChannel(), PerceptionChannel)

    def test_object_without_name_fails_protocol(self) -> None:
        """缺少 name 属性（仅有 sense 方法）的对象 isinstance 为 False。

        Python 3.12 的 runtime_checkable 会把 Protocol 的数据属性注解一并纳入
        __protocol_attrs__（此处为 {'sense', 'name'}），isinstance 时按 hasattr 检查——
        故缺 name 的对象通不过。此测试锁定「name 是契约的一部分」这一约束。
        """

        class SenseOnlyChannel:
            async def sense(self) -> None:
                return None

        assert not isinstance(SenseOnlyChannel(), PerceptionChannel)

    def test_perception_hub_is_not_protocol(self) -> None:
        """PerceptionHub 本身不是 Protocol，是具体类。"""
        prior = _make_prior("m", mu=(0.0, 0.0), precision=(0.5, 0.5))
        hub = PerceptionHub([_make_channel("m", prior)])
        assert isinstance(hub, PerceptionHub)
