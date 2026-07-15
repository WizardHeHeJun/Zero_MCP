"""external_priors 接线接口单测（T1）。

覆盖：
  1. build_external_priors_override：两条 ModalityPrior → 形状/值/顺序正确，
     逐维 tuple 精度保留（不被压成标量）。
  2. M6 上界：max_streams=2 传 3 条 → raise ValueError（消息含上限/条数）；
     max_streams=3 传 3 条 → 通过；max_streams=None → 不检查（传很多条不 raise）。
  3. is_physio_stream：生理类前缀（完整/层级/下划线命名）→ True；
     非生理类 → False；前缀子串但非前缀（如 'hrvx'）→ False（按实现约定）。
  4. 常量：EXTERNAL_PRIOR_SCHEMA_VERSION == 1；PHYSIO_STREAM_PREFIXES 含 5 前缀。
  5. 空 priors → {"external_priors": []}。
"""

from __future__ import annotations

import pytest

from src.agents.models.zero_affect import ModalityPrior
from src.mcp.zero.external_priors import (
    EXTERNAL_PRIOR_SCHEMA_VERSION,
    PHYSIO_STREAM_PREFIXES,
    build_external_priors_override,
    is_physio_stream,
)

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


# ---------------------------------------------------------------------------
# 1. 常量断言
# ---------------------------------------------------------------------------


class TestConstants:
    """EXTERNAL_PRIOR_SCHEMA_VERSION / PHYSIO_STREAM_PREFIXES 常量约束。"""

    def test_schema_version_is_one(self) -> None:
        """EXTERNAL_PRIOR_SCHEMA_VERSION 必须为 1（M5 锚点版本）。"""
        assert EXTERNAL_PRIOR_SCHEMA_VERSION == 1

    def test_physio_stream_prefixes_has_five_entries(self) -> None:
        """PHYSIO_STREAM_PREFIXES 应包含 5 个前缀。"""
        assert len(PHYSIO_STREAM_PREFIXES) == 5

    def test_physio_stream_prefixes_contains_expected(self) -> None:
        """PHYSIO_STREAM_PREFIXES 包含 physio / eda / hrv / pupil / scr。"""
        expected = {"physio", "eda", "hrv", "pupil", "scr"}
        assert expected == set(PHYSIO_STREAM_PREFIXES)


# ---------------------------------------------------------------------------
# 2. build_external_priors_override：形状/值/顺序/精度
# ---------------------------------------------------------------------------


class TestBuildExternalPriorsOverride:
    """build_external_priors_override 正常路径——形状、值、顺序、精度保留。"""

    def test_two_priors_shape_and_key(self) -> None:
        """两条先验 → 返回 dict，唯一键为 'external_priors'，列表长度 == 2。"""
        priors = [
            _make_prior("vision", mu=(0.6, 0.2), precision=(0.8, 0.4)),
            _make_prior("audio", mu=(-0.3, 0.5), precision=(0.5, 0.7)),
        ]
        result = build_external_priors_override(priors)
        assert isinstance(result, dict)
        assert list(result.keys()) == ["external_priors"]
        assert len(result["external_priors"]) == 2

    def test_values_match_priors_first(self) -> None:
        """第一条元组的 name / mu / precision 与原先验一致。"""
        priors = [
            _make_prior("vision", mu=(0.6, 0.2), precision=(0.8, 0.4)),
            _make_prior("audio", mu=(-0.3, 0.5), precision=(0.5, 0.7)),
        ]
        result = build_external_priors_override(priors)
        name, mu, prec = result["external_priors"][0]
        assert name == "vision"
        assert mu == pytest.approx((0.6, 0.2))
        assert prec == pytest.approx((0.8, 0.4))

    def test_values_match_priors_second(self) -> None:
        """第二条元组的 name / mu / precision 与原先验一致。"""
        priors = [
            _make_prior("vision", mu=(0.6, 0.2), precision=(0.8, 0.4)),
            _make_prior("audio", mu=(-0.3, 0.5), precision=(0.5, 0.7)),
        ]
        result = build_external_priors_override(priors)
        name, mu, prec = result["external_priors"][1]
        assert name == "audio"
        assert mu == pytest.approx((-0.3, 0.5))
        assert prec == pytest.approx((0.5, 0.7))

    def test_order_preserved(self) -> None:
        """结果列表顺序与输入先验顺序一致。"""
        names = ["ch_0", "ch_1", "ch_2"]
        priors = [
            _make_prior(n, mu=(0.1 * i, 0.0), precision=(0.3, 0.3)) for i, n in enumerate(names)
        ]
        result = build_external_priors_override(priors)
        result_names = [item[0] for item in result["external_priors"]]
        assert result_names == names

    def test_mu_is_tuple_not_scalar(self) -> None:
        """逐维 tuple 精度：mu 是 (float, float) tuple，不被压成标量。"""
        prior = _make_prior("vision", mu=(0.7, -0.1), precision=(0.5, 0.8))
        result = build_external_priors_override([prior])
        _, mu, _ = result["external_priors"][0]
        assert isinstance(mu, tuple), f"mu 应为 tuple，实际类型 {type(mu)}"
        assert len(mu) == 2

    def test_precision_is_tuple_not_scalar(self) -> None:
        """逐维 tuple 精度：precision 是 (float, float) tuple，不被压成标量。"""
        prior = _make_prior("audio", mu=(0.2, 0.3), precision=(0.6, 0.9))
        result = build_external_priors_override([prior])
        _, _, prec = result["external_priors"][0]
        assert isinstance(prec, tuple), f"precision 应为 tuple，实际类型 {type(prec)}"
        assert len(prec) == 2

    def test_physio_prior_passes_through_unchanged(self) -> None:
        """生理类先验（eda 前缀）的值原样传出，MCP 侧不做精度覆写（Zero 侧才覆写）。"""
        prior = _make_prior("eda/sc", mu=(0.0, 0.8), precision=(0.9, 0.7))
        result = build_external_priors_override([prior])
        name, mu, prec = result["external_priors"][0]
        assert name == "eda/sc"
        assert mu == pytest.approx((0.0, 0.8))
        # MCP 侧不覆写：Πv 保持原值（0.9），Zero 侧才覆写为 MIN_PRECISION
        assert prec == pytest.approx((0.9, 0.7))

    def test_empty_priors_returns_empty_list(self) -> None:
        """空先验列表 → {"external_priors": []}。"""
        result = build_external_priors_override([])
        assert result == {"external_priors": []}

    def test_no_max_streams_large_input_does_not_raise(self) -> None:
        """max_streams=None（默认）时，传入很多条先验也不 raise。"""
        priors = [_make_prior(f"ch_{i}", mu=(0.0, 0.0), precision=(0.5, 0.5)) for i in range(20)]
        result = build_external_priors_override(priors)
        assert len(result["external_priors"]) == 20


# ---------------------------------------------------------------------------
# 3. M6 上界检查
# ---------------------------------------------------------------------------


class TestM6MaxStreams:
    """M6 fail-fast：max_streams 超界时 raise ValueError，含条数与上限。"""

    def test_exceeds_max_streams_raises_value_error(self) -> None:
        """max_streams=2，传 3 条 → raise ValueError。"""
        priors = [_make_prior(f"ch_{i}") for i in range(3)]
        with pytest.raises(ValueError):
            build_external_priors_override(priors, max_streams=2)

    def test_error_message_contains_count_and_limit(self) -> None:
        """ValueError 消息应同时包含实际条数（3）和上限（2）。"""
        priors = [_make_prior(f"ch_{i}") for i in range(3)]
        with pytest.raises(ValueError, match="3") as exc_info:
            build_external_priors_override(priors, max_streams=2)
        # 消息含上限
        assert "2" in str(exc_info.value)

    def test_equal_to_max_streams_does_not_raise(self) -> None:
        """max_streams=3，传 3 条（等于上限）→ 不 raise，正常返回。"""
        priors = [_make_prior(f"ch_{i}") for i in range(3)]
        result = build_external_priors_override(priors, max_streams=3)
        assert len(result["external_priors"]) == 3

    def test_below_max_streams_does_not_raise(self) -> None:
        """max_streams=5，传 2 条（低于上限）→ 不 raise。"""
        priors = [_make_prior(f"ch_{i}") for i in range(2)]
        result = build_external_priors_override(priors, max_streams=5)
        assert len(result["external_priors"]) == 2

    def test_max_streams_none_never_raises(self) -> None:
        """max_streams=None → 不做客户端检查，传大量条数不 raise。"""
        priors = [_make_prior(f"ch_{i}") for i in range(100)]
        result = build_external_priors_override(priors, max_streams=None)
        assert len(result["external_priors"]) == 100

    def test_max_streams_zero_raises_on_any_prior(self) -> None:
        """max_streams=0，传 1 条 → raise ValueError（边界：0 上界仍严格检查）。"""
        priors = [_make_prior("vision")]
        with pytest.raises(ValueError):
            build_external_priors_override(priors, max_streams=0)

    def test_max_streams_zero_with_empty_priors_passes(self) -> None:
        """max_streams=0，传 0 条 → 不 raise（0 <= 0）。"""
        result = build_external_priors_override([], max_streams=0)
        assert result == {"external_priors": []}


# ---------------------------------------------------------------------------
# 4. is_physio_stream
# ---------------------------------------------------------------------------


class TestIsPhysioStream:
    """is_physio_stream 匹配规则：完整匹配 / 前缀+"/" / 前缀+"_" → True；其余 False。"""

    @pytest.mark.parametrize(
        "name",
        [
            # 完整前缀匹配
            "physio",
            "eda",
            "hrv",
            "pupil",
            "scr",
        ],
    )
    def test_exact_prefix_returns_true(self, name: str) -> None:
        """精确匹配前缀 → True。"""
        assert is_physio_stream(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            # 层级命名（前缀 + "/"）
            "hrv/left",
            "eda/sc",
            "physio/heart_rate",
            "pupil/diameter",
            "scr/galvanic",
        ],
    )
    def test_slash_subpath_returns_true(self, name: str) -> None:
        """前缀 + '/' 命名 → True（层级流名）。"""
        assert is_physio_stream(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            # 下划线命名（前缀 + "_"）
            "pupil_diameter",
            "eda_level",
            "hrv_rmssd",
            "physio_signal",
            "scr_amplitude",
        ],
    )
    def test_underscore_suffix_returns_true(self, name: str) -> None:
        """前缀 + '_' 命名 → True（下划线流名）。"""
        assert is_physio_stream(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "vision",
            "audio",
            "text",
        ],
    )
    def test_non_physio_streams_return_false(self, name: str) -> None:
        """非生理类流名 → False。"""
        assert is_physio_stream(name) is False

    @pytest.mark.parametrize(
        "name",
        [
            # 前缀是子串但不是前缀本身（如 'edax', 'hrvx'）→ 按实现约定应为 False
            "edax",
            "hrvx",
            "pupilx",
            "physioX",
            "scrX",
            # 纯空串
            "",
        ],
    )
    def test_prefix_substring_not_prefix_returns_false(self, name: str) -> None:
        """前缀子串但非前缀（无 '/' 或 '_' 分隔符）→ False（严格匹配规则）。"""
        assert is_physio_stream(name) is False

    def test_case_sensitive(self) -> None:
        """匹配区分大小写：'EDA'、'HRV' 不匹配（前缀均小写）。"""
        assert is_physio_stream("EDA") is False
        assert is_physio_stream("HRV") is False
        assert is_physio_stream("Pupil") is False
