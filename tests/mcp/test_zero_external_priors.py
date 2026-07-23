"""external_priors 接线接口单测（T1 + Q3 契约收口）。

覆盖：
  1. 常量：EXTERNAL_PRIOR_SCHEMA_VERSION==1；PHYSIO_STREAM_PREFIXES 含 5 前缀；
     MIN_PRECISION==1e-3；cap/max 默认镜像 Zero（0.8 / 5）。
  2. build_external_priors_override：形状/值/顺序正确，逐维 tuple 精度保留；空 priors；
     physio 高 Πv 原样透传（Zero 侧才覆写）。
  3. M6 上界：显式 max_streams 优先；默认与 None 均对齐 Zero ZERO_MAX_EXTERNAL_STREAMS=5；
     env 覆盖；边界（0、等于上界）。
  4. is_physio_stream：生理类前缀（完整/层级/下划线命名）→ True；非生理/前缀子串/大写 → False
     （严格 advisory 命名自查）。
  5. M3 精度上界（precision_cap）：非生理流 Πv/Πa>cap → raise；生理流 Πv 按 MIN 计豁免
     （镜像 Zero M2-先于-M3，含大写/裸前缀流名），生理流 Πa 仍校验；显式/env cap；env 非法值。
  6. recommended_precision：各模态默认与 design.md §五一致；env 可调；physio Πv 恒 MIN。
  7. build_recommended_prior：以推荐精度构造 ModalityPrior；physio 强制命名前缀；coping 透传。
"""

from __future__ import annotations

import pytest

from src.agents.models.zero_affect import ModalityPrior
from src.mcp.zero.external_priors import (
    EXTERNAL_PRIOR_SCHEMA_VERSION,
    MIN_PRECISION,
    PHYSIO_STREAM_PREFIXES,
    ZERO_EXTERNAL_PRIOR_PRECISION_CAP_DEFAULT,
    ZERO_MAX_EXTERNAL_STREAMS_DEFAULT,
    ModalityKind,
    build_external_priors_override,
    build_recommended_prior,
    is_physio_stream,
    recommended_precision,
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

    def test_min_precision_mirrors_zero(self) -> None:
        """MIN_PRECISION 镜像 Zero affect_math.py MIN_PRECISION == 1e-3。"""
        assert MIN_PRECISION == pytest.approx(1e-3)

    def test_precision_cap_default_is_zero_default(self) -> None:
        """M3 精度上界默认镜像 Zero AffectState.external_prior_precision_cap == 0.8。"""
        assert ZERO_EXTERNAL_PRIOR_PRECISION_CAP_DEFAULT == pytest.approx(0.8)

    def test_max_streams_default_is_zero_default(self) -> None:
        """M6 流数上界默认镜像 Zero AffectState.max_external_streams == 5。"""
        assert ZERO_MAX_EXTERNAL_STREAMS_DEFAULT == 5


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
        prior = _make_prior("audio", mu=(0.2, 0.3), precision=(0.6, 0.7))
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

    def test_explicit_high_max_streams_allows_large_input(self) -> None:
        """显式给足够大的 max_streams 时，传入很多条先验也不 raise。"""
        priors = [_make_prior(f"ch_{i}", mu=(0.0, 0.0), precision=(0.5, 0.5)) for i in range(20)]
        result = build_external_priors_override(priors, max_streams=20)
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

    def test_default_max_streams_enforces_zero_default(self) -> None:
        """不传 max_streams（默认）→ 对齐 Zero ZERO_MAX_EXTERNAL_STREAMS=5：6 条 raise。"""
        priors = [_make_prior(f"ch_{i}") for i in range(6)]
        with pytest.raises(ValueError):
            build_external_priors_override(priors)

    def test_default_max_streams_allows_five(self) -> None:
        """不传 max_streams（默认 5）→ 5 条（等于上界）不 raise。"""
        priors = [_make_prior(f"ch_{i}") for i in range(5)]
        result = build_external_priors_override(priors)
        assert len(result["external_priors"]) == 5

    def test_max_streams_none_resolves_zero_default(self) -> None:
        """max_streams=None 显式传入 → 解析为 Zero 默认 5（非「跳过检查」）：6 条 raise。"""
        priors = [_make_prior(f"ch_{i}") for i in range(6)]
        with pytest.raises(ValueError):
            build_external_priors_override(priors, max_streams=None)

    def test_max_streams_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """env ZERO_MAX_EXTERNAL_STREAMS 覆盖默认上界（两仓同名旋钮同步）。"""
        monkeypatch.setenv("ZERO_MAX_EXTERNAL_STREAMS", "8")
        priors = [_make_prior(f"ch_{i}") for i in range(8)]
        # 8 条在 env=8 下不 raise
        result = build_external_priors_override(priors)
        assert len(result["external_priors"]) == 8
        # 9 条超 env=8 → raise
        priors_over = [_make_prior(f"ch_{i}") for i in range(9)]
        with pytest.raises(ValueError):
            build_external_priors_override(priors_over)

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


# ---------------------------------------------------------------------------
# 5. M3 精度上界（precision_cap）客户端 fail-fast
# ---------------------------------------------------------------------------


class TestM3PrecisionCap:
    """M3 客户端校验：非生理流 Πv/Πa ≤ cap；超上界 raise。

    生理流 Πv 按 MIN_PRECISION 计（镜像 Zero M2-先于-M3），故高 Πv 透传不误报；
    生理流 Πa 仍照常校验上界。
    """

    def test_non_physio_precision_within_cap_passes(self) -> None:
        """非生理流 Πv/Πa 恰等于默认上界 0.8 → 不 raise。"""
        prior = _make_prior("vision", mu=(0.1, 0.2), precision=(0.8, 0.8))
        result = build_external_priors_override([prior])
        assert result["external_priors"][0][2] == pytest.approx((0.8, 0.8))

    def test_non_physio_pi_v_over_cap_raises(self) -> None:
        """非生理流 Πv 超默认上界 0.8 → raise，消息含流名与上界。"""
        prior = _make_prior("vision", mu=(0.1, 0.2), precision=(0.9, 0.5))
        with pytest.raises(ValueError, match="vision") as exc_info:
            build_external_priors_override([prior])
        assert "0.8" in str(exc_info.value)

    def test_non_physio_pi_a_over_cap_raises(self) -> None:
        """非生理流 Πa 超默认上界 0.8 → raise。"""
        prior = _make_prior("audio", mu=(0.1, 0.2), precision=(0.5, 0.95))
        with pytest.raises(ValueError):
            build_external_priors_override([prior])

    def test_physio_high_pi_v_passes_mirrors_zero_m2(self) -> None:
        """生理流高 Πv（0.9>cap）不 raise：按 MIN 计（Zero M2 会覆写）；且原值透传。"""
        prior = _make_prior("eda/sc", mu=(0.0, 0.6), precision=(0.9, 0.5))
        result = build_external_priors_override([prior])
        # 原 Πv 透传（0.9），Zero 侧才覆写为 MIN
        assert result["external_priors"][0][2] == pytest.approx((0.9, 0.5))

    def test_physio_high_pi_a_over_cap_raises(self) -> None:
        """生理流 Πa 超上界仍 raise（Πa 不被 M2 覆写，照常校验）。"""
        prior = _make_prior("hrv/rmssd", mu=(0.0, 0.6), precision=(0.5, 0.9))
        with pytest.raises(ValueError):
            build_external_priors_override([prior])

    def test_uppercase_physio_high_pi_v_passes(self) -> None:
        """大写生理流名（'EDA/SC'）高 Πv 不 raise：M3 豁免忠实镜像 Zero name.lower()。

        回归 BLOCK 1——客户端不得比 Zero 更严：Zero M2 大小写不敏感会先覆写 Πv=MIN 必过 M3。
        """
        prior = _make_prior("EDA/SC", mu=(0.0, 0.6), precision=(0.9, 0.5))
        result = build_external_priors_override([prior])
        # 原值透传，Zero 侧才覆写
        assert result["external_priors"][0][2] == pytest.approx((0.9, 0.5))

    def test_raw_prefix_physio_high_pi_v_passes(self) -> None:
        """裸前缀生理流名（'edax'，无分隔符）高 Πv 不 raise：镜像 Zero 裸 startswith。

        回归 BLOCK 1——Zero M2 用 name.lower().startswith(前缀)（无需 '/'/'_' 分隔符），
        'edax' 在 Zero 侧触发覆写；客户端 M3 豁免须一致，否则误拒。
        """
        prior = _make_prior("edax", mu=(0.0, 0.6), precision=(0.9, 0.5))
        result = build_external_priors_override([prior])
        assert result["external_priors"][0][2] == pytest.approx((0.9, 0.5))

    def test_explicit_precision_cap_lower_raises(self) -> None:
        """显式 precision_cap=0.3，Πv=0.5>0.3 → raise。"""
        prior = _make_prior("vision", mu=(0.0, 0.0), precision=(0.5, 0.2))
        with pytest.raises(ValueError):
            build_external_priors_override([prior], precision_cap=0.3)

    def test_explicit_precision_cap_higher_passes(self) -> None:
        """显式 precision_cap=0.95，Πv=Πa=0.9 → 不 raise。"""
        prior = _make_prior("vision", mu=(0.0, 0.0), precision=(0.9, 0.9))
        result = build_external_priors_override([prior], precision_cap=0.95)
        assert result["external_priors"][0][2] == pytest.approx((0.9, 0.9))

    def test_precision_cap_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """env ZERO_EXTERNAL_PRIOR_PRECISION_CAP 覆盖默认上界（两仓同名旋钮同步）。"""
        monkeypatch.setenv("ZERO_EXTERNAL_PRIOR_PRECISION_CAP", "0.95")
        prior = _make_prior("vision", mu=(0.0, 0.0), precision=(0.9, 0.9))
        # env=0.95 下 0.9 不 raise
        result = build_external_priors_override([prior])
        assert result["external_priors"][0][2] == pytest.approx((0.9, 0.9))

    def test_precision_cap_env_lower_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """env ZERO_EXTERNAL_PRIOR_PRECISION_CAP=0.3 下 Πv=0.5 → raise。"""
        monkeypatch.setenv("ZERO_EXTERNAL_PRIOR_PRECISION_CAP", "0.3")
        prior = _make_prior("vision", mu=(0.0, 0.0), precision=(0.5, 0.2))
        with pytest.raises(ValueError):
            build_external_priors_override([prior])


# ---------------------------------------------------------------------------
# 6. recommended_precision（各模态推荐精度默认）
# ---------------------------------------------------------------------------


class TestRecommendedPrecision:
    """recommended_precision：各模态默认与 design.md §五一致；env 可调；physio Πv 恒 MIN。"""

    def test_face_defaults(self) -> None:
        """FACE 默认 (Πv, Πa) == (0.20, 0.12)。"""
        assert recommended_precision(ModalityKind.FACE) == pytest.approx((0.20, 0.12))

    def test_audio_defaults(self) -> None:
        """AUDIO 默认 (Πv, Πa) == (0.10, 0.25)。"""
        assert recommended_precision(ModalityKind.AUDIO) == pytest.approx((0.10, 0.25))

    def test_physio_defaults(self) -> None:
        """PHYSIO 默认 (Πv, Πa) == (MIN_PRECISION, 0.18)。"""
        pi_v, pi_a = recommended_precision(ModalityKind.PHYSIO)
        assert pi_v == pytest.approx(MIN_PRECISION)
        assert pi_a == pytest.approx(0.18)

    def test_face_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EXTERNAL_FACE_PRECISION_V/A env 覆盖 FACE 推荐精度。"""
        monkeypatch.setenv("EXTERNAL_FACE_PRECISION_V", "0.30")
        monkeypatch.setenv("EXTERNAL_FACE_PRECISION_A", "0.22")
        assert recommended_precision(ModalityKind.FACE) == pytest.approx((0.30, 0.22))

    def test_physio_pi_v_forced_min_even_with_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """physio Πv 恒 MIN：即便 env 调高 EXTERNAL_PHYSIO_PRECISION_V 也归 MIN。"""
        monkeypatch.setenv("EXTERNAL_PHYSIO_PRECISION_V", "0.5")
        monkeypatch.setenv("EXTERNAL_PHYSIO_PRECISION_A", "0.30")
        pi_v, pi_a = recommended_precision(ModalityKind.PHYSIO)
        assert pi_v == pytest.approx(MIN_PRECISION)
        assert pi_a == pytest.approx(0.30)

    def test_recommended_precision_within_default_cap(self) -> None:
        """各模态推荐 Πa 均 ≤ 默认 cap 0.8（保证 build_external_priors_override 不误 raise）。"""
        for kind in ModalityKind:
            pi_v, pi_a = recommended_precision(kind)
            assert pi_v <= ZERO_EXTERNAL_PRIOR_PRECISION_CAP_DEFAULT
            assert pi_a <= ZERO_EXTERNAL_PRIOR_PRECISION_CAP_DEFAULT


# ---------------------------------------------------------------------------
# 7. build_recommended_prior（按模态推荐精度构造 ModalityPrior）
# ---------------------------------------------------------------------------


class TestBuildRecommendedPrior:
    """build_recommended_prior：以推荐精度构造 ModalityPrior；physio 强制命名前缀。"""

    def test_face_prior_uses_recommended_precision(self) -> None:
        """FACE 先验精度 == recommended_precision(FACE)。"""
        prior = build_recommended_prior("vision", (0.5, 0.3), ModalityKind.FACE)
        assert prior.modality == "vision"
        assert prior.mu == pytest.approx((0.5, 0.3))
        assert prior.precision == pytest.approx((0.20, 0.12))

    def test_audio_prior_uses_recommended_precision(self) -> None:
        """AUDIO 先验精度 == recommended_precision(AUDIO)。"""
        prior = build_recommended_prior("audio", (-0.2, 0.6), ModalityKind.AUDIO)
        assert prior.precision == pytest.approx((0.10, 0.25))

    def test_physio_prior_with_prefix_ok(self) -> None:
        """physio 先验带生理前缀 → 成功；Πv=MIN。"""
        prior = build_recommended_prior("eda/sc", (0.0, 0.7), ModalityKind.PHYSIO)
        assert prior.precision[0] == pytest.approx(MIN_PRECISION)
        assert prior.precision[1] == pytest.approx(0.18)

    def test_physio_prior_without_prefix_raises(self) -> None:
        """physio kind 但 name 无生理前缀 → raise ValueError（Zero M2 无法触发）。"""
        with pytest.raises(ValueError, match="physio"):
            build_recommended_prior("skin", (0.0, 0.7), ModalityKind.PHYSIO)

    def test_coping_passthrough(self) -> None:
        """coping 参数透传到 ModalityPrior.coping。"""
        prior = build_recommended_prior("vision", (0.1, 0.2), ModalityKind.FACE, coping=0.4)
        assert prior.coping == pytest.approx(0.4)

    def test_recommended_prior_accepted_by_build_override(self) -> None:
        """build_recommended_prior 产物能过 build_external_priors_override（精度默认在 cap 内）。"""
        priors = [
            build_recommended_prior("vision", (0.5, 0.3), ModalityKind.FACE),
            build_recommended_prior("audio", (-0.2, 0.6), ModalityKind.AUDIO),
            build_recommended_prior("eda/sc", (0.0, 0.7), ModalityKind.PHYSIO),
        ]
        result = build_external_priors_override(priors)
        assert len(result["external_priors"]) == 3


# ---------------------------------------------------------------------------
# 8. env 解析健壮性（非法/越界值 fail-fast 带语境，区别于 M3/M6 业务错误）
# ---------------------------------------------------------------------------


class TestEnvResolution:
    """cap/max 的 env 解析：非法值/越界值 raise 带 env 名的 ValueError（回归 BLOCK 2 + W2）。"""

    def test_invalid_env_precision_cap_raises_named_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ZERO_EXTERNAL_PRIOR_PRECISION_CAP 非数值 → raise，消息含 env 名（可与 M3 错误区分）。"""
        monkeypatch.setenv("ZERO_EXTERNAL_PRIOR_PRECISION_CAP", "abc")
        with pytest.raises(ValueError, match="ZERO_EXTERNAL_PRIOR_PRECISION_CAP"):
            build_external_priors_override([_make_prior("vision")])

    def test_nonpositive_env_precision_cap_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_EXTERNAL_PRIOR_PRECISION_CAP≤0 → raise（镜像 Zero gt=0.0 约束）。"""
        monkeypatch.setenv("ZERO_EXTERNAL_PRIOR_PRECISION_CAP", "-0.5")
        with pytest.raises(ValueError, match="ZERO_EXTERNAL_PRIOR_PRECISION_CAP"):
            build_external_priors_override([_make_prior("vision")])

    def test_invalid_env_max_streams_raises_named_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ZERO_MAX_EXTERNAL_STREAMS 非整数 → raise，消息含 env 名。"""
        monkeypatch.setenv("ZERO_MAX_EXTERNAL_STREAMS", "abc")
        with pytest.raises(ValueError, match="ZERO_MAX_EXTERNAL_STREAMS"):
            build_external_priors_override([_make_prior("vision")])

    def test_negative_env_max_streams_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZERO_MAX_EXTERNAL_STREAMS<0 → raise（镜像 Zero ge=0 约束）。"""
        monkeypatch.setenv("ZERO_MAX_EXTERNAL_STREAMS", "-1")
        with pytest.raises(ValueError, match="ZERO_MAX_EXTERNAL_STREAMS"):
            build_external_priors_override([_make_prior("vision")])

    def test_valid_env_values_apply(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """合法 env 值正常生效：cap=0.95 放行 0.9，max=1 放行 1 条。"""
        monkeypatch.setenv("ZERO_EXTERNAL_PRIOR_PRECISION_CAP", "0.95")
        monkeypatch.setenv("ZERO_MAX_EXTERNAL_STREAMS", "1")
        prior = _make_prior("vision", mu=(0.0, 0.0), precision=(0.9, 0.9))
        result = build_external_priors_override([prior])
        assert result["external_priors"][0][2] == pytest.approx((0.9, 0.9))
