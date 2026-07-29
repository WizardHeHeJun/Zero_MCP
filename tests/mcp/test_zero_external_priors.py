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

import logging
import math  # 自点燃硬顶的期望值一律现算（禁手抄 0.2536，回件 §6-8 数值订正 1）
from typing import Any

import pytest

from src.agents.models.zero_affect import ModalityPrior
from src.mcp.zero import external_priors as external_priors_module
from src.mcp.zero.external_priors import (
    _MERGE_OMEGA_WARN_MARKER,
    EXTERNAL_PRIOR_SCHEMA_VERSION,
    MIN_PRECISION,
    PHYSIO_MERGE_OMEGA_DEFAULT,
    PHYSIO_MERGED_MODALITY,
    PHYSIO_PRECISION_A_SELF_IGNITE_BOUND,
    PHYSIO_STREAM_PREFIXES,
    PHYSIO_SUBSOURCE_PRECISION_A,
    ZERO_EXTERNAL_PRIOR_PRECISION_CAP_DEFAULT,
    ZERO_MAX_EXTERNAL_STREAMS_DEFAULT,
    ZERO_SALIENCE_THRESHOLD,
    ModalityKind,
    _assert_merge_arity_invariant,
    _physio_self_ignite_salience,
    _triggers_zero_m2,
    build_external_priors_override,
    build_recommended_prior,
    is_physio_stream,
    merge_physio_priors,
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
        """生理类先验（eda 前缀）的值原样传出，MCP 侧不做精度覆写（Zero 侧才覆写）。

        ⚠ Πa 取 0.20（原为 0.70）：M8 自点燃硬顶落地后，physio 流的 Πa 上界不再是 M3 的
        cap=0.8 而是 ~0.359。本例的**被测语义是「Πv 原样透传、MCP 不覆写」**，Πa 只是陪跑
        取值，故按新契约取一个合法值即可——不是放宽断言，透传断言逐值不变。
        """
        prior = _make_prior("eda/sc", mu=(0.0, 0.8), precision=(0.9, 0.20))
        result = build_external_priors_override([prior])
        name, mu, prec = result["external_priors"][0]
        assert name == "eda/sc"
        assert mu == pytest.approx((0.0, 0.8))
        # MCP 侧不覆写：Πv 保持原值（0.9），Zero 侧才覆写为 MIN_PRECISION
        assert prec == pytest.approx((0.9, 0.20))

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


class TestM7MuDomain:
    """M7 客户端校验：出线 μ 各维 ∈[-1,1]，镜像 Zero
    `src/agents/affect_math.py::expand_external_priors` 的 M7 μ 域 fail-fast。

    背景：Zero commit `0d4edb1`（2026-07-28 20:00，已在其 main）新增 M7 fail-fast——
    越界 μ 从「静默降级」变成 `raise ValueError`。经其 server 包成 ToolError → 我方
    `graceful_step` 降级 None = **整轮 step 静默丢失**（不是只丢一条流），故必须在客户端拦住。

    **本类的存在理由是「契约类有校验 ≠ 无绕过路径」**：`ModalityPrior` 的构造期校验
    与 frozen 只覆盖「正常构造 + 构造后赋值」，而 `model_construct` / `model_copy` /
    鸭子类型伪造均可绕过；且本函数默认 `merge_physio=True`，合并会**产出新的 μ**，
    校验入参根本看不到它。故守卫必须作用于 `as_stream()` 的**出线 tuple**。
    以下每个用例对应一条**实测跑通过**的绕过口（守卫落地前它们把 (7.7, nan) 原样送出网）。
    """

    def test_model_construct_bypass_is_caught(self) -> None:
        """`model_construct()` 跳过全部校验造出的越界先验，被出境守卫拦下。"""
        rogue = ModalityPrior.model_construct(modality="audio", mu=(9.9, 0.0), precision=(0.2, 0.2))
        with pytest.raises(ValueError, match="M7") as exc_info:
            build_external_priors_override([rogue])
        assert "audio" in str(exc_info.value)

    def test_model_copy_bypass_is_caught(self) -> None:
        """`model_copy(update=)` 不触发 validator，越界 μ 被出境守卫拦下。"""
        rogue = _make_prior("vision").model_copy(update={"mu": (5.0, -9.0)})
        with pytest.raises(ValueError, match="M7"):
            build_external_priors_override([rogue])

    def test_duck_typed_fake_prior_is_caught(self) -> None:
        """鸭子类型伪造（非 ModalityPrior 实例）也被拦下。

        `PerceptionHub` 收集通道产物时只判 `BaseException`/`None`、**不判类型**，
        故伪造对象能混入 priors 列表——守卫读 `as_stream()` 输出而非模型实例，正好覆盖。

        ⚠ 伪造体必须带 `modality` 属性才构成真实威胁：缺它会先在
        `merge_physio_priors`（`external_priors.py:481` 读 `prior.modality`）炸
        `AttributeError`，那是**崩在别处**、不能算本守卫拦下的（初版本例正是这样
        误报为通过——「测试红了要先看它红在哪一行」）。
        """

        class FakePrior:
            modality = "vision"  # 非生理前缀 → merge 阶段原样透传，得以走到 M7 守卫

            def as_stream(self) -> tuple[str, tuple[float, float], tuple[float, float]]:
                return ("vision", (7.7, float("nan")), (0.2, 0.1))

        with pytest.raises(ValueError, match="M7"):
            build_external_priors_override([FakePrior()])  # type: ignore[list-item]

    def test_nan_mu_is_caught(self) -> None:
        """NaN 的 μ 被拦下（`-1.0 <= nan <= 1.0` 恒 False → 取反成立）。"""
        rogue = ModalityPrior.model_construct(
            modality="vision", mu=(0.0, float("nan")), precision=(0.2, 0.2)
        )
        with pytest.raises(ValueError, match="M7"):
            build_external_priors_override([rogue])

    def test_boundary_mu_passes_not_stricter_than_zero(self) -> None:
        """μ=±1.0 边界放行——不得比 Zero M7 更严（其判据用 `<=`，同样放行）。

        守卫过严会把合法载荷拦在客户端，是与漏拦同样真实的失败模式。
        """
        prior = _make_prior("vision", mu=(-1.0, 1.0), precision=(0.2, 0.2))
        result = build_external_priors_override([prior])
        assert result["external_priors"][0][1] == pytest.approx((-1.0, 1.0))

    def test_guard_sees_post_merge_mu_not_input_mu(self) -> None:
        """守卫作用于**合并后**的出线 μ——证明它没被装在函数入口。

        默认 `merge_physio=True` 会把 EDA/HRV 合成单条 physio 流并产出新的 μ_a
        （精度加权）与 μ_v（硬置 0.0）。本例两条入参各自合法，合并后仍合法 →
        不该 raise；同时断言出线 μ 确实**是合并产物而非任一入参原值**，
        从而证明守卫读到的是合并之后的东西。
        """
        eda = _make_prior("eda/sc", mu=(0.0, 0.76), precision=(1e-3, 0.15))
        hrv = _make_prior("hrv/rmssd", mu=(0.0, 0.91), precision=(1e-3, 0.20))
        result = build_external_priors_override([eda, hrv])
        streams = result["external_priors"]
        assert len(streams) == 1, "默认应合并为单条 physio 流"
        merged_mu_a = streams[0][1][1]
        assert merged_mu_a not in (pytest.approx(0.76), pytest.approx(0.91)), (
            "出线 μ_a 应是合并产物，若等于某条入参原值说明守卫/合并顺序有误"
        )
        assert 0.76 < merged_mu_a < 0.91, f"精度加权均值应落在两入参之间，实际 {merged_mu_a}"


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

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_precision_is_caught(self, bad: float) -> None:
        """NaN/inf 的 Π 被出境守卫拦下——`value > cap` 对 NaN **恒 False**，会静默穿过上界关。

        ⚠ 这条比越界 μ 更隐蔽：Zero `src/agents/affect_math.py::expand_external_priors`
        的 M7 只守 μ 不守 Π，其 M3 两关（`pi_v <= 0.0` 正值关、`pi_v > precision_cap`
        上界关）对 NaN 同样恒 False → NaN 精度会一路进入融合数学产出 NaN 后验。

        本守卫是 MCP 侧**单边兜底**，不依赖对方状态。「对方有守卫」是随时会变的运行时
        事实——2026-07-29 一天内实测经历三态：Zero main `11c25b0` 无 → 其未提交工作树
        出现 M3′（`math.isfinite` 前置于两条比较）→ main `332cb40` 起落地。出网收口点
        必须在我方：删掉本守卫，我方就把 NaN 拦截权交给了一个当天变了三次的外部状态。
        """
        rogue = ModalityPrior.model_construct(
            modality="vision", mu=(0.1, 0.1), precision=(bad, 0.2)
        )
        with pytest.raises(ValueError, match="有限值"):
            build_external_priors_override([rogue])

    # ⚠ 下面三条 physio「高 Πv 豁免」用例的 Πa 由 0.5 降为 0.20：M8 自点燃硬顶落地后
    # physio 的 Πa 上界是 ~0.359 而非 M3 的 cap=0.8。三条的**被测语义都是 Πv 那一维的
    # M3 豁免**（0.9 > cap 仍放行），Πa 只是陪跑；保持 0.5 会让它们红在 M8 上，反而**测不到
    # 原本要测的 M3 豁免**。故降 Πa 是为保住判别力，不是放宽断言——Πv=0.9>cap 逐值不变。
    def test_physio_high_pi_v_passes_mirrors_zero_m2(self) -> None:
        """生理流高 Πv（0.9>cap）不 raise：按 MIN 计（Zero M2 会覆写）；且原值透传。"""
        prior = _make_prior("eda/sc", mu=(0.0, 0.6), precision=(0.9, 0.20))
        result = build_external_priors_override([prior])
        # 原 Πv 透传（0.9），Zero 侧才覆写为 MIN
        assert result["external_priors"][0][2] == pytest.approx((0.9, 0.20))

    def test_physio_high_pi_a_over_cap_raises(self) -> None:
        """生理流 Πa 超上界仍 raise（Πa 不被 M2 覆写，照常校验）。

        ⚠ Πa=0.9 现在**同时**越 M3 cap（0.8）与 M8 自点燃硬顶（~0.359），两条守卫都会拦。
        故断言必须 `match="M3"` 钉住是**哪一条**在拦——否则本例会在 M3 被误删时靠 M8 假绿，
        变成一条测不到自己名字里那条守卫的用例。M3 在循环里先于 M8 执行，此为其执行序 pin。
        """
        prior = _make_prior("hrv/rmssd", mu=(0.0, 0.6), precision=(0.5, 0.9))
        with pytest.raises(ValueError, match="M3 精度超上界"):
            build_external_priors_override([prior])

    def test_uppercase_physio_high_pi_v_passes(self) -> None:
        """大写生理流名（'EDA/SC'）高 Πv 不 raise：M3 豁免忠实镜像 Zero name.lower()。

        回归 BLOCK 1——客户端不得比 Zero 更严：Zero M2 大小写不敏感会先覆写 Πv=MIN 必过 M3。
        """
        prior = _make_prior("EDA/SC", mu=(0.0, 0.6), precision=(0.9, 0.20))
        result = build_external_priors_override([prior])
        # 原值透传，Zero 侧才覆写
        assert result["external_priors"][0][2] == pytest.approx((0.9, 0.20))

    def test_raw_prefix_physio_high_pi_v_passes(self) -> None:
        """裸前缀生理流名（'edax'，无分隔符）高 Πv 不 raise：镜像 Zero 裸 startswith。

        回归 BLOCK 1——Zero M2 用 name.lower().startswith(前缀)（无需 '/'/'_' 分隔符），
        'edax' 在 Zero 侧触发覆写；客户端 M3 豁免须一致，否则误拒。
        """
        prior = _make_prior("edax", mu=(0.0, 0.6), precision=(0.9, 0.20))
        result = build_external_priors_override([prior])
        assert result["external_priors"][0][2] == pytest.approx((0.9, 0.20))

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


# ---------------------------------------------------------------------------
# EDA/HRV 协方差交叉预合并（Zero 议会 2026-07-28 终裁 · MCP 侧执行项）
# ---------------------------------------------------------------------------


class TestPhysioPreMerge:
    """EDA 与 HRV 高度相关（同测交感唤醒），须预合并为单条 physio 再注入。

    不合并时朴素 Σπ = 0.15+0.20 = 0.35，**相当于把合并精度虚增 2 倍**（议会数学席）。
    合并式（CI 信息形式，ω 固定）：
        Π_merged = ω·Π_eda + (1-ω)·Π_hrv
        μ_merged = (ω·Π_eda·μ_eda + (1-ω)·Π_hrv·μ_hrv) / Π_merged
    """

    def test_merge_precision_matches_council_value(self) -> None:
        """ω=0.5 + 可靠度分层 (0.15, 0.20) → Π_merged=0.175（议会给出的确切值）。"""
        merged = merge_physio_priors(
            [
                _make_prior("eda/sc", mu=(0.0, 0.76), precision=(MIN_PRECISION, 0.18)),
                _make_prior("hrv/rmssd", mu=(0.0, 0.91), precision=(MIN_PRECISION, 0.18)),
            ]
        )
        assert len(merged) == 1, f"EDA+HRV 应合并为单条，实际 {[p.modality for p in merged]}"
        assert merged[0].modality == PHYSIO_MERGED_MODALITY
        assert merged[0].precision[1] == pytest.approx(0.175)
        # 效价维：两者均对效价盲 → μv=0、Πv=MIN（与 Zero M2 最终形状一致）
        assert merged[0].mu[0] == 0.0
        assert merged[0].precision[0] == pytest.approx(MIN_PRECISION)

    def test_merged_mu_is_precision_weighted(self) -> None:
        """μ_merged 为精度加权（非算术平均）——HRV 可靠度更高故更靠近 HRV 读数。"""
        mu_eda, mu_hrv = 0.76, 0.91
        merged = merge_physio_priors(
            [
                _make_prior("eda/sc", mu=(0.0, mu_eda), precision=(MIN_PRECISION, 0.18)),
                _make_prior("hrv/rmssd", mu=(0.0, mu_hrv), precision=(MIN_PRECISION, 0.18)),
            ]
        )
        w_eda = 0.5 * PHYSIO_SUBSOURCE_PRECISION_A["eda"]
        w_hrv = 0.5 * PHYSIO_SUBSOURCE_PRECISION_A["hrv"]
        expected = (w_eda * mu_eda + w_hrv * mu_hrv) / (w_eda + w_hrv)
        assert merged[0].mu[1] == pytest.approx(expected)
        # 判别性：精度加权 ≠ 算术平均（否则本例退化为无差别断言）
        assert merged[0].mu[1] != pytest.approx((mu_eda + mu_hrv) / 2.0)
        assert mu_eda < merged[0].mu[1] < mu_hrv, "加权结果应落两读数之间且偏向高可靠度侧"

    def test_merged_stream_still_triggers_zero_m2(self) -> None:
        """合并流名须仍落生理前缀集内，否则 Zero M2（Πv→MIN）不再触发。"""
        assert is_physio_stream(PHYSIO_MERGED_MODALITY)
        assert PHYSIO_MERGED_MODALITY.startswith(PHYSIO_STREAM_PREFIXES[0])

    def test_single_source_is_not_merged(self) -> None:
        """只有 EDA（或只有 HRV）时不合并——无相关性双计问题。"""
        only_eda = [_make_prior("eda/sc", mu=(0.0, 0.5), precision=(MIN_PRECISION, 0.18))]
        assert [p.modality for p in merge_physio_priors(only_eda)] == ["eda/sc"]
        only_hrv = [_make_prior("hrv/rmssd", mu=(0.0, 0.5), precision=(MIN_PRECISION, 0.18))]
        assert [p.modality for p in merge_physio_priors(only_hrv)] == ["hrv/rmssd"]

    def test_non_physio_streams_untouched_and_order_kept(self) -> None:
        """非生理流原样保留且保持顺序；合并流落在首个生理流原位置。"""
        priors = [
            _make_prior("audio", mu=(0.1, 0.4), precision=(0.10, 0.25)),
            _make_prior("eda/sc", mu=(0.0, 0.7), precision=(MIN_PRECISION, 0.18)),
            _make_prior("vision", mu=(0.5, 0.1), precision=(0.20, 0.12)),
            _make_prior("hrv/rmssd", mu=(0.0, 0.9), precision=(MIN_PRECISION, 0.18)),
        ]
        names = [p.modality for p in merge_physio_priors(priors)]
        assert names == ["audio", PHYSIO_MERGED_MODALITY, "vision"]

    def test_omega_env_override_and_bounds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ω 走 env 可覆盖；端点 0/1（1 维 CI 退化角，议会已排除）fail-fast。"""
        pair = [
            _make_prior("eda/sc", mu=(0.0, 0.0), precision=(MIN_PRECISION, 0.18)),
            _make_prior("hrv/rmssd", mu=(0.0, 1.0), precision=(MIN_PRECISION, 0.18)),
        ]
        monkeypatch.setenv("ZERO_PHYSIO_MERGE_OMEGA", "0.571")
        merged = merge_physio_priors(pair)
        w_eda = 0.571 * PHYSIO_SUBSOURCE_PRECISION_A["eda"]
        w_hrv = (1.0 - 0.571) * PHYSIO_SUBSOURCE_PRECISION_A["hrv"]
        assert merged[0].precision[1] == pytest.approx(w_eda + w_hrv)
        for bad in ("0.0", "1.0", "-0.2"):
            monkeypatch.setenv("ZERO_PHYSIO_MERGE_OMEGA", bad)
            with pytest.raises(ValueError, match="开区间"):
                merge_physio_priors(pair)

    def test_override_merges_by_default_and_can_opt_out(self) -> None:
        """注入边界默认合并；merge_physio=False 保留旧行为（逃生阀）。"""
        priors = [
            _make_prior("eda/sc", mu=(0.0, 0.7), precision=(MIN_PRECISION, 0.18)),
            _make_prior("hrv/rmssd", mu=(0.0, 0.9), precision=(MIN_PRECISION, 0.18)),
        ]
        default = build_external_priors_override(priors)
        assert [name for name, _mu, _p in default["external_priors"]] == [PHYSIO_MERGED_MODALITY]
        opted_out = build_external_priors_override(priors, merge_physio=False)
        assert [name for name, _mu, _p in opted_out["external_priors"]] == ["eda/sc", "hrv/rmssd"]

    def test_omega_half_gives_hrv_its_exact_reliability_weight(self) -> None:
        """ω=0.5 时 HRV 实际权重**恒等于** Π_hrv/(Π_eda+Π_hrv)——故再按可靠度设 ω 是二次施加。

        对任意 (Π_eda, Π_hrv) 成立的恒等式（Zero 议会 ω 档位终裁的代数核心）：
            w_hrv(ω) = (1-ω)·Π_hrv / [ω·Π_eda + (1-ω)·Π_hrv]，ω=0.5 → Π_hrv/(Π_eda+Π_hrv)
        """
        pi_eda = PHYSIO_SUBSOURCE_PRECISION_A["eda"]
        pi_hrv = PHYSIO_SUBSOURCE_PRECISION_A["hrv"]
        w_hrv = 0.5 * pi_hrv / (0.5 * pi_eda + 0.5 * pi_hrv)
        assert w_hrv == pytest.approx(pi_hrv / (pi_eda + pi_hrv))
        # 判别性：该权重确实 ≠ 0.5（否则本例退化为「对称权重=对称权重」的空断言）
        assert w_hrv != pytest.approx(0.5)

    def test_omega_half_preserves_mu_and_halves_precision(self) -> None:
        """ω=0.5 只调保守度：μ 与「不合并双流进 fuse_terms」逐位相同，Π 精确减半。

        任何 ω≠0.5 会同时扰动 μ 与 Π（把本该正交的两个自由度耦合）——一并作反例守卫。
        """
        mu_eda, mu_hrv = 0.76, 0.91
        pi_eda = PHYSIO_SUBSOURCE_PRECISION_A["eda"]
        pi_hrv = PHYSIO_SUBSOURCE_PRECISION_A["hrv"]
        # Zero fuse_terms 对两条独立流的等效结果
        mu_unmerged = (pi_eda * mu_eda + pi_hrv * mu_hrv) / (pi_eda + pi_hrv)
        pair = [
            _make_prior("eda/sc", mu=(0.0, mu_eda), precision=(MIN_PRECISION, 0.18)),
            _make_prior("hrv/rmssd", mu=(0.0, mu_hrv), precision=(MIN_PRECISION, 0.18)),
        ]
        merged = merge_physio_priors(pair)[0]
        assert merged.mu[1] == pytest.approx(mu_unmerged, abs=1e-12), "ω=0.5 应保持 μ 不变"
        assert (pi_eda + pi_hrv) / merged.precision[1] == pytest.approx(2.0), "Π 应精确减半"
        # 反例：ω≠0.5 会同时挪动 μ（证明「只调一个维度」是 ω=0.5 独有性质）
        shifted = merge_physio_priors(pair, omega=0.4286)[0]
        assert shifted.mu[1] != pytest.approx(mu_unmerged, abs=1e-6)

    def test_merge_arity_invariant_currently_holds(self) -> None:
        """当前子源集合与可靠度权重表一致于二元推导——不变量成立。"""
        _assert_merge_arity_invariant()  # 不抛即通过

    def test_adding_source_without_rederivation_fails_loudly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """⚠ 治理守卫：只加同源子通道、不重走 N 元推导 → 硬失败（非静默降保守度）。

        背景（Zero 议会转 CS 席治理项）：M6 按合并后计数=1，其前提是相关性已被 CI 推导吸收。
        若第三个源被塞进二元式复用 ω=0.5，"physio" 背后会藏 3 条朴素求和的证据，
        **两仓都感知不到 M6 失效**。故此处必须 raise 而非放行。
        """
        monkeypatch.setattr(
            external_priors_module, "_PHYSIO_MERGE_SOURCES", ("eda", "hrv", "rsp"), raising=True
        )
        monkeypatch.setattr(
            external_priors_module,
            "PHYSIO_SUBSOURCE_PRECISION_A",
            {"eda": 0.15, "hrv": 0.20, "rsp": 0.10},
            raising=True,
        )
        with pytest.raises(NotImplementedError, match="N 元 CI 推导"):
            _assert_merge_arity_invariant()
        # 合并入口本身也必须被守住（不是只有辅助函数会抛）
        with pytest.raises(NotImplementedError, match="N 元 CI 推导"):
            merge_physio_priors(
                [
                    _make_prior("eda/sc", mu=(0.0, 0.7), precision=(MIN_PRECISION, 0.18)),
                    _make_prior("hrv/rmssd", mu=(0.0, 0.9), precision=(MIN_PRECISION, 0.18)),
                ]
            )

    def test_merge_avoids_naive_precision_inflation(self) -> None:
        """合并后 Πa 严格小于不合并时的朴素 Σπ——这正是要规避的 2× 虚增。"""
        priors = [
            _make_prior("eda/sc", mu=(0.0, 0.7), precision=(MIN_PRECISION, 0.18)),
            _make_prior("hrv/rmssd", mu=(0.0, 0.9), precision=(MIN_PRECISION, 0.18)),
        ]
        naive_sum = sum(PHYSIO_SUBSOURCE_PRECISION_A.values())
        merged_pi = merge_physio_priors(priors)[0].precision[1]
        assert merged_pi < naive_sum
        assert merged_pi == pytest.approx(naive_sum / 2.0), "对称 ω 下合并精度应为朴素和的一半"


# ---------------------------------------------------------------------------
# 非默认 ω 告警（D-5(b)·Zero 2026-07-29 回执 R6，零回归）
#
# `PHYSIO_MERGE_OMEGA_DEFAULT` docstring 的「仅供实验/对照，生产不应改」原是**自律文字**，
# 代码里非默认值静默接受。R6 要求升级为可执行守卫：warn（不 raise——实验/对照是正当用途，
# 且 :742 的对照用例会设 0.571 并期望成功）。数值行为一字未动。
#
# 判别力四格（两条入口 × 是否默认，逐格实证）：
#   env=0.571        → 发      | 显式 omega=0.6 → 发（env 判据会漏掉的那条通路）
#   env 未设（默认）  → 不发    | 显式 omega=0.5 → 不发
# 两条「不发」都先跑正控，证明该 warn 在同一 caplog 会话里确实可见（防恒真式）。
# ---------------------------------------------------------------------------

_PRIORS_LOGGER = "src.mcp.zero.external_priors"


def _omega_warnings(caplog: Any) -> list[str]:
    """只挑「非默认 ω」那一类 WARNING（按产品侧稳定前缀筛，避免被别的原因染绿/染红）。"""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and _MERGE_OMEGA_WARN_MARKER in record.getMessage()
    ]


def _merge_pair() -> list[ModalityPrior]:
    """EDA+HRV 两条流（会触发合并 ⇒ 会走到 _resolve_merge_omega）。"""
    return [
        _make_prior("eda/sc", mu=(0.0, 0.7), precision=(MIN_PRECISION, 0.18)),
        _make_prior("hrv/rmssd", mu=(0.0, 0.9), precision=(MIN_PRECISION, 0.18)),
    ]


class TestMergeOmegaOverrideWarning:
    """ω≠终裁默认必须出声：它直接改发往 Zero 的 wire Πa，而 Zero 侧无同名旋钮可观测。"""

    def test_env_override_warns(self, monkeypatch: pytest.MonkeyPatch, caplog: Any) -> None:
        """env 通路：ZERO_PHYSIO_MERGE_OMEGA=0.571 → 出声，且数值仍按 0.571 算（不 raise）。"""
        monkeypatch.setenv("ZERO_PHYSIO_MERGE_OMEGA", "0.571")
        with caplog.at_level(logging.WARNING, logger=_PRIORS_LOGGER):
            merged = merge_physio_priors(_merge_pair())
        messages = _omega_warnings(caplog)
        assert len(messages) == 1, f"应恰有一条非默认 ω warn，实际日志：{caplog.text!r}"
        assert "0.571" in messages[0] and "私有旋钮" in messages[0], (
            f"warn 须点明取值与「Zero 无同名旋钮可观测」，实际：{messages[0]!r}"
        )
        # 零回归：只加声音不改数值（与 :742 对照用例同式）
        expected = (
            0.571 * PHYSIO_SUBSOURCE_PRECISION_A["eda"]
            + (1.0 - 0.571) * (PHYSIO_SUBSOURCE_PRECISION_A["hrv"])
        )
        assert merged[0].precision[1] == pytest.approx(expected)

    def test_explicit_argument_override_warns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: Any
    ) -> None:
        """入参通路：omega=0.6 → 出声。

        判据必须写在**解析后的值**上：若按「env 读到没有」判，这条通路会整条漏掉（env 未设，
        照样改了 wire Πa）。故此处刻意把 env 删干净。
        """
        monkeypatch.delenv("ZERO_PHYSIO_MERGE_OMEGA", raising=False)
        with caplog.at_level(logging.WARNING, logger=_PRIORS_LOGGER):
            merge_physio_priors(_merge_pair(), omega=0.6)
        messages = _omega_warnings(caplog)
        assert len(messages) == 1, f"显式入参也须出声，实际日志：{caplog.text!r}"
        assert "0.6" in messages[0]

    def test_default_path_is_silent(self, monkeypatch: pytest.MonkeyPatch, caplog: Any) -> None:
        """**零回归主证**：env 未设、不传 omega（生产默认路径）→ 不出声。

        先跑正控（同一 caplog 会话里 env=0.571 必出声）再断言静默，否则是恒真式。
        """
        with caplog.at_level(logging.WARNING, logger=_PRIORS_LOGGER):
            monkeypatch.setenv("ZERO_PHYSIO_MERGE_OMEGA", "0.571")
            merge_physio_priors(_merge_pair())
            assert _omega_warnings(caplog), "正控失败：warn 不可见，静默断言将是恒真式"
            caplog.clear()
            monkeypatch.delenv("ZERO_PHYSIO_MERGE_OMEGA", raising=False)
            merged = merge_physio_priors(_merge_pair())
        assert _omega_warnings(caplog) == []
        assert merged[0].precision[1] == pytest.approx(
            sum(PHYSIO_SUBSOURCE_PRECISION_A.values()) / 2.0
        )

    def test_explicit_default_value_is_silent(
        self, monkeypatch: pytest.MonkeyPatch, caplog: Any
    ) -> None:
        """显式传入的值**等于**默认 → 不出声（判据是「值非默认」，不是「有没有显式传」）。"""
        monkeypatch.delenv("ZERO_PHYSIO_MERGE_OMEGA", raising=False)
        with caplog.at_level(logging.WARNING, logger=_PRIORS_LOGGER):
            merge_physio_priors(_merge_pair(), omega=0.6)  # 正控
            assert _omega_warnings(caplog), "正控失败：warn 不可见，静默断言将是恒真式"
            caplog.clear()
            merge_physio_priors(_merge_pair(), omega=PHYSIO_MERGE_OMEGA_DEFAULT)
        assert _omega_warnings(caplog) == []

    def test_warn_is_not_deduplicated(self, monkeypatch: pytest.MonkeyPatch, caplog: Any) -> None:
        """有意**不去重**：每次合并各出声一次。

        模块级 `_warned` 标志会让用例顺序相关（先跑的吃掉 warning、后跑的假绿），代价（非默认 ω
        下逐次刷屏）可接受——那本就不该出现在生产。
        """
        monkeypatch.delenv("ZERO_PHYSIO_MERGE_OMEGA", raising=False)
        with caplog.at_level(logging.WARNING, logger=_PRIORS_LOGGER):
            merge_physio_priors(_merge_pair(), omega=0.6)
            merge_physio_priors(_merge_pair(), omega=0.6)
        assert len(_omega_warnings(caplog)) == 2


# ---------------------------------------------------------------------------
# D-6：0.359 自点燃上界的前提锚 —— μv≡0 是**我方**硬写，不是 Zero 的保证
# ---------------------------------------------------------------------------


class TestPhysioOutboundMuVZeroPremise:
    """`PHYSIO_PRECISION_A_SELF_IGNITE_BOUND=0.359` 成立的唯一前提：出线 physio 流 μv≡0。

    归因订正（2026-07-29 跨仓现场核验，Zero 指认成立）：Zero
    `src/agents/affect_math.py::expand_external_priors` 的 M2 分支**只覆写 Πv、从不碰 μ**，
    紧邻的 M7 也只做 μ∈[-1,1] 域校验。μv≡0 完全来自**我方三处硬写**——EdaChannel /
    HrvChannel 各一处 `mu_v = 0.0`，以及 `merge_physio_priors` 出线的 `mu=(0.0, mu_merged_a)`。
    前提一旦破，hypot(μ) 最大到 √2，真实自点燃上界收紧到 `2·T/√2 − MIN`（0.359 松约 30%；
    今日阈值下 = 0.2535584412271571，对外文书写作 0.2536 是**向上取整口径**，判据一律现算）。

    ⚠ **本类的定位已随 M8/M9 落地而变（2026-07-29）**：原文案说「按 μv≡0 闭式复算的既有守卫
    不会报错——文案说缺口在、断言却测不到」，那是 M8/M9 之前的事实。今天出网口有两道运行期
    守卫（M8 按实测 μv 现算收紧、M9 直接拒绝非零 μv），前提破了会**当场红**。本类保留的价值
    是守在**更上游**：`merge_physio_priors` 这一层的硬写锚点本身（出网口之前），且它是三处
    μv 硬写里唯一由本仓算式产出的一处。

    ⚠ 本类补的是一处**实测存在的恒真式缺口**：既有
    `test_merge_precision_matches_council_value` 给两条入参的 μv 都写 0.0，故其
    `assert merged[0].mu[0] == 0.0` 对「把入参 μv 按精度加权带出」这一变异**恒绿**
    （pitfalls⑥ 原型）。本类改喂非零 μv，使该变异真正驱红。
    """

    def test_merge_forces_mu_v_zero_against_nonzero_inputs(self) -> None:
        """入参带**非零且异号**的 μv，合并出线仍须硬置 0.0。

        异号（+0.9 / -0.7）而非同号：若实现改成精度加权带出，同号入参也许仍落在某个
        接近 0 的值上，异号则让加权结果显著偏离 0，变异更易被 `== 0.0` 接住。
        """
        eda = _make_prior("eda/sc", mu=(0.9, 0.4), precision=(MIN_PRECISION, 0.15))
        hrv = _make_prior("hrv/rmssd", mu=(-0.7, 0.6), precision=(MIN_PRECISION, 0.20))
        # 正控：被观测量（入参 μv）必须真的非零，否则下面的断言退化成恒真式。
        assert eda.mu[0] != 0.0 and hrv.mu[0] != 0.0, "入参 μv 全零 → 本用例无判别力"

        merged = merge_physio_priors([eda, hrv])

        assert len(merged) == 1, f"EDA+HRV 应合并为单条，实际 {[p.modality for p in merged]}"
        assert merged[0].mu[0] == 0.0, (
            f"合并出线 μv={merged[0].mu[0]} ≠ 0 —— PHYSIO_PRECISION_A_SELF_IGNITE_BOUND=0.359 "
            "的前提已破，且**违反跨仓协议 §6-8「physio 对效价盲」**（出网口的 M9 会当场拒绝"
            "整条载荷）。若这是有意的口径变更，须先跨仓重开 §6-8，再按 hypot(μ)≤√2 把上界"
            "收紧到 2·T/√2 − MIN（≈0.2536 是向上取整的对外口径，判据现算不手抄）。"
        )
        # μa 仍按 CI 加权正常产出（证明本用例观测的是 μv 那一维，不是「整个 mu 被清空」）
        assert merged[0].mu[1] != 0.0


# ---------------------------------------------------------------------------
# M8：physio 自点燃硬顶 —— 出网收口点的运行期守卫
#
# 背景（Zero 2026-07-29 18:25 裁定件）：我方交付 hrv 残差 σ=1.6 ⇒ Πa=0.39，只核了 Zero 的
# `ZERO_EXTERNAL_PRIOR_PRECISION_CAP=0.8`，漏了本仓自己这条更低的硬顶 0.359。裁定件原话要点：
# **「真正先撞到的是点火阈值，不是精度上界，而后者不会报错、只会静默让 physio 过门。」**
# 本节把那句承诺变成可执行守卫的回归面。
#
# ⚠ 恒真式防线（本仓 pitfalls ⑥）：本节所有「不该红」的用例都必须先证明**被观测量存在**
# （载荷里真有 physio 流 / 真走了合并路径），否则「没有 physio 流时断言恒真」会让整节假绿。
# ---------------------------------------------------------------------------


def _physio_streams(payload: dict[str, list[Any]]) -> list[Any]:
    """从载荷里取出会触发 Zero M2 的 physio 流（按 Zero 的裸前缀判定，非 advisory 命名）。"""
    return [s for s in payload["external_priors"] if _triggers_zero_m2(s[0])]


class TestM8PhysioSelfIgniteBound:
    """出线 physio 流的 Πa 不得高到使其**不经开门动作**即越过 Zero 点燃门。"""

    # -- 判据本身（纯函数层）------------------------------------------------

    def test_bound_reproduces_the_cross_repo_constant_at_zero_mu_v(self) -> None:
        """μv=0 时现算硬顶逐值等于对外承诺的 0.359，且此时 salience 恰好等于阈值。

        这条把「现算式」与「跨仓承诺的常量」焊在一起：任何一侧漂移都红。
        """
        worst, ceiling = _physio_self_ignite_salience(
            (0.0, 0.0), MIN_PRECISION, PHYSIO_PRECISION_A_SELF_IGNITE_BOUND
        )
        assert ceiling == pytest.approx(PHYSIO_PRECISION_A_SELF_IGNITE_BOUND, abs=1e-12), (
            f"μv=0 下现算硬顶 {ceiling} ≠ 对外承诺常量 "
            f"{PHYSIO_PRECISION_A_SELF_IGNITE_BOUND}——两者须成对修改。"
        )
        # 取等即点燃（Zero `_select_fired` 判据是 `s >= threshold`）
        assert worst == pytest.approx(ZERO_SALIENCE_THRESHOLD, abs=1e-12)

    def test_ceiling_tightens_automatically_when_mu_v_nonzero(self) -> None:
        """**选项 B 的核心收益**：μv 变非零 → 硬顶自动收紧，无需任何人回来改常量。

        μv=±1 时 hypot(μv,1)=√2 ⇒ 硬顶 = 2·T/√2 − MIN，即 0.359 松了约 30%。
        原 docstring 说的「缺口在、断言却测不到」正是这一格；按常量比（选项 A）测不到它。

        ⚠ 期望值**现算不手抄**（回件 §6-8 数值订正 1，两仓同款约定）：`0.2536` 是向上取整的
        对外口径、比真值 0.2535584412271571 松 4.16e-5，手抄它等于给断言留一条测不到的缝；
        且 Zero 调 SALIENCE_THRESHOLD 时手抄值不会跟随。现算后容差可收到 1e-12。
        """
        _, ceiling_at_zero = _physio_self_ignite_salience((0.0, 0.0), MIN_PRECISION, 0.1)
        _, ceiling_at_one = _physio_self_ignite_salience((1.0, 0.0), MIN_PRECISION, 0.1)
        _, ceiling_at_half = _physio_self_ignite_salience((0.5, 0.0), MIN_PRECISION, 0.1)

        expected_at_one = 2 * ZERO_SALIENCE_THRESHOLD / math.sqrt(2) - MIN_PRECISION
        assert ceiling_at_one == pytest.approx(expected_at_one, abs=1e-12), (
            f"μv=±1 下硬顶应收紧到现算值 {expected_at_one}，实际 {ceiling_at_one}"
        )
        # 单调收紧：|μv| 越大 → 硬顶越低（不是只有端点对）
        assert ceiling_at_zero > ceiling_at_half > ceiling_at_one
        # 负 μv 与正 μv 对称（hypot 取模，符号无关）
        _, ceiling_at_neg_one = _physio_self_ignite_salience((-1.0, 0.0), MIN_PRECISION, 0.1)
        assert ceiling_at_neg_one == pytest.approx(ceiling_at_one, abs=1e-12)

    def test_guard_is_inclusive_at_the_bound(self) -> None:
        """边界语义 pin：**≥ 硬顶即红**（不是 >），因 Zero `_select_fired` 用 `s >= threshold`。

        逐值三态：0.359 红 / 略低于界绿 / 略高于界红。中间那一格排除「守卫恒红」这一竞争解释。
        """
        at_bound = _make_prior(
            "physio",
            mu=(0.0, 0.5),
            precision=(MIN_PRECISION, PHYSIO_PRECISION_A_SELF_IGNITE_BOUND),
        )
        with pytest.raises(ValueError, match="M8 physio 自点燃越界"):
            build_external_priors_override([at_bound])

        just_below = _make_prior(
            "physio",
            mu=(0.0, 0.5),
            precision=(MIN_PRECISION, PHYSIO_PRECISION_A_SELF_IGNITE_BOUND - 1e-6),
        )
        payload = build_external_priors_override([just_below])
        assert _physio_streams(payload), "正控：恰低于界的用例里必须真有 physio 流被观测到"

        just_above = _make_prior(
            "physio",
            mu=(0.0, 0.5),
            precision=(MIN_PRECISION, PHYSIO_PRECISION_A_SELF_IGNITE_BOUND + 1e-6),
        )
        with pytest.raises(ValueError, match="M8 physio 自点燃越界"):
            build_external_priors_override([just_above])

    # -- env 通路（裁定件点名的那条）----------------------------------------

    def test_env_precision_a_over_bound_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """**判别力①**：env EXTERNAL_PHYSIO_PRECISION_A=0.39（=hrv σ=1.6 的交付值）必红。

        这正是 Zero 裁定件点名的缺口：既有跨仓守卫只看源码常量
        `_RECOMMENDED_PRECISION_DEFAULTS`，对 env 覆盖**恒绿**。
        """
        monkeypatch.setenv("EXTERNAL_PHYSIO_PRECISION_A", "0.39")
        # 正控：env 确实生效了（否则下面的红可能来自别处）
        assert recommended_precision(ModalityKind.PHYSIO)[1] == pytest.approx(0.39)

        prior = build_recommended_prior("hrv/rmssd", (0.0, 0.4), ModalityKind.PHYSIO)
        with pytest.raises(ValueError, match="M8 physio 自点燃越界") as exc:
            build_external_priors_override([prior])

        message = str(exc.value)
        # 原因串必须可读且带归因，不是干巴巴一句
        assert "0.39" in message, f"错误消息未给出实测 Πa：{message}"
        assert "点火阈值" in message and "cap" in message, (
            f"错误消息缺「先撞到的是点火阈值而非精度上界 cap」这句归因：{message}"
        )
        assert "静默" in message, f"错误消息未说明 cap 不会报错、只会静默放行：{message}"
        assert "D7" in message, f"错误消息未说明越界后果（绕过 D7 跨仓承诺）：{message}"

    def test_env_precision_a_below_bound_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """**判别力②**：env EXTERNAL_PHYSIO_PRECISION_A=0.35（< 0.359）必绿。

        与 ① 只差 0.04，证明守卫拦的是**越界**而非「凡 env 覆盖就拦」。
        """
        monkeypatch.setenv("EXTERNAL_PHYSIO_PRECISION_A", "0.35")
        assert recommended_precision(ModalityKind.PHYSIO)[1] == pytest.approx(0.35)

        prior = build_recommended_prior("hrv/rmssd", (0.0, 0.4), ModalityKind.PHYSIO)
        payload = build_external_priors_override([prior])
        # 正控：被观测量存在——载荷里真有一条 physio 流走过了 M8
        streams = _physio_streams(payload)
        assert len(streams) == 1, f"正控失败：载荷里没有 physio 流 {payload}"
        assert streams[0][2][1] == pytest.approx(0.35)

    # -- 零回归主证 ---------------------------------------------------------

    def test_default_recommended_and_merged_precision_stay_green(self) -> None:
        """**零回归主证**：默认推荐态（Πa=0.18）与合并态（Πa=0.175）都不触发。

        两态都带正控，避免「没有 physio 流 ⇒ 恒绿」。
        """
        single = build_recommended_prior("eda/sc", (0.0, 0.9), ModalityKind.PHYSIO)
        payload = build_external_priors_override([single])
        streams = _physio_streams(payload)
        assert len(streams) == 1 and streams[0][2][1] == pytest.approx(0.18), (
            f"正控失败：推荐态 physio 流未出现在载荷里 {payload}"
        )

        eda = _make_prior("eda/sc", mu=(0.0, 0.8), precision=(MIN_PRECISION, 0.18))
        hrv = _make_prior("hrv/rmssd", mu=(0.0, 0.9), precision=(MIN_PRECISION, 0.18))
        merged_payload = build_external_priors_override([eda, hrv])
        merged_streams = _physio_streams(merged_payload)
        assert len(merged_streams) == 1, f"正控失败：合并路径未产出 physio 流 {merged_payload}"
        assert merged_streams[0][0] == PHYSIO_MERGED_MODALITY
        assert merged_streams[0][2][1] == pytest.approx(0.175), (
            "正控失败：合并态 Πa 不是议会确切值 0.175，本例已不在测合并路径"
        )

    def test_no_physio_stream_never_false_positives(self) -> None:
        """**判别力④**：无 physio 流时不误报——即便 Πa 远高于硬顶。

        vision/audio 过阈点燃是它们的本职，M8 只管 physio。Πa=0.7 ≫ 0.359 但 < cap 0.8。
        """
        vision = _make_prior("vision", mu=(0.0, 1.0), precision=(0.7, 0.7))
        audio = _make_prior("audio/prosody", mu=(0.9, 0.9), precision=(0.7, 0.7))
        payload = build_external_priors_override([vision, audio])

        # 正控：确认这两条**确实**不被判为 physio（否则本例变成「因为没流所以不红」的恒真式）
        assert _physio_streams(payload) == [], "vision/audio 不应被判为 physio 流"
        assert len(payload["external_priors"]) == 2
        # 且它们的 Πa 确实越过了 physio 硬顶——证明「不误报」不是因为取值本来就合规
        for _name, _mu, prec in payload["external_priors"]:
            assert prec[1] > PHYSIO_PRECISION_A_SELF_IGNITE_BOUND, (
                "本例的判别力依赖非 physio 流的 Πa 越过 physio 硬顶，否则不红是平凡的"
            )

    # -- 合并路径（校验入参根本看不到的那个值）------------------------------

    def test_merged_precision_over_bound_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """**判别力⑤**：子源可靠度抬高 → 合并出线 Πa 越界 → 必红。

        这一格是「守卫必须落在出网收口点」的正面证明：入参两条先验的 Πa 各为 0.30
        （**都低于硬顶 0.359，逐条看完全合规**），但 `merge_physio_priors` 用的是子源常量
        `PHYSIO_SUBSOURCE_PRECISION_A`、不是入参精度，合并出线 Πa=0.40 才越界。
        任何「校验入参 priors」的实现在这一格上都会假绿。
        """
        monkeypatch.setitem(external_priors_module.PHYSIO_SUBSOURCE_PRECISION_A, "eda", 0.40)
        monkeypatch.setitem(external_priors_module.PHYSIO_SUBSOURCE_PRECISION_A, "hrv", 0.40)

        eda = _make_prior("eda/sc", mu=(0.0, 0.8), precision=(MIN_PRECISION, 0.30))
        hrv = _make_prior("hrv/rmssd", mu=(0.0, 0.9), precision=(MIN_PRECISION, 0.30))
        # 正控：入参逐条合规——红必须来自合并产物，不是来自入参
        for prior in (eda, hrv):
            assert prior.precision[1] < PHYSIO_PRECISION_A_SELF_IGNITE_BOUND

        with pytest.raises(ValueError, match="M8 physio 自点燃越界") as exc:
            build_external_priors_override([eda, hrv])
        # 红在合并后的那条流上（名字是 "physio"，不是 "eda/sc"/"hrv/rmssd"）
        assert PHYSIO_MERGED_MODALITY in str(exc.value)
        assert "0.4" in str(exc.value), f"错误消息未给出合并后的实测 Πa：{exc.value}"

    def test_model_construct_bypass_is_still_caught(self) -> None:
        """构造期校验的四条绕过路径之一（`model_construct`）仍被出网口拦下。

        `ModalityPrior` 是 frozen + validator，但 `model_construct` 完全跳过 validator。
        这正是「唯一必经收口点是出网函数」的理由——守卫若写在模型层，这一格假绿。
        """
        forged = ModalityPrior.model_construct(
            modality="physio/forged", mu=(0.0, 0.1), precision=(MIN_PRECISION, 0.75)
        )
        # 正控：伪造确实绕过了模型校验（对象真的带着越界 Πa 存在）
        assert forged.precision[1] == 0.75
        with pytest.raises(ValueError, match="M8 physio 自点燃越界"):
            build_external_priors_override([forged])

    def test_nonzero_mu_v_path_is_now_owned_by_m9_but_m8_still_tightens(self) -> None:
        """**执行序 pin**：非零 μv 的端到端红，自 M9 落地起归 **M9**（不再是 M8）。

        本用例的前身断言 `match="M8 ..."`——M9（physio 出线 μv 必须恒 0）排在 M8 之前后，
        同一载荷改红在 M9。**这不是判别力缩水**，两件事分别验：
        1. **M8 的收紧判据本身没变**：下面直接调纯函数 `_physio_self_ignite_salience`，
           证明 μv=1.0 时硬顶确实收紧到 < Πa=0.30 —— 即「M8 也会拦」仍成立，只是 M9 先报。
        2. **端到端谁先报**：由 `pytest.raises(match="M9 ...")` 钉死。
        为什么 M9 该先报：M8 的消息会按那个**本就不该存在**的 μv 现算出收紧后的硬顶，把结论
        导向「降 Πa」；照做则契约违反原样上 wire，只是不再点燃——修症状不修病。
        """
        pi_a = 0.30
        assert pi_a < PHYSIO_PRECISION_A_SELF_IGNITE_BOUND, "前提：该 Πa 在 μv=0 下必须合规"

        # μv=0：绿（正控，证明红确实由 μv 引起而非 Πa 本身）
        payload = build_external_priors_override(
            [_make_prior("physio", mu=(0.0, 0.5), precision=(MIN_PRECISION, pi_a))]
        )
        assert _physio_streams(payload), "正控：μv=0 这一格必须真有 physio 流走过 M8"

        # ① M8 判据层：μv=1.0 时硬顶收紧到该 Πa 之下 —— M8 的一般性一字未改
        worst, ceiling = _physio_self_ignite_salience((1.0, 0.5), MIN_PRECISION, pi_a)
        assert ceiling < pi_a and worst >= ZERO_SALIENCE_THRESHOLD, (
            f"M8 判据层已失效：μv=1.0 下硬顶 {ceiling} 应 < Πa={pi_a} 且最坏 salience "
            f"{worst} 应 ≥ {ZERO_SALIENCE_THRESHOLD}"
        )

        # ② 端到端层：M7 允许 μv=1.0，故不会被 M7 抢先；M9 先于 M8 报
        with pytest.raises(ValueError, match="M9 physio 效价契约违反") as exc:
            build_external_priors_override(
                [_make_prior("physio", mu=(1.0, 0.5), precision=(MIN_PRECISION, pi_a))]
            )
        assert "M8" not in str(exc.value), (
            f"执行序错位：非零 μv 应由 M9 独占报错，消息里不该出现 M8：{exc.value}"
        )


# ---------------------------------------------------------------------------
# M9：physio 效价契约（出线 μv ≡ 0.0）—— 跨仓协议里**我方那半**的 fail-fast
#
# 来源：我方 2026-07-29 回件 `notes/2026-07-29-mcp-reply-to-zero-asks.md` §6-8 明确
# **选 (a) 入协议 + 我方同时在出境侧加 fail-fast**（原话：「不是让你方一家兜……两侧各自
# 封一半，0.359 才成为**两侧各自的**结构不变量」）。Zero 已落它那半（M2 分支
# `mu = (0.0, mu[1])`，commit 8043176）；本节是我方那半的回归面。
#
# ⚠ **M8 顶不了 M9**（本节最核心的一格）：M8 拦的是「Πa 高到能自点燃」，一条
# μv=0.9、Πa=0.05 的伪造 physio 流最坏 salience≈0.034 ≪ 0.18 —— **M8 结构上必然放行**，
# 却违反了「生理对效价盲」这条被承诺要 fail-fast 的契约。下面 ① 那格用**正控**把
# 「M8 放行」显式测出来，而不是嘴上声称。
#
# ⚠ 恒真式防线（pitfalls ⑥）：所有「不该红」的用例必须先证明被观测量存在（载荷里真有
# physio 流 / 该流的 μv 真是非零），否则「没有 physio 流 ⇒ 断言恒真」会让整节假绿。
# ---------------------------------------------------------------------------


class TestM9PhysioValenceContract:
    """出线 physio 流的 μv 必须恒 0.0，否则出网收口点 fail-fast。"""

    # -- 前提：模型层不设防，出网口是唯一收口点 ------------------------------

    def test_model_layer_does_not_enforce_the_contract(self) -> None:
        """正控前提：`ModalityPrior` 层**允许** physio 流带非零 μv（μv=0.9 合法构造）。

        这解释了 M9 为何必须落在出网收口点：模型层的 `_validate_ranges` 只校验 μ∈[-1,1]
        与 Π 有限>0，**没有任何一条把 physio 的 μv 钉 0**；再叠加 model_construct /
        model_copy / 鸭子类型三条绕过口，信任模型实例等于没有守卫。
        """
        legit = ModalityPrior(
            modality="physio/forged", mu=(0.9, 0.4), precision=(MIN_PRECISION, 0.05)
        )
        assert legit.mu[0] == 0.9, "前提破：模型层若已拦下非零 μv，M9 的存在理由需重新论证"

    # -- 判别力① 伪造 physio 流必红，且红在 M9（M8 结构上够不着）--------------

    def test_forged_nonzero_mu_v_raises_m9_where_m8_structurally_cannot(self) -> None:
        """**判别力①**：μv=0.9、Πa=0.05 的伪造 physio 流必红，且**红在 M9 不是 M8**。

        正控是本例的重点：先用纯函数把「M8 会放行这条载荷」测出来（最坏 salience≈0.034
        ≪ 0.18），再断言它仍被拦下 —— 从而证明拦下它的**只能**是 M9。若哪天有人把 M9 删掉
        而寄望「M8 顶一下」，本例会红在 `pytest.raises` 上。
        """
        mu = (0.9, 0.4)
        pi_a = 0.05
        # 正控：M8 判据对这条载荷**放行**（不是「恰好也拦了」）
        worst, _ceiling = _physio_self_ignite_salience(mu, MIN_PRECISION, pi_a)
        assert worst < ZERO_SALIENCE_THRESHOLD, (
            f"正控失败：M8 最坏 salience={worst} 已 ≥ 阈值 {ZERO_SALIENCE_THRESHOLD}，"
            "本例测不出「M8 顶不了 M9」"
        )

        forged = _make_prior("physio/forged", mu=mu, precision=(MIN_PRECISION, pi_a))
        with pytest.raises(ValueError, match="M9 physio 效价契约违反") as exc:
            build_external_priors_override([forged])

        message = str(exc.value)
        # 实测 μv 必须出现在消息里（定位用）
        assert "0.9" in message, f"错误消息未给出实测 μv：{message}"
        # 归因：契约 / Kreibig 是建模依据 / 落成 0.0 的是我方硬写而非 Zero M2
        assert "对效价盲" in message and "Kreibig" in message, f"错误消息缺契约归因：{message}"
        assert "M2 只覆写 Πv" in message, (
            f"错误消息未澄清「不是 Zero M2 的保证」这条已订正的假依据：{message}"
        )
        # Zero 已落其半，但我方不依赖对方状态
        assert "8043176" in message and "不依赖对方状态" in message, (
            f"错误消息未写明 Zero 侧落地点与「两侧各封一半」：{message}"
        )
        # 危害：不换取后验影响力却单买点燃资格
        assert "单买点燃资格" in message and "hypot" in message, (
            f"错误消息未说明非零 μv 的危害机制：{message}"
        )

    # -- 判别力② 正常流绿 ---------------------------------------------------

    def test_zero_mu_v_streams_pass(self) -> None:
        """**判别力②**：μv=0.0 的正常 physio 流全绿——单流与默认合并路径双查（各带正控）。

        与 ① 的差只有 μv 那一维（同为 physio 前缀、同为合法 Πa），证明 M9 拦的是**非零 μv**
        而非「凡 physio 流就拦」。
        """
        single = _make_prior("physio/forged", mu=(0.0, 0.4), precision=(MIN_PRECISION, 0.05))
        payload = build_external_priors_override([single])
        streams = _physio_streams(payload)
        assert len(streams) == 1, f"正控失败：载荷里没有 physio 流 {payload}"
        assert streams[0][1] == pytest.approx((0.0, 0.4))

        # 默认合并路径（EDA+HRV → physio）同样绿
        eda = _make_prior("eda/sc", mu=(0.0, 0.8), precision=(MIN_PRECISION, 0.15))
        hrv = _make_prior("hrv/rmssd", mu=(0.0, 0.9), precision=(MIN_PRECISION, 0.20))
        merged_payload = build_external_priors_override([eda, hrv])
        merged_streams = _physio_streams(merged_payload)
        assert len(merged_streams) == 1 and merged_streams[0][0] == PHYSIO_MERGED_MODALITY, (
            f"正控失败：合并路径未产出 physio 流 {merged_payload}"
        )
        assert merged_streams[0][1][0] == 0.0
        assert merged_streams[0][1][1] != 0.0, "正控：μa 须非平凡，否则观测的不是 μv 那一维"

    # -- 判别力④ 非 physio 流带非零 μv 是合法的，不得误报 --------------------

    def test_non_physio_streams_may_carry_nonzero_mu_v(self) -> None:
        """**判别力④**：vision/audio 带非零 μv 是**本职**，M9 不得误报。

        效价正是这两条流的主信息（face valence 强），拦它们等于把守卫写错了对象。
        """
        vision = _make_prior("vision", mu=(0.9, 0.3), precision=(0.20, 0.12))
        audio = _make_prior("audio/prosody", mu=(-0.8, 0.5), precision=(0.10, 0.25))
        payload = build_external_priors_override([vision, audio])

        # 正控①：这两条确实**不被判为** physio（否则「不红」是因为没被观测，恒真式）
        assert _physio_streams(payload) == [], "vision/audio 不应被判为 physio 流"
        # 正控②：它们的 μv 确实非零（否则「不红」是平凡的）
        assert [s[1][0] for s in payload["external_priors"]] == pytest.approx([0.9, -0.8])

    # -- 判别力⑤ 合并路径产出非零 μv 也要红 ---------------------------------

    def test_merged_output_with_nonzero_mu_v_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """**判别力⑤**：合并式若改口径把子源 μv 带出，出网口仍必须红。

        这一格正是回件 §6-8 说「合并产出的新 μ……只有我方出境侧看得见」的可执行版：
        入参两条先验的 μv **都是 0.0（逐条完全合规）**，红只可能来自合并产物。
        通过 monkeypatch 换掉 `merge_physio_priors` 模拟「v2 改口径 / 新增 RSP 子源」这类
        未来改动——三处 μv 硬写锚点中的第三处（合并出线）被改时的真实形态。
        """
        original_merge = external_priors_module.merge_physio_priors

        def leaking_merge(
            priors: list[ModalityPrior], *, omega: float | None = None
        ) -> list[ModalityPrior]:
            """模拟「合并式把子源 μv 按精度加权带出」——即第三处硬写锚点被删。"""
            merged = original_merge(priors, omega=omega)
            return [
                p.model_copy(update={"mu": (0.42, p.mu[1])})
                if p.modality == PHYSIO_MERGED_MODALITY
                else p
                for p in merged
            ]

        monkeypatch.setattr(external_priors_module, "merge_physio_priors", leaking_merge)

        eda = _make_prior("eda/sc", mu=(0.0, 0.8), precision=(MIN_PRECISION, 0.15))
        hrv = _make_prior("hrv/rmssd", mu=(0.0, 0.9), precision=(MIN_PRECISION, 0.20))
        # 正控：入参逐条合规——红必须来自合并产物
        for prior in (eda, hrv):
            assert prior.mu[0] == 0.0

        with pytest.raises(ValueError, match="M9 physio 效价契约违反") as exc:
            build_external_priors_override([eda, hrv])
        # 红在合并后的那条流上（名字是 "physio"，不是 "eda/sc"/"hrv/rmssd"）
        assert PHYSIO_MERGED_MODALITY in str(exc.value)
        assert "0.42" in str(exc.value), f"错误消息未给出合并产物的实测 μv：{exc.value}"

    # -- 执行序 -------------------------------------------------------------

    def test_m9_precedes_m8_when_both_are_violated(self) -> None:
        """两条同时违反时**先报 M9**（契约违反是根因，Πa 越顶是并发症）。

        正控：先证明这条载荷**确实也**越了 M8 的硬顶（否则「报 M9」是平凡的、与执行序无关）。
        """
        mu = (0.9, 0.4)
        pi_a = 0.5  # 远超 μv=0 下的 0.359，更超 μv=0.9 下收紧后的硬顶
        worst, ceiling = _physio_self_ignite_salience(mu, MIN_PRECISION, pi_a)
        assert worst >= ZERO_SALIENCE_THRESHOLD and pi_a > ceiling, (
            f"正控失败：该载荷未同时违反 M8（最坏 salience={worst}、硬顶={ceiling}），"
            "本例测不到执行序"
        )

        with pytest.raises(ValueError, match="M9 physio 效价契约违反") as exc:
            build_external_priors_override([_make_prior("physio", mu=mu, precision=(0.5, pi_a))])
        assert "M8" not in str(exc.value), f"执行序错位：应由 M9 报，实得 {exc.value}"

    def test_m7_precedes_m9_for_out_of_domain_mu_v(self) -> None:
        """**M7 仍先于 M9**：越域/NaN 的 μv 报「M7 μ 越界」，不报契约违反。

        理由：域错误的诊断更具体；且 `nan != 0.0` 恒 True，若无 M7 在前，NaN μv 会被
        误报成「契约违反」而把排查引向跨仓协议——那是错的方向。
        """
        out_of_domain = ModalityPrior.model_construct(
            modality="physio", mu=(7.7, 0.4), precision=(MIN_PRECISION, 0.05)
        )
        with pytest.raises(ValueError, match="M7 μ 越界"):
            build_external_priors_override([out_of_domain])

        nan_mu_v = ModalityPrior.model_construct(
            modality="physio", mu=(float("nan"), 0.4), precision=(MIN_PRECISION, 0.05)
        )
        with pytest.raises(ValueError, match="M7 μ 越界"):
            build_external_priors_override([nan_mu_v])

    # -- 施加集合 == Zero 的 M2 集合 ----------------------------------------

    @pytest.mark.parametrize("name", ["physio", "EDA/SC", "edax", "HRV_RMSSD", "pupil/diam", "scr"])
    def test_covers_exactly_zeros_m2_prefix_set(self, name: str) -> None:
        """M9 的施加集合须与 Zero M2 的判定集合**同集**（大小写不敏感 + 裸前缀）。

        用 `_triggers_zero_m2` 而非 advisory 的 `is_physio_stream`：后者区分大小写、要求
        分隔符，"EDA/SC"/"edax" 在它那里是 False，而 Zero **会**对这两条覆写 Πv——集合错位
        会让「Zero 认定的 physio」逃出我方契约守卫。
        """
        # 正控：这些流名在 Zero 侧确实触发 M2（本例的前提，不是被测项）
        assert _triggers_zero_m2(name), f"前提破：{name!r} 在 Zero 侧不触发 M2，本例无意义"
        prior = _make_prior(name, mu=(0.6, 0.4), precision=(MIN_PRECISION, 0.05))
        with pytest.raises(ValueError, match="M9 physio 效价契约违反"):
            build_external_priors_override([prior])
