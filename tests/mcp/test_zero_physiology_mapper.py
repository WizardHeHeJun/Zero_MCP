"""LinearPhysiologyMapper 单测（蓝图 P1 生理映射器，canonical=WESAD）。

canonical=WESAD 真 physiology_decoder 口径（2026-07-23 拍板）：
skin_conductance 是 μS 物理单位（归一上界 skin_conductance_max_us，默认 20），temperature_c 皮肤温度
（归一范围 temperature_range，默认 (30,40)），pupil_mm 过渡期可选（缺省→None）。

覆盖：
  1. 协议符合性：isinstance PhysiologyMapper；map 是协程函数；默认/自定义配置。
  2. 呼吸速率：breath_rate_bpm = HR / cardio_respiratory_ratio；自定义比；HR≤0 → clamp 0。
  3. 皮肤电导（μS）：skin_conductance_level = clamp01(sc / max_us)；clamp；自定义上界。
  4. 皮肤温度：skin_temperature_level（temperature_range 归一）；clamp；temp/退化范围 → None。
  5. 瞳孔（过渡期可选）：pupil_dilation 归一；pupil_mm=None → None；退化范围 None。
  6. 心率透传：heart_rate_bpm 原样透传；负值 clamp 0。
  7. 配置校验：cardio_respiratory_ratio / skin_conductance_max_us ≤0 → ValueError。
  8. PhysiologyParams 模型：frozen、extra=forbid、字段范围、temp/pupil 可空。
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from src.agents.models.zero_affect import ExpressionHead
from src.mcp.zero.expression_sink import PhysiologyMapper
from src.mcp.zero.mappers.physiology import LinearPhysiologyMapper, PhysiologyParams

# ---------------------------------------------------------------------------
# 辅助构造函数
# ---------------------------------------------------------------------------


def _make_expression_head(
    physiology: dict[str, float] | None = None,
    text_label: str = "content",
) -> ExpressionHead:
    """构造合法 ExpressionHead 实例，可传自定义 physiology。

    默认 physiology 用 canonical WESAD 形状 {heart_rate_bpm, skin_conductance(μS), temperature_c}。
    """
    return ExpressionHead(
        facs_au={"AU12": 0.8, "AU06": 0.6, "intensity": 0.7},
        text_label=text_label,
        physiology=physiology
        if physiology is not None
        else {"heart_rate_bpm": 80.0, "skin_conductance": 10.0, "temperature_c": 35.0},
        prosody={"speech_rate": 1.0, "pitch": 1.0, "energy": 0.7},
    )


# ---------------------------------------------------------------------------
# 1. 协议符合性
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """LinearPhysiologyMapper 满足 PhysiologyMapper Protocol（runtime_checkable）。"""

    def test_satisfies_physiology_mapper_protocol(self) -> None:
        """isinstance(LinearPhysiologyMapper(), PhysiologyMapper) 为 True。"""
        assert isinstance(LinearPhysiologyMapper(), PhysiologyMapper)

    def test_object_without_map_fails_protocol(self) -> None:
        """无 map 方法的对象不满足 PhysiologyMapper Protocol。"""

        class NoMap:
            pass

        assert not isinstance(NoMap(), PhysiologyMapper)

    def test_map_is_coroutine_function(self) -> None:
        """LinearPhysiologyMapper.map 是协程函数（async def）。"""
        assert asyncio.iscoroutinefunction(LinearPhysiologyMapper().map)

    def test_default_config_stored(self) -> None:
        """默认 ratio=4.0、sc_max_us=20.0、temperature_range=(30,40)、pupil_mm_range=(3,5)。"""
        mapper = LinearPhysiologyMapper()
        assert mapper.cardio_respiratory_ratio == pytest.approx(4.0)
        assert mapper.skin_conductance_max_us == pytest.approx(20.0)
        assert mapper.temperature_range == (30.0, 40.0)
        assert mapper.pupil_mm_range == (3.0, 5.0)

    def test_custom_config_stored(self) -> None:
        """自定义配置被正确存储。"""
        mapper = LinearPhysiologyMapper(
            cardio_respiratory_ratio=5.0,
            skin_conductance_max_us=40.0,
            temperature_range=(28.0, 42.0),
            pupil_mm_range=(2.0, 8.0),
        )
        assert mapper.cardio_respiratory_ratio == pytest.approx(5.0)
        assert mapper.skin_conductance_max_us == pytest.approx(40.0)
        assert mapper.temperature_range == (28.0, 42.0)
        assert mapper.pupil_mm_range == (2.0, 8.0)

    async def test_map_returns_physiology_params(self) -> None:
        """map 返回 PhysiologyParams 实例。"""
        result = await LinearPhysiologyMapper().map(_make_expression_head())
        assert isinstance(result, PhysiologyParams)


# ---------------------------------------------------------------------------
# 2. 呼吸速率映射（心肺耦合比）
# ---------------------------------------------------------------------------


class TestBreathRate:
    """breath_rate_bpm = max(0, heart_rate_bpm / cardio_respiratory_ratio)。"""

    async def test_breath_rate_default_ratio(self) -> None:
        """HR=80，ratio=4.0（默认）→ breath_rate_bpm=20.0。"""
        head = _make_expression_head({"heart_rate_bpm": 80.0, "skin_conductance": 10.0})
        result = await LinearPhysiologyMapper().map(head)
        assert result.breath_rate_bpm == pytest.approx(20.0)

    async def test_breath_rate_custom_ratio(self) -> None:
        """HR=100，ratio=5.0 → breath_rate_bpm=20.0。"""
        head = _make_expression_head({"heart_rate_bpm": 100.0, "skin_conductance": 10.0})
        result = await LinearPhysiologyMapper(cardio_respiratory_ratio=5.0).map(head)
        assert result.breath_rate_bpm == pytest.approx(20.0)

    async def test_breath_rate_zero_hr(self) -> None:
        """HR=0 → breath_rate_bpm=0.0。"""
        head = _make_expression_head({"heart_rate_bpm": 0.0, "skin_conductance": 10.0})
        result = await LinearPhysiologyMapper().map(head)
        assert result.breath_rate_bpm == pytest.approx(0.0)

    async def test_breath_rate_negative_hr_clamped_to_zero(self) -> None:
        """负 HR（异常）→ breath_rate_bpm 被钳到 0.0（不出负呼吸速率）。"""
        head = _make_expression_head({"heart_rate_bpm": -10.0, "skin_conductance": 10.0})
        result = await LinearPhysiologyMapper().map(head)
        assert result.breath_rate_bpm == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. 皮肤电导（μS 归一，canonical=WESAD）
# ---------------------------------------------------------------------------


class TestSkinConductance:
    """skin_conductance_level = clamp01(skin_conductance / skin_conductance_max_us)。"""

    async def test_skin_conductance_us_normalized(self) -> None:
        """sc=10μS，上界 20 → level=0.5（μS 归一，非原样透传）。"""
        head = _make_expression_head({"heart_rate_bpm": 80.0, "skin_conductance": 10.0})
        result = await LinearPhysiologyMapper().map(head)
        assert result.skin_conductance_level == pytest.approx(0.5)

    async def test_skin_conductance_at_ceiling(self) -> None:
        """sc=20μS（上界）→ level=1.0。"""
        head = _make_expression_head({"heart_rate_bpm": 80.0, "skin_conductance": 20.0})
        result = await LinearPhysiologyMapper().map(head)
        assert result.skin_conductance_level == pytest.approx(1.0)

    async def test_skin_conductance_above_ceiling_clamps(self) -> None:
        """sc=30μS（越上界）→ level 被钳到 1.0。"""
        head = _make_expression_head({"heart_rate_bpm": 80.0, "skin_conductance": 30.0})
        result = await LinearPhysiologyMapper().map(head)
        assert result.skin_conductance_level == pytest.approx(1.0)

    async def test_skin_conductance_negative_clamps_to_zero(self) -> None:
        """sc=-2μS（越界）→ level 被钳到 0.0。"""
        head = _make_expression_head({"heart_rate_bpm": 80.0, "skin_conductance": -2.0})
        result = await LinearPhysiologyMapper().map(head)
        assert result.skin_conductance_level == pytest.approx(0.0)

    async def test_skin_conductance_custom_ceiling(self) -> None:
        """自定义上界 40，sc=10μS → level=0.25。"""
        head = _make_expression_head({"heart_rate_bpm": 80.0, "skin_conductance": 10.0})
        result = await LinearPhysiologyMapper(skin_conductance_max_us=40.0).map(head)
        assert result.skin_conductance_level == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 4. 皮肤温度归一（canonical=WESAD 新增）
# ---------------------------------------------------------------------------


class TestSkinTemperature:
    """skin_temperature_level = clamp01((temperature_c - lo) / (hi - lo))，temp=None → None。"""

    async def test_temperature_midpoint(self) -> None:
        """temp=35°C 在 (30,40) → level=0.5。"""
        head = _make_expression_head(
            {"heart_rate_bpm": 80.0, "skin_conductance": 10.0, "temperature_c": 35.0}
        )
        result = await LinearPhysiologyMapper().map(head)
        assert result.skin_temperature_level == pytest.approx(0.5)

    async def test_temperature_bounds(self) -> None:
        """temp=30（下界）→ 0.0；temp=40（上界）→ 1.0。"""
        lo = await LinearPhysiologyMapper().map(
            _make_expression_head(
                {"heart_rate_bpm": 80.0, "skin_conductance": 10.0, "temperature_c": 30.0}
            )
        )
        hi = await LinearPhysiologyMapper().map(
            _make_expression_head(
                {"heart_rate_bpm": 80.0, "skin_conductance": 10.0, "temperature_c": 40.0}
            )
        )
        assert lo.skin_temperature_level == pytest.approx(0.0)
        assert hi.skin_temperature_level == pytest.approx(1.0)

    async def test_temperature_out_of_range_clamps(self) -> None:
        """temp=45（越上界）→ 1.0；temp=25（越下界）→ 0.0。"""
        over = await LinearPhysiologyMapper().map(
            _make_expression_head(
                {"heart_rate_bpm": 80.0, "skin_conductance": 10.0, "temperature_c": 45.0}
            )
        )
        under = await LinearPhysiologyMapper().map(
            _make_expression_head(
                {"heart_rate_bpm": 80.0, "skin_conductance": 10.0, "temperature_c": 25.0}
            )
        )
        assert over.skin_temperature_level == pytest.approx(1.0)
        assert under.skin_temperature_level == pytest.approx(0.0)

    async def test_temperature_absent_returns_none(self) -> None:
        """无 temperature_c（过渡期占位形状）→ skin_temperature_level=None（不驱动该维）。"""
        head = _make_expression_head(
            {"heart_rate_bpm": 80.0, "skin_conductance": 10.0, "pupil_mm": 4.0}
        )
        result = await LinearPhysiologyMapper().map(head)
        assert result.skin_temperature_level is None

    async def test_temperature_custom_range(self) -> None:
        """自定义范围 (28,42)，temp=35 → (35-28)/(42-28)=7/14=0.5。"""
        head = _make_expression_head(
            {"heart_rate_bpm": 80.0, "skin_conductance": 10.0, "temperature_c": 35.0}
        )
        result = await LinearPhysiologyMapper(temperature_range=(28.0, 42.0)).map(head)
        assert result.skin_temperature_level == pytest.approx(0.5)

    async def test_degenerate_temperature_range_returns_none(self) -> None:
        """退化范围 (35,35)（span=0）→ level 回退 None，不除零。"""
        head = _make_expression_head(
            {"heart_rate_bpm": 80.0, "skin_conductance": 10.0, "temperature_c": 35.0}
        )
        result = await LinearPhysiologyMapper(temperature_range=(35.0, 35.0)).map(head)
        assert result.skin_temperature_level is None


# ---------------------------------------------------------------------------
# 5. 瞳孔扩张（过渡期兼容旧 avatar 契约，可选）
# ---------------------------------------------------------------------------


class TestPupilDilation:
    """pupil_dilation = clamp01((pupil_mm - lo) / (hi - lo))；pupil_mm=None → None。"""

    async def test_pupil_midpoint(self) -> None:
        """pupil_mm=4.0 在 (3,5) → dilation=0.5。"""
        head = _make_expression_head(
            {"heart_rate_bpm": 80.0, "skin_conductance": 10.0, "pupil_mm": 4.0}
        )
        result = await LinearPhysiologyMapper().map(head)
        assert result.pupil_dilation == pytest.approx(0.5)

    async def test_pupil_bounds_and_clamp(self) -> None:
        """pupil_mm=3(下界)→0.0；5(上界)→1.0；2(越下)→0.0；6(越上)→1.0。"""
        for pupil, expected in ((3.0, 0.0), (5.0, 1.0), (2.0, 0.0), (6.0, 1.0)):
            head = _make_expression_head(
                {"heart_rate_bpm": 80.0, "skin_conductance": 10.0, "pupil_mm": pupil}
            )
            result = await LinearPhysiologyMapper().map(head)
            assert result.pupil_dilation == pytest.approx(expected), f"pupil={pupil}"

    async def test_pupil_custom_range(self) -> None:
        """自定义范围 (2,8)，pupil_mm=5.0 → dilation=0.5。"""
        head = _make_expression_head(
            {"heart_rate_bpm": 80.0, "skin_conductance": 10.0, "pupil_mm": 5.0}
        )
        result = await LinearPhysiologyMapper(pupil_mm_range=(2.0, 8.0)).map(head)
        assert result.pupil_dilation == pytest.approx(0.5)

    async def test_pupil_absent_returns_none(self) -> None:
        """canonical WESAD 无 pupil_mm → pupil_dilation=None（不驱动该维）。"""
        head = _make_expression_head(
            {"heart_rate_bpm": 80.0, "skin_conductance": 10.0, "temperature_c": 35.0}
        )
        result = await LinearPhysiologyMapper().map(head)
        assert result.pupil_dilation is None

    async def test_degenerate_pupil_range_returns_none(self) -> None:
        """退化范围 (4,4)（span=0）→ dilation 回退 None，不除零。"""
        head = _make_expression_head(
            {"heart_rate_bpm": 80.0, "skin_conductance": 10.0, "pupil_mm": 4.0}
        )
        result = await LinearPhysiologyMapper(pupil_mm_range=(4.0, 4.0)).map(head)
        assert result.pupil_dilation is None


# ---------------------------------------------------------------------------
# 6. 心率透传
# ---------------------------------------------------------------------------


class TestHeartRatePassthrough:
    """heart_rate_bpm 原样透传（供引擎直接驱动脉搏）。"""

    async def test_heart_rate_passthrough(self) -> None:
        """heart_rate_bpm 非负值原样透传到 PhysiologyParams。"""
        head = _make_expression_head({"heart_rate_bpm": 95.0, "skin_conductance": 10.0})
        result = await LinearPhysiologyMapper().map(head)
        assert result.heart_rate_bpm == pytest.approx(95.0)

    async def test_negative_heart_rate_clamped_to_zero(self) -> None:
        """负 HR（异常）→ heart_rate_bpm 被钳到 0.0（非负透传，防负值传入下游）。"""
        head = _make_expression_head({"heart_rate_bpm": -10.0, "skin_conductance": 10.0})
        result = await LinearPhysiologyMapper().map(head)
        assert result.heart_rate_bpm == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 7. 配置校验
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """cardio_respiratory_ratio / skin_conductance_max_us ≤0 → ValueError（作除数须正）。"""

    def test_zero_ratio_raises(self) -> None:
        """cardio_respiratory_ratio=0 → ValueError。"""
        with pytest.raises(ValueError, match="cardio_respiratory_ratio"):
            LinearPhysiologyMapper(cardio_respiratory_ratio=0.0)

    def test_negative_ratio_raises(self) -> None:
        """cardio_respiratory_ratio<0 → ValueError。"""
        with pytest.raises(ValueError, match="cardio_respiratory_ratio"):
            LinearPhysiologyMapper(cardio_respiratory_ratio=-1.0)

    def test_zero_sc_max_us_raises(self) -> None:
        """skin_conductance_max_us=0 → ValueError（作除数归一皮电）。"""
        with pytest.raises(ValueError, match="skin_conductance_max_us"):
            LinearPhysiologyMapper(skin_conductance_max_us=0.0)

    def test_negative_sc_max_us_raises(self) -> None:
        """skin_conductance_max_us<0 → ValueError。"""
        with pytest.raises(ValueError, match="skin_conductance_max_us"):
            LinearPhysiologyMapper(skin_conductance_max_us=-1.0)


# ---------------------------------------------------------------------------
# 8. PhysiologyParams 模型约束
# ---------------------------------------------------------------------------


class TestPhysiologyParamsModel:
    """PhysiologyParams frozen / extra=forbid / 字段范围约束 / temp·pupil 可空。"""

    def test_valid_construction_full(self) -> None:
        """全字段（含 temp/pupil）可直接构造。"""
        params = PhysiologyParams(
            heart_rate_bpm=80.0,
            breath_rate_bpm=20.0,
            skin_conductance_level=0.5,
            skin_temperature_level=0.5,
            pupil_dilation=0.5,
        )
        assert params.breath_rate_bpm == pytest.approx(20.0)

    def test_valid_construction_canonical_optional_none(self) -> None:
        """canonical 最小集（temp/pupil 缺省 None）可构造。"""
        params = PhysiologyParams(
            heart_rate_bpm=80.0,
            breath_rate_bpm=20.0,
            skin_conductance_level=0.5,
        )
        assert params.skin_temperature_level is None
        assert params.pupil_dilation is None

    def test_frozen(self) -> None:
        """PhysiologyParams frozen——赋值抛 ValidationError。"""
        params = PhysiologyParams(
            heart_rate_bpm=80.0, breath_rate_bpm=20.0, skin_conductance_level=0.5
        )
        with pytest.raises(ValidationError):
            params.breath_rate_bpm = 30.0  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        """extra=forbid——非法额外字段抛 ValidationError。"""
        with pytest.raises(ValidationError):
            PhysiologyParams(
                heart_rate_bpm=80.0,
                breath_rate_bpm=20.0,
                skin_conductance_level=0.5,
                unknown="x",  # type: ignore[call-arg]
            )

    def test_negative_breath_rate_rejected(self) -> None:
        """breath_rate_bpm<0 → ValidationError（ge=0）。"""
        with pytest.raises(ValidationError):
            PhysiologyParams(heart_rate_bpm=80.0, breath_rate_bpm=-1.0, skin_conductance_level=0.5)

    def test_skin_temperature_out_of_range_rejected(self) -> None:
        """skin_temperature_level>1 → ValidationError（le=1）。"""
        with pytest.raises(ValidationError):
            PhysiologyParams(
                heart_rate_bpm=80.0,
                breath_rate_bpm=20.0,
                skin_conductance_level=0.5,
                skin_temperature_level=1.5,
            )

    def test_pupil_dilation_out_of_range_rejected(self) -> None:
        """pupil_dilation>1 → ValidationError（le=1）。"""
        with pytest.raises(ValidationError):
            PhysiologyParams(
                heart_rate_bpm=80.0,
                breath_rate_bpm=20.0,
                skin_conductance_level=0.5,
                pupil_dilation=1.5,
            )


# ---------------------------------------------------------------------------
# 9. 顶层包导出
# ---------------------------------------------------------------------------


class TestTopLevelExport:
    """LinearPhysiologyMapper / PhysiologyParams 通过 src.mcp.zero 顶层可访问。"""

    def test_mapper_exported(self) -> None:
        """从 src.mcp.zero 顶层导入 LinearPhysiologyMapper 成功。"""
        from src.mcp.zero import LinearPhysiologyMapper as M  # noqa: PLC0415

        assert M is LinearPhysiologyMapper

    def test_params_exported(self) -> None:
        """从 src.mcp.zero 顶层导入 PhysiologyParams 成功。"""
        from src.mcp.zero import PhysiologyParams as P  # noqa: PLC0415

        assert P is PhysiologyParams
