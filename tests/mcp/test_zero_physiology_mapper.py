"""LinearPhysiologyMapper 单测（蓝图 P1 生理映射器）。

覆盖：
  1. 协议符合性：isinstance(LinearPhysiologyMapper(), PhysiologyMapper)；map 是协程函数。
  2. 呼吸速率：breath_rate_bpm = HR / cardio_respiratory_ratio；自定义比；HR≤0 → clamp 0。
  3. 瞳孔扩张：pupil_mm 在范围内归一 [0,1]；边界/越界 clamp；自定义范围；退化范围回退 0。
  4. 皮肤电导：skin_conductance_level = clamp01(skin_conductance)；越界 clamp。
  5. 心率透传：heart_rate_bpm 原样透传。
  6. 配置校验：cardio_respiratory_ratio ≤0 → ValueError。
  7. PhysiologyParams 模型：frozen、extra=forbid、字段范围约束。
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
    """构造合法 ExpressionHead 实例，可传自定义 physiology。"""
    return ExpressionHead(
        facs_au={"AU12": 0.8, "AU06": 0.6, "intensity": 0.7},
        text_label=text_label,
        physiology=physiology
        if physiology is not None
        else {"heart_rate_bpm": 80.0, "skin_conductance": 0.5, "pupil_mm": 4.0},
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
        """默认 cardio_respiratory_ratio=4.0、pupil_mm_range=(3.0, 5.0)。"""
        mapper = LinearPhysiologyMapper()
        assert mapper.cardio_respiratory_ratio == pytest.approx(4.0)
        assert mapper.pupil_mm_range == (3.0, 5.0)

    def test_custom_config_stored(self) -> None:
        """自定义配置被正确存储。"""
        mapper = LinearPhysiologyMapper(cardio_respiratory_ratio=5.0, pupil_mm_range=(2.0, 8.0))
        assert mapper.cardio_respiratory_ratio == pytest.approx(5.0)
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
        head = _make_expression_head(
            {"heart_rate_bpm": 80.0, "skin_conductance": 0.5, "pupil_mm": 4.0}
        )
        result = await LinearPhysiologyMapper().map(head)
        assert result.breath_rate_bpm == pytest.approx(20.0)

    async def test_breath_rate_custom_ratio(self) -> None:
        """HR=100，ratio=5.0 → breath_rate_bpm=20.0。"""
        head = _make_expression_head(
            {"heart_rate_bpm": 100.0, "skin_conductance": 0.5, "pupil_mm": 4.0}
        )
        result = await LinearPhysiologyMapper(cardio_respiratory_ratio=5.0).map(head)
        assert result.breath_rate_bpm == pytest.approx(20.0)

    async def test_breath_rate_zero_hr(self) -> None:
        """HR=0 → breath_rate_bpm=0.0。"""
        head = _make_expression_head(
            {"heart_rate_bpm": 0.0, "skin_conductance": 0.5, "pupil_mm": 4.0}
        )
        result = await LinearPhysiologyMapper().map(head)
        assert result.breath_rate_bpm == pytest.approx(0.0)

    async def test_breath_rate_negative_hr_clamped_to_zero(self) -> None:
        """负 HR（异常）→ breath_rate_bpm 被钳到 0.0（不出负呼吸速率）。"""
        head = _make_expression_head(
            {"heart_rate_bpm": -10.0, "skin_conductance": 0.5, "pupil_mm": 4.0}
        )
        result = await LinearPhysiologyMapper().map(head)
        assert result.breath_rate_bpm == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. 瞳孔扩张归一
# ---------------------------------------------------------------------------


class TestPupilDilation:
    """pupil_dilation = clamp01((pupil_mm - lo) / (hi - lo))，(lo,hi)=pupil_mm_range。"""

    async def test_pupil_midpoint(self) -> None:
        """pupil_mm=4.0 在 (3,5) → dilation=0.5。"""
        head = _make_expression_head(
            {"heart_rate_bpm": 80.0, "skin_conductance": 0.5, "pupil_mm": 4.0}
        )
        result = await LinearPhysiologyMapper().map(head)
        assert result.pupil_dilation == pytest.approx(0.5)

    async def test_pupil_lower_bound(self) -> None:
        """pupil_mm=3.0（下界）→ dilation=0.0。"""
        head = _make_expression_head(
            {"heart_rate_bpm": 80.0, "skin_conductance": 0.5, "pupil_mm": 3.0}
        )
        result = await LinearPhysiologyMapper().map(head)
        assert result.pupil_dilation == pytest.approx(0.0)

    async def test_pupil_upper_bound(self) -> None:
        """pupil_mm=5.0（上界）→ dilation=1.0。"""
        head = _make_expression_head(
            {"heart_rate_bpm": 80.0, "skin_conductance": 0.5, "pupil_mm": 5.0}
        )
        result = await LinearPhysiologyMapper().map(head)
        assert result.pupil_dilation == pytest.approx(1.0)

    async def test_pupil_below_range_clamps_to_zero(self) -> None:
        """pupil_mm=2.0（低于下界）→ dilation 被钳到 0.0。"""
        head = _make_expression_head(
            {"heart_rate_bpm": 80.0, "skin_conductance": 0.5, "pupil_mm": 2.0}
        )
        result = await LinearPhysiologyMapper().map(head)
        assert result.pupil_dilation == pytest.approx(0.0)

    async def test_pupil_above_range_clamps_to_one(self) -> None:
        """pupil_mm=6.0（高于上界）→ dilation 被钳到 1.0。"""
        head = _make_expression_head(
            {"heart_rate_bpm": 80.0, "skin_conductance": 0.5, "pupil_mm": 6.0}
        )
        result = await LinearPhysiologyMapper().map(head)
        assert result.pupil_dilation == pytest.approx(1.0)

    async def test_pupil_custom_range(self) -> None:
        """自定义范围 (2,8)，pupil_mm=5.0 → dilation=0.5。"""
        head = _make_expression_head(
            {"heart_rate_bpm": 80.0, "skin_conductance": 0.5, "pupil_mm": 5.0}
        )
        result = await LinearPhysiologyMapper(pupil_mm_range=(2.0, 8.0)).map(head)
        assert result.pupil_dilation == pytest.approx(0.5)

    async def test_degenerate_pupil_range_returns_zero(self) -> None:
        """退化范围 (4,4)（span=0）→ dilation 回退 0.0，不除零。"""
        head = _make_expression_head(
            {"heart_rate_bpm": 80.0, "skin_conductance": 0.5, "pupil_mm": 4.0}
        )
        result = await LinearPhysiologyMapper(pupil_mm_range=(4.0, 4.0)).map(head)
        assert result.pupil_dilation == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. 皮肤电导
# ---------------------------------------------------------------------------


class TestSkinConductance:
    """skin_conductance_level = clamp01(skin_conductance)。"""

    async def test_skin_conductance_passthrough_in_range(self) -> None:
        """skin_conductance=0.5（[0,1] 内）→ level=0.5。"""
        head = _make_expression_head(
            {"heart_rate_bpm": 80.0, "skin_conductance": 0.5, "pupil_mm": 4.0}
        )
        result = await LinearPhysiologyMapper().map(head)
        assert result.skin_conductance_level == pytest.approx(0.5)

    async def test_skin_conductance_above_one_clamps(self) -> None:
        """skin_conductance=1.5（越界）→ level 被钳到 1.0。

        PhysiologyChannel 不硬卡 skin_conductance 上界，可直接构造越界值测 clamp。
        """
        head = _make_expression_head(
            {"heart_rate_bpm": 80.0, "skin_conductance": 1.5, "pupil_mm": 4.0}
        )
        result = await LinearPhysiologyMapper().map(head)
        assert result.skin_conductance_level == pytest.approx(1.0)

    async def test_skin_conductance_negative_clamps_to_zero(self) -> None:
        """skin_conductance=-0.2（越界）→ level 被钳到 0.0。"""
        head = _make_expression_head(
            {"heart_rate_bpm": 80.0, "skin_conductance": -0.2, "pupil_mm": 4.0}
        )
        result = await LinearPhysiologyMapper().map(head)
        assert result.skin_conductance_level == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. 心率透传
# ---------------------------------------------------------------------------


class TestHeartRatePassthrough:
    """heart_rate_bpm 原样透传（供引擎直接驱动脉搏）。"""

    async def test_heart_rate_passthrough(self) -> None:
        """heart_rate_bpm 非负值原样透传到 PhysiologyParams。"""
        head = _make_expression_head(
            {"heart_rate_bpm": 95.0, "skin_conductance": 0.5, "pupil_mm": 4.0}
        )
        result = await LinearPhysiologyMapper().map(head)
        assert result.heart_rate_bpm == pytest.approx(95.0)

    async def test_negative_heart_rate_clamped_to_zero(self) -> None:
        """负 HR（异常）→ heart_rate_bpm 被钳到 0.0（非负透传，防负值传入下游）。"""
        head = _make_expression_head(
            {"heart_rate_bpm": -10.0, "skin_conductance": 0.5, "pupil_mm": 4.0}
        )
        result = await LinearPhysiologyMapper().map(head)
        assert result.heart_rate_bpm == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. 配置校验
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """cardio_respiratory_ratio ≤0 → ValueError（作除数须正）。"""

    def test_zero_ratio_raises(self) -> None:
        """cardio_respiratory_ratio=0 → ValueError。"""
        with pytest.raises(ValueError, match="cardio_respiratory_ratio"):
            LinearPhysiologyMapper(cardio_respiratory_ratio=0.0)

    def test_negative_ratio_raises(self) -> None:
        """cardio_respiratory_ratio<0 → ValueError。"""
        with pytest.raises(ValueError, match="cardio_respiratory_ratio"):
            LinearPhysiologyMapper(cardio_respiratory_ratio=-1.0)


# ---------------------------------------------------------------------------
# 7. PhysiologyParams 模型约束
# ---------------------------------------------------------------------------


class TestPhysiologyParamsModel:
    """PhysiologyParams frozen / extra=forbid / 字段范围约束。"""

    def test_valid_construction(self) -> None:
        """合法字段可直接构造。"""
        params = PhysiologyParams(
            heart_rate_bpm=80.0,
            breath_rate_bpm=20.0,
            pupil_dilation=0.5,
            skin_conductance_level=0.5,
        )
        assert params.breath_rate_bpm == pytest.approx(20.0)

    def test_frozen(self) -> None:
        """PhysiologyParams frozen——赋值抛 ValidationError。"""
        params = PhysiologyParams(
            heart_rate_bpm=80.0,
            breath_rate_bpm=20.0,
            pupil_dilation=0.5,
            skin_conductance_level=0.5,
        )
        with pytest.raises(ValidationError):
            params.breath_rate_bpm = 30.0  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        """extra=forbid——非法额外字段抛 ValidationError。"""
        with pytest.raises(ValidationError):
            PhysiologyParams(
                heart_rate_bpm=80.0,
                breath_rate_bpm=20.0,
                pupil_dilation=0.5,
                skin_conductance_level=0.5,
                unknown="x",  # type: ignore[call-arg]
            )

    def test_negative_breath_rate_rejected(self) -> None:
        """breath_rate_bpm<0 → ValidationError（ge=0）。"""
        with pytest.raises(ValidationError):
            PhysiologyParams(
                heart_rate_bpm=80.0,
                breath_rate_bpm=-1.0,
                pupil_dilation=0.5,
                skin_conductance_level=0.5,
            )

    def test_pupil_dilation_out_of_range_rejected(self) -> None:
        """pupil_dilation>1 → ValidationError（le=1）。"""
        with pytest.raises(ValidationError):
            PhysiologyParams(
                heart_rate_bpm=80.0,
                breath_rate_bpm=20.0,
                pupil_dilation=1.5,
                skin_conductance_level=0.5,
            )


# ---------------------------------------------------------------------------
# 8. 顶层包导出
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
