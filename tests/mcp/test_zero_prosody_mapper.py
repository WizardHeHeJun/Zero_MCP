"""ProsodyParams 与 LinearProsodyMapper 单测（蓝图 T1 韵律映射器）。

覆盖：
  1. ratio 分支（prosody_scale="ratio"）：speech_rate 直传、pitch=log2 映射、energy=lerp。
  2. None 分支（prosody_scale 缺省，Zero 占位无标注）：与 ratio 同行为。
  3. normalized 分支（prosody_scale="normalized"）：三值均 [0,1]、线性映射到各自目标范围。
  4. 边界：pitch=0.0 + ratio 口径 → 不抛异常（log2 兜底 1e-6），pitch_semitones 为兜底计算值。
  5. SSML 输出：to_ssml_prosody_attrs() 格式正确（rate=%、pitch=±N.NNst、volume=±N.NNdB）。
  6. 协议符合性：isinstance(LinearProsodyMapper(), ProsodyMapper) 为 True。
  7. 自定义 range：构造参数 rate_range / pitch_semitone_range / gain_db_range 生效。
  8. ProsodyParams 不可变 / extra forbid：多余字段被拒、frozen 赋值抛 ValidationError。
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError

from src.agents.models.zero_affect import ExpressionHead
from src.mcp.zero.expression_sink import ProsodyMapper
from src.mcp.zero.mappers.prosody import LinearProsodyMapper, ProsodyParams

# ---------------------------------------------------------------------------
# 辅助构造函数
# ---------------------------------------------------------------------------


def _make_physiology(
    heart_rate_bpm: float = 80.0,
    skin_conductance: float = 0.5,
    pupil_mm: float = 4.0,
) -> dict[str, Any]:
    """构造合法 PhysiologyChannel dict。"""
    return {
        "heart_rate_bpm": heart_rate_bpm,
        "skin_conductance": skin_conductance,
        "pupil_mm": pupil_mm,
    }


def _make_prosody_dict(
    speech_rate: float = 1.0,
    pitch: float = 1.0,
    energy: float = 0.7,
) -> dict[str, Any]:
    """构造合法 ProsodyChannel dict。"""
    return {"speech_rate": speech_rate, "pitch": pitch, "energy": energy}


def _make_expression_head(
    speech_rate: float = 1.0,
    pitch: float = 1.0,
    energy: float = 0.7,
    prosody_scale: str | None = None,
    facs_au: dict[str, float] | None = None,
    text_label: str = "content",
) -> ExpressionHead:
    """构造合法 ExpressionHead（默认 legacy 3 键，prosody_scale 可选）。

    ratio/None 口径时 speech_rate/pitch 可超 [0,1]；
    normalized 口径时需调用方确保三值在 [0,1] 内。
    """
    data: dict[str, Any] = {
        "facs_au": facs_au or {"AU12": 0.8, "AU06": 0.6, "intensity": 0.7},
        "text_label": text_label,
        "physiology": _make_physiology(),
        "prosody": _make_prosody_dict(speech_rate=speech_rate, pitch=pitch, energy=energy),
    }
    if prosody_scale is not None:
        data["prosody_scale"] = prosody_scale
    return ExpressionHead(**data)


# ---------------------------------------------------------------------------
# 1. ratio 分支（prosody_scale="ratio"）
# ---------------------------------------------------------------------------


class TestRatioBranch:
    """prosody_scale="ratio"：speech_rate 直传、pitch=12*log2(pitch)、energy=lerp。"""

    async def test_rate_ratio_equals_speech_rate(self) -> None:
        """ratio 口径：rate_ratio 直接等于 speech_rate，不做映射。"""
        head = _make_expression_head(speech_rate=1.2, pitch=1.0, energy=0.5, prosody_scale="ratio")
        mapper = LinearProsodyMapper()
        result = await mapper.map(head)
        assert result.rate_ratio == pytest.approx(1.2)

    async def test_pitch_one_gives_zero_semitones(self) -> None:
        """ratio 口径：pitch=1.0 → pitch_semitones=0.0（12*log2(1)=0）。"""
        head = _make_expression_head(pitch=1.0, prosody_scale="ratio")
        result = await LinearProsodyMapper().map(head)
        assert result.pitch_semitones == pytest.approx(0.0)

    async def test_pitch_1_3_gives_approx_4_5_semitones(self) -> None:
        """ratio 口径：pitch=1.3 → pitch_semitones≈12*log2(1.3)≈4.504。"""
        head = _make_expression_head(pitch=1.3, prosody_scale="ratio")
        result = await LinearProsodyMapper().map(head)
        expected = 12.0 * math.log2(1.3)
        assert result.pitch_semitones == pytest.approx(expected, rel=1e-5)

    async def test_energy_07_lerp_in_gain_range(self) -> None:
        """ratio 口径：energy=0.7 → gain_db = lerp((-6,6), 0.7) = -6 + 12*0.7 = 2.4 dB。"""
        head = _make_expression_head(energy=0.7, prosody_scale="ratio")
        result = await LinearProsodyMapper().map(head)
        expected = -6.0 + 12.0 * 0.7  # 2.4
        assert result.gain_db == pytest.approx(expected)

    async def test_energy_0_gives_min_gain(self) -> None:
        """ratio 口径：energy=0.0 → gain_db == gain_db_range[0] == -6.0 dB。"""
        head = _make_expression_head(energy=0.0, prosody_scale="ratio")
        result = await LinearProsodyMapper().map(head)
        assert result.gain_db == pytest.approx(-6.0)

    async def test_energy_1_gives_max_gain(self) -> None:
        """ratio 口径：energy=1.0 → gain_db == gain_db_range[1] == 6.0 dB。"""
        head = _make_expression_head(energy=1.0, prosody_scale="ratio")
        result = await LinearProsodyMapper().map(head)
        assert result.gain_db == pytest.approx(6.0)

    async def test_pitch_2_gives_plus_12_semitones(self) -> None:
        """ratio 口径：pitch=2.0 → pitch_semitones == +12.0 st（倍频=12半音）。"""
        head = _make_expression_head(pitch=2.0, prosody_scale="ratio")
        result = await LinearProsodyMapper().map(head)
        assert result.pitch_semitones == pytest.approx(12.0)

    async def test_pitch_half_gives_minus_12_semitones(self) -> None:
        """ratio 口径：pitch=0.5 → pitch_semitones == -12.0 st（半频=-12半音）。"""
        head = _make_expression_head(pitch=0.5, prosody_scale="ratio")
        result = await LinearProsodyMapper().map(head)
        assert result.pitch_semitones == pytest.approx(-12.0)


# ---------------------------------------------------------------------------
# 2. None 分支（prosody_scale 缺省）
# ---------------------------------------------------------------------------


class TestNoneBranch:
    """prosody_scale=None（缺省，decoder 未标注），应与 ratio 分支同行为。"""

    async def test_none_scale_rate_ratio_equals_speech_rate(self) -> None:
        """None 口径：rate_ratio 直接等于 speech_rate，行为同 ratio。"""
        head = _make_expression_head(speech_rate=0.9, pitch=1.0, energy=0.5)
        # prosody_scale 不传，默认 None
        assert head.prosody_scale is None
        result = await LinearProsodyMapper().map(head)
        assert result.rate_ratio == pytest.approx(0.9)

    async def test_none_scale_pitch_log2_mapping(self) -> None:
        """None 口径：pitch 走 12*log2 映射，同 ratio 分支。"""
        head = _make_expression_head(pitch=1.3)
        result = await LinearProsodyMapper().map(head)
        expected = 12.0 * math.log2(1.3)
        assert result.pitch_semitones == pytest.approx(expected, rel=1e-5)

    async def test_none_scale_energy_lerp(self) -> None:
        """None 口径：energy 走 lerp(gain_db_range, energy)，同 ratio 分支。"""
        head = _make_expression_head(energy=0.5)
        result = await LinearProsodyMapper().map(head)
        assert result.gain_db == pytest.approx(0.0)  # lerp((-6,6), 0.5) = 0.0


# ---------------------------------------------------------------------------
# 3. normalized 分支（prosody_scale="normalized"）
# ---------------------------------------------------------------------------


class TestNormalizedBranch:
    """prosody_scale="normalized"：三值均 [0,1]，线性映射到各自目标范围。"""

    async def test_speech_rate_05_gives_center_rate(self) -> None:
        """normalized：speech_rate=0.5 → rate_ratio=lerp((0.5,1.5),0.5)=1.0（中点）。"""
        head = _make_expression_head(
            speech_rate=0.5, pitch=0.5, energy=0.5, prosody_scale="normalized"
        )
        result = await LinearProsodyMapper().map(head)
        assert result.rate_ratio == pytest.approx(1.0)

    async def test_speech_rate_0_gives_min_rate(self) -> None:
        """normalized：speech_rate=0.0 → rate_ratio=0.5（范围下界）。"""
        head = _make_expression_head(
            speech_rate=0.0, pitch=0.5, energy=0.5, prosody_scale="normalized"
        )
        result = await LinearProsodyMapper().map(head)
        assert result.rate_ratio == pytest.approx(0.5)

    async def test_speech_rate_1_gives_max_rate(self) -> None:
        """normalized：speech_rate=1.0 → rate_ratio=1.5（范围上界）。"""
        head = _make_expression_head(
            speech_rate=1.0, pitch=0.5, energy=0.5, prosody_scale="normalized"
        )
        result = await LinearProsodyMapper().map(head)
        assert result.rate_ratio == pytest.approx(1.5)

    async def test_pitch_05_gives_zero_semitones(self) -> None:
        """normalized：pitch=0.5 → pitch_semitones=lerp((-4,+4),0.5)=0.0 st（中点）。"""
        head = _make_expression_head(
            speech_rate=0.5, pitch=0.5, energy=0.5, prosody_scale="normalized"
        )
        result = await LinearProsodyMapper().map(head)
        assert result.pitch_semitones == pytest.approx(0.0)

    async def test_pitch_1_gives_plus_4_semitones(self) -> None:
        """normalized：pitch=1.0 → pitch_semitones=+4.0 st（范围上界）。"""
        head = _make_expression_head(
            speech_rate=0.5, pitch=1.0, energy=0.5, prosody_scale="normalized"
        )
        result = await LinearProsodyMapper().map(head)
        assert result.pitch_semitones == pytest.approx(4.0)

    async def test_pitch_0_gives_minus_4_semitones(self) -> None:
        """normalized：pitch=0.0 → pitch_semitones=-4.0 st（范围下界）。"""
        head = _make_expression_head(
            speech_rate=0.5, pitch=0.0, energy=0.5, prosody_scale="normalized"
        )
        result = await LinearProsodyMapper().map(head)
        assert result.pitch_semitones == pytest.approx(-4.0)

    async def test_energy_1_gives_max_gain(self) -> None:
        """normalized：energy=1.0 → gain_db=6.0 dB（范围上界）。"""
        head = _make_expression_head(
            speech_rate=0.5, pitch=0.5, energy=1.0, prosody_scale="normalized"
        )
        result = await LinearProsodyMapper().map(head)
        assert result.gain_db == pytest.approx(6.0)

    async def test_energy_0_gives_min_gain(self) -> None:
        """normalized：energy=0.0 → gain_db=-6.0 dB（范围下界）。"""
        head = _make_expression_head(
            speech_rate=0.5, pitch=0.5, energy=0.0, prosody_scale="normalized"
        )
        result = await LinearProsodyMapper().map(head)
        assert result.gain_db == pytest.approx(-6.0)

    async def test_normalized_all_midpoints(self) -> None:
        """normalized：三值均 0.5 → rate=1.0、pitch=0.0、gain=0.0（三值均中点）。"""
        head = _make_expression_head(
            speech_rate=0.5, pitch=0.5, energy=0.5, prosody_scale="normalized"
        )
        result = await LinearProsodyMapper().map(head)
        assert result.rate_ratio == pytest.approx(1.0)
        assert result.pitch_semitones == pytest.approx(0.0)
        assert result.gain_db == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. 边界：pitch=0.0 + ratio 口径 → 不抛异常（log2 兜底）
# ---------------------------------------------------------------------------


class TestRatioPitchZeroBoundary:
    """pitch=0.0（或负值）在 ratio/None 口径下不抛异常，兜底取 1e-6 再计算。"""

    async def test_pitch_zero_ratio_does_not_raise(self) -> None:
        """ratio 口径：pitch=0.0 → 不抛异常，兜底 1e-6 后计算 pitch_semitones。"""
        # pitch=0.0 在 ratio 口径下需要数据模型放宽（ProsodyChannel 只做 >=0 sanity）
        # 注：ProsodyChannel 本身接受 0.0，只有 normalized 下才校验上界
        head = _make_expression_head(speech_rate=1.0, pitch=0.0, energy=0.5, prosody_scale="ratio")
        result = await LinearProsodyMapper().map(head)
        # 兜底取 1e-6，pitch_semitones = 12 * log2(1e-6) ≈ -239.15
        expected = 12.0 * math.log2(1e-6)
        assert result.pitch_semitones == pytest.approx(expected, rel=1e-4)

    async def test_pitch_zero_none_scale_does_not_raise(self) -> None:
        """None 口径：pitch=0.0 → 不抛异常，同 ratio 兜底行为。"""
        head = _make_expression_head(speech_rate=1.0, pitch=0.0, energy=0.5)
        result = await LinearProsodyMapper().map(head)
        expected = 12.0 * math.log2(1e-6)
        assert result.pitch_semitones == pytest.approx(expected, rel=1e-4)

    async def test_pitch_zero_logs_warning(self, caplog: Any) -> None:
        """pitch=0.0 + ratio 口径 → logger.warning 记录原始值与 scale。"""
        import logging

        head = _make_expression_head(pitch=0.0, prosody_scale="ratio")
        mapper = LinearProsodyMapper()
        with caplog.at_level(logging.WARNING, logger="src.mcp.zero.mappers.prosody"):
            await mapper.map(head)
        assert caplog.records, "pitch<=0 时应有 WARNING 日志"
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    async def test_pitch_zero_result_is_valid_prosody_params(self) -> None:
        """pitch=0.0 兜底后，返回值仍是有效 ProsodyParams 实例。"""
        head = _make_expression_head(pitch=0.0, prosody_scale="ratio")
        result = await LinearProsodyMapper().map(head)
        assert isinstance(result, ProsodyParams)


# ---------------------------------------------------------------------------
# 5. SSML 输出格式
# ---------------------------------------------------------------------------


class TestToSsmlProsodyAttrs:
    """to_ssml_prosody_attrs() 格式正确（rate=%、pitch=±N.Nst、volume=±N.NdB）。"""

    def test_baseline_format(self) -> None:
        """基线值（rate=1.0、pitch=0.0、gain=0.0）→ 正确格式字符串。"""
        params = ProsodyParams(rate_ratio=1.0, pitch_semitones=0.0, gain_db=0.0)
        result = params.to_ssml_prosody_attrs()
        assert result == 'rate="100%" pitch="+0.0st" volume="+0.0dB"'

    def test_rate_formatting(self) -> None:
        """rate_ratio=1.5 → rate="150%"（整数百分比，无小数点）。"""
        params = ProsodyParams(rate_ratio=1.5, pitch_semitones=0.0, gain_db=0.0)
        result = params.to_ssml_prosody_attrs()
        assert 'rate="150%"' in result

    def test_rate_low(self) -> None:
        """rate_ratio=0.5 → rate="50%"。"""
        params = ProsodyParams(rate_ratio=0.5, pitch_semitones=0.0, gain_db=0.0)
        result = params.to_ssml_prosody_attrs()
        assert 'rate="50%"' in result

    def test_positive_pitch_has_plus_sign(self) -> None:
        """正 pitch_semitones 有显式 '+' 号（:+.1f 格式）。"""
        params = ProsodyParams(rate_ratio=1.0, pitch_semitones=4.0, gain_db=0.0)
        result = params.to_ssml_prosody_attrs()
        assert 'pitch="+4.0st"' in result

    def test_negative_pitch_has_minus_sign(self) -> None:
        """负 pitch_semitones 有 '-' 号。"""
        params = ProsodyParams(rate_ratio=1.0, pitch_semitones=-4.0, gain_db=0.0)
        result = params.to_ssml_prosody_attrs()
        assert 'pitch="-4.0st"' in result

    def test_positive_gain_has_plus_sign(self) -> None:
        """正 gain_db 有显式 '+' 号。"""
        params = ProsodyParams(rate_ratio=1.0, pitch_semitones=0.0, gain_db=6.0)
        result = params.to_ssml_prosody_attrs()
        assert 'volume="+6.0dB"' in result

    def test_negative_gain_has_minus_sign(self) -> None:
        """负 gain_db 有 '-' 号。"""
        params = ProsodyParams(rate_ratio=1.0, pitch_semitones=0.0, gain_db=-6.0)
        result = params.to_ssml_prosody_attrs()
        assert 'volume="-6.0dB"' in result

    def test_three_components_all_present(self) -> None:
        """输出包含 rate=…%、pitch=…st、volume=…dB 三段。"""
        params = ProsodyParams(rate_ratio=1.2, pitch_semitones=2.5, gain_db=-3.0)
        result = params.to_ssml_prosody_attrs()
        assert "rate=" in result
        assert "pitch=" in result
        assert "volume=" in result
        assert "%" in result
        assert "st" in result
        assert "dB" in result

    async def test_ssml_from_mapper_output(self) -> None:
        """mapper.map() 输出的 ProsodyParams.to_ssml_prosody_attrs() 格式正确。"""
        head = _make_expression_head(speech_rate=1.0, pitch=1.0, energy=0.5, prosody_scale="ratio")
        result = await LinearProsodyMapper().map(head)
        ssml = result.to_ssml_prosody_attrs()
        # rate=1.0 → 100%，pitch=1.0 → 0.0st，energy=0.5 → 0.0dB
        assert 'rate="100%"' in ssml
        assert 'pitch="+0.0st"' in ssml
        assert 'volume="+0.0dB"' in ssml


# ---------------------------------------------------------------------------
# 6. 协议符合性
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """LinearProsodyMapper 满足 ProsodyMapper Protocol（runtime_checkable）。"""

    def test_linear_prosody_mapper_satisfies_prosody_mapper_protocol(self) -> None:
        """isinstance(LinearProsodyMapper(), ProsodyMapper) 为 True。"""
        mapper = LinearProsodyMapper()
        assert isinstance(mapper, ProsodyMapper)

    def test_object_without_map_fails_protocol(self) -> None:
        """无 map 方法的对象不满足 ProsodyMapper Protocol。"""

        class NoMap:
            pass

        assert not isinstance(NoMap(), ProsodyMapper)

    def test_map_is_coroutine_function(self) -> None:
        """LinearProsodyMapper.map 是协程函数（async def）。"""
        import asyncio

        mapper = LinearProsodyMapper()
        assert asyncio.iscoroutinefunction(mapper.map)


# ---------------------------------------------------------------------------
# 7. 自定义 range 参数
# ---------------------------------------------------------------------------


class TestCustomRangeParams:
    """构造参数 rate_range / pitch_semitone_range / gain_db_range 生效。"""

    def test_custom_rate_range_stored(self) -> None:
        """rate_range=(0.8,1.2) 被正确存储在成员变量 self.rate_range。"""
        mapper = LinearProsodyMapper(rate_range=(0.8, 1.2))
        assert mapper.rate_range == (0.8, 1.2)

    def test_custom_pitch_semitone_range_stored(self) -> None:
        """pitch_semitone_range=6.0 被正确存储。"""
        mapper = LinearProsodyMapper(pitch_semitone_range=6.0)
        assert mapper.pitch_semitone_range == pytest.approx(6.0)

    def test_custom_gain_db_range_stored(self) -> None:
        """gain_db_range=(-3.0, 3.0) 被正确存储。"""
        mapper = LinearProsodyMapper(gain_db_range=(-3.0, 3.0))
        assert mapper.gain_db_range == (-3.0, 3.0)

    async def test_custom_rate_range_affects_normalized_output(self) -> None:
        """rate_range=(0.8,1.2) 生效：normalized speech_rate=0.5 → rate_ratio=1.0（中点）。"""
        mapper = LinearProsodyMapper(rate_range=(0.8, 1.2))
        head = _make_expression_head(
            speech_rate=0.5, pitch=0.5, energy=0.5, prosody_scale="normalized"
        )
        result = await mapper.map(head)
        # lerp((0.8,1.2), 0.5) = 0.8 + 0.4*0.5 = 1.0
        assert result.rate_ratio == pytest.approx(1.0)

    async def test_custom_rate_range_min_output(self) -> None:
        """rate_range=(0.8,1.2)：normalized speech_rate=0.0 → rate_ratio=0.8。"""
        mapper = LinearProsodyMapper(rate_range=(0.8, 1.2))
        head = _make_expression_head(
            speech_rate=0.0, pitch=0.5, energy=0.5, prosody_scale="normalized"
        )
        result = await mapper.map(head)
        assert result.rate_ratio == pytest.approx(0.8)

    async def test_custom_pitch_range_affects_normalized_output(self) -> None:
        """pitch_semitone_range=6.0：normalized pitch=1.0 → pitch_semitones=+6.0 st。"""
        mapper = LinearProsodyMapper(pitch_semitone_range=6.0)
        head = _make_expression_head(
            speech_rate=0.5, pitch=1.0, energy=0.5, prosody_scale="normalized"
        )
        result = await mapper.map(head)
        assert result.pitch_semitones == pytest.approx(6.0)

    async def test_custom_gain_range_affects_output(self) -> None:
        """gain_db_range=(-3.0, 3.0)：normalized energy=1.0 → gain_db=3.0 dB。"""
        mapper = LinearProsodyMapper(gain_db_range=(-3.0, 3.0))
        head = _make_expression_head(
            speech_rate=0.5, pitch=0.5, energy=1.0, prosody_scale="normalized"
        )
        result = await mapper.map(head)
        assert result.gain_db == pytest.approx(3.0)

    def test_default_rate_range(self) -> None:
        """默认 rate_range=(0.5, 1.5)。"""
        mapper = LinearProsodyMapper()
        assert mapper.rate_range == (0.5, 1.5)

    def test_default_pitch_semitone_range(self) -> None:
        """默认 pitch_semitone_range=4.0。"""
        mapper = LinearProsodyMapper()
        assert mapper.pitch_semitone_range == pytest.approx(4.0)

    def test_default_gain_db_range(self) -> None:
        """默认 gain_db_range=(-6.0, 6.0)。"""
        mapper = LinearProsodyMapper()
        assert mapper.gain_db_range == (-6.0, 6.0)


# ---------------------------------------------------------------------------
# 8. ProsodyParams 不可变 / extra forbid
# ---------------------------------------------------------------------------


class TestProsodyParamsImmutabilityAndValidation:
    """ProsodyParams：extra="forbid" 拒绝多余字段；frozen=True 拒绝赋值。"""

    def test_extra_field_rejected(self) -> None:
        """ProsodyParams 传入多余字段 → ValidationError。"""
        with pytest.raises(ValidationError):
            ProsodyParams(
                rate_ratio=1.0,
                pitch_semitones=0.0,
                gain_db=0.0,
                unknown_field=99.0,  # type: ignore[call-arg]
            )

    def test_frozen_assignment_raises(self) -> None:
        """frozen=True：尝试赋值已有字段 → ValidationError（pydantic frozen model）。"""
        params = ProsodyParams(rate_ratio=1.0, pitch_semitones=0.0, gain_db=0.0)
        with pytest.raises(ValidationError):
            params.rate_ratio = 2.0  # type: ignore[misc]

    def test_valid_construction_passes(self) -> None:
        """合法三字段构造通过，值域自由（映射层自行管理）。"""
        params = ProsodyParams(rate_ratio=0.5, pitch_semitones=-12.0, gain_db=6.0)
        assert params.rate_ratio == pytest.approx(0.5)
        assert params.pitch_semitones == pytest.approx(-12.0)
        assert params.gain_db == pytest.approx(6.0)

    def test_is_pydantic_base_model(self) -> None:
        """ProsodyParams 是 pydantic BaseModel 实例。"""
        from pydantic import BaseModel

        params = ProsodyParams(rate_ratio=1.0, pitch_semitones=0.0, gain_db=0.0)
        assert isinstance(params, BaseModel)

    def test_model_config_frozen(self) -> None:
        """ProsodyParams.model_config frozen=True。"""
        assert ProsodyParams.model_config.get("frozen") is True

    def test_model_config_extra_forbid(self) -> None:
        """ProsodyParams.model_config extra="forbid"。"""
        assert ProsodyParams.model_config.get("extra") == "forbid"
