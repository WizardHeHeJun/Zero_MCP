"""ArkitFacsMapper 单测（蓝图 F1 FACS 映射器）。

覆盖：
  1. 协议符合性：isinstance(ArkitFacsMapper(), FacsMapper) 为 True。
  2. 双侧 AU（AU12）→ 对称 L/R blendshape 同值。
  3. 单侧 AU（AU01→browInnerUp、AU17→mouthShrugLower、AU26→jawOpen）只出 1 个 blendshape。
  4. 全 13 AU（值 0.5，intensity=1.0）→ 所有 AU_TO_ARKIT 目标 blendshape 均在结果；
     "intensity" 不产出名为 intensity 的 blendshape。
  5. intensity 全局乘子：facs_au={"AU12":0.8,"intensity":0.5} → mouthSmile*=0.4（0.8*0.5）。
  6. apply_intensity=False → 忽略 intensity，输出原始值。
  7. intensity 缺省（facs_au 中无 intensity 键）→ gain=1.0，结果不变。
  8. 象限子集：只含 {"AU15":0.5,"AU04":0.3,"intensity":0.6} →
     只驱动 mouthFrown*/browDown*，不含 smile 等未出现 AU 的 blendshape。
  9. clamp：facs_au={"AU12":0.9,"intensity":2.0} → mouthSmile* 被钳到 1.0。
  10. 空 facs_au → 结果为空 dict。
  11. 仅含 intensity 键 → 结果为空 dict（无 AU 被驱动）。
  12. map() 是协程函数（async def）。
"""

from __future__ import annotations

import asyncio

import pytest

from src.agents.models.zero_affect import ExpressionHead
from src.mcp.zero.expression_sink import FacsMapper
from src.mcp.zero.mappers.facs import AU_TO_ARKIT, ArkitFacsMapper

# ---------------------------------------------------------------------------
# 辅助构造函数
# ---------------------------------------------------------------------------


def _make_expression_head(
    facs_au: dict[str, float] | None = None,
    text_label: str = "content",
) -> ExpressionHead:
    """构造合法 ExpressionHead 实例，可传自定义 facs_au。"""
    return ExpressionHead(
        facs_au=facs_au if facs_au is not None else {"AU12": 0.8, "AU06": 0.6, "intensity": 0.7},
        text_label=text_label,
        physiology={"heart_rate_bpm": 80.0, "skin_conductance": 0.5, "pupil_mm": 4.0},
        prosody={"speech_rate": 1.0, "pitch": 1.0, "energy": 0.7},
    )


def _all_au_facs(intensity: float = 1.0, value: float = 0.5) -> dict[str, float]:
    """构造包含全 12 个 AU + intensity 的 facs_au dict。"""
    result: dict[str, float] = {au: value for au in AU_TO_ARKIT}
    result["intensity"] = intensity
    return result


# ---------------------------------------------------------------------------
# 1. 协议符合性
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """ArkitFacsMapper 满足 FacsMapper Protocol（runtime_checkable）。"""

    def test_arkit_facs_mapper_satisfies_facs_mapper_protocol(self) -> None:
        """isinstance(ArkitFacsMapper(), FacsMapper) 为 True。"""
        mapper = ArkitFacsMapper()
        assert isinstance(mapper, FacsMapper)

    def test_object_without_map_fails_protocol(self) -> None:
        """无 map 方法的对象不满足 FacsMapper Protocol。"""

        class NoMap:
            pass

        assert not isinstance(NoMap(), FacsMapper)

    def test_map_is_coroutine_function(self) -> None:
        """ArkitFacsMapper.map 是协程函数（async def）。"""
        mapper = ArkitFacsMapper()
        assert asyncio.iscoroutinefunction(mapper.map)

    def test_apply_intensity_default_true(self) -> None:
        """默认 apply_intensity=True。"""
        mapper = ArkitFacsMapper()
        assert mapper.apply_intensity is True

    def test_apply_intensity_false_stored(self) -> None:
        """apply_intensity=False 被正确存储。"""
        mapper = ArkitFacsMapper(apply_intensity=False)
        assert mapper.apply_intensity is False


# ---------------------------------------------------------------------------
# 2. 双侧 AU 映射（L/R 对称）
# ---------------------------------------------------------------------------


class TestSymmetricAUMapping:
    """双侧 AU 映射到对称 L/R blendshape，且两值相等。"""

    async def test_au12_maps_to_mouth_smile_lr(self) -> None:
        """AU12（0.8，intensity=1.0）→ mouthSmileLeft=0.8、mouthSmileRight=0.8。"""
        head = _make_expression_head(facs_au={"AU12": 0.8, "intensity": 1.0})
        mapper = ArkitFacsMapper()
        result = await mapper.map(head)

        assert "mouthSmileLeft" in result
        assert "mouthSmileRight" in result
        assert result["mouthSmileLeft"] == pytest.approx(0.8)
        assert result["mouthSmileRight"] == pytest.approx(0.8)

    async def test_au12_symmetric_values_equal(self) -> None:
        """AU12 → mouthSmileLeft == mouthSmileRight（对称同值）。"""
        head = _make_expression_head(facs_au={"AU12": 0.6, "intensity": 1.0})
        result = await ArkitFacsMapper().map(head)

        assert result["mouthSmileLeft"] == pytest.approx(result["mouthSmileRight"])

    async def test_au04_maps_to_brow_down_lr(self) -> None:
        """AU04 → browDownLeft 与 browDownRight 均在结果。"""
        head = _make_expression_head(facs_au={"AU04": 0.5, "intensity": 1.0})
        result = await ArkitFacsMapper().map(head)

        assert "browDownLeft" in result
        assert "browDownRight" in result
        assert result["browDownLeft"] == pytest.approx(0.5)
        assert result["browDownRight"] == pytest.approx(0.5)

    async def test_au15_maps_to_mouth_frown_lr(self) -> None:
        """AU15 → mouthFrownLeft 与 mouthFrownRight 均在结果。"""
        head = _make_expression_head(facs_au={"AU15": 0.7, "intensity": 1.0})
        result = await ArkitFacsMapper().map(head)

        assert "mouthFrownLeft" in result
        assert "mouthFrownRight" in result
        assert result["mouthFrownLeft"] == pytest.approx(0.7)
        assert result["mouthFrownRight"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# 3. 单侧 AU（只输出 1 个 blendshape）
# ---------------------------------------------------------------------------


class TestSingleBlendshapeAU:
    """单一控制点 AU 只产出 1 个 blendshape 键。"""

    async def test_au01_maps_to_brow_inner_up_only(self) -> None:
        """AU01 → 只有 browInnerUp，无 L/R 后缀。"""
        head = _make_expression_head(facs_au={"AU01": 0.5, "intensity": 1.0})
        result = await ArkitFacsMapper().map(head)

        assert list(result.keys()) == ["browInnerUp"]
        assert result["browInnerUp"] == pytest.approx(0.5)

    async def test_au17_maps_to_mouth_shrug_lower_only(self) -> None:
        """AU17 → 只有 mouthShrugLower，无 L/R 后缀。"""
        head = _make_expression_head(facs_au={"AU17": 0.4, "intensity": 1.0})
        result = await ArkitFacsMapper().map(head)

        assert list(result.keys()) == ["mouthShrugLower"]
        assert result["mouthShrugLower"] == pytest.approx(0.4)

    async def test_au26_maps_to_jaw_open_only(self) -> None:
        """AU26 → 只有 jawOpen，无 L/R 后缀。"""
        head = _make_expression_head(facs_au={"AU26": 0.9, "intensity": 1.0})
        result = await ArkitFacsMapper().map(head)

        assert list(result.keys()) == ["jawOpen"]
        assert result["jawOpen"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# 4. 全 AU 集合：所有 AU_TO_ARKIT 目标 blendshape 均出现
# ---------------------------------------------------------------------------


class TestFullAUSet:
    """全 13 AU（值 0.5，intensity=1.0）→ AU_TO_ARKIT 所有目标 blendshape 均在结果。"""

    async def test_all_blendshapes_present(self) -> None:
        """全 12 AU + intensity=1.0 → 所有目标 blendshape 键均出现在结果中。"""
        head = _make_expression_head(facs_au=_all_au_facs(intensity=1.0, value=0.5))
        result = await ArkitFacsMapper().map(head)

        for au, blendshapes in AU_TO_ARKIT.items():
            for bs in blendshapes:
                assert bs in result, f"AU {au} 的目标 blendshape {bs!r} 不在结果中"

    async def test_all_blendshape_values_correct(self) -> None:
        """全 12 AU（值 0.5，intensity=1.0）→ 所有 blendshape 值均约等于 0.5。"""
        head = _make_expression_head(facs_au=_all_au_facs(intensity=1.0, value=0.5))
        result = await ArkitFacsMapper().map(head)

        for au, blendshapes in AU_TO_ARKIT.items():
            for bs in blendshapes:
                assert result[bs] == pytest.approx(0.5), (
                    f"AU {au} → {bs!r} 值期望 0.5，实际 {result[bs]}"
                )

    async def test_intensity_key_not_in_result(self) -> None:
        """全 AU 集合下，结果中不含名为 'intensity' 的 blendshape 键。"""
        head = _make_expression_head(facs_au=_all_au_facs(intensity=1.0, value=0.5))
        result = await ArkitFacsMapper().map(head)

        assert "intensity" not in result

    async def test_result_count_matches_all_blendshapes(self) -> None:
        """全 AU 集合 → 结果 blendshape 数量等于 AU_TO_ARKIT 中所有 blendshape 之和。"""
        expected_count = sum(len(bs) for bs in AU_TO_ARKIT.values())
        head = _make_expression_head(facs_au=_all_au_facs(intensity=1.0, value=0.5))
        result = await ArkitFacsMapper().map(head)

        assert len(result) == expected_count


# ---------------------------------------------------------------------------
# 5. intensity 全局乘子
# ---------------------------------------------------------------------------


class TestIntensityMultiplier:
    """intensity 字段作全局增益乘子（apply_intensity=True 时）。"""

    async def test_au12_with_intensity_half(self) -> None:
        """AU12=0.8，intensity=0.5 → mouthSmile*=0.4（0.8×0.5）。"""
        head = _make_expression_head(facs_au={"AU12": 0.8, "intensity": 0.5})
        result = await ArkitFacsMapper().map(head)

        assert result["mouthSmileLeft"] == pytest.approx(0.4)
        assert result["mouthSmileRight"] == pytest.approx(0.4)

    async def test_au12_with_intensity_one(self) -> None:
        """AU12=0.8，intensity=1.0 → mouthSmile*=0.8（增益中性）。"""
        head = _make_expression_head(facs_au={"AU12": 0.8, "intensity": 1.0})
        result = await ArkitFacsMapper().map(head)

        assert result["mouthSmileLeft"] == pytest.approx(0.8)
        assert result["mouthSmileRight"] == pytest.approx(0.8)

    async def test_au12_with_intensity_zero(self) -> None:
        """AU12=0.8，intensity=0.0 → mouthSmile*=0.0（增益归零）。"""
        head = _make_expression_head(facs_au={"AU12": 0.8, "intensity": 0.0})
        result = await ArkitFacsMapper().map(head)

        assert result["mouthSmileLeft"] == pytest.approx(0.0)
        assert result["mouthSmileRight"] == pytest.approx(0.0)

    async def test_apply_intensity_false_ignores_intensity(self) -> None:
        """apply_intensity=False → intensity 被忽略，输出原始 AU 值。"""
        head = _make_expression_head(facs_au={"AU12": 0.8, "intensity": 0.5})
        result = await ArkitFacsMapper(apply_intensity=False).map(head)

        # 无乘子：coeff = clamp(0.8 * 1.0) = 0.8
        assert result["mouthSmileLeft"] == pytest.approx(0.8)
        assert result["mouthSmileRight"] == pytest.approx(0.8)

    async def test_intensity_missing_defaults_to_gain_one(self) -> None:
        """facs_au 中无 intensity 键 → gain=1.0，输出原始 AU 值。"""
        head = _make_expression_head(facs_au={"AU12": 0.6})
        result = await ArkitFacsMapper().map(head)

        # gain 默认 1.0 → coeff = 0.6
        assert result["mouthSmileLeft"] == pytest.approx(0.6)
        assert result["mouthSmileRight"] == pytest.approx(0.6)

    async def test_apply_intensity_false_missing_intensity_key(self) -> None:
        """apply_intensity=False 且 facs_au 无 intensity → 输出原始 AU 值（不报 KeyError）。"""
        head = _make_expression_head(facs_au={"AU12": 0.7})
        result = await ArkitFacsMapper(apply_intensity=False).map(head)

        assert result["mouthSmileLeft"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# 6. 象限子集——只驱动 facs_au 中出现的 AU
# ---------------------------------------------------------------------------


class TestSubsetAUMapping:
    """facs_au 只含部分 AU 时，结果只含对应 blendshape，不含未出现 AU 的 blendshape。"""

    async def test_negative_valence_subset_only_drives_frown_and_brow_down(self) -> None:
        """负效价子集 {AU15, AU04, intensity=0.6} → 只有 mouthFrown*/browDown*，无 smile。"""
        head = _make_expression_head(facs_au={"AU15": 0.5, "AU04": 0.3, "intensity": 0.6})
        result = await ArkitFacsMapper().map(head)

        # 应驱动的 blendshape
        assert "mouthFrownLeft" in result
        assert "mouthFrownRight" in result
        assert "browDownLeft" in result
        assert "browDownRight" in result

        # 不应驱动的 blendshape（smile、eyeWide 等）
        assert "mouthSmileLeft" not in result
        assert "mouthSmileRight" not in result
        assert "eyeWideLeft" not in result
        assert "jawOpen" not in result

    async def test_negative_valence_subset_values_correct(self) -> None:
        """AU15=0.5，intensity=0.6 → mouthFrown*=0.3（0.5×0.6）。"""
        head = _make_expression_head(facs_au={"AU15": 0.5, "AU04": 0.3, "intensity": 0.6})
        result = await ArkitFacsMapper().map(head)

        assert result["mouthFrownLeft"] == pytest.approx(0.3)
        assert result["mouthFrownRight"] == pytest.approx(0.3)
        # AU04=0.3 × intensity=0.6 → browDown*=0.18
        assert result["browDownLeft"] == pytest.approx(0.18)
        assert result["browDownRight"] == pytest.approx(0.18)

    async def test_single_au_only_one_entry_group(self) -> None:
        """单 AU 输入 {AU26:0.7} → 结果只含 jawOpen（1 个 blendshape）。"""
        head = _make_expression_head(facs_au={"AU26": 0.7})
        result = await ArkitFacsMapper().map(head)

        assert list(result.keys()) == ["jawOpen"]


# ---------------------------------------------------------------------------
# 7. clamp：系数上界 1.0、下界 0.0
# ---------------------------------------------------------------------------


class TestClampBehavior:
    """coeff 被钳制到 [0.0, 1.0]——上溢和下溢均安全。"""

    async def test_intensity_overflow_clamps_to_one(self) -> None:
        """AU12=0.9，intensity=2.0（上游异常）→ mouthSmile* 被钳到 1.0。

        ExpressionHead pydantic 校验会拒绝 intensity>1.0，因此用 MagicMock
        绕过数据模型，直接测试 ArkitFacsMapper._clamp01 的保护效果。
        """
        from unittest.mock import MagicMock  # noqa: PLC0415

        # 构造一个 duck-type ExpressionHead：facs_au 含越界 intensity
        mock_head = MagicMock()
        mock_head.facs_au = {"AU12": 0.9, "intensity": 2.0}

        result = await ArkitFacsMapper().map(mock_head)

        assert result["mouthSmileLeft"] == pytest.approx(1.0)
        assert result["mouthSmileRight"] == pytest.approx(1.0)

    async def test_large_au_value_clamps_to_one(self) -> None:
        """AU12=1.5（越界）+ intensity=1.0 → mouthSmile* 被钳到 1.0（_clamp01 保护）。

        ExpressionHead pydantic 校验会拒绝 AU 值 >1.0，用 MagicMock 直接测映射层。
        """
        from unittest.mock import MagicMock  # noqa: PLC0415

        mock_head = MagicMock()
        mock_head.facs_au = {"AU12": 1.5, "intensity": 1.0}

        result = await ArkitFacsMapper().map(mock_head)

        assert result["mouthSmileLeft"] == pytest.approx(1.0)
        assert result["mouthSmileRight"] == pytest.approx(1.0)

    async def test_negative_au_value_clamps_to_zero(self) -> None:
        """AU12=-0.3（负值）+ intensity=1.0 → mouthSmile* 被钳到 0.0（_clamp01 保护）。

        ExpressionHead pydantic 校验会拒绝负 AU 值，用 MagicMock 直接测映射层。
        """
        from unittest.mock import MagicMock  # noqa: PLC0415

        mock_head = MagicMock()
        mock_head.facs_au = {"AU12": -0.3, "intensity": 1.0}

        result = await ArkitFacsMapper().map(mock_head)

        assert result["mouthSmileLeft"] == pytest.approx(0.0)
        assert result["mouthSmileRight"] == pytest.approx(0.0)

    async def test_intensity_almost_zero_does_not_go_negative(self) -> None:
        """AU12=0.5，intensity=0.0 → 结果 ≥ 0.0（不因 clamp 下界翻负）。"""
        head = _make_expression_head(facs_au={"AU12": 0.5, "intensity": 0.0})
        result = await ArkitFacsMapper().map(head)

        assert result["mouthSmileLeft"] >= 0.0
        assert result["mouthSmileLeft"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 8. 空 facs_au / 仅含 intensity 键
# ---------------------------------------------------------------------------


class TestEmptyOrIntensityOnlyFacsAu:
    """空 facs_au 或仅含 intensity 键 → 结果为空 dict。"""

    async def test_empty_facs_au_returns_empty_dict(self) -> None:
        """facs_au={} → map 返回 {}（无 AU，无 blendshape）。

        空 facs_au 是合法 ExpressionHead（键 ⊆ 全集、无越界值），直接构造无需兜底。
        """
        head = _make_expression_head(facs_au={})
        result = await ArkitFacsMapper().map(head)
        assert result == {}

    async def test_intensity_only_returns_empty_dict(self) -> None:
        """facs_au={"intensity":0.8} → map 返回 {}（intensity 不是 AU，无 blendshape 被驱动）。"""
        head = _make_expression_head(facs_au={"intensity": 0.8})
        result = await ArkitFacsMapper().map(head)
        assert result == {}


# ---------------------------------------------------------------------------
# 9. AU_TO_ARKIT 常量结构断言
# ---------------------------------------------------------------------------


class TestAuToArkitConstant:
    """AU_TO_ARKIT 常量键名和 blendshape 名满足规范。"""

    def test_au_to_arkit_has_12_entries(self) -> None:
        """AU_TO_ARKIT 包含恰好 12 个 AU 条目。"""
        assert len(AU_TO_ARKIT) == 12

    def test_intensity_not_in_au_to_arkit(self) -> None:
        """'intensity' 不在 AU_TO_ARKIT 键中（作乘子处理，非 AU）。"""
        assert "intensity" not in AU_TO_ARKIT

    def test_au_keys_have_au_prefix(self) -> None:
        """所有 AU_TO_ARKIT 键以 'AU' 前缀开头。"""
        for key in AU_TO_ARKIT:
            assert key.startswith("AU"), f"{key!r} 不以 'AU' 开头"

    def test_blendshape_names_are_camel_case(self) -> None:
        """所有 blendshape 名首字母小写（ARKit camelCase 规范）。"""
        for au, blendshapes in AU_TO_ARKIT.items():
            for bs in blendshapes:
                assert bs[0].islower(), f"AU {au} → {bs!r} 首字母不是小写（非 camelCase）"

    def test_symmetric_aus_have_two_blendshapes(self) -> None:
        """双侧 AU（如 AU02/AU04/AU05/AU06/AU07/AU12/AU15/AU20/AU23）有 2 个 blendshape。"""
        two_sided = {"AU02", "AU04", "AU05", "AU06", "AU07", "AU12", "AU15", "AU20", "AU23"}
        for au in two_sided:
            assert len(AU_TO_ARKIT[au]) == 2, (
                f"AU {au} 期望 2 个 blendshape，实际 {len(AU_TO_ARKIT[au])}"
            )

    def test_single_au_have_one_blendshape(self) -> None:
        """单侧 AU（AU01/AU17/AU26）只有 1 个 blendshape。"""
        one_sided = {"AU01", "AU17", "AU26"}
        for au in one_sided:
            assert len(AU_TO_ARKIT[au]) == 1, (
                f"AU {au} 期望 1 个 blendshape，实际 {len(AU_TO_ARKIT[au])}"
            )
