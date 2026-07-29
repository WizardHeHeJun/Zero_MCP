"""Zero↔MCP 边界契约数据模型单测（T6）。

覆盖：
  1. 值域校验：AffectStimulus / ModalityPrior 越界 raise ValidationError；合法值通过。
  2. facs_au 校验：接受 legacy 3 键子集、extended 9 键子集、全 13 键；
     未知键 "AU99" 被拒；值超 [0,1] 被拒。
  3. text_label 枚举门控：合法值通过；非法标签被拒。
  4. list→tuple 兼容：valence_arousal / mu 传 list 能存入 tuple 字段。
  5. LanguageOutput 可选：ExpressionBundle 不带 language 合法；带合法。
  6. from_step_output：完整 step 返回 dict（含外层 expression 键）与直接 expression
     子 dict 两种输入都能正确解析；缺 language 时 bundle.language is None。
  7. 真实形状 fixture：贴近 Zero 占位输出的 round-trip 测试。
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from src.agents.models.zero_affect import (
    FACS_KEYS,
    FACS_KEYS_EXT,
    TEXT_LABELS,
    AffectStimulus,
    ExpressionBundle,
    ExpressionHead,
    LanguageOutput,
    ModalityPrior,
    PhysiologyChannel,
)

# ---------------------------------------------------------------------------
# 辅助构造函数
# ---------------------------------------------------------------------------


def _make_physiology(
    heart_rate_bpm: float = 80.0,
    skin_conductance: float = 0.5,
    pupil_mm: float = 4.0,
) -> dict[str, Any]:
    """构造合法 PhysiologyChannel dict（**legacy avatar 口径**：sc 是旧 [0,1] 单位、非 μS）。

    ⚠ canonical=WESAD 的 sc 是 μS（[0,20]），mapper 按 μS 归一——此 helper 默认值仅供「解析零回归」
    类断言（不涉 mapper 值语义）；测 WESAD canonical 值语义见 TestPhysiologyChannelContract 与
    test_zero_physiology_mapper.py 的字面量（code-review W2）。
    """
    return {
        "heart_rate_bpm": heart_rate_bpm,
        "skin_conductance": skin_conductance,
        "pupil_mm": pupil_mm,
    }


def _make_prosody(
    speech_rate: float = 1.0,
    pitch: float = 1.0,
    energy: float = 0.7,
) -> dict[str, Any]:
    """构造合法 ProsodyChannel dict（倍率口径）。"""
    return {"speech_rate": speech_rate, "pitch": pitch, "energy": energy}


def _make_expression_head(
    facs_au: dict[str, float] | None = None,
    text_label: str = "content",
) -> dict[str, Any]:
    """构造合法 ExpressionHead dict（默认 legacy 3 键子集）。"""
    if facs_au is None:
        facs_au = {"AU12": 0.8, "AU06": 0.6, "intensity": 0.7}
    return {
        "facs_au": facs_au,
        "text_label": text_label,
        "physiology": _make_physiology(),
        "prosody": _make_prosody(),
    }


def _make_expression_bundle_dict(
    valence: float = 0.5,
    arousal: float = 0.3,
    *,
    include_language: bool = False,
) -> dict[str, Any]:
    """构造合法 ExpressionBundle dict，可选包含 language 字段。"""
    data: dict[str, Any] = {
        "valence_arousal": [valence, arousal],
        "spontaneous": _make_expression_head(text_label="content"),
        "voluntary": _make_expression_head(text_label="excited"),
    }
    if include_language:
        data["language"] = {
            "text": "你好",
            "affect": [valence, arousal],
            "iters": 1,
            "consistency": 0.9,
        }
    return data


# ---------------------------------------------------------------------------
# 1. AffectStimulus 值域校验
# ---------------------------------------------------------------------------


class TestAffectStimulusValidation:
    """AffectStimulus 值域：v/a 各维 [-1,1]，coping 可 None 或 [-1,1]。"""

    def test_valid_stimulus_passes(self) -> None:
        """合法构造通过。"""
        stim = AffectStimulus(valence=0.5, arousal=-0.3)
        assert stim.valence == pytest.approx(0.5)
        assert stim.arousal == pytest.approx(-0.3)
        assert stim.coping_potential is None

    def test_valid_stimulus_with_coping(self) -> None:
        """带 coping_potential 的合法构造通过。"""
        stim = AffectStimulus(valence=0.0, arousal=0.0, coping_potential=0.5)
        assert stim.coping_potential == pytest.approx(0.5)

    def test_valence_below_minus_one_raises(self) -> None:
        """valence < -1 raise ValidationError。"""
        with pytest.raises(ValidationError):
            AffectStimulus(valence=-1.01, arousal=0.0)

    def test_valence_above_one_raises(self) -> None:
        """valence > 1 raise ValidationError。"""
        with pytest.raises(ValidationError):
            AffectStimulus(valence=1.01, arousal=0.0)

    def test_arousal_below_minus_one_raises(self) -> None:
        """arousal < -1 raise ValidationError。"""
        with pytest.raises(ValidationError):
            AffectStimulus(valence=0.0, arousal=-1.5)

    def test_arousal_above_one_raises(self) -> None:
        """arousal > 1 raise ValidationError。"""
        with pytest.raises(ValidationError):
            AffectStimulus(valence=0.0, arousal=2.0)

    def test_coping_above_one_raises(self) -> None:
        """coping_potential > 1 raise ValidationError。"""
        with pytest.raises(ValidationError):
            AffectStimulus(valence=0.0, arousal=0.0, coping_potential=1.5)

    def test_coping_below_minus_one_raises(self) -> None:
        """coping_potential < -1 raise ValidationError。"""
        with pytest.raises(ValidationError):
            AffectStimulus(valence=0.0, arousal=0.0, coping_potential=-2.0)

    def test_boundary_values_pass(self) -> None:
        """边界值 -1.0 / 1.0 / None 均合法。"""
        stim = AffectStimulus(valence=-1.0, arousal=1.0, coping_potential=-1.0)
        assert stim.valence == pytest.approx(-1.0)
        assert stim.coping_potential == pytest.approx(-1.0)

    def test_extra_fields_forbidden(self) -> None:
        """extra="forbid" 拒绝未知字段。"""
        with pytest.raises(ValidationError):
            AffectStimulus(valence=0.0, arousal=0.0, unknown_field=1.0)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 2. ModalityPrior 值域校验
# ---------------------------------------------------------------------------


class TestModalityPriorValidation:
    """ModalityPrior：mu 各维 [-1,1]，precision 各维 > 0。"""

    def test_valid_prior_passes(self) -> None:
        """合法构造通过。"""
        prior = ModalityPrior(
            modality="vision",
            mu=(0.8, 0.2),
            precision=(0.5, 0.5),
        )
        assert prior.modality == "vision"
        assert prior.mu == (pytest.approx(0.8), pytest.approx(0.2))

    def test_mu_above_one_raises(self) -> None:
        """mu 中任一维 > 1 raise ValidationError。"""
        with pytest.raises(ValidationError):
            ModalityPrior(modality="vision", mu=(1.1, 0.0), precision=(0.5, 0.5))

    def test_mu_below_minus_one_raises(self) -> None:
        """mu 中任一维 < -1 raise ValidationError。"""
        with pytest.raises(ValidationError):
            ModalityPrior(modality="vision", mu=(0.0, -1.2), precision=(0.5, 0.5))

    def test_precision_zero_raises(self) -> None:
        """precision 中任一维 == 0 raise ValidationError。"""
        with pytest.raises(ValidationError):
            ModalityPrior(modality="audio", mu=(0.0, 0.0), precision=(0.0, 0.5))

    def test_precision_negative_raises(self) -> None:
        """precision 中任一维 < 0 raise ValidationError。"""
        with pytest.raises(ValidationError):
            ModalityPrior(modality="audio", mu=(0.0, 0.0), precision=(0.5, -0.1))

    def test_boundary_mu_values_pass(self) -> None:
        """mu 恰好 -1.0 / 1.0 合法。"""
        prior = ModalityPrior(modality="physio", mu=(-1.0, 1.0), precision=(0.1, 0.1))
        assert prior.mu == (pytest.approx(-1.0), pytest.approx(1.0))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_mu_raises(self, bad: float) -> None:
        """NaN/±inf 的 μ 被构造期拒绝。

        判别力见 `test_non_finite_mu_rejection_is_load_bearing`——本例单独绿不足以
        证明是 `_validate_ranges` 在挡（可能是 pydantic 顺手拒的），故配对照组。
        """
        with pytest.raises(ValidationError):
            ModalityPrior(modality="vision", mu=(bad, 0.0), precision=(0.5, 0.5))

    def test_non_finite_mu_rejection_is_load_bearing(self) -> None:
        """判别性对照：去掉 `_validate_ranges` 的孪生模型**接受** NaN μ。

        证明上一例的绿由我方 validator 承担，而非 pydantic 对 float 的默认行为
        （pydantic v2 默认 `allow_inf_nan=True`，裸 float 字段照收 NaN）。
        本仓沉淀「绿灯必须先证明它能红」的标准做法：可疑的绿必须能指出谁在挡。
        """

        class TwinWithoutValidator(BaseModel):
            model_config = ConfigDict(extra="forbid")

            modality: str
            mu: tuple[float, float]
            precision: tuple[float, float]

        twin = TwinWithoutValidator(modality="vision", mu=(float("nan"), 0.0), precision=(0.5, 0.5))
        assert math.isnan(twin.mu[0]), "对照模型应当接受 NaN——否则上一例的绿与 validator 无关"

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_precision_raises(self, bad: float) -> None:
        """NaN/±inf 的 Π 被构造期拒绝。

        ⚠ 这条曾是**双侧全漏**：我方 `pv <= 0.0` 与 Zero
        `src/agents/affect_math.py::expand_external_priors` 的 M3 判据
        （`pi_v <= 0.0` 正值关、`pi_v > precision_cap` 上界关）对 NaN 皆恒 False，
        而其 M7 只守 μ 不守 Π → NaN 精度会静默进入融合数学产出 NaN 后验，
        两侧都不 fail-fast（对比：越界 μ 至少被 M7 响亮 raise）。

        本条是 MCP 侧**单边兜底**，判据与对方状态解耦：不论 Zero 的
        expand_external_priors 是否自带有限性校验，我方都不依赖对方。
        「对方有守卫」是随时会变的运行时事实——2026-07-29 一天内实测经历三态：
        main `11c25b0` 无 → 其未提交工作树出现 M3′ `math.isfinite` → main `332cb40`
        起落地。本条是出网收口点，不随对方处于哪一态而增减。
        """
        with pytest.raises(ValidationError):
            ModalityPrior(modality="audio", mu=(0.0, 0.0), precision=(bad, 0.5))

    def test_frozen_blocks_post_construction_mutation(self) -> None:
        """构造后赋值被拒（frozen）。

        动机：构造期校验只在**构造那一刻**成立；此前本类是同仓唯一未上锁的契约模型
        （兄弟 ProsodyParams / PhysiologyParams 均 frozen=True），实测
        `prior.mu = (5.0, -9.0)` 静默生效并一路穿到发往 Zero 的载荷里。
        ⚠ frozen 挡不住 model_construct / 鸭子类型伪造，**不能替代**出境侧 M7 守卫
        （见 tests/mcp/test_zero_external_priors.py::TestM7MuDomain）。
        """
        prior = ModalityPrior(modality="vision", mu=(0.5, 0.5), precision=(0.2, 0.2))
        with pytest.raises(ValidationError):
            prior.mu = (5.0, -9.0)  # type: ignore[misc]
        assert ModalityPrior.model_config.get("frozen") is True

    def test_coping_field_optional(self) -> None:
        """coping 默认 None，显式设置合法值通过。"""
        prior = ModalityPrior(modality="text", mu=(0.0, 0.0), precision=(0.3, 0.3), coping=0.7)
        assert prior.coping == pytest.approx(0.7)

    def test_coping_out_of_range_raises(self) -> None:
        """coping > 1 raise ValidationError。"""
        with pytest.raises(ValidationError):
            ModalityPrior(modality="text", mu=(0.0, 0.0), precision=(0.3, 0.3), coping=1.5)

    def test_as_stream_returns_triple(self) -> None:
        """as_stream() 返回 (modality, mu, precision) 三元组。"""
        prior = ModalityPrior(modality="audio", mu=(0.4, -0.2), precision=(0.6, 0.8))
        name, mu, prec = prior.as_stream()
        assert name == "audio"
        assert mu == (pytest.approx(0.4), pytest.approx(-0.2))
        assert prec == (pytest.approx(0.6), pytest.approx(0.8))

    def test_list_to_tuple_conversion(self) -> None:
        """mu 与 precision 传 list，pydantic v2 自动强转为 tuple。"""
        prior = ModalityPrior(
            modality="vision",
            mu=[0.5, -0.3],  # type: ignore[arg-type]
            precision=[0.4, 0.6],  # type: ignore[arg-type]
        )
        assert isinstance(prior.mu, tuple)
        assert isinstance(prior.precision, tuple)


# ---------------------------------------------------------------------------
# 3. ExpressionHead facs_au 校验
# ---------------------------------------------------------------------------


class TestExpressionHeadFacsAu:
    """facs_au 键集与值域校验，兼容 legacy / extended 子集。"""

    def test_legacy_3_key_subset_accepted(self) -> None:
        """legacy 3 键子集 {AU12, AU06, intensity}（v >= 0 象限）被接受。"""
        head = ExpressionHead(**_make_expression_head({"AU12": 0.8, "AU06": 0.6, "intensity": 0.7}))
        assert "AU12" in head.facs_au

    def test_extended_9_key_subset_accepted(self) -> None:
        """extended 9 键子集（部分 FACS_KEYS_EXT）被接受。"""
        nine_keys = {
            "AU01": 0.1,
            "AU02": 0.2,
            "AU05": 0.3,
            "AU06": 0.4,
            "AU07": 0.1,
            "AU12": 0.9,
            "AU20": 0.3,
            "AU23": 0.5,
            "intensity": 0.8,
        }
        head = ExpressionHead(**_make_expression_head(nine_keys))
        assert len(head.facs_au) == 9

    def test_all_ext_keys_accepted(self) -> None:
        """全 13 键（FACS_KEYS_EXT 完整集）被接受。"""
        all_keys = {k: 0.5 for k in FACS_KEYS_EXT}
        head = ExpressionHead(**_make_expression_head(all_keys))
        assert set(head.facs_au.keys()) == set(FACS_KEYS_EXT)

    def test_all_5_legacy_keys_accepted(self) -> None:
        """全 5 键（FACS_KEYS 完整集）被接受。"""
        legacy_keys = {k: 0.5 for k in FACS_KEYS}
        head = ExpressionHead(**_make_expression_head(legacy_keys))
        assert set(head.facs_au.keys()) == set(FACS_KEYS)

    def test_unknown_au_key_rejected(self) -> None:
        """未知键 AU99 被 ValidationError 拒绝。"""
        with pytest.raises(ValidationError, match="AU99"):
            ExpressionHead(**_make_expression_head({"AU12": 0.8, "AU99": 0.5}))

    def test_facs_value_below_zero_rejected(self) -> None:
        """facs_au 值 < 0 被拒绝。"""
        with pytest.raises(ValidationError):
            ExpressionHead(**_make_expression_head({"AU12": -0.1, "intensity": 0.5}))

    def test_facs_value_above_one_rejected(self) -> None:
        """facs_au 值 > 1 被拒绝。"""
        with pytest.raises(ValidationError):
            ExpressionHead(**_make_expression_head({"AU12": 1.01, "intensity": 0.5}))

    def test_facs_boundary_values_accepted(self) -> None:
        """facs_au 值恰好为 0.0 / 1.0 被接受。"""
        head = ExpressionHead(**_make_expression_head({"AU12": 0.0, "intensity": 1.0}))
        assert head.facs_au["AU12"] == pytest.approx(0.0)
        assert head.facs_au["intensity"] == pytest.approx(1.0)

    def test_empty_facs_au_accepted(self) -> None:
        """空 facs_au dict 被接受（不要求全集，AD-4）。"""
        head = ExpressionHead(**_make_expression_head({}))
        assert head.facs_au == {}


# ---------------------------------------------------------------------------
# 4. text_label 枚举门控
# ---------------------------------------------------------------------------


class TestExpressionHeadTextLabel:
    """text_label ∈ TEXT_LABELS 校验。"""

    @pytest.mark.parametrize("label", sorted(TEXT_LABELS))
    def test_valid_labels_pass(self, label: str) -> None:
        """TEXT_LABELS 中每个合法标签通过。"""
        head = ExpressionHead(**_make_expression_head(text_label=label))
        assert head.text_label == label

    def test_invalid_label_raises(self) -> None:
        """非合法标签被拒绝。"""
        with pytest.raises(ValidationError, match="text_label"):
            ExpressionHead(**_make_expression_head(text_label="happy"))

    def test_empty_label_raises(self) -> None:
        """空字符串标签被拒绝。"""
        with pytest.raises(ValidationError):
            ExpressionHead(**_make_expression_head(text_label=""))


# ---------------------------------------------------------------------------
# 4b. Q1 prosody_scale 量纲标记（Zero 回传 2026-07-14）
# ---------------------------------------------------------------------------


class TestProsodyScale:
    """Q1：prosody_scale 是 ExpressionHead/ExpressionBundle 的**兄弟键**（非 prosody 子 dict 内，
    Zero `src/agents/affect_math.py::decode_channels` 把它与 prosody 平级输出，刻意如此）。
    normalized→prosody 三值收窄 [0,1]；ratio/缺省放宽（兼容当前 Zero 输出）。
    """

    def test_absent_scale_allows_ratio_prosody(self) -> None:
        """缺省 prosody_scale（decoder 未标注，如 mock）→ 放宽，倍率 prosody 通过。"""
        head = ExpressionHead(**_make_expression_head())
        assert head.prosody_scale is None

    def test_ratio_scale_allows_over_one_prosody(self) -> None:
        """prosody_scale="ratio" + 倍率 prosody（speech_rate>1）→ 通过。"""
        data = _make_expression_head()
        data["prosody"] = {"speech_rate": 1.5, "pitch": 1.3, "energy": 0.7}
        data["prosody_scale"] = "ratio"
        head = ExpressionHead(**data)
        assert head.prosody_scale == "ratio"

    def test_normalized_scale_accepts_in_range(self) -> None:
        """prosody_scale="normalized" 且 prosody 三值 ∈ [0,1] → 通过。"""
        data = _make_expression_head()
        data["prosody"] = {"speech_rate": 0.8, "pitch": 0.5, "energy": 0.9}
        data["prosody_scale"] = "normalized"
        head = ExpressionHead(**data)
        assert head.prosody_scale == "normalized"

    def test_normalized_scale_rejects_over_one(self) -> None:
        """prosody_scale="normalized" 但 prosody.speech_rate>1 → 拒绝。"""
        data = _make_expression_head()
        data["prosody"] = {"speech_rate": 1.5, "pitch": 0.5, "energy": 0.5}
        data["prosody_scale"] = "normalized"
        with pytest.raises(ValidationError, match="normalized"):
            ExpressionHead(**data)

    def test_invalid_scale_value_rejected(self) -> None:
        """非法 prosody_scale 值被拒绝。"""
        data = _make_expression_head()
        data["prosody_scale"] = "linear"
        with pytest.raises(ValidationError):
            ExpressionHead(**data)

    def test_scale_inside_prosody_dict_rejected(self) -> None:
        """兄弟键约束：塞进 prosody 子 dict 会被 ProsodyChannel extra=forbid 拒。"""
        data = _make_expression_head()
        data["prosody"] = {
            "speech_rate": 1.0,
            "pitch": 1.0,
            "energy": 0.7,
            "prosody_scale": "ratio",
        }
        with pytest.raises(ValidationError):
            ExpressionHead(**data)

    def test_bundle_top_level_prosody_scale(self) -> None:
        """ExpressionBundle 顶层接受 Zero 提升的 prosody_scale。

        提升点：Zero `src/agents/expression.py::ExpressionAgent` 的
        `expression["prosody_scale"] = spontaneous["prosody_scale"]`（仅当 spontaneous
        带该键时提升）。注：该模块 2026-07 已从 `src/expression.py` 迁到 `src/agents/`。
        """
        data = _make_expression_bundle_dict()
        data["prosody_scale"] = "ratio"
        bundle = ExpressionBundle(**data)
        assert bundle.prosody_scale == "ratio"


# ---------------------------------------------------------------------------
# 4.5 PhysiologyChannel canonical=WESAD 契约（physiology 对称接线 2026-07-23）
# ---------------------------------------------------------------------------


class TestPhysiologyChannelContract:
    """PhysiologyChannel canonical=WESAD {hr, sc(μS), temperature_c}；temp/pupil_mm 过渡期可选。

    跨仓迁移不原子：本模型须**同时**接受 canonical WESAD 形状与旧 avatar 占位形状（零回归），
    hr+sc 必填，temp/pupil 可选。extra=forbid 拒未知键。
    """

    def test_accepts_wesad_canonical_shape(self) -> None:
        """canonical WESAD {hr, sc(μS), temperature_c} → 解析、pupil_mm 缺省 None。"""
        ch = PhysiologyChannel(heart_rate_bpm=85.0, skin_conductance=10.0, temperature_c=35.0)
        assert ch.temperature_c == pytest.approx(35.0)
        assert ch.pupil_mm is None

    def test_accepts_legacy_placeholder_shape(self) -> None:
        """旧 avatar 占位 {hr, sc, pupil_mm} → 解析、temperature_c 缺省 None（零回归）。"""
        ch = PhysiologyChannel(heart_rate_bpm=80.0, skin_conductance=0.5, pupil_mm=4.0)
        assert ch.pupil_mm == pytest.approx(4.0)
        assert ch.temperature_c is None

    def test_accepts_minimal_required_only(self) -> None:
        """仅 hr+sc（必填）→ 解析，temp/pupil 皆 None。"""
        ch = PhysiologyChannel(heart_rate_bpm=80.0, skin_conductance=10.0)
        assert ch.temperature_c is None and ch.pupil_mm is None

    def test_rejects_missing_required(self) -> None:
        """缺 heart_rate_bpm 或 skin_conductance（必填）→ ValidationError。"""
        with pytest.raises(ValidationError):
            PhysiologyChannel(skin_conductance=10.0, temperature_c=35.0)  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            PhysiologyChannel(heart_rate_bpm=80.0, temperature_c=35.0)  # type: ignore[call-arg]

    def test_rejects_extra_key(self) -> None:
        """extra=forbid——未知键（如 respiration）→ ValidationError（防跨仓漂移悄悄塞新字段）。"""
        with pytest.raises(ValidationError):
            PhysiologyChannel(
                heart_rate_bpm=80.0,
                skin_conductance=10.0,
                temperature_c=35.0,
                respiration=0.5,  # type: ignore[call-arg]
            )


# ---------------------------------------------------------------------------
# 5. list→tuple 兼容
# ---------------------------------------------------------------------------


class TestListToTupleCompatibility:
    """pydantic v2 自动将 list 强转为 tuple 字段。"""

    def test_valence_arousal_from_list(self) -> None:
        """valence_arousal 传 [v, a] list 能存入 tuple 字段。"""
        bundle = ExpressionBundle(**_make_expression_bundle_dict(0.5, -0.3))
        assert isinstance(bundle.valence_arousal, tuple)
        assert bundle.valence_arousal == (pytest.approx(0.5), pytest.approx(-0.3))

    def test_language_affect_from_list(self) -> None:
        """LanguageOutput.affect 传 [v, a] list 能存入 tuple 字段。"""
        lang = LanguageOutput(text="hello", affect=[0.2, -0.1])  # type: ignore[arg-type]
        assert isinstance(lang.affect, tuple)
        assert lang.affect == (pytest.approx(0.2), pytest.approx(-0.1))

    def test_modality_prior_mu_from_list(self) -> None:
        """ModalityPrior.mu 传 list 能存入 tuple 字段。"""
        prior = ModalityPrior(
            modality="m",
            mu=[0.5, -0.3],  # type: ignore[arg-type]
            precision=[0.4, 0.6],  # type: ignore[arg-type]
        )
        assert isinstance(prior.mu, tuple)


# ---------------------------------------------------------------------------
# 6. LanguageOutput 可选
# ---------------------------------------------------------------------------


class TestLanguageOutputOptional:
    """ExpressionBundle.language 可选（None 或 LanguageOutput）。"""

    def test_bundle_without_language_is_valid(self) -> None:
        """不带 language 的 ExpressionBundle 合法，language is None。"""
        bundle = ExpressionBundle(**_make_expression_bundle_dict())
        assert bundle.language is None

    def test_bundle_with_language_is_valid(self) -> None:
        """带 language 的 ExpressionBundle 合法，各字段正确。"""
        bundle = ExpressionBundle(**_make_expression_bundle_dict(include_language=True))
        assert bundle.language is not None
        assert bundle.language.text == "你好"
        assert bundle.language.iters == 1
        assert bundle.language.consistency == pytest.approx(0.9)

    def test_language_output_minimal(self) -> None:
        """LanguageOutput 只需 text 字段，其余可选。"""
        lang = LanguageOutput(text="回复")
        assert lang.text == "回复"
        assert lang.affect is None
        assert lang.iters == 0
        assert lang.consistency is None


# ---------------------------------------------------------------------------
# 7. from_step_output 两种输入形态
# ---------------------------------------------------------------------------


class TestFromStepOutput:
    """from_step_output 兼容 Zero step() 完整返回 dict 与直接 expression 子 dict。"""

    def test_from_full_step_dict_with_expression_key(self) -> None:
        """喂含 'expression' 键的完整 step 返回 dict，正确解析。"""
        expression_data = _make_expression_bundle_dict(0.6, 0.4, include_language=True)
        step_out = {
            "expression": expression_data,
            "trace": {"step_id": "abc", "latency_ms": 120},
            "other_metadata": "ignored",
        }
        bundle = ExpressionBundle.from_step_output(step_out)
        assert bundle.valence_arousal == (pytest.approx(0.6), pytest.approx(0.4))
        assert bundle.language is not None

    def test_from_direct_expression_dict(self) -> None:
        """直接传 expression 子 dict（无外层键），正确解析。"""
        expression_data = _make_expression_bundle_dict(-0.3, 0.7)
        bundle = ExpressionBundle.from_step_output(expression_data)
        assert bundle.valence_arousal == (pytest.approx(-0.3), pytest.approx(0.7))

    def test_from_step_output_no_language_gives_none(self) -> None:
        """缺 language 时 bundle.language is None。"""
        step_out = {"expression": _make_expression_bundle_dict()}
        bundle = ExpressionBundle.from_step_output(step_out)
        assert bundle.language is None

    def test_extra_keys_in_expression_dict_are_rejected(self) -> None:
        """expression 子 dict 含 extra 键时 extra='forbid' 拒绝。"""
        bad_data = _make_expression_bundle_dict()
        bad_data["unknown_extra_field"] = "should_fail"
        with pytest.raises(ValidationError):
            ExpressionBundle.from_step_output(bad_data)

    def test_outer_extra_keys_are_not_forwarded(self) -> None:
        """外层 step dict 的 extra 键不会透传到 ExpressionBundle（隔离于外层 expression 取值）。"""
        expression_data = _make_expression_bundle_dict(0.1, 0.2)
        step_out = {
            "expression": expression_data,
            "trace": "外层无关字段",
        }
        # 不应 raise；外层 extra 键被 from_step_output 自然隔离
        bundle = ExpressionBundle.from_step_output(step_out)
        assert bundle.valence_arousal == (pytest.approx(0.1), pytest.approx(0.2))


# ---------------------------------------------------------------------------
# 8. 真实形状 fixture round-trip
# ---------------------------------------------------------------------------


class TestRealShapeRoundtrip:
    """贴近 Zero 占位输出的真实形状 fixture，断言 round-trip 正确。"""

    def test_zero_like_expression_round_trip(self) -> None:
        """构造贴近 Zero step() 占位输出的 expression dict，解析后字段值与原始一致。

        形状：
        - valence_arousal = (0.6, 0.4)（v >= 0, a >= 0 → excited 象限）
        - spontaneous/voluntary 各用 legacy 3 键（AU12/AU06/intensity）
        - physiology 用参考值域（hr=90, sc=0.6, pupil=4.0）
        - prosody 用倍率口径（speech_rate=1.1, pitch=1.05, energy=0.8）
        """
        v, a = 0.6, 0.4

        # 占位路径 v>=0 象限的 legacy 3 键
        facs_positive = {"AU12": 0.75, "AU06": 0.60, "intensity": 0.70}
        physio = {"heart_rate_bpm": 90.0, "skin_conductance": 0.6, "pupil_mm": 4.0}
        prosody = {"speech_rate": 1.1, "pitch": 1.05, "energy": 0.8}

        expression_data: dict[str, Any] = {
            "valence_arousal": [v, a],
            "spontaneous": {
                "facs_au": facs_positive,
                "text_label": "excited",
                "physiology": physio,
                "prosody": prosody,
            },
            "voluntary": {
                "facs_au": facs_positive,
                "text_label": "content",
                "physiology": physio,
                "prosody": prosody,
            },
        }

        bundle = ExpressionBundle.model_validate(expression_data)

        assert bundle.valence_arousal == (pytest.approx(v), pytest.approx(a))
        assert bundle.spontaneous.text_label == "excited"
        assert bundle.voluntary.text_label == "content"
        assert bundle.spontaneous.facs_au["AU12"] == pytest.approx(0.75)
        assert bundle.spontaneous.physiology.heart_rate_bpm == pytest.approx(90.0)
        assert bundle.voluntary.prosody.speech_rate == pytest.approx(1.1)
        assert bundle.language is None

    def test_zero_like_expression_with_language(self) -> None:
        """带 language 字段的 Zero 占位输出 round-trip，affect 从 list 转 tuple。"""
        v, a = -0.4, 0.7
        facs_negative = {"AU15": 0.6, "AU04": 0.5, "intensity": 0.8}
        physio = {"heart_rate_bpm": 100.0, "skin_conductance": 0.8, "pupil_mm": 4.5}
        prosody = {"speech_rate": 1.3, "pitch": 1.2, "energy": 0.9}

        expression_data: dict[str, Any] = {
            "valence_arousal": [v, a],
            "spontaneous": {
                "facs_au": facs_negative,
                "text_label": "angry",
                "physiology": physio,
                "prosody": prosody,
            },
            "voluntary": {
                "facs_au": facs_negative,
                "text_label": "sad",
                "physiology": physio,
                "prosody": prosody,
            },
            "language": {
                "text": "对不起，我现在很难受。",
                "affect": [v, a],
                "iters": 2,
                "consistency": 0.85,
            },
        }

        bundle = ExpressionBundle.model_validate(expression_data)
        assert bundle.language is not None
        assert bundle.language.text == "对不起，我现在很难受。"
        assert bundle.language.affect == (pytest.approx(v), pytest.approx(a))
        assert bundle.language.iters == 2

    def test_from_step_output_with_outer_wrapper(self) -> None:
        """完整 step_out（含 trace 等外层字段）通过 from_step_output 正确解析。"""
        v, a = 0.3, -0.2
        step_out: dict[str, Any] = {
            "expression": {
                "valence_arousal": [v, a],
                "spontaneous": {
                    "facs_au": {"AU12": 0.5, "intensity": 0.6},
                    "text_label": "content",
                    "physiology": {
                        "heart_rate_bpm": 75.0,
                        "skin_conductance": 0.4,
                        "pupil_mm": 3.5,
                    },
                    "prosody": {"speech_rate": 0.9, "pitch": 0.95, "energy": 0.6},
                },
                "voluntary": {
                    "facs_au": {"AU06": 0.4, "intensity": 0.5},
                    "text_label": "content",
                    "physiology": {
                        "heart_rate_bpm": 75.0,
                        "skin_conductance": 0.4,
                        "pupil_mm": 3.5,
                    },
                    "prosody": {"speech_rate": 0.9, "pitch": 0.95, "energy": 0.6},
                },
            },
            "step_id": "run-001",
            "latency_ms": 234,
        }
        bundle = ExpressionBundle.from_step_output(step_out)
        assert bundle.valence_arousal == (pytest.approx(v), pytest.approx(a))
        assert bundle.spontaneous.text_label == "content"


# ---------------------------------------------------------------------------
# D-7：ModalityPrior.as_stream() 的**丢弃面**（external_prior schema v1 不传 coping）
# ---------------------------------------------------------------------------

_AS_STREAM_CARRIED_FIELDS: frozenset[str] = frozenset({"modality", "mu", "precision"})
"""`as_stream()` 三元组实际携带的字段名集合（丢弃面 = model_fields − 本集合）。

之所以显式维护这份常量而不是从返回值里推：三元组里装的是**值**不是字段名，
`assert "coping" not in prior.as_stream()` 对任何实现都恒真（pitfalls⑥ 原型）。
"""


class TestAsStreamDiscardSurface:
    """`coping` 写进 ModalityPrior 会被 as_stream() 静默丢弃 —— 这是**已知且有意**的事实。

    Zero 侧 ExternalPrior 是 `(name, μ, Π)` 三元组、无 coping 槽位，故 schema v1 不传输。
    字段保留而非删除：`build_recommended_prior(coping=...)` 是既有公开入口，删属 breaking。
    本类把「丢弃面」钉成可执行记录——日后若 Zero 进 schema v2 把 coping 纳入三元组，
    或本模型新增字段，本用例会红并提示同步 Field description 与跨仓 schema 认知。
    """

    def test_as_stream_drops_exactly_coping(self) -> None:
        """丢弃面精确等于 {"coping"}——多一个少一个都要红。"""
        dropped = set(ModalityPrior.model_fields) - _AS_STREAM_CARRIED_FIELDS
        assert dropped == {"coping"}, (
            f"as_stream 的丢弃面变为 {sorted(dropped)}——须同步更新 ModalityPrior.coping 的 "
            "Field description（『v1 不传输』文案）与跨仓 schema 认知"
        )

    def test_as_stream_payload_is_exactly_the_carried_fields(self) -> None:
        """携带面：三元组逐值等于 (modality, mu, precision)，且元数与携带字段数一致。

        ⚠ 判别力：`coping=0.5` 显式传入（非默认 None），若实现改成把 coping 追加进元组，
        `len(...)` 那行接住；若改成挪用某一槽位装 coping，逐值相等那行接住。
        """
        prior = ModalityPrior(modality="vision", mu=(0.1, 0.2), precision=(0.2, 0.12), coping=0.5)
        # 正控：被丢弃的字段确实被赋了**非默认**值，否则「丢没丢」不可观测。
        assert prior.coping == 0.5

        stream = prior.as_stream()

        assert stream == ("vision", (0.1, 0.2), (0.2, 0.12))
        assert len(stream) == len(_AS_STREAM_CARRIED_FIELDS)
