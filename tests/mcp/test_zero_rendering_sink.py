"""RenderingExpressionSink / RenderFrame 单测（zero-link T2）。

覆盖：
  1. isinstance(RenderingExpressionSink(), ExpressionSink) 为 True。
  2. VOLUNTARY_ONLY：render 后 frames 长度 1、frame.head=="voluntary"、is_micro False。
  3. SPONTANEOUS_ONLY：1 帧、head=="spontaneous"、is_micro False。
  4. DUAL：2 帧——主帧 head=="voluntary" is_micro False + 微帧 head=="spontaneous" is_micro True。
  5. 无 prosody_mapper：frame.prosody is None。
  6. 带 LinearProsodyMapper()：frame.prosody 是 ProsodyParams 且值与直接调 mapper.map(head) 一致。
  7. facs_au/physiology 原样透传（frame.facs_au == head.facs_au dict、physiology == model_dump()）。
  8. text_label 透传。
  9. 多次 render 累积 frames。
  10. RenderFrame extra=forbid（非法字段抛 ValidationError）。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agents.models.zero_affect import ExpressionBundle, ExpressionHead
from src.mcp.zero.expression_sink import ExpressionSink, HeadPolicy
from src.mcp.zero.mappers.prosody import LinearProsodyMapper, ProsodyParams
from src.mcp.zero.sinks import RenderFrame, RenderingExpressionSink

# ---------------------------------------------------------------------------
# 辅助构造函数
# ---------------------------------------------------------------------------


def _make_expression_head(
    facs_au: dict[str, float] | None = None,
    text_label: str = "content",
    prosody_scale: str | None = "ratio",
) -> ExpressionHead:
    """构造合法 ExpressionHead 实例（默认 legacy 3 键 + ratio 量纲）。"""
    return ExpressionHead(
        facs_au=facs_au or {"AU12": 0.8, "AU06": 0.6, "intensity": 0.7},
        text_label=text_label,
        physiology={"heart_rate_bpm": 80.0, "skin_conductance": 0.5, "pupil_mm": 4.0},
        prosody={"speech_rate": 1.0, "pitch": 1.0, "energy": 0.7},
        prosody_scale=prosody_scale,
    )


def _make_expression_bundle(
    voluntary_label: str = "content",
    spontaneous_label: str = "excited",
    prosody_scale: str | None = "ratio",
) -> ExpressionBundle:
    """构造合法 ExpressionBundle 实例。"""
    return ExpressionBundle(
        valence_arousal=(0.5, 0.3),
        voluntary=_make_expression_head(text_label=voluntary_label, prosody_scale=prosody_scale),
        spontaneous=_make_expression_head(
            text_label=spontaneous_label, prosody_scale=prosody_scale
        ),
        prosody_scale=prosody_scale,
    )


# ---------------------------------------------------------------------------
# 1. ExpressionSink 协议符合性
# ---------------------------------------------------------------------------


class TestRenderingExpressionSinkProtocol:
    """RenderingExpressionSink 满足 ExpressionSink Protocol（runtime_checkable）。"""

    def test_isinstance_expression_sink(self) -> None:
        """isinstance(RenderingExpressionSink(), ExpressionSink) 为 True。"""
        sink = RenderingExpressionSink()
        assert isinstance(sink, ExpressionSink)

    def test_has_render_method(self) -> None:
        """RenderingExpressionSink 实例具有 render 方法。"""
        sink = RenderingExpressionSink()
        assert callable(sink.render)

    def test_initial_frames_empty(self) -> None:
        """构造后 frames 列表初始为空。"""
        sink = RenderingExpressionSink()
        assert sink.frames == []

    def test_prosody_mapper_none_by_default(self) -> None:
        """默认 prosody_mapper 为 None。"""
        sink = RenderingExpressionSink()
        assert sink.prosody_mapper is None


# ---------------------------------------------------------------------------
# 2. HeadPolicy 三档渲染行为
# ---------------------------------------------------------------------------


class TestRenderHeadPolicy:
    """render() 按 HeadPolicy 正确取头、构造帧。"""

    async def test_voluntary_only_one_frame(self) -> None:
        """VOLUNTARY_ONLY：render 后 frames 长度为 1。"""
        sink = RenderingExpressionSink()
        bundle = _make_expression_bundle()
        await sink.render(bundle, policy=HeadPolicy.VOLUNTARY_ONLY)

        assert len(sink.frames) == 1

    async def test_voluntary_only_head_name(self) -> None:
        """VOLUNTARY_ONLY：frame.head == "voluntary"。"""
        sink = RenderingExpressionSink()
        bundle = _make_expression_bundle()
        await sink.render(bundle, policy=HeadPolicy.VOLUNTARY_ONLY)

        assert sink.frames[0].head == "voluntary"

    async def test_voluntary_only_is_micro_false(self) -> None:
        """VOLUNTARY_ONLY：frame.is_micro == False。"""
        sink = RenderingExpressionSink()
        bundle = _make_expression_bundle()
        await sink.render(bundle, policy=HeadPolicy.VOLUNTARY_ONLY)

        assert sink.frames[0].is_micro is False

    async def test_spontaneous_only_one_frame(self) -> None:
        """SPONTANEOUS_ONLY：render 后 frames 长度为 1。"""
        sink = RenderingExpressionSink()
        bundle = _make_expression_bundle()
        await sink.render(bundle, policy=HeadPolicy.SPONTANEOUS_ONLY)

        assert len(sink.frames) == 1

    async def test_spontaneous_only_head_name(self) -> None:
        """SPONTANEOUS_ONLY：frame.head == "spontaneous"。"""
        sink = RenderingExpressionSink()
        bundle = _make_expression_bundle()
        await sink.render(bundle, policy=HeadPolicy.SPONTANEOUS_ONLY)

        assert sink.frames[0].head == "spontaneous"

    async def test_spontaneous_only_is_micro_false(self) -> None:
        """SPONTANEOUS_ONLY：frame.is_micro == False（非微表情泄漏帧）。"""
        sink = RenderingExpressionSink()
        bundle = _make_expression_bundle()
        await sink.render(bundle, policy=HeadPolicy.SPONTANEOUS_ONLY)

        assert sink.frames[0].is_micro is False

    async def test_dual_two_frames(self) -> None:
        """DUAL：render 后 frames 长度为 2。"""
        sink = RenderingExpressionSink()
        bundle = _make_expression_bundle()
        await sink.render(bundle, policy=HeadPolicy.DUAL)

        assert len(sink.frames) == 2

    async def test_dual_main_frame_voluntary(self) -> None:
        """DUAL：第 0 帧（主帧）head == "voluntary"、is_micro False。"""
        sink = RenderingExpressionSink()
        bundle = _make_expression_bundle()
        await sink.render(bundle, policy=HeadPolicy.DUAL)

        main = sink.frames[0]
        assert main.head == "voluntary"
        assert main.is_micro is False

    async def test_dual_micro_frame_spontaneous(self) -> None:
        """DUAL：第 1 帧（微表情泄漏帧）head == "spontaneous"、is_micro True。"""
        sink = RenderingExpressionSink()
        bundle = _make_expression_bundle()
        await sink.render(bundle, policy=HeadPolicy.DUAL)

        micro = sink.frames[1]
        assert micro.head == "spontaneous"
        assert micro.is_micro is True


# ---------------------------------------------------------------------------
# 3. ProsodyMapper 行为
# ---------------------------------------------------------------------------


class TestRenderProsodyMapper:
    """无 prosody_mapper 时 prosody None；带 LinearProsodyMapper 时值一致。"""

    async def test_no_mapper_prosody_none(self) -> None:
        """无 prosody_mapper：frame.prosody is None。"""
        sink = RenderingExpressionSink()
        bundle = _make_expression_bundle()
        await sink.render(bundle, policy=HeadPolicy.VOLUNTARY_ONLY)

        assert sink.frames[0].prosody is None

    async def test_with_linear_mapper_prosody_is_params(self) -> None:
        """带 LinearProsodyMapper：frame.prosody 是 ProsodyParams 实例。"""
        mapper = LinearProsodyMapper()
        sink = RenderingExpressionSink(prosody_mapper=mapper)
        bundle = _make_expression_bundle(prosody_scale="ratio")
        await sink.render(bundle, policy=HeadPolicy.VOLUNTARY_ONLY)

        assert isinstance(sink.frames[0].prosody, ProsodyParams)

    async def test_prosody_values_match_mapper_output(self) -> None:
        """带 LinearProsodyMapper：frame.prosody 值与直接调 mapper.map(head) 一致。"""
        mapper = LinearProsodyMapper()
        sink = RenderingExpressionSink(prosody_mapper=mapper)
        head = _make_expression_head(prosody_scale="ratio")
        # 用真实 ExpressionBundle 包装这个 head
        bundle = ExpressionBundle(
            valence_arousal=(0.5, 0.3),
            voluntary=head,
            spontaneous=_make_expression_head(prosody_scale="ratio"),
            prosody_scale="ratio",
        )
        await sink.render(bundle, policy=HeadPolicy.VOLUNTARY_ONLY)

        expected = await mapper.map(head)
        actual = sink.frames[0].prosody
        assert actual is not None
        assert actual.rate_ratio == pytest.approx(expected.rate_ratio)
        assert actual.pitch_semitones == pytest.approx(expected.pitch_semitones)
        assert actual.gain_db == pytest.approx(expected.gain_db)

    async def test_rate_ratio_equals_speech_rate_for_ratio_scale(self) -> None:
        """ratio 量纲下 rate_ratio 应等于 head.prosody.speech_rate（LinearProsodyMapper 约定）。"""
        mapper = LinearProsodyMapper()
        sink = RenderingExpressionSink(prosody_mapper=mapper)
        # speech_rate=1.2 倍率
        head = ExpressionHead(
            facs_au={"AU12": 0.8, "AU06": 0.6, "intensity": 0.7},
            text_label="content",
            physiology={"heart_rate_bpm": 80.0, "skin_conductance": 0.5, "pupil_mm": 4.0},
            prosody={"speech_rate": 1.2, "pitch": 1.0, "energy": 0.7},
            prosody_scale="ratio",
        )
        bundle = ExpressionBundle(
            valence_arousal=(0.5, 0.3),
            voluntary=head,
            spontaneous=_make_expression_head(prosody_scale="ratio"),
            prosody_scale="ratio",
        )
        await sink.render(bundle, policy=HeadPolicy.VOLUNTARY_ONLY)

        prosody = sink.frames[0].prosody
        assert prosody is not None
        assert prosody.rate_ratio == pytest.approx(1.2)

    async def test_dual_both_frames_have_prosody(self) -> None:
        """DUAL + mapper：两帧都有 prosody。"""
        mapper = LinearProsodyMapper()
        sink = RenderingExpressionSink(prosody_mapper=mapper)
        bundle = _make_expression_bundle(prosody_scale="ratio")
        await sink.render(bundle, policy=HeadPolicy.DUAL)

        assert sink.frames[0].prosody is not None
        assert sink.frames[1].prosody is not None

    async def test_dual_no_mapper_both_frames_prosody_none(self) -> None:
        """DUAL 无 mapper：两帧 prosody 均为 None。"""
        sink = RenderingExpressionSink()
        bundle = _make_expression_bundle()
        await sink.render(bundle, policy=HeadPolicy.DUAL)

        assert sink.frames[0].prosody is None
        assert sink.frames[1].prosody is None


# ---------------------------------------------------------------------------
# 4. facs_au / physiology / text_label 原样透传
# ---------------------------------------------------------------------------


class TestRenderFramePassthrough:
    """facs_au / physiology / text_label 原样透传到 RenderFrame。"""

    async def test_facs_au_passthrough(self) -> None:
        """frame.facs_au == dict(head.facs_au)，原样透传。"""
        facs = {"AU12": 0.9, "AU06": 0.5, "intensity": 0.6}
        sink = RenderingExpressionSink()
        head = _make_expression_head(facs_au=facs)
        bundle = ExpressionBundle(
            valence_arousal=(0.5, 0.3),
            voluntary=head,
            spontaneous=_make_expression_head(),
        )
        await sink.render(bundle, policy=HeadPolicy.VOLUNTARY_ONLY)

        assert sink.frames[0].facs_au == facs

    async def test_physiology_passthrough(self) -> None:
        """frame.physiology == head.physiology.model_dump()，原样透传。"""
        sink = RenderingExpressionSink()
        head = _make_expression_head()
        bundle = ExpressionBundle(
            valence_arousal=(0.5, 0.3),
            voluntary=head,
            spontaneous=_make_expression_head(),
        )
        await sink.render(bundle, policy=HeadPolicy.VOLUNTARY_ONLY)

        expected_physiology = head.physiology.model_dump()
        assert sink.frames[0].physiology == expected_physiology

    async def test_text_label_passthrough(self) -> None:
        """frame.text_label == head.text_label，原样透传。"""
        sink = RenderingExpressionSink()
        bundle = _make_expression_bundle(voluntary_label="excited")
        await sink.render(bundle, policy=HeadPolicy.VOLUNTARY_ONLY)

        assert sink.frames[0].text_label == "excited"

    async def test_spontaneous_facs_passthrough(self) -> None:
        """SPONTANEOUS_ONLY：spontaneous 头的 facs_au 透传正确。"""
        facs = {"AU04": 0.7, "AU15": 0.5, "intensity": 0.4}
        sink = RenderingExpressionSink()
        spont_head = _make_expression_head(facs_au=facs, text_label="angry")
        bundle = ExpressionBundle(
            valence_arousal=(-0.3, 0.5),
            voluntary=_make_expression_head(),
            spontaneous=spont_head,
        )
        await sink.render(bundle, policy=HeadPolicy.SPONTANEOUS_ONLY)

        assert sink.frames[0].facs_au == facs
        assert sink.frames[0].text_label == "angry"

    async def test_dual_voluntary_and_spontaneous_labels(self) -> None:
        """DUAL：主帧取 voluntary 的 text_label，微帧取 spontaneous 的 text_label。"""
        sink = RenderingExpressionSink()
        bundle = _make_expression_bundle(voluntary_label="content", spontaneous_label="excited")
        await sink.render(bundle, policy=HeadPolicy.DUAL)

        assert sink.frames[0].text_label == "content"
        assert sink.frames[1].text_label == "excited"


# ---------------------------------------------------------------------------
# 5. 多次 render 累积 frames
# ---------------------------------------------------------------------------


class TestRenderFrameAccumulation:
    """多次 render 调用累积 frames 到同一列表。"""

    async def test_two_renders_accumulate_frames(self) -> None:
        """VOLUNTARY_ONLY 调用两次，frames 累积为 2 条。"""
        sink = RenderingExpressionSink()
        bundle = _make_expression_bundle()

        await sink.render(bundle, policy=HeadPolicy.VOLUNTARY_ONLY)
        await sink.render(bundle, policy=HeadPolicy.VOLUNTARY_ONLY)

        assert len(sink.frames) == 2

    async def test_mixed_policy_renders_accumulate(self) -> None:
        """VOLUNTARY_ONLY + DUAL 各调一次，frames 累积为 1+2=3 条。"""
        sink = RenderingExpressionSink()
        bundle = _make_expression_bundle()

        await sink.render(bundle, policy=HeadPolicy.VOLUNTARY_ONLY)
        await sink.render(bundle, policy=HeadPolicy.DUAL)

        assert len(sink.frames) == 3

    async def test_frames_are_same_list_object(self) -> None:
        """frames 引用在多次 render 后始终是同一列表对象。"""
        sink = RenderingExpressionSink()
        bundle = _make_expression_bundle()
        original_frames = sink.frames

        await sink.render(bundle, policy=HeadPolicy.VOLUNTARY_ONLY)

        assert sink.frames is original_frames

    async def test_clear_empties_frames(self) -> None:
        """clear() 清空已收集帧，供多轮复用时防无界增长。"""
        sink = RenderingExpressionSink()
        bundle = _make_expression_bundle()
        await sink.render(bundle, policy=HeadPolicy.DUAL)
        assert len(sink.frames) == 2

        sink.clear()

        assert sink.frames == []
        # 清空后仍可继续累积
        await sink.render(bundle, policy=HeadPolicy.VOLUNTARY_ONLY)
        assert len(sink.frames) == 1


# ---------------------------------------------------------------------------
# 6. RenderFrame 模型约束
# ---------------------------------------------------------------------------


class TestRenderFrameModel:
    """RenderFrame extra=forbid 约束及字段校验。"""

    def test_render_frame_extra_forbid(self) -> None:
        """RenderFrame 拒绝非法的额外字段（extra="forbid"）。"""
        with pytest.raises(ValidationError):
            RenderFrame(
                head="voluntary",
                is_micro=False,
                text_label="content",
                facs_au={"AU12": 0.8},
                physiology={"heart_rate_bpm": 80.0, "skin_conductance": 0.5, "pupil_mm": 4.0},
                prosody=None,
                unknown_field="forbidden",  # type: ignore[call-arg]
            )

    def test_render_frame_valid_construction(self) -> None:
        """合法字段可以直接构造 RenderFrame。"""
        frame = RenderFrame(
            head="voluntary",
            is_micro=False,
            text_label="content",
            facs_au={"AU12": 0.8, "AU06": 0.6},
            physiology={"heart_rate_bpm": 80.0, "skin_conductance": 0.5, "pupil_mm": 4.0},
            prosody=None,
        )
        assert frame.head == "voluntary"
        assert frame.is_micro is False
        assert frame.prosody is None

    def test_render_frame_is_micro_default_false(self) -> None:
        """RenderFrame.is_micro 默认值为 False。"""
        frame = RenderFrame(
            head="spontaneous",
            text_label="excited",
            facs_au={"AU12": 0.5},
            physiology={"heart_rate_bpm": 75.0, "skin_conductance": 0.3, "pupil_mm": 3.5},
            prosody=None,
        )
        assert frame.is_micro is False

    def test_render_frame_head_literal_constraint(self) -> None:
        """RenderFrame.head 只接受 'spontaneous' / 'voluntary'（Literal 约束）。"""
        with pytest.raises(ValidationError):
            RenderFrame(
                head="unknown",  # type: ignore[arg-type]
                is_micro=False,
                text_label="content",
                facs_au={"AU12": 0.8},
                physiology={"heart_rate_bpm": 80.0, "skin_conductance": 0.5, "pupil_mm": 4.0},
                prosody=None,
            )


# ---------------------------------------------------------------------------
# 7. 顶层包导出验证
# ---------------------------------------------------------------------------


class TestTopLevelExport:
    """RenderFrame 与 RenderingExpressionSink 通过 src.mcp.zero 顶层包可访问。"""

    def test_render_frame_exported(self) -> None:
        """从 src.mcp.zero 顶层导入 RenderFrame 成功。"""
        from src.mcp.zero import RenderFrame as RF  # noqa: PLC0415

        assert RF is RenderFrame

    def test_rendering_sink_exported(self) -> None:
        """从 src.mcp.zero 顶层导入 RenderingExpressionSink 成功。"""
        from src.mcp.zero import RenderingExpressionSink as RES  # noqa: PLC0415

        assert RES is RenderingExpressionSink
