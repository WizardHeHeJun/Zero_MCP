"""zero-link 端到端集成测试（T3）——纯内存，无外部依赖。

覆盖：
  表达半程：
  1. 贴近 Zero 真实形状的 step_out（含 expression + prosody_scale="ratio"）
     → ExpressionRouter([RenderingExpressionSink(prosody_mapper=LinearProsodyMapper())],
       policy=HeadPolicy.DUAL).route(step_out)
     → 返回 ExpressionBundle 且 sink.frames 有 2 帧、prosody 已映射。
  2. rate_ratio == voluntary.prosody.speech_rate（ratio 量纲 pass-through）。

  感知半程：
  3. 两个 CallablePerceptionChannel（sense_fn 返回不同 ModalityPrior）
     → PerceptionHub.collect() → as_zero_streams() 形状正确、两条独立、不均值。

  契约回路：
  4. 整条通路（刺激注入形态 / mock expression→渲染帧）可跑通、无异常。
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.agents.models.zero_affect import ExpressionBundle, ModalityPrior
from src.mcp.zero.channels import CallablePerceptionChannel
from src.mcp.zero.expression_sink import ExpressionRouter, HeadPolicy
from src.mcp.zero.mappers.facs import (
    ArkitFacsMapper,  # noqa: F401 – used in TestDualProsodyFacsPipeline
)
from src.mcp.zero.mappers.prosody import LinearProsodyMapper, ProsodyParams
from src.mcp.zero.perception import PerceptionHub
from src.mcp.zero.sinks import RenderFrame, RenderingExpressionSink

# ---------------------------------------------------------------------------
# 辅助构造函数
# ---------------------------------------------------------------------------


def _make_expression_head_dict(
    facs_au: dict[str, float] | None = None,
    text_label: str = "content",
    speech_rate: float = 1.0,
    pitch: float = 1.0,
    energy: float = 0.7,
) -> dict[str, Any]:
    """构造合法 ExpressionHead dict（默认 legacy 3 键，ratio 量纲）。"""
    return {
        "facs_au": facs_au or {"AU12": 0.8, "AU06": 0.6, "intensity": 0.7},
        "text_label": text_label,
        "physiology": {"heart_rate_bpm": 80.0, "skin_conductance": 0.5, "pupil_mm": 4.0},
        "prosody": {"speech_rate": speech_rate, "pitch": pitch, "energy": energy},
        "prosody_scale": "ratio",
    }


def _make_step_out(
    valence: float = 0.5,
    arousal: float = 0.3,
    *,
    voluntary_label: str = "content",
    spontaneous_label: str = "excited",
    voluntary_speech_rate: float = 1.0,
    spontaneous_speech_rate: float = 0.8,
) -> dict[str, Any]:
    """构造贴近 Zero 真实形状的 step_out dict。

    含外层 expression 键 + prosody_scale="ratio"（Q1 已定 2026-07-14）。
    """
    return {
        "expression": {
            "valence_arousal": [valence, arousal],
            "prosody_scale": "ratio",
            "voluntary": _make_expression_head_dict(
                text_label=voluntary_label,
                speech_rate=voluntary_speech_rate,
            ),
            "spontaneous": _make_expression_head_dict(
                text_label=spontaneous_label,
                speech_rate=spontaneous_speech_rate,
            ),
        },
        "trace": {"step": 1},
    }


def _make_prior(
    modality: str,
    mu: tuple[float, float],
    precision: tuple[float, float] = (0.5, 0.5),
) -> ModalityPrior:
    """构造合法 ModalityPrior。"""
    return ModalityPrior(modality=modality, mu=mu, precision=precision)


# ---------------------------------------------------------------------------
# 1. 表达半程：DUAL + LinearProsodyMapper
# ---------------------------------------------------------------------------


class TestExpressionPipelineDual:
    """表达半程端到端：step_out → ExpressionRouter → RenderingExpressionSink。"""

    async def test_route_returns_expression_bundle(self) -> None:
        """route(step_out) 返回正确 ExpressionBundle。"""
        mapper = LinearProsodyMapper()
        sink = RenderingExpressionSink(prosody_mapper=mapper)
        router = ExpressionRouter([sink], policy=HeadPolicy.DUAL)

        step_out = _make_step_out(valence=0.5, arousal=0.3)
        bundle = await router.route(step_out)

        assert isinstance(bundle, ExpressionBundle)
        assert bundle.valence_arousal == (pytest.approx(0.5), pytest.approx(0.3))

    async def test_dual_produces_two_frames(self) -> None:
        """DUAL 策略：sink.frames 有 2 帧。"""
        mapper = LinearProsodyMapper()
        sink = RenderingExpressionSink(prosody_mapper=mapper)
        router = ExpressionRouter([sink], policy=HeadPolicy.DUAL)

        await router.route(_make_step_out())

        assert len(sink.frames) == 2

    async def test_dual_main_frame_voluntary(self) -> None:
        """DUAL 第 0 帧：head=="voluntary"、is_micro False。"""
        sink = RenderingExpressionSink(prosody_mapper=LinearProsodyMapper())
        router = ExpressionRouter([sink], policy=HeadPolicy.DUAL)

        await router.route(_make_step_out())

        main = sink.frames[0]
        assert main.head == "voluntary"
        assert main.is_micro is False

    async def test_dual_micro_frame_spontaneous(self) -> None:
        """DUAL 第 1 帧：head=="spontaneous"、is_micro True。"""
        sink = RenderingExpressionSink(prosody_mapper=LinearProsodyMapper())
        router = ExpressionRouter([sink], policy=HeadPolicy.DUAL)

        await router.route(_make_step_out())

        micro = sink.frames[1]
        assert micro.head == "spontaneous"
        assert micro.is_micro is True

    async def test_prosody_mapped_on_frames(self) -> None:
        """带 LinearProsodyMapper 时两帧 prosody 均已映射（非 None）。"""
        sink = RenderingExpressionSink(prosody_mapper=LinearProsodyMapper())
        router = ExpressionRouter([sink], policy=HeadPolicy.DUAL)

        await router.route(_make_step_out())

        assert isinstance(sink.frames[0].prosody, ProsodyParams)
        assert isinstance(sink.frames[1].prosody, ProsodyParams)

    async def test_rate_ratio_matches_voluntary_speech_rate(self) -> None:
        """ratio 量纲：主帧 prosody.rate_ratio == voluntary.prosody.speech_rate。"""
        voluntary_rate = 1.3
        sink = RenderingExpressionSink(prosody_mapper=LinearProsodyMapper())
        router = ExpressionRouter([sink], policy=HeadPolicy.DUAL)

        await router.route(_make_step_out(voluntary_speech_rate=voluntary_rate))

        main_prosody = sink.frames[0].prosody
        assert main_prosody is not None
        assert main_prosody.rate_ratio == pytest.approx(voluntary_rate)

    async def test_rate_ratio_matches_spontaneous_speech_rate(self) -> None:
        """ratio 量纲：微帧 prosody.rate_ratio == spontaneous.prosody.speech_rate。"""
        spontaneous_rate = 0.8
        sink = RenderingExpressionSink(prosody_mapper=LinearProsodyMapper())
        router = ExpressionRouter([sink], policy=HeadPolicy.DUAL)

        await router.route(_make_step_out(spontaneous_speech_rate=spontaneous_rate))

        micro_prosody = sink.frames[1].prosody
        assert micro_prosody is not None
        assert micro_prosody.rate_ratio == pytest.approx(spontaneous_rate)

    async def test_text_labels_correctly_routed(self) -> None:
        """主帧 text_label 来自 voluntary、微帧来自 spontaneous。"""
        sink = RenderingExpressionSink()
        router = ExpressionRouter([sink], policy=HeadPolicy.DUAL)

        await router.route(_make_step_out(voluntary_label="content", spontaneous_label="excited"))

        assert sink.frames[0].text_label == "content"
        assert sink.frames[1].text_label == "excited"

    async def test_facs_au_passthrough_in_e2e(self) -> None:
        """端到端 facs_au 原样透传到 RenderFrame。"""
        sink = RenderingExpressionSink()
        router = ExpressionRouter([sink], policy=HeadPolicy.VOLUNTARY_ONLY)

        step_out = _make_step_out()
        bundle = await router.route(step_out)

        expected_facs = dict(bundle.voluntary.facs_au)
        assert sink.frames[0].facs_au == expected_facs

    async def test_physiology_passthrough_in_e2e(self) -> None:
        """端到端 physiology 原样透传（model_dump()）到 RenderFrame。"""
        sink = RenderingExpressionSink()
        router = ExpressionRouter([sink], policy=HeadPolicy.VOLUNTARY_ONLY)

        step_out = _make_step_out()
        bundle = await router.route(step_out)

        expected_physiology = bundle.voluntary.physiology.model_dump()
        assert sink.frames[0].physiology == expected_physiology


# ---------------------------------------------------------------------------
# 2. 表达半程：VOLUNTARY_ONLY 与 SPONTANEOUS_ONLY
# ---------------------------------------------------------------------------


class TestExpressionPipelineSingleHead:
    """VOLUNTARY_ONLY 和 SPONTANEOUS_ONLY 端到端单帧验证。"""

    async def test_voluntary_only_one_frame(self) -> None:
        """VOLUNTARY_ONLY：端到端后 sink.frames 恰好 1 帧。"""
        sink = RenderingExpressionSink()
        router = ExpressionRouter([sink], policy=HeadPolicy.VOLUNTARY_ONLY)

        await router.route(_make_step_out())

        assert len(sink.frames) == 1
        assert sink.frames[0].head == "voluntary"

    async def test_spontaneous_only_one_frame(self) -> None:
        """SPONTANEOUS_ONLY：端到端后 sink.frames 恰好 1 帧。"""
        sink = RenderingExpressionSink()
        router = ExpressionRouter([sink], policy=HeadPolicy.SPONTANEOUS_ONLY)

        await router.route(_make_step_out())

        assert len(sink.frames) == 1
        assert sink.frames[0].head == "spontaneous"


# ---------------------------------------------------------------------------
# 3. 感知半程：CallablePerceptionChannel + PerceptionHub
# ---------------------------------------------------------------------------


class TestPerceptionPipeline:
    """感知半程端到端：CallablePerceptionChannel → PerceptionHub → as_zero_streams。"""

    async def test_two_channels_collect_both_priors(self) -> None:
        """两个 CallablePerceptionChannel → collect() 收集 2 条独立先验。"""
        prior_a = _make_prior("vision", mu=(0.8, 0.2), precision=(0.6, 0.6))
        prior_b = _make_prior("audio", mu=(-0.6, 0.5), precision=(0.4, 0.4))

        hub = PerceptionHub(
            [
                CallablePerceptionChannel("vision", AsyncMock(return_value=prior_a)),
                CallablePerceptionChannel("audio", AsyncMock(return_value=prior_b)),
            ]
        )
        priors = await hub.collect()

        assert len(priors) == 2

    async def test_no_averaging_ad3(self) -> None:
        """AD-3 核心断言：两条先验独立保留，不做均值融合。"""
        prior_a = _make_prior("vision", mu=(0.8, 0.2), precision=(0.6, 0.6))
        prior_b = _make_prior("audio", mu=(-0.6, 0.5), precision=(0.4, 0.4))

        hub = PerceptionHub(
            [
                CallablePerceptionChannel("vision", AsyncMock(return_value=prior_a)),
                CallablePerceptionChannel("audio", AsyncMock(return_value=prior_b)),
            ]
        )
        priors = await hub.collect()
        mus = [p.mu for p in priors]

        # 两条 mu 都应在结果中
        assert (pytest.approx(0.8), pytest.approx(0.2)) in mus
        assert (pytest.approx(-0.6), pytest.approx(0.5)) in mus

        # 均值 (0.1, 0.35) 不应出现
        avg_v = (0.8 + -0.6) / 2
        avg_a = (0.2 + 0.5) / 2
        for mu in mus:
            assert not (mu[0] == pytest.approx(avg_v) and mu[1] == pytest.approx(avg_a)), (
                f"collect 返回了均值 mu={mu}，违反 AD-3"
            )

    async def test_as_zero_streams_shape_two_entries(self) -> None:
        """as_zero_streams 返回两条独立流，每条三元组 (name, (v,a), (Πv,Πa))。"""
        prior_a = _make_prior("vision", mu=(0.8, 0.2), precision=(0.6, 0.6))
        prior_b = _make_prior("audio", mu=(-0.6, 0.5), precision=(0.4, 0.4))

        hub = PerceptionHub(
            [
                CallablePerceptionChannel("vision", AsyncMock(return_value=prior_a)),
                CallablePerceptionChannel("audio", AsyncMock(return_value=prior_b)),
            ]
        )
        priors = await hub.collect()
        streams = PerceptionHub.as_zero_streams(priors)

        assert len(streams) == 2
        for stream in streams:
            name, mu, prec = stream
            assert isinstance(name, str)
            assert isinstance(mu, tuple) and len(mu) == 2
            assert isinstance(prec, tuple) and len(prec) == 2

    async def test_stream_values_match_priors(self) -> None:
        """as_zero_streams 的每条流值与原先验精确对应。"""
        prior_a = _make_prior("vision", mu=(0.8, 0.2), precision=(0.6, 0.6))
        prior_b = _make_prior("audio", mu=(-0.6, 0.5), precision=(0.4, 0.4))

        hub = PerceptionHub(
            [
                CallablePerceptionChannel("vision", AsyncMock(return_value=prior_a)),
                CallablePerceptionChannel("audio", AsyncMock(return_value=prior_b)),
            ]
        )
        priors = await hub.collect()
        streams = PerceptionHub.as_zero_streams(priors)
        stream_map = {s[0]: s for s in streams}

        assert stream_map["vision"][1] == (pytest.approx(0.8), pytest.approx(0.2))
        assert stream_map["vision"][2] == (pytest.approx(0.6), pytest.approx(0.6))
        assert stream_map["audio"][1] == (pytest.approx(-0.6), pytest.approx(0.5))
        assert stream_map["audio"][2] == (pytest.approx(0.4), pytest.approx(0.4))

    async def test_exception_in_one_channel_other_preserved(self) -> None:
        """单通道抛异常被跳过，另一通道先验正常收集。"""
        good_prior = _make_prior("audio", mu=(0.3, 0.4), precision=(0.5, 0.5))

        hub = PerceptionHub(
            [
                CallablePerceptionChannel("vision", AsyncMock(side_effect=RuntimeError("fail"))),
                CallablePerceptionChannel("audio", AsyncMock(return_value=good_prior)),
            ]
        )
        priors = await hub.collect()

        assert len(priors) == 1
        assert priors[0].modality == "audio"


# ---------------------------------------------------------------------------
# 4. 端到端契约回路（完整路径）
# ---------------------------------------------------------------------------


class TestFullContractLoop:
    """完整契约回路：刺激输入形态 + mock expression→渲染帧跑通验证。"""

    async def test_full_expression_loop_no_exception(self) -> None:
        """完整表达半程（step_out → router → sink）无异常抛出。"""
        mapper = LinearProsodyMapper()
        sink = RenderingExpressionSink(prosody_mapper=mapper)
        router = ExpressionRouter([sink], policy=HeadPolicy.DUAL)
        step_out = _make_step_out(valence=0.6, arousal=0.4)

        bundle = await router.route(step_out)

        assert bundle is not None
        assert len(sink.frames) == 2
        for frame in sink.frames:
            assert isinstance(frame, RenderFrame)

    async def test_full_perception_loop_no_exception(self) -> None:
        """完整感知半程（两通道 → collect → as_zero_streams）无异常抛出。"""
        hub = PerceptionHub(
            [
                CallablePerceptionChannel(
                    "vision",
                    AsyncMock(return_value=_make_prior("vision", mu=(0.5, 0.2))),
                ),
                CallablePerceptionChannel(
                    "audio",
                    AsyncMock(return_value=_make_prior("audio", mu=(-0.3, 0.6))),
                ),
            ]
        )
        priors = await hub.collect()
        streams = PerceptionHub.as_zero_streams(priors)

        assert len(streams) == 2
        assert all(len(s) == 3 for s in streams)

    async def test_sink_frames_are_render_frame_instances(self) -> None:
        """端到端输出的帧均为合法 RenderFrame 实例（Pydantic 模型，extra=forbid）。"""
        sink = RenderingExpressionSink(prosody_mapper=LinearProsodyMapper())
        router = ExpressionRouter([sink], policy=HeadPolicy.DUAL)

        await router.route(_make_step_out())

        for frame in sink.frames:
            assert isinstance(frame, RenderFrame)
            # RenderFrame 有所有必要字段
            assert frame.head in ("voluntary", "spontaneous")
            assert isinstance(frame.is_micro, bool)
            assert isinstance(frame.text_label, str)
            assert isinstance(frame.facs_au, dict)
            assert isinstance(frame.physiology, dict)

    async def test_bundle_voluntary_label_matches_step_out(self) -> None:
        """从 step_out 解析的 bundle voluntary.text_label 与输入一致。"""
        sink = RenderingExpressionSink()
        router = ExpressionRouter([sink], policy=HeadPolicy.VOLUNTARY_ONLY)

        bundle = await router.route(
            _make_step_out(voluntary_label="content", spontaneous_label="excited")
        )

        assert bundle.voluntary.text_label == "content"
        assert bundle.spontaneous.text_label == "excited"

    async def test_multiple_step_outs_accumulate_frames(self) -> None:
        """多次 route 调用累积 frames（模拟 Zero 连续 step()）。"""
        sink = RenderingExpressionSink()
        router = ExpressionRouter([sink], policy=HeadPolicy.VOLUNTARY_ONLY)

        for _ in range(3):
            await router.route(_make_step_out())

        assert len(sink.frames) == 3

    async def test_perception_and_expression_pipelines_independent(self) -> None:
        """感知半程与表达半程互不干扰——可在同一测试中并行执行。"""
        # 感知半程
        hub = PerceptionHub(
            [
                CallablePerceptionChannel(
                    "vision",
                    AsyncMock(return_value=_make_prior("vision", mu=(0.7, 0.1))),
                ),
            ]
        )
        priors = await hub.collect()
        streams = PerceptionHub.as_zero_streams(priors)

        # 表达半程
        sink = RenderingExpressionSink(prosody_mapper=LinearProsodyMapper())
        router = ExpressionRouter([sink], policy=HeadPolicy.DUAL)
        bundle = await router.route(_make_step_out())

        # 两条路径结果独立正确
        assert len(streams) == 1
        assert len(sink.frames) == 2
        assert isinstance(bundle, ExpressionBundle)


# ---------------------------------------------------------------------------
# 5. 双映射端到端：prosody + facs 同时驱动
# ---------------------------------------------------------------------------


class TestDualProsodyFacsPipeline:
    """端到端：expression → prosody + facs 双映射 → RenderFrame 同时含两字段。"""

    async def test_dual_mapper_frame_has_prosody_and_facs_mapped(self) -> None:
        """RenderingExpressionSink(prosody_mapper=..., facs_mapper=...) + DUAL
        → 每帧同时有 prosody(ProsodyParams) 与 facs_mapped(ARKit dict)。"""
        sink = RenderingExpressionSink(
            prosody_mapper=LinearProsodyMapper(),
            facs_mapper=ArkitFacsMapper(),
        )
        router = ExpressionRouter([sink], policy=HeadPolicy.DUAL)

        await router.route(_make_step_out())

        assert len(sink.frames) == 2
        for frame in sink.frames:
            assert isinstance(frame.prosody, ProsodyParams), (
                f"frame.prosody 应为 ProsodyParams，实际 {type(frame.prosody)}"
            )
            assert isinstance(frame.facs_mapped, dict), (
                f"frame.facs_mapped 应为 dict，实际 {type(frame.facs_mapped)}"
            )
            assert len(frame.facs_mapped) > 0, "facs_mapped 不应为空 dict（AU12/AU06 均应被驱动）"

    async def test_dual_mapper_facs_mapped_contains_smile_and_cheek(self) -> None:
        """默认 facs_au（AU12/AU06/intensity）→ facs_mapped 含 mouthSmile* 与 cheekSquint*。"""
        sink = RenderingExpressionSink(
            prosody_mapper=LinearProsodyMapper(),
            facs_mapper=ArkitFacsMapper(),
        )
        router = ExpressionRouter([sink], policy=HeadPolicy.VOLUNTARY_ONLY)

        await router.route(_make_step_out())

        facs_mapped = sink.frames[0].facs_mapped
        assert facs_mapped is not None
        assert "mouthSmileLeft" in facs_mapped
        assert "mouthSmileRight" in facs_mapped
        assert "cheekSquintLeft" in facs_mapped
        assert "cheekSquintRight" in facs_mapped

    async def test_dual_mapper_prosody_rate_ratio_correct(self) -> None:
        """双映射时 prosody.rate_ratio 仍等于 speech_rate（ratio 量纲，无干扰）。"""
        voluntary_rate = 1.2
        sink = RenderingExpressionSink(
            prosody_mapper=LinearProsodyMapper(),
            facs_mapper=ArkitFacsMapper(),
        )
        router = ExpressionRouter([sink], policy=HeadPolicy.DUAL)

        await router.route(_make_step_out(voluntary_speech_rate=voluntary_rate))

        main_prosody = sink.frames[0].prosody
        assert main_prosody is not None
        assert main_prosody.rate_ratio == pytest.approx(voluntary_rate)

    async def test_dual_mapper_facs_au_passthrough_unaffected(self) -> None:
        """双映射时 facs_au 原样透传字段不受 facs_mapped 影响。"""
        sink = RenderingExpressionSink(
            prosody_mapper=LinearProsodyMapper(),
            facs_mapper=ArkitFacsMapper(),
        )
        router = ExpressionRouter([sink], policy=HeadPolicy.VOLUNTARY_ONLY)

        step_out = _make_step_out()
        bundle = await router.route(step_out)

        expected_facs_au = dict(bundle.voluntary.facs_au)
        assert sink.frames[0].facs_au == expected_facs_au

    async def test_dual_mapper_no_exception_full_pipeline(self) -> None:
        """prosody + facs 双映射完整链路无异常，返回有效 ExpressionBundle。"""
        sink = RenderingExpressionSink(
            prosody_mapper=LinearProsodyMapper(),
            facs_mapper=ArkitFacsMapper(),
        )
        router = ExpressionRouter([sink], policy=HeadPolicy.DUAL)

        bundle = await router.route(_make_step_out(valence=0.6, arousal=0.4))

        assert isinstance(bundle, ExpressionBundle)
        for frame in sink.frames:
            assert isinstance(frame, RenderFrame)
