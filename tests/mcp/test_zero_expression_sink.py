"""ExpressionRouter 分发、HeadPolicy 行为与 ExpressionSink 协议单测（T7）。

覆盖：
  1. ExpressionRouter.route 用真实形状 step_out，返回正确 ExpressionBundle。
  2. 各 sink.render 被调用（传入 bundle 与 policy）。
  3. HeadPolicy 三档（voluntary_only / spontaneous_only / dual）传给 sink 的 policy 符合预期。
  4. 单 sink 抛异常不拖垮其他 sink（其余仍被调用）。
  5. isinstance 协议符合性（ExpressionSink / FacsMapper / ProsodyMapper / PhysiologyMapper）。
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.models.zero_affect import ExpressionBundle, ExpressionHead
from src.mcp.zero.expression_sink import (
    ExpressionRouter,
    ExpressionSink,
    FacsMapper,
    HeadPolicy,
    PhysiologyMapper,
    ProsodyMapper,
)

# ---------------------------------------------------------------------------
# 辅助构造函数
# ---------------------------------------------------------------------------


def _make_expression_head_dict(
    facs_au: dict[str, float] | None = None,
    text_label: str = "content",
) -> dict[str, Any]:
    """构造合法 ExpressionHead dict（默认 legacy 3 键）。"""
    return {
        "facs_au": facs_au or {"AU12": 0.8, "AU06": 0.6, "intensity": 0.7},
        "text_label": text_label,
        "physiology": {"heart_rate_bpm": 80.0, "skin_conductance": 0.5, "pupil_mm": 4.0},
        "prosody": {"speech_rate": 1.0, "pitch": 1.0, "energy": 0.7},
    }


def _make_step_out(
    valence: float = 0.5,
    arousal: float = 0.3,
    *,
    with_expression_wrapper: bool = True,
    with_language: bool = False,
    spontaneous_label: str = "excited",
    voluntary_label: str = "content",
) -> dict[str, Any]:
    """构造真实形状的 step_out dict。

    with_expression_wrapper=True 模拟 Zero session.step() 完整返回（含 'expression' 键）。
    with_expression_wrapper=False 模拟直接传 expression 子 dict。
    """
    expression: dict[str, Any] = {
        "valence_arousal": [valence, arousal],
        "spontaneous": _make_expression_head_dict(text_label=spontaneous_label),
        "voluntary": _make_expression_head_dict(text_label=voluntary_label),
    }
    if with_language:
        expression["language"] = {
            "text": "你好，我很开心。",
            "affect": [valence, arousal],
            "iters": 1,
            "consistency": 0.9,
        }
    if with_expression_wrapper:
        return {"expression": expression, "trace": {"step": 1}}
    return expression


def _make_sink() -> Any:
    """构造满足 ExpressionSink Protocol 结构的假 sink。"""
    sink = MagicMock()
    sink.render = AsyncMock(return_value=None)
    return sink


# ---------------------------------------------------------------------------
# 1. ExpressionRouter.route 基础行为
# ---------------------------------------------------------------------------


class TestExpressionRouterRoute:
    """route() 解析 step_out 并分发各 sink。"""

    async def test_route_returns_correct_bundle(self) -> None:
        """route 返回正确解析的 ExpressionBundle。"""
        sink = _make_sink()
        router = ExpressionRouter([sink])
        step_out = _make_step_out(valence=0.6, arousal=0.4)
        bundle = await router.route(step_out)

        assert isinstance(bundle, ExpressionBundle)
        assert bundle.valence_arousal == (pytest.approx(0.6), pytest.approx(0.4))

    async def test_route_calls_each_sink_render(self) -> None:
        """route 调用每个 sink.render 恰好一次。"""
        sink_a = _make_sink()
        sink_b = _make_sink()
        router = ExpressionRouter([sink_a, sink_b])
        step_out = _make_step_out()
        await router.route(step_out)

        sink_a.render.assert_awaited_once()
        sink_b.render.assert_awaited_once()

    async def test_sink_render_receives_bundle_and_policy(self) -> None:
        """sink.render 收到 bundle 与 policy 关键字参数。"""
        sink = _make_sink()
        router = ExpressionRouter([sink], policy=HeadPolicy.VOLUNTARY_ONLY)
        step_out = _make_step_out()
        bundle = await router.route(step_out)

        sink.render.assert_awaited_once_with(bundle, policy=HeadPolicy.VOLUNTARY_ONLY)

    async def test_route_with_direct_expression_dict(self) -> None:
        """直接传 expression 子 dict（无外层 expression 键）同样正确解析。"""
        sink = _make_sink()
        router = ExpressionRouter([sink])
        step_out = _make_step_out(with_expression_wrapper=False)
        bundle = await router.route(step_out)

        assert isinstance(bundle, ExpressionBundle)
        sink.render.assert_awaited_once()

    async def test_route_with_language_field(self) -> None:
        """step_out 含 language 字段时，bundle.language 正确填充。"""
        sink = _make_sink()
        router = ExpressionRouter([sink])
        step_out = _make_step_out(with_language=True)
        bundle = await router.route(step_out)

        assert bundle.language is not None
        assert bundle.language.text == "你好，我很开心。"

    async def test_route_no_sinks_still_returns_bundle(self) -> None:
        """无 sink 时 route 仍返回正确 bundle。"""
        router = ExpressionRouter([])
        step_out = _make_step_out()
        bundle = await router.route(step_out)
        assert isinstance(bundle, ExpressionBundle)

    async def test_route_returns_bundle_for_caller_reuse(self) -> None:
        """route 返回 bundle 供调用方复用（如记录 metrics），不要求重新解析。"""
        sink = _make_sink()
        router = ExpressionRouter([sink])
        step_out = _make_step_out(valence=-0.3, arousal=0.7)
        bundle = await router.route(step_out)

        # 调用方可直接使用 bundle 而无需重解析
        assert bundle.spontaneous.text_label in {"excited", "content", "angry", "sad"}
        assert bundle.voluntary.text_label in {"excited", "content", "angry", "sad"}


# ---------------------------------------------------------------------------
# 2. HeadPolicy 三档行为
# ---------------------------------------------------------------------------


class TestHeadPolicyDispatch:
    """HeadPolicy 三档值传给 sink 的 policy 符合预期。"""

    async def test_voluntary_only_policy_transmitted(self) -> None:
        """VOLUNTARY_ONLY 策略正确传给 sink。"""
        sink = _make_sink()
        router = ExpressionRouter([sink], policy=HeadPolicy.VOLUNTARY_ONLY)
        await router.route(_make_step_out())

        _, kwargs = sink.render.await_args
        assert kwargs["policy"] == HeadPolicy.VOLUNTARY_ONLY

    async def test_spontaneous_only_policy_transmitted(self) -> None:
        """SPONTANEOUS_ONLY 策略正确传给 sink。"""
        sink = _make_sink()
        router = ExpressionRouter([sink], policy=HeadPolicy.SPONTANEOUS_ONLY)
        await router.route(_make_step_out())

        _, kwargs = sink.render.await_args
        assert kwargs["policy"] == HeadPolicy.SPONTANEOUS_ONLY

    async def test_dual_policy_transmitted(self) -> None:
        """DUAL 策略正确传给 sink。"""
        sink = _make_sink()
        router = ExpressionRouter([sink], policy=HeadPolicy.DUAL)
        await router.route(_make_step_out())

        _, kwargs = sink.render.await_args
        assert kwargs["policy"] == HeadPolicy.DUAL

    def test_default_policy_is_voluntary_only(self) -> None:
        """ExpressionRouter 默认策略为 VOLUNTARY_ONLY。"""
        router = ExpressionRouter([])
        assert router.policy == HeadPolicy.VOLUNTARY_ONLY

    def test_head_policy_str_values(self) -> None:
        """HeadPolicy StrEnum 字符串值正确（供序列化/配置使用）。"""
        assert HeadPolicy.VOLUNTARY_ONLY == "voluntary_only"
        assert HeadPolicy.SPONTANEOUS_ONLY == "spontaneous_only"
        assert HeadPolicy.DUAL == "dual"

    async def test_policy_passed_to_all_sinks(self) -> None:
        """多个 sink 都收到相同 policy。"""
        sinks = [_make_sink() for _ in range(3)]
        router = ExpressionRouter(sinks, policy=HeadPolicy.DUAL)
        await router.route(_make_step_out())

        for sink in sinks:
            _, kwargs = sink.render.await_args
            assert kwargs["policy"] == HeadPolicy.DUAL


# ---------------------------------------------------------------------------
# 3. 单 sink 异常不拖垮其他
# ---------------------------------------------------------------------------


class TestExpressionRouterFaultIsolation:
    """单 sink 抛异常时，其余 sink 仍正常调用，不 raise。"""

    async def test_failing_sink_does_not_affect_others(self, caplog: Any) -> None:
        """第一个 sink 抛 RuntimeError，第二个 sink 仍被调用。"""
        failing_sink = MagicMock()
        failing_sink.render = AsyncMock(side_effect=RuntimeError("渲染引擎崩溃"))
        good_sink = _make_sink()

        router = ExpressionRouter([failing_sink, good_sink])

        with caplog.at_level(logging.WARNING, logger="src.mcp.zero.expression_sink"):
            bundle = await router.route(_make_step_out())

        good_sink.render.assert_awaited_once()
        assert isinstance(bundle, ExpressionBundle)
        assert "渲染引擎崩溃" in caplog.text or "render" in caplog.text.lower()

    async def test_all_sinks_fail_still_returns_bundle(self) -> None:
        """所有 sink 都失败时，route 仍返回 bundle，不 raise。"""
        sinks = [MagicMock() for _ in range(3)]
        for s in sinks:
            s.render = AsyncMock(side_effect=RuntimeError("all bad"))

        router = ExpressionRouter(sinks)
        bundle = await router.route(_make_step_out())
        assert isinstance(bundle, ExpressionBundle)

    async def test_middle_sink_fails_others_called(self, caplog: Any) -> None:
        """中间 sink 失败，前后 sink 均被调用。"""
        sink_a = _make_sink()
        failing_sink = MagicMock()
        failing_sink.render = AsyncMock(side_effect=ValueError("中间 sink 错误"))
        sink_b = _make_sink()

        router = ExpressionRouter([sink_a, failing_sink, sink_b])
        with caplog.at_level(logging.WARNING, logger="src.mcp.zero.expression_sink"):
            await router.route(_make_step_out())

        sink_a.render.assert_awaited_once()
        sink_b.render.assert_awaited_once()

    async def test_warning_logged_on_sink_failure(self, caplog: Any) -> None:
        """sink 失败时有 WARNING 级日志，不 raise。"""
        bad_sink = MagicMock()
        bad_sink.render = AsyncMock(side_effect=Exception("unexpected error"))
        router = ExpressionRouter([bad_sink])

        with caplog.at_level(logging.WARNING, logger="src.mcp.zero.expression_sink"):
            await router.route(_make_step_out())

        assert caplog.records, "应有至少一条 warning 日志"
        assert any(r.levelno >= logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# 4. isinstance 协议符合性
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """ExpressionSink / FacsMapper / ProsodyMapper / PhysiologyMapper 协议结构符合性。"""

    def test_mock_sink_satisfies_expression_sink_protocol(self) -> None:
        """具有 render 方法的 Mock 对象满足 ExpressionSink Protocol。"""
        sink = _make_sink()
        assert isinstance(sink, ExpressionSink)

    def test_object_without_render_fails_sink_protocol(self) -> None:
        """无 render 方法的对象不满足 ExpressionSink Protocol。"""

        class NoRender:
            pass

        assert not isinstance(NoRender(), ExpressionSink)

    def test_mock_facs_mapper_satisfies_protocol(self) -> None:
        """具有 async map 方法的 Mock 满足 FacsMapper Protocol。"""
        mapper = MagicMock()
        mapper.map = AsyncMock(return_value={})
        assert isinstance(mapper, FacsMapper)

    def test_mock_prosody_mapper_satisfies_protocol(self) -> None:
        """具有 async map 方法的 Mock 满足 ProsodyMapper Protocol。"""
        mapper = MagicMock()
        mapper.map = AsyncMock(return_value={})
        assert isinstance(mapper, ProsodyMapper)

    def test_mock_physiology_mapper_satisfies_protocol(self) -> None:
        """具有 async map 方法的 Mock 满足 PhysiologyMapper Protocol。"""
        mapper = MagicMock()
        mapper.map = AsyncMock(return_value={})
        assert isinstance(mapper, PhysiologyMapper)

    def test_expression_router_is_concrete_class(self) -> None:
        """ExpressionRouter 是具体类（非 Protocol），可直接实例化。"""
        router = ExpressionRouter([])
        assert isinstance(router, ExpressionRouter)


# ---------------------------------------------------------------------------
# 5. Bundle 字段与双通路 ExpressionHead 正确性
# ---------------------------------------------------------------------------


class TestBundleHeadFields:
    """验证 route 解析后 bundle 的 spontaneous / voluntary 头字段正确。"""

    async def test_spontaneous_and_voluntary_heads_parsed(self) -> None:
        """route 解析后 bundle 的 spontaneous 和 voluntary 都是 ExpressionHead 实例。"""
        sink = _make_sink()
        router = ExpressionRouter([sink])
        step_out = _make_step_out(
            spontaneous_label="angry",
            voluntary_label="content",
        )
        bundle = await router.route(step_out)

        assert isinstance(bundle.spontaneous, ExpressionHead)
        assert isinstance(bundle.voluntary, ExpressionHead)
        assert bundle.spontaneous.text_label == "angry"
        assert bundle.voluntary.text_label == "content"

    async def test_bundle_passed_to_sink_is_same_object(self) -> None:
        """sink.render 收到的 bundle 与 route 返回值是同一个对象。"""
        sink = _make_sink()
        router = ExpressionRouter([sink])
        bundle = await router.route(_make_step_out())

        passed_bundle = sink.render.await_args[0][0]
        assert passed_bundle is bundle
