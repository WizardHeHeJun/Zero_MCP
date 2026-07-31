"""VTS 行为叠加层 sink 挂点集成测试（离线假 VTS server——蓝图 2026-07-31 §8.2 · T8）。

覆盖：
  1. 零回归守卫：``behavior_overlay=None`` 时注入帧参数集恒 == set(GOVERNED_PARAMS)
     ——即便所连部署提供全部可选参数，无手势也不注入（现有用例语义不动，AD-5）。
  2. 挂 engine 后短跑：注入帧含手势偏移（nod → FaceAngleY 负向）；可选参数
     （BodyAngleY）仅活跃期注入，release 归零、包络剪除后停发（AD-5 交还语义）。
  3. FakeVtsServer RANGES 含/不含 BodyAngle 两态：缺席 warning 一次不 raise、
     ``__aenter__`` 成功（优雅回退纪律）；在场全收进 ``ranges`` 且无告警。
  4. 引擎 apply 抛异常：本帧 overlay 丢弃、``sink.healthy`` 保持 True、warning
     一次（AD-4 局部防御——引擎 bug 不得杀死整条表情通道）。

复用 ``test_vts_expression_sink`` 的 FakeVtsServer/fake_vts 手法；其 fixture 绑死
基础 RANGES，本文件需值域表可配（含/不含可选参数两态），故以子类只重载
``InputParameterListRequest`` 应答、自装 ``_connect`` 缝（与原 fixture 同一手法）。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import pytest

from src.agents.models.vts_behavior import BehaviorRequest
from src.mcp.zero.sinks.behavior_overlay import BehaviorOverlayEngine
from src.mcp.zero.sinks.vts import (
    GOVERNED_PARAMS,
    OPTIONAL_OVERLAY_PARAMS,
    VtsExpressionSink,
)
from tests.mcp.test_vts_expression_sink import RANGES, FakeVtsServer

# ---------------------------------------------------------------------------
# 值域表两态 + ranges 可配的假 VTS
# ---------------------------------------------------------------------------

FULL_OPTIONAL_RANGES: dict[str, tuple[float, float, float]] = {
    **RANGES,
    "BodyAngleX": (-30.0, 30.0, 0.0),
    "BodyAngleY": (-30.0, 30.0, 0.0),
    "BodyAngleZ": (-30.0, 30.0, 0.0),
    "EyeLeftX": (-1.0, 1.0, 0.0),
    "EyeLeftY": (-1.0, 1.0, 0.0),
    "EyeRightX": (-1.0, 1.0, 0.0),
    "EyeRightY": (-1.0, 1.0, 0.0),
}
"""含 BodyAngle 态：基础 RANGES + 全部 7 个可选叠加参数（量程为工程占位值）。"""


class RangesFakeVtsServer(FakeVtsServer):
    """值域表可配的假 VTS：只重载 ``InputParameterListRequest`` 应答，其余沿用父类。"""

    def __init__(self, ranges: dict[str, tuple[float, float, float]]) -> None:
        super().__init__()
        self.ranges = ranges

    def _respond(self, req: dict[str, Any]) -> dict[str, Any]:
        if req["messageType"] == "InputParameterListRequest":
            return {
                "requestID": req["requestID"],
                "messageType": "InputParameterListResponse",
                "data": {
                    "defaultParameters": [
                        {"name": n, "min": lo, "max": hi, "defaultValue": rest}
                        for n, (lo, hi, rest) in self.ranges.items()
                    ]
                },
            }
        return super()._respond(req)


def _install(monkeypatch: pytest.MonkeyPatch, fake: RangesFakeVtsServer) -> RangesFakeVtsServer:
    """把假 VTS 装进 ``vts._connect`` 缝（与 fake_vts fixture 同一手法）。"""
    import src.mcp.zero.sinks.vts as vts_mod

    async def fake_connect(url: str) -> RangesFakeVtsServer:
        return fake

    monkeypatch.setattr(vts_mod, "_connect", fake_connect)
    return fake


def _inject_frames(fake: FakeVtsServer) -> list[dict[str, Any]]:
    return [m for m in fake.sent if m["messageType"] == "InjectParameterDataRequest"]


def _frame_ids(frame: dict[str, Any]) -> set[str]:
    return {p["id"] for p in frame["data"]["parameterValues"]}


def _frame_value(frame: dict[str, Any], param: str) -> float:
    return next(p["value"] for p in frame["data"]["parameterValues"] if p["id"] == param)


# ---------------------------------------------------------------------------
# 1. 零回归守卫（AD-5：无手势时注入集恒等于 GOVERNED_PARAMS）
# ---------------------------------------------------------------------------


class TestZeroRegressionGuard:
    async def test_overlay_none_injects_exactly_governed_params(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """部署提供全部可选参数、但未挂 engine——注入帧参数集恒 == GOVERNED_PARAMS。"""
        fake = _install(monkeypatch, RangesFakeVtsServer(FULL_OPTIONAL_RANGES))
        sink = VtsExpressionSink(token_path=tmp_path / "tok", render_hz=100.0)
        async with sink:
            assert sink.behavior_overlay is None  # 默认无手势叠加
            await asyncio.sleep(0.05)
        frames = _inject_frames(fake)
        assert frames, "渲染循环应至少注入一帧"
        for frame in frames:
            assert _frame_ids(frame) == set(GOVERNED_PARAMS)


# ---------------------------------------------------------------------------
# 2. 挂 engine 后短跑：手势偏移进帧 / 可选参数仅活跃期注入
# ---------------------------------------------------------------------------


class TestOverlayInjection:
    async def test_gesture_offset_reaches_inject_frames(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """nod 触发后，注入帧的 FaceAngleY 出现负向（低头）偏移。

        关 ambient_motion 排除呼吸/OU 噪声——无手势时 FaceAngleY 恒为静息 0，
        帧内非零负值只能来自 overlay 挂点。
        """
        fake = _install(monkeypatch, RangesFakeVtsServer(FULL_OPTIONAL_RANGES))
        sink = VtsExpressionSink(token_path=tmp_path / "tok", render_hz=100.0, ambient_motion=False)
        engine = BehaviorOverlayEngine()
        async with sink:
            sink.behavior_overlay = engine
            receipt = engine.trigger(
                BehaviorRequest(name="nod"), now=time.monotonic(), ranges=sink.ranges
            )
            assert receipt.status == "accepted"
            await asyncio.sleep(0.1)  # 700ms 包络的前段，stroke 已展开
        values = [_frame_value(f, "FaceAngleY") for f in _inject_frames(fake)]
        assert any(v < -0.5 for v in values), "手势偏移未进注入帧"

    async def test_optional_param_injected_only_while_active_then_stops(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """lean_in（BodyAngleY 在场，不降级）：可选参数仅活跃期注入，包络结束停发。"""
        fake = _install(monkeypatch, RangesFakeVtsServer(FULL_OPTIONAL_RANGES))
        sink = VtsExpressionSink(token_path=tmp_path / "tok", render_hz=100.0, ambient_motion=False)
        engine = BehaviorOverlayEngine()
        async with sink:
            sink.behavior_overlay = engine
            await asyncio.sleep(0.03)  # 触发前的帧：不含可选参数
            pre = len(_inject_frames(fake))
            receipt = engine.trigger(
                BehaviorRequest(name="lean_in", duration_ms=400),
                now=time.monotonic(),
                ranges=sink.ranges,
            )
            assert receipt.status == "accepted"
            assert receipt.degraded_channels == []  # BodyAngleY 在场，无降级
            await asyncio.sleep(0.15)  # 活跃期采样窗
            mid = len(_inject_frames(fake))
            await asyncio.sleep(0.5)  # 越过 400ms 总长——包络已剪除
            post = len(_inject_frames(fake))
            await asyncio.sleep(0.05)  # 结束后的帧：应已停发可选参数
        frames = _inject_frames(fake)
        pre_frames, active_frames, tail_frames = frames[:pre], frames[pre:mid], frames[post:]
        assert pre_frames and all(_frame_ids(f) == set(GOVERNED_PARAMS) for f in pre_frames)
        assert any("BodyAngleY" in _frame_ids(f) for f in active_frames), "活跃期未注入可选参数"
        active_vals = [
            _frame_value(f, "BodyAngleY") for f in active_frames if "BodyAngleY" in _frame_ids(f)
        ]
        assert any(v < -1.0 for v in active_vals), "可选参数注入值应含前倾偏移（非 defaultValue）"
        assert tail_frames and all(_frame_ids(f) == set(GOVERNED_PARAMS) for f in tail_frames), (
            "release 归零后应停发可选参数（借 VTS 1s lost 回收交还控制权）"
        )


# ---------------------------------------------------------------------------
# 3. 可选参数值域两态：含 / 不含 BodyAngle
# ---------------------------------------------------------------------------


class TestOptionalRangesStates:
    async def test_full_optional_ranges_collected_without_warning(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        _install(monkeypatch, RangesFakeVtsServer(FULL_OPTIONAL_RANGES))
        sink = VtsExpressionSink(token_path=tmp_path / "tok", render_hz=100.0)
        with caplog.at_level("WARNING"):
            async with sink:
                assert sink.healthy
        assert sink.unavailable_params == []
        for param in OPTIONAL_OVERLAY_PARAMS:
            assert param in sink.ranges
        assert not [r for r in caplog.records if "缺可选叠加参数" in r.getMessage()]

    async def test_missing_optional_warns_once_when_behavior_overlay_active(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """基础 RANGES（无 BodyAngle/眼球参数）+ 已挂 ``behavior_overlay``
        （行为层用户路径，对齐 ``BehaviorService.connect()`` 的挂载顺序）：
        WARNING 一次、不 raise、注入照常（INFO2：两态之一，见同类
        ``test_missing_optional_downgrades_to_debug_without_behavior_overlay``）。
        """
        fake = _install(monkeypatch, RangesFakeVtsServer(dict(RANGES)))
        sink = VtsExpressionSink(token_path=tmp_path / "tok", render_hz=100.0)
        sink.behavior_overlay = BehaviorOverlayEngine()  # 模拟已挂行为层
        with caplog.at_level("DEBUG"):
            async with sink:
                assert sink.healthy  # __aenter__ 成功——可选参数缺席不拦启动
                await asyncio.sleep(0.03)
        assert sink.unavailable_params == list(OPTIONAL_OVERLAY_PARAMS)
        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "缺可选叠加参数" in r.getMessage()
        ]
        assert len(warnings) == 1
        frames = _inject_frames(fake)
        assert frames and all(_frame_ids(f) == set(GOVERNED_PARAMS) for f in frames)

    async def test_missing_optional_downgrades_to_debug_without_behavior_overlay(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """基础 RANGES + 未挂 ``behavior_overlay``（纯表情通路用户，未启用
        行为层）：缺可选参数对该用户是无关噪音，降为 DEBUG——不应在默认
        WARNING 级可见（INFO2）。"""
        fake = _install(monkeypatch, RangesFakeVtsServer(dict(RANGES)))
        sink = VtsExpressionSink(token_path=tmp_path / "tok", render_hz=100.0)
        assert sink.behavior_overlay is None
        with caplog.at_level("DEBUG"):
            async with sink:
                assert sink.healthy
                await asyncio.sleep(0.03)
        assert sink.unavailable_params == list(OPTIONAL_OVERLAY_PARAMS)
        assert not [
            r for r in caplog.records if r.levelno >= logging.WARNING and "缺可选" in r.getMessage()
        ]
        debug_records = [
            r
            for r in caplog.records
            if r.levelno == logging.DEBUG and "缺可选参数" in r.getMessage()
        ]
        assert len(debug_records) == 1
        frames = _inject_frames(fake)
        assert frames and all(_frame_ids(f) == set(GOVERNED_PARAMS) for f in frames)


# ---------------------------------------------------------------------------
# 4. 引擎故障隔离（AD-4 局部防御）
# ---------------------------------------------------------------------------


class BrokenEngine:
    """apply 恒抛的假引擎——验证引擎 bug 只丢本帧 overlay、不杀 sink。"""

    def apply(self, now: float) -> Any:
        raise RuntimeError("boom")


class TestEngineFailureIsolation:
    async def test_apply_exception_drops_overlay_keeps_sink_healthy_warns_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        fake = _install(monkeypatch, RangesFakeVtsServer(FULL_OPTIONAL_RANGES))
        sink = VtsExpressionSink(token_path=tmp_path / "tok", render_hz=100.0)
        async with sink:
            with caplog.at_level("WARNING"):
                sink.behavior_overlay = BrokenEngine()  # type: ignore[assignment]
                await asyncio.sleep(0.05)  # 每帧 apply 都抛——应只告警一次
            assert sink.healthy, "引擎异常不得沿渲染循环广谱兜底杀死 sink"
        frames = _inject_frames(fake)
        assert frames and all(_frame_ids(f) == set(GOVERNED_PARAMS) for f in frames), (
            "overlay 丢弃后表情注入应照常（无半应用态）"
        )
        warnings = [r for r in caplog.records if "丢弃本帧叠加" in r.getMessage()]
        assert len(warnings) == 1
