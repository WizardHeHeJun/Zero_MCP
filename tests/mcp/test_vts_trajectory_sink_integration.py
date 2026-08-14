"""裸参数轨迹通道 sink 挂点集成测试（离线假 VTS server，2026-07-31 二期）。

覆盖：
  1. absolute 接管：挂 ``TrajectoryPlayer`` 喂一段常值轨迹，strength 满后
     注入帧该参数 = 轨迹值；
  2. 非 governed 参数按需注入：轨迹驱动 ``all_params`` 内、``GOVERNED_PARAMS``
     外的自定义参数（夹具加 MouthX）——活跃期注入、播尽+缓出后从注入帧消失；
  3. offset 模式：注入值 = 表情语义静息基线 + 偏移 × strength；
  4. 回放器异常隔离（AD-4 局部防御同思路）：monkeypatch 出的假回放器
     ``apply()`` 恒抛异常——本帧叠加丢弃、``sink.healthy`` 保持 True、
     warning 一次；
  5. 零回归守卫：``sink.trajectory=None`` 时注入帧参数集恒 ==
     ``set(GOVERNED_PARAMS)``。

复用 ``test_vts_expression_sink`` 的 ``FakeVtsServer``/``RANGES`` 手法；本文件
需值域表含一个 GOVERNED 之外的自定义参数（MouthX），故子类化重载
``InputParameterListRequest`` 应答——与 ``test_vts_behavior_sink_integration.py``
的 ``RangesFakeVtsServer`` 同一先例。时间用真实 ``time.monotonic()``（对齐
``test_vts_behavior_sink_integration.py`` 的 sink 集成层惯例，而非
``test_vts_trajectory.py`` 纯回放器单测的注入时间）。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from src.mcp.zero.sinks.trajectory import TrajectoryPlayer
from src.mcp.zero.sinks.vts import GOVERNED_PARAMS, SEMANTIC_REST, VtsExpressionSink
from tests.mcp.test_vts_expression_sink import RANGES, FakeVtsServer

# ---------------------------------------------------------------------------
# 值域表：GOVERNED 之外、all_params 内的自定义参数（MouthX）
# ---------------------------------------------------------------------------

RANGES_WITH_MOUTH_X: dict[str, tuple[float, float, float]] = {
    **RANGES,
    "MouthX": (-1.0, 1.0, 0.0),
}
"""MouthX 既非 GOVERNED_PARAMS 也非 OPTIONAL_OVERLAY_PARAMS——只在
``sink.all_params``（轨迹通道的作用空间）现身，不进 ``sink.ranges``。"""


class TrajectoryFakeVtsServer(FakeVtsServer):
    """值域表可配的假 VTS：只重载 ``InputParameterListRequest`` 应答（同
    ``test_vts_behavior_sink_integration.RangesFakeVtsServer`` 先例）。"""

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


def _install(
    monkeypatch: pytest.MonkeyPatch, fake: TrajectoryFakeVtsServer
) -> TrajectoryFakeVtsServer:
    """把假 VTS 装进 ``vts._connect`` 缝（与仓内既有 fixture 同一手法）。"""
    import src.mcp.zero.sinks.vts as vts_mod

    async def fake_connect(url: str) -> TrajectoryFakeVtsServer:
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
# 1. absolute 接管
# ---------------------------------------------------------------------------


class TestAbsoluteTakeover:
    async def test_constant_trajectory_takes_over_governed_param(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        fake = _install(monkeypatch, TrajectoryFakeVtsServer(dict(RANGES)))
        sink = VtsExpressionSink(token_path=tmp_path / "tok", render_hz=100.0, ambient_motion=False)
        player = TrajectoryPlayer()
        async with sink:
            sink.trajectory = player
            result = player.feed(
                [(0.0, {"MouthSmile": 0.9}), (1.0, {"MouthSmile": 0.9})],
                mode="absolute",
                append=True,
                now=time.monotonic(),
                known_params=frozenset(sink.ranges) | frozenset(sink.all_params),
            )
            assert result.ok
            await asyncio.sleep(0.2)  # 越过 ATTACK_S(0.12s)，strength 已满
        values = [_frame_value(f, "MouthSmile") for f in _inject_frames(fake)]
        assert values, "渲染循环应至少注入一帧"
        assert values[-1] == pytest.approx(0.9, abs=0.02)


# ---------------------------------------------------------------------------
# 2. 非 governed 参数按需注入
# ---------------------------------------------------------------------------


class TestNonGovernedParamInjection:
    async def test_trajectory_drives_non_governed_param_active_then_removed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        fake = _install(monkeypatch, TrajectoryFakeVtsServer(RANGES_WITH_MOUTH_X))
        sink = VtsExpressionSink(token_path=tmp_path / "tok", render_hz=100.0, ambient_motion=False)
        player = TrajectoryPlayer()
        async with sink:
            assert "MouthX" not in sink.ranges  # 非 GOVERNED、非可选叠加白名单
            assert "MouthX" in sink.all_params  # 但在全量参数表内（轨迹作用空间）
            sink.trajectory = player
            await asyncio.sleep(0.03)  # 触发前的帧：不含自定义参数
            pre = len(_inject_frames(fake))
            result = player.feed(
                [(0.0, {"MouthX": 0.5}), (0.2, {"MouthX": 0.5})],
                mode="absolute",
                append=True,
                now=time.monotonic(),
                known_params=frozenset(sink.ranges) | frozenset(sink.all_params),
            )
            assert result.ok
            await asyncio.sleep(0.15)  # 活跃期采样窗（含 attack 结束）
            mid = len(_inject_frames(fake))
            await asyncio.sleep(0.2 + 0.25 + 0.1)  # 越过 0.2s 段长 + RELEASE_S 缓出
            post = len(_inject_frames(fake))
            await asyncio.sleep(0.03)  # 结束后的帧：应已停发
        frames = _inject_frames(fake)
        pre_frames, active_frames, tail_frames = frames[:pre], frames[pre:mid], frames[post:]
        assert pre_frames and all("MouthX" not in _frame_ids(f) for f in pre_frames)
        assert any("MouthX" in _frame_ids(f) for f in active_frames), "活跃期未注入自定义参数"
        active_vals = [
            _frame_value(f, "MouthX") for f in active_frames if "MouthX" in _frame_ids(f)
        ]
        assert any(v > 0.3 for v in active_vals), "注入值应趋近轨迹目标值 0.5"
        assert tail_frames and all("MouthX" not in _frame_ids(f) for f in tail_frames), (
            "release 缓出耗尽后应停发自定义参数（借 VTS 1s lost 回收交还控制权）"
        )


# ---------------------------------------------------------------------------
# 3. offset 模式
# ---------------------------------------------------------------------------


class TestOffsetMode:
    async def test_offset_adds_to_expression_baseline(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        fake = _install(monkeypatch, TrajectoryFakeVtsServer(dict(RANGES)))
        sink = VtsExpressionSink(token_path=tmp_path / "tok", render_hz=100.0, ambient_motion=False)
        player = TrajectoryPlayer()
        async with sink:
            sink.trajectory = player
            result = player.feed(
                [(0.0, {"MouthSmile": 0.2}), (0.5, {"MouthSmile": 0.2})],
                mode="offset",
                append=True,
                now=time.monotonic(),
                known_params=frozenset(sink.ranges) | frozenset(sink.all_params),
            )
            assert result.ok
            await asyncio.sleep(0.2)  # 越过 ATTACK_S，strength 已满
        values = [_frame_value(f, "MouthSmile") for f in _inject_frames(fake)]
        assert values
        assert values[-1] == pytest.approx(SEMANTIC_REST["MouthSmile"] + 0.2, abs=0.02)


# ---------------------------------------------------------------------------
# 4. 回放器故障隔离
# ---------------------------------------------------------------------------


class BrokenTrajectoryPlayer:
    """apply 恒抛的假回放器——验证回放器 bug 只丢本帧叠加、不杀 sink（同
    ``behavior_overlay`` 的 ``BrokenEngine`` 先例，AD-4 局部防御）。"""

    def apply(self, now: float) -> Any:
        raise RuntimeError("boom-trajectory")


class TestTrajectoryFailureIsolation:
    async def test_apply_exception_drops_frame_keeps_sink_healthy_warns_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        fake = _install(monkeypatch, TrajectoryFakeVtsServer(dict(RANGES)))
        sink = VtsExpressionSink(token_path=tmp_path / "tok", render_hz=100.0)
        async with sink:
            with caplog.at_level("WARNING"):
                sink.trajectory = BrokenTrajectoryPlayer()  # type: ignore[assignment]
                await asyncio.sleep(0.05)  # 每帧 apply 都抛——应只告警一次
            assert sink.healthy, "回放器异常不得沿渲染循环广谱兜底杀死 sink"
        frames = _inject_frames(fake)
        assert frames and all(_frame_ids(f) == set(GOVERNED_PARAMS) for f in frames), (
            "回放器叠加丢弃后表情注入应照常（无半应用态）"
        )
        warnings = [r for r in caplog.records if "丢弃本帧叠加" in r.getMessage()]
        assert len(warnings) == 1


# ---------------------------------------------------------------------------
# 5. 零回归守卫
# ---------------------------------------------------------------------------


class TestZeroRegressionGuard:
    async def test_no_trajectory_injects_exactly_governed_params(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """所连部署提供 MouthX，但 ``sink.trajectory`` 保持默认 None——
        注入帧参数集恒 == ``GOVERNED_PARAMS``（现有零回归语义不动）。"""
        fake = _install(monkeypatch, TrajectoryFakeVtsServer(RANGES_WITH_MOUTH_X))
        sink = VtsExpressionSink(token_path=tmp_path / "tok", render_hz=100.0)
        async with sink:
            assert sink.trajectory is None
            await asyncio.sleep(0.05)
        frames = _inject_frames(fake)
        assert frames, "渲染循环应至少注入一帧"
        for frame in frames:
            assert _frame_ids(frame) == set(GOVERNED_PARAMS)


# ---------------------------------------------------------------------------
# 6. 语音口型独占层（speech-play 蓝图 2026-08-14 §T2）
# ---------------------------------------------------------------------------


class TestSpeechMouthExclusiveOverride:
    """``speech_mouth`` 合于 ``trajectory`` **之后**——对其涉及的键有最终话语权
    （AD-5「最后应用者赢」的结构性独占语义），非涉及键不受影响。"""

    async def test_speech_mouth_wins_over_trajectory_and_behavior_overlay_on_shared_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        from src.agents.models.vts_behavior import BehaviorRequest  # noqa: PLC0415
        from src.mcp.zero.sinks.behavior_overlay import BehaviorOverlayEngine  # noqa: PLC0415

        fake = _install(monkeypatch, TrajectoryFakeVtsServer(dict(RANGES)))
        sink = VtsExpressionSink(token_path=tmp_path / "tok", render_hz=100.0, ambient_motion=False)
        engine = BehaviorOverlayEngine()
        trajectory = TrajectoryPlayer()
        speech_mouth = TrajectoryPlayer()
        async with sink:
            sink.behavior_overlay = engine
            sink.trajectory = trajectory
            sink.speech_mouth = speech_mouth
            known = frozenset(sink.ranges) | frozenset(sink.all_params)
            receipt = engine.trigger(
                BehaviorRequest(name="nod"), now=time.monotonic(), ranges=sink.ranges
            )
            assert receipt.status == "accepted"
            trajectory.feed(
                [(0.0, {"MouthSmile": 0.2}), (1.0, {"MouthSmile": 0.2})],
                mode="absolute",
                append=True,
                now=time.monotonic(),
                known_params=known,
            )
            speech_mouth.feed(
                [(0.0, {"MouthSmile": 0.9}), (1.0, {"MouthSmile": 0.9})],
                mode="absolute",
                append=False,
                now=time.monotonic(),
                known_params=known,
            )
            await asyncio.sleep(0.2)  # 越过各自 ATTACK_S(0.12s)，strength 均已满
        frames = _inject_frames(fake)
        mouth_values = [_frame_value(f, "MouthSmile") for f in frames]
        face_y_values = [_frame_value(f, "FaceAngleY") for f in frames]
        assert mouth_values
        assert mouth_values[-1] == pytest.approx(0.9, abs=0.02), (
            "共享键 MouthSmile 应由 speech_mouth 最终覆盖（trajectory=0.2 被压过）"
        )
        assert any(v < -0.5 for v in face_y_values), (
            "非嘴键 FaceAngleY 不受 speech_mouth 影响，behavior_overlay 的手势偏移仍在"
        )

    async def test_non_mouth_key_untouched_by_speech_mouth(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """speech_mouth 只声明 MouthOpen 键——trajectory 驱动的 Brows 不受牵连。"""
        fake = _install(monkeypatch, TrajectoryFakeVtsServer(dict(RANGES)))
        sink = VtsExpressionSink(token_path=tmp_path / "tok", render_hz=100.0, ambient_motion=False)
        trajectory = TrajectoryPlayer()
        speech_mouth = TrajectoryPlayer()
        async with sink:
            sink.trajectory = trajectory
            sink.speech_mouth = speech_mouth
            known = frozenset(sink.ranges) | frozenset(sink.all_params)
            trajectory.feed(
                [(0.0, {"Brows": 0.8}), (1.0, {"Brows": 0.8})],
                mode="absolute",
                append=True,
                now=time.monotonic(),
                known_params=known,
            )
            speech_mouth.feed(
                [(0.0, {"MouthOpen": 0.6}), (1.0, {"MouthOpen": 0.6})],
                mode="absolute",
                append=False,
                now=time.monotonic(),
                known_params=known,
            )
            await asyncio.sleep(0.2)
        frames = _inject_frames(fake)
        brows_values = [_frame_value(f, "Brows") for f in frames]
        mouth_open_values = [_frame_value(f, "MouthOpen") for f in frames]
        assert brows_values[-1] == pytest.approx(0.8, abs=0.02), (
            "speech_mouth 未声明 Brows——trajectory 对它仍有最终话语权"
        )
        assert mouth_open_values[-1] == pytest.approx(0.6, abs=0.02)


class TestSpeechMouthNoneRegression:
    async def test_speech_mouth_none_trajectory_still_has_final_say_on_shared_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """正对照：``sink.speech_mouth`` 保持默认 None 时，trajectory 对共享键
        仍有最终话语权——本次改动新增覆盖层后，未使用该层的既有路径逐帧行为
        不变（零回归的显式落点，非仅靠"未测到"佐证）。"""
        fake = _install(monkeypatch, TrajectoryFakeVtsServer(dict(RANGES)))
        sink = VtsExpressionSink(token_path=tmp_path / "tok", render_hz=100.0, ambient_motion=False)
        trajectory = TrajectoryPlayer()
        async with sink:
            assert sink.speech_mouth is None
            sink.trajectory = trajectory
            trajectory.feed(
                [(0.0, {"MouthSmile": 0.2}), (1.0, {"MouthSmile": 0.2})],
                mode="absolute",
                append=True,
                now=time.monotonic(),
                known_params=frozenset(sink.ranges) | frozenset(sink.all_params),
            )
            await asyncio.sleep(0.2)
        values = [_frame_value(f, "MouthSmile") for f in _inject_frames(fake)]
        assert values
        assert values[-1] == pytest.approx(0.2, abs=0.02)


class BrokenSpeechMouth:
    """apply 恒抛的假回放器——验证 speech_mouth 层 bug 只丢本帧叠加、不杀 sink
    （同 ``BrokenTrajectoryPlayer``/``behavior_overlay.BrokenEngine`` 先例）。"""

    def apply(self, now: float) -> Any:
        raise RuntimeError("boom-speech-mouth")


class TestSpeechMouthFailureIsolation:
    async def test_apply_exception_drops_frame_keeps_sink_healthy_warns_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        fake = _install(monkeypatch, TrajectoryFakeVtsServer(dict(RANGES)))
        sink = VtsExpressionSink(token_path=tmp_path / "tok", render_hz=100.0)
        async with sink:
            with caplog.at_level("WARNING"):
                sink.speech_mouth = BrokenSpeechMouth()  # type: ignore[assignment]
                await asyncio.sleep(0.05)  # 每帧 apply 都抛——应只告警一次
            assert sink.healthy, "speech_mouth 异常不得沿渲染循环广谱兜底杀死 sink"
        frames = _inject_frames(fake)
        assert frames and all(_frame_ids(f) == set(GOVERNED_PARAMS) for f in frames), (
            "speech_mouth 叠加丢弃后表情注入应照常（无半应用态）"
        )
        warnings = [r for r in caplog.records if "丢弃本帧叠加" in r.getMessage()]
        assert len(warnings) == 1
