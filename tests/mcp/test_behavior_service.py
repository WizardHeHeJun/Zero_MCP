"""BehaviorService 业务层测试（code-review W3：service 层测试覆盖缺口）。

复用 ``tests.mcp.test_vts_expression_sink`` 的 ``FakeVtsServer``/``RANGES`` 手法
（子类化补齐 ``CurrentModelRequest``/``HotkeysInCurrentModelRequest``/
``HotkeyTriggerRequest`` 应答，与 ``test_vts_behavior_sink_integration.py`` 的
``RangesFakeVtsServer`` 同一先例）。

覆盖：
  1. connect/disconnect 幂等（双次 connect 只走一次真实 ``__aenter__``）；
  2. 并发双 connect（W1 lifecycle_lock）——先证能红纪律，见
     ``TestConcurrentConnect`` docstring；
  3. 热键触发三态：accepted / 我方 5s 冷却预拦 ``[vtsb:cooldown]`` /
     VTS APIError errorID 映射（202→hotkey_unavailable，表外码→透传
     VtsApiError 供 server 映射 ``[vtsb:vts_error]``）；
  4. ``VTS_BEHAVIOR_HOTKEYS=false``：不枚举、``hotkey:`` 触发回
     ``[vtsb:hotkey_unavailable]``；
  5. ``_read_model_id`` 两级回退（``CurrentModelRequest`` 成功/失败）；
  6. W2/W5 口径一致性回归：同一份 ranges 下 catalog 判定与
     ``engine.trigger`` 每个 direction 的实际回执一致（审查员复现场景）；
  7. INFO2 端到端核验：``BehaviorService.connect()`` 内 behavior_overlay 先于
     ``__aenter__`` 挂上，缺可选参数告警应为 WARNING（非 DEBUG）。
  8. 裸参数轨迹通道面（2026-07-31 二期）：``animate``/``clear_params``/
     ``list_params``/``status`` 的未连接协议性拒绝、回执透传、幂等、全参数表
     governed 标注、trajectory_active/remaining 聚合。
  9. 语音播放面（speech-play 蓝图 2026-08-14 §T4）：未连接协议性拒绝、
     ``status`` 三新字段默认值、``speech_playback`` 首调前懒加载零回归、
     成功路径回执透传。

不标 ``zerorepo``（离线假 ws，无关 D:\\Zero）；无未标记的 skip。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from src.agents.models.vts_behavior import (
    VTSB_COOLDOWN,
    VTSB_HOTKEY_UNAVAILABLE,
    VTSB_NOT_CONNECTED,
    BehaviorRequest,
    SpeechRequest,
    TrajectoryKeyframe,
    TrajectoryRequest,
    extract_vtsb_code,
)
from src.mcp.behavior.service import BehaviorService, _behavior_info
from src.mcp.zero.sinks.behavior_overlay import VOCABULARY, BehaviorOverlayEngine, Ranges
from src.mcp.zero.sinks.vts import GOVERNED_PARAMS, VtsApiError
from tests.mcp.test_vts_expression_sink import RANGES, FakeVtsServer

# ---------------------------------------------------------------------------
# 假 VTS：补齐 BehaviorService 依赖的三个消息类型
# ---------------------------------------------------------------------------


class BehaviorFakeVtsServer(FakeVtsServer):
    """扩展 ``FakeVtsServer``：补齐 ``CurrentModelRequest`` /
    ``HotkeysInCurrentModelRequest`` / ``HotkeyTriggerRequest`` 应答。

    ``recv()`` 强制一次 ``await asyncio.sleep(0)``——父类 ``send()``/``recv()``
    全程同步完成（响应在 ``send()`` 时已同步入队，``recv()`` 若队列非空则不
    经过任何真实挂起点），若不手工补一个真实挂起点，两个通过
    ``asyncio.gather`` 并发发起的 ``connect()`` 协程在假 server 上永远不会
    真正交错（真实 WebSocket I/O 总会让出事件循环）——``TestConcurrentConnect``
    正是靠这个强制让出点获得判别力，手法同
    ``test_vts_expression_sink.py::TestVtsApiClientConcurrency`` 的对抗式响应器。
    """

    def __init__(
        self,
        ranges: dict[str, tuple[float, float, float]] | None = None,
        *,
        hotkeys: list[dict[str, Any]] | None = None,
        model_id: str = "fake-model-id",
        model_loaded: bool = True,
    ) -> None:
        super().__init__()
        self.ranges = dict(ranges) if ranges is not None else dict(RANGES)
        self.hotkeys = hotkeys if hotkeys is not None else []
        self.model_id = model_id
        self.model_loaded = model_loaded
        self.current_model_error = False
        self.hotkey_trigger_error_id: int | None = None

    async def recv(self) -> str:
        await asyncio.sleep(0)  # 强制真实挂起点（见类 docstring）
        return await super().recv()

    def _respond(self, req: dict[str, Any]) -> dict[str, Any]:
        mt, rid = req["messageType"], req["requestID"]
        if mt == "InputParameterListRequest":
            return {
                "requestID": rid,
                "messageType": "InputParameterListResponse",
                "data": {
                    "defaultParameters": [
                        {"name": n, "min": lo, "max": hi, "defaultValue": rest}
                        for n, (lo, hi, rest) in self.ranges.items()
                    ]
                },
            }
        if mt == "CurrentModelRequest":
            if self.current_model_error:
                return {"requestID": rid, "messageType": "APIError", "data": {"errorID": 8}}
            return {
                "requestID": rid,
                "messageType": "CurrentModelResponse",
                "data": {"modelLoaded": self.model_loaded, "modelID": self.model_id},
            }
        if mt == "AvailableModelsRequest":
            return {
                "requestID": rid,
                "messageType": "AvailableModelsResponse",
                "data": {
                    "availableModels": [
                        {
                            "modelID": self.model_id,
                            "modelName": "Hiyori_A",
                            "modelLoaded": self.model_loaded,
                        }
                    ]
                },
            }
        if mt == "HotkeysInCurrentModelRequest":
            return {
                "requestID": rid,
                "messageType": "HotkeysInCurrentModelResponse",
                "data": {"availableHotkeys": self.hotkeys},
            }
        if mt == "HotkeyTriggerRequest":
            if self.hotkey_trigger_error_id is not None:
                return {
                    "requestID": rid,
                    "messageType": "APIError",
                    "data": {"errorID": self.hotkey_trigger_error_id},
                }
            return {
                "requestID": rid,
                "messageType": "HotkeyTriggerResponse",
                "data": {"hotkeyID": req["data"]["hotkeyID"]},
            }
        return super()._respond(req)


def _install(monkeypatch: pytest.MonkeyPatch, fake: BehaviorFakeVtsServer) -> BehaviorFakeVtsServer:
    """把假 VTS 装进 ``vts._connect`` 缝（与仓内既有 fixture 同一手法）。"""
    import src.mcp.zero.sinks.vts as vts_mod

    async def fake_connect(url: str) -> BehaviorFakeVtsServer:
        return fake

    monkeypatch.setattr(vts_mod, "_connect", fake_connect)
    return fake


_WAVE_HOTKEY = {
    "hotkeyID": "hk-wave",
    "name": "Wave",
    "type": "TriggerAnimation",
    "file": "wave.motion3.json",
}


@pytest.fixture(autouse=True)
def _isolated_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """隔离 token 落盘路径（standalone BehaviorService 经 ``kwargs_from_env()``
    读 ``VTS_TOKEN_FILE`` 自建 sink）——避免污染仓内 ``.vts_token``。"""
    monkeypatch.setenv("VTS_TOKEN_FILE", str(tmp_path / "tok"))
    monkeypatch.setenv("VTS_SINK_AMBIENT_MOTION", "false")


# ---------------------------------------------------------------------------
# 1. connect/disconnect 幂等
# ---------------------------------------------------------------------------


class TestConnectDisconnectIdempotent:
    async def test_double_connect_reuses_single_sink(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _install(monkeypatch, BehaviorFakeVtsServer(hotkeys=[_WAVE_HOTKEY]))
        service = BehaviorService()
        first = await service.connect()
        second = await service.connect()
        try:
            assert first.connected is True
            assert second == first
            auth_requests = [m for m in fake.sent if m["messageType"] == "AuthenticationRequest"]
            assert len(auth_requests) == 1  # 第二次 connect 是幂等 no-op
        finally:
            await service.disconnect()

    async def test_double_disconnect_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, BehaviorFakeVtsServer())
        service = BehaviorService()
        await service.connect()
        first = await service.disconnect()
        second = await service.disconnect()
        assert first.connected is False
        assert second.connected is False


# ---------------------------------------------------------------------------
# 2. 并发双 connect（W1 lifecycle_lock）
# ---------------------------------------------------------------------------


class TestConcurrentConnect:
    """并发双 ``connect()`` 在 ``lifecycle_lock`` 保护下只产生一次真实连接建立
    （W1）——修复前的失败形态：两次 ``connect()`` 在同一 sink 上并发进入
    ``__aenter__``，各自的认证握手都真的发生，后完成者覆盖先完成者持有的
    ``render_task`` 句柄（永不被 ``__aexit__`` 等待/取消，泄漏）。

    **先证能红纪律**：已手工验证——临时把 ``BehaviorService.connect()`` 里的
    ``async with self.lifecycle_lock:`` 换成不做任何事的 no-op 上下文管理器
    （等效去锁）后，本用例的 ``AuthenticationRequest`` 计数从 1 变为 2（两次
    并发 ``__aenter__`` 都真的发起了认证握手）——判别力确认该用例在无锁下
    确实能抓到交错；恢复锁后计数回到 1，用例转绿。
    """

    async def test_concurrent_double_connect_authenticates_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _install(monkeypatch, BehaviorFakeVtsServer())
        service = BehaviorService()
        first, second = await asyncio.gather(service.connect(), service.connect())
        assert first.connected is True
        assert second.connected is True
        auth_requests = [m for m in fake.sent if m["messageType"] == "AuthenticationRequest"]
        assert len(auth_requests) == 1, (
            "lifecycle_lock 应把并发 connect() 串行化——后完成者应落到"
            "「已连接且健康」的早退分支，不应重新走 __aenter__ 认证握手"
        )
        current_model_requests = [m for m in fake.sent if m["messageType"] == "CurrentModelRequest"]
        assert len(current_model_requests) == 1
        assert service.sink is not None
        render_task = service.sink.render_task
        assert render_task is not None and not render_task.done()
        status = await service.disconnect()
        assert status.connected is False
        assert render_task.done(), "disconnect() 应等待/收尾了这唯一的 render_task，无遗留句柄"


# ---------------------------------------------------------------------------
# 3. 热键触发三态
# ---------------------------------------------------------------------------


class TestHotkeyTrigger:
    async def test_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _install(monkeypatch, BehaviorFakeVtsServer(hotkeys=[_WAVE_HOTKEY]))
        service = BehaviorService()
        await service.connect()
        try:
            receipt = await service.trigger(BehaviorRequest(name="hotkey:hk-wave"))
            assert receipt.status == "accepted"
            assert receipt.code is None
            trigger_requests = [m for m in fake.sent if m["messageType"] == "HotkeyTriggerRequest"]
            assert len(trigger_requests) == 1
            assert trigger_requests[0]["data"]["hotkeyID"] == "hk-wave"
        finally:
            await service.disconnect()

    async def test_our_precooldown_rejects_second_trigger(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """我方 5s 保守冷却预拦：紧随的第二次同热键触发被拒，不撞 VTS 侧。"""
        fake = _install(monkeypatch, BehaviorFakeVtsServer(hotkeys=[_WAVE_HOTKEY]))
        service = BehaviorService()
        await service.connect()
        try:
            first = await service.trigger(BehaviorRequest(name="hotkey:hk-wave"))
            assert first.status == "accepted"
            second = await service.trigger(BehaviorRequest(name="hotkey:hk-wave"))
            assert second.status == "rejected"
            assert second.code == VTSB_COOLDOWN
            trigger_requests = [m for m in fake.sent if m["messageType"] == "HotkeyTriggerRequest"]
            assert len(trigger_requests) == 1  # 第二次被预拦，未转发给 VTS
        finally:
            await service.disconnect()

    async def test_vts_202_id_not_found_maps_to_hotkey_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _install(monkeypatch, BehaviorFakeVtsServer(hotkeys=[_WAVE_HOTKEY]))
        fake.hotkey_trigger_error_id = 202
        service = BehaviorService()
        await service.connect()
        try:
            receipt = await service.trigger(BehaviorRequest(name="hotkey:hk-wave"))
            assert receipt.status == "rejected"
            assert receipt.code == VTSB_HOTKEY_UNAVAILABLE
        finally:
            await service.disconnect()

    async def test_out_of_table_error_code_reraises_as_vts_api_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """表外错误码（如 204 DataInvalid）是协议性失败：``VtsApiError`` 透传，
        由 server 映射 ``[vtsb:vts_error]``——不该被我方码表悄悄吞掉。"""
        fake = _install(monkeypatch, BehaviorFakeVtsServer(hotkeys=[_WAVE_HOTKEY]))
        fake.hotkey_trigger_error_id = 204
        service = BehaviorService()
        await service.connect()
        try:
            with pytest.raises(VtsApiError):
                await service.trigger(BehaviorRequest(name="hotkey:hk-wave"))
        finally:
            await service.disconnect()


# ---------------------------------------------------------------------------
# 4. VTS_BEHAVIOR_HOTKEYS=false
# ---------------------------------------------------------------------------


class TestHotkeysDisabled:
    async def test_disabled_skips_enumeration_and_rejects_trigger(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VTS_BEHAVIOR_HOTKEYS", "false")
        fake = _install(monkeypatch, BehaviorFakeVtsServer(hotkeys=[_WAVE_HOTKEY]))
        service = BehaviorService()
        status = await service.connect()
        try:
            assert status.hotkey_count == 0
            enum_requests = [
                m for m in fake.sent if m["messageType"] == "HotkeysInCurrentModelRequest"
            ]
            assert enum_requests == []  # 关闭时不枚举
            receipt = await service.trigger(BehaviorRequest(name="hotkey:hk-wave"))
            assert receipt.status == "rejected"
            assert receipt.code == VTSB_HOTKEY_UNAVAILABLE
        finally:
            await service.disconnect()


# ---------------------------------------------------------------------------
# 5. _read_model_id 两级回退
# ---------------------------------------------------------------------------


class TestReadModelIdFallback:
    async def test_current_model_request_success_used_directly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _install(monkeypatch, BehaviorFakeVtsServer(model_id="model-a"))
        service = BehaviorService()
        status = await service.connect()
        try:
            assert status.model_id == "model-a"
            current_requests = [m for m in fake.sent if m["messageType"] == "CurrentModelRequest"]
            fallback_requests = [
                m for m in fake.sent if m["messageType"] == "AvailableModelsRequest"
            ]
            assert len(current_requests) == 1
            assert fallback_requests == []  # 首选成功，不走回退
        finally:
            await service.disconnect()

    async def test_current_model_request_failure_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _install(monkeypatch, BehaviorFakeVtsServer(model_id="model-b"))
        fake.current_model_error = True
        service = BehaviorService()
        status = await service.connect()
        try:
            assert status.model_id == "model-b"
            fallback_requests = [
                m for m in fake.sent if m["messageType"] == "AvailableModelsRequest"
            ]
            assert len(fallback_requests) == 1
        finally:
            await service.disconnect()


# ---------------------------------------------------------------------------
# 6. W2/W5 口径一致性回归（审查员复现场景）
# ---------------------------------------------------------------------------

RANGES_ONLY_EYE_X: Ranges = {
    **RANGES,
    "EyeLeftX": (-1.0, 1.0, 0.0),
    "EyeRightX": (-1.0, 1.0, 0.0),
}
"""审查员复现场景：ranges 只提供 EyeLeftX/EyeRightX（无 Y 轴眼球参数）——修复前
catalog 对 glance(direction=left) 误报 degraded，而引擎实际 accepted 无降级。"""


class TestCatalogEngineCalibrationConsistency:
    """同一份 ranges 下，catalog（`_behavior_info`）判定与
    ``BehaviorOverlayEngine.trigger`` 每个合法 direction 的实际回执一致。"""

    def test_glance_catalog_matches_engine_per_direction(self) -> None:
        spec = VOCABULARY["glance"]
        info = _behavior_info(spec, RANGES_ONLY_EYE_X)
        engine = BehaviorOverlayEngine()
        for i, direction in enumerate(spec.directions or ()):
            receipt = engine.trigger(
                BehaviorRequest(name="glance", direction=direction),
                now=i * 10.0,  # 每次间隔 10s：跳出冷却（1.0s）与全局节流（0.25s）
                ranges=RANGES_ONLY_EYE_X,
            )
            assert receipt.status == "accepted", f"direction={direction}"
            engine_degraded = bool(receipt.degraded_channels)
            if direction in ("left", "right"):
                assert engine_degraded is False, f"{direction}: X 轴参数在场，引擎应无降级"
            else:
                assert engine_degraded is True, f"{direction}: 缺 Y 轴眼球参数，引擎应降级"
        # catalog：available 恒 True（每个 direction 都有路可走——非降级即降级）；
        # degraded 文本须点名哪些 direction 降级、哪些完整（W2 要求）
        assert info.available is True
        assert info.degraded is not None
        assert "up" in info.degraded and "down" in info.degraded
        assert "left" in info.degraded and "right" in info.degraded


# ---------------------------------------------------------------------------
# 7. INFO2 端到端核验：behavior_overlay 先于 __aenter__ 挂上 → WARNING 级
# ---------------------------------------------------------------------------


class TestOptionalParamWarningLevel:
    async def test_connect_missing_optional_params_warns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``BehaviorService.connect()`` 先挂 ``behavior_overlay`` 再进
        ``__aenter__``（见其 docstring）——``_read_ranges`` 内「缺可选参数」
        告警应为 WARNING（非 DEBUG，INFO2）。默认 ``ranges=RANGES`` 不含 7 个
        可选叠加参数，天然命中该分支。"""
        _install(monkeypatch, BehaviorFakeVtsServer())
        service = BehaviorService()
        with caplog.at_level("DEBUG"):
            await service.connect()
        try:
            warnings = [
                r
                for r in caplog.records
                if r.levelno == logging.WARNING and "缺可选叠加参数" in r.getMessage()
            ]
            assert len(warnings) == 1
        finally:
            await service.disconnect()


# ---------------------------------------------------------------------------
# 8. 裸参数轨迹通道面（2026-07-31 二期）：
#    animate / clear_params / list_params / status
# ---------------------------------------------------------------------------

RANGES_WITH_MOUTH_X: dict[str, tuple[float, float, float]] = {
    **RANGES,
    "MouthX": (-1.0, 1.0, 0.0),
}
"""全参数表非降级用例：MouthX 既非 GOVERNED_PARAMS 也非可选叠加白名单，仅在
``sink.all_params``（``list_params`` 的作用空间）现身。"""


class TestTrajectoryServiceFace:
    """service 层轨迹面：未连接协议性拒绝、回执透传、``clear_params`` 幂等、
    全参数表 governed 标注、``status`` 的 trajectory_active/remaining 聚合。"""

    def test_animate_not_connected_raises_tool_error(self) -> None:
        service = BehaviorService()
        request = TrajectoryRequest(
            keyframes=[TrajectoryKeyframe(t_ms=0, params={"MouthSmile": 0.5})]
        )
        with pytest.raises(ToolError) as exc_info:
            service.animate(request)
        assert extract_vtsb_code(str(exc_info.value)) == VTSB_NOT_CONNECTED

    def test_list_params_not_connected_returns_none(self) -> None:
        service = BehaviorService()
        catalog = service.list_params()
        assert catalog.connected is False
        assert catalog.params is None

    async def test_animate_accepted_passes_through_queue_depth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch, BehaviorFakeVtsServer())
        service = BehaviorService()
        await service.connect()
        try:
            first = service.animate(
                TrajectoryRequest(
                    keyframes=[
                        TrajectoryKeyframe(t_ms=0, params={"MouthSmile": 0.2}),
                        TrajectoryKeyframe(t_ms=500, params={"MouthSmile": 0.2}),
                    ]
                )
            )
            assert first.status == "accepted"
            assert first.queue_depth == 1  # 首次投喂无队可入，直接接管
            second = service.animate(
                TrajectoryRequest(
                    keyframes=[
                        TrajectoryKeyframe(t_ms=0, params={"MouthSmile": 0.3}),
                        TrajectoryKeyframe(t_ms=500, params={"MouthSmile": 0.3}),
                    ],
                    append=True,
                )
            )
            assert second.status == "accepted"
            assert second.queue_depth == 2  # 透传 TrajectoryPlayer.feed 的实际队深，非硬编码
        finally:
            await service.disconnect()

    async def test_clear_params_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, BehaviorFakeVtsServer())
        service = BehaviorService()
        await service.connect()
        try:
            first = service.clear_params()
            second = service.clear_params()
            assert first.status == "accepted"
            assert second.status == "accepted"
        finally:
            await service.disconnect()

    async def test_list_params_connected_returns_full_table_with_governed_flags(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch, BehaviorFakeVtsServer(ranges=RANGES_WITH_MOUTH_X))
        service = BehaviorService()
        await service.connect()
        try:
            catalog = service.list_params()
            assert catalog.connected is True
            assert catalog.params is not None
            by_name = {p.name: p for p in catalog.params}
            assert set(by_name) == set(RANGES_WITH_MOUTH_X)
            for name in GOVERNED_PARAMS:
                assert by_name[name].governed is True
            assert by_name["MouthX"].governed is False
        finally:
            await service.disconnect()

    async def test_status_reflects_active_trajectory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, BehaviorFakeVtsServer())
        service = BehaviorService()
        await service.connect()
        try:
            service.animate(
                TrajectoryRequest(
                    keyframes=[
                        TrajectoryKeyframe(t_ms=0, params={"MouthSmile": 0.5}),
                        TrajectoryKeyframe(t_ms=500, params={"MouthSmile": 0.5}),
                    ]
                )
            )
            status = service.status()
            assert status.trajectory_active is True
            assert status.trajectory_remaining_ms > 0
        finally:
            await service.disconnect()


# ---------------------------------------------------------------------------
# 9. 语音播放面（speech-play 蓝图 2026-08-14 §T4）
# ---------------------------------------------------------------------------


def _speech_request(wav_path: str = "/abs/x.wav") -> SpeechRequest:
    return SpeechRequest(
        wav_path=wav_path,
        mouth_track=[TrajectoryKeyframe(t_ms=0, params={"MouthOpen": 0.5})],
    )


class TestSpeechPlayService:
    """未连接协议性拒绝、``status`` 默认值、懒加载零回归、成功路径回执。"""

    async def test_speech_play_not_connected_raises_tool_error(self) -> None:
        service = BehaviorService()
        with pytest.raises(ToolError) as exc_info:
            await service.speech_play(_speech_request())
        assert extract_vtsb_code(str(exc_info.value)) == VTSB_NOT_CONNECTED

    async def test_status_speech_fields_default_before_any_speech_play(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch, BehaviorFakeVtsServer())
        service = BehaviorService()
        await service.connect()
        try:
            status = service.status()
            assert status.speech_active is False
            assert status.speech_queue_depth == 0
            assert status.speech_last_error is None
        finally:
            await service.disconnect()

    async def test_speech_playback_module_not_imported_before_first_speech_play(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """懒加载零回归：``connect()`` 前后、首次 ``speech_play()`` 之前，
        ``src.mcp.behavior.speech_playback`` 不得进 ``sys.modules``。

        测试进程内其它用例（如 ``test_vts_speech_playback.py``）可能已 import
        过该模块——先从 ``sys.modules`` 快照剔除，模拟"进程内尚未触达过"的
        状态，测后原样恢复（不泄漏给其它用例）。
        """
        import sys

        target = "src.mcp.behavior.speech_playback"
        saved = sys.modules.pop(target, None)
        try:
            _install(monkeypatch, BehaviorFakeVtsServer())
            service = BehaviorService()
            await service.connect()
            try:
                assert target not in sys.modules, (
                    "connect() 不得触发 speech_playback 懒加载（首次 speech_play() 才该拉起）"
                )
            finally:
                await service.disconnect()
        finally:
            if saved is not None:
                sys.modules[target] = saved

    async def test_speech_play_success_returns_receipt_and_enqueues_job(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """成功路径：fake sink + monkeypatch ``read_wav_meta``/``SpeechQueue`` →
        回执 ``accepted=True``、``duration_ms`` 与 wav 读取结果一致，且入队的
        ``SpeechJob`` 携带同一 duration_ms（不是硬编码/巧合相等）。"""
        import src.mcp.behavior.speech_playback as speech_playback_mod

        _install(monkeypatch, BehaviorFakeVtsServer())
        service = BehaviorService()
        await service.connect()
        try:
            fake_duration_ms = 3210.0

            def _fake_read_wav_meta(wav_path: str) -> tuple[bytes, float]:
                assert wav_path == "/abs/x.wav"
                return b"\x00" * 128, fake_duration_ms

            monkeypatch.setattr(speech_playback_mod, "read_wav_meta", _fake_read_wav_meta)

            enqueued: list[Any] = []

            class _FakeSpeechQueue:
                def __init__(self, speech_mouth: Any) -> None:
                    self.speech_mouth = speech_mouth

                async def enqueue(self, job: Any) -> None:
                    enqueued.append(job)

                async def aclose(self) -> None:
                    return None

            monkeypatch.setattr(speech_playback_mod, "SpeechQueue", _FakeSpeechQueue)

            receipt = await service.speech_play(_speech_request())
            assert receipt.accepted is True
            assert receipt.duration_ms == fake_duration_ms
            assert len(enqueued) == 1
            assert enqueued[0].duration_ms == fake_duration_ms
            assert enqueued[0].frames == b"\x00" * 128
        finally:
            await service.disconnect()
