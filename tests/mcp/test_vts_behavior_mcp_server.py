"""VTS 行为 MCP server 工具层测试（蓝图 2026-07-31 §8.4 · T10）。

被测：``src/mcp/vts_behavior_mcp_server.py``（薄工具层）+ 其对
``src/mcp/behavior/service.py`` 的转发/错误映射契约（AD-8 / AD-11）。

覆盖：
  1. 注册面：六工具全注册、readOnlyHint 集合正确、无 destructiveHint。
  2. feature flag 关（默认）→ 六工具全 ToolError，且 ``[vtsb:disabled]`` 令牌在
     **FastMCP 加壳后的真 wire 形态**（``"Error executing tool <name>: ..."``）上
     经 ``re.search`` 可提取——夹具经 ``Tool.run`` 走 SDK 真实加壳分支，不手拼
     未加壳原文（`rules/mcp-integration.md`「绿灯从没能红」教训）。
  3. flag 开 + 真 service 未 connect → ``behavior_list`` 仍完整返回 12 词静态词表
     （hotkeys=None）；trigger/interrupt 抛 ``[vtsb:not_connected]``（原样透传，
     不被二次包成 ``[vtsb:vts_error]``）。
  4. flag 开 + 假 service（monkeypatch ``_SERVICE``）→ 回执三态透传（rejected 是
     正常返回不抛异常）、逐码透传、参数转发、connect 幂等、catalog 含热键。
  5. 错误映射逐码：解析层 ValidationError → ``[vtsb:invalid_params]``；服务层
     ToolError 原样透传；其余异常 → ``[vtsb:vts_error]``。

不标 ``zerorepo``（无关 D:\\Zero）；无真 VTS 依赖（假 service / 未连接路径）。
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from mcp.server.fastmcp.exceptions import ToolError

import src.mcp.vts_behavior_mcp_server as server_mod
from src.agents.models.vts_behavior import (
    VTSB_CHANNEL_BUSY,
    VTSB_CODE_RE,
    VTSB_COOLDOWN,
    VTSB_DISABLED,
    VTSB_HOTKEY_UNAVAILABLE,
    VTSB_INVALID_PARAMS,
    VTSB_NOT_CONNECTED,
    VTSB_THROTTLED,
    VTSB_UNKNOWN_BEHAVIOR,
    VTSB_VTS_ERROR,
    BehaviorCatalog,
    BehaviorInfo,
    BehaviorReceipt,
    BehaviorRequest,
    BehaviorStatus,
    HotkeyInfo,
    ParamCatalog,
    ParamInfo,
    TrajectoryReceipt,
    TrajectoryRequest,
    extract_vtsb_code,
)
from src.mcp.behavior.service import BehaviorService
from src.mcp.vts_behavior_mcp_server import _is_enabled, _require_enabled, mcp

FLAG_ENV = "VTS_BEHAVIOR_ENABLED"

EXPECTED_TOOLS = {
    "behavior_list",
    "behavior_trigger",
    "behavior_interrupt",
    "behavior_status",
    "params_list",
    "params_animate",
    "params_clear",
    "vts_connect",
    "vts_disconnect",
}

READ_ONLY_TOOLS = {"behavior_list", "behavior_status", "params_list"}

# 各工具经 `Tool.run` 调用时的最小合法入参（缺省全走默认值）。
MINIMAL_ARGS: dict[str, dict[str, Any]] = {
    "behavior_list": {},
    "behavior_trigger": {"name": "nod"},
    "behavior_interrupt": {},
    "behavior_status": {},
    "params_list": {},
    "params_animate": {"keyframes": [{"t_ms": 0, "params": {"FaceAngleX": 0.0}}]},
    "params_clear": {},
    "vts_connect": {},
    "vts_disconnect": {},
}


def _get_tool(name: str) -> Any:
    return mcp._tool_manager._tools[name]


async def _run_on_wire(name: str, arguments: dict[str, Any]) -> Any:
    """经 SDK ``Tool.run`` 调用工具——异常按 FastMCP 真实分支加壳。

    🛑 夹具纪律（AD-11 / `rules/mcp-integration.md`）：wire 上的 ToolError 文本
    永远带 ``"Error executing tool <name>: "`` 外壳（``Tool.run`` 的
    ``except Exception`` 分支，ToolError 自身也继承 Exception 照样被再包一层）。
    直接调工具函数拿到的是未加壳原文——用它验令牌判据会重演「单测绿、真 wire
    恒 False」的生产死码。本文件所有令牌提取断言一律走本 helper。
    """
    return await _get_tool(name).run(arguments)


# ---------------------------------------------------------------------------
# 假 service：只回放预置模型 / 预置异常，记录转发入参
# ---------------------------------------------------------------------------

_STATUS_CONNECTED = BehaviorStatus(
    connected=True,
    healthy=True,
    hotkey_count=1,
    model_id="fake-model-id",
)

_HOTKEY = HotkeyInfo(
    hotkey_id="hk-wave",
    name="Wave",
    type="TriggerAnimation",
    file="wave.motion3.json",
    kind="animation",
)

_CATALOG_WITH_HOTKEYS = BehaviorCatalog(
    behaviors=[
        BehaviorInfo(
            name="nod",
            definition="肯定/附和：低头-回位",
            typical_duration_ms=700,
            cooldown_s=1.5,
            channels=["head"],
        )
    ],
    hotkeys=[_HOTKEY],
    connected=True,
)

_RECEIPT_ACCEPTED = BehaviorReceipt(
    status="accepted",
    behavior_id="fake-behavior-id",
    channels=["head"],
    estimated_duration_ms=700,
)


class FakeBehaviorService:
    """回放型假 service：与 ``BehaviorService`` 同签名面，记录每次转发。

    ``raise_exc`` 置异常时六方法统一改为抛出——用于逐工具验 server 层错误映射。
    """

    def __init__(self) -> None:
        self.raise_exc: Exception | None = None
        self.receipt: BehaviorReceipt = _RECEIPT_ACCEPTED
        self.catalog: BehaviorCatalog = _CATALOG_WITH_HOTKEYS
        self.status_model: BehaviorStatus = _STATUS_CONNECTED
        self.trigger_requests: list[BehaviorRequest] = []
        self.interrupt_channels: list[str | None] = []
        self.list_refreshes: list[bool] = []
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.trajectory_receipt: TrajectoryReceipt = TrajectoryReceipt(
            status="accepted", duration_ms=500, queue_depth=1
        )
        self.param_catalog: ParamCatalog = ParamCatalog(
            params=[
                ParamInfo(name="FaceAngleX", min=-30.0, max=30.0, default_value=0.0, governed=True)
            ],
            connected=True,
        )
        self.animate_requests: list[TrajectoryRequest] = []
        self.clear_params_calls = 0

    def _maybe_raise(self) -> None:
        if self.raise_exc is not None:
            raise self.raise_exc

    async def connect(self) -> BehaviorStatus:
        self._maybe_raise()
        self.connect_calls += 1
        return self.status_model

    async def disconnect(self) -> BehaviorStatus:
        self._maybe_raise()
        self.disconnect_calls += 1
        return self.status_model

    async def trigger(self, request: BehaviorRequest) -> BehaviorReceipt:
        self._maybe_raise()
        self.trigger_requests.append(request)
        return self.receipt

    def interrupt(self, channel: str | None = None) -> BehaviorReceipt:
        self._maybe_raise()
        self.interrupt_channels.append(channel)
        return self.receipt

    async def list_catalog(self, refresh: bool = False) -> BehaviorCatalog:
        self._maybe_raise()
        self.list_refreshes.append(refresh)
        return self.catalog

    def status(self) -> BehaviorStatus:
        self._maybe_raise()
        return self.status_model

    # ── 轨迹通道（2026-07-31 二期）──────────────────────────────────────────

    def animate(self, request: TrajectoryRequest) -> TrajectoryReceipt:
        self._maybe_raise()
        self.animate_requests.append(request)
        return self.trajectory_receipt

    def clear_params(self) -> TrajectoryReceipt:
        self._maybe_raise()
        self.clear_params_calls += 1
        return self.trajectory_receipt

    def list_params(self) -> ParamCatalog:
        self._maybe_raise()
        return self.param_catalog


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


@pytest.fixture
def flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """flag 关（默认态）：并清空单例——工具体在 flag 检查处即断，不得触到 service。"""
    monkeypatch.delenv(FLAG_ENV, raising=False)
    monkeypatch.setattr(server_mod, "_SERVICE", None)


@pytest.fixture
def fake_service(monkeypatch: pytest.MonkeyPatch) -> FakeBehaviorService:
    """flag 开 + 假 service（monkeypatch 单例，测后自动还原为 None）。"""
    monkeypatch.setenv(FLAG_ENV, "true")
    fake = FakeBehaviorService()
    monkeypatch.setattr(server_mod, "_SERVICE", fake)
    return fake


@pytest.fixture
def real_unconnected_service(monkeypatch: pytest.MonkeyPatch) -> BehaviorService:
    """flag 开 + **真** service（未 connect、无 sink）：走真实的未连接协议路径。"""
    monkeypatch.setenv(FLAG_ENV, "true")
    service = BehaviorService()
    monkeypatch.setattr(server_mod, "_SERVICE", service)
    return service


# ---------------------------------------------------------------------------
# 注册面
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_all_tools_registered(self) -> None:
        """九工具全部注册，无多无少（六行为/连接 + 三轨迹通道，2026-07-31 二期）。"""
        assert set(mcp._tool_manager._tools.keys()) == EXPECTED_TOOLS

    def test_read_only_hints(self) -> None:
        """behavior_list / behavior_status 为 readOnly，其余不得标 readOnly。"""
        for name in EXPECTED_TOOLS:
            ann = _get_tool(name).annotations
            assert ann is not None, f"{name} 缺 annotations"
            if name in READ_ONLY_TOOLS:
                assert ann.readOnlyHint is True, f"{name} 应为 readOnlyHint=True"
            else:
                assert ann.readOnlyHint is not True, f"{name} 不应为 readOnlyHint=True"

    def test_no_destructive_hint(self) -> None:
        """行为层无破坏性工具（瞬态动作自动回基准，disconnect 幂等可重连）。"""
        for name in EXPECTED_TOOLS:
            ann = _get_tool(name).annotations
            assert ann is None or ann.destructiveHint is not True, (
                f"{name} 不应为 destructiveHint=True"
            )


# ---------------------------------------------------------------------------
# feature flag
# ---------------------------------------------------------------------------


class TestFeatureFlag:
    def test_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """未设 env 时 flag 关（默认关零回归，AD-12）。"""
        monkeypatch.delenv(FLAG_ENV, raising=False)
        assert _is_enabled() is False

    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "YES"])
    def test_enabled_variants(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv(FLAG_ENV, value)
        assert _is_enabled() is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "", "on2", "enabled"])
    def test_disabled_variants(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        """真值集外的值一律视为关（与仓内先例一致的保守语义）。"""
        monkeypatch.setenv(FLAG_ENV, value)
        assert _is_enabled() is False

    def test_require_enabled_raises_when_off(self, flag_off: None) -> None:
        with pytest.raises(ToolError) as excinfo:
            _require_enabled()
        assert extract_vtsb_code(str(excinfo.value)) == VTSB_DISABLED

    def test_require_enabled_ok_when_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(FLAG_ENV, "true")
        _require_enabled()  # 不抛


class TestFlagOffAllToolsOnWire:
    """flag 关 → 六工具全 ToolError，令牌在**加壳后的真 wire 形态**上可提取。"""

    @pytest.mark.parametrize("tool_name", sorted(EXPECTED_TOOLS))
    async def test_tool_rejected_with_disabled_token(self, flag_off: None, tool_name: str) -> None:
        with pytest.raises(ToolError) as excinfo:
            await _run_on_wire(tool_name, MINIMAL_ARGS[tool_name])
        wire = str(excinfo.value)
        # ① 加壳确实发生（否则夹具没判别力——等于回到未加壳假绿）
        assert wire.startswith(f"Error executing tool {tool_name}: ")
        # ② 位置 0 裸前缀判据在真 wire 上恒 False（mcp-integration 死码教训复现）
        assert wire.lstrip().startswith(VTSB_DISABLED) is False
        # ③ 位置无关令牌经 re.search 可提取（契约唯一真相 VTSB_CODE_RE / extract_vtsb_code）
        match = VTSB_CODE_RE.search(wire)
        assert match is not None and match.group(0) == VTSB_DISABLED
        assert extract_vtsb_code(wire) == VTSB_DISABLED

    async def test_wire_shape_matches_sdk_format(self, flag_off: None) -> None:
        """外壳文案逐字对齐 SDK `f"Error executing tool {name}: {e}"`——防 SDK 升级悄改壳形。"""
        with pytest.raises(ToolError) as excinfo:
            await _run_on_wire("behavior_status", {})
        assert re.fullmatch(
            r"Error executing tool behavior_status: \[vtsb:disabled\] .+",
            str(excinfo.value),
            flags=re.DOTALL,
        )


# ---------------------------------------------------------------------------
# flag 开 + 真 service 未 connect：静态词表 / [vtsb:not_connected]
# ---------------------------------------------------------------------------


class TestEnabledUnconnected:
    async def test_behavior_list_full_vocabulary_without_connection(
        self, real_unconnected_service: BehaviorService
    ) -> None:
        """词表是静态知识：未连接也完整返回 12 词（hotkeys=None=尚未枚举）。"""
        from src.mcp.zero.sinks.behavior_overlay import VOCABULARY  # noqa: PLC0415

        payload = json.loads(await server_mod.behavior_list())
        assert payload["connected"] is False
        assert payload["hotkeys"] is None
        names = [b["name"] for b in payload["behaviors"]]
        assert names == list(VOCABULARY)
        assert len(names) == 12
        for behavior in payload["behaviors"]:
            assert behavior["definition"], f"{behavior['name']} 缺定义文本"
            assert behavior["typical_duration_ms"] > 0
            assert behavior["available"] is True  # 未连接按静态知识乐观返回

    async def test_status_works_without_connection(
        self, real_unconnected_service: BehaviorService
    ) -> None:
        payload = json.loads(await server_mod.behavior_status())
        assert payload["connected"] is False
        assert payload["healthy"] is False
        assert payload["hotkey_count"] is None
        assert payload["model_id"] is None

    @pytest.mark.parametrize(
        ("tool_name", "arguments"),
        [
            ("behavior_trigger", {"name": "nod"}),
            ("behavior_interrupt", {}),
        ],
    )
    async def test_not_connected_token_on_wire(
        self,
        real_unconnected_service: BehaviorService,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """未 connect 时 trigger/interrupt → ``[vtsb:not_connected]``，且是服务层
        ToolError **原样透传**（不被 server 二次包成 ``[vtsb:vts_error]``）。"""
        with pytest.raises(ToolError) as excinfo:
            await _run_on_wire(tool_name, arguments)
        wire = str(excinfo.value)
        assert wire.startswith(f"Error executing tool {tool_name}: ")
        assert extract_vtsb_code(wire) == VTSB_NOT_CONNECTED
        assert VTSB_VTS_ERROR not in wire


# ---------------------------------------------------------------------------
# flag 开 + 假 service：转发与回执透传
# ---------------------------------------------------------------------------


class TestTriggerPassthrough:
    async def test_request_fields_forwarded(self, fake_service: FakeBehaviorService) -> None:
        """工具入参逐字段进 ``BehaviorRequest`` 转发给 service（传输层零业务逻辑）。"""
        await server_mod.behavior_trigger(
            name="glance", intensity=0.8, repeat=2, duration_ms=1500, direction="left"
        )
        (request,) = fake_service.trigger_requests
        assert isinstance(request, BehaviorRequest)
        assert request.name == "glance"
        assert request.intensity == 0.8
        assert request.repeat == 2
        assert request.duration_ms == 1500
        assert request.direction == "left"

    @pytest.mark.parametrize("status", ["accepted", "replaced"])
    async def test_positive_receipt_passthrough(
        self, fake_service: FakeBehaviorService, status: str
    ) -> None:
        fake_service.receipt = BehaviorReceipt(
            status=status,
            behavior_id="bid-1",
            channels=["head"],
            estimated_duration_ms=700,
        )
        payload = json.loads(await server_mod.behavior_trigger(name="nod"))
        assert payload["status"] == status
        assert payload["behavior_id"] == "bid-1"
        assert payload["channels"] == ["head"]
        assert payload["code"] is None

    @pytest.mark.parametrize(
        "code",
        [
            VTSB_UNKNOWN_BEHAVIOR,
            VTSB_INVALID_PARAMS,
            VTSB_COOLDOWN,
            VTSB_THROTTLED,
            VTSB_CHANNEL_BUSY,
            VTSB_HOTKEY_UNAVAILABLE,
        ],
    )
    async def test_rejected_receipt_is_normal_return_per_code(
        self, fake_service: FakeBehaviorService, code: str
    ) -> None:
        """业务性拒绝逐码透传进回执 ``code`` 字段——是正常返回，**不抛 ToolError**（AD-11）。"""
        fake_service.receipt = BehaviorReceipt(
            status="rejected",
            behavior_id="bid-2",
            channels=[],
            estimated_duration_ms=0,
            code=code,
            detail="业务性拒绝",
        )
        payload = json.loads(await server_mod.behavior_trigger(name="nod"))
        assert payload["status"] == "rejected"
        assert payload["code"] == code
        assert extract_vtsb_code(payload["code"]) == code

    async def test_degraded_channels_passthrough(self, fake_service: FakeBehaviorService) -> None:
        fake_service.receipt = BehaviorReceipt(
            status="accepted",
            behavior_id="bid-3",
            channels=["head"],
            estimated_duration_ms=2500,
            degraded_channels=["body"],
        )
        payload = json.loads(await server_mod.behavior_trigger(name="lean_in"))
        assert payload["degraded_channels"] == ["body"]


class TestCatalogAndStatusPassthrough:
    async def test_behavior_list_contains_vocabulary_and_hotkeys(
        self, fake_service: FakeBehaviorService
    ) -> None:
        """已连接 catalog：程序化词表与热键在**同一张清单**（AD-7）。"""
        payload = json.loads(await server_mod.behavior_list())
        assert payload["connected"] is True
        assert [b["name"] for b in payload["behaviors"]] == ["nod"]
        (hotkey,) = payload["hotkeys"]
        assert hotkey["hotkey_id"] == "hk-wave"
        assert hotkey["type"] == "TriggerAnimation"
        assert hotkey["kind"] == "animation"
        assert fake_service.list_refreshes == [False]

    async def test_behavior_list_forwards_refresh(self, fake_service: FakeBehaviorService) -> None:
        await server_mod.behavior_list(refresh=True)
        assert fake_service.list_refreshes == [True]

    async def test_interrupt_forwards_channel(self, fake_service: FakeBehaviorService) -> None:
        await server_mod.behavior_interrupt(channel="head")
        await server_mod.behavior_interrupt()
        assert fake_service.interrupt_channels == ["head", None]

    async def test_status_passthrough(self, fake_service: FakeBehaviorService) -> None:
        payload = json.loads(await server_mod.behavior_status())
        assert payload["connected"] is True
        assert payload["healthy"] is True
        assert payload["hotkey_count"] == 1
        assert payload["model_id"] == "fake-model-id"


class TestConnectionTools:
    async def test_connect_idempotent(self, fake_service: FakeBehaviorService) -> None:
        """vts_connect 幂等：重复调用均成功且返回一致状态（幂等语义在 service，
        server 层验证的是重复调用不炸、逐次如实转发）。"""
        first = json.loads(await server_mod.vts_connect())
        second = json.loads(await server_mod.vts_connect())
        assert first == second
        assert first["connected"] is True
        assert fake_service.connect_calls == 2

    async def test_disconnect_returns_status(self, fake_service: FakeBehaviorService) -> None:
        fake_service.status_model = BehaviorStatus(connected=False, healthy=False)
        payload = json.loads(await server_mod.vts_disconnect())
        assert payload["connected"] is False
        assert fake_service.disconnect_calls == 1


# ---------------------------------------------------------------------------
# 错误映射逐码（AD-11）
# ---------------------------------------------------------------------------


class TestErrorMapping:
    async def test_invalid_request_maps_to_invalid_params(
        self, fake_service: FakeBehaviorService
    ) -> None:
        """解析层 ValidationError → ``[vtsb:invalid_params]``，service 不被触到。"""
        with pytest.raises(ToolError) as excinfo:
            await _run_on_wire("behavior_trigger", {"name": "nod", "intensity": 1.5})
        wire = str(excinfo.value)
        assert wire.startswith("Error executing tool behavior_trigger: ")
        assert extract_vtsb_code(wire) == VTSB_INVALID_PARAMS
        assert fake_service.trigger_requests == []

    @pytest.mark.parametrize(
        "arguments",
        [
            {"name": ""},
            {"name": "nod", "repeat": 9},
            {"name": "nod", "duration_ms": 10_001},
        ],
    )
    async def test_boundary_violations_map_to_invalid_params(
        self, fake_service: FakeBehaviorService, arguments: dict[str, Any]
    ) -> None:
        with pytest.raises(ToolError) as excinfo:
            await _run_on_wire("behavior_trigger", arguments)
        assert extract_vtsb_code(str(excinfo.value)) == VTSB_INVALID_PARAMS
        assert fake_service.trigger_requests == []

    @pytest.mark.parametrize("tool_name", sorted(EXPECTED_TOOLS))
    async def test_generic_exception_maps_to_vts_error(
        self, fake_service: FakeBehaviorService, tool_name: str
    ) -> None:
        """服务层任意异常 → ``[vtsb:vts_error]``（六工具逐一，含原始错误文本）。"""
        fake_service.raise_exc = RuntimeError("ws 撕裂")
        with pytest.raises(ToolError) as excinfo:
            await _run_on_wire(tool_name, MINIMAL_ARGS[tool_name])
        wire = str(excinfo.value)
        assert wire.startswith(f"Error executing tool {tool_name}: ")
        assert extract_vtsb_code(wire) == VTSB_VTS_ERROR
        assert "ws 撕裂" in wire

    @pytest.mark.parametrize("tool_name", sorted(EXPECTED_TOOLS))
    async def test_service_toolerror_passthrough_not_double_wrapped(
        self, fake_service: FakeBehaviorService, tool_name: str
    ) -> None:
        """服务层 ToolError 原样透传（``except ToolError: raise``）——首个令牌保持
        服务层给出的码，不被 server 兜底改写为 ``[vtsb:vts_error]``。"""
        fake_service.raise_exc = ToolError(f"{VTSB_NOT_CONNECTED} 尚未连接 VTS")
        with pytest.raises(ToolError) as excinfo:
            await _run_on_wire(tool_name, MINIMAL_ARGS[tool_name])
        wire = str(excinfo.value)
        assert extract_vtsb_code(wire) == VTSB_NOT_CONNECTED
        assert VTSB_VTS_ERROR not in wire
