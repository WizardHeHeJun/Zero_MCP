"""test_client_contract.py — DesktopMCPClient 10 方法 round-trip + 异常 + 缓存测试。

覆盖：
- flag=false 时 __aenter__ 抛 DesktopCapabilityDisabledError（零回归）
- AsyncMock ClientSession 10 方法 round-trip（返回正确 pydantic 模型）
- get_capability_flags 使用缓存、不重新调用 server
- _extract_text：isError=True 抛 DesktopMCPCallError
- _extract_text：空 content 抛 DesktopMCPCallError
- __aexit__ 后 capability_cache=None、session=None
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp.desktop_mcp_client import (
    DesktopCapabilityDisabledError,
    DesktopMCPCallError,
    DesktopMCPClient,
    _extract_text,
    _is_enabled,
)

# ── feature flag ──────────────────────────────────────────────────────────────


def test_client_is_enabled_false_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCREEN_CAPABILITY_ENABLED", raising=False)
    assert _is_enabled() is False


async def test_aenter_raises_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """flag=false 时 __aenter__ 立即抛 DesktopCapabilityDisabledError。"""
    monkeypatch.delenv("SCREEN_CAPABILITY_ENABLED", raising=False)
    client = DesktopMCPClient()
    with pytest.raises(DesktopCapabilityDisabledError):
        await client.__aenter__()


# ── _extract_text ─────────────────────────────────────────────────────────────


def _make_call_result(
    is_error: bool = False,
    text: str = "",
    empty_content: bool = False,
    non_text_content: bool = False,
) -> MagicMock:
    """构造 MCP CallToolResult mock。"""
    from mcp.types import TextContent

    result = MagicMock()
    result.isError = is_error
    if empty_content:
        result.content = []
    elif non_text_content:
        fake = MagicMock(spec=[])  # 非 TextContent
        result.content = [fake]
    else:
        tc = MagicMock(spec=TextContent)
        tc.text = text
        result.content = [tc]
    return result


def test_extract_text_ok() -> None:
    r = _make_call_result(text="hello")
    assert _extract_text(r, "tool") == "hello"


def test_extract_text_is_error_raises() -> None:
    r = _make_call_result(is_error=True, text="boom")
    with pytest.raises(DesktopMCPCallError) as exc_info:
        _extract_text(r, "my_tool")
    assert exc_info.value.tool == "my_tool"
    assert "boom" in str(exc_info.value)


def test_extract_text_empty_content_raises() -> None:
    r = _make_call_result(empty_content=True)
    with pytest.raises(DesktopMCPCallError):
        _extract_text(r, "tool")


def test_extract_text_non_text_content_raises() -> None:
    r = _make_call_result(non_text_content=True)
    with pytest.raises(DesktopMCPCallError):
        _extract_text(r, "tool")


# ── 10 方法 round-trip（AsyncMock session） ───────────────────────────────────


# 预制各工具返回的 JSON fixture
def _make_screen_snapshot_json() -> str:
    return json.dumps(
        {
            "snapshot_id": "snap-001",
            "timestamp_ms": 1000,
            "screen_width": 1920,
            "screen_height": 1080,
            "active_window_title": "Test",
            "uia_elements": [],
            "text_blocks": [],
            "visual_objects": [],
            "screenshot_path": None,
            "perception_mode": "uia_ocr",
            "capability_flags": {"ocr": True},
            "is_untrusted": True,
            "uia_hollow": False,
        }
    )


def _make_uia_tree_json() -> str:
    return json.dumps(
        [
            {
                "element_id": "uia_1234_0",
                "control_type": "ButtonControl",
                "name": "OK",
                "automation_id": None,
                "bbox": {"x": 10, "y": 20, "width": 80, "height": 30},
                "is_enabled": True,
                "is_visible": True,
                "value": None,
                "source": "uia",
            }
        ]
    )


def _make_action_result_json(success: bool = True) -> str:
    return json.dumps(
        {
            "action_id": "act-001",
            "success": success,
            "error_message": None,
            "ui_changed": False,
        }
    )


def _make_window_list_json() -> str:
    return json.dumps(
        [
            {
                "hwnd": 99999,
                "title": "Notepad",
                "class_name": "Notepad",
                "visible": True,
                "rect": {"left": 0, "top": 0, "right": 800, "bottom": 600},
            }
        ]
    )


def _make_capability_flags_json() -> str:
    return json.dumps(
        {
            "ocr": True,
            "omniparser": False,
            "cuda_accel": False,
            "dml_accel": False,
            "mss_available": True,
            "effective_device": "cpu",
        }
    )


def _make_ocr_result_json() -> str:
    return json.dumps(
        [
            {
                "block_id": "ocr_snap_0",
                "text": "Hello",
                "bbox": {"x": 5, "y": 5, "width": 50, "height": 20},
                "confidence": 0.98,
                "source": "ocr_rapidocr",
            }
        ]
    )


def _build_mock_client(monkeypatch: pytest.MonkeyPatch) -> tuple[DesktopMCPClient, AsyncMock]:
    """构建一个绕过真实 stdio 连接的 DesktopMCPClient + mock session。"""
    monkeypatch.setenv("SCREEN_CAPABILITY_ENABLED", "true")

    mock_session = AsyncMock()
    client = DesktopMCPClient()
    # 直接注入 session 和 capability_cache，绕过 __aenter__ 的 spawn 逻辑
    client.session = mock_session
    client.capability_cache = {"ocr": True, "omniparser": False, "mss_available": True}
    return client, mock_session


def _set_tool_return(mock_session: AsyncMock, text: str) -> None:
    """设置 session.call_tool 返回指定文本内容。"""
    from mcp.types import TextContent

    tc = MagicMock(spec=TextContent)
    tc.text = text
    call_result = MagicMock()
    call_result.isError = False
    call_result.content = [tc]
    mock_session.call_tool.return_value = call_result


async def test_screen_snapshot_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.agents.models.screen_snapshot import ScreenSnapshot

    client, mock_session = _build_mock_client(monkeypatch)
    _set_tool_return(mock_session, _make_screen_snapshot_json())
    result = await client.screen_snapshot(mode="uia_ocr")
    assert isinstance(result, ScreenSnapshot)
    assert result.snapshot_id == "snap-001"
    mock_session.call_tool.assert_called_once_with(
        "screen_snapshot", {"mode": "uia_ocr", "capture_screenshot": False}
    )


async def test_screen_snapshot_with_window_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    """window_handle 指定时随参数下发（定向感知，解除前台耦合）；None 时不带键。"""
    client, mock_session = _build_mock_client(monkeypatch)
    _set_tool_return(mock_session, _make_screen_snapshot_json())
    await client.screen_snapshot(mode="uia_ocr", window_handle=0x1234)
    mock_session.call_tool.assert_called_once_with(
        "screen_snapshot",
        {"mode": "uia_ocr", "capture_screenshot": False, "window_handle": 0x1234},
    )


async def test_get_uia_tree_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.agents.models.screen_snapshot import UIAElement

    client, mock_session = _build_mock_client(monkeypatch)
    _set_tool_return(mock_session, _make_uia_tree_json())
    result = await client.get_uia_tree(max_depth=3)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], UIAElement)
    assert result[0].control_type == "ButtonControl"


async def test_ocr_region_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.agents.models.screen_snapshot import BBox, TextBlock

    client, mock_session = _build_mock_client(monkeypatch)
    _set_tool_return(mock_session, _make_ocr_result_json())
    bbox = BBox(x=0, y=0, width=100, height=100)
    result = await client.ocr_region(bbox=bbox, screenshot_path="/fake.png")
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], TextBlock)
    assert result[0].text == "Hello"


async def test_click_element_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.agents.models.screen_snapshot import ActionResult

    client, mock_session = _build_mock_client(monkeypatch)
    _set_tool_return(mock_session, _make_action_result_json())
    result = await client.click_element(coordinates=(100, 200), method="coordinate")
    assert isinstance(result, ActionResult)
    assert result.success is True
    # 零回归：未提供 expected_root_hwnd 时参数不带该键（server 侧走现行为不核验）
    mock_session.call_tool.assert_called_once_with(
        "click_element", {"method": "coordinate", "coordinates": [100, 200]}
    )


async def test_click_element_expected_root_hwnd_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """K7 批2：expected_root_hwnd 提供时随参数透传到 server 工具。"""
    client, mock_session = _build_mock_client(monkeypatch)
    _set_tool_return(mock_session, _make_action_result_json())
    await client.click_element(
        coordinates=(100, 200), method="coordinate", expected_root_hwnd=0xAAA
    )
    mock_session.call_tool.assert_called_once_with(
        "click_element",
        {"method": "coordinate", "coordinates": [100, 200], "expected_root_hwnd": 0xAAA},
    )


async def test_type_text_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.agents.models.screen_snapshot import ActionResult

    client, mock_session = _build_mock_client(monkeypatch)
    _set_tool_return(mock_session, _make_action_result_json())
    result = await client.type_text(text="你好", method="clipboard")
    assert isinstance(result, ActionResult)
    assert result.success is True
    mock_session.call_tool.assert_called_once_with(
        "type_text", {"text": "你好", "method": "clipboard"}
    )


async def test_send_key_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.agents.models.screen_snapshot import ActionResult

    client, mock_session = _build_mock_client(monkeypatch)
    _set_tool_return(mock_session, _make_action_result_json())
    result = await client.send_key("ctrl+c")
    assert isinstance(result, ActionResult)
    mock_session.call_tool.assert_called_once_with("send_key", {"key_combo": "ctrl+c"})


async def test_window_list_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    client, mock_session = _build_mock_client(monkeypatch)
    _set_tool_return(mock_session, _make_window_list_json())
    result = await client.window_list()
    assert isinstance(result, list)
    assert result[0]["title"] == "Notepad"
    assert result[0]["hwnd"] == 99999


async def test_focus_window_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.agents.models.screen_snapshot import ActionResult

    client, mock_session = _build_mock_client(monkeypatch)
    _set_tool_return(mock_session, _make_action_result_json())
    result = await client.focus_window(window_handle=99999)
    assert isinstance(result, ActionResult)
    # 零回归：pin_topmost 默认 False 不随参数下发
    mock_session.call_tool.assert_called_once_with("focus_window", {"window_handle": 99999})


async def test_focus_window_pin_topmost_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    """K8：pin_topmost=True 时随参数透传。"""
    client, mock_session = _build_mock_client(monkeypatch)
    _set_tool_return(mock_session, _make_action_result_json())
    await client.focus_window(window_handle=99999, pin_topmost=True)
    mock_session.call_tool.assert_called_once_with(
        "focus_window", {"window_handle": 99999, "pin_topmost": True}
    )


async def test_screen_snapshot_hardening_fields_server_json_to_client_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """K2 贯穿断言：server 侧 model_dump_json（_dump_model 同口径）→ client
    model_validate_json，desktop_locked / window_captured / degradations 三个
    加固契约字段原样到位（契约模型单一真相，无白名单丢字段）。"""
    from src.agents.models.screen_snapshot import ScreenSnapshot

    server_side = ScreenSnapshot(
        snapshot_id="snap-locked",
        timestamp_ms=2000,
        screen_width=1920,
        screen_height=1080,
        active_window_title=None,
        uia_elements=[],
        text_blocks=[],
        visual_objects=[],
        screenshot_path=None,
        perception_mode="uia_ocr",
        capability_flags={"ocr": True},
        is_untrusted=True,
        uia_hollow=False,
        desktop_locked=True,
        window_captured=True,
        degradations=["desktop_locked", "ocr_unavailable"],
    )

    client, mock_session = _build_mock_client(monkeypatch)
    _set_tool_return(mock_session, server_side.model_dump_json())
    result = await client.screen_snapshot()

    assert result.desktop_locked is True
    assert result.window_captured is True
    assert result.degradations == ["desktop_locked", "ocr_unavailable"]


async def test_screen_snapshot_old_payload_defaults_hardening_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """旧 payload（无三个加固字段，见 _make_screen_snapshot_json）反序列化零回归：
    默认 desktop_locked=False / window_captured=False / degradations=[]。"""
    client, mock_session = _build_mock_client(monkeypatch)
    _set_tool_return(mock_session, _make_screen_snapshot_json())
    result = await client.screen_snapshot()
    assert result.desktop_locked is False
    assert result.window_captured is False
    assert result.degradations == []


async def test_close_window_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.agents.models.screen_snapshot import ActionResult

    client, mock_session = _build_mock_client(monkeypatch)
    _set_tool_return(mock_session, _make_action_result_json())
    result = await client.close_window(window_handle=99999)
    assert isinstance(result, ActionResult)
    mock_session.call_tool.assert_called_once_with("close_window", {"window_handle": 99999})


async def test_get_capability_flags_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_capability_flags 使用实例缓存，不重新调用 server。"""
    client, mock_session = _build_mock_client(monkeypatch)
    # capability_cache 已预填充，不应调用 call_tool
    result = await client.get_capability_flags()
    mock_session.call_tool.assert_not_called()
    assert result["ocr"] is True
    assert result["omniparser"] is False


async def test_get_capability_flags_cache_not_mutated(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_capability_flags 返回副本，修改不影响内部缓存。"""
    client, _ = _build_mock_client(monkeypatch)
    result = await client.get_capability_flags()
    result["ocr"] = False  # 修改返回值
    # 内部缓存不变
    assert client.capability_cache is not None
    assert client.capability_cache["ocr"] is True


# ── 工具调用失败路径 ──────────────────────────────────────────────────────────


async def test_call_tool_raises_on_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """server 返回 isError=True 时 _call_tool 抛 DesktopMCPCallError。"""
    client, mock_session = _build_mock_client(monkeypatch)
    # 构造 isError=True 的返回
    from mcp.types import TextContent

    tc = MagicMock(spec=TextContent)
    tc.text = "内部错误"
    err_result = MagicMock()
    err_result.isError = True
    err_result.content = [tc]
    mock_session.call_tool.return_value = err_result

    with pytest.raises(DesktopMCPCallError) as exc_info:
        await client.screen_snapshot()
    assert "screen_snapshot" in str(exc_info.value)


# ── aexit 清理 ────────────────────────────────────────────────────────────────


async def test_aexit_clears_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """__aexit__ 后 session 和 capability_cache 清为 None。"""
    client, _ = _build_mock_client(monkeypatch)
    # 注入假的 exit_stack
    mock_stack = AsyncMock()
    client.exit_stack = mock_stack

    await client.__aexit__(None, None, None)

    assert client.session is None
    assert client.capability_cache is None
    mock_stack.__aexit__.assert_called_once_with(None, None, None)
