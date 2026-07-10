"""test_control_unit.py — control.py 单元测试（mock 不走真实 win32/pyautogui）。

覆盖：
- clipboard 输入：pyperclip.copy + pyautogui.hotkey 调用顺序正确
- key_events 输入：pyautogui.typewrite 被调用
- 未知 method → ActionResult.success=False
- 降级链：uia_invoke 失败 → uia_click 失败 → coordinate
- uia_invoke 成功 → 直接返回 success=True
- uia_click element_id=None → 直接降级 coordinate
- do_window_list：mock EnumWindows 返回结构正确
- do_close_window：mock IsWindow=True → PostMessage → 等待 → IsWindow=False → ui_changed=True
- do_close_window：hwnd 已无效 → success=False
- do_send_key：单键 press / 多键 hotkey

中文 clipboard round-trip 在实机环境标注为 realenv，此文件不含。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import src.mcp.desktop.tools.control as ctrl_mod

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_win32gui(
    is_window: bool = True,
    is_iconic: bool = False,
    is_visible: bool = True,
    window_text: str = "TestWindow",
    class_name: str = "TestClass",
    rect: tuple[int, int, int, int] = (0, 0, 800, 600),
) -> MagicMock:
    w = MagicMock()
    w.IsWindow.return_value = is_window
    w.IsIconic.return_value = is_iconic
    w.IsWindowVisible.return_value = is_visible
    w.GetWindowText.return_value = window_text
    w.GetClassName.return_value = class_name
    w.GetWindowRect.return_value = rect
    return w


# ── do_type_text ──────────────────────────────────────────────────────────────


async def test_type_text_clipboard_calls_pyperclip_then_paste() -> None:
    """clipboard 模式：先 pyperclip.copy，再 pyautogui.hotkey('ctrl','v')。"""
    fake_pyperclip = MagicMock()
    fake_pyautogui = MagicMock()

    call_order: list[str] = []
    fake_pyperclip.copy.side_effect = lambda t: call_order.append("copy")
    fake_pyautogui.hotkey.side_effect = lambda *a: call_order.append("paste")

    with patch.dict(
        "sys.modules",
        {"pyperclip": fake_pyperclip, "pyautogui": fake_pyautogui},
    ):
        result = await ctrl_mod.do_type_text(text="hello", method="clipboard")

    assert result.success is True
    assert result.error_message is None
    fake_pyperclip.copy.assert_called_once_with("hello")
    fake_pyautogui.hotkey.assert_called_once_with("ctrl", "v")
    assert call_order == ["copy", "paste"], f"调用顺序错误：{call_order}"


async def test_type_text_key_events_calls_typewrite() -> None:
    """key_events 模式：调用 pyautogui.typewrite。"""
    fake_pyautogui = MagicMock()
    fake_pyperclip = MagicMock()

    with patch.dict(
        "sys.modules",
        {"pyautogui": fake_pyautogui, "pyperclip": fake_pyperclip},
    ):
        result = await ctrl_mod.do_type_text(text="hello", method="key_events")

    assert result.success is True
    fake_pyautogui.typewrite.assert_called_once_with("hello", interval=0.02)
    fake_pyperclip.copy.assert_not_called()


async def test_type_text_unknown_method_returns_failure() -> None:
    """未知 method → success=False，错误信息含 method 名。"""
    result = await ctrl_mod.do_type_text(text="x", method="magic")
    assert result.success is False
    assert result.error_message is not None
    assert "magic" in result.error_message


async def test_type_text_clipboard_exception_returns_failure() -> None:
    """clipboard 调用抛异常 → success=False。"""
    fake_pyperclip = MagicMock()
    fake_pyperclip.copy.side_effect = RuntimeError("clipboard error")
    fake_pyautogui = MagicMock()

    with patch.dict(
        "sys.modules",
        {"pyperclip": fake_pyperclip, "pyautogui": fake_pyautogui},
    ):
        result = await ctrl_mod.do_type_text(text="fail", method="clipboard")

    assert result.success is False
    assert result.error_message is not None
    assert "clipboard" in result.error_message.lower()


# ── do_send_key ───────────────────────────────────────────────────────────────


async def test_send_key_single_key_calls_press() -> None:
    """单键（无 +）调用 pyautogui.press。"""
    fake_pyautogui = MagicMock()
    with patch.dict("sys.modules", {"pyautogui": fake_pyautogui}):
        result = await ctrl_mod.do_send_key("enter")
    assert result.success is True
    fake_pyautogui.press.assert_called_once_with("enter")
    fake_pyautogui.hotkey.assert_not_called()


async def test_send_key_combo_calls_hotkey() -> None:
    """组合键（含 +）调用 pyautogui.hotkey。"""
    fake_pyautogui = MagicMock()
    with patch.dict("sys.modules", {"pyautogui": fake_pyautogui}):
        result = await ctrl_mod.do_send_key("ctrl+c")
    assert result.success is True
    fake_pyautogui.hotkey.assert_called_once_with("ctrl", "c")
    fake_pyautogui.press.assert_not_called()


async def test_send_key_three_part_combo() -> None:
    """三键组合正确拆分。"""
    fake_pyautogui = MagicMock()
    with patch.dict("sys.modules", {"pyautogui": fake_pyautogui}):
        result = await ctrl_mod.do_send_key("ctrl+shift+esc")
    assert result.success is True
    fake_pyautogui.hotkey.assert_called_once_with("ctrl", "shift", "esc")


# ── do_click_element 降级链 ───────────────────────────────────────────────────


async def test_click_coordinate_mode_calls_pyautogui() -> None:
    """coordinate 模式直接调 pyautogui.click(x, y)。"""
    fake_pyautogui = MagicMock()
    with patch.dict("sys.modules", {"pyautogui": fake_pyautogui}):
        result = await ctrl_mod.do_click_element(coordinates=(100, 200), method="coordinate")
    assert result.success is True
    fake_pyautogui.click.assert_called_once_with(100, 200)


async def test_click_coordinate_mode_no_coords_fails() -> None:
    """coordinate 模式无坐标 → success=False。"""
    result = await ctrl_mod.do_click_element(coordinates=None, method="coordinate")
    assert result.success is False
    assert result.error_message is not None


async def test_click_uia_invoke_success() -> None:
    """uia_invoke 成功 → success=True，不降级。"""
    with patch.object(ctrl_mod, "_uia_invoke", return_value=(True, None)) as mock_invoke:
        with patch.object(ctrl_mod, "_uia_click") as mock_click:
            result = await ctrl_mod.do_click_element(automation_id="btn_ok", method="uia_invoke")
    assert result.success is True
    mock_invoke.assert_called_once_with("btn_ok")
    mock_click.assert_not_called()


async def test_click_uia_invoke_fails_downgrades_to_uia_click() -> None:
    """uia_invoke 失败 → 降级 uia_click 成功 → success=True，error_message 含降级信息。"""
    with patch.object(ctrl_mod, "_uia_invoke", return_value=(False, "invoke error")):
        with patch.object(ctrl_mod, "_uia_click", return_value=(True, None)):
            result = await ctrl_mod.do_click_element(
                automation_id="btn_ok",
                coordinates=(50, 100),
                method="uia_invoke",
            )
    assert result.success is True
    assert result.error_message is not None
    assert "uia_click" in result.error_message


async def test_click_uia_invoke_and_click_both_fail_downgrades_to_coordinate() -> None:
    """uia_invoke + uia_click 均失败 → 降级 coordinate → success=True。"""
    fake_pyautogui = MagicMock()
    with patch.object(ctrl_mod, "_uia_invoke", return_value=(False, "invoke fail")):
        with patch.object(ctrl_mod, "_uia_click", return_value=(False, "click fail")):
            with patch.dict("sys.modules", {"pyautogui": fake_pyautogui}):
                result = await ctrl_mod.do_click_element(
                    automation_id="btn_ok",
                    coordinates=(10, 20),
                    method="uia_invoke",
                )
    assert result.success is True
    fake_pyautogui.click.assert_called_once_with(10, 20)
    assert result.error_message is not None
    assert "coordinate" in result.error_message


async def test_click_uia_invoke_all_fail_no_coords() -> None:
    """uia_invoke + uia_click 均失败且无坐标 → success=False。"""
    with patch.object(ctrl_mod, "_uia_invoke", return_value=(False, "invoke fail")):
        with patch.object(ctrl_mod, "_uia_click", return_value=(False, "click fail")):
            result = await ctrl_mod.do_click_element(
                automation_id="btn_ok",
                coordinates=None,
                method="uia_invoke",
            )
    assert result.success is False


async def test_click_uia_click_automation_id_none_downgrades_coordinate() -> None:
    """uia_click automation_id=None → 直接降级 coordinate。"""
    fake_pyautogui = MagicMock()
    with patch.dict("sys.modules", {"pyautogui": fake_pyautogui}):
        result = await ctrl_mod.do_click_element(
            automation_id=None,
            coordinates=(30, 40),
            method="uia_click",
        )
    assert result.success is True
    fake_pyautogui.click.assert_called_once_with(30, 40)


async def test_click_unknown_method_returns_failure() -> None:
    """未知 method → success=False。"""
    result = await ctrl_mod.do_click_element(method="teleport")
    assert result.success is False
    assert result.error_message is not None
    assert "teleport" in result.error_message


# ── do_window_list ────────────────────────────────────────────────────────────


async def test_window_list_returns_visible_windows() -> None:
    """mock EnumWindows 返回包含标题的可见窗口结构正确。"""

    def fake_enum_windows(callback, lparam):  # type: ignore[no-untyped-def]
        # 模拟两个可见窗口
        callback(1001, 0)
        callback(1002, 0)

    fake_win32gui = MagicMock()
    fake_win32gui.IsWindowVisible.side_effect = lambda hwnd: True
    fake_win32gui.GetWindowText.side_effect = lambda hwnd: {
        1001: "Notepad",
        1002: "Calculator",
    }[hwnd]
    fake_win32gui.GetClassName.side_effect = lambda hwnd: {
        1001: "Notepad",
        1002: "CalcFrame",
    }[hwnd]
    fake_win32gui.GetWindowRect.side_effect = lambda hwnd: (0, 0, 800, 600)
    fake_win32gui.EnumWindows.side_effect = fake_enum_windows

    with patch.dict("sys.modules", {"win32gui": fake_win32gui}):
        result = await ctrl_mod.do_window_list()

    assert len(result) == 2
    titles = {w["title"] for w in result}
    assert titles == {"Notepad", "Calculator"}
    # 验证结构字段
    w0 = result[0]
    assert "hwnd" in w0
    assert "title" in w0
    assert "class_name" in w0
    assert "visible" in w0
    assert "rect" in w0
    assert w0["visible"] is True
    assert w0["rect"]["left"] == 0


async def test_window_list_filters_invisible() -> None:
    """不可见窗口被过滤，仅返回 IsWindowVisible=True 的窗口。"""

    def fake_enum_windows(callback, lparam):  # type: ignore[no-untyped-def]
        callback(2001, 0)  # visible
        callback(2002, 0)  # invisible

    fake_win32gui = MagicMock()
    fake_win32gui.IsWindowVisible.side_effect = lambda hwnd: hwnd == 2001
    fake_win32gui.GetWindowText.return_value = "SomeTitle"
    fake_win32gui.GetClassName.return_value = "SomeClass"
    fake_win32gui.GetWindowRect.return_value = (0, 0, 100, 100)
    fake_win32gui.EnumWindows.side_effect = fake_enum_windows

    with patch.dict("sys.modules", {"win32gui": fake_win32gui}):
        result = await ctrl_mod.do_window_list()

    assert len(result) == 1
    assert result[0]["hwnd"] == 2001


async def test_window_list_filters_empty_title() -> None:
    """无标题窗口（空字符串）被过滤。"""

    def fake_enum_windows(callback, lparam):  # type: ignore[no-untyped-def]
        callback(3001, 0)  # empty title
        callback(3002, 0)  # has title

    fake_win32gui = MagicMock()
    fake_win32gui.IsWindowVisible.return_value = True
    fake_win32gui.GetWindowText.side_effect = lambda hwnd: "" if hwnd == 3001 else "Valid"
    fake_win32gui.GetClassName.return_value = "Cls"
    fake_win32gui.GetWindowRect.return_value = (0, 0, 100, 100)
    fake_win32gui.EnumWindows.side_effect = fake_enum_windows

    with patch.dict("sys.modules", {"win32gui": fake_win32gui}):
        result = await ctrl_mod.do_window_list()

    assert len(result) == 1
    assert result[0]["hwnd"] == 3002


# ── do_close_window ───────────────────────────────────────────────────────────


async def test_close_window_success_ui_changed() -> None:
    """PostMessage 后窗口销毁（IsWindow=False）→ ui_changed=True, success=True。"""
    call_count = 0

    def is_window_side_effect(hwnd: int) -> bool:
        nonlocal call_count
        call_count += 1
        # 第一次（_do_close 开头校验）= True（窗口存在）
        # 第二次（500ms 后检查）= False（已销毁）
        return call_count == 1

    fake_win32gui = MagicMock()
    fake_win32gui.IsWindow.side_effect = is_window_side_effect

    fake_win32api = MagicMock()

    with patch.dict(
        "sys.modules",
        {"win32gui": fake_win32gui, "win32api": fake_win32api},
    ):
        with patch("time.sleep"):  # 跳过真实 sleep
            result = await ctrl_mod.do_close_window(window_handle=0xABCD)

    assert result.success is True
    assert result.ui_changed is True
    assert result.error_message is None
    fake_win32api.PostMessage.assert_called_once()


async def test_close_window_window_still_alive() -> None:
    """PostMessage 后窗口仍存在 → ui_changed=False, success=True，error_message 非空。"""
    fake_win32gui = MagicMock()
    fake_win32gui.IsWindow.return_value = True  # 始终存在

    fake_win32api = MagicMock()

    with patch.dict(
        "sys.modules",
        {"win32gui": fake_win32gui, "win32api": fake_win32api},
    ):
        with patch("time.sleep"):
            result = await ctrl_mod.do_close_window(window_handle=0x1234)

    assert result.success is True
    assert result.ui_changed is False
    assert result.error_message is not None
    assert "仍存在" in result.error_message


async def test_close_window_invalid_hwnd() -> None:
    """hwnd 无效（IsWindow=False）→ success=False，不发送 PostMessage。"""
    fake_win32gui = MagicMock()
    fake_win32gui.IsWindow.return_value = False

    fake_win32api = MagicMock()

    with patch.dict(
        "sys.modules",
        {"win32gui": fake_win32gui, "win32api": fake_win32api},
    ):
        result = await ctrl_mod.do_close_window(window_handle=0xDEAD)

    assert result.success is False
    assert result.error_message is not None
    fake_win32api.PostMessage.assert_not_called()


# ── do_focus_window ───────────────────────────────────────────────────────────


async def test_focus_window_success() -> None:
    """有效 hwnd → SetForegroundWindow 调用 → ui_changed=True, success=True。"""
    fake_win32gui = MagicMock()
    fake_win32gui.IsWindow.return_value = True
    fake_win32gui.IsIconic.return_value = False

    with patch.dict("sys.modules", {"win32gui": fake_win32gui}):
        result = await ctrl_mod.do_focus_window(window_handle=0x5678)

    assert result.success is True
    assert result.ui_changed is True
    fake_win32gui.SetForegroundWindow.assert_called_once_with(0x5678)


async def test_focus_window_invalid_hwnd() -> None:
    """hwnd 无效 → success=False。"""
    fake_win32gui = MagicMock()
    fake_win32gui.IsWindow.return_value = False

    with patch.dict("sys.modules", {"win32gui": fake_win32gui}):
        result = await ctrl_mod.do_focus_window(window_handle=0xBAD)

    assert result.success is False
    fake_win32gui.SetForegroundWindow.assert_not_called()
