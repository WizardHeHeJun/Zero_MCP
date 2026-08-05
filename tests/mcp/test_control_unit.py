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
- K2 锁屏门：五个写动作入口锁定态全拒 + [desk:desktop_locked] 令牌 re.search 可提取
- K8 前台唤回梯级：逐级核验/梯级按序推进/AttachThreadInput 异常路径必分离/全梯失败令牌
- K7 批2 落点核验：命中放行/不符拒点 + [desk:landing_mismatch]/未设期望零回归/GA_ROOTOWNER 兜弹窗

Win32 纪律：一律 monkeypatch 模块内私有函数（_fl_*/_resolve_click_root/
session_state._is_desktop_locked_sync）注桩，不真调 Win32（CI 无桌面会话）。
autouse fixture 默认注桩「未锁」，锁屏用例内再覆写。

中文 clipboard round-trip 在实机环境标注为 realenv，此文件不含。
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

import src.mcp.desktop.session_state as session_state_mod
import src.mcp.desktop.tools.control as ctrl_mod

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def desktop_unlocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认注桩「桌面未锁」：既避免单测真调 Win32（CI 无桌面会话会被判锁定而

    全线误红），也把 K2 锁屏门的行为收敛为显式覆写才触发。"""
    monkeypatch.setattr(session_state_mod, "_is_desktop_locked_sync", lambda: (False, False))


def _set_locked(monkeypatch: pytest.MonkeyPatch) -> None:
    """把锁屏探测桩覆写为「锁定」。"""
    monkeypatch.setattr(session_state_mod, "_is_desktop_locked_sync", lambda: (True, False))


LOCK_TOKEN = re.compile(r"\[desk:desktop_locked\]")
FOCUS_TOKEN = re.compile(r"\[desk:focus_unverified\]")
LANDING_TOKEN = re.compile(r"\[desk:landing_mismatch\]")

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


# ── do_focus_window（K8 前台唤回梯级） ────────────────────────────────────────


def _patch_ladder(
    monkeypatch: pytest.MonkeyPatch,
    verify_results: list[bool],
    set_fg_raises_on_call: int | None = None,
) -> tuple[list[str], list[bool]]:
    """给 _focus_ladder 的全部 Win32 辅助注桩，返回 (调用序列, attach 布尔序列)。

    verify_results 按核验次序消费；set_fg_raises_on_call=N 时第 N 次
    _fl_set_foreground 抛 RuntimeError（验 finally 分离路径）。
    """
    calls: list[str] = []
    attach_flags: list[bool] = []

    def fake_set_fg(hwnd: int) -> None:
        calls.append("set_fg")
        n = sum(1 for c in calls if c == "set_fg")
        if set_fg_raises_on_call is not None and n == set_fg_raises_on_call:
            raise RuntimeError("SetForegroundWindow boom")

    verify_iter = iter(verify_results)

    monkeypatch.setattr(ctrl_mod, "_fl_set_foreground", fake_set_fg)
    monkeypatch.setattr(ctrl_mod, "_fl_send_shift_up", lambda: calls.append("shift_up"))
    monkeypatch.setattr(ctrl_mod, "_fl_get_foreground", lambda: 0xFFFF)
    monkeypatch.setattr(ctrl_mod, "_fl_thread_ids", lambda fg: (111, 222))

    def fake_attach(tid_self: int, tid_fg: int, attach: bool) -> bool:
        attach_flags.append(attach)
        calls.append(f"attach_{attach}")
        return True

    monkeypatch.setattr(ctrl_mod, "_fl_attach_thread_input", fake_attach)
    monkeypatch.setattr(ctrl_mod, "_fl_show_window", lambda hwnd, cmd: calls.append(f"show_{cmd}"))
    monkeypatch.setattr(ctrl_mod, "_fl_verify", lambda hwnd: next(verify_iter))
    return calls, attach_flags


def test_focus_ladder_rung1_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    """梯级①核验即通过 → (1, True)，不发 SendInput/Attach/ShowWindow。"""
    calls, _ = _patch_ladder(monkeypatch, verify_results=[True])
    assert ctrl_mod._focus_ladder(0x5678) == (1, True)
    assert calls == ["set_fg"]


def test_focus_ladder_escalates_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """①②③连败 → ②SendInput ③Attach（成对）④最小化-恢复按序真被走到，(4, True)。"""
    calls, attach_flags = _patch_ladder(monkeypatch, verify_results=[False, False, False, True])
    assert ctrl_mod._focus_ladder(0x5678) == (4, True)
    assert calls == [
        "set_fg",  # ①
        "shift_up",  # ②
        "set_fg",
        "attach_True",  # ③ 附加
        "set_fg",
        "attach_False",  # ③ finally 分离
        f"show_{ctrl_mod.SW_MINIMIZE}",  # ④ 强恢复
        f"show_{ctrl_mod.SW_RESTORE}",
        "set_fg",
    ]
    assert attach_flags == [True, False]


def test_focus_ladder_all_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """四级全败 → (4, False)。"""
    _patch_ladder(monkeypatch, verify_results=[False, False, False, False])
    assert ctrl_mod._focus_ladder(0x5678) == (4, False)


def test_focus_ladder_attach_detached_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """梯级③内 SetForegroundWindow 抛异常 → AttachThreadInput 仍在 finally 分离。"""
    # 第 3 次 set_fg（=梯级③附加后那次）抛
    _, attach_flags = _patch_ladder(
        monkeypatch, verify_results=[False, False], set_fg_raises_on_call=3
    )
    with pytest.raises(RuntimeError, match="boom"):
        ctrl_mod._focus_ladder(0x5678)
    assert attach_flags == [True, False], "异常路径下 detach（attach=False）必须仍被调用"


async def test_focus_window_success_verified_rung1(monkeypatch: pytest.MonkeyPatch) -> None:
    """梯级①核验通过 → success=True, ui_changed=True, error_message=None。"""
    fake_win32gui = MagicMock()
    fake_win32gui.IsWindow.return_value = True
    monkeypatch.setattr(ctrl_mod, "_focus_ladder", lambda hwnd: (1, True))

    with patch.dict("sys.modules", {"win32gui": fake_win32gui}):
        result = await ctrl_mod.do_focus_window(window_handle=0x5678)

    assert result.success is True
    assert result.ui_changed is True
    assert result.error_message is None


async def test_focus_window_success_higher_rung_reports_rung(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """梯级③核验通过 → success=True，error_message 写明到达梯级。"""
    fake_win32gui = MagicMock()
    fake_win32gui.IsWindow.return_value = True
    monkeypatch.setattr(ctrl_mod, "_focus_ladder", lambda hwnd: (3, True))

    with patch.dict("sys.modules", {"win32gui": fake_win32gui}):
        result = await ctrl_mod.do_focus_window(window_handle=0x5678)

    assert result.success is True
    assert result.error_message is not None
    assert "梯级 3" in result.error_message


async def test_focus_window_all_rungs_fail_unverified_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全梯失败（假前台）→ success=False + [desk:focus_unverified] + 到达梯级。"""
    fake_win32gui = MagicMock()
    fake_win32gui.IsWindow.return_value = True
    monkeypatch.setattr(ctrl_mod, "_focus_ladder", lambda hwnd: (4, False))

    with patch.dict("sys.modules", {"win32gui": fake_win32gui}):
        result = await ctrl_mod.do_focus_window(window_handle=0x5678)

    assert result.success is False
    assert result.ui_changed is False
    assert result.error_message is not None
    assert FOCUS_TOKEN.search(result.error_message), result.error_message
    assert "梯级 4" in result.error_message


async def test_focus_window_ladder_exception_returns_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """梯级抛异常 → success=False，异常信息进 error_message。"""
    fake_win32gui = MagicMock()
    fake_win32gui.IsWindow.return_value = True

    def raise_ladder(hwnd: int) -> tuple[int, bool]:
        raise RuntimeError("ladder boom")

    monkeypatch.setattr(ctrl_mod, "_focus_ladder", raise_ladder)

    with patch.dict("sys.modules", {"win32gui": fake_win32gui}):
        result = await ctrl_mod.do_focus_window(window_handle=0x5678)

    assert result.success is False
    assert result.error_message is not None
    assert "ladder boom" in result.error_message


async def test_focus_window_invalid_hwnd(monkeypatch: pytest.MonkeyPatch) -> None:
    """hwnd 无效 → success=False，不进梯级。"""
    fake_win32gui = MagicMock()
    fake_win32gui.IsWindow.return_value = False

    def fail_ladder(hwnd: int) -> tuple[int, bool]:
        raise AssertionError("hwnd 无效时不得进入梯级")

    monkeypatch.setattr(ctrl_mod, "_focus_ladder", fail_ladder)

    with patch.dict("sys.modules", {"win32gui": fake_win32gui}):
        result = await ctrl_mod.do_focus_window(window_handle=0xBAD)

    assert result.success is False


async def test_focus_window_pin_topmost_paired(monkeypatch: pytest.MonkeyPatch) -> None:
    """pin_topmost=True → HWND_TOPMOST 置顶与撤销成对（含梯级异常路径）。"""
    fake_win32gui = MagicMock()
    fake_win32gui.IsWindow.return_value = True
    pin_calls: list[bool] = []
    monkeypatch.setattr(
        ctrl_mod, "_fl_pin_topmost", lambda hwnd, pin: (pin_calls.append(pin), True)[1]
    )
    monkeypatch.setattr(ctrl_mod, "_focus_ladder", lambda hwnd: (1, True))

    with patch.dict("sys.modules", {"win32gui": fake_win32gui}):
        result = await ctrl_mod.do_focus_window(window_handle=0x5678, pin_topmost=True)

    assert result.success is True
    assert pin_calls == [True, False]

    # 梯级异常路径同样成对撤销
    pin_calls.clear()

    def raise_ladder(hwnd: int) -> tuple[int, bool]:
        raise RuntimeError("boom")

    monkeypatch.setattr(ctrl_mod, "_focus_ladder", raise_ladder)
    with patch.dict("sys.modules", {"win32gui": fake_win32gui}):
        result = await ctrl_mod.do_focus_window(window_handle=0x5678, pin_topmost=True)
    assert result.success is False
    assert pin_calls == [True, False]


# ── K2：锁屏门（五个写动作入口全拒，不发任何输入事件） ────────────────────────


async def test_click_rejected_when_desktop_locked(monkeypatch: pytest.MonkeyPatch) -> None:
    """锁定态 click → success=False + 令牌可提取，pyautogui.click 不被调。"""
    _set_locked(monkeypatch)
    fake_pyautogui = MagicMock()
    with patch.dict("sys.modules", {"pyautogui": fake_pyautogui}):
        result = await ctrl_mod.do_click_element(coordinates=(100, 200), method="coordinate")
    assert result.success is False
    assert result.error_message is not None
    assert LOCK_TOKEN.search(result.error_message), result.error_message
    fake_pyautogui.click.assert_not_called()


async def test_type_text_rejected_when_desktop_locked(monkeypatch: pytest.MonkeyPatch) -> None:
    """锁定态 type_text → 拒绝，pyperclip/pyautogui 不被碰。"""
    _set_locked(monkeypatch)
    fake_pyautogui = MagicMock()
    fake_pyperclip = MagicMock()
    with patch.dict("sys.modules", {"pyautogui": fake_pyautogui, "pyperclip": fake_pyperclip}):
        result = await ctrl_mod.do_type_text(text="hello", method="clipboard")
    assert result.success is False
    assert result.error_message is not None
    assert LOCK_TOKEN.search(result.error_message)
    fake_pyperclip.copy.assert_not_called()
    fake_pyautogui.hotkey.assert_not_called()


async def test_send_key_rejected_when_desktop_locked(monkeypatch: pytest.MonkeyPatch) -> None:
    """锁定态 send_key → 拒绝，press/hotkey 不被调。"""
    _set_locked(monkeypatch)
    fake_pyautogui = MagicMock()
    with patch.dict("sys.modules", {"pyautogui": fake_pyautogui}):
        result = await ctrl_mod.do_send_key("ctrl+c")
    assert result.success is False
    assert result.error_message is not None
    assert LOCK_TOKEN.search(result.error_message)
    fake_pyautogui.press.assert_not_called()
    fake_pyautogui.hotkey.assert_not_called()


async def test_focus_window_rejected_when_desktop_locked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """锁定态 focus_window → 拒绝，不进梯级。"""
    _set_locked(monkeypatch)

    def fail_ladder(hwnd: int) -> tuple[int, bool]:
        raise AssertionError("锁定态不得进入前台唤回梯级")

    monkeypatch.setattr(ctrl_mod, "_focus_ladder", fail_ladder)
    result = await ctrl_mod.do_focus_window(window_handle=0x5678)
    assert result.success is False
    assert result.error_message is not None
    assert LOCK_TOKEN.search(result.error_message)


async def test_close_window_rejected_when_desktop_locked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """锁定态 close_window → 拒绝，WM_CLOSE 不被发送。"""
    _set_locked(monkeypatch)
    fake_win32gui = MagicMock()
    fake_win32api = MagicMock()
    with patch.dict("sys.modules", {"win32gui": fake_win32gui, "win32api": fake_win32api}):
        result = await ctrl_mod.do_close_window(window_handle=0x1234)
    assert result.success is False
    assert result.error_message is not None
    assert LOCK_TOKEN.search(result.error_message)
    fake_win32api.PostMessage.assert_not_called()


async def test_lock_probe_failed_does_not_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """探测失败（probe_failed=True）按未锁放行——不为防护增强新增硬失败面。"""
    monkeypatch.setattr(session_state_mod, "_is_desktop_locked_sync", lambda: (False, True))
    fake_pyautogui = MagicMock()
    with patch.dict("sys.modules", {"pyautogui": fake_pyautogui}):
        result = await ctrl_mod.do_click_element(coordinates=(10, 20), method="coordinate")
    assert result.success is True
    fake_pyautogui.click.assert_called_once_with(10, 20)


# ── K7 批2：坐标落点核验 ──────────────────────────────────────────────────────


async def test_click_landing_match_clicks(monkeypatch: pytest.MonkeyPatch) -> None:
    """落点 GA_ROOT == 期望 → 照常点击。"""
    monkeypatch.setattr(
        ctrl_mod, "_resolve_click_root", lambda x, y, ga_flag=ctrl_mod.GA_ROOT: 0xAAA
    )
    fake_pyautogui = MagicMock()
    with patch.dict("sys.modules", {"pyautogui": fake_pyautogui}):
        result = await ctrl_mod.do_click_element(
            coordinates=(100, 200), method="coordinate", expected_root_hwnd=0xAAA
        )
    assert result.success is True
    fake_pyautogui.click.assert_called_once_with(100, 200)


async def test_click_landing_mismatch_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """落点 = 其他窗口 → 拒绝点击 + [desk:landing_mismatch] + 实际 hwnd/类名可见。"""
    monkeypatch.setattr(
        ctrl_mod, "_resolve_click_root", lambda x, y, ga_flag=ctrl_mod.GA_ROOT: 0x111
    )
    monkeypatch.setattr(ctrl_mod, "_get_class_name", lambda hwnd: "Chrome_WidgetWin_1")
    fake_pyautogui = MagicMock()
    with patch.dict("sys.modules", {"pyautogui": fake_pyautogui}):
        result = await ctrl_mod.do_click_element(
            coordinates=(100, 200), method="coordinate", expected_root_hwnd=0xAAA
        )
    assert result.success is False
    assert result.error_message is not None
    assert LANDING_TOKEN.search(result.error_message), result.error_message
    assert "0x111" in result.error_message
    assert "Chrome_WidgetWin_1" in result.error_message
    fake_pyautogui.click.assert_not_called()


async def test_click_no_expected_hwnd_zero_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    """未提供 expected_root_hwnd → 完全不做落点探测，照常点击（零回归判别用例）。"""

    def fail_resolve(x: int, y: int, ga_flag: int = ctrl_mod.GA_ROOT) -> int | None:
        raise AssertionError("未设期望值时不得做落点探测")

    monkeypatch.setattr(ctrl_mod, "_resolve_click_root", fail_resolve)
    fake_pyautogui = MagicMock()
    with patch.dict("sys.modules", {"pyautogui": fake_pyautogui}):
        result = await ctrl_mod.do_click_element(coordinates=(100, 200), method="coordinate")
    assert result.success is True
    fake_pyautogui.click.assert_called_once_with(100, 200)


async def test_click_landing_rootowner_popup_not_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """有 owner 的主弹窗：GA_ROOT 不中但 GA_ROOTOWNER 归回主窗 → 命中不误拒。"""

    def fake_resolve(x: int, y: int, ga_flag: int = ctrl_mod.GA_ROOT) -> int | None:
        # 弹窗自身是顶层（GA_ROOT=0x111），owner 链归回主窗 0xAAA
        return 0x111 if ga_flag == ctrl_mod.GA_ROOT else 0xAAA

    monkeypatch.setattr(ctrl_mod, "_resolve_click_root", fake_resolve)
    fake_pyautogui = MagicMock()
    with patch.dict("sys.modules", {"pyautogui": fake_pyautogui}):
        result = await ctrl_mod.do_click_element(
            coordinates=(100, 200), method="coordinate", expected_root_hwnd=0xAAA
        )
    assert result.success is True
    fake_pyautogui.click.assert_called_once_with(100, 200)


async def test_click_landing_probe_exception_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """落点探测自身异常 → 宁拒不误点（核验是显式 opt-in）。"""

    def raise_resolve(x: int, y: int, ga_flag: int = ctrl_mod.GA_ROOT) -> int | None:
        raise OSError("WindowFromPoint boom")

    monkeypatch.setattr(ctrl_mod, "_resolve_click_root", raise_resolve)
    fake_pyautogui = MagicMock()
    with patch.dict("sys.modules", {"pyautogui": fake_pyautogui}):
        result = await ctrl_mod.do_click_element(
            coordinates=(100, 200), method="coordinate", expected_root_hwnd=0xAAA
        )
    assert result.success is False
    assert result.error_message is not None
    assert LANDING_TOKEN.search(result.error_message)
    fake_pyautogui.click.assert_not_called()


async def test_click_downgrade_chain_also_verifies_landing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """uia 降级到 coordinate 的兜底路径同样过落点核验（不符拒点）。"""
    monkeypatch.setattr(
        ctrl_mod, "_resolve_click_root", lambda x, y, ga_flag=ctrl_mod.GA_ROOT: 0x111
    )
    monkeypatch.setattr(ctrl_mod, "_get_class_name", lambda hwnd: "Other")
    fake_pyautogui = MagicMock()
    with patch.object(ctrl_mod, "_uia_invoke", return_value=(False, "invoke fail")):
        with patch.object(ctrl_mod, "_uia_click", return_value=(False, "click fail")):
            with patch.dict("sys.modules", {"pyautogui": fake_pyautogui}):
                result = await ctrl_mod.do_click_element(
                    automation_id="btn_ok",
                    coordinates=(10, 20),
                    method="uia_invoke",
                    expected_root_hwnd=0xAAA,
                )
    assert result.success is False
    assert result.error_message is not None
    assert LANDING_TOKEN.search(result.error_message)
    fake_pyautogui.click.assert_not_called()
