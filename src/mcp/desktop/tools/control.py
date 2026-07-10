"""桌面操控原语实现。

本文件是 pyautogui / pyperclip / win32api / win32gui / uiautomation（invoke/click 降级）
等操控库 import 的**唯一出现处**（红线），感知库 import 只在 perception.py。

全部阻塞 I/O / 系统调用用 asyncio.to_thread 包装（python-code.md async 规范）。

公开函数：
  do_click_element  — 点击元素，降级链：uia_invoke → uia_click → coordinate
  do_type_text      — 文字输入，默认 clipboard（支持中文）；key_events 仅 ASCII
  do_send_key       — 键盘组合键
  do_window_list    — EnumWindows 枚举可见顶层窗口
  do_focus_window   — 将指定窗口置前台
  do_close_window   — PostMessage WM_CLOSE + 500ms 验证

已知局限（O6）：clipboard 方案会临时覆盖系统剪贴板内容，并发写时有竞争风险；
这是中文输入的必要代价（pyautogui issue #237），不作为 bug 修复。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agents.models.screen_snapshot import ActionResult

logger = logging.getLogger(__name__)

# ── Win32 常量 ────────────────────────────────────────────────────────────────

WM_CLOSE = 0x0010


# ── 内部 Win32 辅助（不 import Zero，不共享感知库）─────────────────────────────


def _is_window_visible(hwnd: int) -> bool:
    """检查窗口是否可见。"""
    import win32gui  # noqa: PLC0415

    return bool(win32gui.IsWindowVisible(hwnd))


def _get_window_rect(hwnd: int) -> tuple[int, int, int, int]:
    """获取窗口矩形（left, top, right, bottom）。"""
    import win32gui  # noqa: PLC0415

    return win32gui.GetWindowRect(hwnd)


def _get_window_text(hwnd: int) -> str:
    """获取窗口标题。"""
    import win32gui  # noqa: PLC0415

    return win32gui.GetWindowText(hwnd)


def _get_class_name(hwnd: int) -> str:
    """获取窗口类名。"""
    import win32gui  # noqa: PLC0415

    # win32gui stub 声明返回 str | None，实践中只会返回 str（失败抛异常）
    return win32gui.GetClassName(hwnd)  # type: ignore[return-value]


def _is_window(hwnd: int) -> bool:
    """检查句柄是否仍为有效窗口。"""
    import win32gui  # noqa: PLC0415

    return bool(win32gui.IsWindow(hwnd))


def _enum_windows_impl() -> list[dict[str, Any]]:
    """EnumWindows 枚举所有可见顶层窗口。

    复用 test_uia_coverage.py 的 Win32 辅助模式，改写为独立实现，
    不 import 该测试文件（poc 脚本不是业务依赖）。

    返回每项：hwnd, title, class_name, visible, rect（left,top,right,bottom）。
    """
    import win32gui  # noqa: PLC0415

    results: list[dict[str, Any]] = []

    def enum_cb(hwnd: int, _lparam: int) -> bool:
        # 只保留可见且有标题的顶层窗口
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return True
        class_name = win32gui.GetClassName(hwnd)
        try:
            rect = win32gui.GetWindowRect(hwnd)
        except Exception:  # noqa: BLE001
            rect = (0, 0, 0, 0)
        results.append(
            {
                "hwnd": hwnd,
                "title": title,
                "class_name": class_name,
                "visible": True,
                "rect": {
                    "left": rect[0],
                    "top": rect[1],
                    "right": rect[2],
                    "bottom": rect[3],
                },
            }
        )
        return True

    win32gui.EnumWindows(enum_cb, 0)
    return results


def _post_message_close(hwnd: int) -> None:
    """向窗口发送 WM_CLOSE 消息（PostMessage 非阻塞）。"""
    import win32api  # noqa: PLC0415

    win32api.PostMessage(hwnd, WM_CLOSE, 0, 0)


def _set_foreground_window(hwnd: int) -> None:
    """将窗口置前台。"""
    import win32gui  # noqa: PLC0415

    # ShowWindow 先确保不最小化，再 SetForegroundWindow
    SW_RESTORE = 9
    if win32gui.IsIconic(hwnd):
        win32gui.ShowWindow(hwnd, SW_RESTORE)
    win32gui.SetForegroundWindow(hwnd)


def _uia_find_element(automation_id: str) -> Any | None:
    """通过 Windows AutomationId 在 UIA 树中查找元素。

    重要语义说明：
      - 此函数接受的是真正的 Windows AutomationId 字符串（即 UIAElement.automation_id 字段），
        不是本地 snapshot 内的位置索引（UIAElement.element_id = "uia_{hwnd}_{i}"）。
      - UIAElement.element_id 是本地索引键，永远不应传入此函数。
      - caller 应传 UIAElement.automation_id；若该字段为 None 或空字符串，
        则直接调用方应跳过 UIA 路径并降级到坐标点击。

    使用 auto.Control(AutomationId=...) 构造器方式搜索（现场核验的正确 API）。
    返回 uiautomation Control 对象（已 Exists() 确认）或 None。
    """
    if not automation_id:
        # AutomationId 为空（微信等 mmui 应用 AutomationId 为空字符串），无法用 UIA 查找
        logger.warning("uia_find_element: automation_id 为空，无法查找（请降级到坐标模式）")
        return None
    try:
        import uiautomation as auto  # noqa: PLC0415

        ctrl = auto.Control(searchDepth=10, AutomationId=automation_id)
        if ctrl is None or not ctrl.Exists(0):
            return None
        return ctrl
    except Exception as exc:
        logger.debug("uia_find_element(automation_id=%r) 失败：%s", automation_id, exc)
        return None


def _uia_invoke(automation_id: str) -> tuple[bool, str | None]:
    """尝试 UIA Invoke 模式点击，返回 (success, error_message)。

    Args:
        automation_id: Windows AutomationId（UIAElement.automation_id 字段），
                       不是本地索引 element_id。
    """
    ctrl = _uia_find_element(automation_id)
    if ctrl is None:
        return False, f"UIA 未找到元素 automation_id={automation_id!r}"
    try:
        import uiautomation as auto  # noqa: PLC0415

        # PatternId.InvokePattern（现场核验：属性名为 InvokePattern，不是 InvokePatternId）
        pattern = ctrl.GetPattern(auto.PatternId.InvokePattern)
        if pattern is None:
            return False, f"元素不支持 InvokePattern（automation_id={automation_id!r}）"
        pattern.Invoke()
        return True, None
    except Exception as exc:
        return False, f"UIA Invoke 失败：{exc}"


def _uia_click(automation_id: str) -> tuple[bool, str | None]:
    """尝试 UIA 元素点击（Click()），返回 (success, error_message)。

    Args:
        automation_id: Windows AutomationId（UIAElement.automation_id 字段），
                       不是本地索引 element_id。
    """
    ctrl = _uia_find_element(automation_id)
    if ctrl is None:
        return False, f"UIA 未找到元素 automation_id={automation_id!r}"
    try:
        ctrl.Click()
        return True, None
    except Exception as exc:
        return False, f"UIA Click 失败：{exc}"


def _coordinate_click(x: int, y: int) -> None:
    """pyautogui 坐标点击（阻塞，在 asyncio.to_thread 内调用）。"""
    import pyautogui  # noqa: PLC0415

    pyautogui.click(x, y)


def _clipboard_type(text: str) -> None:
    """clipboard 方案输入文字：pyperclip.copy → ctrl+v → sleep 50ms。

    已知局限（O6）：临时覆盖系统剪贴板；pyautogui 原生 typewrite 对中文无效，
    必须走剪贴板方案。参见 pyautogui issue #237。
    """
    import pyautogui  # noqa: PLC0415
    import pyperclip  # noqa: PLC0415

    pyperclip.copy(text)
    pyautogui.hotkey("ctrl", "v")
    # 等待粘贴生效（50ms，实测足够）
    time.sleep(0.05)


def _key_events_type(text: str) -> None:
    """key_events 方案输入（仅 ASCII 安全；中文字符会静默丢失或乱码）。"""
    import pyautogui  # noqa: PLC0415

    pyautogui.typewrite(text, interval=0.02)


def _send_hotkey(key_combo: str) -> None:
    """发送键盘组合键。key_combo 格式如 'ctrl+c'、'alt+F4'、'enter'。

    pyautogui.hotkey 接受多个 key 字符串；单 key 直接 press。
    """
    import pyautogui  # noqa: PLC0415

    parts = [k.strip() for k in key_combo.split("+") if k.strip()]
    if len(parts) == 1:
        pyautogui.press(parts[0])
    else:
        pyautogui.hotkey(*parts)


# ── ActionResult 构造辅助 ─────────────────────────────────────────────────────


def _make_result(
    success: bool,
    error_message: str | None = None,
    ui_changed: bool = False,
) -> ActionResult:
    """构造 ActionResult（延迟 import 契约模型，避免循环依赖风险）。"""
    from src.agents.models.screen_snapshot import ActionResult  # noqa: PLC0415

    return ActionResult(
        action_id=str(uuid.uuid4()),
        success=success,
        error_message=error_message,
        ui_changed=ui_changed,
    )


# ── 公开 async 接口 ───────────────────────────────────────────────────────────


async def do_click_element(
    coordinates: tuple[int, int] | None = None,
    automation_id: str | None = None,
    method: str = "coordinate",
) -> ActionResult:
    """点击元素，降级链：uia_invoke → uia_click → coordinate。

    method="uia_invoke" 且 automation_id 非 None/空：
      1. 尝试 uia_invoke；失败 → 降级 uia_click。
      2. uia_click 失败 → 降级 coordinate（若有 coordinates）。
      降级原因写入 ActionResult.error_message（追加）。

    method="uia_click" 且 automation_id 非 None/空：
      直接 uia_click；失败 → 降级 coordinate。

    method="coordinate"（默认）：
      直接 pyautogui.click(coordinates)。

    Args:
        coordinates: 目标物理像素坐标 (x, y)，坐标模式或降级兜底时使用。
        automation_id: Windows AutomationId（即 UIAElement.automation_id 字段），
                       不是本地 snapshot 内的位置索引（UIAElement.element_id）。
                       微信等 mmui 应用 AutomationId 为空字符串，此时 UIA 路径
                       自动降级到 coordinate 模式。
        method: 点击方式，"coordinate"（默认）| "uia_invoke" | "uia_click"。

    Returns:
        ActionResult，ui_changed 恒为 False（点击后状态由感知层感知）。
    """
    degradation_notes: list[str] = []

    def _coord_click() -> ActionResult:
        """坐标兜底点击。"""
        if coordinates is None:
            return _make_result(
                success=False,
                error_message=(
                    "无法降级到 coordinate 模式：coordinates 未提供。"
                    + ("降级链：" + " → ".join(degradation_notes) if degradation_notes else "")
                ),
            )
        try:
            _coordinate_click(coordinates[0], coordinates[1])
            msg = None
            if degradation_notes:
                msg = "已降级到 coordinate 模式。降级链：" + " → ".join(degradation_notes)
            return _make_result(success=True, error_message=msg, ui_changed=False)
        except Exception as exc:
            full_msg = f"coordinate 点击失败：{exc}"
            if degradation_notes:
                full_msg += "；降级链：" + " → ".join(degradation_notes)
            return _make_result(success=False, error_message=full_msg)

    if method == "coordinate":
        # 纯坐标模式，不走 UIA
        if coordinates is None:
            return _make_result(
                success=False,
                error_message="coordinate 模式需要提供 coordinates 参数",
            )
        return await asyncio.to_thread(_coord_click)

    if method == "uia_invoke":
        if not automation_id:
            # automation_id 为 None 或空字符串（微信等 mmui 应用）→ 直接降级坐标
            degradation_notes.append("uia_invoke：automation_id 为空，跳过 UIA 路径")
            return await asyncio.to_thread(_coord_click)

        ok, err = await asyncio.to_thread(_uia_invoke, automation_id)
        if ok:
            return _make_result(success=True, ui_changed=False)

        # uia_invoke 失败 → 降级 uia_click
        degradation_notes.append(f"uia_invoke 失败：{err}")
        ok2, err2 = await asyncio.to_thread(_uia_click, automation_id)
        if ok2:
            return _make_result(
                success=True,
                error_message="已降级到 uia_click。降级链：" + " → ".join(degradation_notes),
                ui_changed=False,
            )

        # uia_click 失败 → 降级 coordinate
        degradation_notes.append(f"uia_click 失败：{err2}")
        return await asyncio.to_thread(_coord_click)

    if method == "uia_click":
        if not automation_id:
            degradation_notes.append("uia_click：automation_id 为空，跳过 UIA 路径")
            return await asyncio.to_thread(_coord_click)

        ok, err = await asyncio.to_thread(_uia_click, automation_id)
        if ok:
            return _make_result(success=True, ui_changed=False)

        # uia_click 失败 → 降级 coordinate
        degradation_notes.append(f"uia_click 失败：{err}")
        return await asyncio.to_thread(_coord_click)

    return _make_result(
        success=False,
        error_message=f"未知 method：{method!r}，支持 coordinate | uia_invoke | uia_click",
    )


async def do_type_text(
    text: str,
    method: str = "clipboard",
) -> ActionResult:
    """向当前焦点输入文字。

    method="clipboard"（默认）：
      pyperclip.copy(text) → Ctrl+V → sleep 50ms。
      支持中文、全角符号等任意 Unicode 字符。
      已知局限（O6）：操作期间临时占用系统剪贴板，并发写时有竞争风险。

    method="key_events"：
      pyautogui.typewrite(text, interval=0.02)。
      仅 ASCII 安全；中文/特殊符号会静默丢失或乱码，不建议使用。

    Args:
        text: 要输入的文字内容。
        method: "clipboard"（默认）| "key_events"。

    Returns:
        ActionResult。
    """
    if method == "clipboard":
        try:
            await asyncio.to_thread(_clipboard_type, text)
            return _make_result(success=True, ui_changed=False)
        except Exception as exc:
            return _make_result(
                success=False,
                error_message=f"clipboard 输入失败：{exc}",
            )

    if method == "key_events":
        try:
            await asyncio.to_thread(_key_events_type, text)
            return _make_result(success=True, ui_changed=False)
        except Exception as exc:
            return _make_result(
                success=False,
                error_message=f"key_events 输入失败：{exc}",
            )

    return _make_result(
        success=False,
        error_message=f"未知 method：{method!r}，支持 clipboard | key_events",
    )


async def do_send_key(key_combo: str) -> ActionResult:
    """发送键盘组合键（如 'ctrl+c'、'alt+F4'、'enter'）。

    Args:
        key_combo: 组合键字符串，多键用 '+' 分隔（如 'ctrl+shift+esc'），
                   单键直接写（如 'enter'、'escape'）。

    Returns:
        ActionResult。
    """
    try:
        await asyncio.to_thread(_send_hotkey, key_combo)
        return _make_result(success=True, ui_changed=False)
    except Exception as exc:
        return _make_result(
            success=False,
            error_message=f"发送快捷键失败（key_combo={key_combo!r}）：{exc}",
        )


async def do_window_list() -> list[dict[str, Any]]:
    """枚举当前桌面所有可见顶层窗口（EnumWindows）。

    利用 win32gui.EnumWindows 遍历，过滤条件：
    - IsWindowVisible = True
    - GetWindowText 非空

    Returns:
        窗口信息列表，每项含：
        - hwnd: int（窗口句柄）
        - title: str（窗口标题）
        - class_name: str（窗口类名）
        - visible: bool（恒 True，已过滤）
        - rect: dict{left, top, right, bottom}（物理像素）
    """
    return await asyncio.to_thread(_enum_windows_impl)


async def do_focus_window(window_handle: int) -> ActionResult:
    """将指定窗口置前台并获取焦点。

    先 ShowWindow（SW_RESTORE，防最小化/隐藏），再 SetForegroundWindow。

    Args:
        window_handle: 目标窗口 HWND 句柄（整数）。

    Returns:
        ActionResult，ui_changed=True 表示窗口状态已改变（前台）。
    """

    def _do_focus() -> ActionResult:
        if not _is_window(window_handle):
            return _make_result(
                success=False,
                error_message=f"窗口句柄 {window_handle:#x} 无效或已销毁",
            )
        try:
            _set_foreground_window(window_handle)
            return _make_result(success=True, ui_changed=True)
        except Exception as exc:
            return _make_result(
                success=False,
                error_message=f"SetForegroundWindow({window_handle:#x}) 失败：{exc}",
            )

    return await asyncio.to_thread(_do_focus)


async def do_close_window(window_handle: int) -> ActionResult:
    """向指定窗口发送 WM_CLOSE，等待 500ms 后检测是否销毁。

    使用 PostMessage（非阻塞）而非 SendMessage，避免等待消息处理超时。
    500ms 后检查 IsWindow：
    - False → 窗口已销毁，ui_changed=True
    - True  → 窗口仍存在（可能弹出保存确认框等），ui_changed=False

    Args:
        window_handle: 目标窗口 HWND 句柄（整数）。

    Returns:
        ActionResult，ui_changed 反映实际销毁状态。
    """

    def _do_close() -> ActionResult:
        if not _is_window(window_handle):
            return _make_result(
                success=False,
                error_message=f"窗口句柄 {window_handle:#x} 无效或已销毁",
            )
        try:
            _post_message_close(window_handle)
        except Exception as exc:
            return _make_result(
                success=False,
                error_message=f"PostMessage WM_CLOSE({window_handle:#x}) 失败：{exc}",
            )

        # 等待 500ms 后验证是否销毁
        time.sleep(0.5)
        still_alive = _is_window(window_handle)
        ui_changed = not still_alive

        if ui_changed:
            logger.debug("close_window：hwnd=%#x 已销毁", window_handle)
        else:
            logger.debug(
                "close_window：hwnd=%#x 仍存在（可能弹出确认框）",
                window_handle,
            )

        return _make_result(
            success=True,
            error_message=(
                None
                if ui_changed
                else (
                    f"WM_CLOSE 已发送但窗口 {window_handle:#x} 仍存在（500ms 后），可能等待用户确认"
                )
            ),
            ui_changed=ui_changed,
        )

    return await asyncio.to_thread(_do_close)
