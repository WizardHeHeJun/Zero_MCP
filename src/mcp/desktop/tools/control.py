"""桌面操控原语实现。

本文件是 pyautogui / pyperclip / win32api / win32gui / uiautomation（invoke/click 降级）
等操控库 import 的**唯一出现处**（红线），感知库 import 只在 perception.py。

全部阻塞 I/O / 系统调用用 asyncio.to_thread 包装（python-code.md async 规范）。

公开函数：
  do_click_element  — 点击元素，降级链：uia_invoke → uia_click → coordinate；
                      K7 批2：expected_root_hwnd 落点核验（[desk:landing_mismatch]）
  do_type_text      — 文字输入，默认 clipboard（支持中文）；key_events 仅 ASCII
  do_send_key       — 键盘组合键
  do_window_list    — EnumWindows 枚举可见顶层窗口
  do_focus_window   — K8 前台唤回梯级 + Win32 级核验（[desk:focus_unverified]）
  do_close_window   — PostMessage WM_CLOSE + 500ms 验证

桌面加固（feat/desktop-hardening）：全部写动作入口前置锁屏检测（K2），锁定时
success=False + [desk:desktop_locked] 机读令牌（位置无关，消费侧 re.search 提取），
不发任何 pyautogui/win32 输入事件。

已知局限（O6）：clipboard 方案会临时覆盖系统剪贴板内容，并发写时有竞争风险；
这是中文输入的必要代价（pyautogui issue #237），不作为 bug 修复。
"""

from __future__ import annotations

import asyncio
import ctypes
import ctypes.wintypes as wintypes
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from src.mcp.desktop import session_state

if TYPE_CHECKING:
    from src.agents.models.screen_snapshot import ActionResult

logger = logging.getLogger(__name__)

# ── Win32 常量 ────────────────────────────────────────────────────────────────

WM_CLOSE = 0x0010

# GetAncestor 语义（K7 落点核验 / K8 前台核验）：GA_ROOT=沿父链取顶层窗口；
# GA_ROOTOWNER=再沿 owner 链走到底（有 owner 的主弹窗归回主窗；无 owner 的
# 菜单窗口 #32768 不归并——见 ActionSpec.expected_root_hwnd 契约「菜单点击勿设」）。
GA_ROOT = 2
GA_ROOTOWNER = 3

SW_MINIMIZE = 6
SW_RESTORE = 9

# SendInput（K8 梯级②）：无害 SHIFT key-up——不改变任何按键状态，但让本进程
# 获得「最近发送过输入」资格，绕开 SetForegroundWindow 的前台锁定限制。
INPUT_KEYBOARD = 1
VK_SHIFT = 0x10
KEYEVENTF_KEYUP = 0x0002

# SetWindowPos（do_focus_window pin_topmost 成对置顶/撤销）
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010


# ── SendInput 结构体（全量声明；union 需含最大成员 MOUSEINPUT 保证 sizeof 正确）──

_ULONG_PTR = ctypes.c_size_t  # ULONG_PTR：指针宽度无符号整数（64 位下 8 字节）


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTUNION)]


# 操控侧私有 DLL 实例 + 全量显式签名（K7/K8 新增 Win32 调用）。
# 先例与理由见 perception.py _pw_user32/_pw_gdi32 与 ai-docs/pitfalls.md
# 「ctypes.windll 是进程级共享对象」：共享 windll 的签名会被第三方库
# （pyautogui/pywinauto 等）篡改，且是否踩雷取决于 import 顺序；私有实例 +
# 全量 argtypes/restype 解耦，句柄一律按指针宽度（wintypes.HWND 等）传递。
_ctl_user32 = ctypes.WinDLL("user32", use_last_error=True)
_ctl_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_ctl_user32.GetForegroundWindow.argtypes = []
_ctl_user32.GetForegroundWindow.restype = wintypes.HWND
_ctl_user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
_ctl_user32.GetAncestor.restype = wintypes.HWND
_ctl_user32.IsIconic.argtypes = [wintypes.HWND]
_ctl_user32.IsIconic.restype = wintypes.BOOL
_ctl_user32.SetForegroundWindow.argtypes = [wintypes.HWND]
_ctl_user32.SetForegroundWindow.restype = wintypes.BOOL
_ctl_user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
_ctl_user32.ShowWindow.restype = wintypes.BOOL
_ctl_user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
_ctl_user32.AttachThreadInput.restype = wintypes.BOOL
_ctl_user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, wintypes.LPDWORD]
_ctl_user32.GetWindowThreadProcessId.restype = wintypes.DWORD
_ctl_user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int]
_ctl_user32.SendInput.restype = wintypes.UINT
_ctl_user32.SetWindowPos.argtypes = [
    wintypes.HWND,
    wintypes.HWND,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.UINT,
]
_ctl_user32.SetWindowPos.restype = wintypes.BOOL
# WindowFromPoint 收 POINT **结构体按值传参**：argtypes 直接声明 Structure 类型
# （wintypes.POINT），调用时传 POINT 实例——ctypes 对按值结构体参数的标准写法
# （现场核验 ctypes 文档：argtypes 中给出 Structure 子类即按值传递该结构体）。
_ctl_user32.WindowFromPoint.argtypes = [wintypes.POINT]
_ctl_user32.WindowFromPoint.restype = wintypes.HWND
_ctl_kernel32.GetCurrentThreadId.argtypes = []
_ctl_kernel32.GetCurrentThreadId.restype = wintypes.DWORD


# ── K2：写动作锁屏门 ──────────────────────────────────────────────────────────


async def _reject_if_locked(action_name: str) -> ActionResult | None:
    """写动作前置锁屏检测：锁定→拒绝结果；未锁/探测失败→None（放行）。

    锁屏下注入 pyautogui/win32 输入事件会落到凭据界面或被系统吞掉，且事后
    感知不可信——一律拒绝并带机读令牌 ``[desk:desktop_locked]``（位置无关，
    消费侧用 re.search 提取）。探测自身失败按未锁放行（服务态不为防护增强
    新增硬失败面，见 session_state 模块 docstring）。
    """
    locked, probe_failed = await asyncio.to_thread(session_state._is_desktop_locked_sync)
    if probe_failed:
        logger.warning("锁屏探测失败，按未锁继续执行 %s（lock_probe_failed）", action_name)
    if locked:
        return _make_result(
            success=False,
            error_message=(
                f"桌面会话已锁定（OpenInputDesktop 探测），拒绝执行 {action_name}，"
                "未注入任何输入事件 [desk:desktop_locked]。请解锁桌面后重试。"
            ),
        )
    return None


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


# ── K8：前台唤回梯级（细粒度辅助函数便于单测注桩，不真调 Win32） ──────────────


def _fl_get_foreground() -> int:
    """当前前台窗口 hwnd（无前台窗口返回 0）。"""
    return int(_ctl_user32.GetForegroundWindow() or 0)


def _fl_set_foreground(hwnd: int) -> None:
    """裸 SetForegroundWindow（成败以 _fl_verify 核验为准，返回值不作依据）。"""
    _ctl_user32.SetForegroundWindow(hwnd)


def _fl_verify(hwnd: int) -> bool:
    """核验 hwnd 是否真为前台顶层窗口且非最小化（K8 梯级核验）。

    判据：GetAncestor(GetForegroundWindow(), GA_ROOT) == hwnd 且 not IsIconic(hwnd)。
    ⚠ 这只是 **Win32 级**核验——前台/标题/CLOAKED 状态都可能撒谎
    （notes desktop-win32-state-untrusted），最终真值以后续快照像素锚点为准。
    """
    fg = _fl_get_foreground()
    if not fg:
        return False
    root = int(_ctl_user32.GetAncestor(fg, GA_ROOT) or 0) or fg
    return root == hwnd and not bool(_ctl_user32.IsIconic(hwnd))


def _fl_send_shift_up() -> None:
    """SendInput 发送一次无害 SHIFT key-up（K8 梯级②取前台权）。

    ⚠ SendInput 会重置系统 GetLastInputInfo 空闲计时——将来接空闲门控
    （用户离开检测）联动时需知会此副作用。
    """
    inp = _INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki = _KEYBDINPUT(
        wVk=VK_SHIFT, wScan=0, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=0
    )
    _ctl_user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))


def _fl_thread_ids(fg_hwnd: int) -> tuple[int, int]:
    """返回 (本线程 id, 前台窗口所属线程 id)。"""
    tid_self = int(_ctl_kernel32.GetCurrentThreadId())
    tid_fg = int(_ctl_user32.GetWindowThreadProcessId(fg_hwnd, None))
    return tid_self, tid_fg


def _fl_attach_thread_input(tid_self: int, tid_fg: int, attach: bool) -> bool:
    """AttachThreadInput 附加/分离前台线程输入队列，返回是否成功。"""
    return bool(_ctl_user32.AttachThreadInput(tid_self, tid_fg, attach))


def _fl_show_window(hwnd: int, cmd: int) -> None:
    """ShowWindow（K8 梯级④ SW_MINIMIZE→SW_RESTORE 强恢复用）。"""
    _ctl_user32.ShowWindow(hwnd, cmd)


def _fl_pin_topmost(hwnd: int, pin: bool) -> bool:
    """临时 HWND_TOPMOST 置顶（pin=True）/ HWND_NOTOPMOST 撤销，必须成对调用。"""
    insert_after = HWND_TOPMOST if pin else HWND_NOTOPMOST
    return bool(
        _ctl_user32.SetWindowPos(
            hwnd, insert_after, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE
        )
    )


def _focus_ladder(hwnd: int) -> tuple[int, bool]:
    """前台唤回梯级（K8），返回 (reached_rung, verified)。

    梯级（每级后按 _fl_verify 核验，通过即止）：
      ① 裸 SetForegroundWindow；
      ② SendInput 无害 SHIFT-up 取「最近输入」资格后重试；
      ③ AttachThreadInput 附加前台线程输入队列后重试（分离在 finally，异常路径
         也必分离——两线程输入队列黏连会串键鼠状态）；
      ④ 仍 iconic/假前台：SW_MINIMIZE→SW_RESTORE 强恢复（迫使 DWM 重渲染）后重试。

    ⚠ verified=True 仅代表 Win32 级核验通过，最终以后续快照像素锚点为准；
    ⚠ 梯级② SendInput 会重置 GetLastInputInfo 空闲计时（见 _fl_send_shift_up）。
    """
    # ① 裸 SetForegroundWindow
    _fl_set_foreground(hwnd)
    if _fl_verify(hwnd):
        return 1, True

    # ② SHIFT-up 取前台权后重试
    _fl_send_shift_up()
    _fl_set_foreground(hwnd)
    if _fl_verify(hwnd):
        return 2, True

    # ③ AttachThreadInput 附加前台线程输入队列后重试
    fg = _fl_get_foreground()
    if fg:
        tid_self, tid_fg = _fl_thread_ids(fg)
        if tid_fg and tid_fg != tid_self:
            attached = _fl_attach_thread_input(tid_self, tid_fg, True)
            try:
                _fl_set_foreground(hwnd)
            finally:
                if attached:
                    _fl_attach_thread_input(tid_self, tid_fg, False)
        else:
            _fl_set_foreground(hwnd)
    else:
        _fl_set_foreground(hwnd)
    if _fl_verify(hwnd):
        return 3, True

    # ④ 最小化→恢复强迫 DWM 重渲染（解 iconic / 假前台）后最后一试
    _fl_show_window(hwnd, SW_MINIMIZE)
    _fl_show_window(hwnd, SW_RESTORE)
    _fl_set_foreground(hwnd)
    if _fl_verify(hwnd):
        return 4, True
    return 4, False


# ── K7 批2：坐标落点核验 ──────────────────────────────────────────────────────


def _resolve_click_root(x: int, y: int, ga_flag: int = GA_ROOT) -> int | None:
    """屏幕坐标 (x, y) 当前落点的顶层窗口 hwnd（K7 批2 落点核验原语）。

    WindowFromPoint（POINT 结构体按值传参，见 _ctl_user32 签名注释）→
    GetAncestor(ga_flag)。ga_flag=GA_ROOT 取父链顶层；GA_ROOTOWNER 再沿 owner
    链归并（有 owner 的主弹窗归回主窗）。落点无窗口时返回 None。
    """
    hwnd = _ctl_user32.WindowFromPoint(wintypes.POINT(x, y))
    if not hwnd:
        return None
    root = _ctl_user32.GetAncestor(hwnd, ga_flag)
    return int(root) if root else int(hwnd)


def _verify_landing(x: int, y: int, expected_root_hwnd: int) -> ActionResult | None:
    """点击前一刻核验坐标落点顶层窗口：命中返回 None，否则返回拒绝结果。

    两级都算命中：GA_ROOT==expected（常规）或 GA_ROOTOWNER==expected（有 owner
    的主弹窗兜回主窗）。无 owner 的菜单窗口（#32768）不会归并——契约
    ActionSpec.expected_root_hwnd 已注明「菜单点击勿设期望值」。核验自身异常
    按拒绝处理（核验是显式 opt-in，宁拒不误点）。
    """
    try:
        actual_root = _resolve_click_root(x, y)
        if actual_root == expected_root_hwnd:
            return None
        if (
            actual_root is not None
            and _resolve_click_root(x, y, GA_ROOTOWNER) == expected_root_hwnd
        ):
            return None
    except Exception as exc:  # noqa: BLE001  — 探测异常宁拒不误点
        return _make_result(
            success=False,
            error_message=(
                f"坐标 ({x}, {y}) 落点核验探测异常，拒绝点击 [desk:landing_mismatch]：{exc}"
            ),
        )
    actual_desc = f"{actual_root:#x}" if actual_root is not None else "无窗口"
    class_name = ""
    if actual_root is not None:
        try:
            class_name = _get_class_name(actual_root)
        except Exception:  # noqa: BLE001  — 类名仅诊断用，取不到不影响拒绝
            class_name = "?"
    return _make_result(
        success=False,
        error_message=(
            f"坐标 ({x}, {y}) 实际落点顶层窗口 {actual_desc}（类名 {class_name!r}）"
            f"≠ 期望 {expected_root_hwnd:#x}，已拒绝点击（TOCTOU/遮挡防护）"
            " [desk:landing_mismatch]"
        ),
    )


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
    expected_root_hwnd: int | None = None,
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
        expected_root_hwnd: K7 批2 坐标落点核验的期望顶层窗口句柄；None=不核验
            （零回归）。提供时在 pyautogui.click 前一刻核验 WindowFromPoint 落点
            （GA_ROOT / GA_ROOTOWNER 两级命中皆可），不符则拒绝点击并带
            [desk:landing_mismatch] 令牌。**仅主窗元素点击设期望值**，弹出菜单
            勿设（契约语义见 ActionSpec.expected_root_hwnd docstring）。

    Returns:
        ActionResult，ui_changed 恒为 False（点击后状态由感知层感知）。
    """
    locked = await _reject_if_locked("click_element")
    if locked is not None:
        return locked

    degradation_notes: list[str] = []

    def _coord_click() -> ActionResult:
        """坐标兜底点击（K7 批2：点击前一刻核验落点顶层窗口）。"""
        if coordinates is None:
            return _make_result(
                success=False,
                error_message=(
                    "无法降级到 coordinate 模式：coordinates 未提供。"
                    + ("降级链：" + " → ".join(degradation_notes) if degradation_notes else "")
                ),
            )
        if expected_root_hwnd is not None:
            mismatch = _verify_landing(coordinates[0], coordinates[1], expected_root_hwnd)
            if mismatch is not None:
                return mismatch
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
    locked = await _reject_if_locked("type_text")
    if locked is not None:
        return locked

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
    locked = await _reject_if_locked("send_key")
    if locked is not None:
        return locked

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


async def do_focus_window(window_handle: int, pin_topmost: bool = False) -> ActionResult:
    """将指定窗口置前台并核验（K8 前台唤回梯级）。

    梯级与核验语义见 `_focus_ladder` docstring。两点必读：

    - **success=True 仅表示 Win32 级核验通过**（前台根==目标且非 iconic）——
      Win32 前台/标题状态会撒谎（notes desktop-win32-state-untrusted），最终
      确认以后续快照的像素锚点为准；
    - 梯级② SendInput 会**重置 GetLastInputInfo 空闲计时**，将来接空闲门控
      （用户离开检测）联动时需知会此副作用。

    Args:
        window_handle: 目标窗口 HWND 句柄（整数）。
        pin_topmost: True 时梯级期间临时 HWND_TOPMOST 置顶，结束后（含异常
            路径）成对撤销（默认 False；仅顽固遮挡场景用，撤销失败落日志告警）。

    Returns:
        ActionResult：核验通过→success=True/ui_changed=True（非①级时
        error_message 记录到达梯级）；全梯失败→success=False +
        [desk:focus_unverified] 令牌 + 到达梯级。
    """
    locked = await _reject_if_locked("focus_window")
    if locked is not None:
        return locked

    def _do_focus() -> ActionResult:
        if not _is_window(window_handle):
            return _make_result(
                success=False,
                error_message=f"窗口句柄 {window_handle:#x} 无效或已销毁",
            )
        pinned = False
        try:
            if pin_topmost:
                pinned = _fl_pin_topmost(window_handle, True)
            rung, verified = _focus_ladder(window_handle)
        except Exception as exc:
            return _make_result(
                success=False,
                error_message=f"前台唤回梯级异常（hwnd={window_handle:#x}）：{exc}",
            )
        finally:
            if pinned and not _fl_pin_topmost(window_handle, False):
                logger.warning("pin_topmost 撤销失败（hwnd=%#x），窗口可能残留置顶", window_handle)
        if verified:
            return _make_result(
                success=True,
                error_message=(
                    None
                    if rung == 1
                    else (
                        f"前台唤回于梯级 {rung} 核验通过"
                        "（①裸置前台 ②SendInput 取权 ③AttachThreadInput ④最小化-恢复）"
                    )
                ),
                ui_changed=True,
            )
        return _make_result(
            success=False,
            error_message=(
                f"前台唤回全部梯级失败（已达梯级 {rung}，Win32 核验未通过）"
                f" [desk:focus_unverified]：目标 hwnd={window_handle:#x} "
                "未成为前台根或仍最小化"
            ),
            ui_changed=False,
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
    locked = await _reject_if_locked("close_window")
    if locked is not None:
        return locked

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
