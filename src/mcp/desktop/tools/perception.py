"""感知原语实现（UIA / mss / RapidOCR）。

本文件是 uiautomation、mss、rapidocr_onnxruntime 三个库在 MCP 层的唯一持有者。
其他模块不得直接 import 这些库——经此文件暴露的 async 函数调用即可。

三层降级（规格书 Task 4）：
  L1  UIA 树遍历——始终执行（窗口级定位）；
  L2  RapidOCR OCR —— mode=uia_ocr / full 时执行；
      hollow 且 mode=uia_only → 自动升 uia_ocr（log warning）；
  L3  视觉（模板匹配 / OmniParser）—— mode=full 且 caps.ocr / caps.omniparser 时。

DPI 感知：模块加载时立即调 _setup_dpi()（必须在 uiautomation import 前）。
阻塞计算：UIA 树遍历、mss 截图、RapidOCR 推理 全部走 asyncio.to_thread。

契约唯一真相：src/agents/models/screen_snapshot.py（不得修改已有字段语义）。
"""

from __future__ import annotations

import asyncio
import ctypes
import ctypes.wintypes as wintypes
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.agents.models.screen_snapshot import (
    BBox,
    ScreenSnapshot,
    TextBlock,
    UIAElement,
    VisualObject,
)

if TYPE_CHECKING:
    from src.mcp.desktop.capability_probe import CapabilityFlags

logger = logging.getLogger(__name__)

# Any 在延迟 import 的 uiautomation 控件对象上作类型标注（无 py.typed stubs）。
_AnyCtrl = Any  # 别名，让 mypy 对 uiautomation 控件属性访问静默

# ── DPI 感知（必须在 uiautomation 延迟 import 之前执行） ────────────────────────


def _setup_dpi() -> str:
    """设置进程 DPI 感知级别，返回实际生效的模式名。

    PER_MONITOR_AWARE_V2（-4）保证 UIA BoundingRectangle 为物理像素（蓝图工程假设）。
    必须在 uiautomation 首次 import 前调用——此函数在模块顶层立即执行。
    """
    user32 = ctypes.windll.user32
    try:
        # PER_MONITOR_AWARE_V2 = DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return "per_monitor_v2"
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        return "per_monitor"
    except (AttributeError, OSError):
        pass
    user32.SetProcessDPIAware()
    return "system"


# 模块级立即调用：保证后续延迟 import uiautomation 时 DPI 已设好
_DPI_MODE: str = _setup_dpi()
logger.debug("DPI 感知已设置：mode=%s", _DPI_MODE)

# ── 常量 ──────────────────────────────────────────────────────────────────────

# 可交互控件类型（uiautomation ControlTypeName 口径，与 poc 脚本保持一致）
_INTERACTIVE_TYPES: frozenset[str] = frozenset(
    {
        "ButtonControl",
        "EditControl",
        "ComboBoxControl",
        "CheckBoxControl",
        "RadioButtonControl",
        "ListItemControl",
        "TreeItemControl",
        "TabItemControl",
        "MenuItemControl",
        "HyperlinkControl",
        "SplitButtonControl",
        "SliderControl",
    }
)


def _get_uia_hollow_threshold() -> int:
    """读 UIA_HOLLOW_THRESHOLD 环境变量，默认 3。"""
    try:
        return int(os.environ.get("UIA_HOLLOW_THRESHOLD", "3"))
    except ValueError:
        return 3


# ── Win32 辅助 ─────────────────────────────────────────────────────────────────


def _get_active_window_info() -> tuple[int | None, str | None]:
    """返回当前前台窗口的 (hwnd, title)。失败返回 (None, None)。"""
    user32 = ctypes.windll.user32
    hwnd: int = user32.GetForegroundWindow()
    if not hwnd:
        return None, None
    buf = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, buf, 512)
    return hwnd, buf.value or None


def _get_window_title(hwnd: int) -> str | None:
    """返回指定窗口标题。空标题/失败返回 None。"""
    user32 = ctypes.windll.user32
    buf = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, buf, 512)
    return buf.value or None


def _get_screen_size() -> tuple[int, int]:
    """返回主显示器物理分辨率 (width, height)。"""
    user32 = ctypes.windll.user32
    return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))


def _get_window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """返回窗口物理像素 rect (left, top, right, bottom)，失败返回 None。"""
    user32 = ctypes.windll.user32
    rect = wintypes.RECT()
    if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return rect.left, rect.top, rect.right, rect.bottom
    return None


# ── UIA 树遍历（复用 poc 的 collect_tree 思路，延迟 import uiautomation） ────────


def _collect_uia_tree_sync(
    hwnd: int | None,
    max_depth: int,
) -> list[dict[str, Any]]:
    """同步遍历 UIA 树，返回元素属性 dict 列表（阻塞，应在 to_thread 中调用）。

    每个 dict 包含：
        control_type, name, automation_id, rect(left,top,right,bottom),
        is_enabled, is_offscreen, depth, children_count
    """
    import uiautomation as auto  # 延迟 import：DPI 已在模块级设好

    root: _AnyCtrl
    if hwnd is not None:
        root = auto.ControlFromHandle(hwnd)
    else:
        root = auto.GetRootControl()
        # 取前台窗口，避免遍历整个桌面树
        fg_hwnd = ctypes.windll.user32.GetForegroundWindow()
        if fg_hwnd:
            root = auto.ControlFromHandle(fg_hwnd)

    elements: list[dict[str, Any]] = []
    # DFS 栈：(control, depth)
    stack: list[tuple[_AnyCtrl, int]] = [(root, 0)]
    while stack:
        ctrl, depth = stack.pop()
        try:
            r = ctrl.BoundingRectangle
            elem: dict[str, Any] = {
                "control_type": ctrl.ControlTypeName or "UnknownControl",
                "name": (ctrl.Name or "")[:200].replace("\n", " "),
                "automation_id": (ctrl.AutomationId or "")[:200],
                "rect": (r.left, r.top, r.right, r.bottom),
                "is_enabled": bool(ctrl.IsEnabled),
                "is_offscreen": bool(ctrl.IsOffscreen),
                "depth": depth,
                "children_count": 0,
            }
        except Exception as exc:  # noqa: BLE001  — COM 单点失败只跳过
            logger.debug("UIA 属性读取失败（跳过）：%s", exc)
            continue
        elements.append(elem)
        if depth >= max_depth:
            continue
        try:
            children: list[_AnyCtrl] = ctrl.GetChildren()
        except Exception as exc:  # noqa: BLE001
            logger.debug("GetChildren 失败：%s", exc)
            continue
        elem["children_count"] = len(children)
        for child in reversed(children):
            stack.append((child, depth + 1))

    return elements


def _probe_window_uia_hollow_sync(hwnd: int) -> bool:
    """同步探测目标窗口 UIA 树是否为「空洞」（阻塞，应在 to_thread 中调用）。

    判定标准（规格书 Task 4）：
      - 一层直接子元素数 ≤ UIA_HOLLOW_THRESHOLD（默认 3），或
      - 可交互控件数 = 0
    任一满足即判 hollow=True。
    """
    import uiautomation as auto  # 延迟 import

    threshold = _get_uia_hollow_threshold()
    try:
        root: _AnyCtrl = auto.ControlFromHandle(hwnd)
        children: list[_AnyCtrl] = root.GetChildren()
    except Exception as exc:  # noqa: BLE001
        logger.warning("UIA 空洞探测 GetChildren 失败（hwnd=%#x）：%s，判 hollow=True", hwnd, exc)
        return True

    child_count = len(children)
    if child_count <= threshold:
        logger.debug(
            "UIA 空洞探测：hwnd=%#x 一层子元素=%d ≤ 阈值=%d → hollow",
            hwnd,
            child_count,
            threshold,
        )
        return True

    # 统计全一层+两层内可交互控件数（快速，不做深度遍历）
    interactive_count = 0
    child: _AnyCtrl
    for child in children:
        try:
            if child.ControlTypeName in _INTERACTIVE_TYPES:
                interactive_count += 1
            # 检查一层孙子
            grandchildren: list[_AnyCtrl] = child.GetChildren()
            gc: _AnyCtrl
            for gc in grandchildren:
                try:
                    if gc.ControlTypeName in _INTERACTIVE_TYPES:
                        interactive_count += 1
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass

    if interactive_count == 0:
        logger.debug(
            "UIA 空洞探测：hwnd=%#x 可交互控件=0（两层内）→ hollow",
            hwnd,
        )
        return True

    logger.debug(
        "UIA 空洞探测：hwnd=%#x 子元素=%d 可交互=%d → 非 hollow",
        hwnd,
        child_count,
        interactive_count,
    )
    return False


def _uia_elements_to_models(
    raw_elements: list[dict[str, Any]],
    hwnd: int | None,
) -> list[UIAElement]:
    """将 _collect_uia_tree_sync 返回的 dict 列表转换为 UIAElement 模型列表。"""
    result: list[UIAElement] = []
    for i, elem in enumerate(raw_elements):
        left, top, right, bottom = elem["rect"]
        width = max(0, right - left)
        height = max(0, bottom - top)
        # 跳过零面积元素（不可见/离屏控件）
        if width == 0 or height == 0:
            continue
        element_id = f"uia_{hwnd or 'desktop'}_{i}"
        result.append(
            UIAElement(
                element_id=element_id,
                control_type=elem["control_type"],
                name=elem["name"],
                automation_id=elem["automation_id"] or None,
                bbox=BBox(x=left, y=top, width=width, height=height),
                is_enabled=elem["is_enabled"],
                is_visible=not elem["is_offscreen"],
                value=None,
                source="uia",
            )
        )
    return result


# ── 截图（mss） ────────────────────────────────────────────────────────────────


def _take_screenshot_sync(
    snapshot_id: str,
    screenshot_tmp_dir: str,
) -> tuple[str, tuple[int, int, int, int]]:
    """用 mss 抓全虚拟屏（monitors[0]），PIL 转换后存 PNG（阻塞）。

    多显示器修正（2026-07-11 实测）：mss monitors[1] **不保证**是主显示器
    （本机实测 monitors[1]=副屏、monitors[2]=主屏，枚举顺序不可依赖），
    且主屏单幅截图会漏掉副屏窗口。故固定抓 monitors[0]=全虚拟屏。
    虚拟屏 origin 可为负（显示器排列在主屏左/上方时 SM_XVIRTUALSCREEN<0）。

    Returns:
        (png_path, (left, top, width, height))——第二项是实际抓取的
        虚拟屏 rect（mss monitors[0]），其 (left, top) 即图像坐标系原点
        的屏幕绝对坐标（capture_origin）。
    """
    import mss
    from PIL import Image

    # 确定输出目录
    if screenshot_tmp_dir:
        out_dir = Path(screenshot_tmp_dir)
    else:
        import tempfile

        out_dir = Path(tempfile.gettempdir()) / "zero_mcp_screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"snap_{snapshot_id}.png"

    with mss.mss() as sct:
        monitor = sct.monitors[0]  # monitors[0] = 全虚拟屏（覆盖所有显示器）
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        img.save(str(out_path), "PNG")
        virtual_rect = (
            int(monitor["left"]),
            int(monitor["top"]),
            int(monitor["width"]),
            int(monitor["height"]),
        )

    logger.debug("截图已保存：%s（虚拟屏 rect=%s）", out_path, virtual_rect)
    return str(out_path), virtual_rect


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


# PrintWindow 捕获专用的私有 DLL 实例 + 显式签名。
# Task 13 实测坑：ctypes.windll.* 是进程级共享对象——第三方库（pyautogui/
# pywinauto 等）会给共享的 user32.GetWindowDC 设 restype=c_void_p，返回的
# 64 位 HDC（>2^31）再传给未声明 argtypes 的 gdi32 调用（默认按 c_int 转换）
# 即 OverflowError: int too long to convert，且是否触发取决于 import 顺序
# （进程状态依赖，Task 12 钉钉实测未暴露）。私有 WinDLL 实例 + 全量显式
# argtypes/restype 与第三方库解耦，句柄一律按指针宽度传递。
_pw_user32 = ctypes.WinDLL("user32", use_last_error=True)
_pw_gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

_pw_user32.GetWindowDC.argtypes = [wintypes.HWND]
_pw_user32.GetWindowDC.restype = wintypes.HDC
_pw_user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
_pw_user32.ReleaseDC.restype = ctypes.c_int
_pw_user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
_pw_user32.PrintWindow.restype = wintypes.BOOL
_pw_gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
_pw_gdi32.CreateCompatibleDC.restype = wintypes.HDC
_pw_gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
_pw_gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
_pw_gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
_pw_gdi32.SelectObject.restype = wintypes.HGDIOBJ
_pw_gdi32.GetDIBits.argtypes = [
    wintypes.HDC,
    wintypes.HBITMAP,
    wintypes.UINT,
    wintypes.UINT,
    ctypes.c_void_p,
    ctypes.POINTER(_BITMAPINFOHEADER),
    wintypes.UINT,
]
_pw_gdi32.GetDIBits.restype = ctypes.c_int
_pw_gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
_pw_gdi32.DeleteObject.restype = wintypes.BOOL
_pw_gdi32.DeleteDC.argtypes = [wintypes.HDC]
_pw_gdi32.DeleteDC.restype = wintypes.BOOL


def _capture_window_sync(
    window_handle: int,
    snapshot_id: str,
    screenshot_tmp_dir: str,
) -> str | None:
    """PrintWindow(PW_RENDERFULLCONTENT) 捕获指定窗口自身的 DWM 渲染面（阻塞）。

    与屏幕截图的本质区别：取的是窗口自己的合成表面——**被其他窗口遮挡、
    无焦点、被压 z 序底部都能拿到真实像素**（Task 12 实测：真实多窗口桌面上
    目标窗口几乎总被遮挡，屏幕截图裁剪会拍到覆盖者的像素）。
    仅要求窗口非最小化。失败（最小化/DWM 未渲染出全黑/GDI 失败）返回 None，
    调用方回退全屏截图。
    """
    from PIL import Image

    rect = _get_window_rect(window_handle)
    if rect is None:
        return None
    left, top, right, bottom = rect
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0 or left <= -32000:
        return None  # 最小化哨兵 rect 或空窗口

    hwnd_dc = _pw_user32.GetWindowDC(window_handle)
    if not hwnd_dc:
        return None
    mem_dc = _pw_gdi32.CreateCompatibleDC(hwnd_dc)
    bmp = _pw_gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
    try:
        _pw_gdi32.SelectObject(mem_dc, bmp)
        PW_RENDERFULLCONTENT = 0x00000002
        if not _pw_user32.PrintWindow(window_handle, mem_dc, PW_RENDERFULLCONTENT):
            logger.warning("PrintWindow 失败（hwnd=%#x），回退屏幕截图", window_handle)
            return None
        bmi = _BITMAPINFOHEADER(
            ctypes.sizeof(_BITMAPINFOHEADER), width, -height, 1, 32, 0, 0, 0, 0, 0, 0
        )
        buf = ctypes.create_string_buffer(width * height * 4)
        if _pw_gdi32.GetDIBits(mem_dc, bmp, 0, height, buf, ctypes.byref(bmi), 0) != height:
            logger.warning("GetDIBits 失败（hwnd=%#x），回退屏幕截图", window_handle)
            return None
        img = Image.frombuffer("RGB", (width, height), buf.raw, "raw", "BGRX", 0, 1)
        if img.convert("L").getextrema() == (0, 0):
            logger.warning("PrintWindow 返回全黑（hwnd=%#x 未渲染），回退屏幕截图", window_handle)
            return None

        if screenshot_tmp_dir:
            out_dir = Path(screenshot_tmp_dir)
        else:
            import tempfile

            out_dir = Path(tempfile.gettempdir()) / "zero_mcp_screenshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"snapwin_{snapshot_id}.png"
        img.save(str(out_path), "PNG")
        logger.debug("窗口捕获已保存：%s（hwnd=%#x）", out_path, window_handle)
        return str(out_path)
    finally:
        _pw_gdi32.DeleteObject(bmp)
        _pw_gdi32.DeleteDC(mem_dc)
        _pw_user32.ReleaseDC(window_handle, hwnd_dc)


# ── OCR（RapidOCR） ────────────────────────────────────────────────────────────

# RapidOCR 引擎模块级懒加载单例：构造时加载 ONNX 模型（秒级），每快照重建会把
# 单次感知延迟推高一个量级（Task 12 实测 14.5s/快照，主因即此）。to_thread 下
# 可能并发首建，加锁保证只初始化一次。
_OCR_ENGINE: Any = None
_OCR_ENGINE_LOCK = threading.Lock()


def _get_ocr_engine() -> Any:
    """返回进程级 RapidOCR 单例（首次调用时加载模型，阻塞）。"""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        with _OCR_ENGINE_LOCK:
            if _OCR_ENGINE is None:
                from rapidocr_onnxruntime import RapidOCR

                _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def _run_ocr_on_file_sync(
    screenshot_path: str,
    bbox: dict[str, int] | None,
    snapshot_id: str,
    origin: tuple[int, int] = (0, 0),
) -> list[TextBlock]:
    """对截图文件（或其裁剪区域）执行 RapidOCR，返回 TextBlock 列表（阻塞）。

    Args:
        screenshot_path: PNG 文件路径。
        bbox: 可选裁剪区域 {x,y,width,height}（物理像素）；None=全图。
        snapshot_id: 用于生成 block_id 前缀。
        origin: 图像坐标系原点在屏幕绝对坐标中的位置。全屏截图为 (0,0)；
            PrintWindow 窗口图传窗口左上角，保证 TextBlock.bbox 恒为屏幕绝对坐标。
    """
    from PIL import Image

    img = Image.open(screenshot_path).convert("RGB")
    if bbox is not None:
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        img = img.crop((x, y, x + w, y + h))
        offset_x, offset_y = x + origin[0], y + origin[1]
    else:
        offset_x, offset_y = origin

    engine = _get_ocr_engine()
    # RapidOCR 接受 PIL Image / numpy array / 文件路径
    import numpy as np

    img_arr = np.array(img)
    result, _ = engine(img_arr)

    blocks: list[TextBlock] = []
    if not result:
        return blocks

    for idx, item in enumerate(result):
        # item = [box_points, text, confidence]
        # box_points = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]（顺时针四点）
        try:
            box_pts, text, conf = item[0], item[1], float(item[2])
        except (IndexError, TypeError, ValueError) as exc:
            logger.debug("OCR 结果解析失败（item=%r）：%s", item, exc)
            continue

        # 从四点计算 BBox（min/max 包围盒）
        xs = [pt[0] for pt in box_pts]
        ys = [pt[1] for pt in box_pts]
        x_min, y_min = int(min(xs)), int(min(ys))
        x_max, y_max = int(max(xs)), int(max(ys))

        blocks.append(
            TextBlock(
                block_id=f"ocr_{snapshot_id}_{idx}",
                text=str(text),
                bbox=BBox(
                    x=x_min + offset_x,
                    y=y_min + offset_y,
                    width=max(1, x_max - x_min),
                    height=max(1, y_max - y_min),
                ),
                confidence=min(1.0, max(0.0, conf)),
                source="ocr_rapidocr",
            )
        )

    logger.debug(
        "OCR 识别：%d 个文本块（screenshot=%s bbox=%s）", len(blocks), screenshot_path, bbox
    )
    return blocks


# ── 公开感知接口 ───────────────────────────────────────────────────────────────


async def do_screen_snapshot(
    mode: str,
    capture_screenshot: bool,
    caps: CapabilityFlags,
    screenshot_tmp_dir: str,
    window_handle: int | None = None,
) -> ScreenSnapshot:
    """获取完整屏幕感知快照（三层降级逻辑）。

    Args:
        mode: "uia_only" | "uia_ocr" | "full"。
        capture_screenshot: 是否截图落磁盘。
        caps: 能力探测结果（来自 capability_probe）。
        screenshot_tmp_dir: 截图输出目录（空串时用系统临时目录）。
        window_handle: 目标窗口 HWND；None = 前台窗口。指定时 UIA 树与
            OCR 裁剪均以该窗口为准（Task 12 实测：前台可被第三方窗口/
            自家子进程控制台抢占，前台耦合使感知在多窗口桌面不可靠）。

    Returns:
        ScreenSnapshot 实例（符合 src/agents/models/screen_snapshot.py 契约）。
    """
    if mode not in {"uia_only", "uia_ocr", "full"}:
        logger.warning("do_screen_snapshot: 未知 mode=%r，回退 uia_ocr", mode)
        mode = "uia_ocr"

    snapshot_id = str(uuid.uuid4())
    timestamp_ms = int(time.time() * 1000)
    screen_width, screen_height = _get_screen_size()

    # ── L1：窗口级定位（始终执行；指定 window_handle 时解除前台耦合） ─────────
    if window_handle is not None:
        active_hwnd: int | None = window_handle
        active_title = _get_window_title(window_handle)
    else:
        active_hwnd, active_title = _get_active_window_info()

    # 空洞探测（L1 附属步骤，阻塞 COM，走 to_thread）
    uia_hollow = False
    if active_hwnd is not None:
        uia_hollow = await asyncio.to_thread(_probe_window_uia_hollow_sync, active_hwnd)

    # hollow + uia_only → 自动升 uia_ocr（规格书 Task 4 核心要求）
    effective_mode = mode
    if uia_hollow and mode == "uia_only":
        logger.warning(
            "UIA 空洞（hwnd=%s）且 mode=uia_only，自动升级为 uia_ocr。"
            "目标窗口 UIA 内容树为空（如微信 4.x mmui 自绘），已切 OCR 主通道。",
            f"{active_hwnd:#x}" if active_hwnd else "None",
        )
        effective_mode = "uia_ocr"

    # ── UIA 树遍历（阻塞，走 to_thread） ──────────────────────────────────────
    raw_uia: list[dict[str, Any]] = await asyncio.to_thread(
        _collect_uia_tree_sync,
        active_hwnd,
        5,  # max_depth 固定 5，与 server 默认对齐
    )
    uia_elements = _uia_elements_to_models(raw_uia, active_hwnd)
    logger.debug(
        "UIA 遍历完成：hwnd=%s elements=%d hollow=%s",
        f"{active_hwnd:#x}" if active_hwnd else "None",
        len(uia_elements),
        uia_hollow,
    )

    # ── 截图 ──────────────────────────────────────────────────────────────────
    # 指定 window_handle 时优先 PrintWindow 捕获窗口自身渲染面（被遮挡/无焦点/
    # 压 z 序底都能取到真实像素，Task 12 实测真实桌面上目标窗口几乎总被遮挡）；
    # 失败（最小化/未渲染）回退 mss 全虚拟屏截图 + 窗口 rect 裁剪。
    screenshot_path: str | None = None
    window_captured = False
    capture_origin: tuple[int, int] = (0, 0)
    # mss 全虚拟屏截图实际抓取的 rect（left, top, width, height）；
    # PrintWindow 路径恒为 None。OCR 裁剪 clamp 以它为准。
    virtual_rect: tuple[int, int, int, int] | None = None
    # mode=uia_ocr/full 或 capture_screenshot=True 时截图
    need_screenshot = capture_screenshot or effective_mode in {"uia_ocr", "full"}
    if need_screenshot and window_handle is not None:
        try:
            screenshot_path = await asyncio.to_thread(
                _capture_window_sync,
                window_handle,
                snapshot_id,
                screenshot_tmp_dir,
            )
        except Exception as exc:
            logger.error("窗口捕获异常（非致命，回退屏幕截图）：%s", exc, exc_info=True)
            screenshot_path = None
        if screenshot_path is not None:
            window_captured = True
            wc_rect = _get_window_rect(window_handle)
            if wc_rect is not None:
                capture_origin = (wc_rect[0], wc_rect[1])
    if need_screenshot and screenshot_path is None:
        if caps.mss_available:
            try:
                mss_result: tuple[str, tuple[int, int, int, int]] = await asyncio.to_thread(
                    _take_screenshot_sync,
                    snapshot_id,
                    screenshot_tmp_dir,
                )
                screenshot_path, virtual_rect = mss_result
                # 图像原点 = 虚拟屏 origin（可为负，显示器在主屏左/上方时）
                capture_origin = (mss_result[1][0], mss_result[1][1])
            except Exception as exc:
                logger.error("截图失败（非致命，继续感知）：%s", exc, exc_info=True)
        else:
            logger.warning("mss 不可用，跳过截图（caps.mss_available=False）")

    # ── L2：OCR（uia_ocr / full，需截图文件） ─────────────────────────────────
    text_blocks: list[TextBlock] = []
    if effective_mode in {"uia_ocr", "full"} and screenshot_path is not None:
        if caps.ocr:
            # OCR 与 L1 UIA 同口径：PrintWindow 窗口图直接全图 OCR（origin 补偿回
            # 屏幕绝对坐标）；mss 全虚拟屏截图则裁剪到目标窗口 rect。Task 12 实测
            # （2026-07-10）：全图 OCR 会把其他应用的文本混入 perception_summary——
            # 既误导编排层，也构成跨窗口注入面。
            # mss 路径窗口 rect clamp 到虚拟屏 rect（2026-07-11 多显示器修正：
            # 副屏窗口不再回退全图；虚拟屏 origin 可为负）。裁剪 bbox 传的是
            # **图像坐标**（屏幕绝对坐标 − 虚拟屏 origin），origin 参数传
            # capture_origin，_run_ocr_on_file_sync 会补偿回屏幕绝对坐标。
            ocr_bbox: dict[str, int] | None = None
            if not window_captured and active_hwnd is not None and virtual_rect is not None:
                win_rect = _get_window_rect(active_hwnd)
                if win_rect is not None:
                    vs_left, vs_top, vs_width, vs_height = virtual_rect
                    clamp_x0 = max(vs_left, win_rect[0])
                    clamp_y0 = max(vs_top, win_rect[1])
                    clamp_x1 = min(vs_left + vs_width, win_rect[2])
                    clamp_y1 = min(vs_top + vs_height, win_rect[3])
                    if clamp_x1 > clamp_x0 and clamp_y1 > clamp_y0:
                        ocr_bbox = {
                            "x": clamp_x0 - vs_left,
                            "y": clamp_y0 - vs_top,
                            "width": clamp_x1 - clamp_x0,
                            "height": clamp_y1 - clamp_y0,
                        }
                    else:
                        logger.warning(
                            "窗口 rect=%s 不在虚拟屏截图范围 %s 内，OCR 回退全图",
                            win_rect,
                            virtual_rect,
                        )
            try:
                text_blocks = await asyncio.to_thread(
                    _run_ocr_on_file_sync,
                    screenshot_path,
                    ocr_bbox,
                    snapshot_id,
                    capture_origin,
                )
            except Exception as exc:
                logger.error("OCR 失败（非致命）：%s", exc, exc_info=True)
        else:
            logger.warning("caps.ocr=False，跳过 OCR（RapidOCR 不可用）")

    # ── L3：视觉（full，当前仅模板匹配占位；OmniParser 仅 caps.omniparser） ───
    visual_objects: list[VisualObject] = []
    if effective_mode == "full":
        if caps.omniparser:
            logger.info("OmniParser 可用但当前批次仅做探测，不执行推理（Task 4 范围）")
        # CPU 路径：OpenCV 模板匹配——本批 Task 4 范围不跑推理，占位空列表
        # 后续 Task 4+ 扩展时填充此处

    # ── 能力 flags（写入 snapshot） ──────────────────────────────────────────
    capability_flags: dict[str, bool] = {
        "ocr": caps.ocr,
        "omniparser": caps.omniparser,
        "cuda_accel": caps.cuda_accel,
        "dml_accel": caps.dml_accel,
        "mss_available": caps.mss_available,
        "uia_hollow": uia_hollow,
    }

    return ScreenSnapshot(
        snapshot_id=snapshot_id,
        timestamp_ms=timestamp_ms,
        screen_width=screen_width,
        screen_height=screen_height,
        active_window_title=active_title,
        uia_elements=uia_elements,
        text_blocks=text_blocks,
        visual_objects=visual_objects,
        screenshot_path=screenshot_path,
        perception_mode=effective_mode,  # type: ignore[arg-type]
        capability_flags=capability_flags,
        is_untrusted=True,
        uia_hollow=uia_hollow,
        capture_origin=capture_origin,
    )


async def do_get_uia_tree(
    window_handle: int | None,
    max_depth: int,
    caps: CapabilityFlags,  # noqa: ARG001  — 接口一致性，暂不过滤
) -> list[UIAElement]:
    """获取指定窗口（或前台窗口）的 UIA 控件树。

    Args:
        window_handle: 目标窗口 HWND；None = 前台窗口。
        max_depth: 最大遍历深度。
        caps: 能力标志（接口一致性，当前未用于过滤）。

    Returns:
        UIAElement 列表。
    """
    raw_elements: list[dict[str, Any]] = await asyncio.to_thread(
        _collect_uia_tree_sync,
        window_handle,
        max_depth,
    )
    return _uia_elements_to_models(raw_elements, window_handle)


async def do_ocr_region(
    bbox: dict[str, int],
    screenshot_path: str,
    caps: CapabilityFlags,
) -> list[TextBlock]:
    """对截图文件的指定区域执行 OCR。

    ⚠ 坐标系口径（Task 13）：bbox 与返回的 TextBlock.bbox 均为**该图像自身
    坐标系**（原点=图像左上角），不是屏幕绝对坐标。mss 全虚拟屏截图的图像
    坐标与屏幕绝对坐标仅在虚拟屏 origin=(0,0) 时重合；PrintWindow 窗口图则
    恒不重合。需要屏幕绝对坐标时，调用方自行按快照的 capture_origin 换算
    （screen_xy = image_xy + capture_origin）。

    Args:
        bbox: 裁剪区域 {"x": int, "y": int, "width": int, "height": int}（图像坐标）。
        screenshot_path: 截图文件绝对路径（PNG）。
        caps: 能力标志（ocr=False 时 raise RuntimeError）。

    Returns:
        TextBlock 列表（bbox 为图像坐标）。

    Raises:
        RuntimeError: caps.ocr=False（RapidOCR 不可用）。
        FileNotFoundError: screenshot_path 文件不存在。
    """
    if not caps.ocr:
        raise RuntimeError(
            "RapidOCR 不可用（caps.ocr=False）。请确认 rapidocr-onnxruntime 已安装。"
        )
    if not Path(screenshot_path).is_file():
        raise FileNotFoundError(f"截图文件不存在：{screenshot_path}")

    snapshot_id = str(uuid.uuid4())
    return await asyncio.to_thread(
        _run_ocr_on_file_sync,
        screenshot_path,
        bbox,
        snapshot_id,
    )
