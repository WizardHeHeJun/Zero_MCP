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
) -> str:
    """用 mss 截全屏，PIL 转换后存 PNG，返回绝对路径（阻塞）。"""
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
        monitor = sct.monitors[1]  # monitors[0] = 全虚拟屏，[1] = 主显示器
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        img.save(str(out_path), "PNG")

    logger.debug("截图已保存：%s", out_path)
    return str(out_path)


# ── OCR（RapidOCR） ────────────────────────────────────────────────────────────


def _run_ocr_on_file_sync(
    screenshot_path: str,
    bbox: dict[str, int] | None,
    snapshot_id: str,
) -> list[TextBlock]:
    """对截图文件（或其裁剪区域）执行 RapidOCR，返回 TextBlock 列表（阻塞）。

    Args:
        screenshot_path: PNG 文件路径。
        bbox: 可选裁剪区域 {x,y,width,height}（物理像素）；None=全图。
        snapshot_id: 用于生成 block_id 前缀。
    """
    from PIL import Image
    from rapidocr_onnxruntime import RapidOCR

    img = Image.open(screenshot_path).convert("RGB")
    if bbox is not None:
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        img = img.crop((x, y, x + w, y + h))
        offset_x, offset_y = x, y
    else:
        offset_x, offset_y = 0, 0

    engine = RapidOCR()
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
) -> ScreenSnapshot:
    """获取完整屏幕感知快照（三层降级逻辑）。

    Args:
        mode: "uia_only" | "uia_ocr" | "full"。
        capture_screenshot: 是否截图落磁盘。
        caps: 能力探测结果（来自 capability_probe）。
        screenshot_tmp_dir: 截图输出目录（空串时用系统临时目录）。

    Returns:
        ScreenSnapshot 实例（符合 src/agents/models/screen_snapshot.py 契约）。
    """
    if mode not in {"uia_only", "uia_ocr", "full"}:
        logger.warning("do_screen_snapshot: 未知 mode=%r，回退 uia_ocr", mode)
        mode = "uia_ocr"

    snapshot_id = str(uuid.uuid4())
    timestamp_ms = int(time.time() * 1000)
    screen_width, screen_height = _get_screen_size()

    # ── L1：窗口级定位（始终执行） ────────────────────────────────────────────
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

    # ── 截图（mss，阻塞） ──────────────────────────────────────────────────────
    screenshot_path: str | None = None
    # mode=uia_ocr/full 或 capture_screenshot=True 时截图
    need_screenshot = capture_screenshot or effective_mode in {"uia_ocr", "full"}
    if need_screenshot and caps.mss_available:
        try:
            screenshot_path = await asyncio.to_thread(
                _take_screenshot_sync,
                snapshot_id,
                screenshot_tmp_dir,
            )
        except Exception as exc:
            logger.error("截图失败（非致命，继续感知）：%s", exc, exc_info=True)
    elif need_screenshot and not caps.mss_available:
        logger.warning("mss 不可用，跳过截图（caps.mss_available=False）")

    # ── L2：OCR（uia_ocr / full，需截图文件） ─────────────────────────────────
    text_blocks: list[TextBlock] = []
    if effective_mode in {"uia_ocr", "full"} and screenshot_path is not None:
        if caps.ocr:
            try:
                text_blocks = await asyncio.to_thread(
                    _run_ocr_on_file_sync,
                    screenshot_path,
                    None,  # 全图 OCR
                    snapshot_id,
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

    Args:
        bbox: 裁剪区域 {"x": int, "y": int, "width": int, "height": int}。
        screenshot_path: 截图文件绝对路径（PNG）。
        caps: 能力标志（ocr=False 时 raise RuntimeError）。

    Returns:
        TextBlock 列表。

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
