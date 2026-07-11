"""桌面屏幕能力 MCP Server（FastMCP 骨架）。

feature flag：SCREEN_CAPABILITY_ENABLED（默认 false）。
传输：stdio（供 DesktopMCPClient spawn 子进程）。

设计约束（规格书 Task 3C）：
- 传输层零业务逻辑：工具体只做参数转换 + 转发 + 错误映射。
- 业务/降级逻辑全在 perception.py / control.py。
- 感知库 import 只在 perception.py，操控库 import 只在 control.py（模块级延迟 import）。
- SCREEN_CAPABILITY_ENABLED=false 时始终注册工具、运行时首行 raise（更易测）。
- 阻塞 I/O 在 perception/control 内用 asyncio.to_thread 包装（此层只调 async 接口）。
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from src.mcp.desktop.capability_probe import CapabilityFlags, probe_capabilities

logger = logging.getLogger(__name__)

# ── 全局状态 ──────────────────────────────────────────────────────────────────

_CAPABILITY_FLAGS: CapabilityFlags | None = None

mcp = FastMCP(
    name="desktop-screen-capability",
    instructions=(
        "提供桌面屏幕感知与操控能力：UIA 树解析、OCR 识别、截图、"
        "鼠标/键盘模拟、窗口管理。仅 SCREEN_CAPABILITY_ENABLED=true 时生效。"
    ),
)


def _is_enabled() -> bool:
    """检查 SCREEN_CAPABILITY_ENABLED feature flag。"""
    return os.environ.get("SCREEN_CAPABILITY_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
    }


def _require_enabled() -> None:
    """feature flag 未开时 raise ToolError，阻止工具执行。"""
    if not _is_enabled():
        raise ToolError(
            "桌面屏幕能力未启用（SCREEN_CAPABILITY_ENABLED=false）。"
            "请在 .env 中设置 SCREEN_CAPABILITY_ENABLED=true 后重启 server。"
        )


def _get_flags() -> CapabilityFlags:
    """获取已探测的 CapabilityFlags（server 启动时缓存）。"""
    global _CAPABILITY_FLAGS
    if _CAPABILITY_FLAGS is None:
        _CAPABILITY_FLAGS = probe_capabilities()
    return _CAPABILITY_FLAGS


def _dump_model(obj: Any) -> str:
    """将 pydantic 模型序列化为 JSON 字符串（类型安全包装）。"""
    result: str = obj.model_dump_json()
    return result


# ── 感知工具 ──────────────────────────────────────────────────────────────────


@mcp.tool(
    name="screen_snapshot",
    description=(
        "获取完整屏幕感知快照（UIA 树 + OCR 文字块 + 视觉对象）。"
        "返回 ScreenSnapshot JSON，screenshot_path 为磁盘路径。"
    ),
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
)
async def screen_snapshot(
    mode: str = "uia_ocr",
    capture_screenshot: bool = False,
    window_handle: int | None = None,
) -> str:
    """感知当前屏幕状态。

    Args:
        mode: 感知模式，"uia_only" | "uia_ocr" | "full"（默认 uia_ocr）。
        capture_screenshot: 是否同时截图落磁盘（默认 False）。
        window_handle: 目标窗口 HWND（None = 前台窗口）。指定时 UIA 树
            与 OCR 裁剪均以该窗口为准，感知不再依赖前台归属。

    Returns:
        ScreenSnapshot 序列化 JSON。
    """
    _require_enabled()
    import src.mcp.desktop.tools.perception as perception  # noqa: PLC0415

    flags = _get_flags()
    screenshot_tmp_dir = os.environ.get("SCREENSHOT_TMP_DIR", "")
    try:
        snapshot = await perception.do_screen_snapshot(
            mode=mode,
            capture_screenshot=capture_screenshot,
            caps=flags,
            screenshot_tmp_dir=screenshot_tmp_dir,
            window_handle=window_handle,
        )
        return _dump_model(snapshot)
    except ToolError:
        raise
    except Exception as exc:
        logger.error("screen_snapshot 失败：%s", exc, exc_info=True)
        raise ToolError(f"screen_snapshot 执行失败：{exc}") from exc


@mcp.tool(
    name="get_uia_tree",
    description="获取指定窗口（或当前活动窗口）的 UIA 控件树，返回 UIAElement 列表 JSON。",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
)
async def get_uia_tree(
    window_handle: int | None = None,
    max_depth: int = 5,
) -> str:
    """获取 UIA 控件树。

    Args:
        window_handle: 目标窗口句柄（None = 活动窗口）。
        max_depth: 遍历最大深度（默认 5）。

    Returns:
        UIAElement 列表序列化 JSON。
    """
    _require_enabled()
    import src.mcp.desktop.tools.perception as perception  # noqa: PLC0415

    flags = _get_flags()
    try:
        elements: list[Any] = await perception.do_get_uia_tree(
            window_handle=window_handle,
            max_depth=max_depth,
            caps=flags,
        )
        return json.dumps([e.model_dump() for e in elements], ensure_ascii=False)
    except ToolError:
        raise
    except Exception as exc:
        logger.error("get_uia_tree 失败：%s", exc, exc_info=True)
        raise ToolError(f"get_uia_tree 执行失败：{exc}") from exc


@mcp.tool(
    name="ocr_region",
    description="对指定截图文件的矩形区域执行 OCR 识别，返回 TextBlock 列表 JSON（含置信度）。",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
)
async def ocr_region(
    bbox: dict[str, int],
    screenshot_path: str,
) -> str:
    """OCR 识别截图区域。

    Args:
        bbox: 矩形区域 {"x": int, "y": int, "width": int, "height": int}（物理像素）。
        screenshot_path: 截图文件绝对路径（落磁盘后的 PNG）。

    Returns:
        TextBlock 列表序列化 JSON。
    """
    _require_enabled()
    import src.mcp.desktop.tools.perception as perception  # noqa: PLC0415

    flags = _get_flags()
    try:
        blocks: list[Any] = await perception.do_ocr_region(
            bbox=bbox,
            screenshot_path=screenshot_path,
            caps=flags,
        )
        return json.dumps([b.model_dump() for b in blocks], ensure_ascii=False)
    except ToolError:
        raise
    except Exception as exc:
        logger.error("ocr_region 失败：%s", exc, exc_info=True)
        raise ToolError(f"ocr_region 执行失败：{exc}") from exc


@mcp.tool(
    name="window_list",
    description="枚举当前桌面所有可见顶层窗口，返回窗口信息列表 JSON。",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
)
async def window_list() -> str:
    """枚举桌面可见窗口。

    Returns:
        窗口信息列表 JSON，每项含 hwnd/title/class_name/visible/rect。
    """
    _require_enabled()
    import src.mcp.desktop.tools.control as control  # noqa: PLC0415

    try:
        windows: list[dict[str, Any]] = await control.do_window_list()
        return json.dumps(windows, ensure_ascii=False)
    except ToolError:
        raise
    except Exception as exc:
        logger.error("window_list 失败：%s", exc, exc_info=True)
        raise ToolError(f"window_list 执行失败：{exc}") from exc


@mcp.tool(
    name="get_capability_flags",
    description="返回当前运行环境的能力探测结果（OCR/OmniParser/GPU/mss 可用性）。",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
)
async def get_capability_flags() -> str:
    """获取能力探测结果。

    Returns:
        CapabilityFlags 序列化 JSON。
    """
    _require_enabled()
    flags = _get_flags()
    return json.dumps(dataclasses.asdict(flags), ensure_ascii=False)


# ── 操控工具 ──────────────────────────────────────────────────────────────────


@mcp.tool(
    name="click_element",
    description=(
        "点击指定坐标或 UIA 元素（默认 coordinate 模式，Task 1 修正）。返回 ActionResult JSON。"
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
)
async def click_element(
    coordinates: tuple[int, int] | None = None,
    automation_id: str | None = None,
    method: str = "coordinate",
) -> str:
    """点击操作。

    Args:
        coordinates: 目标物理像素坐标 (x, y)（优先用于坐标模式）。
        automation_id: Windows AutomationId（即 UIAElement.automation_id 字段，
                       非本地索引 UIAElement.element_id）。uia_invoke/uia_click
                       模式使用；微信等 mmui 应用 AutomationId 为空，自动降级坐标。
        method: 点击方式，"coordinate"（默认）| "uia_invoke" | "uia_click"。

    Returns:
        ActionResult 序列化 JSON。
    """
    _require_enabled()
    import src.mcp.desktop.tools.control as control  # noqa: PLC0415

    try:
        result = await control.do_click_element(
            coordinates=coordinates,
            automation_id=automation_id,
            method=method,
        )
        return _dump_model(result)
    except ToolError:
        raise
    except Exception as exc:
        logger.error("click_element 失败：%s", exc, exc_info=True)
        raise ToolError(f"click_element 执行失败：{exc}") from exc


@mcp.tool(
    name="type_text",
    description="向当前焦点输入文字（默认 clipboard 模式以支持中文）。返回 ActionResult JSON。",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
)
async def type_text(
    text: str,
    method: str = "clipboard",
) -> str:
    """文字输入。

    Args:
        text: 要输入的文字内容（支持中文）。
        method: 输入方式，"clipboard"（默认，pyperclip.copy + ctrl+v）
                | "key_events"（仅 ASCII 安全）。
                注意：clipboard 模式会临时污染系统剪贴板（O6 已知局限）。

    Returns:
        ActionResult 序列化 JSON。
    """
    _require_enabled()
    import src.mcp.desktop.tools.control as control  # noqa: PLC0415

    try:
        result = await control.do_type_text(
            text=text,
            method=method,
        )
        return _dump_model(result)
    except ToolError:
        raise
    except Exception as exc:
        logger.error("type_text 失败：%s", exc, exc_info=True)
        raise ToolError(f"type_text 执行失败：{exc}") from exc


@mcp.tool(
    name="send_key",
    description="发送键盘快捷键（如 ctrl+c、alt+F4、enter）。返回 ActionResult JSON。",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
)
async def send_key(key_combo: str) -> str:
    """发送键盘组合键。

    Args:
        key_combo: 快捷键字符串（如 "ctrl+c"、"enter"、"alt+F4"）。

    Returns:
        ActionResult 序列化 JSON。
    """
    _require_enabled()
    import src.mcp.desktop.tools.control as control  # noqa: PLC0415

    try:
        result = await control.do_send_key(key_combo=key_combo)
        return _dump_model(result)
    except ToolError:
        raise
    except Exception as exc:
        logger.error("send_key 失败：%s", exc, exc_info=True)
        raise ToolError(f"send_key 执行失败：{exc}") from exc


@mcp.tool(
    name="focus_window",
    description="将指定句柄的窗口置于前台并获取焦点。返回 ActionResult JSON。",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
)
async def focus_window(window_handle: int) -> str:
    """聚焦指定窗口。

    Args:
        window_handle: 目标窗口 HWND 句柄。

    Returns:
        ActionResult 序列化 JSON。
    """
    _require_enabled()
    import src.mcp.desktop.tools.control as control  # noqa: PLC0415

    try:
        result = await control.do_focus_window(window_handle=window_handle)
        return _dump_model(result)
    except ToolError:
        raise
    except Exception as exc:
        logger.error("focus_window 失败：%s", exc, exc_info=True)
        raise ToolError(f"focus_window 执行失败：{exc}") from exc


@mcp.tool(
    name="close_window",
    description=(
        "向指定窗口发送 WM_CLOSE，等待 500ms 后检测是否销毁。"
        "此操作不可逆，可能导致数据丢失。返回 ActionResult JSON。"
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True),
)
async def close_window(window_handle: int) -> str:
    """关闭指定窗口（PostMessage WM_CLOSE）。

    Args:
        window_handle: 目标窗口 HWND 句柄。

    Returns:
        ActionResult 序列化 JSON（含 ui_changed=True/False 反映实际销毁状态）。
    """
    _require_enabled()
    import src.mcp.desktop.tools.control as control  # noqa: PLC0415

    try:
        result = await control.do_close_window(window_handle=window_handle)
        return _dump_model(result)
    except ToolError:
        raise
    except Exception as exc:
        logger.error("close_window 失败：%s", exc, exc_info=True)
        raise ToolError(f"close_window 执行失败：{exc}") from exc


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    enabled = _is_enabled()
    logger.info(
        "desktop_mcp_server 启动：SCREEN_CAPABILITY_ENABLED=%s",
        enabled,
    )

    if enabled:
        # 启动时探测能力并缓存（幂等）
        startup_flags = _get_flags()
        logger.info(
            "能力探测完成：device=%s cuda=%s dml=%s ocr=%s omniparser=%s mss=%s",
            startup_flags.effective_device,
            startup_flags.cuda_accel,
            startup_flags.dml_accel,
            startup_flags.ocr,
            startup_flags.omniparser,
            startup_flags.mss_available,
        )
    else:
        logger.warning(
            "SCREEN_CAPABILITY_ENABLED=false：所有工具调用将被拒绝（运行时检查），"
            "设置为 true 并重启以启用桌面能力。"
        )

    mcp.run(transport="stdio")
