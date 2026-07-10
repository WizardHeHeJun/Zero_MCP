"""桌面屏幕能力 MCP Client（Python 侧）。

生命周期：async context manager（AsyncExitStack 嵌套 stdio_client + ClientSession）。
工具面：与 desktop_mcp_server.py 完全对应的 10 个 async 方法。
能力缓存：连接建立后调一次 get_capability_flags() 缓存到实例，生命周期与连接绑定（§7.4）。

设计约束（规格书 Task 6）：
- __aenter__：flag 检查 → stdio_client spawn 子进程 → ClientSession.initialize() → 能力缓存。
- call_tool 结果取 content[0].text → model_validate_json 强类型化。
- 三个自定义异常：DesktopCapabilityDisabledError / DesktopMCPConnectionError / DesktopMCPCallError。
- env 传递：StdioServerParameters.env 只含新增 key（库自动与 get_default_environment() 合并），
  加入完整 os.environ 以确保子进程能找到项目 src 包（sys.path / PYTHONPATH 等可能由 conda 设置）。
- Windows ProactorEventLoop（Python 3.12 默认）：anyio 在 Windows 上使用 asyncio backend，
  stdio_client 内部的 create_windows_process 已处理 ProactorEventLoop 兼容性（O4 核验）。
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import types
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent

from src.agents.models.screen_snapshot import (
    ActionResult,
    BBox,
    ScreenSnapshot,
    TextBlock,
    UIAElement,
)

logger = logging.getLogger(__name__)

# ── 自定义异常 ─────────────────────────────────────────────────────────────────


class DesktopCapabilityDisabledError(RuntimeError):
    """SCREEN_CAPABILITY_ENABLED=false 时尝试连接/调用抛出。

    客户端侧 flag 检查（双侧检查策略：client 禁用即拒绝 spawn，不等到 server 报错）。
    """


class DesktopMCPConnectionError(OSError):
    """stdio 子进程 spawn 失败或 ClientSession.initialize() 失败。

    附带 stderr 输出供诊断。
    """

    def __init__(self, message: str, stderr: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr


class DesktopMCPCallError(RuntimeError):
    """工具调用失败（isError=True 或内容解析失败）。"""

    def __init__(self, tool: str, message: str) -> None:
        super().__init__(f"[{tool}] {message}")
        self.tool = tool


# ── 内部辅助 ───────────────────────────────────────────────────────────────────


def _is_enabled() -> bool:
    """检查客户端侧 SCREEN_CAPABILITY_ENABLED feature flag。"""
    return os.environ.get("SCREEN_CAPABILITY_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
    }


def _build_subprocess_env() -> dict[str, str]:
    """构造子进程环境变量。

    StdioServerParameters.env 若非 None，SDK 内部会做 {**get_default_environment(), **env}，
    即只继承有限白名单 env（Windows: APPDATA/PATH/TEMP 等）再叠加我们传入的 key。
    为确保子进程能正确解析 `src.mcp.desktop_mcp_server` 模块（依赖 PYTHONPATH / conda PATH 等），
    直接传递完整 os.environ 的副本（字符串化），再覆写/透传能力 flag。
    """
    env: dict[str, str] = {k: v for k, v in os.environ.items() if isinstance(v, str)}
    # 显式透传 SCREEN_CAPABILITY_ENABLED，保证子进程能看到（parent 进程中可能已设）
    env["SCREEN_CAPABILITY_ENABLED"] = os.environ.get("SCREEN_CAPABILITY_ENABLED", "false")
    return env


def _extract_text(result: Any, tool_name: str) -> str:
    """从 CallToolResult 中提取文本内容。

    结果 isError=True 时抛 DesktopMCPCallError。
    content 为空或首元素非 TextContent 时也抛错。
    """
    if result.isError:
        # content[0].text 通常含错误描述
        err_text = ""
        if result.content and isinstance(result.content[0], TextContent):
            err_text = result.content[0].text
        raise DesktopMCPCallError(tool_name, err_text or "server 返回 isError=True")

    if not result.content:
        raise DesktopMCPCallError(tool_name, "server 返回空 content")

    first = result.content[0]
    if not isinstance(first, TextContent):
        raise DesktopMCPCallError(
            tool_name,
            f"期望 TextContent，得到 {type(first).__name__}",
        )
    return first.text


# ── 主类 ───────────────────────────────────────────────────────────────────────


class DesktopMCPClient:
    """桌面屏幕能力 MCP Client，async context manager。

    用法：
        async with DesktopMCPClient() as client:
            flags = await client.get_capability_flags()
            snapshot = await client.screen_snapshot(mode="uia_ocr")

    生命周期（§7.4）：
        - 实例创建时（__aenter__）spawn server 子进程，initialize() 后缓存能力 flags。
        - 能力缓存绑定到 server 连接，__aexit__ 后失效；不做跨连接独立失效逻辑。
        - server 重启 → 重建 DesktopMCPClient 实例 → 能力缓存自然刷新。
    """

    def __init__(self, repo_root: str | None = None) -> None:
        """初始化 DesktopMCPClient。

        Args:
            repo_root: 仓库根目录（默认自动推断为 src/ 的上一级目录）。
                       作为 StdioServerParameters.cwd，确保子进程模块解析正确。
        """
        if repo_root is None:
            # src/mcp/desktop_mcp_client.py → src/ → repo_root
            repo_root = str(__import__("pathlib").Path(__file__).parent.parent.parent.resolve())
        self.repo_root = repo_root

        self.exit_stack: contextlib.AsyncExitStack | None = None
        self.session: ClientSession | None = None
        self.capability_cache: dict[str, bool] | None = None

    # ── context manager ───────────────────────────────────────────────────────

    async def __aenter__(self) -> DesktopMCPClient:
        # 1. 客户端侧 feature flag 检查
        if not _is_enabled():
            raise DesktopCapabilityDisabledError(
                "桌面屏幕能力未启用（SCREEN_CAPABILITY_ENABLED=false）。"
                "请设置 SCREEN_CAPABILITY_ENABLED=true 后重试。"
            )

        # 2. 构造 server 启动参数
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "src.mcp.desktop_mcp_server"],
            cwd=self.repo_root,
            env=_build_subprocess_env(),
        )

        # 3. 用 AsyncExitStack 嵌套管理 stdio_client + ClientSession
        stack = contextlib.AsyncExitStack()
        try:
            await stack.__aenter__()

            # spawn 子进程，获取 read/write streams
            read_stream, write_stream = await stack.enter_async_context(stdio_client(server_params))

            # 建立 ClientSession
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            try:
                await session.initialize()
            except Exception as exc:
                raise DesktopMCPConnectionError(
                    f"ClientSession 初始化失败：{exc}",
                    stderr="",
                ) from exc

            self.exit_stack = stack
            self.session = session

            # 4. 能力协商：连接建立后调一次，缓存到实例（§7.4）
            try:
                self.capability_cache = await self._fetch_capability_flags()
                logger.info(
                    "DesktopMCPClient 连接成功，能力缓存：%s",
                    self.capability_cache,
                )
            except Exception as exc:
                # 能力缓存失败不阻断连接，仅 log warning
                logger.warning("能力 flags 缓存失败（%s），后续调用可能受影响", exc)
                self.capability_cache = {}

        except Exception:
            if self.exit_stack is None:
                # stack 尚未赋值给 self，统一在此处清理，避免二次 aclose
                try:
                    await stack.aclose()
                except Exception:
                    pass
            raise

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.capability_cache = None
        self.session = None
        if self.exit_stack is not None:
            await self.exit_stack.__aexit__(exc_type, exc_val, exc_tb)
            self.exit_stack = None

    # ── 内部 call_tool 封装 ───────────────────────────────────────────────────

    def _require_session(self) -> ClientSession:
        """断言 session 存在（在 context 内调用时始终满足）。"""
        if self.session is None:
            raise DesktopMCPConnectionError(
                "DesktopMCPClient 尚未初始化，请在 async with 块内使用。"
            )
        return self.session

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """调用 MCP 工具并返回 TextContent.text。

        Args:
            tool_name: 工具名（与 server 注册名一致）。
            arguments: 工具调用参数字典。

        Returns:
            server 返回的 JSON 字符串（调用方负责解析）。

        Raises:
            DesktopMCPCallError: server 返回 isError=True 或内容解析失败。
        """
        session = self._require_session()
        result = await session.call_tool(tool_name, arguments)
        return _extract_text(result, tool_name)

    # ── 能力协商（内部） ──────────────────────────────────────────────────────

    async def _fetch_capability_flags(self) -> dict[str, bool]:
        """向 server 请求能力 flags（仅内部调用一次）。"""
        text = await self._call_tool("get_capability_flags", {})
        raw: dict[str, Any] = json.loads(text)
        # 将 dataclasses.asdict 产生的字段过滤为 bool 值字典
        return {k: bool(v) for k, v in raw.items() if isinstance(v, bool)}

    # ── 公开 async 方法（10 个工具） ─────────────────────────────────────────

    async def get_capability_flags(self) -> dict[str, bool]:
        """返回能力协商结果（来自实例缓存，不重新调用 server）。

        缓存生命周期：与 server 连接绑定（__aexit__ 后失效）。
        server 重启需重建 DesktopMCPClient 实例以刷新缓存（§7.4）。

        Returns:
            能力标志字典，如 {"ocr": True, "omniparser": False, "cuda_accel": False}。
        """
        if self.capability_cache is None:
            # 理论上不会到这里（__aenter__ 已填充），但防御性保障
            self.capability_cache = await self._fetch_capability_flags()
        return dict(self.capability_cache)

    async def screen_snapshot(
        self,
        mode: str = "uia_ocr",
        capture_screenshot: bool = False,
    ) -> ScreenSnapshot:
        """获取完整屏幕感知快照。

        Args:
            mode: 感知模式，"uia_only" | "uia_ocr" | "full"（默认 uia_ocr）。
            capture_screenshot: 是否同时截图落磁盘（默认 False）。

        Returns:
            ScreenSnapshot Pydantic 模型实例。

        Raises:
            DesktopMCPCallError: server 工具调用失败。
        """
        text = await self._call_tool(
            "screen_snapshot",
            {"mode": mode, "capture_screenshot": capture_screenshot},
        )
        return ScreenSnapshot.model_validate_json(text)

    async def get_uia_tree(
        self,
        window_handle: int | None = None,
        max_depth: int = 5,
    ) -> list[UIAElement]:
        """获取指定窗口的 UIA 控件树。

        Args:
            window_handle: 目标窗口句柄（None = 活动窗口）。
            max_depth: 遍历最大深度（默认 5）。

        Returns:
            UIAElement 列表。

        Raises:
            DesktopMCPCallError: server 工具调用失败。
        """
        args: dict[str, Any] = {"max_depth": max_depth}
        if window_handle is not None:
            args["window_handle"] = window_handle
        text = await self._call_tool("get_uia_tree", args)
        raw_list: list[dict[str, Any]] = json.loads(text)
        return [UIAElement.model_validate(item) for item in raw_list]

    async def ocr_region(
        self,
        bbox: BBox,
        screenshot_path: str,
    ) -> list[TextBlock]:
        """对截图指定区域执行 OCR。

        Args:
            bbox: 矩形区域（BBox 模型，物理像素）。
            screenshot_path: 截图文件绝对路径（落磁盘后的 PNG）。

        Returns:
            TextBlock 列表（从 JSON 反序列化）。

        Raises:
            DesktopMCPCallError: server 工具调用失败。
        """

        text = await self._call_tool(
            "ocr_region",
            {
                "bbox": bbox.model_dump(),
                "screenshot_path": screenshot_path,
            },
        )
        raw_list: list[dict[str, Any]] = json.loads(text)
        return [TextBlock.model_validate(item) for item in raw_list]

    async def click_element(
        self,
        automation_id: str | None = None,
        coordinates: tuple[int, int] | None = None,
        method: str = "coordinate",
    ) -> ActionResult:
        """点击指定坐标或 UIA 元素。

        Args:
            automation_id: Windows AutomationId（即 UIAElement.automation_id 字段，
                           非本地索引 UIAElement.element_id）。uia_invoke/uia_click
                           模式使用；None 或空字符串时自动降级到 coordinate 模式。
            coordinates: 目标物理像素坐标 (x, y)。
            method: 点击方式，"coordinate"（默认）| "uia_invoke" | "uia_click"。

        Returns:
            ActionResult 模型实例（含 success / error_message / ui_changed）。

        Raises:
            DesktopMCPCallError: server 工具调用失败。
        """
        args: dict[str, Any] = {"method": method}
        if automation_id is not None:
            args["automation_id"] = automation_id
        if coordinates is not None:
            args["coordinates"] = list(coordinates)
        text = await self._call_tool("click_element", args)
        return ActionResult.model_validate_json(text)

    async def type_text(
        self,
        text: str,
        method: str = "clipboard",
    ) -> ActionResult:
        """向当前焦点输入文字（默认 clipboard 模式支持中文）。

        Args:
            text: 要输入的文字内容（中文必须用 clipboard 模式）。
            method: "clipboard"（默认）| "key_events"（仅 ASCII 安全）。
                    注意：clipboard 模式会临时污染系统剪贴板（O6 已知局限）。

        Returns:
            ActionResult 模型实例。

        Raises:
            DesktopMCPCallError: server 工具调用失败。
        """
        raw_text = await self._call_tool(
            "type_text",
            {"text": text, "method": method},
        )
        return ActionResult.model_validate_json(raw_text)

    async def send_key(self, key_combo: str) -> ActionResult:
        """发送键盘快捷键。

        Args:
            key_combo: 快捷键字符串（如 "ctrl+c"、"enter"、"alt+F4"）。

        Returns:
            ActionResult 模型实例。

        Raises:
            DesktopMCPCallError: server 工具调用失败。
        """
        text = await self._call_tool("send_key", {"key_combo": key_combo})
        return ActionResult.model_validate_json(text)

    async def window_list(self) -> list[dict[str, Any]]:
        """枚举桌面所有可见顶层窗口。

        Returns:
            窗口信息字典列表，每项含 hwnd/title/class_name/visible/rect 字段。

        Raises:
            DesktopMCPCallError: server 工具调用失败。
        """
        text = await self._call_tool("window_list", {})
        result: list[dict[str, Any]] = json.loads(text)
        return result

    async def focus_window(self, window_handle: int) -> ActionResult:
        """将指定窗口置于前台并获取焦点。

        Args:
            window_handle: 目标窗口 HWND 句柄。

        Returns:
            ActionResult 模型实例。

        Raises:
            DesktopMCPCallError: server 工具调用失败。
        """
        text = await self._call_tool("focus_window", {"window_handle": window_handle})
        return ActionResult.model_validate_json(text)

    async def close_window(self, window_handle: int) -> ActionResult:
        """关闭指定窗口（PostMessage WM_CLOSE）。

        注意：此操作不可逆，可能导致数据丢失。
        destructiveHint=True 的工具，应由编排层 action_guard 触发 interrupt 确认后调用。

        Args:
            window_handle: 目标窗口 HWND 句柄。

        Returns:
            ActionResult 模型实例（ui_changed 反映实际销毁状态）。

        Raises:
            DesktopMCPCallError: server 工具调用失败。
        """
        text = await self._call_tool("close_window", {"window_handle": window_handle})
        return ActionResult.model_validate_json(text)
