"""屏幕感知 Worker Agent（Task 8）。

职责：调 DesktopMCPClient.screen_snapshot → 对 text_blocks 执行注入过滤 →
组装 perception_summary → SnapshotStore 打桩存储 → 返回 state 增量。

设计约束：
- 不持感知库句柄（pywinauto/mss/RapidOCR 等），通过注入的 DesktopMCPClient 间接调用。
- Agent 不直连图谱/向量库（SnapshotStore 走 Protocol 打桩）。
- 节点签名 (state) -> dict，只返回增量字段。
- 异常 catch 返回 perception_error 非 None，不崩溃、不静默 retry。
- 增量只含 snapshot_ref / perception_summary / perception_error。

依赖：
- src.agents.text_filter.sanitize_screen_text（agents 层共享工具，纯函数）
- src.mcp.desktop_mcp_client.DesktopMCPClient（Task 6）
- src.agents.models.screen_snapshot.ScreenSnapshot

层依赖校验：agents → mcp client（允许）；不反向调 orchestration/memory。
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Literal

from pydantic import BaseModel

from src.agents.models.screen_snapshot import ScreenSnapshot
from src.agents.protocols import SnapshotStore
from src.agents.text_filter import sanitize_screen_text
from src.mcp.desktop_mcp_client import (
    DesktopMCPCallError,
    DesktopMCPClient,
    DesktopMCPConnectionError,
)

logger = logging.getLogger(__name__)


# ── SnapshotStore 打桩实现 ────────────────────────────────────────────────────
# SnapshotStore Protocol 权威定义在 src.agents.protocols（消除双定义技术债），
# 本模块只 import 使用 + 提供内存打桩实现。


class InMemorySnapshotStore:
    """SnapshotStore 内存实现（测试用打桩，不走真实存储）。"""

    def __init__(self) -> None:
        self.store: dict[str, ScreenSnapshot] = {}

    async def save(self, snapshot: ScreenSnapshot) -> str:
        self.store[snapshot.snapshot_id] = snapshot
        return snapshot.snapshot_id

    async def load(self, snapshot_id: str) -> ScreenSnapshot:
        return self.store[snapshot_id]


# ── 请求模型 ──────────────────────────────────────────────────────────────────


class PerceptionRequest(BaseModel):
    """感知请求参数（感知模式 + 是否截图）。

    节点从 state 提取或使用默认值构造本对象，不直接暴露 MCP 参数到 state。
    """

    mode: Literal["uia_only", "uia_ocr", "full"] = "uia_ocr"
    capture_screenshot: bool = False


# ── ScreenPerceptionAgent ─────────────────────────────────────────────────────


class ScreenPerceptionAgent:
    """屏幕感知 Worker Agent。

    注入 DesktopMCPClient 与 SnapshotStore，不持感知库底层句柄。

    用法（图构建时注入）：
        agent = ScreenPerceptionAgent(client=client, snapshot_store=store)
        node_fn = agent.perceive_node  # 注册到 StateGraph
    """

    def __init__(
        self,
        client: DesktopMCPClient,
        snapshot_store: SnapshotStore | None = None,
    ) -> None:
        """初始化 ScreenPerceptionAgent。

        Args:
            client: 已建立连接的 DesktopMCPClient 实例（async with 块内）。
            snapshot_store: 快照持久化接口；None 时使用 InMemorySnapshotStore 打桩。
        """
        self.client = client
        self.snapshot_store: SnapshotStore = snapshot_store or InMemorySnapshotStore()

    async def perceive(self, request: PerceptionRequest) -> dict[str, Any]:
        """执行一次屏幕感知，返回 state 增量字段。

        流程：
          1. 调 client.screen_snapshot 获取 ScreenSnapshot。
          2. 对 text_blocks 的每个 text 执行 sanitize_screen_text（注入过滤）。
          3. 组装 perception_summary（原始摘要，截断在 Task 11 prompt_loader 层）。
          4. 调 snapshot_store.save 打桩存储，获取 snapshot_ref。
          5. 返回 {"snapshot_ref": ..., "perception_summary": ..., "perception_error": None}。

        异常：DesktopMCPCallError / DesktopMCPConnectionError / 其他异常均 catch，
              返回 perception_error 非 None，不抛出、不静默 retry。

        Args:
            request: 感知请求参数（mode / capture_screenshot）。

        Returns:
            state 增量字典，只含三个字段：snapshot_ref / perception_summary / perception_error。
        """
        try:
            snapshot = await self.client.screen_snapshot(
                mode=request.mode,
                capture_screenshot=request.capture_screenshot,
            )
        except (DesktopMCPCallError, DesktopMCPConnectionError) as exc:
            logger.warning("ScreenPerceptionAgent.perceive: MCP 调用失败：%s", exc)
            return {
                "snapshot_ref": None,
                "perception_summary": None,
                "perception_error": str(exc),
            }
        except Exception as exc:
            logger.warning("ScreenPerceptionAgent.perceive: 意外异常：%s", exc)
            return {
                "snapshot_ref": None,
                "perception_summary": None,
                "perception_error": f"unexpected: {exc}",
            }

        # 对 text_blocks 每个 text 执行注入过滤
        sanitized_texts: list[str] = []
        for block in snapshot.text_blocks:
            sanitized = sanitize_screen_text(block.text)
            sanitized_texts.append(sanitized)

        # 组装 perception_summary（原始摘要，截断留 Task 11 prompt_loader）
        perception_summary = _build_perception_summary(snapshot, sanitized_texts)

        # SnapshotStore 打桩存储（不直连存储层）
        try:
            snapshot_ref = await self.snapshot_store.save(snapshot)
        except Exception as exc:
            # 存储失败不阻断感知流程，使用 snapshot_id 作为降级引用
            logger.warning(
                "ScreenPerceptionAgent.perceive: snapshot_store.save 失败（%s），"
                "使用 snapshot_id 降级",
                exc,
            )
            snapshot_ref = snapshot.snapshot_id

        logger.info(
            "ScreenPerceptionAgent.perceive: snapshot_ref=%s uia_hollow=%s text_blocks=%d",
            snapshot_ref,
            snapshot.uia_hollow,
            len(snapshot.text_blocks),
        )

        return {
            "snapshot_ref": snapshot_ref,
            "perception_summary": perception_summary,
            "perception_error": None,
        }


def _build_perception_summary(snapshot: ScreenSnapshot, sanitized_texts: list[str]) -> str:
    """组装感知摘要字符串（纯函数，可单独测试）。

    摘要结构（面向 LLM 的结构化文本，不假设消费方是多模态）：
      - 活跃窗口标题
      - 感知模式 / uia_hollow 状态
      - UIA 元素摘要（类型/名称列表）
      - 过滤后文本块列表
      - 视觉对象摘要

    截断不在此处（由 Task 11 prompt_loader 统一处理）。

    **注入过滤边界**（蓝图 v2 WARN-1）：本摘要整体会进 Supervisor 的 LLM prompt，故
    **所有由被感知应用填写的自由文本**都在此处过 `sanitize_screen_text`——包括活跃窗口标题、
    UIA 元素的 `name` 与 `control_type`、视觉对象的 `label`。`text_blocks` 例外：它由调用方
    预先过滤后经 `sanitized_texts` 传入（保持既有契约）。不过滤的只有 `VisualObject.source`
    与 `UiaElement.source`——它们是 `Literal` 枚举、由本仓自己填，非外部输入。

    Args:
        snapshot: 原始感知快照。
        sanitized_texts: 过滤后的文本列表（与 snapshot.text_blocks 一一对应）。

    Returns:
        面向 LLM 的感知摘要字符串。
    """
    lines: list[str] = []

    # 基本元信息（窗口标题由外部进程提供 → 与 text_blocks 同为不可信文本，须过滤）
    window_title = (
        sanitize_screen_text(snapshot.active_window_title)
        if snapshot.active_window_title
        else "(无活跃窗口)"
    )
    lines.append(f"[感知摘要] 窗口={window_title}")
    lines.append(f"模式={snapshot.perception_mode}  uia_hollow={snapshot.uia_hollow}")
    lines.append(f"屏幕={snapshot.screen_width}x{snapshot.screen_height}")

    # UIA 元素摘要（uia_hollow=True 时元素可能为空）
    if snapshot.uia_elements:
        lines.append(f"UIA元素({len(snapshot.uia_elements)}):")
        for elem in snapshot.uia_elements[:20]:  # 最多列 20 个，防摘要过长
            # name/control_type 均由被感知应用自行填写 → 不可信，须过滤
            elem_name = sanitize_screen_text(elem.name) if elem.name else "(无名)"
            lines.append(f"  [{sanitize_screen_text(elem.control_type)}] {elem_name}")
        if len(snapshot.uia_elements) > 20:
            lines.append(f"  ...（共 {len(snapshot.uia_elements)} 个，截断显示 20）")
    else:
        lines.append("UIA元素: (空)")

    # 文本块（过滤后）
    if sanitized_texts:
        lines.append(f"文本块({len(sanitized_texts)}):")
        for text in sanitized_texts:
            lines.append(f"  {text}")
    else:
        lines.append("文本块: (空)")

    # 视觉对象摘要
    if snapshot.visual_objects:
        lines.append(f"视觉对象({len(snapshot.visual_objects)}):")
        for obj in snapshot.visual_objects[:10]:
            # label 源自模型/模板匹配对屏幕内容的读出 → 不可信。
            # source 是 Literal 枚举（我方自产）→ 不过滤。
            lines.append(
                f"  [{obj.source}] {sanitize_screen_text(obj.label)} conf={obj.confidence:.2f}"
            )
    else:
        lines.append("视觉对象: (空)")

    return "\n".join(lines)


# ── 节点函数工厂（图构建时调用） ──────────────────────────────────────────────


def make_perceive_node(
    agent: ScreenPerceptionAgent,
    request: PerceptionRequest | None = None,
) -> Any:
    """生成 perceive_node 节点函数（闭包注入 agent 与默认请求参数）。

    Args:
        agent: 已构造的 ScreenPerceptionAgent 实例。
        request: 默认感知请求；None 时使用 PerceptionRequest() 默认值（uia_ocr）。

    Returns:
        符合 LangGraph 节点签名 `(state) -> dict` 的异步函数。
    """
    default_request = request or PerceptionRequest()

    async def perceive_node(state: Any) -> dict[str, Any]:
        """LangGraph 感知节点（只读感知，不写记忆/存储）。

        从 state 提取感知请求参数（若有），否则使用默认请求。
        返回只含增量字段的 dict：snapshot_ref / perception_summary / perception_error。

        Args:
            state: 编排 state（DesktopTaskState 或兼容 dict）。

        Returns:
            state 增量字典。
        """
        # 从 state 提取感知模式（若存在），否则使用默认值
        perception_mode: Literal["uia_only", "uia_ocr", "full"] = default_request.mode
        capture_screenshot: bool = default_request.capture_screenshot

        # state 可能是 Pydantic BaseModel 或 dict（兼容两种形式）
        if hasattr(state, "perception_mode") and state.perception_mode is not None:
            mode_val = state.perception_mode
            if mode_val in ("uia_only", "uia_ocr", "full"):
                perception_mode = mode_val

        req = PerceptionRequest(
            mode=perception_mode,
            capture_screenshot=capture_screenshot,
        )
        return await agent.perceive(req)

    return perceive_node


# ── 顶层便捷节点（图构建时注入 agent 后直接用） ──────────────────────────────


async def perceive_node(state: Any) -> dict[str, Any]:
    """顶层 perceive_node 占位（图构建时需用 make_perceive_node 注入 agent）。

    此函数为 import 便利保留，不应直接注册到图（无 agent 注入会 raise RuntimeError）。

    Args:
        state: 编排 state。

    Returns:
        不会到达此处（raise RuntimeError）。

    Raises:
        RuntimeError: 始终抛出，提示使用 make_perceive_node。
    """
    raise RuntimeError(
        "perceive_node 未注入 agent——请使用 make_perceive_node(agent) 生成节点函数，"
        "再注册到 StateGraph。"
    )


# ── 测试用辅助（不对外，测试文件内直接用 InMemorySnapshotStore） ──────────────


def _make_snapshot_id() -> str:
    """生成唯一快照 ID（测试辅助）。"""
    return f"snap-{uuid.uuid4().hex[:8]}"
