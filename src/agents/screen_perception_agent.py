"""屏幕感知 Worker Agent（Task 8）。

职责：调 DesktopMCPClient.screen_snapshot → 对 text_blocks 执行注入过滤 →
组装 perception_summary → SnapshotStore 打桩存储 → 返回 state 增量。

设计约束：
- 不持感知库句柄（pywinauto/mss/RapidOCR 等），通过注入的 DesktopMCPClient 间接调用。
- Agent 不直连图谱/向量库（SnapshotStore 走 Protocol 打桩）。
- 节点签名 (state) -> dict，只返回增量字段。
- 异常 catch 返回 perception_error 非 None，不崩溃、不静默 retry。
- 增量字段（K4 起）：成功含 snapshot_ref / perception_summary / perception_error /
  uia_hollow / step_history；失败不含 uia_hollow（详见 perceive docstring）。

依赖：
- src.agents.text_filter.sanitize_screen_text（agents 层共享工具，纯函数）
- src.mcp.desktop_mcp_client.DesktopMCPClient（Task 6）
- src.agents.models.screen_snapshot.ScreenSnapshot
- src.agents.models.step_record.append_step / StepRecord（共享契约层，见下）

层依赖校验：agents → mcp client（允许）；不反向调 memory/storage、不反向调
orchestration（code-review F2 根治：StepRecord/append_step 权威定义已挪到
src.agents.models.step_record——agents 与 orchestration 共同下调的契约层，
本文件只下调 agents.models，不 import src.orchestration.state，无反向依赖。
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Literal

from pydantic import BaseModel

from src.agents.models.screen_snapshot import ScreenSnapshot
from src.agents.models.step_record import StepRecord, append_step
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
    """感知请求参数（感知模式 + 是否截图 + 定向窗口）。

    节点从 state 提取或使用默认值构造本对象，不直接暴露 MCP 参数到 state。
    """

    mode: Literal["uia_only", "uia_ocr", "full"] = "uia_ocr"
    capture_screenshot: bool = False
    # K6：定向感知目标窗口 HWND；None=前台窗口（现状口径，零回归）。
    # HWND 生命周期风险见 DesktopTaskState.target_window_handle 注释。
    window_handle: int | None = None


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

    async def perceive(
        self,
        request: PerceptionRequest,
        prev_steps: list[StepRecord] | None = None,
        instruction: str = "",
        task_status: str = "RUNNING",
    ) -> dict[str, Any]:
        """执行一次屏幕感知，返回 state 增量字段。

        流程：
          1. 调 client.screen_snapshot 获取 ScreenSnapshot（K6：透传 window_handle）。
          2. 对 text_blocks 的每个 text 执行 sanitize_screen_text（注入过滤）。
          3. 组装 perception_summary（原始摘要，截断在 Task 11 prompt_loader 层）。
          4. 调 snapshot_store.save 打桩存储，获取 snapshot_ref。
          5. 经 append_step 把本步追加进 step_history（K4，纯函数不 mutate）。

        异常：DesktopMCPCallError / DesktopMCPConnectionError / 其他异常均 catch，
              返回 perception_error 非 None，不抛出、不静默 retry。

        增量字段（K4 起不再只含三字段）：
          成功 — snapshot_ref / perception_summary / perception_error(None) /
                 uia_hollow（随快照刷新）/ step_history（追加后完整 list）。
          失败 — snapshot_ref(None) / perception_summary(None) / perception_error /
                 step_history；**不含 uia_hollow**（感知失败时窗口空洞与否未知，
                 保留 state 现值不覆写）。

        Args:
            request: 感知请求参数（mode / capture_screenshot / window_handle）。
            prev_steps: 追加前的 step_history（None 视为空历史）。
            instruction: 本步执行的指令摘要（记入 StepRecord）。
            task_status: 当前任务状态快照（感知不改状态，原样记入 StepRecord）。

        Returns:
            state 增量字典（字段见上）。
        """
        steps: list[StepRecord] = prev_steps if prev_steps is not None else []
        try:
            snapshot = await self.client.screen_snapshot(
                mode=request.mode,
                capture_screenshot=request.capture_screenshot,
                window_handle=request.window_handle,
            )
        except (DesktopMCPCallError, DesktopMCPConnectionError) as exc:
            logger.warning("ScreenPerceptionAgent.perceive: MCP 调用失败：%s", exc)
            increment: dict[str, Any] = {
                "snapshot_ref": None,
                "perception_summary": None,
                "perception_error": str(exc),
            }
            increment["step_history"] = append_step(
                steps,
                agent="perceive",
                instruction=instruction,
                increment=increment,
                task_status=task_status,
            )
            return increment
        except Exception as exc:
            logger.warning("ScreenPerceptionAgent.perceive: 意外异常：%s", exc)
            increment = {
                "snapshot_ref": None,
                "perception_summary": None,
                "perception_error": f"unexpected: {exc}",
            }
            increment["step_history"] = append_step(
                steps,
                agent="perceive",
                instruction=instruction,
                increment=increment,
                task_status=task_status,
            )
            return increment

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

        increment = {
            "snapshot_ref": snapshot_ref,
            "perception_summary": perception_summary,
            "perception_error": None,
            # K4 ②：uia_hollow 随快照刷新——否则 state 里的空洞标记停留在初值，
            # supervisor 的坐标点击引导块（jinja 模板 uia_hollow 分支）永远不渲染
            "uia_hollow": snapshot.uia_hollow,
        }
        increment["step_history"] = append_step(
            steps,
            agent="perceive",
            instruction=instruction,
            increment=increment,
            task_status=task_status,
        )
        return increment


def _build_perception_summary(snapshot: ScreenSnapshot, sanitized_texts: list[str]) -> str:
    """组装感知摘要字符串（纯函数，可单独测试）。

    摘要结构（面向 LLM 的结构化文本，不假设消费方是多模态）：
      - 活跃窗口标题
      - 感知模式 / uia_hollow 状态
      - UIA 元素摘要（类型/名称列表）
      - 过滤后文本块列表
      - 视觉对象摘要

    截断不在此处（由 Task 11 prompt_loader 统一处理）。

    **注入过滤边界**（蓝图 v2 WARN-1 的**第②道**）：本摘要整体会进 Supervisor 的 LLM prompt，故
    **所有由被感知应用填写的自由文本**都在此处过 `sanitize_screen_text`——包括活跃窗口标题、
    UIA 元素的 `name` 与 `control_type`、视觉对象的 `label`。`text_blocks` 例外：它由调用方
    预先过滤后经 `sanitized_texts` 传入（保持既有契约）。不过滤的只有 `VisualObject.source`
    与 `UIAElement.source`——它们是 `Literal` 枚举、由本仓自己填，非外部输入。

    ⚠ **WARN-1 的第①道（「MCP server 返回解析时入口即净化」）仍未实现**：`ScreenSnapshot`
    里存的仍是未过滤原文，本函数是原文进 LLM 的**唯一**出口，故当前无可达注入面；但一旦出现
    绕过本函数的新消费方（持久化存储重取、`mcp-server/` TS 层直接转发原始 JSON），该保护即失效。
    第①道未做是**待决而非遗漏**——入口即净化会让快照与现场包再也留不下攻击原文，取证保真度
    与纵深防御在此冲突，需产品决策。Task 7 验收「两道注入过滤到位」目前**仅达成第②道**。

    Args:
        snapshot: 原始感知快照。
        sanitized_texts: 过滤后的文本列表（与 snapshot.text_blocks 一一对应）。

    Returns:
        面向 LLM 的感知摘要字符串。
    """
    lines: list[str] = []

    # K2：降级警示行（摘要**首行**，LLM-agnostic 纯文本）。desktop_locked 标记与
    # degradations 机读枚举（契约见 ScreenSnapshot 类 docstring）合并列出——
    # 锁屏/截图失败/OCR 异常下快照文本可能不完整或过期，Supervisor 须先看到警示
    # 再消费下方内容。枚举值由本仓自产（Literal 语义），不过 sanitize。
    degraded_items: list[str] = []
    if snapshot.desktop_locked:
        degraded_items.append("desktop_locked")
    degraded_items.extend(d for d in snapshot.degradations if d not in degraded_items)
    if degraded_items:
        lines.append("⚠ 感知降级: " + ",".join(degraded_items) + "——本快照文本可能不完整或过期")

    # 基本元信息（窗口标题由外部进程提供 → 与 text_blocks 同为不可信文本，须过滤）
    window_title = (
        sanitize_screen_text(snapshot.active_window_title)
        if snapshot.active_window_title
        else "(无活跃窗口)"
    )
    lines.append(f"[感知摘要] 窗口={window_title}")
    # K2：window_captured 口径入既有状态行——True=PrintWindow 窗口直取（含被
    # 遮挡/后台部分），False=全屏截图口径，消费方据此判断坐标换算与遮挡假设
    lines.append(
        f"模式={snapshot.perception_mode}  uia_hollow={snapshot.uia_hollow}"
        f"  window_captured={snapshot.window_captured}"
    )
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
        增量字段见 ScreenPerceptionAgent.perceive docstring（K4：含 step_history，
        成功另含 uia_hollow）。

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

        # K6：定向感知目标窗口（仿 perception_mode 模式：hasattr + 非 None + int 校验；
        # bool 是 int 子类，显式排除防 True/False 混入 HWND）
        target_window_handle: int | None = default_request.window_handle
        if hasattr(state, "target_window_handle"):
            handle_val = state.target_window_handle
            if (
                handle_val is not None
                and isinstance(handle_val, int)
                and not isinstance(handle_val, bool)
            ):
                target_window_handle = handle_val

        # K4：step_history 追加所需上下文（最小 state / dict 不带这些字段时用默认值）
        prev_steps: list[StepRecord] = []
        if hasattr(state, "step_history") and isinstance(state.step_history, list):
            prev_steps = state.step_history
        elif isinstance(state, dict) and isinstance(state.get("step_history"), list):
            prev_steps = state["step_history"]

        instruction = ""
        if hasattr(state, "current_instruction") and isinstance(state.current_instruction, str):
            instruction = state.current_instruction
        elif isinstance(state, dict) and isinstance(state.get("current_instruction"), str):
            instruction = state["current_instruction"]

        task_status = "RUNNING"
        if hasattr(state, "task_status") and state.task_status:
            task_status = str(state.task_status)
        elif isinstance(state, dict) and state.get("task_status"):
            task_status = str(state["task_status"])

        req = PerceptionRequest(
            mode=perception_mode,
            capture_screenshot=capture_screenshot,
            window_handle=target_window_handle,
        )
        return await agent.perceive(
            req,
            prev_steps=prev_steps,
            instruction=instruction,
            task_status=task_status,
        )

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
