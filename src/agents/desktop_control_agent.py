"""桌面控制 Worker Agent（Task 9）。

职责：接收 ActionSpec → 安全门分级 → TOCTOU 验证 → （DESTRUCTIVE 时）interrupt
人工确认 → 执行写操作 → 返回 state 增量。

设计约束：
- interrupt 重放铁律（SDK 核验 §1a, 规格书 §1.2）：
    interrupt() 之前严格只读（classify_risk / toctou_verify），
    高危写操作（click_element / type_text / send_key / close_window）
    必须放 Command(resume=) 之后的 resume 区。
    代码内用 # --- interrupt 前只读区 --- / # --- resume 后写区 --- 注释分区。
- 节点签名 (state) -> dict，只返回增量字段。
- Agent 不直连图谱/向量库/存储层。
- 模型 ID 走 .env，不硬编码（本 Agent 不持 LLM 客户端）。
- I/O 全 async，不阻塞。
- 不 import Zero；不 print（用 logging）；不裸 except。

依赖：
- src.agents.protocols.ActionGuardProtocol（agents 层 Protocol，不 import 具体实现类）
- src.agents.protocols.ConfirmRequest / ConfirmResponse（agents 层权威定义）
- src.mcp.desktop_mcp_client.DesktopMCPClient（Task 6）
- src.agents.models.screen_snapshot.ActionSpec / ActionRisk / ActionResult

层依赖校验：agents → mcp client（允许）；agents → agents.protocols（同层，允许）；
不反向 import orchestration/memory/storage 层。
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt as lg_interrupt

from src.agents.models.screen_snapshot import ActionResult, ActionRisk, ActionSpec
from src.agents.protocols import ActionGuardProtocol, ConfirmRequest, ConfirmResponse
from src.mcp.desktop_mcp_client import (
    DesktopMCPCallError,
    DesktopMCPClient,
    DesktopMCPConnectionError,
)

logger = logging.getLogger(__name__)


# ── DesktopControlAgent ───────────────────────────────────────────────────────


class DesktopControlAgent:
    """桌面控制 Worker Agent。

    注入 DesktopMCPClient 与 ActionGuardProtocol，不持底层系统句柄。
    guard 参数类型为 ActionGuardProtocol（agents 层 Protocol），图构建时由
    orchestration 层注入 ActionGuard 实例（结构子类型满足 Protocol）。

    用法（图构建时注入）：
        agent = DesktopControlAgent(client=client, guard=guard)
        node_fn = make_control_node(agent)  # 注册到 StateGraph
    """

    def __init__(
        self,
        client: DesktopMCPClient,
        guard: ActionGuardProtocol,
    ) -> None:
        """初始化 DesktopControlAgent。

        Args:
            client: 已建立连接的 DesktopMCPClient 实例（async with 块内）。
            guard: ActionGuardProtocol 实例（编排层注入，结构子类型满足即可）。
        """
        self.client = client
        self.guard = guard

    async def execute(self, action: ActionSpec) -> dict[str, Any]:
        """按 interrupt 分区协议执行单个动作，返回 state 增量。

        分区协议（SDK 核验 §1a，规格书 §1.2）：
          [只读区] 1. classify_risk → 得到 effective_risk
          [只读区] 2. toctou_verify → "pass" | "abort"
          [只读区] 3. DESTRUCTIVE 时 lg_interrupt(ConfirmRequest) ← 触发 GraphInterrupt
          [写区]   4. Command(resume=ConfirmResponse) 恢复后执行写操作

        非 DESTRUCTIVE 且 TOCTOU pass → 直接执行写操作（无 interrupt）。
        TOCTOU abort → 返回 control_error，不执行写操作。

        Args:
            action: 待执行的动作规格。

        Returns:
            state 增量字典，含 control_error / task_status（仅拒绝时含 task_status）。
        """
        # --- interrupt 前只读区 ---

        # [只读 1] 风险分级
        effective_risk = await self.guard.classify_risk(action)
        logger.info(
            "DesktopControlAgent.execute: action_id=%r action_type=%r declared=%r effective=%r",
            action.action_id,
            action.action_type,
            action.risk_level,
            effective_risk,
        )

        # [只读 2] TOCTOU 验证（DESTRUCTIVE / LOW_RISK / 坐标点击强制走）
        toctou_verdict = await self.guard.toctou_verify(action)
        if toctou_verdict == "abort":
            logger.warning(
                "DesktopControlAgent.execute: TOCTOU abort action_id=%r",
                action.action_id,
            )
            return {
                "control_error": (
                    f"TOCTOU abort: 界面在执行前发生变化 (action_id={action.action_id})"
                ),
            }

        # [只读 3] DESTRUCTIVE 动作触发 interrupt，等待人工确认
        if effective_risk == ActionRisk.DESTRUCTIVE:
            confirm_req = ConfirmRequest(
                action_id=action.action_id,
                action_type=action.action_type,
                risk_level=effective_risk,
                description=f"高危动作待确认: {action.action_type}",
                coordinates=action.coordinates,
                target_element_id=action.target_element_id,
            )
            # lg_interrupt() 首次调用抛 GraphInterrupt（GraphBubbleUp 子类），
            # 节点整体重放时再次执行到此处取得 resume 值（ConfirmResponse）。
            # 重放时只读区（分级/TOCTOU）幂等重跑，无副作用。
            response: ConfirmResponse = lg_interrupt(confirm_req)

            # --- resume 后写区 ---
            # 以下所有代码仅在 Command(resume=ConfirmResponse(...)) 恢复后执行

            if not response.confirmed:
                logger.warning(
                    "DesktopControlAgent.execute: 人工拒绝 action_id=%r reason=%r",
                    action.action_id,
                    response.reason,
                )
                return {
                    "control_error": f"人工拒绝执行: {response.reason or '无说明'}",
                    "task_status": "FAILED",
                }

            # 已确认 → 执行写操作
            result = await self._dispatch_write(action)
            return _build_control_increment(action, result)

        # --- 非 DESTRUCTIVE：直接执行写操作（无 interrupt） ---
        # TOCTOU 已 pass，LOW_RISK / READ_ONLY 直接放行
        result = await self._dispatch_write(action)
        return _build_control_increment(action, result)

    async def _dispatch_write(self, action: ActionSpec) -> ActionResult:
        """按 action_type 路由执行写操作（所有 client 写调用在此集中）。

        仅在 interrupt resume 后或非 DESTRUCTIVE 直接执行路径调用。

        Args:
            action: 待执行的动作规格。

        Returns:
            ActionResult（成功或失败均含 error_message）。

        Raises:
            DesktopMCPCallError: client 工具调用失败（由调用方捕获）。
            DesktopMCPConnectionError: 连接断开（由调用方捕获）。
        """
        action_type = action.action_type

        if action_type in ("click", "click_element"):
            return await self.client.click_element(
                automation_id=action.target_element_id,
                coordinates=action.coordinates,
            )

        if action_type in ("type", "type_text"):
            payload = action.text_payload or ""
            return await self.client.type_text(text=payload)

        if action_type in ("key", "send_key"):
            payload = action.text_payload or ""
            return await self.client.send_key(key_combo=payload)

        if action_type == "window_close":
            # window_handle 从 target_element_id 解析（约定为整数字符串）
            if action.target_element_id is None:
                return ActionResult(
                    action_id=action.action_id,
                    success=False,
                    error_message="window_close 缺 target_element_id（window handle）",
                    ui_changed=False,
                )
            try:
                hwnd = int(action.target_element_id)
            except ValueError:
                return ActionResult(
                    action_id=action.action_id,
                    success=False,
                    error_message=(
                        f"window_close target_element_id 无法转换为整数:"
                        f" {action.target_element_id!r}"
                    ),
                    ui_changed=False,
                )
            return await self.client.close_window(window_handle=hwnd)

        logger.warning(
            "_dispatch_write: 未知 action_type=%r，跳过执行",
            action_type,
        )
        return ActionResult(
            action_id=action.action_id,
            success=False,
            error_message=f"未知 action_type: {action_type}",
            ui_changed=False,
        )


# ── 辅助函数 ──────────────────────────────────────────────────────────────────


def _build_control_increment(action: ActionSpec, result: ActionResult) -> dict[str, Any]:
    """从 ActionResult 构造 state 增量字典（纯函数，可单独测试）。

    Args:
        action: 原始动作规格（用于错误信息补充）。
        result: 执行结果。

    Returns:
        state 增量字典，含 control_error（成功时为 None）。
    """
    if result.success:
        logger.info(
            "_build_control_increment: action_id=%r SUCCESS",
            action.action_id,
        )
        return {"control_error": None}

    err = result.error_message or "未知错误"
    logger.warning(
        "_build_control_increment: action_id=%r FAILED error=%r",
        action.action_id,
        err,
    )
    return {"control_error": f"action_id={action.action_id}: {err}"}


# ── 节点函数工厂 ──────────────────────────────────────────────────────────────


def make_control_node(agent: DesktopControlAgent) -> Any:
    """生成 control_node 节点函数（闭包注入 agent）。

    注意：GraphBubbleUp（GraphInterrupt 的父类）是 LangGraph 的控制流信号，
    必须向上透传，不可被 except Exception 吞掉。节点内异常捕获只覆盖
    MCP 调用错误，GraphBubbleUp 始终 re-raise。

    Args:
        agent: 已构造的 DesktopControlAgent 实例。

    Returns:
        符合 LangGraph 节点签名 `(state) -> dict` 的异步函数。
    """

    async def _control_node(state: Any) -> dict[str, Any]:
        """LangGraph 控制节点（执行写操作，DESTRUCTIVE 需 interrupt 确认）。

        从 state.pending_action 提取待执行动作，按 interrupt 分区协议执行。
        返回只含增量字段的 dict：control_error / task_status（仅拒绝时含后者）。

        GraphBubbleUp（含 GraphInterrupt）始终透传，不被捕获——这是
        LangGraph interrupt 机制的必要条件。

        Args:
            state: 编排 state（DesktopTaskState 或兼容 dict）。

        Returns:
            state 增量字典。
        """
        # 从 state 提取待执行动作
        pending_action: ActionSpec | None = None
        if hasattr(state, "pending_action"):
            pending_action = state.pending_action
        elif isinstance(state, dict):
            pending_action = state.get("pending_action")

        if pending_action is None:
            logger.warning("control_node: state 无 pending_action，跳过执行")
            return {"control_error": "state 缺 pending_action"}

        try:
            return await agent.execute(pending_action)
        except GraphBubbleUp:
            # LangGraph 控制流信号（GraphInterrupt 等）必须透传，不可捕获
            raise
        except (DesktopMCPCallError, DesktopMCPConnectionError) as exc:
            logger.warning("control_node: MCP 调用失败：%s", exc)
            return {"control_error": f"MCP 调用失败: {exc}"}
        except Exception as exc:
            logger.warning("control_node: 意外异常：%s", exc)
            return {"control_error": f"unexpected: {exc}"}

    return _control_node


# ── 顶层占位节点（图构建时需用 make_control_node 注入 agent） ─────────────────


async def control_node(state: Any) -> dict[str, Any]:
    """顶层 control_node 占位（图构建时需用 make_control_node 注入 agent）。

    此函数为 import 便利保留，不应直接注册到图（无 agent 注入会 raise RuntimeError）。

    Args:
        state: 编排 state。

    Raises:
        RuntimeError: 始终抛出，提示使用 make_control_node。
    """
    raise RuntimeError(
        "control_node 未注入 agent——请使用 make_control_node(agent) 生成节点函数，"
        "再注册到 StateGraph。"
    )


# ── 辅助类型别名（供测试用） ──────────────────────────────────────────────────

ControlVerdict = Literal["pass", "abort"]
"""TOCTOU 验证结果类型别名。"""
