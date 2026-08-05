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
- src.agents.models.step_record.append_step（共享契约层，见下）

层依赖校验：agents → mcp client（允许）；agents → agents.protocols（同层，允许）；
不反向 import memory/storage 层，也不反向 import orchestration（code-review F2
根治：append_step 权威定义已挪到 src.agents.models.step_record——agents 与
orchestration 共同下调的契约层，本文件只下调 agents.models，不 import
src.orchestration.state，无反向依赖。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Literal

from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt as lg_interrupt

from src.agents.models.screen_snapshot import (
    ActionResult,
    ActionRisk,
    ActionSpec,
    ScreenSnapshot,
)
from src.agents.models.step_record import append_step
from src.agents.protocols import (
    ActionGuardProtocol,
    ConfirmRequest,
    ConfirmResponse,
    SnapshotStore,
)
from src.mcp.desktop_mcp_client import (
    DesktopMCPCallError,
    DesktopMCPClient,
    DesktopMCPConnectionError,
)

logger = logging.getLogger(__name__)

# ── 环境配置（K1 ⑤ 新鲜度门）─────────────────────────────────────────────────

TOCTOU_SNAPSHOT_MAX_AGE_MS: int = int(os.environ.get("TOCTOU_SNAPSHOT_MAX_AGE_MS", "5000"))
"""snapshot_before 复用的新鲜度上限（毫秒）。state.snapshot_ref 指向的快照
timestamp_ms 距当前超过此值则不复用（走 toctou_verify 内重拍现行为）。
默认 5000ms 为工程假设：感知→控制两节点间的典型间隔上界。"""


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

    async def execute(
        self,
        action: ActionSpec,
        snapshot_before: ScreenSnapshot | None = None,
    ) -> dict[str, Any]:
        """按 interrupt 分区协议执行单个动作，返回 state 增量。

        分区协议（SDK 核验 §1a，规格书 §1.2）：
          [只读区] 1. classify_risk → 得到 effective_risk
          [只读区] 2. toctou_verify → "pass" | "abort"（K1 ④：传入 effective_risk，
                     TOCTOU 触发与降级裁决按有效风险而非声明值）
          [只读区] 3. DESTRUCTIVE 时 lg_interrupt(ConfirmRequest) ← 触发 GraphInterrupt
          [写区]   4. Command(resume=ConfirmResponse) 恢复后执行写操作

        非 DESTRUCTIVE 且 TOCTOU pass → 直接执行写操作（无 interrupt）。
        TOCTOU abort → 返回 control_error，不执行写操作。

        Args:
            action: 待执行的动作规格。
            snapshot_before: 可选的执行前快照（K1 ⑤：由 make_control_node 经
                snapshot_store 加载并过新鲜度门后传入；None 时 toctou_verify
                自行重拍第一张截图）。

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

        # [只读 2] TOCTOU 验证（有效风险 DESTRUCTIVE / LOW_RISK / 坐标点击强制走）
        toctou_verdict = await self.guard.toctou_verify(
            action,
            snapshot_before=snapshot_before,
            effective_risk=effective_risk,
        )
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
                    # K5 ④：清 pending_action——防同一被拒动作在后续路由中再次
                    # 进入 control 重复 interrupt（配合 supervisor 终态守卫）
                    "pending_action": None,
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
                expected_root_hwnd=action.expected_root_hwnd,
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

    K4 ③：成功/失败均补 ``pending_action: None``——动作已执行完毕（无论成败），
    残留的 pending_action 会让下一次进入 control 节点原样重放同一动作。

    Args:
        action: 原始动作规格（用于错误信息补充）。
        result: 执行结果。

    Returns:
        state 增量字典，含 control_error（成功时为 None）与 pending_action(None)。
    """
    if result.success:
        logger.info(
            "_build_control_increment: action_id=%r SUCCESS",
            action.action_id,
        )
        return {"control_error": None, "pending_action": None}

    err = result.error_message or "未知错误"
    logger.warning(
        "_build_control_increment: action_id=%r FAILED error=%r",
        action.action_id,
        err,
    )
    return {
        "control_error": f"action_id={action.action_id}: {err}",
        "pending_action": None,
    }


# ── 节点函数工厂 ──────────────────────────────────────────────────────────────


async def _load_fresh_snapshot(
    snapshot_store: SnapshotStore | None,
    state: Any,
) -> ScreenSnapshot | None:
    """按 state.snapshot_ref 加载执行前快照，过新鲜度门后返回（K1 ⑤）。

    新鲜度门：snapshot.timestamp_ms 距当前超过 TOCTOU_SNAPSHOT_MAX_AGE_MS
    则不复用（返回 None，toctou_verify 内走现行为重拍第一张截图）——过旧的
    快照做 TOCTOU 基线会把「早已发生的界面变化」误判为执行前突变，或反之
    漏掉真实变化。加载失败同样回退重拍，不阻断执行链路。

    Args:
        snapshot_store: 快照存取接口；None 时直接返回 None（零回归）。
        state: 编排 state（DesktopTaskState 或兼容 dict）。

    Returns:
        新鲜的 ScreenSnapshot，或 None（无 store / 无引用 / 过旧 / 加载失败）。
    """
    if snapshot_store is None:
        return None

    snapshot_ref: str | None = None
    if hasattr(state, "snapshot_ref"):
        snapshot_ref = state.snapshot_ref
    elif isinstance(state, dict):
        snapshot_ref = state.get("snapshot_ref")
    if snapshot_ref is None:
        return None

    try:
        snapshot = await snapshot_store.load(snapshot_ref)
    except Exception as exc:
        logger.warning(
            "_load_fresh_snapshot: 加载快照失败 ref=%r（%s），走重拍",
            snapshot_ref,
            exc,
        )
        return None

    age_ms = int(time.time() * 1000) - snapshot.timestamp_ms
    if age_ms > TOCTOU_SNAPSHOT_MAX_AGE_MS:
        logger.info(
            "_load_fresh_snapshot: 快照过旧 ref=%r age_ms=%d > %d，不复用，走重拍",
            snapshot_ref,
            age_ms,
            TOCTOU_SNAPSHOT_MAX_AGE_MS,
        )
        return None
    return snapshot


def make_control_node(
    agent: DesktopControlAgent,
    snapshot_store: SnapshotStore | None = None,
) -> Any:
    """生成 control_node 节点函数（闭包注入 agent 与可选 snapshot_store）。

    注意：GraphBubbleUp（GraphInterrupt 的父类）是 LangGraph 的控制流信号，
    必须向上透传，不可被 except Exception 吞掉。节点内异常捕获只覆盖
    MCP 调用错误，GraphBubbleUp 始终 re-raise。

    Args:
        agent: 已构造的 DesktopControlAgent 实例。
        snapshot_store: 可选快照存取接口（K1 ⑤，先例：desktop_graph.py stall
            节点的 store 注入模式）。非 None 时从 state.snapshot_ref 加载快照、
            过新鲜度门（TOCTOU_SNAPSHOT_MAX_AGE_MS）后作 snapshot_before 传入
            toctou_verify，省一次截图 RPC；None 时保持现行为（零回归）。

    Returns:
        符合 LangGraph 节点签名 `(state) -> dict` 的异步函数。
    """

    async def _control_node(state: Any) -> dict[str, Any]:
        """LangGraph 控制节点（执行写操作，DESTRUCTIVE 需 interrupt 确认）。

        从 state.pending_action 提取待执行动作，按 interrupt 分区协议执行。
        返回只含增量字段的 dict：control_error / task_status（仅拒绝时含后者）。

        GraphBubbleUp（含 GraphInterrupt）始终透传，不被捕获——这是
        LangGraph interrupt 机制的必要条件。

        code-review F4（MCP 异常分支 at-least-once 风险）：DesktopMCPCallError /
        DesktopMCPConnectionError 分支补 ``pending_action: None``——RPC 已发出后
        无法区分「未到达 server」与「已执行但未收到回执」（网络层的
        at-least-once 语义），对 click/type/key/window_close 这类非幂等写操作
        （尤其发消息类）：若保留 pending_action 让 Supervisor 原样重放，
        坏情形是同一动作被执行两次（如重复发送消息）；清空后 Supervisor 需
        重新规划（可能重新截图核验现状），代价是多一轮感知，但避免了更危险的
        重复副作用。TOCTOU abort 分支**不受此影响、语义不变**——它发生在写操作
        派发之前（execute 只读区），pending_action 保留是安全的：下次重试前
        必经 TOCTOU 重核验，不会跳过安全门盲目重放。

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

        # K1 ⑤：snapshot_before 死码消除——经 store 加载 state.snapshot_ref
        # 指向的快照并过新鲜度门（只读操作，interrupt 重放时幂等重跑）
        snapshot_before = await _load_fresh_snapshot(snapshot_store, state)

        increment: dict[str, Any]
        try:
            increment = await agent.execute(pending_action, snapshot_before=snapshot_before)
        except GraphBubbleUp:
            # LangGraph 控制流信号（GraphInterrupt 等）必须透传，不可捕获
            raise
        except (DesktopMCPCallError, DesktopMCPConnectionError) as exc:
            logger.warning("control_node: MCP 调用失败：%s", exc)
            # F4：RPC 已发出、结果未知（at-least-once）——清 pending_action，
            # 不让 Supervisor 原样重放非幂等写操作（见 _control_node docstring）。
            increment = {"control_error": f"MCP 调用失败: {exc}", "pending_action": None}
        except Exception as exc:
            logger.warning("control_node: 意外异常：%s", exc)
            # 复核追加：兜底分支同样清 pending_action——此处可捕获的异常
            # （如 ActionResult 响应体解析 ValidationError）发生在 RPC 响应
            # **已返回之后**，server 大概率已真执行写操作，比连接断开更接近
            # 「已执行」，at-least-once 论证同上，不让 Supervisor 原样重放。
            increment = {"control_error": f"unexpected: {exc}", "pending_action": None}

        # K4 ③：control 步进入 step_history——append 是 execute **返回后**的纯计算：
        # interrupt 首跑在 execute 内 raise（走不到这里），resume 重放到达时只追加
        # 一次，不进 interrupt 只读区，重放确定性安全。state 不带 step_history
        # （最小测试 state / 裸 dict）时跳过，零耦合。
        prev_steps: Any = None
        if hasattr(state, "step_history"):
            prev_steps = state.step_history
        elif isinstance(state, dict):
            prev_steps = state.get("step_history")
        if isinstance(prev_steps, list):
            instruction: Any = ""
            if hasattr(state, "current_instruction"):
                instruction = state.current_instruction or ""
            elif isinstance(state, dict):
                instruction = state.get("current_instruction") or ""
            status_after: Any = increment.get("task_status")
            if not status_after:
                if hasattr(state, "task_status"):
                    status_after = state.task_status or "RUNNING"
                elif isinstance(state, dict):
                    status_after = state.get("task_status") or "RUNNING"
                else:
                    status_after = "RUNNING"
            increment["step_history"] = append_step(
                prev_steps,
                agent="control",
                instruction=str(instruction),
                increment=increment,
                task_status=str(status_after),
            )
        return increment

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
