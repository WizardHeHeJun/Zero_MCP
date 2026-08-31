"""DesktopControlAgent 单元测试（Task 9）。

测试策略：
- mock DesktopMCPClient + ActionGuard，不依赖真实桌面环境。
- 用 InMemorySaver 建最小单节点图（仅 control 节点）测 interrupt/resume 链路。
- LangGraph 1.2.6 行为说明：ainvoke 对 interrupt 不抛 GraphInterrupt，
  而是在返回值的 __interrupt__ 字段携带 Interrupt 列表。
  resume 通过 ainvoke(Command(resume=v), config=same_config) 传入。

覆盖：
    1. DESTRUCTIVE 动作 → result["__interrupt__"] 非空，payload 是 ConfirmRequest。
    2. interrupt 前无任何 client 写调用（dispatch 未被触发）。
    3. resume abort（confirmed=False）→ task_status=FAILED，无 client 写调用。
    4. resume confirm（confirmed=True）→ client 写调用恰好一次，control_error=None。
    5. TOCTOU abort → control_error 含 "TOCTOU abort"（不带降级令牌），无 client 写调用。
    5b. TOCTOU abort_degraded → control_error 带 [desk:toctou_degraded]，无写调用。
    6. LOW_RISK 动作 → 无 interrupt，直接执行写操作。
    7. state 无 pending_action → control_error。
    8. _build_control_increment 辅助函数独立测试。
    9-15. dispatch 路由、异常捕获、模型验证等单元测试。
"""

from __future__ import annotations

import re
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel

from src.agents.desktop_control_agent import (
    DesktopControlAgent,
    _build_control_increment,
    make_control_node,
)
from src.agents.models.screen_snapshot import (
    ActionResult,
    ActionRisk,
    ActionSpec,
    ScreenSnapshot,
)
from src.agents.protocols import ConfirmRequest, ConfirmResponse
from src.orchestration.safety.action_guard import ActionGuard

# ── 测试辅助 ──────────────────────────────────────────────────────────────────


def _make_action(
    action_type: str = "window_close",
    risk_level: ActionRisk = ActionRisk.DESTRUCTIVE,
    coordinates: tuple[int, int] | None = None,
    target_element_id: str | None = "12345",
) -> ActionSpec:
    """构造测试用 ActionSpec。"""
    return ActionSpec(
        action_id="test-action-001",
        action_type=action_type,
        target_element_id=target_element_id,
        coordinates=coordinates,
        text_payload=None,
        risk_level=risk_level,
    )


def _make_mock_client(write_result: ActionResult | None = None) -> MagicMock:
    """构造 mock DesktopMCPClient。

    write_result 为 None 时默认返回成功的 ActionResult。
    """
    client = MagicMock()
    default_result = write_result or ActionResult(
        action_id="test-action-001",
        success=True,
        error_message=None,
        ui_changed=True,
    )
    client.click_element = AsyncMock(return_value=default_result)
    client.type_text = AsyncMock(return_value=default_result)
    client.send_key = AsyncMock(return_value=default_result)
    client.close_window = AsyncMock(return_value=default_result)
    client.screen_snapshot = AsyncMock()
    return client


def _make_mock_guard(
    risk: ActionRisk = ActionRisk.DESTRUCTIVE,
    toctou_verdict: Literal["pass", "abort", "abort_degraded"] = "pass",
) -> MagicMock:
    """构造 mock ActionGuard。"""
    guard = MagicMock()
    guard.classify_risk = AsyncMock(return_value=risk)
    guard.toctou_verify = AsyncMock(return_value=toctou_verdict)
    return guard


class _MinimalState(BaseModel):
    """最小图 state（无需完整 DesktopTaskState，降低测试耦合）。"""

    pending_action: ActionSpec | None = None
    control_error: str | None = None
    task_status: str | None = None


def _build_graph(agent: DesktopControlAgent) -> Any:
    """构造最小单节点 StateGraph（仅 control 节点），InMemorySaver 持久化。"""
    node_fn = make_control_node(agent)
    builder: StateGraph = StateGraph(_MinimalState)
    builder.add_node("control", node_fn)
    builder.add_edge(START, "control")
    builder.add_edge("control", END)
    return builder.compile(checkpointer=InMemorySaver())


def _extract_interrupt_payload(result: dict[str, Any]) -> ConfirmRequest | None:
    """从 ainvoke 返回值提取 interrupt payload。

    LangGraph 1.2.6：ainvoke 对 interrupt 不抛异常，而是在 result["__interrupt__"]
    携带 Interrupt 列表。payload 在 interrupts[0].value。
    """
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None
    value = interrupts[0].value
    return value if isinstance(value, ConfirmRequest) else None


# ── 1. DESTRUCTIVE → __interrupt__ 字段含 ConfirmRequest ─────────────────────


async def test_destructive_sets_interrupt_field() -> None:
    """DESTRUCTIVE 动作首次 ainvoke 后 result['__interrupt__'] 含 ConfirmRequest。

    LangGraph 1.2.6：interrupt 结果在返回值 __interrupt__ 字段，不抛异常。
    """
    action = _make_action(action_type="window_close", risk_level=ActionRisk.DESTRUCTIVE)
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.DESTRUCTIVE, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)

    graph = _build_graph(agent)
    config = {"configurable": {"thread_id": "test-thread-interrupt"}}

    result = await graph.ainvoke(_MinimalState(pending_action=action), config=config)

    interrupts = result.get("__interrupt__", [])
    assert len(interrupts) > 0, "result['__interrupt__'] 应含至少一个 Interrupt"
    payload = interrupts[0].value
    assert isinstance(payload, ConfirmRequest), (
        f"interrupt payload 应为 ConfirmRequest，得到 {type(payload)}"
    )
    assert payload.action_id == "test-action-001"
    assert payload.risk_level == ActionRisk.DESTRUCTIVE


# ── 2. interrupt 前无 client 写调用 ───────────────────────────────────────────


async def test_no_write_call_before_interrupt() -> None:
    """DESTRUCTIVE 动作触发 interrupt 时，interrupt 前无任何 client 写调用。"""
    action = _make_action(action_type="window_close", risk_level=ActionRisk.DESTRUCTIVE)
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.DESTRUCTIVE, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)

    graph = _build_graph(agent)
    config = {"configurable": {"thread_id": "test-thread-no-write"}}

    result = await graph.ainvoke(_MinimalState(pending_action=action), config=config)

    # 验证确实触发了 interrupt（否则测试前提不成立）
    assert result.get("__interrupt__"), "期望 interrupt 被触发"

    # interrupt 前不应有任何写操作
    client.close_window.assert_not_called()
    client.click_element.assert_not_called()
    client.type_text.assert_not_called()
    client.send_key.assert_not_called()


# ── 3. resume abort → task_status=FAILED，无 client 写调用 ───────────────────


async def test_resume_abort_returns_failed() -> None:
    """resume confirmed=False 时返回 task_status=FAILED，且无 client 写调用。"""
    action = _make_action(action_type="window_close", risk_level=ActionRisk.DESTRUCTIVE)
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.DESTRUCTIVE, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)

    graph = _build_graph(agent)
    config = {"configurable": {"thread_id": "test-thread-abort"}}

    # 第一次调用 → interrupt
    result1 = await graph.ainvoke(_MinimalState(pending_action=action), config=config)
    assert result1.get("__interrupt__"), "期望 interrupt 被触发"

    # resume 拒绝
    abort_response = ConfirmResponse(confirmed=False, reason="测试拒绝")
    result2 = await graph.ainvoke(Command(resume=abort_response), config=config)

    assert result2.get("task_status") == "FAILED", f"期望 FAILED，得到 {result2.get('task_status')}"
    assert result2.get("control_error") is not None
    # K5 ④：拒绝分支清 pending_action——同一被拒动作不得再次进入 control interrupt
    assert result2.get("pending_action") is None

    # 拒绝后无写操作
    client.close_window.assert_not_called()
    client.click_element.assert_not_called()


# ── 4. resume confirm → client 写调用恰好一次 ────────────────────────────────


async def test_resume_confirm_executes_write_once() -> None:
    """resume confirmed=True 时，client 写调用恰好一次，control_error=None。"""
    action = _make_action(action_type="window_close", risk_level=ActionRisk.DESTRUCTIVE)
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.DESTRUCTIVE, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)

    graph = _build_graph(agent)
    config = {"configurable": {"thread_id": "test-thread-confirm"}}

    # 第一次调用 → interrupt
    result1 = await graph.ainvoke(_MinimalState(pending_action=action), config=config)
    assert result1.get("__interrupt__"), "期望 interrupt 被触发"

    # resume 确认
    result2 = await graph.ainvoke(
        Command(resume=ConfirmResponse(confirmed=True, reason="")),
        config=config,
    )

    # 写操作恰好一次
    client.close_window.assert_called_once_with(window_handle=12345)
    assert result2.get("control_error") is None


# ── 5. TOCTOU abort → control_error，无写操作 ────────────────────────────────


async def test_toctou_abort_no_write() -> None:
    """TOCTOU verify 返回 abort 时，control_error 含相关信息，无任何 client 写调用。"""
    action = _make_action(action_type="window_close", risk_level=ActionRisk.DESTRUCTIVE)
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.DESTRUCTIVE, toctou_verdict="abort")
    agent = DesktopControlAgent(client=client, guard=guard)

    # 直接调 execute（TOCTOU abort 不触发 interrupt，直接返回增量）
    result = await agent.execute(action)

    assert "TOCTOU abort" in result.get("control_error", ""), (
        f"control_error 应含 'TOCTOU abort'，得到 {result.get('control_error')!r}"
    )
    # 三态化负对照：界面真变了的 abort 不得带降级令牌（消费侧靠此区分两种拒绝）
    assert not re.search(r"\[desk:toctou_degraded\]", result.get("control_error", "")), (
        f"「界面已变」abort 的 control_error 不应含 [desk:toctou_degraded]，"
        f"得到 {result.get('control_error')!r}"
    )
    # 无写操作
    client.close_window.assert_not_called()
    client.click_element.assert_not_called()
    client.type_text.assert_not_called()
    client.send_key.assert_not_called()


# ── 5b. TOCTOU abort_degraded → control_error 带机读令牌，无写操作 ────────────


async def test_toctou_abort_degraded_carries_token_no_write() -> None:
    """三态化收口：toctou_verify 返回 abort_degraded 时，control_error 带机读
    令牌 [desk:toctou_degraded]（位置无关，re.search 可提取），且无任何写调用。

    与 test_toctou_abort_no_write 构成判别对：同一消费路径，verdict 不同 →
    control_error 令牌有无不同——修复前两种拒绝的 control_error 文案相同，
    本用例在修复前必红。
    """
    action = _make_action(action_type="window_close", risk_level=ActionRisk.DESTRUCTIVE)
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.DESTRUCTIVE, toctou_verdict="abort_degraded")
    agent = DesktopControlAgent(client=client, guard=guard)

    result = await agent.execute(action)

    control_error = result.get("control_error", "")
    assert re.search(r"\[desk:toctou_degraded\]", control_error), (
        f"abort_degraded 的 control_error 应含机读令牌 [desk:toctou_degraded]，"
        f"得到 {control_error!r}"
    )
    assert "TOCTOU abort" in control_error, (
        f"control_error 应保留 'TOCTOU abort' 前缀口径，得到 {control_error!r}"
    )
    # 无写操作（fail-closed）
    client.close_window.assert_not_called()
    client.click_element.assert_not_called()
    client.type_text.assert_not_called()
    client.send_key.assert_not_called()


# ── 6. LOW_RISK 动作 → 无 interrupt，直接写 ──────────────────────────────────


async def test_low_risk_no_interrupt_direct_write() -> None:
    """LOW_RISK 动作不触发 interrupt，直接执行写操作，结果无 __interrupt__ 字段。"""
    action = _make_action(
        action_type="click",
        risk_level=ActionRisk.LOW_RISK,
        coordinates=(100, 200),
        target_element_id=None,
    )
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)

    graph = _build_graph(agent)
    config = {"configurable": {"thread_id": "test-thread-low-risk"}}

    result = await graph.ainvoke(_MinimalState(pending_action=action), config=config)

    # 无 interrupt
    assert not result.get("__interrupt__"), "LOW_RISK 不应触发 interrupt"
    assert result.get("control_error") is None
    client.click_element.assert_called_once()


# ── 7. state 无 pending_action → control_error ───────────────────────────────


async def test_missing_pending_action_returns_error() -> None:
    """state 无 pending_action 时 control_node 返回 control_error。"""
    client = _make_mock_client()
    guard = _make_mock_guard()
    agent = DesktopControlAgent(client=client, guard=guard)

    graph = _build_graph(agent)
    config = {"configurable": {"thread_id": "test-thread-no-action"}}

    result = await graph.ainvoke(_MinimalState(pending_action=None), config=config)

    assert result.get("control_error") is not None
    assert "pending_action" in result["control_error"]


# ── 8. _build_control_increment 辅助函数独立测试 ─────────────────────────────


def test_build_control_increment_success() -> None:
    """_build_control_increment 成功时 control_error=None 且清 pending_action（K4 ③）。"""
    action = _make_action()
    result = ActionResult(
        action_id="test-action-001",
        success=True,
        error_message=None,
        ui_changed=True,
    )
    increment = _build_control_increment(action, result)
    assert increment == {"control_error": None, "pending_action": None}


def test_build_control_increment_failure() -> None:
    """_build_control_increment 失败时 control_error 包含 action_id 和错误信息。"""
    action = _make_action()
    result = ActionResult(
        action_id="test-action-001",
        success=False,
        error_message="窗口不存在",
        ui_changed=False,
    )
    increment = _build_control_increment(action, result)
    assert increment.get("control_error") is not None
    assert "test-action-001" in increment["control_error"]
    assert "窗口不存在" in increment["control_error"]
    # K4 ③：失败同样清 pending_action（动作已执行完毕，不得原样重放）
    assert "pending_action" in increment and increment["pending_action"] is None


# ── 9. dispatch_write 路由测试 ────────────────────────────────────────────────


async def test_dispatch_type_text() -> None:
    """type_text 动作正确路由到 client.type_text。"""
    action = ActionSpec(
        action_id="act-type",
        action_type="type",
        target_element_id=None,
        coordinates=None,
        text_payload="hello world",
        risk_level=ActionRisk.LOW_RISK,
    )
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)

    await agent.execute(action)

    client.type_text.assert_called_once_with(text="hello world")


async def test_dispatch_send_key() -> None:
    """send_key 动作正确路由到 client.send_key。"""
    action = ActionSpec(
        action_id="act-key",
        action_type="key",
        target_element_id=None,
        coordinates=None,
        text_payload="ctrl+c",
        risk_level=ActionRisk.LOW_RISK,
    )
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)

    await agent.execute(action)

    client.send_key.assert_called_once_with(key_combo="ctrl+c")


async def test_dispatch_unknown_action_type_returns_error() -> None:
    """未知 action_type 不崩溃，返回 control_error。"""
    action = ActionSpec(
        action_id="act-unknown",
        action_type="unknown_op",
        target_element_id=None,
        coordinates=None,
        text_payload=None,
        risk_level=ActionRisk.LOW_RISK,
    )
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)

    result = await agent.execute(action)

    assert result.get("control_error") is not None
    assert "未知 action_type" in result["control_error"]


# ── 10. window_close 缺 target_element_id → control_error ───────────────────


async def test_window_close_missing_handle() -> None:
    """window_close 缺 target_element_id 时 control_error 包含提示，无写操作。"""
    action = ActionSpec(
        action_id="act-close",
        action_type="window_close",
        target_element_id=None,
        coordinates=None,
        text_payload=None,
        risk_level=ActionRisk.LOW_RISK,  # LOW_RISK 不触发 interrupt，路径更简单
    )
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)

    result = await agent.execute(action)

    assert result.get("control_error") is not None
    client.close_window.assert_not_called()


# ── 11. MCP 调用异常被 control_node 捕获 ─────────────────────────────────────


async def test_mcp_call_error_caught_by_control_node() -> None:
    """client 写操作抛 DesktopMCPCallError 时，control_node 捕获并返回 control_error。

    F4 语义修订（code-review WARN）：本用例原先只断言 control_error 非空，未
    对 pending_action 表态。新语义下 MCP 异常分支须清空 pending_action（见
    _control_node docstring 的 at-least-once 论证）——本用例补上该断言，
    与专门的 test_mcp_call_error_clears_pending_action 互为正反面覆盖
    （该用例另做变异实证）。
    """
    from src.mcp.desktop_mcp_client import DesktopMCPCallError

    client = _make_mock_client()
    client.click_element = AsyncMock(side_effect=DesktopMCPCallError("click_element", "调用失败"))
    guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)

    graph = _build_graph(agent)
    config = {"configurable": {"thread_id": "test-thread-mcp-error"}}
    initial_state = _MinimalState(
        pending_action=ActionSpec(
            action_id="test-action-001",
            action_type="click",
            target_element_id=None,
            coordinates=(10, 20),
            text_payload=None,
            risk_level=ActionRisk.LOW_RISK,
        )
    )

    result = await graph.ainvoke(initial_state, config=config)

    assert result.get("control_error") is not None
    # F4：MCP 异常分支清 pending_action，防 Supervisor 原样重放非幂等写操作
    assert result.get("pending_action") is None


# ── 12. ConfirmRequest/ConfirmResponse 模型测试 ───────────────────────────────


def test_confirm_request_model() -> None:
    """ConfirmRequest Pydantic 模型基本字段验证。"""
    req = ConfirmRequest(
        action_id="a1",
        action_type="window_close",
        risk_level=ActionRisk.DESTRUCTIVE,
        description="关闭窗口",
    )
    assert req.action_id == "a1"
    assert req.risk_level == ActionRisk.DESTRUCTIVE
    assert req.coordinates is None


def test_confirm_response_default_reason() -> None:
    """ConfirmResponse reason 默认为空字符串。"""
    resp = ConfirmResponse(confirmed=True)
    assert resp.confirmed is True
    assert resp.reason == ""


# ── 13. interrupt payload 字段完整性 ─────────────────────────────────────────


async def test_interrupt_payload_fields() -> None:
    """__interrupt__ payload 包含 action_id / action_type / risk_level / description。"""
    action = _make_action(action_type="window_close", risk_level=ActionRisk.DESTRUCTIVE)
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.DESTRUCTIVE, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)

    graph = _build_graph(agent)
    config = {"configurable": {"thread_id": "test-thread-payload"}}

    result = await graph.ainvoke(_MinimalState(pending_action=action), config=config)

    payload = _extract_interrupt_payload(result)
    assert payload is not None, "未找到 ConfirmRequest payload"
    assert payload.action_type == "window_close"
    assert payload.description, "description 不应为空"
    assert payload.risk_level == ActionRisk.DESTRUCTIVE


# ── 14. 多 thread_id 隔离 ─────────────────────────────────────────────────────


async def test_multiple_threads_isolated() -> None:
    """不同 thread_id 的 interrupt/resume 互不影响（InMemorySaver 隔离）。"""
    action = _make_action(action_type="window_close", risk_level=ActionRisk.DESTRUCTIVE)
    client_a = _make_mock_client()
    client_b = _make_mock_client()
    guard_a = _make_mock_guard(risk=ActionRisk.DESTRUCTIVE, toctou_verdict="pass")
    guard_b = _make_mock_guard(risk=ActionRisk.DESTRUCTIVE, toctou_verdict="pass")
    agent_a = DesktopControlAgent(client=client_a, guard=guard_a)
    agent_b = DesktopControlAgent(client=client_b, guard=guard_b)

    graph_a = _build_graph(agent_a)
    graph_b = _build_graph(agent_b)

    config_a = {"configurable": {"thread_id": "thread-A"}}
    config_b = {"configurable": {"thread_id": "thread-B"}}

    # 两个图各自 interrupt
    result_a1 = await graph_a.ainvoke(_MinimalState(pending_action=action), config=config_a)
    result_b1 = await graph_b.ainvoke(_MinimalState(pending_action=action), config=config_b)

    assert result_a1.get("__interrupt__"), "graph_a 应触发 interrupt"
    assert result_b1.get("__interrupt__"), "graph_b 应触发 interrupt"

    # 只 resume A，确认；B 不 resume
    result_a2 = await graph_a.ainvoke(
        Command(resume=ConfirmResponse(confirmed=True)),
        config=config_a,
    )
    assert result_a2.get("control_error") is None
    client_a.close_window.assert_called_once()
    client_b.close_window.assert_not_called()


# ── 15. patch lg_interrupt 单元测试（不经图，验分支逻辑） ────────────────────


async def test_execute_destructive_calls_lg_interrupt() -> None:
    """直接 patch lg_interrupt，验证 DESTRUCTIVE 路径调用了 interrupt 且参数是 ConfirmRequest。"""
    action = _make_action(action_type="window_close", risk_level=ActionRisk.DESTRUCTIVE)
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.DESTRUCTIVE, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)

    # patch lg_interrupt 使其直接返回 ConfirmResponse（模拟 resume 后的重放取值）
    confirm_resp = ConfirmResponse(confirmed=True)
    with patch(
        "src.agents.desktop_control_agent.lg_interrupt",
        return_value=confirm_resp,
    ) as mock_interrupt:
        result = await agent.execute(action)

    # interrupt 被调用了一次，参数是 ConfirmRequest
    mock_interrupt.assert_called_once()
    call_arg = mock_interrupt.call_args[0][0]
    assert isinstance(call_arg, ConfirmRequest)

    # 确认后 close_window 被调用一次
    client.close_window.assert_called_once_with(window_handle=12345)
    assert result.get("control_error") is None


async def test_execute_destructive_abort_via_patch() -> None:
    """patch lg_interrupt 返回 confirmed=False，验证 FAILED 分支。"""
    action = _make_action(action_type="window_close", risk_level=ActionRisk.DESTRUCTIVE)
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.DESTRUCTIVE, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)

    abort_resp = ConfirmResponse(confirmed=False, reason="手动取消")
    with patch(
        "src.agents.desktop_control_agent.lg_interrupt",
        return_value=abort_resp,
    ):
        result = await agent.execute(action)

    assert result.get("task_status") == "FAILED"
    client.close_window.assert_not_called()


# ── 16. DESTRUCTIVE resume confirm 后节点重放时 guard 被调用两次 ──────────────


async def test_destructive_resume_guard_called_on_replay() -> None:
    """LangGraph 重放铁律：resume 时节点整体重放，classify_risk / toctou_verify 各被调用两次。

    interrupt 前只读操作在重放时幂等重跑，这是预期行为（非副作用）。
    """
    action = _make_action(action_type="window_close", risk_level=ActionRisk.DESTRUCTIVE)
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.DESTRUCTIVE, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)

    graph = _build_graph(agent)
    config = {"configurable": {"thread_id": "test-thread-replay"}}

    # 第一次：interrupt
    await graph.ainvoke(_MinimalState(pending_action=action), config=config)
    # 第二次：resume confirm
    await graph.ainvoke(Command(resume=ConfirmResponse(confirmed=True)), config=config)

    # 节点重放：guard 各被调用两次（第一次 + 重放各一次）
    assert guard.classify_risk.call_count == 2, (
        f"classify_risk 应被调用 2 次（初次+重放），实际 {guard.classify_risk.call_count}"
    )
    assert guard.toctou_verify.call_count == 2, (
        f"toctou_verify 应被调用 2 次（初次+重放），实际 {guard.toctou_verify.call_count}"
    )
    # 写操作只执行一次（resume 后才执行）
    client.close_window.assert_called_once()


# ── 17. window_close target_element_id 非整数字符串 ──────────────────────────


async def test_window_close_invalid_handle() -> None:
    """window_close target_element_id 非整数字符串时返回 control_error，无写操作。"""
    action = ActionSpec(
        action_id="act-invalid",
        action_type="window_close",
        target_element_id="not-a-number",
        coordinates=None,
        text_payload=None,
        risk_level=ActionRisk.LOW_RISK,
    )
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)

    result = await agent.execute(action)

    assert result.get("control_error") is not None
    assert "无法转换为整数" in result["control_error"]
    client.close_window.assert_not_called()


# ── 18. READ_ONLY 动作跳过 TOCTOU，直接执行 ──────────────────────────────────


async def test_read_only_skips_toctou_and_executes() -> None:
    """READ_ONLY 动作 toctou_verify 返回 pass（READ_ONLY 且无坐标跳过 TOCTOU），直接执行。"""
    action = ActionSpec(
        action_id="act-ro",
        action_type="click",
        target_element_id=None,
        coordinates=None,
        text_payload=None,
        risk_level=ActionRisk.READ_ONLY,
    )
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.READ_ONLY, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)

    result = await agent.execute(action)

    # 无 interrupt，直接写
    assert result.get("control_error") is None
    client.click_element.assert_called_once()


# ── 19. click_element 动作名别名测试 ─────────────────────────────────────────


async def test_dispatch_click_element_alias() -> None:
    """action_type='click_element' 与 'click' 等价，均路由到 client.click_element。"""
    action = ActionSpec(
        action_id="act-click-alias",
        action_type="click_element",
        target_element_id="btn-ok",
        coordinates=None,
        text_payload=None,
        risk_level=ActionRisk.LOW_RISK,
    )
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)

    await agent.execute(action)

    client.click_element.assert_called_once_with(
        automation_id="btn-ok",
        coordinates=None,
        expected_root_hwnd=None,
    )


# ── 20. 无 pending_action（dict state）→ control_error ───────────────────────


async def test_missing_pending_action_dict_state() -> None:
    """state 为 dict 且无 pending_action 时，control_node 正确处理。"""
    client = _make_mock_client()
    guard = _make_mock_guard()

    # 直接测 _control_node 对 dict state 的处理
    agent = DesktopControlAgent(client=client, guard=guard)
    node_fn = make_control_node(agent)

    result = await node_fn({"control_error": None})  # dict state，无 pending_action

    assert result.get("control_error") is not None
    assert "pending_action" in result["control_error"]


# ── 21. K1 ④：execute 把 effective_risk 传入 toctou_verify ───────────────────


async def test_execute_passes_effective_risk_to_toctou_verify() -> None:
    """K1 ④：execute 把 classify_risk 结果作为 effective_risk 传入 toctou_verify。

    场景：声明 READ_ONLY，guard.classify_risk 升级为 LOW_RISK——toctou_verify
    必须收到升级后的 LOW_RISK（而非声明值），否则低报动作会跳过 TOCTOU。
    """
    action = ActionSpec(
        action_id="act-eff-risk",
        action_type="click",
        target_element_id=None,
        coordinates=(10, 20),
        text_payload=None,
        risk_level=ActionRisk.READ_ONLY,
    )
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)

    await agent.execute(action)

    kwargs = guard.toctou_verify.call_args.kwargs
    assert kwargs.get("effective_risk") == ActionRisk.LOW_RISK, (
        f"toctou_verify 应收到 classify_risk 升级后的 effective_risk=LOW_RISK，"
        f"实际 kwargs={kwargs!r}"
    )


# ── 22. K1 ⑤：make_control_node snapshot_store 注入与新鲜度门 ────────────────


def _make_store_snapshot(
    screenshot_path: str | None,
    timestamp_ms: int,
) -> ScreenSnapshot:
    """构造 SnapshotStore 返回的最小 ScreenSnapshot（新鲜度门测试用）。"""
    return ScreenSnapshot(
        snapshot_id="snap-ctrl-store",
        timestamp_ms=timestamp_ms,
        screen_width=1920,
        screen_height=1080,
        active_window_title="Test",
        uia_elements=[],
        text_blocks=[],
        visual_objects=[],
        screenshot_path=screenshot_path,
        perception_mode="uia_only",
        capability_flags={},
        uia_hollow=False,
    )


def _write_gray_png(path: str, gray_value: int = 128) -> None:
    """写一张 64x64 单色灰度 PNG（TOCTOU phash 比对用，同 test_action_guard 口径）。"""
    img = np.full((64, 64), gray_value, dtype=np.uint8)
    cv2.imwrite(path, img)


def _make_low_risk_click_no_coords() -> ActionSpec:
    """LOW_RISK 无坐标 click：触发 TOCTOU（整图口径）但不触发 interrupt。"""
    return ActionSpec(
        action_id="act-store-001",
        action_type="click",
        target_element_id=None,
        coordinates=None,
        text_payload=None,
        risk_level=ActionRisk.LOW_RISK,
    )


async def test_control_node_passes_fresh_snapshot_to_guard() -> None:
    """K1 ⑤：新鲜 snapshot_ref → 加载的快照作为 snapshot_before 传入 guard。"""
    import time

    fresh_snap = _make_store_snapshot(None, timestamp_ms=int(time.time() * 1000))
    store = MagicMock()
    store.load = AsyncMock(return_value=fresh_snap)

    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)
    node_fn = make_control_node(agent, snapshot_store=store)

    await node_fn({"pending_action": _make_low_risk_click_no_coords(), "snapshot_ref": "snap-1"})

    store.load.assert_awaited_once_with("snap-1")
    kwargs = guard.toctou_verify.call_args.kwargs
    assert kwargs.get("snapshot_before") is fresh_snap


async def test_control_node_stale_snapshot_passes_none_to_guard() -> None:
    """K1 ⑤ 新鲜度门：过旧快照（timestamp 远古）不复用，snapshot_before=None。"""
    stale_snap = _make_store_snapshot(None, timestamp_ms=1000)  # 1970 年，远超 5000ms
    store = MagicMock()
    store.load = AsyncMock(return_value=stale_snap)

    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)
    node_fn = make_control_node(agent, snapshot_store=store)

    await node_fn({"pending_action": _make_low_risk_click_no_coords(), "snapshot_ref": "snap-2"})

    kwargs = guard.toctou_verify.call_args.kwargs
    assert kwargs.get("snapshot_before") is None


async def test_control_node_fresh_snapshot_saves_one_rpc(tmp_path: Any) -> None:
    """K1 ⑤（真 ActionGuard）：新鲜快照复用为 TOCTOU 基线——screen_snapshot 少调一次。"""
    import time

    img_a = str(tmp_path / "before.png")
    img_b = str(tmp_path / "after.png")
    _write_gray_png(img_a, gray_value=128)
    _write_gray_png(img_b, gray_value=128)

    fresh_snap = _make_store_snapshot(img_a, timestamp_ms=int(time.time() * 1000))
    store = MagicMock()
    store.load = AsyncMock(return_value=fresh_snap)

    client = _make_mock_client()
    client.screen_snapshot = AsyncMock(
        side_effect=[_make_store_snapshot(img_b, timestamp_ms=int(time.time() * 1000))]
    )
    guard = ActionGuard(client=client)  # 真安全门
    agent = DesktopControlAgent(client=client, guard=guard)
    node_fn = make_control_node(agent, snapshot_store=store)

    with patch("src.orchestration.safety.action_guard.asyncio.sleep", new_callable=AsyncMock):
        result = await node_fn(
            {"pending_action": _make_low_risk_click_no_coords(), "snapshot_ref": "snap-3"}
        )

    # 第一张基线来自 store 快照，只重拍了第二张
    assert client.screen_snapshot.call_count == 1
    assert result.get("control_error") is None
    client.click_element.assert_called_once()


async def test_control_node_stale_snapshot_reshoots_both(tmp_path: Any) -> None:
    """K1 ⑤（真 ActionGuard）：过旧快照不复用——两张截图都重拍（现行为）。"""
    import time

    img_a = str(tmp_path / "before.png")
    img_b = str(tmp_path / "after.png")
    _write_gray_png(img_a, gray_value=128)
    _write_gray_png(img_b, gray_value=128)

    stale_snap = _make_store_snapshot(img_a, timestamp_ms=1000)
    store = MagicMock()
    store.load = AsyncMock(return_value=stale_snap)

    client = _make_mock_client()
    client.screen_snapshot = AsyncMock(
        side_effect=[
            _make_store_snapshot(img_a, timestamp_ms=int(time.time() * 1000)),
            _make_store_snapshot(img_b, timestamp_ms=int(time.time() * 1000)),
        ]
    )
    guard = ActionGuard(client=client)
    agent = DesktopControlAgent(client=client, guard=guard)
    node_fn = make_control_node(agent, snapshot_store=store)

    with patch("src.orchestration.safety.action_guard.asyncio.sleep", new_callable=AsyncMock):
        result = await node_fn(
            {"pending_action": _make_low_risk_click_no_coords(), "snapshot_ref": "snap-4"}
        )

    assert client.screen_snapshot.call_count == 2
    assert result.get("control_error") is None


async def test_control_node_store_load_failure_falls_back() -> None:
    """K1 ⑤ 防御：store.load 抛异常时回退重拍（snapshot_before=None），不阻断执行。"""
    store = MagicMock()
    store.load = AsyncMock(side_effect=RuntimeError("存储不可用"))

    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)
    node_fn = make_control_node(agent, snapshot_store=store)

    result = await node_fn(
        {"pending_action": _make_low_risk_click_no_coords(), "snapshot_ref": "snap-5"}
    )

    kwargs = guard.toctou_verify.call_args.kwargs
    assert kwargs.get("snapshot_before") is None
    assert result.get("control_error") is None


async def test_control_node_without_store_keeps_current_behavior() -> None:
    """K1 ⑤ 零回归：不注入 snapshot_store 时 snapshot_before 恒为 None（现行为）。"""
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)
    node_fn = make_control_node(agent)

    result = await node_fn({"pending_action": _make_low_risk_click_no_coords()})

    kwargs = guard.toctou_verify.call_args.kwargs
    assert kwargs.get("snapshot_before") is None
    assert result.get("control_error") is None


# ── 23. F1（code-review BLOCK）：_dispatch_write 落点核验参数透传 ────────────


async def test_dispatch_click_passes_expected_root_hwnd_to_client() -> None:
    """F1：ActionSpec.expected_root_hwnd 设为具体整数时，client.click_element
    必须收到同一个值——_dispatch_write 是唯一集中调用点，漏传会让落点核验
    （K7 批2）在 Agent 执行链路上静默失效（client 侧支持但收不到实参）。
    """
    action = ActionSpec(
        action_id="act-hwnd",
        action_type="click",
        target_element_id=None,
        coordinates=(10, 20),
        text_payload=None,
        risk_level=ActionRisk.LOW_RISK,
        expected_root_hwnd=98765,
    )
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)

    await agent.execute(action)

    client.click_element.assert_called_once_with(
        automation_id=None,
        coordinates=(10, 20),
        expected_root_hwnd=98765,
    )


async def test_dispatch_click_default_expected_root_hwnd_is_none() -> None:
    """F1 零回归对照：ActionSpec.expected_root_hwnd 未设置（默认 None）时，
    client.click_element 收到的 expected_root_hwnd 同样为 None（不核验落点）。
    """
    action = ActionSpec(
        action_id="act-hwnd-default",
        action_type="click_element",
        target_element_id="btn-cancel",
        coordinates=None,
        text_payload=None,
        risk_level=ActionRisk.LOW_RISK,
    )
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)

    await agent.execute(action)

    client.click_element.assert_called_once_with(
        automation_id="btn-cancel",
        coordinates=None,
        expected_root_hwnd=None,
    )


# ── 24. F4（code-review WARN）：MCP 异常分支清 pending_action ────────────────


async def test_mcp_call_error_clears_pending_action() -> None:
    """F4：DesktopMCPCallError 分支须清空 pending_action。

    RPC 已发出、结果未知（at-least-once）——保留 pending_action 会让
    Supervisor 把它当「未执行」原样重放，对非幂等写操作（如发消息）有重复
    副作用风险；清空后 Supervisor 需重新规划（经 node_fn 直接验证节点级
    增量，避开图 state 合并语义）。
    """
    from src.mcp.desktop_mcp_client import DesktopMCPCallError

    client = _make_mock_client()
    client.click_element = AsyncMock(side_effect=DesktopMCPCallError("click_element", "调用失败"))
    guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)
    node_fn = make_control_node(agent)

    action = ActionSpec(
        action_id="act-mcp-err",
        action_type="click",
        target_element_id=None,
        coordinates=(1, 2),
        text_payload=None,
        risk_level=ActionRisk.LOW_RISK,
    )

    increment = await node_fn({"pending_action": action})

    assert increment.get("control_error") is not None
    assert "pending_action" in increment and increment["pending_action"] is None


async def test_mcp_connection_error_clears_pending_action() -> None:
    """F4：DesktopMCPConnectionError 分支同样清空 pending_action（同一 except 元组）。"""
    from src.mcp.desktop_mcp_client import DesktopMCPConnectionError

    client = _make_mock_client()
    client.click_element = AsyncMock(side_effect=DesktopMCPConnectionError("连接断开"))
    guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)
    node_fn = make_control_node(agent)

    action = ActionSpec(
        action_id="act-mcp-conn-err",
        action_type="click",
        target_element_id=None,
        coordinates=(1, 2),
        text_payload=None,
        risk_level=ActionRisk.LOW_RISK,
    )

    increment = await node_fn({"pending_action": action})

    assert increment.get("control_error") is not None
    assert "pending_action" in increment and increment["pending_action"] is None


async def test_unexpected_exception_clears_pending_action() -> None:
    """F4 复核追加：except Exception 兜底分支同样清空 pending_action。

    该分支可捕获「RPC 响应已返回、解析/后处理才抛」的异常（如 ActionResult
    响应体 ValidationError）——server 大概率已真执行写操作，比连接断开更接近
    「已执行」，保留 pending_action 的重放风险更高，故与 MCP 异常分支同语义。
    """
    client = _make_mock_client()
    client.click_element = AsyncMock(side_effect=ValueError("响应体解析失败"))
    guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
    agent = DesktopControlAgent(client=client, guard=guard)
    node_fn = make_control_node(agent)

    action = ActionSpec(
        action_id="act-unexpected-err",
        action_type="click",
        target_element_id=None,
        coordinates=(1, 2),
        text_payload=None,
        risk_level=ActionRisk.LOW_RISK,
    )

    increment = await node_fn({"pending_action": action})

    assert (increment.get("control_error") or "").startswith("unexpected:")
    assert "pending_action" in increment and increment["pending_action"] is None


async def test_toctou_abort_retains_pending_action_at_node_level() -> None:
    """F4 对照（零回归）：TOCTOU abort 发生在写操作派发前（只读区），语义不变。

    node 级增量不含 pending_action 键（LastValue 保留旧值）——重试前仍会
    重新经过 TOCTOU 核验，不构成 at-least-once 风险，故不清空。
    """
    action = _make_low_risk_click_no_coords()
    client = _make_mock_client()
    guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="abort")
    agent = DesktopControlAgent(client=client, guard=guard)
    node_fn = make_control_node(agent)

    increment = await node_fn({"pending_action": action})

    assert "TOCTOU abort" in (increment.get("control_error") or "")
    assert "pending_action" not in increment


# ── pytest 配置提示（不是测试函数） ──────────────────────────────────────────

# asyncio_mode=auto 已在 pyproject.toml 配置，所有 async def test_* 自动异步执行。
# 无需 @pytest.mark.asyncio 装饰器。
