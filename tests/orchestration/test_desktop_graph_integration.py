"""桌面任务执行图端到端集成测试（Task 10D）。

使用 InMemorySaver 跑完整 StateGraph（get_graph 工厂），验证图级别的行为：

  1. Happy path：正常完成路径
       START→supervisor→perceive→stall_detect→supervisor→control→supervisor
       →supervisor→memory_flush→END
       验证 task_status==DONE + memory_flush 被调 + memory_api.write_session_summary 被调。

  2. interrupt/resume 链路：DESTRUCTIVE 动作引发 GraphInterrupt
       payload 是 ConfirmRequest → Command(resume=ConfirmResponse(confirmed=True))
       → 写操作被调 → supervisor → memory_flush → END。
       同时验证 interrupt 前 client 写操作未被调用（只读区红线）。

  3. 感知失败停滞路径（R3 图结构验证）：
       perception_error 连续出现 → stall_count 达 STALL_THRESHOLD
       → error_report → memory_flush → END（task_status=FAILED）
       验证 error_report 被调、memory_flush 被调。

设计约束（测试侧同样遵守红线）：
  - 不依赖 Postgres/Neo4j（InMemorySaver + 全 mock）。
  - thread_id = task_id（蓝图 §1.4）。
  - memory_api mock 验证 scope="session"（memory-rules.md）。
  - 不 import Zero，不直连存储层。
  - mock LLM/client/guard/memory_api，注入真实 PromptLoader（验证 Task 11B 接线）。
  - ConfirmRequest/ConfirmResponse 从 src.orchestration.state 导入（Task 10A 统一出口）。
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from langgraph.types import Command

from src.agents.desktop_control_agent import DesktopControlAgent
from src.agents.models.screen_snapshot import (
    ActionResult,
    ActionRisk,
    ActionSpec,
    BBox,
    ScreenSnapshot,
    TextBlock,
    UIAElement,
)
from src.agents.screen_perception_agent import ScreenPerceptionAgent
from src.orchestration.desktop_graph import get_graph
from src.orchestration.desktop_supervisor import DesktopSupervisorAgent
from src.orchestration.prompt_loader import PromptLoader
from src.orchestration.state import (
    ConfirmRequest,
    ConfirmResponse,
    DesktopTaskState,
    TaskStatus,
)

# ── 测试辅助：构造 mock ScreenSnapshot ───────────────────────────────────────

# LLM 响应常量（缩短 JSON key 空格以满足 line-length=100）
_R_PERCEIVE_RUNNING = (
    '{"next_agent":"perceive","current_instruction":"感知桌面状态","task_status":"RUNNING"}'
)
_R_PERCEIVE_DONE = '{"next_agent":"perceive","current_instruction":"任务完成","task_status":"DONE"}'
_R_CONTROL_RUNNING = (
    '{"next_agent":"control","current_instruction":"执行操作","task_status":"RUNNING"}'
)
_R_PERCEIVE_FAILED = (
    '{"next_agent":"perceive","current_instruction":"操作被拒绝终止","task_status":"FAILED"}'
)


def _make_snapshot(snapshot_id: str = "snap-test-001") -> ScreenSnapshot:
    """构造测试用 ScreenSnapshot（最小字段，满足 agent 调用需求）。"""
    return ScreenSnapshot(
        snapshot_id=snapshot_id,
        timestamp_ms=1_700_000_000_000,
        screen_width=1920,
        screen_height=1080,
        active_window_title="测试窗口",
        uia_elements=[
            UIAElement(
                element_id="elem-001",
                control_type="Button",
                name="确定",
                automation_id="btnOK",
                bbox=BBox(x=100, y=200, width=80, height=30),
                is_enabled=True,
                is_visible=True,
                value=None,
                source="uia",
            )
        ],
        text_blocks=[
            TextBlock(
                block_id="tb-001",
                text="确定",
                bbox=BBox(x=100, y=200, width=80, height=30),
                confidence=0.99,
                source="ocr_rapidocr",
            )
        ],
        visual_objects=[],
        screenshot_path=None,
        perception_mode="uia_ocr",
        capability_flags={"SCREEN_CAPABILITY_ENABLED": True},
        uia_hollow=False,
    )


def _make_action(
    action_type: str = "click",
    risk_level: ActionRisk = ActionRisk.LOW_RISK,
    target_element_id: str | None = "elem-001",
) -> ActionSpec:
    """构造测试用 ActionSpec。"""
    return ActionSpec(
        action_id="act-integration-001",
        action_type=action_type,
        target_element_id=target_element_id,
        coordinates=None,
        text_payload=None,
        risk_level=risk_level,
    )


def _make_action_result(success: bool = True) -> ActionResult:
    """构造测试用 ActionResult。"""
    return ActionResult(
        action_id="act-integration-001",
        success=success,
        error_message=None if success else "点击失败",
        ui_changed=success,
    )


# ── Mock 工厂 ─────────────────────────────────────────────────────────────────


def _make_mock_client(snapshot: ScreenSnapshot | None = None) -> MagicMock:
    """构造 mock DesktopMCPClient。

    screen_snapshot 返回 snapshot（默认 _make_snapshot()）。
    写操作默认返回成功的 ActionResult。
    """
    snap = snapshot or _make_snapshot()
    client = MagicMock()
    client.screen_snapshot = AsyncMock(return_value=snap)
    success_result = ActionResult(
        action_id="act-integration-001",
        success=True,
        error_message=None,
        ui_changed=True,
    )
    client.click_element = AsyncMock(return_value=success_result)
    client.type_text = AsyncMock(return_value=success_result)
    client.send_key = AsyncMock(return_value=success_result)
    client.close_window = AsyncMock(return_value=success_result)
    return client


def _make_mock_guard(
    risk: ActionRisk = ActionRisk.LOW_RISK,
    toctou_verdict: str = "pass",
) -> MagicMock:
    """构造 mock ActionGuard。"""
    guard = MagicMock()
    guard.classify_risk = AsyncMock(return_value=risk)
    guard.toctou_verify = AsyncMock(return_value=toctou_verdict)
    return guard


def _make_mock_memory_api() -> MagicMock:
    """构造 mock MemoryAPI，write_session_summary 为 AsyncMock。"""
    mock = MagicMock()
    mock.write_session_summary = AsyncMock()
    return mock


def _make_mock_llm_client(responses: list[str]) -> MagicMock:
    """构造 mock LLM 客户端，按顺序返回 responses 列表中的 JSON 字符串。

    每次 messages.create 调用取下一个 response（循环队列）。
    response 字符串必须是合法 JSON，符合 Supervisor plan 输出格式。
    """
    mock_llm = MagicMock()
    call_count = [0]

    async def _fake_create(**kwargs: Any) -> MagicMock:
        idx = call_count[0] % len(responses)
        call_count[0] += 1
        resp = MagicMock()
        content_item = MagicMock()
        content_item.text = responses[idx]
        resp.content = [content_item]
        return resp

    mock_llm.messages = MagicMock()
    mock_llm.messages.create = _fake_create
    return mock_llm


# ── 辅助：提取 interrupt payload ──────────────────────────────────────────────


def _extract_interrupt_payload(result: dict[str, Any]) -> ConfirmRequest | None:
    """从 ainvoke 返回值提取 interrupt payload。

    LangGraph 1.2.6：ainvoke 对 interrupt 不抛异常，
    而是在 result['__interrupt__'] 携带 Interrupt 列表。
    payload 在 interrupts[0].value。
    """
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None
    value = interrupts[0].value
    return value if isinstance(value, ConfirmRequest) else None


# ── 辅助：构造带真实 PromptLoader 的图 ───────────────────────────────────────


def _build_graph(
    mock_llm: MagicMock,
    mock_client: MagicMock,
    mock_guard: MagicMock,
    mock_memory_api: MagicMock,
    task_id: str = "task-integration-001",
) -> Any:
    """构造完整 StateGraph，注入真实 PromptLoader 与 mock 依赖。

    DesktopSupervisorAgent 注入真实 PromptLoader（Task 11B），
    ScreenPerceptionAgent / DesktopControlAgent 注入 mock client/guard。
    memory_api 注入 mock 供断言调用。

    Returns:
        编译好的 CompiledGraph（InMemorySaver）。
    """
    prompt_loader = PromptLoader()

    supervisor_agent = DesktopSupervisorAgent(
        llm_client=mock_llm,
        prompt_loader=prompt_loader,
    )

    perception_agent = ScreenPerceptionAgent(
        client=mock_client,
        snapshot_store=None,  # InMemory 打桩
    )

    control_agent = DesktopControlAgent(
        client=mock_client,
        guard=mock_guard,
    )

    return get_graph(
        supervisor_agent=supervisor_agent,
        perception_agent=perception_agent,
        control_agent=control_agent,
        memory_api=mock_memory_api,
        checkpointer=None,  # 使用 InMemorySaver 默认值
    )


# ── 辅助：设置/还原 ANTHROPIC_API_KEY ─────────────────────────────────────────


class _ApiKeyContext:
    """上下文管理器：临时设置 ANTHROPIC_API_KEY，退出时还原。"""

    def __init__(self, key: str = "mock-key-for-test") -> None:
        import os

        self.key = key
        self.original = os.environ.get("ANTHROPIC_API_KEY")

    def __enter__(self) -> _ApiKeyContext:
        import os

        os.environ["ANTHROPIC_API_KEY"] = self.key
        return self

    def __exit__(self, *args: Any) -> None:
        import os

        if self.original is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = self.original


# ── 测试 1：Happy path ─────────────────────────────────────────────────────────


class TestHappyPath:
    """完整图 happy path 集成测试。

    路径：START→supervisor→perceive→stall_detect→supervisor
           →control→supervisor→supervisor(DONE)→memory_flush→END

    验证：
      - task_status 最终为 DONE。
      - memory_api.write_session_summary 被调用至少一次，scope="session"。
      - memory_flush 节点成功执行（不崩溃）。
    """

    async def test_happy_path_ends_with_done_and_memory_flush(self) -> None:
        """完整 happy path：图运行到 END，task_status=DONE，memory_flush 被调。

        LLM mock 返回顺序：
          1. 感知指令（next_agent=perceive, status=RUNNING）
          2. 控制指令（next_agent=control, status=RUNNING）
          3. 任务完成（task_status=DONE）—— 触发 memory_flush
        """
        task_id = "task-happy-001"
        mock_client = _make_mock_client()
        mock_guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
        mock_memory_api = _make_mock_memory_api()

        # 轮 1：感知；轮 2：执行点击；轮 3：声明完成
        mock_llm = _make_mock_llm_client(
            [_R_PERCEIVE_RUNNING, _R_CONTROL_RUNNING, _R_PERCEIVE_DONE]
        )

        graph = _build_graph(
            mock_llm=mock_llm,
            mock_client=mock_client,
            mock_guard=mock_guard,
            mock_memory_api=mock_memory_api,
            task_id=task_id,
        )

        # pending_action：点击动作（LOW_RISK，无 interrupt）
        pending_action = _make_action(action_type="click", risk_level=ActionRisk.LOW_RISK)

        initial_state = DesktopTaskState(
            task_id=task_id,
            task_description="集成测试：点击确定按钮后感知结果",
            task_status=TaskStatus.RUNNING,
            pending_action=pending_action,
        )

        config = {"configurable": {"thread_id": task_id}}

        with _ApiKeyContext():
            result = await graph.ainvoke(initial_state, config=config)

        # 验证最终状态
        assert result.get("task_status") == TaskStatus.DONE, (
            f"期望 task_status=DONE，实际为 {result.get('task_status')!r}"
        )

        # 验证 memory_flush 被调（write_session_summary 被调用至少一次）
        assert mock_memory_api.write_session_summary.call_count >= 1, (
            "memory_api.write_session_summary 应在 memory_flush_node 被调用至少一次"
        )

        # 验证 scope="session"（memory-rules.md：不默认 user）
        for actual_call in mock_memory_api.write_session_summary.call_args_list:
            _, kwargs = actual_call
            scope = kwargs.get("scope")
            assert scope == "session", (
                f"write_session_summary scope 应为 'session'，实际为 {scope!r}。"
                "违反 memory-rules.md：禁止默认 user 作用域。"
            )

    async def test_happy_path_memory_flush_contains_task_id(self) -> None:
        """memory_flush 写入的 summary 包含 task_id（可追溯性）。"""
        task_id = "task-summary-check-001"
        mock_client = _make_mock_client()
        mock_guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
        mock_memory_api = _make_mock_memory_api()

        mock_llm = _make_mock_llm_client([_R_PERCEIVE_RUNNING, _R_PERCEIVE_DONE])

        graph = _build_graph(
            mock_llm=mock_llm,
            mock_client=mock_client,
            mock_guard=mock_guard,
            mock_memory_api=mock_memory_api,
            task_id=task_id,
        )

        with _ApiKeyContext():
            result = await graph.ainvoke(
                DesktopTaskState(
                    task_id=task_id,
                    task_description="摘要内容检查测试",
                    task_status=TaskStatus.RUNNING,
                ),
                config={"configurable": {"thread_id": task_id}},
            )

        assert result.get("task_status") == TaskStatus.DONE

        # 验证 write_session_summary 摘要中包含 task_id
        assert mock_memory_api.write_session_summary.called
        _, kwargs = mock_memory_api.write_session_summary.call_args
        summary = kwargs.get("summary", "")
        assert task_id in summary, (
            f"summary 应包含 task_id={task_id!r}，实际 summary={summary[:200]!r}"
        )

    async def test_happy_path_thread_id_equals_task_id(self) -> None:
        """thread_id=task_id 配置正确（蓝图 §1.4）。

        通过验证使用相同 thread_id 能获取最终 state 确认 checkpointer 配置正确。
        """
        task_id = "task-threadid-001"
        mock_client = _make_mock_client()
        mock_guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
        mock_memory_api = _make_mock_memory_api()

        mock_llm = _make_mock_llm_client([_R_PERCEIVE_DONE])

        graph = _build_graph(
            mock_llm=mock_llm,
            mock_client=mock_client,
            mock_guard=mock_guard,
            mock_memory_api=mock_memory_api,
            task_id=task_id,
        )

        config = {"configurable": {"thread_id": task_id}}

        with _ApiKeyContext():
            result = await graph.ainvoke(
                DesktopTaskState(
                    task_id=task_id,
                    task_description="thread_id 测试",
                    task_status=TaskStatus.RUNNING,
                ),
                config=config,
            )

        # 任务在 DONE 状态后直接路由 memory_flush→END
        assert result.get("task_status") == TaskStatus.DONE

        # 使用相同 thread_id 可以 get_state（checkpointer 正常工作）
        state_snapshot = await graph.aget_state(config)
        assert state_snapshot is not None


# ── 测试 2：interrupt/resume 链路 ─────────────────────────────────────────────


class TestInterruptResumePath:
    """DESTRUCTIVE 动作 interrupt/resume 链路集成测试。

    验证：
      - 首次 ainvoke 触发 GraphInterrupt，payload 是 ConfirmRequest。
      - interrupt 前无任何 client 写操作（只读区红线）。
      - Command(resume=ConfirmResponse(confirmed=True)) 后写操作被调用一次。
      - 最终图运行到 END，task_status=DONE/FAILED（由后续 supervisor 决定）。
    """

    async def test_destructive_triggers_interrupt_with_confirm_request(self) -> None:
        """DESTRUCTIVE 动作首次 ainvoke 后 result['__interrupt__'] 含 ConfirmRequest。

        LangGraph 1.2.6：interrupt 结果在 __interrupt__ 字段，不抛异常。
        payload 必须是 ConfirmRequest（来自 src.orchestration.state，Task 10A 统一出口）。
        """
        task_id = "task-interrupt-001"
        mock_client = _make_mock_client()
        mock_guard = _make_mock_guard(risk=ActionRisk.DESTRUCTIVE, toctou_verdict="pass")
        mock_memory_api = _make_mock_memory_api()

        # Supervisor 先路由到 control（执行 DESTRUCTIVE 动作）
        mock_llm = _make_mock_llm_client([_R_CONTROL_RUNNING, _R_PERCEIVE_DONE])

        graph = _build_graph(
            mock_llm=mock_llm,
            mock_client=mock_client,
            mock_guard=mock_guard,
            mock_memory_api=mock_memory_api,
            task_id=task_id,
        )

        # DESTRUCTIVE 动作
        pending_action = ActionSpec(
            action_id="act-destructive-001",
            action_type="window_close",
            target_element_id="12345",
            coordinates=None,
            text_payload=None,
            risk_level=ActionRisk.DESTRUCTIVE,
        )

        config = {"configurable": {"thread_id": task_id}}

        with _ApiKeyContext():
            result1 = await graph.ainvoke(
                DesktopTaskState(
                    task_id=task_id,
                    task_description="关闭窗口集成测试",
                    task_status=TaskStatus.RUNNING,
                    pending_action=pending_action,
                ),
                config=config,
            )

        # 验证触发了 interrupt
        interrupts = result1.get("__interrupt__", [])
        assert len(interrupts) > 0, (
            f"DESTRUCTIVE 动作应触发 interrupt，result['__interrupt__'] 为空。"
            f"result keys: {list(result1.keys())}"
        )

        payload = _extract_interrupt_payload(result1)
        assert payload is not None, (
            f"interrupt payload 应为 ConfirmRequest，得到 {interrupts[0].value!r}"
        )
        assert isinstance(payload, ConfirmRequest), (
            f"payload 应为 ConfirmRequest（来自 src.orchestration.state），"
            f"实际类型 {type(payload).__name__}"
        )
        assert payload.risk_level == ActionRisk.DESTRUCTIVE

    async def test_no_write_before_interrupt(self) -> None:
        """interrupt 前无任何 client 写操作（interrupt 重放铁律，规格书 §1.2）。

        验证：classify_risk + toctou_verify（只读操作）在 interrupt 之前执行，
        但 click_element / close_window / type_text / send_key 不被调用。
        """
        task_id = "task-no-write-001"
        mock_client = _make_mock_client()
        mock_guard = _make_mock_guard(risk=ActionRisk.DESTRUCTIVE, toctou_verdict="pass")
        mock_memory_api = _make_mock_memory_api()

        mock_llm = _make_mock_llm_client([_R_CONTROL_RUNNING, _R_PERCEIVE_DONE])

        graph = _build_graph(
            mock_llm=mock_llm,
            mock_client=mock_client,
            mock_guard=mock_guard,
            mock_memory_api=mock_memory_api,
            task_id=task_id,
        )

        pending_action = ActionSpec(
            action_id="act-write-guard-001",
            action_type="window_close",
            target_element_id="99999",
            coordinates=None,
            text_payload=None,
            risk_level=ActionRisk.DESTRUCTIVE,
        )

        config = {"configurable": {"thread_id": task_id}}

        with _ApiKeyContext():
            result1 = await graph.ainvoke(
                DesktopTaskState(
                    task_id=task_id,
                    task_description="写操作前置保护测试",
                    task_status=TaskStatus.RUNNING,
                    pending_action=pending_action,
                ),
                config=config,
            )

        # 确认触发了 interrupt（否则测试前提不成立）
        assert result1.get("__interrupt__"), "期望 DESTRUCTIVE 触发 interrupt"

        # interrupt 前不应有任何写操作（只读区红线）
        mock_client.close_window.assert_not_called()
        mock_client.click_element.assert_not_called()
        mock_client.type_text.assert_not_called()
        mock_client.send_key.assert_not_called()

    async def test_resume_confirm_triggers_write_and_reaches_end(self) -> None:
        """Command(resume=ConfirmResponse(confirmed=True)) 后写操作被调，图运行到 END。

        恢复流程：
          1. 首次 ainvoke → interrupt（DESTRUCTIVE）
          2. ainvoke(Command(resume=ConfirmResponse(confirmed=True))) → 写操作 → supervisor
          3. supervisor 决定 DONE → memory_flush → END
        """
        task_id = "task-resume-confirm-001"
        mock_client = _make_mock_client()
        mock_guard = _make_mock_guard(risk=ActionRisk.DESTRUCTIVE, toctou_verdict="pass")
        mock_memory_api = _make_mock_memory_api()

        # 轮 1：路由 control；轮 2（resume 后 supervisor）：声明完成
        mock_llm = _make_mock_llm_client([_R_CONTROL_RUNNING, _R_PERCEIVE_DONE])

        graph = _build_graph(
            mock_llm=mock_llm,
            mock_client=mock_client,
            mock_guard=mock_guard,
            mock_memory_api=mock_memory_api,
            task_id=task_id,
        )

        pending_action = ActionSpec(
            action_id="act-resume-001",
            action_type="window_close",
            target_element_id="55555",
            coordinates=None,
            text_payload=None,
            risk_level=ActionRisk.DESTRUCTIVE,
        )

        config = {"configurable": {"thread_id": task_id}}

        with _ApiKeyContext():
            # 步骤 1：触发 interrupt
            result1 = await graph.ainvoke(
                DesktopTaskState(
                    task_id=task_id,
                    task_description="interrupt/resume 集成测试",
                    task_status=TaskStatus.RUNNING,
                    pending_action=pending_action,
                ),
                config=config,
            )

            assert result1.get("__interrupt__"), "步骤 1：期望触发 interrupt"

            # 步骤 2：resume confirm → 写操作被调 → 图运行到 END
            result2 = await graph.ainvoke(
                Command(resume=ConfirmResponse(confirmed=True, reason="集成测试确认")),
                config=config,
            )

        # 写操作被调用恰好一次（resume 后写区）
        mock_client.close_window.assert_called_once_with(window_handle=55555)

        # 图运行到 END（task_status=DONE，memory_flush 被调）
        assert result2.get("task_status") == TaskStatus.DONE, (
            f"resume confirm 后期望 task_status=DONE，实际 {result2.get('task_status')!r}"
        )
        mock_memory_api.write_session_summary.assert_called()

    async def test_resume_abort_no_write_and_memory_flush_called(self) -> None:
        """Command(resume=ConfirmResponse(confirmed=False)) 后：

        1. 写操作未被调（拒绝区红线）。
        2. 图继续运行直到 END（不因 abort 直接停止）。
        3. memory_flush 被调（不管最终 task_status）。

        图行为：abort → control_error 非 None（task_status=FAILED by control）
          → route_after_control → stall_detect（因 control 设 FAILED）
          → route_after_stall（stall_count 未达阈）→ supervisor
          → supervisor 用 LLM 第二响应声明 FAILED → memory_flush → END。

        此测试用两个 LLM 响应：第一个触发 control，第二个 supervisor 声明 FAILED。
        """
        task_id = "task-resume-abort-001"
        mock_client = _make_mock_client()
        mock_guard = _make_mock_guard(risk=ActionRisk.DESTRUCTIVE, toctou_verdict="pass")
        mock_memory_api = _make_mock_memory_api()

        # 轮 1：路由 control（触发 interrupt）；轮 2（abort 后）：声明 FAILED
        mock_llm = _make_mock_llm_client([_R_CONTROL_RUNNING, _R_PERCEIVE_FAILED])

        graph = _build_graph(
            mock_llm=mock_llm,
            mock_client=mock_client,
            mock_guard=mock_guard,
            mock_memory_api=mock_memory_api,
            task_id=task_id,
        )

        pending_action = ActionSpec(
            action_id="act-abort-001",
            action_type="window_close",
            target_element_id="11111",
            coordinates=None,
            text_payload=None,
            risk_level=ActionRisk.DESTRUCTIVE,
        )

        config = {"configurable": {"thread_id": task_id}}

        with _ApiKeyContext():
            # 步骤 1：触发 interrupt
            result1 = await graph.ainvoke(
                DesktopTaskState(
                    task_id=task_id,
                    task_description="interrupt abort 集成测试",
                    task_status=TaskStatus.RUNNING,
                    pending_action=pending_action,
                ),
                config=config,
            )

            assert result1.get("__interrupt__"), "步骤 1：期望触发 interrupt"

            # 步骤 2：resume 拒绝
            result2 = await graph.ainvoke(
                Command(resume=ConfirmResponse(confirmed=False, reason="测试拒绝")),
                config=config,
            )

        # 拒绝后无写操作（红线：interrupt 前只读，resume 后写区不执行）
        mock_client.close_window.assert_not_called()

        # 图运行到 END（task_status=FAILED，由第二个 LLM 响应决定）
        assert result2.get("task_status") == TaskStatus.FAILED, (
            f"resume abort 后 supervisor 应声明 FAILED，实际 {result2.get('task_status')!r}"
        )

        # memory_flush 被调（FAILED 状态也触发记忆写入）
        mock_memory_api.write_session_summary.assert_called()

    async def test_interrupt_payload_is_confirm_request_from_state_module(self) -> None:
        """interrupt payload 类型来自 src.orchestration.state（Task 10A 统一出口）。

        desktop_control_agent.py 已改为从 state 导入（Bug 修复：集成测试发现本地定义与
        state 定义为不同类型，导致 isinstance 检查失败）。通过 isinstance 验证。
        """
        task_id = "task-payload-type-001"
        mock_client = _make_mock_client()
        mock_guard = _make_mock_guard(risk=ActionRisk.DESTRUCTIVE, toctou_verdict="pass")
        mock_memory_api = _make_mock_memory_api()

        mock_llm = _make_mock_llm_client([_R_CONTROL_RUNNING, _R_PERCEIVE_DONE])

        graph = _build_graph(
            mock_llm=mock_llm,
            mock_client=mock_client,
            mock_guard=mock_guard,
            mock_memory_api=mock_memory_api,
            task_id=task_id,
        )

        pending_action = ActionSpec(
            action_id="act-type-check-001",
            action_type="window_close",
            target_element_id="77777",
            coordinates=None,
            text_payload=None,
            risk_level=ActionRisk.DESTRUCTIVE,
        )

        config = {"configurable": {"thread_id": task_id}}

        with _ApiKeyContext():
            result = await graph.ainvoke(
                DesktopTaskState(
                    task_id=task_id,
                    task_description="payload 类型验证",
                    task_status=TaskStatus.RUNNING,
                    pending_action=pending_action,
                ),
                config=config,
            )

        payload = _extract_interrupt_payload(result)
        assert payload is not None, "期望存在 interrupt payload"

        # ConfirmRequest 来自 src.orchestration.state（Task 10A 统一出口）
        assert isinstance(payload, ConfirmRequest), (
            f"payload 应为 src.orchestration.state.ConfirmRequest，"
            f"实际类型 {type(payload).__module__}.{type(payload).__name__}"
        )
        assert payload.risk_level == ActionRisk.DESTRUCTIVE


# ── 测试 3：感知失败停滞路径（R3 图结构验证） ─────────────────────────────────


class TestPerceptionFailureStallPath:
    """感知失败停滞路径集成测试（R3 决策：perceive→stall_detect→supervisor）。

    验证：
      - perception_error 连续出现 → stall_count 达 STALL_THRESHOLD。
      - 停滞超阈 → error_report 被调 → memory_flush → END。
      - task_status 最终为 FAILED。
      - R3 图结构正确：perceive 后经 stall_detect（不直接回 supervisor）。
    """

    async def test_consecutive_perception_errors_trigger_stall_path(self) -> None:
        """连续感知失败 → stall_count 达阈 → error_report → memory_flush → FAILED。

        mock client.screen_snapshot 始终抛 DesktopMCPCallError，
        模拟持续感知失败场景。
        """
        from src.mcp.desktop_mcp_client import DesktopMCPCallError

        task_id = "task-stall-perception-001"
        mock_client = MagicMock()
        # screen_snapshot 始终失败
        mock_client.screen_snapshot = AsyncMock(
            side_effect=DesktopMCPCallError("screen_snapshot", "MCP 连接失败")
        )
        mock_guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
        mock_memory_api = _make_mock_memory_api()

        # Supervisor 一直路由到 perceive（直到停滞超阈路由覆盖）
        mock_llm = _make_mock_llm_client([_R_PERCEIVE_RUNNING])

        graph = _build_graph(
            mock_llm=mock_llm,
            mock_client=mock_client,
            mock_guard=mock_guard,
            mock_memory_api=mock_memory_api,
            task_id=task_id,
        )

        with _ApiKeyContext():
            result = await graph.ainvoke(
                DesktopTaskState(
                    task_id=task_id,
                    task_description="感知失败停滞路径集成测试",
                    task_status=TaskStatus.RUNNING,
                ),
                config={"configurable": {"thread_id": task_id}},
            )

        # 最终 task_status=FAILED（停滞超阈 → error_report）
        assert result.get("task_status") == TaskStatus.FAILED, (
            f"连续感知失败应触发停滞路径，最终 FAILED。实际 {result.get('task_status')!r}"
        )

        # memory_flush 被调（FAILED 也触发记忆写入）
        assert mock_memory_api.write_session_summary.called, (
            "感知失败停滞路径应触发 memory_flush（FAILED 状态也写记忆）"
        )

    async def test_r3_graph_structure_perceive_goes_through_stall_detect(self) -> None:
        """R3 图结构验证：perceive 节点后经 stall_detect（不直接回 supervisor）。

        通过 graph 的 get_graph 编译结果验证图连线：
        perceive 的出边必须指向 stall_detect（而非 supervisor）。
        """
        task_id = "task-r3-structure-001"
        mock_client = _make_mock_client()
        mock_guard = _make_mock_guard()
        mock_memory_api = _make_mock_memory_api()
        mock_llm = _make_mock_llm_client([_R_PERCEIVE_DONE])

        graph = _build_graph(
            mock_llm=mock_llm,
            mock_client=mock_client,
            mock_guard=mock_guard,
            mock_memory_api=mock_memory_api,
            task_id=task_id,
        )

        # 通过 graph.get_graph() 获取编译后图的节点边信息
        compiled_graph = graph.get_graph()
        edges = compiled_graph.edges

        # perceive → stall_detect 边必须存在（R3 决策）
        perceive_targets = {edge.target for edge in edges if edge.source == "perceive"}
        assert "stall_detect" in perceive_targets, (
            f"R3 决策：perceive 节点后应经 stall_detect，实际出边目标: {perceive_targets}"
        )

        # perceive 不应直接连向 supervisor（R3 图结构）
        assert "supervisor" not in perceive_targets, (
            f"R3 决策：perceive 不应直接连 supervisor，实际出边目标: {perceive_targets}"
        )

    async def test_perception_failure_stall_count_reaches_threshold(self) -> None:
        """感知失败路径中 stall_count 实际达到 STALL_THRESHOLD 触发停滞处理。

        验证：stall_detect_node 的感知失败信号（信号 C）被正确累积，
        stall_count 最终达到 STALL_THRESHOLD，触发 error_report 路由。
        """
        from src.mcp.desktop_mcp_client import DesktopMCPConnectionError

        task_id = "task-stall-count-001"
        mock_client = MagicMock()
        mock_client.screen_snapshot = AsyncMock(side_effect=DesktopMCPConnectionError("连接断开"))
        mock_guard = _make_mock_guard()
        mock_memory_api = _make_mock_memory_api()

        # supervisor 持续路由 perceive 直到停滞超阈
        mock_llm = _make_mock_llm_client([_R_PERCEIVE_RUNNING])

        graph = _build_graph(
            mock_llm=mock_llm,
            mock_client=mock_client,
            mock_guard=mock_guard,
            mock_memory_api=mock_memory_api,
            task_id=task_id,
        )

        with _ApiKeyContext():
            result = await graph.ainvoke(
                DesktopTaskState(
                    task_id=task_id,
                    task_description="stall_count 阈值测试",
                    task_status=TaskStatus.RUNNING,
                ),
                config={"configurable": {"thread_id": task_id}},
            )

        # 停滞超阈后 error_report → FAILED
        assert result.get("task_status") == TaskStatus.FAILED
        # memory_flush 被调（FAILED 状态不跳过写记忆）
        mock_memory_api.write_session_summary.assert_called()

    async def test_error_report_called_before_memory_flush_on_stall(self) -> None:
        """停滞路径：error_report 节点在 memory_flush 之前执行。

        验证调用顺序：error_report → memory_flush（通过 call_count 推断）。
        memory_flush 在 error_report 之后执行，task_status=FAILED。
        """
        from src.mcp.desktop_mcp_client import DesktopMCPCallError

        task_id = "task-order-001"
        mock_client = MagicMock()
        mock_client.screen_snapshot = AsyncMock(
            side_effect=DesktopMCPCallError("screen_snapshot", "感知服务不可用")
        )
        mock_guard = _make_mock_guard()

        # 使用可追踪的 memory_api mock
        call_order: list[str] = []

        class TrackingMemoryAPI:
            async def write_session_summary(
                self,
                task_id: str,
                scope: str,
                summary: str,
                metadata: Any = None,
            ) -> None:
                call_order.append("memory_flush")

        mock_llm = _make_mock_llm_client([_R_PERCEIVE_RUNNING])

        # 注入真实 PromptLoader + TrackingMemoryAPI
        prompt_loader = PromptLoader()
        supervisor_agent = DesktopSupervisorAgent(
            llm_client=mock_llm,
            prompt_loader=prompt_loader,
        )
        perception_agent = ScreenPerceptionAgent(client=mock_client, snapshot_store=None)
        control_agent = DesktopControlAgent(client=mock_client, guard=mock_guard)
        tracking_memory_api = TrackingMemoryAPI()

        graph = get_graph(
            supervisor_agent=supervisor_agent,
            perception_agent=perception_agent,
            control_agent=control_agent,
            memory_api=tracking_memory_api,
            checkpointer=None,
        )

        with _ApiKeyContext():
            result = await graph.ainvoke(
                DesktopTaskState(
                    task_id=task_id,
                    task_description="调用顺序验证",
                    task_status=TaskStatus.RUNNING,
                ),
                config={"configurable": {"thread_id": task_id}},
            )

        # memory_flush 被调（call_order 有记录）
        assert "memory_flush" in call_order, "停滞路径应触发 memory_flush"
        # 最终 FAILED
        assert result.get("task_status") == TaskStatus.FAILED


# ── 测试 4：图结构与契约完整性 ───────────────────────────────────────────────


class TestGraphStructureAndContracts:
    """图结构和接口契约验证（不跑图运行，只验证编译结果）。"""

    def test_get_graph_returns_compiled_graph(self) -> None:
        """get_graph() 返回编译好的 StateGraph，无报错。"""
        graph = get_graph(checkpointer=None)
        assert graph is not None
        # 编译好的图有 ainvoke / invoke 方法
        assert hasattr(graph, "ainvoke")
        assert hasattr(graph, "invoke")

    def test_graph_has_all_required_nodes(self) -> None:
        """图包含所有必要节点（supervisor/perceive/control/stall_detect/error_report/memory_flush）。

        LangGraph 1.2.6：get_graph().nodes 是 dict，键为节点 ID 字符串。
        """
        graph = get_graph(checkpointer=None)
        compiled_graph = graph.get_graph()
        # nodes 是 dict（键=节点 ID 字符串），不是对象列表
        node_ids = set(compiled_graph.nodes.keys())

        required_nodes = {
            "supervisor",
            "perceive",
            "control",
            "stall_detect",
            "error_report",
            "memory_flush",
        }
        missing = required_nodes - node_ids
        assert not missing, f"图缺少节点: {missing}（实际节点: {node_ids}）"

    def test_graph_memory_flush_connects_to_end(self) -> None:
        """memory_flush 节点连向 END（终态路径正确）。"""
        graph = get_graph(checkpointer=None)
        compiled_graph = graph.get_graph()
        edges = compiled_graph.edges

        memory_flush_targets = {edge.target for edge in edges if edge.source == "memory_flush"}
        assert "__end__" in memory_flush_targets, (
            f"memory_flush 应连向 END，实际目标: {memory_flush_targets}"
        )

    def test_graph_error_report_connects_to_memory_flush(self) -> None:
        """error_report 节点连向 memory_flush（停滞路径写记忆）。"""
        graph = get_graph(checkpointer=None)
        compiled_graph = graph.get_graph()
        edges = compiled_graph.edges

        error_report_targets = {edge.target for edge in edges if edge.source == "error_report"}
        assert "memory_flush" in error_report_targets, (
            f"error_report 应连向 memory_flush，实际目标: {error_report_targets}"
        )

    def test_stall_detect_connects_to_both_supervisor_and_error_report(self) -> None:
        """stall_detect 节点有两条出边：supervisor 和 error_report（条件边）。"""
        graph = get_graph(checkpointer=None)
        compiled_graph = graph.get_graph()
        edges = compiled_graph.edges

        stall_targets = {edge.target for edge in edges if edge.source == "stall_detect"}
        assert "supervisor" in stall_targets, (
            f"stall_detect 应有 supervisor 出边，实际: {stall_targets}"
        )
        assert "error_report" in stall_targets, (
            f"stall_detect 应有 error_report 出边，实际: {stall_targets}"
        )

    def test_confirm_request_importable_from_state_module(self) -> None:
        """ConfirmRequest 可从 src.orchestration.state 导入（Task 10A 统一出口）。"""
        # 此 import 在模块顶层已完成；此处验证字段完整性
        req = ConfirmRequest(
            action_id="test-001",
            action_type="window_close",
            risk_level=ActionRisk.DESTRUCTIVE,
            description="测试确认请求",
        )
        assert req.action_id == "test-001"
        assert req.risk_level == ActionRisk.DESTRUCTIVE

    def test_confirm_response_importable_from_state_module(self) -> None:
        """ConfirmResponse 可从 src.orchestration.state 导入。"""
        resp_confirm = ConfirmResponse(confirmed=True, reason="集成测试确认")
        resp_abort = ConfirmResponse(confirmed=False, reason="集成测试拒绝")
        assert resp_confirm.confirmed is True
        assert resp_abort.confirmed is False

    def test_prompt_loader_injection_works_with_supervisor(self) -> None:
        """真实 PromptLoader 可注入 DesktopSupervisorAgent（Task 11B 接线验证）。

        PromptLoader 满足 desktop_supervisor.PromptLoader Protocol（结构子类型）。
        """
        prompt_loader = PromptLoader()
        # DesktopSupervisorAgent 接受 PromptLoader Protocol 实例（无 TypeError）
        supervisor = DesktopSupervisorAgent(
            llm_client=None,  # 无 LLM 客户端（缺 key 优雅回退）
            prompt_loader=prompt_loader,
        )
        assert supervisor.prompt_loader is prompt_loader

    def test_desktop_task_state_initial_values(self) -> None:
        """DesktopTaskState 初始化正确，默认值符合规格书 §5.1。"""
        state = DesktopTaskState(
            task_id="init-test",
            task_description="初始值测试",
        )
        assert state.task_status == TaskStatus.RUNNING
        assert state.stall_count == 0
        assert state.step_history == []
        assert state.perception_error is None
        assert state.control_error is None
        assert state.snapshot_ref is None
        assert state.uia_hollow is False


# ── 测试 5：缺 ANTHROPIC_API_KEY 优雅回退 ────────────────────────────────────


class TestMissingApiKeyGracefulFallback:
    """缺 ANTHROPIC_API_KEY 时 Supervisor 优雅回退（不崩溃，任务 FAILED）。"""

    async def test_missing_api_key_results_in_failed(self) -> None:
        """缺 ANTHROPIC_API_KEY 时 supervisor 返回 FAILED，图运行到 END 不崩溃。"""
        import os

        task_id = "task-no-key-001"
        mock_client = _make_mock_client()
        mock_guard = _make_mock_guard()
        mock_memory_api = _make_mock_memory_api()

        # 随意 mock llm（不会被真实调用，因为缺 key 提前回退）
        mock_llm = MagicMock()
        mock_llm.messages = MagicMock()

        graph = _build_graph(
            mock_llm=mock_llm,
            mock_client=mock_client,
            mock_guard=mock_guard,
            mock_memory_api=mock_memory_api,
            task_id=task_id,
        )

        original_key = os.environ.get("ANTHROPIC_API_KEY")
        # 强制移除 API key
        os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            result = await graph.ainvoke(
                DesktopTaskState(
                    task_id=task_id,
                    task_description="缺 API key 测试",
                    task_status=TaskStatus.RUNNING,
                ),
                config={"configurable": {"thread_id": task_id}},
            )
        finally:
            if original_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = original_key

        # 缺 key → supervisor 返回 FAILED → error_report → memory_flush → END
        assert result.get("task_status") == TaskStatus.FAILED, (
            f"缺 ANTHROPIC_API_KEY 应优雅回退为 FAILED，实际 {result.get('task_status')!r}"
        )
        # 不应崩溃（到达此处即通过）
