"""DesktopSupervisorAgent + supervisor_node 单元测试（Task 10A）。

覆盖：
  1. mock LLM 只返 3 增量字段（next_agent/current_instruction/task_status），
     不含业务路由判断（无 is_browser_task 等）。
  2. 缺 ANTHROPIC_API_KEY 时 plan() 返回 task_status=FAILED，不崩溃。
  3. step_history 超 STATE_STEP_KEEP 时 StepArchive.archive 被调，
     返回最近 K 步完整 list（R2：LastValue 截断）。
  4. supervisor_node 只返 4 个字段增量（3 plan 字段 + step_history）。
  5. _parse_plan_response：正常 JSON / markdown fence / 非法字段 / 非 dict。
  6. _FallbackPromptLoader.render_supervisor 返回两非空字符串。
  7. 顶层 supervisor_node 占位函数始终 raise RuntimeError。
  8. TaskStatus StrEnum 值正确。
  9. DesktopTaskState 构造、字段默认值、step_history LastValue 语义。
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.models.screen_snapshot import ActionRisk
from src.orchestration.desktop_supervisor import (
    MAX_ITERATIONS_EXCEEDED,
    DesktopSupervisorAgent,
    _FallbackPromptLoader,
    _parse_plan_response,
    _truncate_step_history,
    make_supervisor_node,
    supervisor_node,
)
from src.orchestration.state import (
    STATE_STEP_KEEP,
    ConfirmRequest,
    ConfirmResponse,
    DesktopTaskState,
    StepArchive,
    StepRecord,
    TaskStatus,
)

# ── 测试辅助 ──────────────────────────────────────────────────────────────────


def _make_step(index: int, agent: str = "perceive") -> StepRecord:
    """构造测试用 StepRecord。"""
    return StepRecord(
        step_index=index,
        agent=agent,
        instruction=f"step-{index}",
        snapshot_ref=None,
        perception_summary=None,
        control_error=None,
        perception_error=None,
        task_status=TaskStatus.RUNNING,
    )


def _make_state(
    task_id: str = "task-001",
    task_description: str = "打开计算器",
    steps: int = 0,
) -> DesktopTaskState:
    """构造测试用 DesktopTaskState。"""
    return DesktopTaskState(
        task_id=task_id,
        task_description=task_description,
        step_history=[_make_step(i) for i in range(steps)],
    )


def _make_llm_response(
    next_agent: str = "perceive",
    current_instruction: str = "执行感知",
    task_status: str = "RUNNING",
) -> MagicMock:
    """构造 mock LLM response 对象（mimics anthropic SDK 返回值结构）。"""
    content_block = MagicMock()
    content_block.text = json.dumps(
        {
            "next_agent": next_agent,
            "current_instruction": current_instruction,
            "task_status": task_status,
        }
    )
    response = MagicMock()
    response.content = [content_block]
    return response


def _make_mock_llm(
    next_agent: str = "perceive",
    current_instruction: str = "执行感知",
    task_status: str = "RUNNING",
) -> MagicMock:
    """构造 mock Anthropic AsyncAnthropic client。"""
    llm = MagicMock()
    llm.messages = MagicMock()
    llm.messages.create = AsyncMock(
        return_value=_make_llm_response(next_agent, current_instruction, task_status)
    )
    return llm


# ── 1. TaskStatus StrEnum 值正确 ──────────────────────────────────────────────


def test_task_status_values() -> None:
    """TaskStatus StrEnum 枚举值符合规格书 §5.1。"""
    assert TaskStatus.RUNNING == "RUNNING"
    assert TaskStatus.WAITING_CONFIRM == "WAITING_CONFIRM"
    assert TaskStatus.STALLED == "STALLED"
    assert TaskStatus.DONE == "DONE"
    assert TaskStatus.FAILED == "FAILED"


# ── 2. DesktopTaskState 构造与字段默认值 ─────────────────────────────────────


def test_desktop_task_state_defaults() -> None:
    """DesktopTaskState 字段默认值符合规格书 §5.1。"""
    state = DesktopTaskState()
    assert state.task_id == ""
    assert state.task_description == ""
    assert state.task_status == TaskStatus.RUNNING
    assert state.step_history == []
    assert state.snapshot_ref is None
    assert state.perception_error is None
    assert state.control_error is None
    assert state.stall_count == 0
    assert state.uia_hollow is False
    assert state.capability_flags == {}


def test_desktop_task_state_step_history_is_list() -> None:
    """step_history 默认为空 list，不共享实例（pydantic field_factory 隔离）。"""
    s1 = DesktopTaskState()
    s2 = DesktopTaskState()
    s1.step_history.append(_make_step(0))
    assert s2.step_history == [], "step_history 实例不应共享"


# ── 3. ConfirmRequest / ConfirmResponse 定义在 state.py ──────────────────────


def test_confirm_request_fields() -> None:
    """ConfirmRequest 可正常构造，字段符合规格书 §7.6。"""
    req = ConfirmRequest(
        action_id="act-001",
        action_type="window_close",
        risk_level=ActionRisk.DESTRUCTIVE,
        description="关闭目标窗口",
        coordinates=(100, 200),
        target_element_id="hwnd-42",
    )
    assert req.action_id == "act-001"
    assert req.risk_level == ActionRisk.DESTRUCTIVE
    assert req.coordinates == (100, 200)


def test_confirm_response_fields() -> None:
    """ConfirmResponse 可正常构造（confirmed + reason）。"""
    resp = ConfirmResponse(confirmed=True, reason="测试确认")
    assert resp.confirmed is True
    assert resp.reason == "测试确认"

    resp_false = ConfirmResponse(confirmed=False)
    assert resp_false.confirmed is False
    assert resp_false.reason == ""


# ── 4. _parse_plan_response 单元测试 ─────────────────────────────────────────


def test_parse_plan_response_normal() -> None:
    """正常 JSON 字符串 → 提取 3 个字段。"""
    raw = json.dumps(
        {
            "next_agent": "control",
            "current_instruction": "点击确认按钮",
            "task_status": "RUNNING",
        }
    )
    result = _parse_plan_response(raw)
    assert result["next_agent"] == "control"
    assert result["current_instruction"] == "点击确认按钮"
    assert result["task_status"] == "RUNNING"


def test_parse_plan_response_markdown_fence() -> None:
    """LLM 返回 markdown code fence 包裹的 JSON → 自动剥除。"""
    inner = '{"next_agent":"perceive","current_instruction":"感知","task_status":"RUNNING"}'
    raw = f"```json\n{inner}\n```"
    result = _parse_plan_response(raw)
    assert result["next_agent"] == "perceive"
    assert result["task_status"] == "RUNNING"


def test_parse_plan_response_invalid_json() -> None:
    """非法 JSON → next_agent=error_report, task_status=FAILED。"""
    result = _parse_plan_response("not json at all")
    assert result["next_agent"] == "error_report"
    assert result["task_status"] == TaskStatus.FAILED


def test_parse_plan_response_non_dict() -> None:
    """JSON 但非 dict（如 list）→ task_status=FAILED。"""
    result = _parse_plan_response(json.dumps([1, 2, 3]))
    assert result["task_status"] == TaskStatus.FAILED


def test_parse_plan_response_invalid_next_agent() -> None:
    """next_agent 值不在有效集合 → 降级为 error_report。"""
    raw = json.dumps(
        {
            "next_agent": "unknown_agent",
            "current_instruction": "test",
            "task_status": "RUNNING",
        }
    )
    result = _parse_plan_response(raw)
    assert result["next_agent"] == "error_report"


def test_parse_plan_response_invalid_task_status() -> None:
    """task_status 值非法 → 降级为 RUNNING。"""
    raw = json.dumps(
        {
            "next_agent": "perceive",
            "current_instruction": "test",
            "task_status": "INVALID_STATUS",
        }
    )
    result = _parse_plan_response(raw)
    assert result["task_status"] == TaskStatus.RUNNING


def test_parse_plan_response_missing_fields() -> None:
    """部分字段缺失 → 使用安全默认值，不崩溃。"""
    raw = json.dumps({"next_agent": "perceive"})
    result = _parse_plan_response(raw)
    assert result["next_agent"] == "perceive"
    assert result["current_instruction"] == ""
    assert result["task_status"] == TaskStatus.RUNNING


# ── 5. _FallbackPromptLoader 单元测试 ────────────────────────────────────────


def test_fallback_prompt_loader_returns_nonempty() -> None:
    """_FallbackPromptLoader.render_supervisor 返回两非空字符串。"""
    loader = _FallbackPromptLoader()
    state = _make_state(task_description="测试任务", steps=2)
    system, user = loader.render_supervisor(state)
    assert isinstance(system, str) and len(system) > 0
    assert isinstance(user, str) and len(user) > 0


def test_fallback_prompt_loader_contains_task_description() -> None:
    """user prompt 包含任务描述。"""
    loader = _FallbackPromptLoader()
    state = _make_state(task_description="打开 Chrome 浏览器")
    _, user = loader.render_supervisor(state)
    assert "打开 Chrome 浏览器" in user


def test_fallback_prompt_loader_no_steps() -> None:
    """无历史步骤时 user prompt 含无步骤提示，不崩溃。"""
    loader = _FallbackPromptLoader()
    state = _make_state(steps=0)
    _, user = loader.render_supervisor(state)
    assert "无历史步骤" in user


# ── 6. _truncate_step_history 单元测试 ───────────────────────────────────────


async def test_truncate_no_overflow() -> None:
    """步数未超限时不调 archive，返回原 list。"""
    archive = MagicMock(spec=StepArchive)
    archive.archive = AsyncMock()
    steps = [_make_step(i) for i in range(5)]

    result = await _truncate_step_history("t1", steps, keep=10, archive=archive)

    assert result == steps
    archive.archive.assert_not_called()


async def test_truncate_overflow_calls_archive() -> None:
    """步数超限时调 archive，返回最近 K 步。"""
    archive = MagicMock(spec=StepArchive)
    archive.archive = AsyncMock()

    keep = 3
    steps = [_make_step(i) for i in range(7)]  # 7 步，超出 keep=3

    result = await _truncate_step_history("t2", steps, keep=keep, archive=archive)

    # 应返回最近 3 步
    assert len(result) == keep
    assert result == steps[-keep:]

    # archive 应被调一次，传入超出的 4 步（steps[0:4]）
    archive.archive.assert_called_once()
    call_kwargs = archive.archive.call_args.kwargs
    assert call_kwargs["task_id"] == "t2"
    assert len(call_kwargs["steps"]) == 4
    assert call_kwargs["steps"] == steps[:4]


async def test_truncate_archive_failure_does_not_raise() -> None:
    """archive 失败时不抛异常，仍返回截断后 list。"""
    archive = MagicMock(spec=StepArchive)
    archive.archive = AsyncMock(side_effect=RuntimeError("archive 失败"))

    steps = [_make_step(i) for i in range(5)]
    result = await _truncate_step_history("t3", steps, keep=2, archive=archive)

    # 虽然 archive 失败，仍返回截断结果
    assert len(result) == 2
    assert result == steps[-2:]


# ── 7. DesktopSupervisorAgent.plan 单元测试 ──────────────────────────────────


async def test_plan_missing_api_key_returns_failed() -> None:
    """缺 ANTHROPIC_API_KEY 时 plan() 返回 task_status=FAILED，不崩溃。"""
    llm = _make_mock_llm()
    agent = DesktopSupervisorAgent(llm_client=llm)
    state = _make_state()

    with patch.dict("os.environ", {}, clear=False):
        # 确保 key 不存在
        import os

        os.environ.pop("ANTHROPIC_API_KEY", None)
        result = await agent.plan(state)

    assert result["task_status"] == TaskStatus.FAILED
    assert result["next_agent"] == "error_report"
    # LLM 不应被调用
    llm.messages.create.assert_not_called()


async def test_plan_none_llm_client_returns_failed() -> None:
    """llm_client=None 且有 API key 时 plan() 返回 FAILED，不崩溃。"""
    agent = DesktopSupervisorAgent(llm_client=None)
    state = _make_state()

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key"}):
        result = await agent.plan(state)

    assert result["task_status"] == TaskStatus.FAILED
    assert result["next_agent"] == "error_report"


async def test_plan_llm_call_failure_returns_failed() -> None:
    """LLM 调用抛异常时 plan() 返回 FAILED，不崩溃。"""
    llm = MagicMock()
    llm.messages = MagicMock()
    llm.messages.create = AsyncMock(side_effect=RuntimeError("network error"))
    agent = DesktopSupervisorAgent(llm_client=llm)
    state = _make_state()

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key"}):
        result = await agent.plan(state)

    assert result["task_status"] == TaskStatus.FAILED
    assert "LLM 调用失败" in result["current_instruction"]


async def test_plan_returns_only_3_fields() -> None:
    """plan() 返回仅含 3 个字段的增量 dict（无业务路由判断字段）。"""
    llm = _make_mock_llm(
        next_agent="control",
        current_instruction="点击保存按钮",
        task_status="RUNNING",
    )
    agent = DesktopSupervisorAgent(llm_client=llm)
    state = _make_state()

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key"}):
        result = await agent.plan(state)

    # 只有 3 个增量字段
    assert set(result.keys()) == {"next_agent", "current_instruction", "task_status"}
    assert result["next_agent"] == "control"
    assert result["current_instruction"] == "点击保存按钮"
    assert result["task_status"] == "RUNNING"


async def test_plan_no_business_routing_logic() -> None:
    """supervisor plan 不做业务路由判断——LLM 说什么 next_agent 就是什么。

    验证：supervisor 直接透传 LLM 给出的 next_agent，不覆盖/修改。
    """
    # LLM 返回 "playwright"（业务路由标签）
    llm = _make_mock_llm(next_agent="playwright", task_status="RUNNING")
    agent = DesktopSupervisorAgent(llm_client=llm)
    state = _make_state()

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key"}):
        result = await agent.plan(state)

    # supervisor 直接透传，不修改（业务判断在条件边函数）
    assert result["next_agent"] == "playwright"


async def test_plan_with_prerendered_prompt() -> None:
    """plan(rendered_prompt=...) 时使用预渲染提示词，不调 prompt_loader。"""
    llm = _make_mock_llm(next_agent="perceive", task_status="RUNNING")
    mock_loader = MagicMock()
    mock_loader.render_supervisor = MagicMock()

    agent = DesktopSupervisorAgent(llm_client=llm, prompt_loader=mock_loader)
    state = _make_state()

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key"}):
        result = await agent.plan(
            state,
            rendered_prompt=("system text", "user text"),
        )

    # prompt_loader 不应被调用
    mock_loader.render_supervisor.assert_not_called()
    assert result["next_agent"] == "perceive"

    # 验证 LLM 接收了预渲染 prompt
    call_kwargs = llm.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "system text"
    assert call_kwargs["messages"][0]["content"] == "user text"


# ── 8. supervisor_node（通过 make_supervisor_node）单元测试 ───────────────────


async def test_supervisor_node_returns_5_fields() -> None:
    """supervisor_node 返回恰好 5 个字段：3 plan 字段 + step_history + iteration_count。"""
    llm = _make_mock_llm(
        next_agent="perceive",
        current_instruction="感知屏幕",
        task_status="RUNNING",
    )
    archive = MagicMock(spec=StepArchive)
    archive.archive = AsyncMock()
    agent = DesktopSupervisorAgent(llm_client=llm, step_archive=archive)
    node_fn = make_supervisor_node(agent)
    state = _make_state(steps=2)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key"}):
        result = await node_fn(state)

    assert set(result.keys()) == {
        "next_agent",
        "current_instruction",
        "task_status",
        "step_history",
        "iteration_count",
    }


async def test_supervisor_node_step_history_truncated() -> None:
    """supervisor_node：step_history 超 STATE_STEP_KEEP 时截断并调 archive。"""
    llm = _make_mock_llm()
    archive = MagicMock(spec=StepArchive)
    archive.archive = AsyncMock()
    agent = DesktopSupervisorAgent(llm_client=llm, step_archive=archive)
    node_fn = make_supervisor_node(agent)

    # 构造超出 STATE_STEP_KEEP 的 step_history
    overflow_count = 3
    steps = [_make_step(i) for i in range(STATE_STEP_KEEP + overflow_count)]
    state = DesktopTaskState(
        task_id="task-trunc",
        task_description="测试截断",
        step_history=steps,
    )

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key"}):
        result = await node_fn(state)

    # 返回的 step_history 应为最近 STATE_STEP_KEEP 步
    assert len(result["step_history"]) == STATE_STEP_KEEP
    assert result["step_history"] == steps[-STATE_STEP_KEEP:]

    # archive 应被调一次（归档 overflow 部分）
    archive.archive.assert_called_once()
    call_kwargs = archive.archive.call_args.kwargs
    assert len(call_kwargs["steps"]) == overflow_count


async def test_supervisor_node_no_overflow_archive_not_called() -> None:
    """step_history 未超限时 archive 不被调。"""
    llm = _make_mock_llm()
    archive = MagicMock(spec=StepArchive)
    archive.archive = AsyncMock()
    agent = DesktopSupervisorAgent(llm_client=llm, step_archive=archive)
    node_fn = make_supervisor_node(agent)

    state = _make_state(steps=3)  # 3 < STATE_STEP_KEEP（默认 20）

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key"}):
        await node_fn(state)

    archive.archive.assert_not_called()


async def test_supervisor_node_missing_api_key_returns_failed() -> None:
    """缺 ANTHROPIC_API_KEY 时 supervisor_node 返回 task_status=FAILED。"""
    llm = _make_mock_llm()
    agent = DesktopSupervisorAgent(llm_client=llm)
    node_fn = make_supervisor_node(agent)
    state = _make_state()

    import os

    os.environ.pop("ANTHROPIC_API_KEY", None)
    result = await node_fn(state)

    assert result["task_status"] == TaskStatus.FAILED
    assert result["next_agent"] == "error_report"
    # step_history 仍有效（即使 plan 失败，截断仍执行）
    assert "step_history" in result


# ── 8b. 回路硬上限 DESKTOP_MAX_ITERATIONS（K4 紧后 §3.3）─────────────────────


async def test_supervisor_node_increments_iteration_count() -> None:
    """每次非终态调用递增 iteration_count，未达上限不产生 failure_reason。"""
    llm = _make_mock_llm()
    agent = DesktopSupervisorAgent(llm_client=llm)
    node_fn = make_supervisor_node(agent)
    state = _make_state()
    state = state.model_copy(update={"iteration_count": 4})

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key"}):
        result = await node_fn(state)

    assert result["iteration_count"] == 5
    assert "failure_reason" not in result
    llm.messages.create.assert_called_once()


async def test_supervisor_node_max_iterations_cap_hits() -> None:
    """命中硬上限：不调 LLM，返回 failure_reason=max_iterations_exceeded +
    next_agent=error_report，且不设 task_status（终态由 error_report_node 落，
    与 LLM 判定失败可区分——设计输入 §3.3）。"""
    llm = _make_mock_llm()
    agent = DesktopSupervisorAgent(llm_client=llm)
    node_fn = make_supervisor_node(agent)
    state = _make_state().model_copy(update={"iteration_count": 3})

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key"}):
        with patch("src.orchestration.desktop_supervisor.DESKTOP_MAX_ITERATIONS", 3):
            result = await node_fn(state)

    assert result["failure_reason"] == MAX_ITERATIONS_EXCEEDED
    assert result["next_agent"] == "error_report"
    assert "task_status" not in result, "硬上限不直接设终态，由 error_report_node 统一落"
    assert result["iteration_count"] == 4
    llm.messages.create.assert_not_called()


async def test_supervisor_node_below_cap_plan_called() -> None:
    """低于上限一轮（第 cap 轮本身仍允许）：正常调 LLM，无 failure_reason。"""
    llm = _make_mock_llm()
    agent = DesktopSupervisorAgent(llm_client=llm)
    node_fn = make_supervisor_node(agent)
    state = _make_state().model_copy(update={"iteration_count": 2})

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key"}):
        with patch("src.orchestration.desktop_supervisor.DESKTOP_MAX_ITERATIONS", 3):
            result = await node_fn(state)

    assert "failure_reason" not in result
    assert result["iteration_count"] == 3
    llm.messages.create.assert_called_once()


async def test_supervisor_node_terminal_state_no_iteration_increment() -> None:
    """终态守卫优先于硬上限：DONE 时返回空增量，不递增 iteration_count。"""
    llm = _make_mock_llm()
    agent = DesktopSupervisorAgent(llm_client=llm)
    node_fn = make_supervisor_node(agent)
    state = _make_state().model_copy(
        update={"task_status": TaskStatus.DONE, "iteration_count": 99}
    )

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key"}):
        with patch("src.orchestration.desktop_supervisor.DESKTOP_MAX_ITERATIONS", 3):
            result = await node_fn(state)

    assert result == {}


# ── 9. 顶层占位 supervisor_node 始终 raise RuntimeError ─────────────────────


async def test_toplevel_supervisor_node_raises() -> None:
    """顶层 supervisor_node 占位函数始终 raise RuntimeError。"""
    state = _make_state()
    with pytest.raises(RuntimeError, match="make_supervisor_node"):
        await supervisor_node(state)


# ── 10. StepRecord 构造测试 ───────────────────────────────────────────────────


def test_step_record_construction() -> None:
    """StepRecord 可正常构造，metadata 默认为空 dict。"""
    step = StepRecord(
        step_index=0,
        agent="perceive",
        instruction="感知屏幕",
        snapshot_ref="snap-001",
        perception_summary="屏幕摘要",
        control_error=None,
        perception_error=None,
        task_status=TaskStatus.RUNNING,
    )
    assert step.step_index == 0
    assert step.agent == "perceive"
    assert step.metadata == {}


def test_step_record_with_metadata() -> None:
    """StepRecord metadata 字段可存扩展数据，不污染其他实例。"""
    s1 = StepRecord(
        step_index=0,
        agent="control",
        instruction="点击",
        snapshot_ref=None,
        perception_summary=None,
        control_error=None,
        perception_error=None,
        task_status=TaskStatus.RUNNING,
        metadata={"action_id": "act-001"},
    )
    s2 = StepRecord(
        step_index=1,
        agent="perceive",
        instruction="感知",
        snapshot_ref=None,
        perception_summary=None,
        control_error=None,
        perception_error=None,
        task_status=TaskStatus.RUNNING,
    )
    assert s1.metadata == {"action_id": "act-001"}
    assert s2.metadata == {}


# ── 11. StepArchive 打桩（默认无操作，不抛） ─────────────────────────────────


async def test_step_archive_stub_no_op() -> None:
    """StepArchive 打桩的 archive 方法无操作，不抛异常。"""
    archive = StepArchive()
    steps = [_make_step(0), _make_step(1)]
    # 不应抛任何异常
    await archive.archive(task_id="t1", steps=steps)
