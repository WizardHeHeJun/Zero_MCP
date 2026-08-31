"""编排层状态机与接线加固测试（feat/desktop-hardening T1b）。

覆盖四个修复群：
  K5 — 终态状态机：人工拒绝后 FAILED 单调保持（supervisor 终态守卫不再调 LLM）、
       同一被拒动作不再 interrupt（pending_action 清空）。
  K3 — get_graph 默认装配：默认 supervisor 注入真 PromptLoader；ANTHROPIC_API_KEY
       存在且 anthropic 包可用时自动构造 AsyncAnthropic（无包/无 key 优雅回退 None）；
       自动创建的 ScreenPerceptionAgent 与 stall 共用同一 snapshot_store（信号 A
       不再因 store 分裂静默失效——判别性：同图两轮 stall_count 增、异图不增）。
  K4 — state 回路：append_step 纯函数；perceive/control 增量真实写回 step_history
       （负对照：不手工注入历史，跑图后历史非空、supervisor prompt 含步骤行、
       信号 B 六步同 Worker 可触发）；control 成功后再入 control 报缺 pending_action
       而非重放。
  K6 — 定向感知 state 面：target_window_handle 默认值与 checkpoint 序列化。

测试红线同 test_desktop_graph_integration.py：InMemorySaver + 全 mock，
不依赖真实桌面/存储，不 import Zero。
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import cv2
import numpy as np
import pytest
from langgraph.types import Command

from src.agents.desktop_control_agent import DesktopControlAgent, make_control_node
from src.agents.models.screen_snapshot import (
    ActionResult,
    ActionRisk,
    ActionSpec,
    ScreenSnapshot,
)
from src.agents.screen_perception_agent import InMemorySnapshotStore
from src.orchestration import desktop_graph
from src.orchestration.desktop_graph import STALL_THRESHOLD, get_graph
from src.orchestration.desktop_supervisor import (
    DesktopSupervisorAgent,
    make_supervisor_node,
)
from src.orchestration.prompt_loader import PromptLoader
from src.orchestration.state import (
    ConfirmResponse,
    DesktopTaskState,
    TaskStatus,
    append_step,
)

# ── LLM 响应常量（与集成测试同口径）──────────────────────────────────────────

_R_PERCEIVE_RUNNING = (
    '{"next_agent":"perceive","current_instruction":"感知桌面状态","task_status":"RUNNING"}'
)
_R_PERCEIVE_DONE = '{"next_agent":"perceive","current_instruction":"任务完成","task_status":"DONE"}'
_R_CONTROL_RUNNING = (
    '{"next_agent":"control","current_instruction":"执行操作","task_status":"RUNNING"}'
)

# ── 测试辅助 ──────────────────────────────────────────────────────────────────


def _make_snapshot(
    snapshot_id: str = "snap-hard-001",
    screenshot_path: str | None = None,
) -> ScreenSnapshot:
    """构造测试用 ScreenSnapshot（最小字段）。"""
    return ScreenSnapshot(
        snapshot_id=snapshot_id,
        timestamp_ms=1_700_000_000_000,
        screen_width=1920,
        screen_height=1080,
        active_window_title="加固测试窗口",
        uia_elements=[],
        text_blocks=[],
        visual_objects=[],
        screenshot_path=screenshot_path,
        perception_mode="uia_ocr",
        capability_flags={},
        uia_hollow=False,
    )


def _make_mock_client(snapshots: list[ScreenSnapshot] | ScreenSnapshot | None = None) -> MagicMock:
    """构造 mock DesktopMCPClient（写操作默认成功）。"""
    client = MagicMock()
    if isinstance(snapshots, list):
        client.screen_snapshot = AsyncMock(side_effect=snapshots)
    else:
        client.screen_snapshot = AsyncMock(return_value=snapshots or _make_snapshot())
    ok = ActionResult(action_id="act-hard-001", success=True, error_message=None, ui_changed=True)
    client.click_element = AsyncMock(return_value=ok)
    client.type_text = AsyncMock(return_value=ok)
    client.send_key = AsyncMock(return_value=ok)
    client.close_window = AsyncMock(return_value=ok)
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


def _make_passthrough_action_generator() -> MagicMock:
    """构造回显 state.pending_action 的桩 ActionGeneratorAgent（同集成测试先例）。

    ActionSpec 生成层蓝图 PR-β 任务 9：route_after_supervisor 的 "control"
    现在先经 generate_action 节点。本用例测的是拒绝后终态状态机（K5），不是
    生成层，注入原样回显桩保留既有直构 pending_action 语义。
    """
    agent = MagicMock()

    async def _generate(state: Any) -> dict[str, Any]:
        pending = (
            state.pending_action
            if hasattr(state, "pending_action")
            else state.get("pending_action")
        )
        return {"pending_action": pending}

    agent.generate = _generate
    return agent


def _make_mock_memory_api() -> MagicMock:
    mock = MagicMock()
    mock.write_session_summary = AsyncMock()
    return mock


def _make_counting_llm(responses: list[str]) -> tuple[MagicMock, list[int]]:
    """构造带调用计数的 mock LLM（循环取 responses）。

    Returns:
        (mock_llm, calls)——len(calls) 即 messages.create 实际被调次数。
    """
    calls: list[int] = []

    async def _fake_create(**kwargs: Any) -> MagicMock:
        idx = len(calls) % len(responses)
        calls.append(1)
        resp = MagicMock()
        item = MagicMock()
        item.text = responses[idx]
        resp.content = [item]
        return resp

    llm = MagicMock()
    llm.messages = MagicMock()
    llm.messages.create = _fake_create
    return llm, calls


class _RecordingPromptLoader:
    """真 PromptLoader 的记录包装：捕获每轮渲染的 user prompt 供断言。"""

    def __init__(self) -> None:
        self.inner = PromptLoader()
        self.user_prompts: list[str] = []

    def render_supervisor(self, state: DesktopTaskState) -> tuple[str, str]:
        system, user = self.inner.render_supervisor(state)
        self.user_prompts.append(user)
        return system, user


def _make_destructive_action(action_id: str = "act-reject-001") -> ActionSpec:
    return ActionSpec(
        action_id=action_id,
        action_type="window_close",
        target_element_id="13579",
        coordinates=None,
        text_payload=None,
        risk_level=ActionRisk.DESTRUCTIVE,
    )


def _capture_default_supervisors(monkeypatch: pytest.MonkeyPatch) -> list[DesktopSupervisorAgent]:
    """monkeypatch desktop_graph.DesktopSupervisorAgent，捕获 get_graph 默认构造的实例。"""
    created: list[DesktopSupervisorAgent] = []
    original = desktop_graph.DesktopSupervisorAgent

    def factory(*args: Any, **kwargs: Any) -> DesktopSupervisorAgent:
        inst = original(*args, **kwargs)
        created.append(inst)
        return inst

    monkeypatch.setattr(desktop_graph, "DesktopSupervisorAgent", factory)
    return created


def _write_half_png(path: str, half: str) -> None:
    """写一张 64x64 半白 PNG（half='left' 左半白 / 'top' 上半白）。

    两种取向的 average hash 汉明距离为 32（>> PHASH_UNCHANGED_THRESHOLD=10），
    供信号 A 负对照；同一张图对自身距离 0，供正对照。
    """
    img = np.zeros((64, 64), dtype=np.uint8)
    if half == "left":
        img[:, :32] = 255
    else:
        img[:32, :] = 255
    cv2.imwrite(path, img)


# ── K4 ①：append_step 纯函数 ─────────────────────────────────────────────────


class TestAppendStepPure:
    """append_step：返回新 list、不 mutate、字段从增量提取。"""

    def test_returns_new_list_without_mutating_prev(self) -> None:
        prev = append_step(
            [], agent="perceive", instruction="第一步", increment={}, task_status="RUNNING"
        )
        result = append_step(
            prev, agent="control", instruction="第二步", increment={}, task_status="RUNNING"
        )
        assert len(result) == 2
        assert len(prev) == 1, "prev 不得被原地 mutate（interrupt 重放确定性）"
        assert result is not prev

    def test_fields_extracted_from_increment(self) -> None:
        increment = {
            "snapshot_ref": "snap-x",
            "perception_summary": "摘要",
            "perception_error": None,
            "control_error": "点击失败",
        }
        result = append_step(
            [], agent="control", instruction="点击按钮", increment=increment, task_status="FAILED"
        )
        record = result[0]
        assert record.agent == "control"
        assert record.instruction == "点击按钮"
        assert record.snapshot_ref == "snap-x"
        assert record.perception_summary == "摘要"
        assert record.control_error == "点击失败"
        assert record.perception_error is None
        assert record.task_status == "FAILED"

    def test_step_index_is_window_position(self) -> None:
        prev = append_step([], agent="a", instruction="", increment={}, task_status="RUNNING")
        result = append_step(prev, agent="b", instruction="", increment={}, task_status="RUNNING")
        assert [r.step_index for r in result] == [0, 1]


# ── K5 ③：supervisor 终态单调守卫（单元）─────────────────────────────────────


class TestSupervisorTerminalGuard:
    """task_status 已终态时 supervisor_node 不调 LLM、返回空增量。"""

    async def test_done_returns_empty_without_llm_call(self) -> None:
        llm, calls = _make_counting_llm([_R_PERCEIVE_RUNNING])
        agent = DesktopSupervisorAgent(llm_client=llm)
        node_fn = make_supervisor_node(agent)
        state = DesktopTaskState(task_id="t-done", task_status=TaskStatus.DONE)

        result = await node_fn(state)

        assert result == {}
        assert len(calls) == 0, "终态不得再调 LLM"

    async def test_failed_returns_empty_without_llm_call(self) -> None:
        llm, calls = _make_counting_llm([_R_PERCEIVE_RUNNING])
        agent = DesktopSupervisorAgent(llm_client=llm)
        node_fn = make_supervisor_node(agent)
        state = DesktopTaskState(task_id="t-failed", task_status=TaskStatus.FAILED)

        result = await node_fn(state)

        assert result == {}
        assert len(calls) == 0


# ── K5：人工拒绝后的终态状态机（图级）────────────────────────────────────────


class TestRejectionTerminalStateMachine:
    """人工拒绝 → FAILED 单调保持 + 同 action 不再 interrupt + LLM 不再被调。"""

    async def test_rejection_keeps_failed_and_never_reinterrupts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "mock-key-for-test")
        task_id = "task-reject-terminal-001"
        mock_client = _make_mock_client()
        mock_guard = _make_mock_guard(risk=ActionRisk.DESTRUCTIVE, toctou_verdict="pass")
        mock_memory = _make_mock_memory_api()
        # 只给「路由 control」一个响应：终态守卫生效时 LLM 恰好被调 1 次；
        # 守卫失效（变异）时会再次消费同一响应、重路由 control → 计数 >1（判别点）
        mock_llm, llm_calls = _make_counting_llm([_R_CONTROL_RUNNING])

        graph = get_graph(
            supervisor_agent=DesktopSupervisorAgent(
                llm_client=mock_llm, prompt_loader=PromptLoader()
            ),
            # perceive 在本场景不会被路由到，MagicMock 占位即可（工厂只闭包引用）
            perception_agent=MagicMock(),
            action_generator_agent=_make_passthrough_action_generator(),
            control_agent=DesktopControlAgent(client=mock_client, guard=mock_guard),
            memory_api=mock_memory,
            checkpointer=None,
        )
        config = {"configurable": {"thread_id": task_id}}

        result1 = await graph.ainvoke(
            DesktopTaskState(
                task_id=task_id,
                task_description="拒绝终态测试",
                task_status=TaskStatus.RUNNING,
                pending_action=_make_destructive_action(),
            ),
            config=config,
        )
        assert result1.get("__interrupt__"), "前提：DESTRUCTIVE 动作应先触发 interrupt"

        result2 = await graph.ainvoke(
            Command(resume=ConfirmResponse(confirmed=False, reason="人工拒绝")),
            config=config,
        )

        # ① FAILED 单调保持（终态守卫返回 {}，不被新 plan 覆写）
        assert result2.get("task_status") == TaskStatus.FAILED
        # ② 同一被拒动作不再 interrupt（pending_action 已清 + 不再进 control）
        assert result2.get("pending_action") is None
        assert not result2.get("__interrupt__"), "拒绝后不得再次 interrupt"
        # ③ 写操作从未执行
        mock_client.close_window.assert_not_called()
        # ④ LLM 恰好被调一次（首轮路由）；终态守卫跳过后续 plan
        assert len(llm_calls) == 1, f"终态后不得再调 LLM，实际 {len(llm_calls)} 次"
        # ⑤ 终态照常写记忆
        mock_memory.write_session_summary.assert_called()
        # ⑥ 拒绝的 control 步入史（K4：control 增量附本步）
        history = result2.get("step_history") or []
        assert any(s.agent == "control" and s.control_error for s in history)


# ── K3：get_graph 默认装配 ────────────────────────────────────────────────────


class TestGetGraphDefaultAssembly:
    """默认 supervisor 的 PromptLoader / LLM 客户端装配。"""

    def test_default_supervisor_uses_real_prompt_loader(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """K3 ②：默认 supervisor.prompt_loader 是真 PromptLoader（非 fallback）。"""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        created = _capture_default_supervisors(monkeypatch)

        get_graph(checkpointer=None)

        assert len(created) == 1
        assert isinstance(created[0].prompt_loader, PromptLoader)

    def test_default_supervisor_no_key_yields_none_llm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """K3 ③：无 ANTHROPIC_API_KEY → llm_client=None（优雅回退，不崩）。"""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        created = _capture_default_supervisors(monkeypatch)

        get_graph(checkpointer=None)

        assert created[0].llm_client is None

    def test_default_supervisor_wires_async_anthropic_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """K3 ③：key 存在且 anthropic 可 import → 自动构造 AsyncAnthropic(api_key=...)。

        本环境未装 anthropic，注入 fake module 验证「有包」分支的接线形状。
        """

        class FakeAsyncAnthropic:
            def __init__(self, api_key: str) -> None:
                self.api_key = api_key

        fake_module = types.ModuleType("anthropic")
        fake_module.AsyncAnthropic = FakeAsyncAnthropic  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "anthropic", fake_module)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key-hardening-123")
        created = _capture_default_supervisors(monkeypatch)

        get_graph(checkpointer=None)

        assert isinstance(created[0].llm_client, FakeAsyncAnthropic)
        assert created[0].llm_client.api_key == "key-hardening-123"

    def test_default_supervisor_key_without_package_falls_back_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """K3 ③：key 存在但 anthropic 包缺失 → llm_client=None（不崩，图仍可编译）。"""
        # sys.modules 置 None 令 `import anthropic` 必然 ImportError（确定性模拟无包）
        monkeypatch.setitem(sys.modules, "anthropic", None)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key-but-no-package")
        created = _capture_default_supervisors(monkeypatch)

        graph = get_graph(checkpointer=None)

        assert graph is not None
        assert created[0].llm_client is None


class TestSignalADefaultWiring:
    """K3 ①：自动创建的 PerceptionAgent 与 stall 共用 snapshot_store → 信号 A 通电。"""

    async def _run_two_perceive_rounds(
        self,
        monkeypatch: pytest.MonkeyPatch,
        screenshot_paths: tuple[str, str],
    ) -> dict[str, Any]:
        """跑「感知→感知→DONE」三轮，返回最终 state（不注入 perception_agent）。"""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "mock-key-for-test")
        task_id = "task-signal-a-wiring"
        snapshots = [
            _make_snapshot(snapshot_id="snap-r1", screenshot_path=screenshot_paths[0]),
            _make_snapshot(snapshot_id="snap-r2", screenshot_path=screenshot_paths[1]),
        ]
        mock_client = _make_mock_client(snapshots=snapshots)
        mock_llm, _ = _make_counting_llm(
            [_R_PERCEIVE_RUNNING, _R_PERCEIVE_RUNNING, _R_PERCEIVE_DONE]
        )
        store = InMemorySnapshotStore()

        graph = get_graph(
            client=mock_client,  # 不注入 perception_agent：走 K3 ① 自动创建路径
            supervisor_agent=DesktopSupervisorAgent(
                llm_client=mock_llm, prompt_loader=PromptLoader()
            ),
            memory_api=_make_mock_memory_api(),
            snapshot_store=store,
            checkpointer=None,
        )
        return await graph.ainvoke(
            DesktopTaskState(
                task_id=task_id,
                task_description="信号A 默认接线测试",
                task_status=TaskStatus.RUNNING,
            ),
            config={"configurable": {"thread_id": task_id}},
        )

    async def test_same_screenshot_two_rounds_increments_stall(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """同一张图连拍两轮 → 信号 A 触发，stall_count 增（修复前 store 分裂恒 0）。"""
        img = str(tmp_path / "same.png")
        _write_half_png(img, half="left")

        result = await self._run_two_perceive_rounds(monkeypatch, (img, img))

        assert result.get("task_status") == TaskStatus.DONE
        assert result.get("stall_count") == 1, (
            "同图两轮应触发信号 A（stall_count=1）；为 0 说明自动创建的 "
            "PerceptionAgent 与 stall 节点仍在用不同 snapshot_store（store 分裂）"
        )

    async def test_different_screenshots_do_not_increment_stall(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """判别性负对照：两轮不同图（hamming≈32）→ stall_count 不增。"""
        img_a = str(tmp_path / "a.png")
        img_b = str(tmp_path / "b.png")
        _write_half_png(img_a, half="left")
        _write_half_png(img_b, half="top")

        result = await self._run_two_perceive_rounds(monkeypatch, (img_a, img_b))

        assert result.get("task_status") == TaskStatus.DONE
        assert result.get("stall_count") == 0


# ── K4：step_history 回路（图级负对照）───────────────────────────────────────


class TestStepHistoryWiring:
    """不手工注入 step_history，验证 Worker 增量真实写回历史。"""

    async def test_two_perceive_rounds_build_history_and_feed_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """两轮 perceive 后 history len=2，且 supervisor prompt 含步骤行（非空历史）。"""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "mock-key-for-test")
        task_id = "task-history-wiring"
        mock_client = _make_mock_client()
        mock_llm, _ = _make_counting_llm(
            [_R_PERCEIVE_RUNNING, _R_PERCEIVE_RUNNING, _R_PERCEIVE_DONE]
        )
        recording_loader = _RecordingPromptLoader()

        graph = get_graph(
            client=mock_client,
            supervisor_agent=DesktopSupervisorAgent(
                llm_client=mock_llm, prompt_loader=recording_loader
            ),
            memory_api=_make_mock_memory_api(),
            checkpointer=None,
        )
        result = await graph.ainvoke(
            DesktopTaskState(
                task_id=task_id,
                task_description="历史回路测试",
                task_status=TaskStatus.RUNNING,
            ),
            config={"configurable": {"thread_id": task_id}},
        )

        history = result.get("step_history") or []
        assert len(history) == 2, f"两轮 perceive 应写回 2 条历史，实际 {len(history)}"
        assert all(s.agent == "perceive" for s in history)
        # 第三轮 supervisor（决策 DONE 前）看到的 prompt 应含前两步
        final_prompt = recording_loader.user_prompts[-1]
        assert "[步骤 0]" in final_prompt
        assert "[步骤 1]" in final_prompt

    async def test_signal_b_six_same_worker_steps_reaches_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """信号 B 负对照通电：感知成功但任务无进展（同 Worker 连续 6+ 步）→
        步骤重复信号累加至阈值 → error_report → FAILED。

        判别力：去掉 perceive 的 step_history append，信号 B 永不触发，
        本测试必红（图收不了口）。
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "mock-key-for-test")
        task_id = "task-signal-b-wiring"
        # 无 screenshot_path（信号 A 不参与）、无错误（信号 C 不参与）
        mock_client = _make_mock_client(snapshots=_make_snapshot())
        mock_llm, _ = _make_counting_llm([_R_PERCEIVE_RUNNING])

        graph = get_graph(
            client=mock_client,
            supervisor_agent=DesktopSupervisorAgent(
                llm_client=mock_llm, prompt_loader=PromptLoader()
            ),
            memory_api=_make_mock_memory_api(),
            checkpointer=None,
        )
        result = await graph.ainvoke(
            DesktopTaskState(
                task_id=task_id,
                task_description="信号B 通电测试",
                task_status=TaskStatus.RUNNING,
            ),
            # 信号 B 需 6+ 步才开始累加，8 轮 ×3 节点 + 收口 > 默认 25 步上限
            config={"configurable": {"thread_id": task_id}, "recursion_limit": 60},
        )

        assert result.get("task_status") == TaskStatus.FAILED
        assert result.get("stall_count", 0) >= STALL_THRESHOLD
        assert len(result.get("step_history") or []) >= 6


# ── K4 ②：uia_hollow 经 perceive 增量流入 prompt（坐标点击引导块）────────────


class TestUiaHollowFlowsToPrompt:
    """uia_hollow=True 快照 → perceive 增量刷新 state → jinja 渲染坐标点击引导。

    修复前 perceive 增量不含 uia_hollow，state 停留初值 False，
    supervisor_user.jinja2 的 UIA 空洞引导块永不渲染。
    """

    async def test_hollow_snapshot_renders_coordinate_click_guidance(self) -> None:
        hollow_snapshot = _make_snapshot(snapshot_id="snap-hollow").model_copy(
            update={"uia_hollow": True}
        )
        client = _make_mock_client(snapshots=hollow_snapshot)
        from src.agents.screen_perception_agent import (  # noqa: PLC0415 局部导入避免顶部堆积
            PerceptionRequest,
            ScreenPerceptionAgent,
        )

        agent = ScreenPerceptionAgent(client=client)
        increment = await agent.perceive(PerceptionRequest())
        assert increment["uia_hollow"] is True

        state = DesktopTaskState(
            task_id="t-hollow",
            task_description="微信发消息",
            uia_hollow=increment["uia_hollow"],
            perception_summary=increment["perception_summary"],
        )
        _, user_prompt = PromptLoader().render_supervisor(state)
        assert "坐标点击" in user_prompt, "uia_hollow=True 应渲染坐标点击引导块"

        # 负对照：非空洞快照不渲染引导块
        state_normal = DesktopTaskState(task_id="t-normal", uia_hollow=False)
        _, user_normal = PromptLoader().render_supervisor(state_normal)
        assert "坐标点击" not in user_normal


# ── K4 ③：control 成功后再入 control 不重放 ──────────────────────────────────


class TestControlReentryAfterSuccess:
    """control 成功清 pending_action → 再入 control 报缺动作而非重放。"""

    async def test_reentry_reports_missing_pending_action_not_replay(self) -> None:
        client = _make_mock_client()
        guard = _make_mock_guard(risk=ActionRisk.LOW_RISK, toctou_verdict="pass")
        agent = DesktopControlAgent(client=client, guard=guard)
        node_fn = make_control_node(agent)

        action = ActionSpec(
            action_id="act-reentry-001",
            action_type="click",
            target_element_id=None,
            coordinates=(10, 20),
            text_payload=None,
            risk_level=ActionRisk.LOW_RISK,
        )
        state1 = {"pending_action": action, "step_history": [], "task_status": "RUNNING"}
        result1 = await node_fn(state1)

        assert result1.get("control_error") is None
        assert result1.get("pending_action") is None, "成功后必须清 pending_action"
        assert len(result1["step_history"]) == 1
        assert result1["step_history"][0].agent == "control"

        # 再入 control：按增量后的 state（pending_action 已清）
        state2 = {
            "pending_action": result1["pending_action"],
            "step_history": result1["step_history"],
            "task_status": "RUNNING",
        }
        result2 = await node_fn(state2)

        assert result2.get("control_error") is not None
        assert "pending_action" in result2["control_error"]
        # 写操作只发生了第一次（未重放）
        client.click_element.assert_called_once()


# ── K6：state 面（默认值 + checkpoint 序列化）────────────────────────────────


class TestTargetWindowHandleState:
    """target_window_handle / counted_error_fingerprint 的 state 契约。"""

    def test_new_fields_default_values(self) -> None:
        state = DesktopTaskState()
        assert state.target_window_handle is None
        assert state.counted_error_fingerprint is None

    def test_checkpoint_json_round_trip(self) -> None:
        """checkpoint 序列化往返：新字段值不丢（LangGraph JSON 持久化口径）。"""
        state = DesktopTaskState(
            task_id="t-k6",
            target_window_handle=112233,
            counted_error_fingerprint="p='x'|c=None",
            stall_count=2,
        )
        restored = DesktopTaskState.model_validate_json(state.model_dump_json())
        assert restored.target_window_handle == 112233
        assert restored.counted_error_fingerprint == "p='x'|c=None"
        assert restored.stall_count == 2
