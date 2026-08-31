"""PromptLoader 单测（Task 11B）。

验收标准：
  1. render_supervisor 返回两个非空字符串（system / user）。
  2. step_history 15 步 → user_prompt 中只含 ≤10 步（CONTEXT_STEP_WINDOW 默认值）。
  3. perception_summary 2100 字符 → 截断后 ≤2000 字符出现在 user_prompt 中。
  4. 模板路径不依赖 cwd（Path(__file__).parent/"prompts" 固定）。
  5. step_window 参数生效（自定义步数截断）。
  6. summary_max_chars 参数生效（自定义摘要截断）。
  7. perception_summary=None 时不崩溃，正常返回两串。
  8. step_history 为空时不崩溃，正常返回两串。
  9. uia_hollow=True 时 user_prompt 含 UIA 空洞提示。
  10. capability_flags 非空时出现在 user_prompt 中。
  11. 模板路径固定——os.chdir 到临时目录后仍可加载模板（不依赖 cwd）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from src.agents.models.screen_snapshot import BBox, ScreenSnapshot, TextBlock, UIAElement
from src.orchestration.prompt_loader import (
    PromptLoader,
    _serialize_snapshot_compact,
    _truncate_perception_summary,
    _truncate_step_window,
    build_grounding_table,
    derive_last_step_outcome,
)
from src.orchestration.state import DesktopTaskState, StepRecord, TaskStatus

# ── 辅助：构造测试用 State ────────────────────────────────────────────────────


def _make_step(
    index: int,
    agent: str = "perceive",
    instruction: str = "感知屏幕",
) -> StepRecord:
    """构造最简 StepRecord，供测试使用。"""
    return StepRecord(
        step_index=index,
        agent=agent,
        instruction=instruction,
        snapshot_ref=None,
        perception_summary=None,
        control_error=None,
        perception_error=None,
        task_status=TaskStatus.RUNNING,
    )


def _make_state(
    *,
    task_description: str = "打开计算器",
    step_history: list[StepRecord] | None = None,
    perception_summary: str | None = "屏幕显示桌面。",
    perception_error: str | None = None,
    control_error: str | None = None,
    stall_count: int = 0,
    uia_hollow: bool = False,
    capability_flags: dict[str, bool] | None = None,
) -> DesktopTaskState:
    """构造测试用 DesktopTaskState。"""
    return DesktopTaskState(
        task_id="test-task-001",
        task_description=task_description,
        task_status=TaskStatus.RUNNING,
        step_history=step_history if step_history is not None else [],
        perception_summary=perception_summary,
        perception_error=perception_error,
        control_error=control_error,
        stall_count=stall_count,
        uia_hollow=uia_hollow,
        capability_flags=capability_flags if capability_flags is not None else {},
    )


# ── PromptLoader 实例（模块级，复用 Jinja2 Environment）─────────────────────


@pytest.fixture()
def loader() -> PromptLoader:
    """默认 PromptLoader（CONTEXT_STEP_WINDOW=10，PERCEPTION_SUMMARY_MAX_TOKENS=2000）。"""
    return PromptLoader()


# ── 1. 返回两个非空字符串 ──────────────────────────────────────────────────────


def test_render_supervisor_returns_two_nonempty_strings(loader: PromptLoader) -> None:
    """render_supervisor 返回 (system, user) 两个非空字符串。"""
    state = _make_state()
    system, user = loader.render_supervisor(state)

    assert isinstance(system, str), "system 应为 str"
    assert isinstance(user, str), "user 应为 str"
    assert len(system.strip()) > 0, "system 不得为空"
    assert len(user.strip()) > 0, "user 不得为空"


def test_render_supervisor_system_contains_role(loader: PromptLoader) -> None:
    """system prompt 包含 Supervisor 角色声明。"""
    state = _make_state()
    system, _ = loader.render_supervisor(state)
    assert "桌面任务编排器" in system or "Supervisor" in system


def test_render_supervisor_system_contains_json_schema(loader: PromptLoader) -> None:
    """system prompt 包含输出 JSON schema 关键字段名。"""
    state = _make_state()
    system, _ = loader.render_supervisor(state)
    assert "next_agent" in system
    assert "current_instruction" in system
    assert "task_status" in system


# ── 2. step_history 15 步 → user_prompt ≤10 步 ───────────────────────────────


def test_step_history_15_truncated_to_10(loader: PromptLoader) -> None:
    """step_history 传入 15 步，user_prompt 中出现的步骤数 ≤ CONTEXT_STEP_WINDOW（10）。

    标记格式 "SID{i:02d}Z"（零填充两位）确保所有标记互不为子串：
    "SID00Z"～"SID14Z" 两两不重叠，不会因 "SID1Z" 是 "SID10Z" 子串而误报。
    保留区：最近 10 步（index 5-14）；截断区：最早 5 步（index 0-4）。
    """
    steps = [_make_step(i, instruction=f"SID{i:02d}Z") for i in range(15)]
    state = _make_state(step_history=steps)
    _, user = loader.render_supervisor(state)

    # 应保留最近 10 步（index 5-14）
    for i in range(5, 15):
        assert f"SID{i:02d}Z" in user, f"步骤 {i} 应在 user_prompt 中（保留区）"

    # 最早的 5 步（index 0-4）应被截断
    for i in range(5):
        assert f"SID{i:02d}Z" not in user, f"步骤 {i} 不应在 user_prompt 中（截断区）"


def test_step_history_exactly_window_no_truncation(loader: PromptLoader) -> None:
    """step_history 恰好 10 步时，全部保留（边界）。"""
    steps = [_make_step(i, instruction=f"SID{i:02d}Z") for i in range(10)]
    state = _make_state(step_history=steps)
    _, user = loader.render_supervisor(state)

    for i in range(10):
        assert f"SID{i:02d}Z" in user, f"步骤 {i} 应全部保留（恰好 window）"


def test_step_history_fewer_than_window_no_truncation(loader: PromptLoader) -> None:
    """step_history 少于 10 步时，全部保留（无截断）。"""
    steps = [_make_step(i, instruction=f"SID{i:02d}Z") for i in range(3)]
    state = _make_state(step_history=steps)
    _, user = loader.render_supervisor(state)

    for i in range(3):
        assert f"SID{i:02d}Z" in user, f"步骤 {i} 应全部保留（少于 window）"


def test_step_history_custom_window() -> None:
    """自定义 step_window=3 时，15 步中只保留最近 3 步。

    标记格式 "SID{i:02d}Z" 确保不同 index 的标记不互为子串。
    """
    loader_3 = PromptLoader(step_window=3)
    steps = [_make_step(i, instruction=f"SID{i:02d}Z") for i in range(15)]
    state = _make_state(step_history=steps)
    _, user = loader_3.render_supervisor(state)

    # 最近 3 步（index 12-14）应保留
    for i in range(12, 15):
        assert f"SID{i:02d}Z" in user, f"步骤 {i} 应保留（自定义 window=3）"

    # 其余 12 步（index 0-11）应截断
    for i in range(12):
        assert f"SID{i:02d}Z" not in user, f"步骤 {i} 应截断（自定义 window=3）"


# ── 3. perception_summary 2100 字符 → 截断为 2000 字符 ────────────────────────


def test_perception_summary_2100_chars_truncated_to_2000(loader: PromptLoader) -> None:
    """perception_summary 传入 2100 字符，user_prompt 中只含前 2000 字符。

    精确验证：前 2000 字符的结尾标记（MARK_END）出现在 user_prompt 中，
    第 2001 字符起的内容（MARK_OVER）不出现。
    """
    # 构造 2100 字符：前 2000 字符 + 后 100 字符（使用唯一标记）
    body_2000 = "A" * 1990 + "MARK_END__"  # 长度 = 1990 + 10 = 2000
    extra_100 = "MARK_OVER" + "B" * 91  # 长度 = 9 + 91 = 100
    long_summary = body_2000 + extra_100  # 总长 2100

    assert len(long_summary) == 2100, f"测试数据长度错误：{len(long_summary)}"
    assert len(body_2000) == 2000, f"前段长度错误：{len(body_2000)}"

    state = _make_state(perception_summary=long_summary)
    _, user = loader.render_supervisor(state)

    # 前 2000 字符的最后部分（MARK_END__）应出现
    assert "MARK_END__" in user, "截断后的 2000 字符内容应出现在 user_prompt 中"
    # 第 2001 字符起的内容（MARK_OVER）不应出现
    assert "MARK_OVER" not in user, "超出 2000 字符的内容不应出现在 user_prompt 中"


def test_perception_summary_exactly_2000_chars_not_truncated(loader: PromptLoader) -> None:
    """perception_summary 恰好 2000 字符时，不截断（边界）。"""
    summary_2000 = "Z" * 1990 + "EXACT_END_"  # 1990 + 10 = 2000
    assert len(summary_2000) == 2000

    state = _make_state(perception_summary=summary_2000)
    _, user = loader.render_supervisor(state)

    assert "EXACT_END_" in user, "恰好 2000 字符的摘要不应截断"


def test_perception_summary_under_2000_chars_not_truncated(loader: PromptLoader) -> None:
    """perception_summary 少于 2000 字符时，不截断。"""
    summary_short = "短摘要内容：屏幕显示桌面，无目标窗口。"
    state = _make_state(perception_summary=summary_short)
    _, user = loader.render_supervisor(state)

    assert summary_short in user, "短摘要应完整出现在 user_prompt 中"


def test_perception_summary_custom_max_chars() -> None:
    """自定义 summary_max_chars=100 时，200 字符摘要被截断到 100。"""
    loader_100 = PromptLoader(summary_max_chars=100)
    body_100 = "C" * 90 + "CUSTOM_END"  # 90 + 10 = 100
    extra = "CUSTOM_OVER" + "D" * 89  # 11 + 89 = 100
    long_summary = body_100 + extra  # 总长 200

    state = _make_state(perception_summary=long_summary)
    _, user = loader_100.render_supervisor(state)

    assert "CUSTOM_END" in user, "前 100 字符内容应出现在 user_prompt 中"
    assert "CUSTOM_OVER" not in user, "超出 100 字符的内容不应出现"


# ── 4. 路径不依赖 cwd ─────────────────────────────────────────────────────────


def test_template_path_independent_of_cwd(tmp_path: Path) -> None:
    """os.chdir 到临时目录后，PromptLoader 仍可正常加载模板（路径固定，不依赖 cwd）。

    Args:
        tmp_path: pytest 内置临时目录 fixture。
    """
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        # 在与项目目录完全无关的 cwd 下构造 PromptLoader 并渲染
        loader_in_tmp = PromptLoader()
        state = _make_state()
        system, user = loader_in_tmp.render_supervisor(state)
        assert len(system.strip()) > 0, "system 不应为空（cwd 无关）"
        assert len(user.strip()) > 0, "user 不应为空（cwd 无关）"
    finally:
        os.chdir(original_cwd)


# ── 5. None / 空值边界 ────────────────────────────────────────────────────────


def test_perception_summary_none_no_crash(loader: PromptLoader) -> None:
    """perception_summary=None 时不崩溃，正常返回两个非空字符串。"""
    state = _make_state(perception_summary=None)
    system, user = loader.render_supervisor(state)
    assert len(system.strip()) > 0
    assert len(user.strip()) > 0


def test_step_history_empty_no_crash(loader: PromptLoader) -> None:
    """step_history 为空 list 时不崩溃，正常返回两个非空字符串。"""
    state = _make_state(step_history=[])
    system, user = loader.render_supervisor(state)
    assert len(system.strip()) > 0
    assert len(user.strip()) > 0


# ── 6. uia_hollow / capability_flags 注入 ────────────────────────────────────


def test_uia_hollow_true_in_user_prompt(loader: PromptLoader) -> None:
    """uia_hollow=True 时 user_prompt 含 UIA 空洞相关提示（坐标点击）。"""
    state = _make_state(uia_hollow=True)
    _, user = loader.render_supervisor(state)
    # 模板 supervisor_user.jinja2 在 uia_hollow=True 时注入 UIA 相关警告
    assert "UIA" in user or "坐标点击" in user, "uia_hollow=True 时 user_prompt 应含 UIA 空洞提示"


def test_uia_hollow_false_no_uia_warning(loader: PromptLoader) -> None:
    """uia_hollow=False 时 user_prompt 不含 UIA 空洞警告。"""
    state = _make_state(uia_hollow=False)
    _, user = loader.render_supervisor(state)
    assert "UIA 空洞" not in user


def test_capability_flags_appear_in_user_prompt(loader: PromptLoader) -> None:
    """capability_flags 非空时，flag 名称出现在 user_prompt 中。"""
    flags = {"ocr_enabled": True, "screenshot_enabled": False}
    state = _make_state(capability_flags=flags)
    _, user = loader.render_supervisor(state)
    assert "ocr_enabled" in user
    assert "screenshot_enabled" in user


def test_task_description_in_user_prompt(loader: PromptLoader) -> None:
    """task_description 出现在 user_prompt 中。"""
    desc = "打开微信并查看最新消息"
    state = _make_state(task_description=desc)
    _, user = loader.render_supervisor(state)
    assert desc in user


# ── 7. 内部辅助函数单测 ──────────────────────────────────────────────────────


class TestTruncateStepWindow:
    """_truncate_step_window 内部函数单测。"""

    def test_more_than_window_returns_recent(self) -> None:
        """超过 window 步时返回最近 window 步。"""
        steps = [_make_step(i) for i in range(15)]
        result = _truncate_step_window(steps, window=10)
        assert len(result) == 10
        assert result[0]["step_index"] == 5  # 第 6 步（index 5）
        assert result[-1]["step_index"] == 14  # 最后一步

    def test_exactly_window_returns_all(self) -> None:
        """恰好 window 步时返回全部。"""
        steps = [_make_step(i) for i in range(10)]
        result = _truncate_step_window(steps, window=10)
        assert len(result) == 10

    def test_fewer_than_window_returns_all(self) -> None:
        """少于 window 步时返回全部（无截断）。"""
        steps = [_make_step(i) for i in range(3)]
        result = _truncate_step_window(steps, window=10)
        assert len(result) == 3

    def test_empty_returns_empty(self) -> None:
        """空 list 返回空 dict list。"""
        result = _truncate_step_window([], window=10)
        assert result == []

    def test_returns_dict_list(self) -> None:
        """返回值为 dict 列表（供模板消费，与 StepRecord 解耦）。"""
        steps = [_make_step(0)]
        result = _truncate_step_window(steps, window=10)
        assert isinstance(result[0], dict)
        assert "step_index" in result[0]
        assert "agent" in result[0]
        assert "instruction" in result[0]
        assert "task_status" in result[0]
        assert "control_error" in result[0]
        assert "perception_error" in result[0]


class TestTruncatePerceptionSummary:
    """_truncate_perception_summary 内部函数单测。"""

    def test_over_limit_truncated(self) -> None:
        """超出 max_chars 时截断到 max_chars。"""
        summary = "A" * 2100
        result = _truncate_perception_summary(summary, max_chars=2000)
        assert result is not None
        assert len(result) == 2000

    def test_exactly_limit_not_truncated(self) -> None:
        """恰好 max_chars 时不截断（边界）。"""
        summary = "B" * 2000
        result = _truncate_perception_summary(summary, max_chars=2000)
        assert result == summary

    def test_under_limit_not_truncated(self) -> None:
        """少于 max_chars 时不截断。"""
        summary = "短摘要"
        result = _truncate_perception_summary(summary, max_chars=2000)
        assert result == summary

    def test_none_returns_none(self) -> None:
        """None 原样返回 None。"""
        result = _truncate_perception_summary(None, max_chars=2000)
        assert result is None

    def test_empty_string_not_truncated(self) -> None:
        """空字符串不截断（返回空字符串）。"""
        result = _truncate_perception_summary("", max_chars=2000)
        assert result == ""


# ── derive_last_step_outcome（K4 紧后 §3.2，纯函数）──────────────────────────


class TestDeriveLastStepOutcome:
    """上一步结果三态派生：initial / succeeded / failed。"""

    def test_empty_history_is_initial(self) -> None:
        """空 step_history → initial。"""
        assert derive_last_step_outcome([]) == "initial"

    def test_clean_last_step_is_succeeded(self) -> None:
        """最后一步无错误 → succeeded。"""
        steps = [_make_step(0), _make_step(1)]
        assert derive_last_step_outcome(steps) == "succeeded"

    def test_last_step_control_error_is_failed(self) -> None:
        """最后一步带 control_error → failed。"""
        bad = _make_step(1).model_copy(update={"control_error": "TOCTOU abort"})
        assert derive_last_step_outcome([_make_step(0), bad]) == "failed"

    def test_last_step_perception_error_is_failed(self) -> None:
        """最后一步带 perception_error → failed。"""
        bad = _make_step(1).model_copy(update={"perception_error": "截图失败"})
        assert derive_last_step_outcome([_make_step(0), bad]) == "failed"

    def test_earlier_error_but_last_clean_is_succeeded(self) -> None:
        """判别负对照：历史中段有错误、最后一步干净 → succeeded（只看最后一步，
        不被平铺 history 中的旧错误污染——这正是三态存在的意义）。"""
        bad = _make_step(0).model_copy(update={"control_error": "旧错误"})
        assert derive_last_step_outcome([bad, _make_step(1)]) == "succeeded"


class TestRenderLastStepOutcome:
    """render_supervisor 集成：三态驱动 user_prompt 段落。"""

    def test_failed_state_renders_guidance(self, loader: PromptLoader) -> None:
        """最后一步失败 → user_prompt 含失败指导文案（禁原样重试）。"""
        bad = _make_step(1).model_copy(update={"control_error": "元素不可交互"})
        state = _make_state(step_history=[_make_step(0), bad])
        _, user = loader.render_supervisor(state)
        assert "上一步执行**失败**" in user
        assert "不要原样重试同一动作" in user

    def test_clean_state_renders_succeeded(self, loader: PromptLoader) -> None:
        """最后一步干净 → user_prompt 含成功段落，无失败指导文案。"""
        state = _make_state(step_history=[_make_step(0), _make_step(1)])
        _, user = loader.render_supervisor(state)
        assert "上一步执行**成功**" in user
        assert "不要原样重试同一动作" not in user

    def test_empty_history_renders_initial(self, loader: PromptLoader) -> None:
        """空历史 → user_prompt 含第一步提示。"""
        state = _make_state(step_history=[])
        _, user = loader.render_supervisor(state)
        assert "任务的第一步" in user


# ── render_action_generation / _serialize_snapshot_compact（蓝图 PR-β 任务 7/10）


def _make_snapshot(
    uia_elements: list[Any] | None = None,
    text_blocks: list[Any] | None = None,
    visual_objects: list[Any] | None = None,
    uia_hollow: bool = False,
) -> ScreenSnapshot:
    """构造最小 ScreenSnapshot（供 grounding 序列化测试）。"""
    return ScreenSnapshot(
        snapshot_id="snap-render-001",
        timestamp_ms=1_700_000_000_000,
        screen_width=1920,
        screen_height=1080,
        active_window_title="测试窗口",
        uia_elements=uia_elements or [],
        text_blocks=text_blocks or [],
        visual_objects=visual_objects or [],
        screenshot_path=None,
        perception_mode="uia_ocr",
        capability_flags={},
        uia_hollow=uia_hollow,
    )


def _make_uia_element(name: str = "确定", bbox: BBox | None = None) -> UIAElement:
    return UIAElement(
        element_id="uia_1234_0",
        control_type="Button",
        name=name,
        automation_id="btnOK",
        bbox=bbox or BBox(x=100, y=200, width=80, height=30),
        is_enabled=True,
        is_visible=True,
        value=None,
        source="uia",
    )


def _make_text_block(text: str = "搜索", bbox: BBox | None = None) -> TextBlock:
    return TextBlock(
        block_id="ocr_snap_0",
        text=text,
        bbox=bbox or BBox(x=10, y=20, width=40, height=15),
        confidence=0.95,
        source="ocr_rapidocr",
    )


class TestSerializeSnapshotCompact:
    """`_serialize_snapshot_compact` 纯函数测试。"""

    def test_empty_snapshot_returns_empty_string(self) -> None:
        snapshot = _make_snapshot()
        assert _serialize_snapshot_compact(snapshot) == ""

    def test_uia_and_ocr_get_distinct_prefixes(self) -> None:
        """三类元素合并进统一表，compact id 按类型分前缀（uia:/ocr:），互不冲突。"""
        snapshot = _make_snapshot(
            uia_elements=[_make_uia_element()],
            text_blocks=[_make_text_block()],
        )
        text = _serialize_snapshot_compact(snapshot)
        assert 'id="uia:0"' in text
        assert 'id="ocr:0"' in text

    def test_rendered_line_contains_role_text_bbox(self) -> None:
        snapshot = _make_snapshot(uia_elements=[_make_uia_element(name="确定")])
        text = _serialize_snapshot_compact(snapshot)
        assert 'role="Button"' in text
        assert 'text="确定"' in text
        assert 'bbox="100,200,80,30"' in text

    def test_truncation_drops_overflow_elements_and_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """超出字符预算的元素被丢弃，不出现在渲染文本中，记 warning。"""
        elements = [_make_uia_element(name=f"元素{i}") for i in range(50)]
        snapshot = _make_snapshot(uia_elements=elements)

        with caplog.at_level("WARNING"):
            text = _serialize_snapshot_compact(snapshot, max_chars=200)

        assert len(text) <= 200 + 100  # 允许最后一行边界宽裕，核心断言在下方
        assert "uia:0" in text
        assert "uia:49" not in text, "超出预算的元素不应出现在渲染文本中"
        assert any("截断" in r.message for r in caplog.records)


class TestBuildGroundingTable:
    """`build_grounding_table` 与渲染文本一致性测试。"""

    def test_table_matches_rendered_ids(self) -> None:
        snapshot = _make_snapshot(
            uia_elements=[_make_uia_element()],
            text_blocks=[_make_text_block()],
        )
        table = build_grounding_table(snapshot)
        assert set(table.keys()) == {"uia:0", "ocr:0"}
        assert table["uia:0"].x == 100
        assert table["ocr:0"].x == 10

    def test_truncated_elements_absent_from_table(self) -> None:
        """截断口径与渲染文本一致：被丢弃的元素不进查找表（LLM 只能引用它
        实际看到的 id，核验才有意义）。"""
        elements = [_make_uia_element(name=f"元素{i}") for i in range(50)]
        snapshot = _make_snapshot(uia_elements=elements)

        table = build_grounding_table(snapshot, max_chars=200)

        assert "uia:0" in table
        assert "uia:49" not in table

    def test_empty_snapshot_returns_empty_table(self) -> None:
        assert build_grounding_table(_make_snapshot()) == {}


class TestRenderActionGeneration:
    """`PromptLoader.render_action_generation` 集成测试。"""

    def test_returns_nonempty_system_and_user(self, loader: PromptLoader) -> None:
        state = _make_state(task_description="测试任务")
        state = state.model_copy(update={"current_instruction": "点击确定按钮"})
        snapshot = _make_snapshot(uia_elements=[_make_uia_element()])

        system, user = loader.render_action_generation(state, snapshot)

        assert isinstance(system, str) and len(system) > 0
        assert isinstance(user, str) and len(user) > 0
        assert "点击确定按钮" in user

    def test_grounding_table_text_injected_into_user_prompt(self, loader: PromptLoader) -> None:
        state = _make_state()
        snapshot = _make_snapshot(uia_elements=[_make_uia_element(name="确定")])

        _, user = loader.render_action_generation(state, snapshot)

        assert 'id="uia:0"' in user
        assert "确定" in user

    def test_no_elements_renders_fallback_hint(self, loader: PromptLoader) -> None:
        state = _make_state()
        snapshot = _make_snapshot()

        _, user = loader.render_action_generation(state, snapshot)

        assert "未提取到任何可定位元素" in user

    def test_uia_hollow_renders_coordinate_only_guidance(self, loader: PromptLoader) -> None:
        state = _make_state(uia_hollow=True)
        snapshot = _make_snapshot(uia_elements=[_make_uia_element()], uia_hollow=True)

        _, user = loader.render_action_generation(state, snapshot)

        assert "UIA 空洞" in user
        assert "不要引用 `target_element_id`" in user

    def test_uia_not_hollow_no_coordinate_only_guidance(self, loader: PromptLoader) -> None:
        state = _make_state(uia_hollow=False)
        snapshot = _make_snapshot(uia_elements=[_make_uia_element()], uia_hollow=False)

        _, user = loader.render_action_generation(state, snapshot)

        assert "UIA 空洞" not in user

    def test_failed_last_step_renders_no_retry_guidance(self, loader: PromptLoader) -> None:
        bad = _make_step(0).model_copy(update={"control_error": "元素不可交互"})
        state = _make_state(step_history=[bad])
        snapshot = _make_snapshot()

        _, user = loader.render_action_generation(state, snapshot)

        assert "上一步执行**失败**" in user
        assert "请勿原样重试同一动作" in user
