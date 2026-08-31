"""Jinja2 提示词模板测试（Task 11A）。

验收标准：
  1. Jinja2 FileSystemLoader 可从 TEMPLATES_DIR 加载两个模板文件。
  2. supervisor_system.jinja2 语法解析通过（无语法错误）。
  3. supervisor_user.jinja2 语法解析通过（无语法错误）。
  4. system 模板渲染（无变量注入）返回非空字符串，含关键词：
       - 角色声明（"桌面任务编排器" / "Supervisor"）
       - 输出 JSON schema 字段（"next_agent" / "current_instruction" / "task_status"）
       - 安全提醒（"[FILTERED]"）
  5. user 模板渲染含关键变量注入：
       - task_description
       - perception_summary
       - stall_count
       - errors.perception_error / errors.control_error
       - capability_flags
       - step_history_window 步骤列表（step_index / agent / instruction）
  6. uia_hollow=True 时 user 模板注入 UIA 空洞提示（"UIA 空洞" / "坐标点击"）。
  7. uia_hollow=False 时 user 模板不含 UIA 空洞提示。
  8. step_history_window 为空时模板不崩溃，输出无历史步骤提示。
  9. perception_summary 为 None / 空时模板不崩溃，输出替代提示。
  10. capability_flags 为空 dict 时模板不崩溃，输出默认配置提示。
"""

from __future__ import annotations

import pytest
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from src.orchestration.prompts import TEMPLATES_DIR

# ── 辅助：构建 Jinja2 Environment ────────────────────────────────────────────


def _make_env() -> Environment:
    """用 FileSystemLoader 从 TEMPLATES_DIR 构建 Jinja2 Environment。"""
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=False,
        keep_trailing_newline=True,
    )


def _make_default_user_vars(
    *,
    task_description: str = "打开计算器并计算 1+1",
    step_history_window: list[dict] | None = None,
    perception_summary: str | None = "屏幕显示桌面，无目标窗口。",
    last_step_outcome: str = "initial",
    errors: dict | None = None,
    capability_flags: dict | None = None,
    stall_count: int = 0,
    uia_hollow: bool = False,
) -> dict:
    """构造 supervisor_user.jinja2 所需的渲染变量（含合理默认值）。"""
    return {
        "task_description": task_description,
        "step_history_window": step_history_window if step_history_window is not None else [],
        "perception_summary": perception_summary,
        "last_step_outcome": last_step_outcome,
        "errors": errors
        if errors is not None
        else {"perception_error": None, "control_error": None},
        "capability_flags": capability_flags if capability_flags is not None else {},
        "stall_count": stall_count,
        "uia_hollow": uia_hollow,
    }


def _make_step(
    step_index: int = 0,
    agent: str = "perceive",
    instruction: str = "感知屏幕",
    task_status: str = "RUNNING",
    perception_error: str | None = None,
    control_error: str | None = None,
) -> dict:
    """构造渲染用的步骤 dict（模板只消费 dict，不依赖 StepRecord 对象）。"""
    return {
        "step_index": step_index,
        "agent": agent,
        "instruction": instruction,
        "task_status": task_status,
        "perception_error": perception_error,
        "control_error": control_error,
    }


# ── 1. FileSystemLoader 可加载两个模板文件 ─────────────────────────────────


def test_templates_dir_exists() -> None:
    """TEMPLATES_DIR 指向实际存在的目录。"""
    assert TEMPLATES_DIR.is_dir(), f"TEMPLATES_DIR 不存在：{TEMPLATES_DIR}"


def test_system_template_loadable() -> None:
    """FileSystemLoader 可加载 supervisor_system.jinja2，不抛 TemplateNotFound。"""
    env = _make_env()
    template = env.get_template("supervisor_system.jinja2")
    assert template is not None


def test_user_template_loadable() -> None:
    """FileSystemLoader 可加载 supervisor_user.jinja2，不抛 TemplateNotFound。"""
    env = _make_env()
    template = env.get_template("supervisor_user.jinja2")
    assert template is not None


def test_nonexistent_template_raises() -> None:
    """不存在的模板文件应抛 TemplateNotFound（验证 loader 正常工作）。"""
    env = _make_env()
    with pytest.raises(TemplateNotFound):
        env.get_template("nonexistent_template.jinja2")


# ── 2. 模板语法解析通过（parse，不渲染）────────────────────────────────────


def test_system_template_syntax_valid() -> None:
    """supervisor_system.jinja2 语法解析通过（Environment.parse 无异常）。"""
    env = _make_env()
    source = env.loader.get_source(env, "supervisor_system.jinja2")[0]  # type: ignore[union-attr]
    env.parse(source)  # 无异常即通过


def test_user_template_syntax_valid() -> None:
    """supervisor_user.jinja2 语法解析通过（Environment.parse 无异常）。"""
    env = _make_env()
    source = env.loader.get_source(env, "supervisor_user.jinja2")[0]  # type: ignore[union-attr]
    env.parse(source)  # 无异常即通过


# ── 3. system 模板渲染（无变量注入）────────────────────────────────────────


def test_system_template_renders_nonempty() -> None:
    """supervisor_system.jinja2 渲染结果为非空字符串。"""
    env = _make_env()
    rendered = env.get_template("supervisor_system.jinja2").render()
    assert isinstance(rendered, str)
    assert len(rendered.strip()) > 0


def test_system_template_contains_role_declaration() -> None:
    """system 模板含角色声明关键词（"桌面任务编排器" 或 "Supervisor"）。"""
    env = _make_env()
    rendered = env.get_template("supervisor_system.jinja2").render()
    assert "桌面任务编排器" in rendered or "Supervisor" in rendered


def test_system_template_contains_json_schema_fields() -> None:
    """system 模板含输出 JSON schema 的三个必要字段名。"""
    env = _make_env()
    rendered = env.get_template("supervisor_system.jinja2").render()
    assert "next_agent" in rendered
    assert "current_instruction" in rendered
    assert "task_status" in rendered


def test_system_template_contains_security_reminder() -> None:
    """system 模板含安全提醒，提到 [FILTERED] 标记。"""
    env = _make_env()
    rendered = env.get_template("supervisor_system.jinja2").render()
    assert "[FILTERED]" in rendered


# ── 4. user 模板渲染——含关键变量注入 ────────────────────────────────────


def test_user_template_renders_nonempty() -> None:
    """supervisor_user.jinja2 渲染结果为非空字符串。"""
    env = _make_env()
    rendered = env.get_template("supervisor_user.jinja2").render(_make_default_user_vars())
    assert isinstance(rendered, str)
    assert len(rendered.strip()) > 0


def test_user_template_contains_task_description() -> None:
    """user 模板渲染结果包含 task_description 内容。"""
    env = _make_env()
    desc = "打开微信并发送消息"
    rendered = env.get_template("supervisor_user.jinja2").render(
        _make_default_user_vars(task_description=desc)
    )
    assert desc in rendered


def test_user_template_contains_perception_summary() -> None:
    """user 模板渲染结果包含 perception_summary 内容。"""
    env = _make_env()
    summary = "屏幕显示：桌面，任务栏可见，无目标应用窗口。"
    rendered = env.get_template("supervisor_user.jinja2").render(
        _make_default_user_vars(perception_summary=summary)
    )
    assert summary in rendered


def test_user_template_contains_stall_count() -> None:
    """user 模板渲染结果包含 stall_count 数值。"""
    env = _make_env()
    rendered = env.get_template("supervisor_user.jinja2").render(
        _make_default_user_vars(stall_count=2)
    )
    assert "2" in rendered


def test_user_template_contains_perception_error() -> None:
    """user 模板包含 errors.perception_error 内容（failed 分支才渲染，PR #26 WARN①）。"""
    env = _make_env()
    err_msg = "MCP 连接超时，无法获取屏幕快照"
    rendered = env.get_template("supervisor_user.jinja2").render(
        _make_default_user_vars(
            last_step_outcome="failed",
            errors={"perception_error": err_msg, "control_error": None},
        )
    )
    assert err_msg in rendered


def test_user_template_contains_control_error() -> None:
    """user 模板包含 errors.control_error 内容（failed 分支才渲染，PR #26 WARN①）。"""
    env = _make_env()
    err_msg = "元素不可点击：按钮已禁用"
    rendered = env.get_template("supervisor_user.jinja2").render(
        _make_default_user_vars(
            last_step_outcome="failed",
            errors={"perception_error": None, "control_error": err_msg},
        )
    )
    assert err_msg in rendered


def test_user_template_contains_capability_flags() -> None:
    """user 模板渲染结果包含 capability_flags 中的 flag 名称。"""
    env = _make_env()
    flags = {"ocr_enabled": True, "screenshot_enabled": False}
    rendered = env.get_template("supervisor_user.jinja2").render(
        _make_default_user_vars(capability_flags=flags)
    )
    assert "ocr_enabled" in rendered
    assert "screenshot_enabled" in rendered


def test_user_template_contains_step_history() -> None:
    """user 模板渲染结果包含 step_history_window 中的步骤信息。"""
    env = _make_env()
    steps = [
        _make_step(0, "perceive", "感知桌面", "RUNNING"),
        _make_step(1, "control", "点击计算器图标", "RUNNING"),
    ]
    rendered = env.get_template("supervisor_user.jinja2").render(
        _make_default_user_vars(step_history_window=steps)
    )
    # 步骤 agent 名和指令应出现在渲染结果中
    assert "perceive" in rendered
    assert "感知桌面" in rendered
    assert "control" in rendered
    assert "点击计算器图标" in rendered


def test_user_template_step_history_shows_step_index() -> None:
    """step_history_window 中的 step_index 出现在渲染结果中。"""
    env = _make_env()
    steps = [_make_step(step_index=42, agent="perceive", instruction="测试步骤")]
    rendered = env.get_template("supervisor_user.jinja2").render(
        _make_default_user_vars(step_history_window=steps)
    )
    assert "42" in rendered


# ── 5. uia_hollow 相关测试 ──────────────────────────────────────────────────


def test_user_template_uia_hollow_true_shows_warning() -> None:
    """uia_hollow=True 时 user 模板注入 UIA 空洞警告（含"UIA 空洞"和"坐标点击"）。"""
    env = _make_env()
    rendered = env.get_template("supervisor_user.jinja2").render(
        _make_default_user_vars(uia_hollow=True)
    )
    assert "UIA 空洞" in rendered
    assert "坐标点击" in rendered


def test_user_template_uia_hollow_false_no_warning() -> None:
    """uia_hollow=False 时 user 模板不含 UIA 空洞警告。"""
    env = _make_env()
    rendered = env.get_template("supervisor_user.jinja2").render(
        _make_default_user_vars(uia_hollow=False)
    )
    assert "UIA 空洞" not in rendered
    assert "坐标点击" not in rendered


# ── 6. 边界情况——空/None 值时模板不崩溃 ────────────────────────────────


def test_user_template_empty_step_history_no_crash() -> None:
    """step_history_window 为空 list 时模板渲染不崩溃，输出无历史步骤提示。"""
    env = _make_env()
    rendered = env.get_template("supervisor_user.jinja2").render(
        _make_default_user_vars(step_history_window=[])
    )
    assert isinstance(rendered, str)
    # 应有"无历史步骤"或等价提示
    assert "无历史步骤" in rendered or "第一步" in rendered


def test_user_template_none_perception_summary_no_crash() -> None:
    """perception_summary=None 时模板渲染不崩溃，输出替代提示。"""
    env = _make_env()
    rendered = env.get_template("supervisor_user.jinja2").render(
        _make_default_user_vars(perception_summary=None)
    )
    assert isinstance(rendered, str)
    # 应有感知数据缺失的提示
    assert "perceive" in rendered or "暂无" in rendered or "感知" in rendered


def test_user_template_empty_capability_flags_no_crash() -> None:
    """capability_flags 为空 dict 时模板渲染不崩溃，输出默认配置提示。"""
    env = _make_env()
    rendered = env.get_template("supervisor_user.jinja2").render(
        _make_default_user_vars(capability_flags={})
    )
    assert isinstance(rendered, str)
    assert len(rendered.strip()) > 0


def test_user_template_zero_errors_no_crash() -> None:
    """errors 均为 None 时模板渲染不崩溃（正常情况）。"""
    env = _make_env()
    rendered = env.get_template("supervisor_user.jinja2").render(
        _make_default_user_vars(errors={"perception_error": None, "control_error": None})
    )
    assert isinstance(rendered, str)
    assert len(rendered.strip()) > 0


def test_user_template_step_with_errors_renders() -> None:
    """step_history_window 中含 perception_error/control_error 的步骤正确渲染。"""
    env = _make_env()
    steps = [
        _make_step(
            step_index=3,
            agent="perceive",
            instruction="感知屏幕",
            perception_error="截图失败：屏幕锁定",
        ),
        _make_step(
            step_index=4,
            agent="control",
            instruction="点击按钮",
            control_error="元素不可交互",
        ),
    ]
    rendered = env.get_template("supervisor_user.jinja2").render(
        _make_default_user_vars(step_history_window=steps)
    )
    assert "截图失败：屏幕锁定" in rendered
    assert "元素不可交互" in rendered


# ── last_step_outcome 三态分支（K4 紧后 §3.2）────────────────────────────────


def test_user_template_outcome_initial_branch() -> None:
    """last_step_outcome=initial → 渲染「第一步」提示，不含成功/失败段落。"""
    env = _make_env()
    rendered = env.get_template("supervisor_user.jinja2").render(
        _make_default_user_vars(last_step_outcome="initial")
    )
    assert "任务的第一步" in rendered
    assert "上一步执行**成功**" not in rendered
    assert "上一步执行**失败**" not in rendered


def test_user_template_outcome_succeeded_branch() -> None:
    """last_step_outcome=succeeded → 渲染成功段落，不含失败指导文案。"""
    env = _make_env()
    rendered = env.get_template("supervisor_user.jinja2").render(
        _make_default_user_vars(last_step_outcome="succeeded")
    )
    assert "上一步执行**成功**" in rendered
    assert "不要原样重试同一动作" not in rendered


def test_user_template_outcome_failed_branch_has_guidance() -> None:
    """last_step_outcome=failed → 渲染失败段落 + 可恢复/不可恢复指导文案
    （§3.2：错误文案写「为什么失败 + 下一步可做什么」）。"""
    env = _make_env()
    rendered = env.get_template("supervisor_user.jinja2").render(
        _make_default_user_vars(last_step_outcome="failed")
    )
    assert "上一步执行**失败**" in rendered
    assert "不要原样重试同一动作" in rendered
    assert "可恢复" in rendered
    assert "FAILED" in rendered


def test_user_template_succeeded_hides_stale_errors() -> None:
    """PR #26 审查 WARN① 收口：上一步成功时 LastValue 残留的 errors 不渲染——
    避免「上一步执行成功」与「控制错误：…」并列的自相矛盾上下文。
    旧错误原文仍经执行历史表逐步保留（不丢信息）。"""
    env = _make_env()
    stale = "TOCTOU abort: 界面在执行前发生变化 (action_id=old-001)"
    rendered = env.get_template("supervisor_user.jinja2").render(
        _make_default_user_vars(
            last_step_outcome="succeeded",
            errors={"perception_error": None, "control_error": stale},
        )
    )
    assert "上一步执行**成功**" in rendered
    assert stale not in rendered, "succeeded 分支不得渲染 LastValue 残留错误"


def test_user_template_unknown_outcome_falls_back_conservative() -> None:
    """PR #26 审查 INFO 收口：last_step_outcome 意外值走 else 兜底（保守按失败处理），
    不再静默缺整节。"""
    env = _make_env()
    rendered = env.get_template("supervisor_user.jinja2").render(
        _make_default_user_vars(last_step_outcome="")
    )
    assert "上一步结果未知" in rendered
