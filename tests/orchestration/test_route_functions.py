"""路由函数全分支单元测试（Task 10BC）。

覆盖三个条件边路由函数的全部分支：
  route_after_supervisor — 6 个路径（DONE/FAILED/stall超阈/perceive/control/browser/默认）
  route_after_control   — 2 个路径（有错误→stall_detect / 正常→supervisor）
  route_after_stall     — 2 个路径（超阈→error_report / 未超→supervisor）

重点覆盖：
  - 感知失败停滞路径（perception_error 非 None + stall_count 超阈 → error_report）
  - is_browser_task 纯函数（多关键词）
  - DONE/FAILED 均路由 memory_flush
  - 默认 next_agent 路由 error_report
"""

from __future__ import annotations

from src.orchestration.desktop_graph import (
    STALL_THRESHOLD,
    is_browser_task,
    route_after_control,
    route_after_stall,
    route_after_supervisor,
)
from src.orchestration.desktop_supervisor import MAX_ITERATIONS_EXCEEDED
from src.orchestration.state import DesktopTaskState, TaskStatus

# ── 辅助构造 ──────────────────────────────────────────────────────────────────


def _make_state(**kwargs: object) -> DesktopTaskState:
    """构造 DesktopTaskState，只覆盖指定字段。"""
    defaults: dict[str, object] = {
        "task_id": "test-task-1",
        "task_description": "测试任务",
        "task_status": TaskStatus.RUNNING,
        "next_agent": "perceive",
        "current_instruction": "截图感知当前桌面",
        "stall_count": 0,
        "perception_error": None,
        "control_error": None,
    }
    defaults.update(kwargs)
    return DesktopTaskState(**defaults)


# ── route_after_supervisor 测试 ────────────────────────────────────────────────


class TestRouteAfterSupervisor:
    """route_after_supervisor 全分支测试。"""

    def test_done_routes_to_memory_flush(self) -> None:
        """task_status=DONE → memory_flush（规则 1）。"""
        state = _make_state(task_status=TaskStatus.DONE, next_agent="perceive")
        assert route_after_supervisor(state) == "memory_flush"

    def test_failed_routes_to_memory_flush(self) -> None:
        """task_status=FAILED → memory_flush（规则 1，不是 error_report）。"""
        state = _make_state(task_status=TaskStatus.FAILED, next_agent="perceive")
        assert route_after_supervisor(state) == "memory_flush"

    def test_stall_threshold_routes_to_error_report(self) -> None:
        """stall_count >= STALL_THRESHOLD → error_report（规则 2）。"""
        state = _make_state(stall_count=STALL_THRESHOLD, task_status=TaskStatus.RUNNING)
        assert route_after_supervisor(state) == "error_report"

    def test_stall_above_threshold_routes_to_error_report(self) -> None:
        """stall_count > STALL_THRESHOLD 也 → error_report（规则 2）。"""
        state = _make_state(stall_count=STALL_THRESHOLD + 5)
        assert route_after_supervisor(state) == "error_report"

    def test_stall_below_threshold_continues_normal_routing(self) -> None:
        """stall_count < STALL_THRESHOLD 时走正常路由（规则 3）。"""
        state = _make_state(stall_count=STALL_THRESHOLD - 1, next_agent="perceive")
        assert route_after_supervisor(state) == "perceive"

    def test_next_agent_perceive(self) -> None:
        """next_agent="perceive" → perceive（规则 3）。"""
        state = _make_state(next_agent="perceive")
        assert route_after_supervisor(state) == "perceive"

    def test_next_agent_control(self) -> None:
        """next_agent="control" → control（规则 4）。"""
        state = _make_state(next_agent="control")
        assert route_after_supervisor(state) == "control"

    def test_browser_instruction_routes_to_playwright(self) -> None:
        """current_instruction 含浏览器关键词 → playwright（规则 5）。"""
        state = _make_state(
            next_agent="unknown_agent",
            current_instruction="打开 https://example.com 搜索内容",
        )
        assert route_after_supervisor(state) == "playwright"

    def test_unknown_next_agent_routes_to_error_report(self) -> None:
        """未识别 next_agent 且非浏览器任务 → error_report（默认）。"""
        state = _make_state(
            next_agent="mystery_agent",
            current_instruction="执行某个未知操作",
        )
        assert route_after_supervisor(state) == "error_report"

    def test_done_takes_priority_over_stall(self) -> None:
        """DONE + stall_count 超阈 → memory_flush（规则 1 优先于规则 3）。"""
        state = _make_state(
            task_status=TaskStatus.DONE,
            stall_count=STALL_THRESHOLD + 10,
        )
        assert route_after_supervisor(state) == "memory_flush"

    def test_failure_reason_routes_to_error_report(self) -> None:
        """failure_reason 非 None → error_report（规则 2），优先于 next_agent。

        K4 紧后 §3.3：回路硬上限命中后 supervisor 设 failure_reason，
        即使 next_agent 是合法 Worker 也不得继续派发。
        """
        state = _make_state(
            failure_reason=MAX_ITERATIONS_EXCEEDED,
            next_agent="perceive",
        )
        assert route_after_supervisor(state) == "error_report"

    def test_terminal_takes_priority_over_failure_reason(self) -> None:
        """FAILED + failure_reason → memory_flush（规则 1 优先于规则 2）。

        error_report_node 落 FAILED 后 failure_reason 仍残留在 LastValue state，
        下一轮 supervisor 必须按终态收口，不得再进 error_report 死循环。
        """
        state = _make_state(
            task_status=TaskStatus.FAILED,
            failure_reason=MAX_ITERATIONS_EXCEEDED,
        )
        assert route_after_supervisor(state) == "memory_flush"

    def test_failed_takes_priority_over_stall(self) -> None:
        """FAILED + stall_count 超阈 → memory_flush（规则 1 优先于规则 2）。"""
        state = _make_state(
            task_status=TaskStatus.FAILED,
            stall_count=STALL_THRESHOLD + 10,
        )
        assert route_after_supervisor(state) == "memory_flush"

    def test_perception_failure_stall_path(self) -> None:
        """感知失败停滞路径：stall_count 达阈 → error_report。

        R3 决策：感知失败经 perceive→stall_detect→supervisor，
        stall_count 达阈后 route_after_supervisor 路由 error_report。
        此测试验证路由函数本身的分支（stall_count 已超阈时的判断）。
        """
        state = _make_state(
            perception_error="MCP 调用失败：连接超时",
            stall_count=STALL_THRESHOLD,
            task_status=TaskStatus.RUNNING,
            next_agent="perceive",
        )
        # stall_count 达阈 → error_report（不管 next_agent 是什么）
        assert route_after_supervisor(state) == "error_report"

    def test_perception_failure_below_stall_threshold_routes_perceive(self) -> None:
        """感知失败但 stall_count 未达阈 → 按 next_agent 路由（继续感知）。"""
        state = _make_state(
            perception_error="MCP 调用失败",
            stall_count=STALL_THRESHOLD - 1,
            next_agent="perceive",
        )
        assert route_after_supervisor(state) == "perceive"

    def test_waiting_confirm_routes_by_next_agent(self) -> None:
        """WAITING_CONFIRM 状态不触发终态路由（非 DONE/FAILED），按 next_agent 走。"""
        state = _make_state(
            task_status=TaskStatus.WAITING_CONFIRM,
            next_agent="control",
        )
        assert route_after_supervisor(state) == "control"

    def test_stalled_status_routes_by_next_agent_if_count_below(self) -> None:
        """STALLED 状态但 stall_count 未达阈时按 next_agent 路由。"""
        state = _make_state(
            task_status=TaskStatus.STALLED,
            stall_count=1,
            next_agent="perceive",
        )
        assert route_after_supervisor(state) == "perceive"


# ── route_after_control 测试 ───────────────────────────────────────────────────


class TestRouteAfterControl:
    """route_after_control 全分支测试。"""

    def test_no_error_routes_to_supervisor(self) -> None:
        """control_error=None + 非 FAILED → supervisor。"""
        state = _make_state(control_error=None, task_status=TaskStatus.RUNNING)
        assert route_after_control(state) == "supervisor"

    def test_control_error_routes_to_stall_detect(self) -> None:
        """control_error 非 None → stall_detect（累加停滞信号）。"""
        state = _make_state(control_error="点击失败：元素不可见")
        assert route_after_control(state) == "stall_detect"

    def test_failed_status_routes_to_stall_detect(self) -> None:
        """task_status=FAILED → stall_detect（无论 control_error）。"""
        state = _make_state(control_error=None, task_status=TaskStatus.FAILED)
        assert route_after_control(state) == "stall_detect"

    def test_failed_with_error_routes_to_stall_detect(self) -> None:
        """task_status=FAILED + control_error 非 None → stall_detect。"""
        state = _make_state(
            control_error="人工拒绝执行",
            task_status=TaskStatus.FAILED,
        )
        assert route_after_control(state) == "stall_detect"

    def test_toctou_abort_routes_to_stall_detect(self) -> None:
        """TOCTOU abort 产生 control_error → stall_detect。"""
        state = _make_state(control_error="TOCTOU abort: 界面在执行前发生变化 (action_id=act-001)")
        assert route_after_control(state) == "stall_detect"

    def test_done_status_routes_to_stall_detect(self) -> None:
        """task_status=DONE（视为终态）→ stall_detect。

        注：DONE 状态一般由 supervisor 路由 memory_flush，不经 control；
        此处验证 control→stall_detect 的条件（FAILED 判断）不误判 DONE。
        """
        # DONE 不等于 FAILED，control_error=None → supervisor
        state = _make_state(control_error=None, task_status=TaskStatus.DONE)
        assert route_after_control(state) == "supervisor"


# ── route_after_stall 测试 ─────────────────────────────────────────────────────


class TestRouteAfterStall:
    """route_after_stall 全分支测试。"""

    def test_below_threshold_routes_to_supervisor(self) -> None:
        """stall_count < STALL_THRESHOLD → supervisor。"""
        state = _make_state(stall_count=STALL_THRESHOLD - 1)
        assert route_after_stall(state) == "supervisor"

    def test_at_threshold_routes_to_error_report(self) -> None:
        """stall_count == STALL_THRESHOLD → error_report。"""
        state = _make_state(stall_count=STALL_THRESHOLD)
        assert route_after_stall(state) == "error_report"

    def test_above_threshold_routes_to_error_report(self) -> None:
        """stall_count > STALL_THRESHOLD → error_report。"""
        state = _make_state(stall_count=STALL_THRESHOLD + 3)
        assert route_after_stall(state) == "error_report"

    def test_zero_stall_count_routes_to_supervisor(self) -> None:
        """stall_count=0 → supervisor（正常初始状态）。"""
        state = _make_state(stall_count=0)
        assert route_after_stall(state) == "supervisor"

    def test_stall_count_one_routes_to_supervisor(self) -> None:
        """stall_count=1（单次停滞信号，未达阈）→ supervisor。"""
        state = _make_state(stall_count=1)
        assert route_after_stall(state) == "supervisor"

    def test_perception_failure_stall_path_via_stall_detect(self) -> None:
        """感知失败停滞路径最后一段：stall_detect 累积到阈 → error_report。

        R3 决策：perceive→stall_detect→route_after_stall。
        当 stall_count >= STALL_THRESHOLD 时路由 error_report。
        """
        state = _make_state(
            stall_count=STALL_THRESHOLD,
            perception_error="DesktopMCPConnectionError: 连接断开",
        )
        assert route_after_stall(state) == "error_report"


# ── is_browser_task 测试 ──────────────────────────────────────────────────────


class TestIsBrowserTask:
    """is_browser_task 纯函数多关键词测试。"""

    def test_http_url_is_browser_task(self) -> None:
        """包含 http:// → True。"""
        assert is_browser_task("打开 http://example.com")

    def test_https_url_is_browser_task(self) -> None:
        """包含 https:// → True。"""
        assert is_browser_task("访问 https://www.baidu.com 搜索内容")

    def test_browser_keyword_is_browser_task(self) -> None:
        """包含 browser → True。"""
        assert is_browser_task("use the browser to navigate")

    def test_chrome_keyword(self) -> None:
        """包含 chrome → True。"""
        assert is_browser_task("open chrome and search")

    def test_firefox_keyword(self) -> None:
        """包含 firefox → True。"""
        assert is_browser_task("launch firefox browser")

    def test_edge_keyword(self) -> None:
        """包含 edge → True。"""
        assert is_browser_task("open edge browser")

    def test_playwright_keyword(self) -> None:
        """包含 playwright → True。"""
        assert is_browser_task("use playwright to click button")

    def test_chinese_browser_keyword(self) -> None:
        """包含中文浏览器 → True。"""
        assert is_browser_task("打开浏览器访问百度")

    def test_chinese_open_webpage(self) -> None:
        """包含「打开网页」→ True。"""
        assert is_browser_task("打开网页 www.example.com")

    def test_chinese_visit_url(self) -> None:
        """包含「访问网址」→ True。"""
        assert is_browser_task("访问网址 example.com")

    def test_desktop_task_is_not_browser(self) -> None:
        """桌面操作指令 → False。"""
        assert not is_browser_task("点击桌面上的记事本图标")

    def test_empty_instruction_is_not_browser(self) -> None:
        """空字符串 → False。"""
        assert not is_browser_task("")

    def test_generic_instruction_is_not_browser(self) -> None:
        """通用操作指令 → False。"""
        assert not is_browser_task("截图并分析当前桌面状态")

    def test_case_insensitive_matching(self) -> None:
        """关键词大小写不敏感。"""
        assert is_browser_task("Open CHROME and navigate")
        assert is_browser_task("Use FIREFOX to access")
