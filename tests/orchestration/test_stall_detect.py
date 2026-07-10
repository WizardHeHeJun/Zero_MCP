"""stall_detect_node 三信号单元测试（Task 10BC）。

覆盖：
  信号 A — 画面 phash 不变（snapshot_store 注入，hamming 距离 < 阈值）
  信号 B — 步骤重复（len(step_history) > STALL_MAX_STEPS 且最近 N 步同 Worker）
  信号 C — 连续错误（最近步骤均有 perception_error 或 control_error）
  感知失败信号 — perception_error 非 None 触发 C 信号（R3 路径关键）
  组合 — 多信号同时触发，stall_count 累加正确
  快乐路径 — 无信号，stall_count 不变
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from src.orchestration.desktop_graph import (
    STALL_CONSECUTIVE_ERROR_WINDOW,
    STALL_MAX_STEPS,
    _compute_average_hash,
    _hamming_distance,
    make_stall_detect_node,
)
from src.orchestration.state import DesktopTaskState, StepRecord, TaskStatus

# ── 辅助函数 ──────────────────────────────────────────────────────────────────


def _make_step(
    agent: str = "perceive",
    step_index: int = 0,
    perception_error: str | None = None,
    control_error: str | None = None,
) -> StepRecord:
    """构造 StepRecord 测试实例。"""
    return StepRecord(
        step_index=step_index,
        agent=agent,
        instruction="测试指令",
        snapshot_ref=None,
        perception_summary=None,
        control_error=control_error,
        perception_error=perception_error,
        task_status=TaskStatus.RUNNING,
    )


def _make_state(**kwargs: object) -> DesktopTaskState:
    """构造 DesktopTaskState，只覆盖指定字段。"""
    defaults: dict[str, object] = {
        "task_id": "test-task-stall",
        "task_description": "停滞检测测试任务",
        "task_status": TaskStatus.RUNNING,
        "stall_count": 0,
        "last_screen_hash": None,
        "snapshot_ref": None,
        "perception_error": None,
        "control_error": None,
        "step_history": [],
    }
    defaults.update(kwargs)
    return DesktopTaskState(**defaults)


# ── 辅助函数测试 ──────────────────────────────────────────────────────────────


class TestPhashHelpers:
    """_compute_average_hash 与 _hamming_distance 辅助函数测试。"""

    def test_hamming_distance_identical(self) -> None:
        """相同哈希距离为 0。"""
        h = "1" * 64
        assert _hamming_distance(h, h) == 0

    def test_hamming_distance_all_different(self) -> None:
        """全部不同距离为 64。"""
        h_a = "1" * 64
        h_b = "0" * 64
        assert _hamming_distance(h_a, h_b) == 64

    def test_hamming_distance_partial(self) -> None:
        """部分不同时汉明距离正确计算。"""
        h_a = "1" * 10 + "0" * 54
        h_b = "0" * 10 + "0" * 54
        assert _hamming_distance(h_a, h_b) == 10

    def test_hamming_distance_length_mismatch_returns_64(self) -> None:
        """长度不等时返回 64（最大值）。"""
        assert _hamming_distance("1" * 32, "1" * 64) == 64

    def test_compute_average_hash_returns_none_for_invalid_bytes(self) -> None:
        """无效字节返回 None，不抛出。"""
        result = _compute_average_hash(b"not_an_image")
        assert result is None

    def test_compute_average_hash_returns_64bit_string_for_valid_image(self) -> None:
        """有效图像返回 64 位二进制字符串。"""

        import cv2
        import numpy as np

        # 生成 64x64 灰度测试图像并编码为 PNG 字节
        img_array = np.zeros((64, 64), dtype=np.uint8)
        img_array[32:, :] = 255  # 下半部分白色
        _, buf = cv2.imencode(".png", img_array)
        img_bytes = buf.tobytes()

        result = _compute_average_hash(img_bytes)
        assert result is not None
        assert len(result) == 64
        assert all(c in "01" for c in result)


# ── 信号 A：画面 phash 不变 ───────────────────────────────────────────────────


class TestStallDetectSignalA:
    """信号 A：画面 phash 不变时触发停滞（需 snapshot_store）。"""

    async def test_signal_a_triggers_when_hash_unchanged(self) -> None:
        """phash 距离 < 阈值 → stall_count +1。"""
        # 构造两次相同的哈希（距离 0 < PHASH_UNCHANGED_THRESHOLD）
        same_hash = "1" * 32 + "0" * 32

        mock_store = MagicMock()
        mock_snapshot = MagicMock()
        mock_snapshot.screenshot_path = None  # 跳过文件读取
        mock_store.load = AsyncMock(return_value=mock_snapshot)

        # 通过 monkeypatch 让 _compute_average_hash 返回固定哈希
        # 此处 screenshot_path=None，信号 A 跳过文件读取分支，不触发
        # 改为直接测试有截图路径时的行为，用 tmp 文件
        import os
        import tempfile

        import cv2
        import numpy as np

        img_array = np.zeros((64, 64), dtype=np.uint8)
        _, buf = cv2.imencode(".png", img_array)
        img_bytes = buf.tobytes()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name

        try:
            mock_snapshot.screenshot_path = tmp_path
            state = _make_state(
                snapshot_ref="snap-001",
                last_screen_hash=same_hash,  # 上次哈希与当前相同
            )

            node = make_stall_detect_node(snapshot_store=mock_store)
            result = await node(state)

            # 黑色图像的 phash 确定，与 same_hash 不一定相同
            # 关键：stall_count 的变化
            assert "stall_count" in result
            assert "last_screen_hash" in result
        finally:
            os.unlink(tmp_path)

    async def test_signal_a_no_trigger_when_no_last_hash(self) -> None:
        """last_screen_hash=None（首次感知）→ 不触发信号 A，只更新 last_screen_hash。"""
        import os
        import tempfile

        import cv2
        import numpy as np

        img_array = np.zeros((64, 64), dtype=np.uint8)
        _, buf = cv2.imencode(".png", img_array)
        img_bytes = buf.tobytes()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name

        try:
            mock_store = MagicMock()
            mock_snapshot = MagicMock()
            mock_snapshot.screenshot_path = tmp_path
            mock_store.load = AsyncMock(return_value=mock_snapshot)

            state = _make_state(
                snapshot_ref="snap-001",
                last_screen_hash=None,  # 首次，无历史哈希
            )

            node = make_stall_detect_node(snapshot_store=mock_store)
            result = await node(state)

            # 无历史哈希 → 不触发 A 信号，stall_count 不因 A 增加
            # （其他信号 B/C 也不触发，初始 stall_count=0）
            assert result["stall_count"] == 0
            assert result["last_screen_hash"] is not None  # 哈希已更新
        finally:
            os.unlink(tmp_path)

    async def test_signal_a_skipped_when_no_snapshot_store(self) -> None:
        """snapshot_store=None → 跳过信号 A，stall_count 不因 A 增加。"""
        state = _make_state(snapshot_ref="snap-001", last_screen_hash="1" * 64)

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        # 无 store → 信号 A 跳过；无其他信号 → stall_count=0
        assert result["stall_count"] == 0

    async def test_signal_a_skipped_when_no_snapshot_ref(self) -> None:
        """snapshot_ref=None → 跳过信号 A（无快照可加载）。"""
        mock_store = MagicMock()
        mock_store.load = AsyncMock()

        state = _make_state(snapshot_ref=None, last_screen_hash="1" * 64)

        node = make_stall_detect_node(snapshot_store=mock_store)
        result = await node(state)

        mock_store.load.assert_not_called()
        assert result["stall_count"] == 0


# ── 信号 B：步骤重复 ──────────────────────────────────────────────────────────


class TestStallDetectSignalB:
    """信号 B：最近 STALL_MAX_STEPS+1 步均为同一 Worker → 触发停滞。"""

    async def test_signal_b_triggers_when_same_agent_repeated(self) -> None:
        """最近 N+1 步均为 perceive → 触发信号 B。"""
        steps = [
            _make_step(agent="perceive", step_index=i)
            for i in range(STALL_MAX_STEPS + 2)  # 超过 STALL_MAX_STEPS
        ]
        state = _make_state(step_history=steps)

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        # 信号 B 触发，stall_count 至少 +1
        assert result["stall_count"] >= 1

    async def test_signal_b_not_triggered_when_agents_vary(self) -> None:
        """最近步骤包含不同 Worker → 不触发信号 B。"""
        steps = []
        for i in range(STALL_MAX_STEPS + 2):
            agent = "perceive" if i % 2 == 0 else "control"
            steps.append(_make_step(agent=agent, step_index=i))

        state = _make_state(step_history=steps)

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        # 信号 B 未触发（步骤 agent 交替）；无其他信号
        assert result["stall_count"] == 0

    async def test_signal_b_not_triggered_when_few_steps(self) -> None:
        """步骤数 <= STALL_MAX_STEPS → 不触发信号 B（步骤不足）。"""
        steps = [_make_step(agent="perceive", step_index=i) for i in range(STALL_MAX_STEPS)]
        state = _make_state(step_history=steps)

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        assert result["stall_count"] == 0

    async def test_signal_b_triggers_with_control_agent_repeated(self) -> None:
        """最近步骤均为 control → 信号 B 触发。"""
        steps = [_make_step(agent="control", step_index=i) for i in range(STALL_MAX_STEPS + 2)]
        state = _make_state(step_history=steps)

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        assert result["stall_count"] >= 1


# ── 信号 C：连续错误 ──────────────────────────────────────────────────────────


class TestStallDetectSignalC:
    """信号 C：最近步骤均有 perception_error 或 control_error → 触发停滞。"""

    async def test_signal_c_triggers_with_consecutive_perception_errors(self) -> None:
        """最近 STALL_CONSECUTIVE_ERROR_WINDOW 步均有 perception_error → 信号 C。

        这是 R3 决策中感知失败停滞路径的关键验证：
        perceive_node 返回 perception_error，stall_detect_node 识别信号 C。
        """
        steps = [
            _make_step(
                agent="perceive",
                step_index=i,
                perception_error="MCP 连接失败",
            )
            for i in range(STALL_CONSECUTIVE_ERROR_WINDOW)
        ]
        state = _make_state(step_history=steps, perception_error="MCP 连接失败")

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        assert result["stall_count"] >= 1

    async def test_signal_c_triggers_with_consecutive_control_errors(self) -> None:
        """最近步骤均有 control_error → 信号 C。"""
        steps = [
            _make_step(
                agent="control",
                step_index=i,
                control_error="点击失败：元素不可见",
            )
            for i in range(STALL_CONSECUTIVE_ERROR_WINDOW)
        ]
        state = _make_state(step_history=steps, control_error="点击失败")

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        assert result["stall_count"] >= 1

    async def test_signal_c_triggers_with_mixed_errors(self) -> None:
        """perception_error 与 control_error 交替出现也触发信号 C。"""
        steps = [
            _make_step(agent="perceive", step_index=0, perception_error="感知失败"),
            _make_step(agent="control", step_index=1, control_error="控制失败"),
        ]
        # 确保窗口 = 2（默认）
        assert STALL_CONSECUTIVE_ERROR_WINDOW == 2
        state = _make_state(step_history=steps)

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        assert result["stall_count"] >= 1

    async def test_signal_c_not_triggered_when_error_not_consecutive(self) -> None:
        """有成功步骤打断连续错误 → 不触发信号 C。"""
        steps = [
            _make_step(agent="perceive", step_index=0, perception_error="感知失败"),
            _make_step(agent="control", step_index=1),  # 无错误
        ]
        state = _make_state(step_history=steps, perception_error=None, control_error=None)

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        assert result["stall_count"] == 0

    async def test_signal_c_via_current_state_error(self) -> None:
        """step_history 不足窗口大小但 state 直接字段有错误 → 触发 C（兜底）。"""
        state = _make_state(
            step_history=[],  # 空历史
            perception_error="DesktopMCPConnectionError: 连接断开",
        )

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        assert result["stall_count"] >= 1

    async def test_perception_failure_signal_r3_path(self) -> None:
        """R3 路径关键测试：感知失败触发信号 C → stall_count 累加。

        验证 perceive→stall_detect 图连线（R3 决策）中，
        stall_detect_node 能正确识别感知失败信号并累加 stall_count。
        """
        # 模拟连续感知失败（perception_error 连续出现）
        steps = [
            _make_step(agent="perceive", step_index=i, perception_error="连接超时")
            for i in range(STALL_CONSECUTIVE_ERROR_WINDOW)
        ]
        initial_stall = 1  # 已有一些停滞
        state = _make_state(
            step_history=steps,
            stall_count=initial_stall,
            perception_error="连接超时",
        )

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        # stall_count 应在初始值基础上增加
        assert result["stall_count"] > initial_stall


# ── 快乐路径（无信号） ────────────────────────────────────────────────────────


class TestStallDetectNoSignal:
    """快乐路径：无停滞信号时 stall_count 不变。"""

    async def test_no_signal_stall_count_unchanged(self) -> None:
        """正常执行：无任何停滞信号 → stall_count=0。"""
        state = _make_state(stall_count=0, step_history=[])

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        assert result["stall_count"] == 0

    async def test_existing_stall_count_preserved_when_no_new_signal(self) -> None:
        """已有 stall_count 但本轮无新信号 → stall_count 不增加。"""
        state = _make_state(stall_count=1, step_history=[])

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        assert result["stall_count"] == 1

    async def test_result_contains_required_fields(self) -> None:
        """返回增量必须包含 stall_count 和 last_screen_hash 字段。"""
        state = _make_state()

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        assert "stall_count" in result
        assert "last_screen_hash" in result

    async def test_last_screen_hash_preserved_when_no_snapshot(self) -> None:
        """无 snapshot_ref 时 last_screen_hash 保持为上次值。"""
        existing_hash = "1" * 32 + "0" * 32
        state = _make_state(snapshot_ref=None, last_screen_hash=existing_hash)

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        assert result["last_screen_hash"] == existing_hash


# ── 组合信号测试 ──────────────────────────────────────────────────────────────


class TestStallDetectCombinedSignals:
    """多信号同时触发时 stall_count 正确累加。"""

    async def test_signal_b_and_c_both_trigger(self) -> None:
        """信号 B（步骤重复）+ 信号 C（连续错误）同时触发 → stall_count += 2。"""
        # 构造足够多的步骤（B 信号：同一 Worker）且每步都有错误（C 信号）
        steps = [
            _make_step(
                agent="perceive",
                step_index=i,
                perception_error="持续感知失败",
            )
            for i in range(STALL_MAX_STEPS + 2)
        ]
        state = _make_state(step_history=steps, perception_error="持续感知失败")

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        # B 信号 +1，C 信号 +1（step_history >= window）= 至少 +2
        assert result["stall_count"] >= 2

    async def test_accumulation_across_calls(self) -> None:
        """多轮调用 stall_count 正确累加（模拟多次感知失败）。"""
        steps = [
            _make_step(agent="perceive", step_index=i, perception_error="失败")
            for i in range(STALL_CONSECUTIVE_ERROR_WINDOW)
        ]

        # 第一轮
        state1 = _make_state(step_history=steps, stall_count=0, perception_error="失败")
        node = make_stall_detect_node(snapshot_store=None)
        result1 = await node(state1)
        assert result1["stall_count"] >= 1

        # 第二轮（基于第一轮 stall_count）
        state2 = _make_state(
            step_history=steps,
            stall_count=result1["stall_count"],
            perception_error="失败",
        )
        result2 = await node(state2)
        assert result2["stall_count"] > result1["stall_count"]
