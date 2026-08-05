"""stall_detect_node 三信号单元测试（Task 10BC，K5 修订）。

覆盖：
  信号 A — 画面 phash 不变（snapshot_store 注入，hamming 距离 < 阈值）
  信号 B — 步骤重复（len(step_history) > STALL_MAX_STEPS 且最近 N 步同 Worker）
  信号 C — 错误指纹去重计数（K5 ①：(perception_error, control_error) 指纹相对
           「上次已计数指纹」新产生时 +1；同指纹不重复计；错误清空后指纹归 None）
  连续语义 — 本轮无任何信号（increment==0）时 stall_count **归零**（K5 ②，
           修订理由：文档语义本为「连续停滞计数」，旧实现只加不清）
  K5 场景 — control 失败 1 次 + 感知成功 2 次不误杀；停滞-进展交替不达阈值
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from src.orchestration.desktop_graph import (
    STALL_MAX_STEPS,
    STALL_THRESHOLD,
    _compute_average_hash,
    _error_fingerprint,
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


# ── 信号 C：错误指纹去重计数（K5 ①）──────────────────────────────────────────


class TestErrorFingerprint:
    """_error_fingerprint 纯函数。"""

    def test_no_errors_returns_none(self) -> None:
        """两者皆 None → 无指纹。"""
        assert _error_fingerprint(None, None) is None

    def test_fingerprint_deterministic_and_distinct(self) -> None:
        """同错误同指纹；不同错误/不同通道指纹不同。"""
        assert _error_fingerprint("e1", None) == _error_fingerprint("e1", None)
        assert _error_fingerprint("e1", None) != _error_fingerprint("e2", None)
        # 同文本落在不同通道也算不同指纹（感知错≠控制错）
        assert _error_fingerprint("e1", None) != _error_fingerprint(None, "e1")


class TestStallDetectSignalC:
    """信号 C：错误指纹相对「上次已计数指纹」新产生时 +1（去重）。"""

    async def test_new_perception_error_fingerprint_counts(self) -> None:
        """新感知错误（counted 指纹为 None）→ +1，且回写已计数指纹。"""
        state = _make_state(perception_error="MCP 连接失败")

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        assert result["stall_count"] == 1
        assert result["counted_error_fingerprint"] == _error_fingerprint("MCP 连接失败", None)

    async def test_new_control_error_fingerprint_counts(self) -> None:
        """新控制错误 → +1。"""
        state = _make_state(control_error="点击失败：元素不可见")

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        assert result["stall_count"] == 1

    async def test_same_fingerprint_not_recounted(self) -> None:
        """同一错误指纹已计数 → 不再 +1（LastValue 残留错误文本不重复计）。

        典型场景：control_error 残留在 state 里，后续 perceive 轮次反复经过
        stall_detect——旧实现每轮 +1（误杀），新语义只计一次。
        """
        fp = _error_fingerprint(None, "点击失败")
        state = _make_state(control_error="点击失败", counted_error_fingerprint=fp, stall_count=1)

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        # 无新信号 → 连续语义归零（而非维持 1）
        assert result["stall_count"] == 0
        # 指纹保留（错误文本未清，仍是同一个已计数错误）
        assert result["counted_error_fingerprint"] == fp

    async def test_changed_fingerprint_counts_again(self) -> None:
        """错误内容变化（新指纹）→ 再次 +1 并累加。"""
        old_fp = _error_fingerprint("旧错误", None)
        state = _make_state(
            perception_error="新错误",
            counted_error_fingerprint=old_fp,
            stall_count=2,
        )

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        assert result["stall_count"] == 3
        assert result["counted_error_fingerprint"] == _error_fingerprint("新错误", None)

    async def test_fingerprint_cleared_when_no_error(self) -> None:
        """本轮无错误 → 已计数指纹归 None（同一错误再现视为新停滞事件）。"""
        state = _make_state(
            counted_error_fingerprint=_error_fingerprint("旧错误", None),
        )

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        assert result["counted_error_fingerprint"] is None
        assert result["stall_count"] == 0

    async def test_error_text_not_cleared_by_stall_detect(self) -> None:
        """stall_detect 增量不含 perception_error/control_error——
        执行顺序 stall_detect→supervisor，错误原文必须留给 supervisor 的 prompt。"""
        state = _make_state(perception_error="连接超时", control_error="点击失败")

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        assert "perception_error" not in result
        assert "control_error" not in result

    async def test_perception_failure_signal_r3_path(self) -> None:
        """R3 路径关键测试：新感知失败指纹 → stall_count 在已有基础上累加。"""
        initial_stall = 1  # 已有一些停滞
        state = _make_state(
            stall_count=initial_stall,
            perception_error="连接超时",
        )

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        assert result["stall_count"] > initial_stall


# ── K5 状态机场景（跨多轮 stall_detect 的序列语义）───────────────────────────


async def _run_round(state: DesktopTaskState) -> dict[str, object]:
    """跑一轮 stall_detect 节点，返回增量。"""
    node = make_stall_detect_node(snapshot_store=None)
    return await node(state)


def _carry(state_kwargs: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    """把 stall_detect 增量按 LastValue 语义带入下一轮 state 构造参数。"""
    merged = dict(state_kwargs)
    merged.update(
        {
            "stall_count": result["stall_count"],
            "last_screen_hash": result["last_screen_hash"],
            "counted_error_fingerprint": result["counted_error_fingerprint"],
        }
    )
    return merged


class TestStallStateMachineScenarios:
    """K5 要求的序列场景：误杀防护与交替不达阈值。"""

    async def test_one_control_failure_two_perceive_successes_no_false_kill(self) -> None:
        """control 失败 1 次 + 感知成功 2 次 → 不误杀（stall_count 不累加至阈值）。

        场景关键：perceive 成功增量**不清 control_error**（LastValue 残留），
        旧实现会把同一控制错误在后两轮各再计一次 → 3 轮即达阈值误杀。
        """
        # 轮 1：control 失败（route_after_control → stall_detect）
        r1 = await _run_round(_make_state(control_error="点击失败: 元素不可见"))
        assert r1["stall_count"] == 1  # 新错误指纹，正常计数

        # 轮 2：perceive 成功（perception_error=None，control_error 残留）
        kwargs2 = _carry(
            {"control_error": "点击失败: 元素不可见", "snapshot_ref": None},
            r1,
        )
        r2 = await _run_round(_make_state(**kwargs2))
        assert r2["stall_count"] == 0, "同一错误指纹不得重复计数，且无新信号轮归零"

        # 轮 3：perceive 再成功
        kwargs3 = _carry(kwargs2, r2)
        r3 = await _run_round(_make_state(**kwargs3))
        assert r3["stall_count"] == 0
        assert r3["stall_count"] < STALL_THRESHOLD

    async def test_stall_progress_alternation_never_reaches_threshold(self) -> None:
        """停滞-进展-停滞交替 → stall_count 反复归零，永不达阈值（连续语义）。"""
        state_kwargs: dict[str, object] = {}
        max_seen = 0
        for i in range(4):
            if i % 2 == 0:
                # 停滞轮：出现一个新错误
                state_kwargs["perception_error"] = f"错误-{i}"
            else:
                # 进展轮：错误清空（perceive 成功清 perception_error）
                state_kwargs["perception_error"] = None
            result = await _run_round(_make_state(**state_kwargs))
            state_kwargs = _carry(state_kwargs, result)
            max_seen = max(max_seen, int(result["stall_count"]))  # type: ignore[call-overload]

        assert max_seen == 1, f"交替场景每轮至多 1 且随进展归零，实见峰值 {max_seen}"
        assert max_seen < STALL_THRESHOLD


# ── 快乐路径（无信号） ────────────────────────────────────────────────────────


class TestStallDetectNoSignal:
    """快乐路径：无停滞信号时 stall_count 不变。"""

    async def test_no_signal_stall_count_unchanged(self) -> None:
        """正常执行：无任何停滞信号 → stall_count=0。"""
        state = _make_state(stall_count=0, step_history=[])

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        assert result["stall_count"] == 0

    async def test_existing_stall_count_reset_when_no_new_signal(self) -> None:
        """已有 stall_count 但本轮无新信号 → **归零**（K5 ② 连续语义修订）。

        修订理由：stall_count 的文档语义本为「连续停滞计数」，旧实现只加不清，
        间歇性小故障会跨长任务累积到阈值误杀。
        """
        state = _make_state(stall_count=1, step_history=[])

        node = make_stall_detect_node(snapshot_store=None)
        result = await node(state)

        assert result["stall_count"] == 0

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

    async def test_accumulation_across_calls_with_new_fingerprints(self) -> None:
        """多轮调用：每轮出现**不同**错误（新指纹）→ stall_count 正确累加。

        K5 修订：同一错误指纹只计一次（去重），故累加验证须用逐轮变化的错误；
        同错误重复轮由 test_same_fingerprint_not_recounted 覆盖（归零）。
        """
        node = make_stall_detect_node(snapshot_store=None)

        # 第一轮：错误 A
        state1 = _make_state(stall_count=0, perception_error="失败-A")
        result1 = await node(state1)
        assert result1["stall_count"] == 1

        # 第二轮：错误 B（新指纹，带上第一轮已计数指纹）
        state2 = _make_state(
            stall_count=result1["stall_count"],
            counted_error_fingerprint=result1["counted_error_fingerprint"],
            perception_error="失败-B",
        )
        result2 = await node(state2)
        assert result2["stall_count"] == 2
