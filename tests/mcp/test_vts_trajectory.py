"""裸参数轨迹回放器确定性单测（TrajectoryPlayer，2026-07-31 二期）。

纯同步、无 I/O：时间全部由测试注入 ``now``（与 ``BehaviorOverlayEngine`` 同一
可测性约定，见 ``test_vts_behavior_overlay.py``）。

覆盖：
  1. 契约层（pydantic）校验：t_ms 非升序/负值/超单段时长上限/超关键帧数上限/
     空 keyframes/NaN 值 → ``ValidationError``（构造期即拒收，不进回放器）；
  2. ``feed()`` 业务性拒绝：未知 mode / 同段键集不一致 / 全部参数所连部署缺席 /
     部分参数缺席（只丢弃缺席者）/ 队列满 → ``[vtsb:invalid_params]`` 或
     ``[vtsb:throttled]``；
  3. 回放确定性：两帧线性插值中点值精确、首帧 t_ms>0 时从 0 起持首帧值；
  4. ``_Segment.sample()`` 端点夹取契约（白盒：``TrajectoryPlayer.apply()`` 在
     正常时序下因不变式保证不会触达此分支，此处直接对纯函数辅助验证）；
  5. attack：起播 strength 按 ``_ease`` 缓入，``ATTACK_S`` 后恒为 1；
  6. append 续接：新段起点严格衔接前段终点，跨段取样连续无跳变；
  7. replace（``append=False``）桥接：同 mode 时旧输出经 ``BRIDGE_S`` 缓变到新
     轨迹；异 mode 不桥接、直接跳变；
  8. release：播尽后 ``RELEASE_S`` 内 strength 1→0、值持末帧，之后 ``apply()``
     归 ``None``；``clear()`` 同语义且幂等；
  9. snapshot：在播 (True, 剩余 ms 含缓出)；空闲 (False, 0)。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agents.models.vts_behavior import (
    TRAJECTORY_MAX_KEYFRAMES,
    TRAJECTORY_MAX_QUEUE,
    TRAJECTORY_MAX_SEGMENT_MS,
    VTSB_INVALID_PARAMS,
    VTSB_THROTTLED,
    TrajectoryKeyframe,
    TrajectoryRequest,
)
from src.mcp.zero.sinks.trajectory import (
    ATTACK_S,
    BRIDGE_S,
    RELEASE_S,
    TrajectoryPlayer,
    _ease,
    _Segment,
)

# ---------------------------------------------------------------------------
# 1. 契约层（pydantic）校验
# ---------------------------------------------------------------------------


class TestKeyframeAndRequestValidation:
    """非法输入在构造期即拒收（`ValidationError`），不进入回放器。"""

    def test_t_ms_non_ascending_rejected(self) -> None:
        """t_ms 非升序（含相等/倒序）：`TrajectoryRequest` 的严格升序校验拒收。"""
        kf_a = TrajectoryKeyframe(t_ms=200, params={"X": 0.0})
        kf_b = TrajectoryKeyframe(t_ms=100, params={"X": 1.0})
        with pytest.raises(ValidationError):
            TrajectoryRequest(keyframes=[kf_a, kf_b])

    def test_negative_t_ms_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrajectoryKeyframe(t_ms=-1, params={"X": 0.0})

    def test_segment_duration_exceeds_max_rejected(self) -> None:
        kf_a = TrajectoryKeyframe(t_ms=0, params={"X": 0.0})
        kf_b = TrajectoryKeyframe(t_ms=TRAJECTORY_MAX_SEGMENT_MS + 1, params={"X": 1.0})
        with pytest.raises(ValidationError):
            TrajectoryRequest(keyframes=[kf_a, kf_b])

    def test_keyframes_exceed_max_count_rejected(self) -> None:
        keyframes = [
            TrajectoryKeyframe(t_ms=i, params={"X": 0.0})
            for i in range(TRAJECTORY_MAX_KEYFRAMES + 2)
        ]
        with pytest.raises(ValidationError):
            TrajectoryRequest(keyframes=keyframes)

    def test_empty_keyframes_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrajectoryRequest(keyframes=[])

    def test_nan_param_value_rejected(self) -> None:
        """NaN 幅度须显式 isfinite 判——比较式对 NaN 恒 False 会静默通过。"""
        with pytest.raises(ValidationError):
            TrajectoryKeyframe(t_ms=0, params={"X": float("nan")})


# ---------------------------------------------------------------------------
# 2. feed() 业务性拒绝（回放器自身语义校验，非 pydantic 层）
# ---------------------------------------------------------------------------


class TestFeedBusinessRejections:
    def test_unknown_mode_rejected(self) -> None:
        player = TrajectoryPlayer()
        result = player.feed(
            [(0.0, {"X": 0.0})], mode="bogus", append=True, now=0.0, known_params={"X"}
        )
        assert result.ok is False
        assert result.code == VTSB_INVALID_PARAMS

    def test_inconsistent_key_set_within_segment_rejected(self) -> None:
        player = TrajectoryPlayer()
        result = player.feed(
            [(0.0, {"X": 0.0, "Y": 0.0}), (1.0, {"X": 1.0})],
            mode="absolute",
            append=True,
            now=0.0,
            known_params={"X", "Y"},
        )
        assert result.ok is False
        assert result.code == VTSB_INVALID_PARAMS

    def test_all_params_missing_rejected_with_dropped(self) -> None:
        player = TrajectoryPlayer()
        result = player.feed(
            [(0.0, {"Z": 0.5})], mode="absolute", append=True, now=0.0, known_params={"X"}
        )
        assert result.ok is False
        assert result.code == VTSB_INVALID_PARAMS
        assert result.dropped_params == ["Z"]

    def test_partial_missing_params_dropped_only_absent_ones(self) -> None:
        player = TrajectoryPlayer()
        result = player.feed(
            [(0.0, {"X": 0.1, "Z": 0.9})],
            mode="absolute",
            append=True,
            now=0.0,
            known_params={"X"},
        )
        assert result.ok is True
        assert result.dropped_params == ["Z"]
        frame = player.apply(0.0)
        assert frame is not None
        assert set(frame.values) == {"X"}  # 缺席参数已被丢弃，其余照常回放

    def test_append_queue_full_rejected_throttled(self) -> None:
        player = TrajectoryPlayer()
        known = {"X"}
        first = player.feed(
            [(0.0, {"X": 0.0}), (10.0, {"X": 1.0})],
            mode="absolute",
            append=True,
            now=0.0,
            known_params=known,
        )
        assert first.ok
        assert first.queue_depth == 1  # 首次投喂无队可入，直接接管
        for i in range(TRAJECTORY_MAX_QUEUE - 1):  # 追加到队满（TRAJECTORY_MAX_QUEUE 段）
            result = player.feed(
                [(0.0, {"X": float(i)}), (1.0, {"X": float(i + 1)})],
                mode="absolute",
                append=True,
                now=0.1,
                known_params=known,
            )
            assert result.ok, f"第 {i} 次追加不应被拒"
        assert len(player.segments) == TRAJECTORY_MAX_QUEUE
        overflow = player.feed(
            [(0.0, {"X": 9.0}), (1.0, {"X": 10.0})],
            mode="absolute",
            append=True,
            now=0.1,
            known_params=known,
        )
        assert overflow.ok is False
        assert overflow.code == VTSB_THROTTLED
        assert overflow.queue_depth == TRAJECTORY_MAX_QUEUE


# ---------------------------------------------------------------------------
# 3. 回放确定性：线性插值 / 首帧持值
# ---------------------------------------------------------------------------


class TestPlaybackDeterminism:
    def test_linear_interpolation_midpoint_exact(self) -> None:
        player = TrajectoryPlayer()
        player.feed(
            [(0.0, {"X": 0.0}), (1.0, {"X": 10.0})],
            mode="absolute",
            append=True,
            now=100.0,
            known_params={"X"},
        )
        frame = player.apply(100.5)
        assert frame is not None
        assert frame.values["X"] == pytest.approx(5.0)

    def test_first_keyframe_after_zero_holds_value_from_zero(self) -> None:
        """首帧 t_ms>0：feed() 前置 (0, 首帧值)——不做起点外推猜测，纯持值。"""
        player = TrajectoryPlayer()
        player.feed(
            [(0.3, {"X": 2.0}), (1.0, {"X": 5.0})],
            mode="absolute",
            append=True,
            now=0.0,
            known_params={"X"},
        )
        assert player.apply(0.0).values["X"] == pytest.approx(2.0)
        assert player.apply(0.1).values["X"] == pytest.approx(2.0)
        assert player.apply(0.3).values["X"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 4. _Segment.sample() 端点夹取契约（白盒）
# ---------------------------------------------------------------------------


class TestSegmentSampleClamp:
    """越过端点夹取端点帧——`TrajectoryPlayer.apply()` 在正常时序下因
    `_active()`/队列不变式保证 `t_local` 恒落在 `[0, times[-1])` 内，不会触达
    这两条边界分支；直接对 `_Segment.sample()` 这一纯函数辅助做白盒验证，
    确保该契约不因未来重构悄悄退化为越界外推。"""

    def test_below_first_keyframe_clamps_to_first_frame(self) -> None:
        seg = _Segment(mode="absolute", times=[0.5, 1.5], frames=[{"X": 1.0}, {"X": 3.0}])
        assert seg.sample(-1.0) == {"X": 1.0}
        assert seg.sample(0.5) == {"X": 1.0}

    def test_beyond_last_keyframe_clamps_to_last_frame(self) -> None:
        seg = _Segment(mode="absolute", times=[0.0, 1.0], frames=[{"X": 1.0}, {"X": 3.0}])
        assert seg.sample(1.0) == {"X": 3.0}
        assert seg.sample(5.0) == {"X": 3.0}


# ---------------------------------------------------------------------------
# 5. attack：起播 takeover 强度爬升
# ---------------------------------------------------------------------------


class TestAttackRamp:
    def test_strength_follows_ease_curve_then_saturates(self) -> None:
        player = TrajectoryPlayer()
        player.feed(
            [(0.0, {"X": 0.0}), (1.0, {"X": 1.0})],
            mode="absolute",
            append=True,
            now=10.0,
            known_params={"X"},
        )
        assert player.apply(10.0).strength == pytest.approx(_ease(0.0))
        half_way = 10.0 + ATTACK_S / 2.0
        assert player.apply(half_way).strength == pytest.approx(_ease(0.5))
        after = 10.0 + ATTACK_S + 0.01
        frame = player.apply(after)
        assert frame.strength == pytest.approx(1.0)
        assert player.attack_from is None  # 达峰后复位，不再重复计算 ease


# ---------------------------------------------------------------------------
# 6. append 续接：跨段连续取样
# ---------------------------------------------------------------------------


class TestAppendContinuation:
    def test_second_segment_starts_at_first_segment_end_and_samples_continuously(self) -> None:
        player = TrajectoryPlayer()
        player.feed(
            [(0.0, {"X": 0.0}), (1.0, {"X": 1.0})],
            mode="absolute",
            append=True,
            now=0.0,
            known_params={"X"},
        )
        result = player.feed(
            [(0.0, {"X": 1.0}), (1.0, {"X": 2.0})],
            mode="absolute",
            append=True,
            now=0.5,
            known_params={"X"},
        )
        assert result.ok
        assert result.queue_depth == 2
        assert player.segments[1].start_s == pytest.approx(player.segments[0].end_s)
        assert player.segments[1].start_s == pytest.approx(1.0)
        # 跨段边界：段2 起点值 == 段1 终点值，无跳变
        assert player.apply(1.0).values["X"] == pytest.approx(1.0)
        # 段2 中点（局部 t=0.5）：1.0 与 2.0 间插值 = 1.5
        assert player.apply(1.5).values["X"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# 7. replace（append=False）桥接
# ---------------------------------------------------------------------------


class TestReplaceBridging:
    def test_same_mode_bridges_old_output_into_new_trajectory(self) -> None:
        player = TrajectoryPlayer()
        player.feed(
            [(0.0, {"X": 0.0}), (1.0, {"X": 1.0})],
            mode="absolute",
            append=True,
            now=0.0,
            known_params={"X"},
        )
        now_replace = 0.5
        old_value = player.apply(now_replace).values["X"]
        assert old_value == pytest.approx(0.5)
        result = player.feed(
            [(0.0, {"X": 5.0}), (1.0, {"X": 6.0})],
            mode="absolute",
            append=False,
            now=now_replace,
            known_params={"X"},
        )
        assert result.ok
        assert player.bridge_values == pytest.approx({"X": old_value})
        # 桥接窗口起点（w=0）：值仍等于旧输出，未跳变
        assert player.apply(now_replace).values["X"] == pytest.approx(old_value)
        # 桥接窗口末尾（w=1）：值已完全过渡到新轨迹采样值，bridge 状态清空
        after_bridge = now_replace + BRIDGE_S + 0.01
        local_t = after_bridge - now_replace
        expected = 5.0 + local_t * (6.0 - 5.0)
        assert player.apply(after_bridge).values["X"] == pytest.approx(expected)
        assert player.bridge_values is None

    def test_different_mode_skips_bridge_and_jumps_immediately(self) -> None:
        """异 mode 不桥接——`current.mode == mode` 判据不成立，直接跳变到新段值。"""
        player = TrajectoryPlayer()
        player.feed(
            [(0.0, {"X": 0.0}), (1.0, {"X": 1.0})],
            mode="absolute",
            append=True,
            now=0.0,
            known_params={"X"},
        )
        now_replace = 0.5
        result = player.feed(
            [(0.0, {"X": 10.0}), (1.0, {"X": 20.0})],
            mode="offset",
            append=False,
            now=now_replace,
            known_params={"X"},
        )
        assert result.ok
        assert player.bridge_values is None
        frame = player.apply(now_replace)
        assert frame.mode == "offset"
        assert frame.values["X"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 8. release / clear
# ---------------------------------------------------------------------------


class TestRelease:
    def test_release_holds_last_frame_and_decays_strength_to_none(self) -> None:
        player = TrajectoryPlayer()
        player.feed(
            [(0.0, {"X": 0.0}), (0.4, {"X": 1.0})],
            mode="absolute",
            append=True,
            now=0.0,
            known_params={"X"},
        )
        start = player.apply(0.4)  # 恰好播尽：转入交还缓出，u=0
        assert start is not None
        assert start.strength == pytest.approx(1.0)
        assert start.values["X"] == pytest.approx(1.0)
        mid = player.apply(0.4 + RELEASE_S / 2.0)
        assert mid is not None
        assert 0.0 < mid.strength < 1.0
        assert mid.values["X"] == pytest.approx(1.0)  # 值恒持末帧，只有 strength 缓出
        after = player.apply(0.4 + RELEASE_S + 0.01)
        assert after is None


class TestClear:
    """clear() 与自然播尽同一交还语义：立即捕获当前输出值进入 RELEASE_S 缓出；
    幂等——已空闲时再次 clear() 不产生新的缓出窗口。"""

    def test_clear_mid_playback_holds_current_value_and_decays(self) -> None:
        player = TrajectoryPlayer()
        player.feed(
            [(0.0, {"X": 0.0}), (1.0, {"X": 1.0})],
            mode="absolute",
            append=True,
            now=0.0,
            known_params={"X"},
        )
        now_clear = 0.3
        before = player.apply(now_clear)
        assert before is not None
        player.clear(now_clear)
        assert player.segments == []
        assert player.timeline_start is None
        after = player.apply(now_clear)
        assert after is not None
        assert after.values["X"] == pytest.approx(before.values["X"])
        assert after.strength == pytest.approx(1.0)  # 刚清除：u=0
        assert player.apply(now_clear + RELEASE_S + 0.01) is None

    def test_clear_when_already_idle_is_noop(self) -> None:
        player = TrajectoryPlayer()
        player.feed(
            [(0.0, {"X": 0.0}), (0.2, {"X": 1.0})],
            mode="absolute",
            append=True,
            now=0.0,
            known_params={"X"},
        )
        player.clear(0.1)
        exhausted_at = 0.1 + RELEASE_S + 0.01
        assert player.apply(exhausted_at) is None  # 缓出已耗尽，彻底空闲
        player.clear(exhausted_at + 1.0)  # 空闲态再次 clear：幂等 no-op
        assert player.apply(exhausted_at + 1.0) is None


# ---------------------------------------------------------------------------
# 9. snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_idle_before_any_feed(self) -> None:
        player = TrajectoryPlayer()
        assert player.snapshot(0.0) == (False, 0)

    def test_active_reports_remaining_including_release_tail(self) -> None:
        player = TrajectoryPlayer()
        player.feed(
            [(0.0, {"X": 0.0}), (1.0, {"X": 1.0})],
            mode="absolute",
            append=True,
            now=0.0,
            known_params={"X"},
        )
        active, remaining = player.snapshot(0.5)
        assert active is True
        expected = int(round((1.0 + RELEASE_S - 0.5) * 1000.0))
        assert remaining == pytest.approx(expected, abs=1)

    def test_snapshot_without_prior_apply_reports_idle_at_exact_exhaustion(self) -> None:
        """已知细节（非缺陷断言）：队列播尽但从未调用过 `apply()` 驱动「转入
        交还缓出」的状态迁移（该迁移只在 `apply()` 内发生）时，`snapshot()`
        如实报告空闲——不代表 RELEASE_S 缓出未发生，只是本实例还未被驱动
        感知到。生产链路 `_render_loop` 每 tick 都调 `apply()`，该窗口通常
        < 一帧（render_hz 量级）；此处仅记录实测行为。"""
        player = TrajectoryPlayer()
        player.feed(
            [(0.0, {"X": 0.0}), (0.2, {"X": 1.0})],
            mode="absolute",
            append=True,
            now=0.0,
            known_params={"X"},
        )
        assert player.snapshot(0.25) == (False, 0)

    def test_idle_after_release_completes(self) -> None:
        player = TrajectoryPlayer()
        player.feed(
            [(0.0, {"X": 0.0}), (0.2, {"X": 1.0})],
            mode="absolute",
            append=True,
            now=0.0,
            known_params={"X"},
        )
        player.apply(0.2)  # 驱动转入缓出
        player.apply(0.2 + RELEASE_S + 0.01)  # 驱动缓出耗尽
        assert player.snapshot(0.2 + RELEASE_S + 0.02) == (False, 0)
