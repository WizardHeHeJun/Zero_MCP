"""BehaviorOverlayEngine 引擎确定性单测（蓝图 2026-07-31 §8.1 · T7）。

纯同步、无 ws：时间全部由测试注入 ``now`` 序列驱动（AD-3 可测性约定）。引擎无随机
相位——``behavior_id`` 的 uuid 不参与任何数值路径，故无需 ``random.seed``。

覆盖：
  1. 包络确定性：hold 型 attack 峰值时刻 / sustain 保持 / release 收敛到 0 且键消失；
     stroke 型 repeat 周期数（nod 同号拍、shake 一去一回变号）；duration_ms 覆盖典型值。
  2. 仲裁矩阵（AD-6）：同通道低优先级 rejected ``[vtsb:channel_busy]`` / 高·同优先级
     replaced 且交叉淡化期两包络共存（同参数逐点加和）/ 异通道 MERGE 并行叠加。
  3. 冷却与节流：per-behavior 冷却剩余 ms、250ms 全局节流 ``[vtsb:throttled]``、
     rejected 不消耗节流窗口。
  4. 降级映射（AD-5）：ranges 无 BodyAngle → body 词借 FaceAngle 近似（约 1/3 幅度）
     + ``degraded_channels=["body"]``；无眼球参数 → glance 借 head 微偏。
  5. eye_gate 乘法语义：blink 走乘法门 0→1 包络（不进 offsets）；eyes_widen 走加法上推。
  6. 未知行为名与非法 direction 的 rejected 回执（机读码按符号名 pin）。
  7. interrupt：淡出回基准、未知通道拒绝、无活跃时幂等。
"""

from __future__ import annotations

import pytest

from src.agents.models.vts_behavior import (
    VTSB_CHANNEL_BUSY,
    VTSB_COOLDOWN,
    VTSB_INVALID_PARAMS,
    VTSB_THROTTLED,
    VTSB_UNKNOWN_BEHAVIOR,
    BehaviorReceipt,
    BehaviorRequest,
)
from src.mcp.zero.sinks.behavior_overlay import (
    ATTACK_FRACTION,
    CROSSFADE_S,
    DEGRADED_BODY_RATIO,
    GLANCE_EYE_SCALE,
    GLANCE_HEAD_SCALE,
    GLOBAL_THROTTLE_S,
    LEAN_BODY_SCALE,
    LEAN_HEAD_SCALE,
    NOD_SCALE,
    RELEASE_FRACTION,
    SHAKE_SCALE,
    SMILE_SCALE,
    BehaviorOverlayEngine,
    Ranges,
    adsr_envelope,
    cosine_ease01,
)

# ---------------------------------------------------------------------------
# 辅助构造
# ---------------------------------------------------------------------------

FULL_RANGES: Ranges = {
    "MouthSmile": (0.0, 1.0, 0.0),
    "MouthOpen": (0.0, 1.0, 0.0),
    "Brows": (0.0, 1.0, 0.0),
    "BrowLeftY": (0.0, 1.0, 0.0),
    "BrowRightY": (0.0, 1.0, 0.0),
    "EyeOpenLeft": (0.0, 1.0, 0.0),
    "EyeOpenRight": (0.0, 1.0, 0.0),
    "FaceAngleX": (-30.0, 30.0, 0.0),
    "FaceAngleY": (-30.0, 30.0, 0.0),
    "FaceAngleZ": (-90.0, 90.0, 0.0),
    "EyeLeftX": (-1.0, 1.0, 0.0),
    "EyeLeftY": (-1.0, 1.0, 0.0),
    "EyeRightX": (-1.0, 1.0, 0.0),
    "EyeRightY": (-1.0, 1.0, 0.0),
    "BodyAngleX": (-30.0, 30.0, 0.0),
    "BodyAngleY": (-30.0, 30.0, 0.0),
    "BodyAngleZ": (-30.0, 30.0, 0.0),
}
"""可选参数（BodyAngle*/眼球）全在场的部署——量程沿用实测 VTS 夹具形状。"""

_OPTIONAL_PARAMS = (
    "BodyAngleX",
    "BodyAngleY",
    "BodyAngleZ",
    "EyeLeftX",
    "EyeLeftY",
    "EyeRightX",
    "EyeRightY",
)

GOVERNED_RANGES: Ranges = {k: v for k, v in FULL_RANGES.items() if k not in _OPTIONAL_PARAMS}
"""仅标准治理参数在场的部署（无 BodyAngle、无眼球）——body/gaze 词走降级路径。"""


def _half(ranges: Ranges, param: str) -> float:
    """角度参数定标用半量程 ``(max-min)/2``（与引擎 `_resolve_tracks` 同式）。"""
    lo, hi, _ = ranges[param]
    return (hi - lo) / 2.0


def _trigger(
    engine: BehaviorOverlayEngine,
    name: str,
    now: float,
    *,
    ranges: Ranges = FULL_RANGES,
    **params: object,
) -> BehaviorReceipt:
    return engine.trigger(BehaviorRequest(name=name, **params), now=now, ranges=ranges)


# ---------------------------------------------------------------------------
# 1. 包络确定性 —— hold 型（ADSR + 余弦缓动）
# ---------------------------------------------------------------------------


class TestEnvelopeHold:
    def test_attack_ramps_to_peak_at_attack_end(self) -> None:
        """smile（hold 2000ms）：attack 段余弦缓入，attack 结束时刻恰达峰值。"""
        engine = BehaviorOverlayEngine()
        receipt = _trigger(engine, "smile", 0.0, intensity=1.0)
        assert receipt.status == "accepted"
        assert receipt.code is None
        assert receipt.estimated_duration_ms == 2000
        attack_s = 2.0 * ATTACK_FRACTION
        # attack 中点 = 峰值 × cosine_ease01(0.5)（半程幅度）
        mid = engine.apply(attack_s / 2.0).offsets["MouthSmile"]
        assert mid == pytest.approx(SMILE_SCALE * cosine_ease01(0.5))
        # attack 结束时刻 = 满峰值 intensity × SMILE_SCALE（[0,1] 参数直乘，不做量程定标）
        assert engine.apply(attack_s).offsets["MouthSmile"] == pytest.approx(SMILE_SCALE)

    def test_sustain_holds_peak(self) -> None:
        engine = BehaviorOverlayEngine()
        _trigger(engine, "smile", 0.0, intensity=1.0)
        # sustain 段 [attack, attack+sustain) = [0.5, 1.3)：任意时刻恒为峰值
        assert engine.apply(0.6).offsets["MouthSmile"] == pytest.approx(SMILE_SCALE)
        assert engine.apply(1.29).offsets["MouthSmile"] == pytest.approx(SMILE_SCALE)

    def test_release_converges_to_zero_and_key_disappears(self) -> None:
        """release 末端严格收敛到 0，包络结束后键从 offsets 消失（AD-5 交还语义）。"""
        engine = BehaviorOverlayEngine()
        _trigger(engine, "smile", 0.0, intensity=1.0)
        near_end = engine.apply(1.99).offsets["MouthSmile"]
        assert 0.0 < near_end < 0.01  # 余弦缓出，末端已近 0（无观感跳变）
        assert engine.apply(2.0).offsets == {}  # 到期即剪除，键消失 → sink 停发

    def test_duration_ms_overrides_typical_duration(self) -> None:
        """duration_ms 覆盖词表典型值：包络总长按覆盖值收缩，相位随之重定标。"""
        engine = BehaviorOverlayEngine()
        receipt = _trigger(engine, "smile", 0.0, intensity=1.0, duration_ms=1000)
        assert receipt.estimated_duration_ms == 1000
        # 覆盖后 attack = 0.25s，峰值时刻同步前移
        assert engine.apply(1.0 * ATTACK_FRACTION).offsets["MouthSmile"] == pytest.approx(
            SMILE_SCALE
        )
        assert "MouthSmile" in engine.apply(0.9).offsets  # 仍在 release 内
        assert engine.apply(1.0).offsets == {}  # 1s 即结束（典型值 2s 已被覆盖）

    def test_head_tilt_direction_flips_sign(self) -> None:
        """direction 决定 FaceAngleZ 偏移符号（left=负、right=正，角度参数乘半量程）。"""
        peak = 0.30 * _half(FULL_RANGES, "FaceAngleZ")  # HEAD_TILT_SCALE × 半量程
        engine = BehaviorOverlayEngine()
        _trigger(engine, "head_tilt", 0.0, intensity=1.0)  # 缺省 direction=left
        assert engine.apply(1.0).offsets["FaceAngleZ"] == pytest.approx(-peak)
        engine2 = BehaviorOverlayEngine()
        _trigger(engine2, "head_tilt", 0.0, intensity=1.0, direction="right")
        assert engine2.apply(1.0).offsets["FaceAngleZ"] == pytest.approx(peak)


# ---------------------------------------------------------------------------
# 1b. 包络确定性 —— stroke 型（正弦节律，repeat = 拍数）
# ---------------------------------------------------------------------------


class TestEnvelopeStroke:
    def test_nod_repeat_beats_peak_same_sign(self) -> None:
        """nod repeat=2：半周期正弦每拍 0→峰→0 同号（低头），峰值在每拍中点。"""
        peak = NOD_SCALE * _half(FULL_RANGES, "FaceAngleY")
        engine = BehaviorOverlayEngine()
        receipt = _trigger(engine, "nod", 0.0, intensity=1.0, repeat=2)
        assert receipt.estimated_duration_ms == 1400  # 700ms/拍 × 2
        # 第一拍中点 u=0.25 → 满幅低头（FaceAngleY 负向）
        assert engine.apply(0.35).offsets["FaceAngleY"] == pytest.approx(-peak)
        # 两拍交界 u=0.5 → 回位过零，但键不消失（防 sink 停发/复发抖动）
        boundary = engine.apply(0.70).offsets
        assert boundary["FaceAngleY"] == pytest.approx(0.0, abs=1e-9)
        # 第二拍中点 u=0.75 → 再次同号满幅（半周期正弦不变号）
        assert engine.apply(1.05).offsets["FaceAngleY"] == pytest.approx(-peak)
        # 总长到期键消失
        assert engine.apply(1.40).offsets == {}

    def test_shake_full_sine_alternates_sign(self) -> None:
        """shake：整周期正弦一拍内一去一回变号，端点为 0 天然无突跳。"""
        peak = SHAKE_SCALE * _half(FULL_RANGES, "FaceAngleX")
        engine = BehaviorOverlayEngine()
        _trigger(engine, "shake", 0.0, intensity=1.0, repeat=1)  # 600ms 一拍
        assert engine.apply(0.15).offsets["FaceAngleX"] == pytest.approx(peak)  # u=0.25
        assert engine.apply(0.45).offsets["FaceAngleX"] == pytest.approx(-peak)  # u=0.75
        assert engine.apply(0.60).offsets == {}

    def test_repeat_scales_stroke_duration(self) -> None:
        engine = BehaviorOverlayEngine()
        receipt = _trigger(engine, "nod", 0.0, intensity=1.0, repeat=3)
        assert receipt.estimated_duration_ms == 2100
        assert "FaceAngleY" in engine.apply(2.09).offsets
        assert engine.apply(2.10).offsets == {}


# ---------------------------------------------------------------------------
# 2. 仲裁矩阵（AD-6）
# ---------------------------------------------------------------------------


class TestArbitration:
    def test_same_channel_lower_priority_rejected_channel_busy(self) -> None:
        """brows 通道被 reactive 档 brow_raise 占用时，deliberate 档 brow_furrow 被拒。"""
        engine = BehaviorOverlayEngine()
        _trigger(engine, "brow_raise", 0.0)  # priority 3，活跃 900ms
        receipt = _trigger(engine, "brow_furrow", 0.3)  # priority 2，同通道
        assert receipt.status == "rejected"
        assert receipt.code == VTSB_CHANNEL_BUSY
        assert receipt.detail is not None and "brow_raise" in receipt.detail
        # 被拒行为未入场：活跃包络仍只有 brow_raise 一条
        assert [env.name for env in engine.envelopes] == ["brow_raise"]

    def test_same_channel_equal_priority_replaced_with_crossfade(self) -> None:
        """同优先级后到覆盖：replaced 回执 + 交叉淡化期新旧两包络共存。"""
        engine = BehaviorOverlayEngine()
        _trigger(engine, "nod", 0.0, intensity=1.0)  # head，700ms
        receipt = _trigger(engine, "head_tilt", 0.3, intensity=1.0)  # head，同 priority
        assert receipt.status == "replaced"
        assert receipt.detail is not None and "nod" in receipt.detail
        # 淡化窗口内（0.3–0.45）：nod 淡出中、head_tilt 淡入中，两包络同时输出
        mid = engine.apply(0.35).offsets
        assert mid["FaceAngleY"] != 0.0  # 旧包络（nod）仍在场
        assert mid["FaceAngleZ"] != 0.0  # 新包络（head_tilt）已入场
        # 淡出窗口结束后旧包络剪除，只剩新包络
        after = engine.apply(0.3 + CROSSFADE_S + 0.01).offsets
        assert "FaceAngleY" not in after
        assert "FaceAngleZ" in after

    def test_higher_priority_replaces_lower(self) -> None:
        engine = BehaviorOverlayEngine()
        _trigger(engine, "brow_furrow", 0.0)  # priority 2
        receipt = _trigger(engine, "brow_raise", 0.3)  # priority 3 → 抢占
        assert receipt.status == "replaced"
        assert receipt.detail is not None and "brow_furrow" in receipt.detail

    def test_crossfade_same_param_offsets_sum_both_envelopes(self) -> None:
        """交叉淡化期同参数逐点加和：lean_in 淡出 + lean_back 淡入共写 FaceAngleY
        （降级部署下两词同轨），offsets = 旧×淡出因子 + 新×淡入因子。"""
        engine = BehaviorOverlayEngine()
        _trigger(engine, "lean_in", 0.0, intensity=1.0, ranges=GOVERNED_RANGES)
        receipt = _trigger(engine, "lean_back", 0.5, intensity=1.0, ranges=GOVERNED_RANGES)
        assert receipt.status == "replaced"
        # 期望值用引擎同款包络原语独立重算（验证的是「线性加和」而非曲线形状）
        amp = LEAN_HEAD_SCALE * _half(GOVERNED_RANGES, "FaceAngleY")
        attack_s = 2.5 * ATTACK_FRACTION
        release_s = 2.5 * RELEASE_FRACTION
        sustain_s = 2.5 - attack_s - release_s
        dt = 0.05  # 淡化窗口内取样点 now=0.55
        old_part = (
            -amp
            * adsr_envelope(0.55, attack_s, sustain_s, release_s)
            * (1.0 - cosine_ease01(dt / CROSSFADE_S))
        )
        new_part = (
            amp
            * adsr_envelope(dt, attack_s, sustain_s, release_s)
            * cosine_ease01(dt / CROSSFADE_S)
        )
        frame = engine.apply(0.55)
        assert frame.offsets["FaceAngleY"] == pytest.approx(old_part + new_part)

    def test_different_channels_merge_coexist(self) -> None:
        """异通道 MERGE：head 与 mouth 并行叠加，第二次触发是 accepted 而非 replaced。"""
        engine = BehaviorOverlayEngine()
        _trigger(engine, "nod", 0.0, intensity=1.0)
        receipt = _trigger(engine, "smile", 0.3, intensity=1.0)
        assert receipt.status == "accepted"
        frame = engine.apply(0.4).offsets
        assert frame["FaceAngleY"] != 0.0
        assert frame["MouthSmile"] != 0.0


# ---------------------------------------------------------------------------
# 3. 冷却与全局节流
# ---------------------------------------------------------------------------


class TestCooldownAndThrottle:
    def test_per_behavior_cooldown_rejects_with_remaining_ms(self) -> None:
        """nod 冷却 1.5s：1.0s 时再触发 → rejected + 剩余 500ms 进 detail。"""
        engine = BehaviorOverlayEngine()
        _trigger(engine, "nod", 0.0)
        receipt = _trigger(engine, "nod", 1.0)
        assert receipt.status == "rejected"
        assert receipt.code == VTSB_COOLDOWN
        assert receipt.detail is not None and "500ms" in receipt.detail

    def test_cooldown_expires_then_accepts(self) -> None:
        engine = BehaviorOverlayEngine()
        _trigger(engine, "nod", 0.0)
        assert _trigger(engine, "nod", 1.6).status == "accepted"  # 1.6 > 冷却 1.5

    def test_global_throttle_rejects_within_window(self) -> None:
        """异通道、异行为也受 250ms 全局节流约束（防 LLM 侧连珠炮）。"""
        engine = BehaviorOverlayEngine()
        assert _trigger(engine, "nod", 0.0).status == "accepted"
        receipt = _trigger(engine, "smile", 0.1)
        assert receipt.status == "rejected"
        assert receipt.code == VTSB_THROTTLED

    def test_throttle_window_boundary_accepts(self) -> None:
        engine = BehaviorOverlayEngine()
        _trigger(engine, "nod", 0.0)
        # 恰在窗口边界（间隔 == GLOBAL_THROTTLE_S）→ 放行（判据为严格小于）
        assert _trigger(engine, "smile", GLOBAL_THROTTLE_S).status == "accepted"

    def test_rejection_does_not_reset_throttle_window(self) -> None:
        """被节流拒绝的触发不消耗节流窗口（只有 accepted 才更新时戳）。"""
        engine = BehaviorOverlayEngine()
        _trigger(engine, "nod", 0.0)
        assert _trigger(engine, "smile", 0.1).code == VTSB_THROTTLED
        # 距上次 accepted（0.0）已过 250ms——若 0.1 的拒绝重置了窗口，此处将被误拒
        assert _trigger(engine, "smile", 0.26).status == "accepted"


# ---------------------------------------------------------------------------
# 4. 降级映射（AD-5：触发时按 ranges 键集决定）
# ---------------------------------------------------------------------------


class TestDegradation:
    def test_lean_in_without_body_angle_borrows_head(self) -> None:
        """缺 BodyAngleY → lean_in 借 FaceAngleY 微量近似 + degraded_channels=['body']。"""
        engine = BehaviorOverlayEngine()
        receipt = _trigger(engine, "lean_in", 0.0, intensity=1.0, ranges=GOVERNED_RANGES)
        assert receipt.status == "accepted"
        assert receipt.degraded_channels == ["body"]
        assert receipt.channels == ["body", "head"]  # 降级后通道集含实际借用的 head
        frame = engine.apply(1.2).offsets  # sustain 段
        assert frame["FaceAngleY"] == pytest.approx(
            -LEAN_HEAD_SCALE * _half(GOVERNED_RANGES, "FaceAngleY")
        )
        assert "BodyAngleY" not in frame

    def test_lean_in_with_body_angle_uses_primary_plan(self) -> None:
        engine = BehaviorOverlayEngine()
        receipt = _trigger(engine, "lean_in", 0.0, intensity=1.0, ranges=FULL_RANGES)
        assert receipt.degraded_channels == []
        assert receipt.channels == ["body"]
        frame = engine.apply(1.2).offsets
        assert frame["BodyAngleY"] == pytest.approx(
            -LEAN_BODY_SCALE * _half(FULL_RANGES, "BodyAngleY")
        )
        assert "FaceAngleY" not in frame

    def test_degraded_amplitude_is_one_third_of_primary(self) -> None:
        """降级幅度 ≈ 主计划的 DEGRADED_BODY_RATIO（同半量程部署下逐值可比）。"""
        full = BehaviorOverlayEngine()
        _trigger(full, "lean_in", 0.0, intensity=1.0, ranges=FULL_RANGES)
        degraded = BehaviorOverlayEngine()
        _trigger(degraded, "lean_in", 0.0, intensity=1.0, ranges=GOVERNED_RANGES)
        primary = full.apply(1.2).offsets["BodyAngleY"]
        borrowed = degraded.apply(1.2).offsets["FaceAngleY"]
        assert borrowed == pytest.approx(primary * DEGRADED_BODY_RATIO)

    def test_glance_without_eye_params_borrows_head(self) -> None:
        """缺眼球参数 → glance 借 FaceAngleX 微偏 + degraded_channels=['gaze']。"""
        engine = BehaviorOverlayEngine()
        receipt = _trigger(
            engine, "glance", 0.0, intensity=1.0, direction="right", ranges=GOVERNED_RANGES
        )
        assert receipt.degraded_channels == ["gaze"]
        assert receipt.channels == ["gaze", "head"]
        frame = engine.apply(0.5).offsets  # sustain 段（1200ms，attack 0.3s）
        assert frame["FaceAngleX"] == pytest.approx(
            GLANCE_HEAD_SCALE * _half(GOVERNED_RANGES, "FaceAngleX")  # right=正向
        )
        assert "EyeLeftX" not in frame

    def test_glance_with_eye_params_moves_eyes_only(self) -> None:
        engine = BehaviorOverlayEngine()
        receipt = _trigger(
            engine, "glance", 0.0, intensity=1.0, direction="left", ranges=FULL_RANGES
        )
        assert receipt.degraded_channels == []
        assert receipt.channels == ["gaze"]
        frame = engine.apply(0.5).offsets
        eye_peak = -GLANCE_EYE_SCALE * _half(FULL_RANGES, "EyeLeftX")  # left=负向
        assert frame["EyeLeftX"] == pytest.approx(eye_peak)
        assert frame["EyeRightX"] == pytest.approx(eye_peak)
        # 方向过滤：left/right 只动 X 轴轨道；未降级不碰 head
        assert "EyeLeftY" not in frame
        assert "FaceAngleX" not in frame


# ---------------------------------------------------------------------------
# 5. eye_gate 乘法语义（blink 乘法门 vs eyes_widen 加法上推）
# ---------------------------------------------------------------------------


class TestEyeGate:
    def test_blink_gate_closes_and_reopens(self) -> None:
        """blink：门值 1→0→1 半周期正弦包络；gate 轨道不进 offsets。"""
        engine = BehaviorOverlayEngine()
        receipt = _trigger(engine, "blink", 0.0, intensity=1.0, repeat=1)  # 220ms 一拍
        assert receipt.status == "accepted"
        assert receipt.channels == ["eyelid"]
        assert engine.apply(0.0).eye_gate == pytest.approx(1.0)  # 起点未闭合
        quarter = engine.apply(0.055)  # u=0.25：闭合深度 sin(π/4)
        assert quarter.eye_gate == pytest.approx(1.0 - 0.5**0.5)
        assert quarter.offsets == {}  # 乘法门不走加性偏移
        assert engine.apply(0.11).eye_gate == pytest.approx(0.0)  # u=0.5 全闭
        assert engine.apply(0.22).eye_gate == pytest.approx(1.0)  # 包络结束，门回中性

    def test_blink_depth_scales_with_intensity(self) -> None:
        """intensity 即闭合深度：0.4 → 最深处门值 0.6（不全闭）。"""
        engine = BehaviorOverlayEngine()
        _trigger(engine, "blink", 0.0, intensity=0.4, repeat=1)
        assert engine.apply(0.11).eye_gate == pytest.approx(0.6)

    def test_blink_repeat_closes_per_beat(self) -> None:
        engine = BehaviorOverlayEngine()
        receipt = _trigger(engine, "blink", 0.0, intensity=1.0, repeat=2)
        assert receipt.estimated_duration_ms == 440
        assert engine.apply(0.11).eye_gate == pytest.approx(0.0)  # 第一拍最深
        assert engine.apply(0.22).eye_gate == pytest.approx(1.0)  # 两拍交界睁开
        assert engine.apply(0.33).eye_gate == pytest.approx(0.0)  # 第二拍最深

    def test_eyes_widen_is_additive_not_gate(self) -> None:
        """eyes_widen 的睁大走 EyeOpen 加法上推（AD-4），不动乘法门。"""
        engine = BehaviorOverlayEngine()
        _trigger(engine, "eyes_widen", 0.0, intensity=1.0)
        frame = engine.apply(0.3)  # sustain 段（800ms，attack 0.2s）
        assert frame.eye_gate == pytest.approx(1.0)
        assert frame.offsets["EyeOpenLeft"] == pytest.approx(0.50)  # EYES_WIDEN_EYE_SCALE
        assert frame.offsets["EyeOpenRight"] == pytest.approx(0.50)
        assert frame.offsets["BrowLeftY"] > 0.0  # 伴随微扬眉


# ---------------------------------------------------------------------------
# 6. 未知名与非法参数回执（AD-11：业务性拒绝走 code 字段，不抛异常）
# ---------------------------------------------------------------------------


class TestRejectionReceipts:
    def test_unknown_behavior_rejected(self) -> None:
        engine = BehaviorOverlayEngine()
        receipt = _trigger(engine, "moonwalk", 0.0)
        assert receipt.status == "rejected"
        assert receipt.code == VTSB_UNKNOWN_BEHAVIOR
        assert receipt.channels == []
        assert receipt.estimated_duration_ms == 0

    def test_direction_on_directionless_behavior_rejected(self) -> None:
        """nod 不接受 direction——传了即执行侧 invalid_params（契约层有意不管）。"""
        engine = BehaviorOverlayEngine()
        receipt = _trigger(engine, "nod", 0.0, direction="left")
        assert receipt.status == "rejected"
        assert receipt.code == VTSB_INVALID_PARAMS

    def test_illegal_direction_rejected_with_options(self) -> None:
        engine = BehaviorOverlayEngine()
        receipt = _trigger(engine, "glance", 0.0, direction="behind")
        assert receipt.status == "rejected"
        assert receipt.code == VTSB_INVALID_PARAMS
        assert receipt.detail is not None and "left" in receipt.detail  # 回执列出合法值集

    def test_rejection_consumes_neither_cooldown_nor_throttle(self) -> None:
        """rejected 不启动冷却、不占节流窗口——同一时刻紧随的合法触发照常接受。"""
        engine = BehaviorOverlayEngine()
        assert _trigger(engine, "moonwalk", 0.0).status == "rejected"
        assert _trigger(engine, "nod", 0.0).status == "accepted"


# ---------------------------------------------------------------------------
# 7. interrupt（AD-6 第 4 层：淡出回语义静息基准）
# ---------------------------------------------------------------------------


class TestInterrupt:
    def test_interrupt_all_fades_out_then_clears(self) -> None:
        engine = BehaviorOverlayEngine()
        _trigger(engine, "nod", 0.0, intensity=1.0)
        receipt = engine.interrupt(None, now=0.1)
        assert receipt.status == "accepted"
        assert receipt.channels == ["head"]
        assert receipt.detail is not None and "nod" in receipt.detail
        # 淡出窗口内仍有衰减中的偏移（非硬切）
        fading = engine.apply(0.15).offsets["FaceAngleY"]
        assert fading != 0.0
        assert abs(fading) < NOD_SCALE * _half(FULL_RANGES, "FaceAngleY")
        # 淡出结束（0.1 + CROSSFADE_S）后彻底归零、键消失
        assert engine.apply(0.1 + CROSSFADE_S).offsets == {}

    def test_interrupt_channel_only_clears_matching(self) -> None:
        engine = BehaviorOverlayEngine()
        _trigger(engine, "nod", 0.0, intensity=1.0)
        _trigger(engine, "smile", 0.3, intensity=1.0)
        receipt = engine.interrupt("mouth", now=0.4)
        assert receipt.channels == ["mouth"]
        after = engine.apply(0.4 + CROSSFADE_S).offsets
        assert "MouthSmile" not in after  # mouth 已清
        assert "FaceAngleY" in after  # head 不受影响

    def test_interrupt_unknown_channel_rejected(self) -> None:
        engine = BehaviorOverlayEngine()
        receipt = engine.interrupt("tail", now=0.0)
        assert receipt.status == "rejected"
        assert receipt.code == VTSB_INVALID_PARAMS

    def test_interrupt_idempotent_without_active(self) -> None:
        engine = BehaviorOverlayEngine()
        receipt = engine.interrupt(None, now=0.0)
        assert receipt.status == "accepted"
        assert receipt.channels == []
        assert receipt.estimated_duration_ms == 0
