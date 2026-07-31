"""裸参数轨迹回放器——Zero 侧动作模型 → VTS 参数级直驱（2026-07-31 二期）。

定位（用户拍板 2026-07-31）：自然度交给 Zero 侧的专业动作模型，本仓只做忠实的
参数通道——模型输出关键帧轨迹，经 MCP 分块投喂（``params_animate``），本回放器
在 ``VtsExpressionSink`` 渲染循环内按时间轴插值回放。与 12 词程序化通道
（``behavior_overlay``）正交共存：词表是模型未上线时的即用通道与降级兜底。

设计要点（对齐 behavior_overlay 的引擎纪律）：
- **纯同步、无 I/O、时间全由调用方注入**（``now`` 单调秒）——确定性可测；
- **takeover 强度渐变**：absolute 模式按 ``frame + (value-frame)×strength`` 混合，
  strength 起播 ``ATTACK_S`` 缓入、队列播尽/清除后 ``RELEASE_S`` 缓出——接管与
  交还都无跳变；offset 模式加性叠加同乘 strength；
- **流式续接**：``append=True`` 排到队列末尾无缝续播（动作模型分窗投喂）；
  ``append=False`` 清队即刻接管，旧值经 ``BRIDGE_S`` 桥接淡入新轨迹防跳变；
- **释放语义**：播尽后 strength 归零，键从输出消失——非 GOVERNED 参数停止注入，
  1s 后 VTS 自动收回控制权（同 AD-5 交还机制）。

契约层限值（段长/帧数/队列深）见 ``src.agents.models.vts_behavior``。
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

from src.agents.models.vts_behavior import (
    TRAJECTORY_MAX_QUEUE,
    TRAJECTORY_MODES,
    VTSB_INVALID_PARAMS,
    VTSB_THROTTLED,
)

ATTACK_S: float = 0.12
"""起播接管缓入时长（秒）——strength 0→1。"""

RELEASE_S: float = 0.25
"""播尽/清除后交还缓出时长（秒）——strength 1→0。"""

BRIDGE_S: float = 0.12
"""``append=False`` 即刻接管时，旧输出值向新轨迹的桥接淡入时长（秒）。"""


def _ease(u: float) -> float:
    """余弦缓动（与 behavior_overlay.cosine_ease01 同形，本模块自持避免环依赖面）。"""
    u = max(0.0, min(1.0, u))
    return 0.5 * (1.0 - math.cos(math.pi * u))


@dataclass(frozen=True, kw_only=True)
class TrajectoryFrame:
    """单帧回放输出：``values`` 参数值（按 mode 解释）、``strength`` 接管强度 [0,1]。"""

    values: dict[str, float]
    strength: float
    mode: str


@dataclass(kw_only=True)
class _Segment:
    """内部轨迹段：keyframes 为 (t_s, params) 且 t 严格升序、参数键集统一。"""

    mode: str
    times: list[float]
    frames: list[dict[str, float]]
    start_s: float = 0.0  # 在队列时间轴上的起点

    @property
    def duration_s(self) -> float:
        return self.times[-1]

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    def sample(self, t_local: float) -> dict[str, float]:
        """段内取样：帧间线性插值；越过端点夹取端点帧。"""
        if t_local <= self.times[0]:
            return dict(self.frames[0])
        if t_local >= self.times[-1]:
            return dict(self.frames[-1])
        idx = bisect.bisect_right(self.times, t_local)
        t0, t1 = self.times[idx - 1], self.times[idx]
        f0, f1 = self.frames[idx - 1], self.frames[idx]
        w = (t_local - t0) / (t1 - t0)
        return {k: f0[k] + (f1[k] - f0[k]) * w for k in f0}


@dataclass(kw_only=True)
class FeedResult:
    """feed() 的结构化结果（service 层折算成 TrajectoryReceipt）。"""

    ok: bool
    code: str | None = None
    detail: str | None = None
    dropped_params: list[str] = field(default_factory=list)
    duration_ms: int = 0
    queue_depth: int = 0


class TrajectoryPlayer:
    """轨迹段队列回放器（纯同步；时间由调用方注入）。"""

    def __init__(self) -> None:
        self.segments: list[_Segment] = []
        self.timeline_start: float | None = None
        self.attack_from: float | None = None
        self.release_from: float | None = None
        self.release_values: dict[str, float] = {}
        self.release_mode: str = "absolute"
        self.bridge_values: dict[str, float] | None = None
        self.bridge_from: float = 0.0

    # ── 投喂 / 清除 ─────────────────────────────────────────────────────────

    def feed(
        self,
        keyframes: list[tuple[float, dict[str, float]]],
        *,
        mode: str,
        append: bool,
        now: float,
        known_params: frozenset[str] | set[str],
    ) -> FeedResult:
        """投喂一段轨迹。业务性拒绝以 FeedResult(ok=False, code=...) 返回。

        丢弃所连部署不存在的参数（dropped_params 报告）；全部不存在才整段拒绝。
        键集必须统一（契约 docstring 约定，执行侧核验）。
        """
        if mode not in TRAJECTORY_MODES:
            return FeedResult(
                ok=False,
                code=VTSB_INVALID_PARAMS,
                detail=f"未知 mode {mode!r}，可选 {sorted(TRAJECTORY_MODES)}",
                queue_depth=len(self.segments),
            )
        key_set = set(keyframes[0][1])
        if any(set(params) != key_set for _, params in keyframes):
            return FeedResult(
                ok=False,
                code=VTSB_INVALID_PARAMS,
                detail="同段关键帧参数键集不一致（契约要求稠密统一帧）",
                queue_depth=len(self.segments),
            )
        dropped = sorted(key_set - set(known_params))
        kept = key_set - set(dropped)
        if not kept:
            return FeedResult(
                ok=False,
                code=VTSB_INVALID_PARAMS,
                detail=f"全部参数在所连部署缺席：{dropped}",
                dropped_params=dropped,
                queue_depth=len(self.segments),
            )
        if append and len(self.segments) >= TRAJECTORY_MAX_QUEUE:
            return FeedResult(
                ok=False,
                code=VTSB_THROTTLED,
                detail=f"轨迹队列已满（{TRAJECTORY_MAX_QUEUE}），请按 queue_depth 退避",
                queue_depth=len(self.segments),
            )

        times = [t for t, _ in keyframes]
        frames = [{k: params[k] for k in kept} for _, params in keyframes]
        if times[0] > 0.0:  # 首帧非 0：从 t=0 起持首帧值（不做起点外推猜测）
            times = [0.0, *times]
            frames = [dict(frames[0]), *frames]
        seg = _Segment(mode=mode, times=times, frames=frames)

        active = self._active(now)
        if append and active:
            seg.start_s = self.segments[-1].end_s
            self.segments.append(seg)
        else:
            if active:  # 即刻接管：桥接旧输出值防跳变
                current = self.apply(now)
                if current is not None and current.mode == mode:
                    self.bridge_values = dict(current.values)
                    self.bridge_from = now
            else:
                self.attack_from = now
                self.bridge_values = None
            self.segments = [seg]
            self.timeline_start = now
            seg.start_s = 0.0
            self.release_from = None
        return FeedResult(
            ok=True,
            dropped_params=dropped,
            duration_ms=int(round(seg.duration_s * 1000.0)),
            queue_depth=len(self.segments),
        )

    def clear(self, now: float) -> None:
        """清空队列并进入交还缓出（幂等；未在播时 no-op）。"""
        current = self.apply(now)
        self.segments = []
        self.timeline_start = None
        self.bridge_values = None
        if current is not None:
            self.release_from = now
            self.release_values = dict(current.values)
            self.release_mode = current.mode

    # ── 回放 ────────────────────────────────────────────────────────────────

    def apply(self, now: float) -> TrajectoryFrame | None:
        """当前时刻回放帧；空闲（含交还完成）返回 None。"""
        if self._active(now):
            assert self.timeline_start is not None
            t = now - self.timeline_start
            seg = next(s for s in self.segments if t < s.end_s or s is self.segments[-1])
            values = seg.sample(t - seg.start_s)
            if self.bridge_values is not None:
                w = _ease((now - self.bridge_from) / BRIDGE_S)
                values = {
                    k: self.bridge_values.get(k, v) + (v - self.bridge_values.get(k, v)) * w
                    for k, v in values.items()
                }
                if w >= 1.0:
                    self.bridge_values = None
            strength = 1.0
            if self.attack_from is not None:
                strength = _ease((now - self.attack_from) / ATTACK_S)
                if strength >= 1.0:
                    self.attack_from = None
            return TrajectoryFrame(values=values, strength=strength, mode=seg.mode)

        # 队列播尽：转入交还缓出
        if self.segments:
            assert self.timeline_start is not None
            last = self.segments[-1]
            self.release_from = self.timeline_start + last.end_s
            self.release_values = dict(last.frames[-1])
            self.release_mode = last.mode
            self.segments = []
            self.timeline_start = None
            self.bridge_values = None
        if self.release_from is not None:
            u = (now - self.release_from) / RELEASE_S
            if u < 1.0:
                return TrajectoryFrame(
                    values=dict(self.release_values),
                    strength=1.0 - _ease(u),
                    mode=self.release_mode,
                )
            self.release_from = None
            self.release_values = {}
        return None

    def snapshot(self, now: float) -> tuple[bool, int]:
        """(是否在播, 距播尽含交还缓出的剩余 ms)——BehaviorStatus 可观测性用。"""
        if self._active(now):
            assert self.timeline_start is not None
            end = self.timeline_start + self.segments[-1].end_s
            return True, max(0, int(round((end + RELEASE_S - now) * 1000.0)))
        if self.release_from is not None:
            return True, max(0, int(round((self.release_from + RELEASE_S - now) * 1000.0)))
        return False, 0

    def _active(self, now: float) -> bool:
        if not self.segments or self.timeline_start is None:
            return False
        return now - self.timeline_start < self.segments[-1].end_s
