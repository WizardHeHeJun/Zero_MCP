"""语音播放引擎测试（speech-play 蓝图 2026-08-14 §T1/§T3）。

被测：
  - ``src/agents/models/vts_behavior.py`` 的 ``SpeechRequest``/``SpeechReceipt``
    契约（T1 面：mouth_track 校验复用 ``TrajectoryRequest`` validator、fps 校验、
    回执恰好两字段的字面形状）；
  - ``src/mcp/behavior/speech_playback.py`` 的 ``read_wav_meta``（wav 头表驱动）、
    ``AudioPlayer``（play 可注入，异常归一）、``SpeechQueue``（FIFO/锚点传导/
    线程边界/失败面/aclose，全 fake player，不触真声卡）。

不标 ``zerorepo``（无关 D:\\Zero）；零 sounddevice 依赖（`AudioPlayer(play=...)`
注入缝，见 speech_playback 模块 docstring）。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import ValidationError

from src.agents.models.vts_behavior import (
    TRAJECTORY_MAX_KEYFRAMES,
    TRAJECTORY_MAX_SEGMENT_MS,
    VTSB_SPEECH_DEVICE_ERROR,
    VTSB_SPEECH_FILE_ERROR,
    VTSB_SPEECH_FORMAT_ERROR,
    VTSB_THROTTLED,
    SpeechReceipt,
    SpeechRequest,
    extract_vtsb_code,
)
from src.mcp.behavior.speech_playback import (
    AudioPlayer,
    SpeechJob,
    SpeechQueue,
    read_wav_meta,
)
from src.mcp.zero.sinks.trajectory import FeedResult

# ---------------------------------------------------------------------------
# T1 — 契约：SpeechRequest / SpeechReceipt
# ---------------------------------------------------------------------------


def _kf(t_ms: int, value: float = 0.5) -> dict[str, Any]:
    return {"t_ms": t_ms, "params": {"MouthOpen": value}}


class TestSpeechRequestContract:
    def test_valid_request_passes(self) -> None:
        req = SpeechRequest(wav_path="/abs/path.wav", mouth_track=[_kf(0), _kf(100)])
        assert req.fps == 20.0
        assert req.wav_path == "/abs/path.wav"

    def test_empty_mouth_track_raises(self) -> None:
        with pytest.raises(ValidationError):
            SpeechRequest(wav_path="/abs/path.wav", mouth_track=[])

    def test_non_ascending_t_ms_raises(self) -> None:
        with pytest.raises(ValidationError):
            SpeechRequest(wav_path="/abs/path.wav", mouth_track=[_kf(100), _kf(50)])

    def test_equal_t_ms_raises(self) -> None:
        """严格升序——相等也不行（继承 TrajectoryRequest 的 ``<=`` 判据）。"""
        with pytest.raises(ValidationError):
            SpeechRequest(wav_path="/abs/path.wav", mouth_track=[_kf(0), _kf(0)])

    def test_over_max_keyframes_raises(self) -> None:
        track = [_kf(i) for i in range(TRAJECTORY_MAX_KEYFRAMES + 1)]
        with pytest.raises(ValidationError):
            SpeechRequest(wav_path="/abs/path.wav", mouth_track=track)

    def test_at_max_keyframes_passes(self) -> None:
        track = [_kf(i) for i in range(TRAJECTORY_MAX_KEYFRAMES)]
        req = SpeechRequest(wav_path="/abs/path.wav", mouth_track=track)
        assert len(req.mouth_track) == TRAJECTORY_MAX_KEYFRAMES

    def test_over_max_duration_raises(self) -> None:
        track = [_kf(0), _kf(TRAJECTORY_MAX_SEGMENT_MS + 1)]
        with pytest.raises(ValidationError):
            SpeechRequest(wav_path="/abs/path.wav", mouth_track=track)

    @pytest.mark.parametrize("fps", [0.0, -1.0, float("inf"), float("nan")])
    def test_invalid_fps_raises(self, fps: float) -> None:
        with pytest.raises(ValidationError):
            SpeechRequest(wav_path="/abs/path.wav", mouth_track=[_kf(0)], fps=fps)

    def test_empty_wav_path_raises(self) -> None:
        with pytest.raises(ValidationError):
            SpeechRequest(wav_path="", mouth_track=[_kf(0)])


class TestSpeechReceiptShape:
    def test_serializes_to_exact_two_keys(self) -> None:
        """AD-3：恰好 ``{accepted, duration_ms}`` 两字段——Zero 已锁定的字面形状。"""
        receipt = SpeechReceipt(accepted=True, duration_ms=3210.0)
        payload = json.loads(receipt.model_dump_json())
        assert payload == {"accepted": True, "duration_ms": 3210.0}
        assert set(payload.keys()) == {"accepted", "duration_ms"}


# ---------------------------------------------------------------------------
# T3 — read_wav_meta：wav 头表驱动
# ---------------------------------------------------------------------------


def _write_wav(
    path: Path,
    *,
    channels: int = 1,
    sampwidth: int = 2,
    framerate: int = 44100,
    n_frames: int = 100,
) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(b"\x00" * n_frames * channels * sampwidth)


class TestReadWavMeta:
    def test_valid_pcm16_mono_44100_returns_correct_duration(self, tmp_path: Path) -> None:
        path = tmp_path / "ok.wav"
        n_frames = 44100  # 恰好 1 秒
        _write_wav(path, channels=1, sampwidth=2, framerate=44100, n_frames=n_frames)
        frames, duration_ms = read_wav_meta(str(path))
        assert len(frames) == n_frames * 2  # sampwidth=2 字节/帧
        assert duration_ms == pytest.approx(1000.0)

    @pytest.mark.parametrize(
        ("channels", "sampwidth", "framerate", "label"),
        [
            (2, 2, 44100, "stereo"),
            (1, 1, 44100, "8bit"),
            (1, 2, 48000, "48kHz"),
        ],
    )
    def test_format_mismatch_raises_format_error(
        self, tmp_path: Path, channels: int, sampwidth: int, framerate: int, label: str
    ) -> None:
        path = tmp_path / f"bad_{label}.wav"
        _write_wav(path, channels=channels, sampwidth=sampwidth, framerate=framerate, n_frames=100)
        with pytest.raises(ToolError) as excinfo:
            read_wav_meta(str(path))
        assert extract_vtsb_code(str(excinfo.value)) == VTSB_SPEECH_FORMAT_ERROR

    def test_nonexistent_absolute_path_raises_file_error(self, tmp_path: Path) -> None:
        path = tmp_path / "nope.wav"
        with pytest.raises(ToolError) as excinfo:
            read_wav_meta(str(path))
        assert extract_vtsb_code(str(excinfo.value)) == VTSB_SPEECH_FILE_ERROR

    def test_relative_path_raises_file_error(self) -> None:
        with pytest.raises(ToolError) as excinfo:
            read_wav_meta("relative/path.wav")
        assert extract_vtsb_code(str(excinfo.value)) == VTSB_SPEECH_FILE_ERROR

    def test_corrupted_file_raises_format_error(self, tmp_path: Path) -> None:
        """随便写 bytes（不构成 RIFF 头）——现场核验实际走 ``wave.Error`` 分支，
        按实现语义映射 FORMAT_ERROR（非 FILE_ERROR，见 read_wav_meta docstring）。"""
        path = tmp_path / "corrupt.wav"
        path.write_bytes(b"not a real wav file, just garbage bytes 1234567890")
        with pytest.raises(ToolError) as excinfo:
            read_wav_meta(str(path))
        assert extract_vtsb_code(str(excinfo.value)) == VTSB_SPEECH_FORMAT_ERROR


# ---------------------------------------------------------------------------
# AudioPlayer：异常归一
# ---------------------------------------------------------------------------


class TestAudioPlayer:
    def test_tool_error_from_play_passes_through_unchanged(self) -> None:
        original = ToolError(f"{VTSB_SPEECH_DEVICE_ERROR} 设备探测失败")

        def _play(frames: bytes, on_anchor: Callable[[float], None]) -> None:
            raise original

        player = AudioPlayer(play=_play)
        with pytest.raises(ToolError) as excinfo:
            player.play_blocking(b"\x00", lambda _t0: None)
        assert excinfo.value is original  # 原样透传，不重复包壳

    def test_arbitrary_exception_normalized_to_device_error(self) -> None:
        def _play(frames: bytes, on_anchor: Callable[[float], None]) -> None:
            raise RuntimeError("设备驱动崩了")

        player = AudioPlayer(play=_play)
        with pytest.raises(ToolError) as excinfo:
            player.play_blocking(b"\x00", lambda _t0: None)
        assert extract_vtsb_code(str(excinfo.value)) == VTSB_SPEECH_DEVICE_ERROR
        assert "设备驱动崩了" in str(excinfo.value)


# ---------------------------------------------------------------------------
# SpeechQueue：全 fake player
# ---------------------------------------------------------------------------


@dataclass
class FeedCall:
    keyframes: list[tuple[float, dict[str, float]]]
    mode: str
    append: bool
    now: float
    known_params: frozenset[str]


class FakeSpeechMouth:
    """记录调用的假 ``TrajectoryPlayer``——不做任何真实回放数学，只捕获转发。"""

    def __init__(self) -> None:
        self.feed_calls: list[FeedCall] = []
        self.clear_calls: list[float] = []
        self.feed_thread_ids: list[int] = []

    def feed(
        self,
        keyframes: list[tuple[float, dict[str, float]]],
        *,
        mode: str,
        append: bool,
        now: float,
        known_params: frozenset[str],
    ) -> FeedResult:
        self.feed_thread_ids.append(threading.get_ident())
        self.feed_calls.append(FeedCall(keyframes, mode, append, now, known_params))
        return FeedResult(ok=True, duration_ms=100, queue_depth=1)

    def clear(self, now: float) -> None:
        self.clear_calls.append(now)


class RejectingSpeechMouth(FakeSpeechMouth):
    """feed 返回业务性拒绝的假嘴部通道——验证拒绝被留痕而非静默吞掉（审查 WARN-1）。"""

    def feed(self, *args: Any, **kwargs: Any) -> FeedResult:
        super().feed(*args, **kwargs)
        return FeedResult(
            ok=False,
            code="[vtsb:invalid_params]",
            detail="同段关键帧参数键集不一致（契约要求稠密统一帧）",
            queue_depth=0,
        )


def _make_job(
    *,
    frames: bytes = b"\x00" * 10,
    duration_ms: float = 100.0,
    mouth_keyframes: list[tuple[float, dict[str, float]]] | None = None,
    known_params: frozenset[str] | None = None,
    fps: float = 20.0,
) -> SpeechJob:
    return SpeechJob(
        frames=frames,
        duration_ms=duration_ms,
        mouth_keyframes=mouth_keyframes or [(0.0, {"MouthOpen": 0.5})],
        known_params=known_params or frozenset({"MouthOpen"}),
        fps=fps,
    )


class AnchorPlayer:
    """立即回传固定锚点、不阻塞——用于验证锚点传导形状。"""

    def __init__(self, t0: float) -> None:
        self.t0 = t0

    def play(self, frames: bytes, on_anchor: Callable[[float], None]) -> None:
        on_anchor(self.t0)


class ThreadRecordingPlayer:
    """记录 ``play()`` 自身执行时的线程 id——用于对照 feed() 是否在另一线程。"""

    def __init__(self) -> None:
        self.play_thread_id: int | None = None

    def play(self, frames: bytes, on_anchor: Callable[[float], None]) -> None:
        self.play_thread_id = threading.get_ident()
        on_anchor(time.monotonic())


@dataclass
class OrderedPlayer:
    """记录 play() 开始顺序（按 frames 内容区分 job），阻塞直到测试显式释放
    （``threading.Event``，一次 set 后对全部后续 job 均视为"已释放"——够用，
    本测试只关心谁先开始播放，不需要每 job 独立控制）。"""

    order: list[bytes] = field(default_factory=list)
    gate: threading.Event = field(default_factory=threading.Event)

    def play(self, frames: bytes, on_anchor: Callable[[float], None]) -> None:
        self.order.append(frames)
        on_anchor(time.monotonic())
        self.gate.wait(timeout=5.0)


class RaisingBeforeAnchorPlayer:
    """播放线程在报锚点**之前**就失败——命中 ``_propagate_early_failure``。"""

    def play(self, frames: bytes, on_anchor: Callable[[float], None]) -> None:
        raise RuntimeError("boom-before-anchor")


class RaisingAfterAnchorPlayer:
    """播放线程报锚点（feed 已被调用）**之后**才失败——命中正常路径的
    ``await play_task`` 异常分支。"""

    def play(self, frames: bytes, on_anchor: Callable[[float], None]) -> None:
        on_anchor(time.monotonic())
        raise RuntimeError("boom-mid-playback")


async def _drain(queue: SpeechQueue) -> None:
    """等待队列内已入队的全部 job 处理完成（含 feed/clear 的事件循环线程回调）。"""
    await queue.queue.join()
    await asyncio.sleep(0)  # 放行一次事件循环，确保 finally 块内的状态更新落地


class TestSpeechQueueAnchorPropagation:
    async def test_anchor_reaches_speech_mouth_feed_as_absolute_no_append(self) -> None:
        mouth = FakeSpeechMouth()
        anchor_t0 = 123.456
        sq = SpeechQueue(mouth, player=AudioPlayer(play=AnchorPlayer(anchor_t0).play))
        try:
            await sq.enqueue(_make_job())
            await _drain(sq)
            assert len(mouth.feed_calls) == 1
            call = mouth.feed_calls[0]
            assert call.now == anchor_t0
            assert call.mode == "absolute"
            assert call.append is False
        finally:
            await sq.aclose()


class TestSpeechQueueThreadBoundary:
    async def test_feed_called_on_event_loop_thread_not_player_thread(self) -> None:
        """AD-7 线程边界不变式：``play()`` 在独立线程执行（``asyncio.to_thread``
        派发），但 ``feed()`` 的实际调用必须回到事件循环线程——记录各自执行时的
        线程 id 对照，不得相同。"""
        mouth = FakeSpeechMouth()
        player_impl = ThreadRecordingPlayer()
        sq = SpeechQueue(mouth, player=AudioPlayer(play=player_impl.play))
        loop_thread_id = threading.get_ident()
        try:
            await sq.enqueue(_make_job())
            await _drain(sq)
            assert player_impl.play_thread_id is not None
            assert player_impl.play_thread_id != loop_thread_id, (
                "play() 应在独立线程执行（asyncio.to_thread 派发）"
            )
            assert mouth.feed_thread_ids == [loop_thread_id], (
                "feed() 必须只在事件循环线程被调用（AD-7）"
            )
        finally:
            await sq.aclose()


class TestSpeechQueueFifoOrdering:
    async def test_second_job_not_dequeued_before_first_playback_thread_returns(self) -> None:
        mouth = FakeSpeechMouth()
        player_impl = OrderedPlayer()
        sq = SpeechQueue(mouth, player=AudioPlayer(play=player_impl.play), maxsize=5)
        try:
            await sq.enqueue(_make_job(frames=b"job1"))
            for _ in range(200):  # 轮询等待 job1 的 play() 真正开始（进入阻塞）
                if player_impl.order:
                    break
                await asyncio.sleep(0.01)
            assert player_impl.order == [b"job1"]

            await sq.enqueue(_make_job(frames=b"job2"))
            await asyncio.sleep(0.05)  # 给 worker 一个"若会误抢"就会露馅的窗口
            assert player_impl.order == [b"job1"], "job1 播放线程未返回时 job2 不得被 dequeue"
            assert sq.queue.qsize() == 1, "job2 应仍在队列里等待，不是被 worker 取走"

            player_impl.gate.set()  # 释放 job1（Event 常驻 set，job2 的 wait 也会立即通过）
            for _ in range(200):
                if len(player_impl.order) == 2:
                    break
                await asyncio.sleep(0.01)
            assert player_impl.order == [b"job1", b"job2"]
        finally:
            player_impl.gate.set()
            await sq.aclose()


class TestSpeechQueueFull:
    async def test_enqueue_when_full_raises_throttled(self) -> None:
        mouth = FakeSpeechMouth()
        player_impl = OrderedPlayer()
        sq = SpeechQueue(mouth, player=AudioPlayer(play=player_impl.play), maxsize=1)
        try:
            await sq.enqueue(_make_job(frames=b"active"))  # 立即被 worker 取走、阻塞在播放线程
            for _ in range(200):
                if player_impl.order:
                    break
                await asyncio.sleep(0.01)
            assert player_impl.order == [b"active"]

            await sq.enqueue(_make_job(frames=b"waiting"))  # 填满 maxsize=1 的内部队列
            assert sq.queue.qsize() == 1

            with pytest.raises(ToolError) as excinfo:
                await sq.enqueue(_make_job(frames=b"overflow"))
            assert extract_vtsb_code(str(excinfo.value)) == VTSB_THROTTLED
        finally:
            player_impl.gate.set()
            await sq.aclose()


class TestSpeechQueueFailureIsolation:
    async def test_failure_before_anchor_clears_mouth_without_feed(self) -> None:
        mouth = FakeSpeechMouth()
        sq = SpeechQueue(mouth, player=AudioPlayer(play=RaisingBeforeAnchorPlayer().play))
        try:
            await sq.enqueue(_make_job())
            await _drain(sq)
            assert mouth.feed_calls == [], "起播锚点从未到达，不应调用 feed"
            assert mouth.clear_calls, "提前失败也应释放嘴部独占（防御性 clear）"
            assert sq.last_error is not None
            assert "boom-before-anchor" in sq.last_error
        finally:
            await sq.aclose()

    async def test_failure_after_anchor_clears_mouth_worker_continues(self) -> None:
        """④ 播放中途异常 → clear 被调、last_error 记录、worker 不崩、继续消化下一条。"""
        mouth = FakeSpeechMouth()
        sq = SpeechQueue(mouth, player=AudioPlayer(play=RaisingAfterAnchorPlayer().play))
        try:
            await sq.enqueue(_make_job(frames=b"job1"))
            await _drain(sq)
            assert len(mouth.feed_calls) == 1, "锚点已到达——feed 应被调用过"
            assert mouth.clear_calls, "播放中途异常应提前释放嘴部独占"
            assert sq.last_error is not None
            assert "boom-mid-playback" in sq.last_error

            # worker 未崩溃：继续消化下一条 job（同样会失败，但不应卡死/不再消费）
            await sq.enqueue(_make_job(frames=b"job2"))
            await _drain(sq)
            assert len(mouth.feed_calls) == 2
            assert len(mouth.clear_calls) == 2
        finally:
            await sq.aclose()


class TestSpeechQueueFeedRejectionObservability:
    async def test_feed_rejection_recorded_not_swallowed(self) -> None:
        """审查 WARN-1：口型侧业务性拒绝（键集不一致/参数全缺席）无处回传
        （回执已提前返回，AD-4），必须留痕到 last_error + warning 日志——
        否则口型静默丢失、验收线①不可诊断。音频不因口型拒绝中断（不 clear）。"""
        mouth = RejectingSpeechMouth()
        sq = SpeechQueue(mouth, player=AudioPlayer(play=AnchorPlayer(1.0).play))
        try:
            await sq.enqueue(_make_job())
            await _drain(sq)
            assert len(mouth.feed_calls) == 1
            assert sq.last_error is not None, "feed 拒绝必须留痕（AD-12 可观测性）"
            assert "[vtsb:invalid_params]" in sq.last_error
            assert not mouth.clear_calls, "口型拒绝不该中断音频（无异常路径，不 clear）"

            # worker 未受影响：继续消化下一条
            await sq.enqueue(_make_job())
            await _drain(sq)
            assert len(mouth.feed_calls) == 2
        finally:
            await sq.aclose()


class TestSpeechQueueAclose:
    async def test_aclose_idempotent_cancels_worker_releases_mouth(self) -> None:
        mouth = FakeSpeechMouth()
        sq = SpeechQueue(mouth, player=AudioPlayer(play=AnchorPlayer(1.0).play))
        await sq.aclose()
        assert sq.worker_task.done()
        assert mouth.clear_calls, "aclose 应释放嘴部独占"
        calls_before = len(mouth.clear_calls)
        await sq.aclose()  # 幂等：不抛
        assert len(mouth.clear_calls) >= calls_before

    async def test_aclose_drops_pending_queue_items(self) -> None:
        mouth = FakeSpeechMouth()
        player_impl = OrderedPlayer()
        sq = SpeechQueue(mouth, player=AudioPlayer(play=player_impl.play), maxsize=5)
        try:
            await sq.enqueue(_make_job(frames=b"active"))
            for _ in range(200):
                if player_impl.order:
                    break
                await asyncio.sleep(0.01)
            await sq.enqueue(_make_job(frames=b"never-played"))
            assert sq.queue.qsize() == 1
        finally:
            player_impl.gate.set()
            await sq.aclose()
        assert sq.queue.empty(), "aclose 应清空未播放的排队 job"
