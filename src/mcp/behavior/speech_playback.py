"""语音播放引擎——`speech_play` 的音频 I/O + 队列编排（speech-play 蓝图 2026-08-14 §T3）。

## 定位（AD-6/7/8 浓缩）

真实设备 I/O + 队列编排落在本文件（`src/mcp/behavior/` 业务层，不进 `sinks/`）：
`trajectory.py` 明确自我约束"纯同步、无 I/O"，把设备 I/O 塞进 `sinks/` 会破坏
这条既有契约的单一性。本模块与 `BehaviorService` 同层：业务编排（队列、时序、
失败面），不碰 VTS 协议本身。

播放选型 = `sounddevice.RawOutputStream`（bytes 直写）+ stdlib `wave` 解析头与
帧，全程**零 numpy 依赖**（AD-7）：`wave.readframes()` 直接返回 `bytes`，
`RawOutputStream.write(bytes)` 直接吃 bytes，不需要经 numpy 数组中转。

## 线程边界不变式（AD-7，审查重点）

跨线程边界是本模块**唯一真实的并发面**：真正做阻塞 I/O 的是独立 OS 线程
（经 `asyncio.to_thread` 派发）；该线程唯一被允许触碰共享可变状态的入口是
`loop.call_soon_threadsafe(...)`，用来把"流已起播的 monotonic 锚点"和
"提前失败"两类事件转交回事件循环线程处理。⚠ **`TrajectoryPlayer.feed()`/
`.clear()` 的调用只在事件循环线程发生**——`SpeechQueue._play_job` 里，
两次调用都在 `await` 恢复后的协程体内执行，播放线程本身（`AudioPlayer.play`
及其默认实现）永不直接触碰 `TrajectoryPlayer` 实例。

## numpy 死锁风险（2026-08-14 集成核验：经证据关闭，而非经预热机制关闭）

`sounddevice` 延迟 import 于 `_default_play` **函数体**内——本模块虽被
`vts_behavior_mcp_server.__main__` 的预热清单收录，但 `importlib.import_module`
不执行函数体，故预热**并未**真正 import 过 sounddevice（`-X importtime` 实测
确认）。死锁风险由三重独立证据关闭：① `sys.modules` diff 实测
`import sounddevice` 不传递 import numpy；② 最小复现 server 在事件循环内经
`to_thread` 首次触达 sounddevice 未复现死锁；③ `speech_play` 强制先开
`VTS_BEHAVIOR_ENABLED`，该分支预热 `src.mcp.behavior.service` 时已真正加载
numpy（死锁判据是"是否首次触达 numpy"，见 `src/mcp/native_warmup.py`）。
⚠ 若未来把某个**真正传递拉 numpy** 的依赖做成函数体内延迟 import，"预热
整模块"救不了它——必须让预热路径真正执行到那次 import。

## 时钟锚点（AD-8，工程论证非硬件实测）

`t0 = time.monotonic() + stream.latency`，在 `stream.start()` 后立即取值
（见 `_default_play`）。整句预合成、时长通常数秒：`stream.latency` 给出固定
偏移的最佳估计，剩余误差来源是设备时钟相对 `monotonic()` 的 ppm 级漂移，
远小于验收线 80ms 预算——这是工程论证，真机验收以 Zero 固定脚本 + 逐帧
录屏为准，不在本模块单测范围。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mcp.server.fastmcp.exceptions import ToolError

from src.agents.models.vts_behavior import (
    SPEECH_MAX_QUEUE,
    SPEECH_WAV_CHANNELS,
    SPEECH_WAV_SAMPLE_RATE,
    SPEECH_WAV_SAMPLE_WIDTH,
    VTSB_SPEECH_DEVICE_ERROR,
    VTSB_SPEECH_FILE_ERROR,
    VTSB_SPEECH_FORMAT_ERROR,
    VTSB_THROTTLED,
)
from src.mcp.zero.sinks.trajectory import TrajectoryPlayer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# wav 读取（阻塞 I/O——调用方须经 asyncio.to_thread 包）
# ---------------------------------------------------------------------------


def read_wav_meta(wav_path: str) -> tuple[bytes, float]:
    """读取 wav 文件帧数据与时长。**阻塞**——调用方须经 `asyncio.to_thread` 包
    （对齐仓内 io_adapters 惯例，enqueue 前读好，不在事件循环里直接跑）。

    校验 PCM16/mono/44100（规格常量见 `src.agents.models.vts_behavior`）：

    - 路径非绝对 / 不存在 / 不可读 → `ToolError(VTSB_SPEECH_FILE_ERROR)`；
    - 声道/位宽/采样率不符，或 `wave.Error`（文件头/数据损坏）→
      `ToolError(VTSB_SPEECH_FORMAT_ERROR)`。

    Returns:
        ``(frames, duration_ms)``：帧字节数据（`wave.readframes` 直出，可直接
        喂 `RawOutputStream.write`）与音频时长（毫秒）。
    """
    path = Path(wav_path)
    if not path.is_absolute():
        raise ToolError(f"{VTSB_SPEECH_FILE_ERROR} wav_path 须为同机绝对路径：{wav_path!r}")
    if not path.exists():
        raise ToolError(f"{VTSB_SPEECH_FILE_ERROR} wav 文件不存在：{wav_path}")
    if not os.access(path, os.R_OK):
        raise ToolError(f"{VTSB_SPEECH_FILE_ERROR} wav 文件不可读：{wav_path}")
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            if (
                channels != SPEECH_WAV_CHANNELS
                or sample_width != SPEECH_WAV_SAMPLE_WIDTH
                or sample_rate != SPEECH_WAV_SAMPLE_RATE
            ):
                raise ToolError(
                    f"{VTSB_SPEECH_FORMAT_ERROR} wav 规格不符（要求 "
                    f"{SPEECH_WAV_CHANNELS} 声道 / {SPEECH_WAV_SAMPLE_WIDTH * 8}-bit / "
                    f"{SPEECH_WAV_SAMPLE_RATE}Hz），实际 {channels} 声道 / "
                    f"{sample_width * 8}-bit / {sample_rate}Hz：{wav_path}"
                )
            frames = wf.readframes(n_frames)
    except wave.Error as exc:
        raise ToolError(
            f"{VTSB_SPEECH_FORMAT_ERROR} wav 文件损坏/非法：{wav_path}（{exc}）"
        ) from exc
    except OSError as exc:
        raise ToolError(f"{VTSB_SPEECH_FILE_ERROR} wav 文件读取失败：{wav_path}（{exc}）") from exc
    duration_ms = (n_frames / SPEECH_WAV_SAMPLE_RATE) * 1000.0
    return frames, duration_ms


# ---------------------------------------------------------------------------
# 音频设备播放（play 函数可注入——供纯逻辑测试用 fake player 替换真实设备）
# ---------------------------------------------------------------------------

PlayCallable = Callable[[bytes, Callable[[float], None]], None]
"""play 函数签名：``(frames, on_anchor) -> None``，**同步阻塞**执行，须由调用方
经 `asyncio.to_thread` 派发。``on_anchor`` 在真正起播那一刻被调用一次，携带
monotonic 时钟锚点；实现必须是线程安全的（默认实现内部走
`loop.call_soon_threadsafe`，由 `SpeechQueue` 构造并注入，见其 `_play_job`）。"""


def _default_play(frames: bytes, on_anchor: Callable[[float], None]) -> None:
    """默认播放实现：延迟 import `sounddevice`（AD-7 零 numpy 依赖判据不适用于
    本函数本身——sounddevice 是否传递性拉 numpy 未核验，见蓝图风险 7；本仓的
    应对不是避免它，而是把"整模块预热"覆盖到 mcp.run() 之前，见模块 docstring）。

    独立函数（不内联进 `AudioPlayer`）——与 `vts.py._connect` 同款风格：单测
    可替换的缝。设备探测失败（`check_output_settings`）与起播/写入失败统一
    包成 `ToolError(VTSB_SPEECH_DEVICE_ERROR)`，由 `AudioPlayer.play_blocking`
    的外层兜底捕获（本函数直接 raise 也会被那层原样透传，见其 docstring）。
    """
    import sounddevice as sd  # noqa: PLC0415  延迟 import：默认关的能力默认不装依赖

    try:
        sd.check_output_settings(
            samplerate=SPEECH_WAV_SAMPLE_RATE,
            channels=SPEECH_WAV_CHANNELS,
            dtype="int16",
        )
    except Exception as exc:
        raise ToolError(f"{VTSB_SPEECH_DEVICE_ERROR} 播放设备不可用：{exc}") from exc
    with sd.RawOutputStream(
        samplerate=SPEECH_WAV_SAMPLE_RATE,
        channels=SPEECH_WAV_CHANNELS,
        dtype="int16",
    ) as stream:
        # AD-8：起播（__enter__ 内部已 start()）后立即取锚点——stream.latency
        # 是设备报告的固定偏移最佳估计，见模块 docstring「时钟锚点」一节。
        t0 = time.monotonic() + stream.latency
        on_anchor(t0)
        stream.write(frames)


class AudioPlayer:
    """音频设备播放器：``play`` 函数可注入（默认 `_default_play`），供纯逻辑
    测试用 fake player 替换真实设备（不装 `sounddevice` 也能测）。"""

    def __init__(self, play: PlayCallable | None = None) -> None:
        self.play: PlayCallable = play or _default_play

    def play_blocking(self, frames: bytes, on_anchor: Callable[[float], None]) -> None:
        """执行一次阻塞播放（须由调用方经 `asyncio.to_thread` 派发）。

        任何非 `ToolError` 的底层异常统一归一为
        `ToolError(VTSB_SPEECH_DEVICE_ERROR)`——`_default_play` 自身的设备探测
        已抛 `ToolError`（原样透传，不重复包壳）；此处兜底 fake player 在测试里
        模拟的任意异常，统一按设备错误语义处理。
        """
        try:
            self.play(frames, on_anchor)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"{VTSB_SPEECH_DEVICE_ERROR} 播放失败：{exc}") from exc


# ---------------------------------------------------------------------------
# 播放任务 + 队列
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class SpeechJob:
    """一次 `speech_play` 调度的播放任务（内部纯数据，无 I/O）。

    ``mouth_keyframes`` 为 `TrajectoryPlayer.feed` 吃的 ``(t_s, params)`` 形态
    （由 service 层从 `SpeechRequest.mouth_track` 转换，对齐 `animate()` 的
    ``kf.t_ms / 1000.0`` 换算）；``known_params`` 同样由 service 在入队时按当前
    连接的 `sink.ranges`/`sink.all_params` 算好（对齐 `animate()` 的口径）——
    读取时机固定在入队那一刻，不在 worker 里回头再查一次 sink 状态。
    """

    frames: bytes
    duration_ms: float
    mouth_keyframes: list[tuple[float, dict[str, float]]]
    known_params: frozenset[str]
    fps: float = 20.0


class SpeechQueue:
    """语音播放队列：`asyncio.Queue` + 单一后台 worker task。

    - `enqueue()`：FIFO 入队；队满（`SPEECH_MAX_QUEUE`）→
      `ToolError(VTSB_THROTTLED)`（不阻塞等待，立即拒绝——我方自加防御，
      speech-play 蓝图风险 6）；
    - worker（`_run`/`_play_job`）：取 job → 经 `asyncio.to_thread` 起播放
      线程 → 收到起播锚点后**在事件循环线程**调
      `speech_mouth.feed(mode="absolute", append=False, now=t0, ...)` →
      等播放线程结束（自然播尽由 `TrajectoryPlayer._settle` 自迁移释放，
      不显式 clear）；播放异常/中途失败 → 显式 `speech_mouth.clear(now)`
      提前释放 + 记 `last_error`，worker **不崩溃**、继续下一条（AD-7 线程
      边界不变式：`feed`/`clear` 只在事件循环线程调用）。
    """

    def __init__(
        self,
        speech_mouth: TrajectoryPlayer,
        *,
        player: AudioPlayer | None = None,
        maxsize: int = SPEECH_MAX_QUEUE,
    ) -> None:
        self.speech_mouth = speech_mouth
        self.player = player or AudioPlayer()
        self.queue: asyncio.Queue[SpeechJob] = asyncio.Queue(maxsize=maxsize)
        self.active = False
        self.last_error: str | None = None
        # python-code.md：不裸 asyncio.create_task 丢句柄——持有引用供 aclose()
        # 取消/等待（对齐 vts.py.render_task 惯例）。
        self.worker_task: asyncio.Task[None] = asyncio.create_task(self._run())

    async def enqueue(self, job: SpeechJob) -> None:
        """FIFO 入队；满 → `ToolError(VTSB_THROTTLED)`。"""
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise ToolError(
                f"{VTSB_THROTTLED} 语音播放队列已满（maxsize={self.queue.maxsize}），"
                "请按上一次 duration_ms 节流后重试"
            ) from exc

    def snapshot(self) -> tuple[bool, int, str | None]:
        """``(是否在播, 队列深度含在播, 最后一次失败信息)``——`BehaviorStatus` 聚合用。"""
        depth = self.queue.qsize() + (1 if self.active else 0)
        return self.active, depth, self.last_error

    async def aclose(self) -> None:
        """关闭：取消 worker、清空队列、释放嘴部独占（`disconnect()`/进程退出调用）。"""
        self.worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.worker_task
        discarded = self.queue.qsize()
        while not self.queue.empty():
            self.queue.get_nowait()
        self.active = False
        self.speech_mouth.clear(time.monotonic())
        logger.info("语音播放队列已关闭：丢弃未播 job %d 个，嘴部独占已释放。", discarded)

    # ── 内部 ─────────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        while True:
            job = await self.queue.get()
            self.active = True
            try:
                await self._play_job(job)
            except asyncio.CancelledError:
                raise  # worker 被 aclose() 取消：不吞，交由其 await worker_task 捕获
            except Exception as exc:
                self.last_error = repr(exc)
                logger.warning("语音播放失败——提前释放嘴部独占，继续下一条：%r", exc)
                self.speech_mouth.clear(time.monotonic())
            finally:
                self.active = False
                self.queue.task_done()

    async def _play_job(self, job: SpeechJob) -> None:
        """派发一次播放：线程↔事件循环边界的唯一交汇点（AD-7）。

        ``anchor`` 是本次播放的"起播锚点到达"信号——正常路径由播放线程经
        ``on_anchor``（线程安全）解出；若播放线程在报锚点**之前**就失败/结束
        （`_propagate_early_failure`），同样解出为异常，避免 `await anchor`
        无限期悬挂（蓝图 T3 验收④「播放中途异常→worker 不崩」的前置窗口）。
        """
        loop = asyncio.get_running_loop()
        anchor: asyncio.Future[float] = loop.create_future()

        def _set_anchor_result(t0: float) -> None:
            if not anchor.done():
                anchor.set_result(t0)

        def _on_anchor(t0: float) -> None:
            # 播放线程调用——唯一允许触碰事件循环状态的入口，必须走
            # call_soon_threadsafe（AD-7 线程边界不变式）。
            loop.call_soon_threadsafe(_set_anchor_result, t0)

        play_task: asyncio.Task[None] = asyncio.create_task(
            asyncio.to_thread(self.player.play_blocking, job.frames, _on_anchor)
        )

        def _propagate_early_failure(task: asyncio.Task[None]) -> None:
            if anchor.done():
                return
            if task.cancelled():
                anchor.cancel()
                return
            exc = task.exception()
            anchor.set_exception(
                exc if exc is not None else RuntimeError("播放线程未报起播锚点即正常结束")
            )

        play_task.add_done_callback(_propagate_early_failure)
        logger.info(
            "语音播放起播中：%.0fms，口型 %d 帧（fps=%g）",
            job.duration_ms,
            len(job.mouth_keyframes),
            job.fps,
        )
        try:
            t0 = await anchor
            logger.debug("起播锚点已到达：t0=%.3f（含设备 latency 偏移）", t0)
            result = self.speech_mouth.feed(
                job.mouth_keyframes,
                mode="absolute",
                append=False,
                now=t0,
                known_params=job.known_params,
            )
            if not result.ok:
                # 口型侧业务性拒绝（键集不一致/参数全缺席）：音频已在播、回执已
                # 提前返回（AD-4），无处回传——至少留痕（AD-12 可观测性承诺），
                # 否则口型静默丢失且 speech_last_error 恒 None，验收线①不可诊断。
                # 不升级为异常：音频不该因口型拒绝而中断。
                self.last_error = f"口型轨迹被拒（音频照常播放）：{result.code} {result.detail}"
                logger.warning("speech_play 口型注入被拒——%s", self.last_error)
            await play_task
            logger.debug("语音播放完成（%.0fms），口型轨迹自然播尽释放。", job.duration_ms)
        finally:
            if not play_task.done():
                play_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await play_task
