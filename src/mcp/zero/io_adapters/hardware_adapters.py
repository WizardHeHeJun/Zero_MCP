"""真设备采集 I/O 适配器 —— mic / camera / wearable 实时信号源工厂。

定位：感知输入 I/O 适配层（T3 硬件档），与 `audio_file_adapter` / `vision_file_adapter` /
`physio_synthetic_adapter` 同层同形——各工厂返回 async callable，经 Channel 的
``signal_source=`` 注入，**Channel 核心一字不改**。

设计约束（与同层其余适配器一致）：
- 只依赖硬件库 + 标准库/科学栈；不反向依赖编排层 / 记忆层 / Zero。
- 硬件库一律**延迟 import**（模块加载不碰）；缺库 / 无设备 / 读取失败 → warning + 返回 None，
  由 Channel 走既有「本轮无证据」优雅回退，**不抛给上层**。
- 阻塞采集走 ``asyncio.to_thread``，不堵事件循环。
- 设备参数经函数入参传入，**不读 env**（配置由调用方持有）。

依赖与安装（**均为可选 extra，默认不装**）：
    uv pip install -e ".[hardware-audio]"     # sounddevice（麦克风）
    uv pip install -e ".[hardware-wearable]"  # pyserial（可穿戴串口）
    # 摄像头走 opencv-python，已随 perception-vision extra 提供

⚠ 真机验证边界（诚实标注）：本模块的三个工厂**均可在无设备环境下安全构造**（返回的 callable
返回 None）。截至落地时本仓环境 **未装 sounddevice / pyserial、无可穿戴设备**，故
mic/wearable 路径只经 mock 单测覆盖；camera 路径依赖 cv2（在环境内）但**是否有可用摄像头
取决于运行机器**。真机端到端验证须在具备对应硬件的机器上跑 ``-m realenv`` 用例。
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Coroutine
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# 信号源 callable 的类型别名（与同层文件/合成适配器的返回类型对齐）
SignalSource = Callable[[], Coroutine[Any, Any, Any]]


# ---------------------------------------------------------------------------
# 麦克风（sounddevice）→ AudioChannel
# ---------------------------------------------------------------------------


def make_mic_source(
    device: int | str | None = None,
    sample_rate: int = 16000,
    chunk_duration_s: float = 3.0,
) -> SignalSource:
    """构造「录一段麦克风音频」的 async callable（AudioChannel 期望的 float32 mono 帧）。

    每次调用录 ``chunk_duration_s`` 秒并返回 1-D float32 ndarray（幅度约 [-1,1]），
    形状与 ``make_audio_file_source`` 一致，故 AudioChannel 无需任何改动。

    Args:
        device:           输入设备 index 或名称；None = 系统默认输入设备。
        sample_rate:      采样率 Hz。默认 16000（audeering w2v2 期望值）。
        chunk_duration_s: 单次采集时长（秒）。默认 3.0。

    Returns:
        async callable ``() -> np.ndarray | None``：
        - 成功：1-D float32 ndarray（长度 ≈ sample_rate × chunk_duration_s）。
        - sounddevice 缺失 / 无输入设备 / 录制失败：None（warning 后回退）。
    """
    frames = max(1, int(sample_rate * chunk_duration_s))

    def _record() -> np.ndarray | None:
        """阻塞录音（走线程池）。缺库/无设备/失败 → None。"""
        try:
            import sounddevice as sd  # 延迟 import：缺库不影响模块加载
        except ImportError as exc:
            logger.warning("make_mic_source: sounddevice 不可用，本轮跳过: %s", exc)
            return None
        try:
            buf = sd.rec(
                frames,
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=device,
            )
            sd.wait()
        except Exception as exc:  # sounddevice 抛 PortAudioError 等库内异常，统一优雅回退
            logger.warning("make_mic_source: 麦克风采集失败，本轮跳过: %s", exc)
            return None
        arr = np.asarray(buf, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            logger.warning("make_mic_source: 采集到空缓冲，本轮跳过")
            return None
        return arr

    async def _source() -> np.ndarray | None:
        """async：录一段麦克风音频并返回 float32 mono 帧。"""
        return await asyncio.to_thread(_record)

    return _source


# ---------------------------------------------------------------------------
# 摄像头（OpenCV VideoCapture + 可选 YuNet 人脸裁剪）→ VisionChannel
# ---------------------------------------------------------------------------


def make_camera_source(
    device: int = 0,
    yunet_model_path: str | os.PathLike[str] | None = None,
) -> SignalSource:
    """构造「抓一帧摄像头画面」的 async callable（VisionChannel 期望的 RGB uint8 帧）。

    与 ``make_vision_file_source`` 同形：BGR→RGB 转换；给了 YuNet 模型路径则做人脸检测并
    返回**人脸裁剪**，否则返回整帧。无脸时回退整帧（对齐文件适配器的 "whole_image" 语义）。

    ⚠ 每次调用**即开即关** ``VideoCapture``：牺牲少量延迟换「不长期占用摄像头」，
    与本层「适配器无状态、不持设备句柄」的取向一致（长期持有会挡住其它进程用摄像头）。

    Args:
        device:           摄像头 index。默认 0。
        yunet_model_path: YuNet ONNX 路径；None = 跳过人脸检测、返回整帧。

    Returns:
        async callable ``() -> np.ndarray | None``：
        - 成功：RGB uint8 ndarray（H×W×3）。
        - cv2 缺失 / 打不开设备 / 读帧失败：None（warning 后回退）。
    """

    def _grab() -> np.ndarray | None:
        """阻塞抓帧（走线程池）。缺库/无设备/读失败 → None。"""
        try:
            import cv2  # 延迟 import
        except ImportError as exc:
            logger.warning("make_camera_source: opencv 不可用，本轮跳过: %s", exc)
            return None

        cap = None
        try:
            cap = cv2.VideoCapture(device)
            if not cap.isOpened():
                logger.warning("make_camera_source: 打不开摄像头 device=%r，本轮跳过", device)
                return None
            ok, frame_bgr = cap.read()
        except Exception as exc:
            logger.warning("make_camera_source: 摄像头读帧失败，本轮跳过: %s", exc)
            return None
        finally:
            if cap is not None:
                cap.release()  # 即开即关：不长期占用设备

        if not ok or frame_bgr is None:
            logger.warning("make_camera_source: 读到空帧，本轮跳过")
            return None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if yunet_model_path is None:
            return np.asarray(frame_rgb, dtype=np.uint8)
        return _crop_face_or_whole(cv2, frame_rgb, str(yunet_model_path))

    async def _source() -> np.ndarray | None:
        """async：抓一帧摄像头画面并返回 RGB uint8 帧。"""
        return await asyncio.to_thread(_grab)

    return _source


def _crop_face_or_whole(cv2: Any, frame_rgb: np.ndarray, model_path: str) -> np.ndarray:
    """YuNet 检测人脸并裁剪；无脸/检测失败 → 回退整帧（不抛）。"""
    height, width = frame_rgb.shape[:2]
    try:
        detector = cv2.FaceDetectorYN.create(model_path, "", (width, height))
        # YuNet 吃 BGR，此处已转 RGB → 转回去喂检测器
        _, faces = detector.detect(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    except Exception as exc:
        logger.warning("make_camera_source: YuNet 检测失败，回退整帧: %s", exc)
        return np.asarray(frame_rgb, dtype=np.uint8)

    if faces is None or len(faces) == 0:
        logger.debug("make_camera_source: 未检出人脸，回退整帧")
        return np.asarray(frame_rgb, dtype=np.uint8)

    x, y, w, h = (int(v) for v in faces[0][:4])
    # 钳到画面内，防越界切出空数组
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        logger.warning("make_camera_source: 人脸框退化，回退整帧")
        return np.asarray(frame_rgb, dtype=np.uint8)
    return np.asarray(frame_rgb[y0:y1, x0:x1], dtype=np.uint8)


# ---------------------------------------------------------------------------
# 可穿戴设备（串口）→ HrvChannel / EdaChannel
# ---------------------------------------------------------------------------


def make_wearable_source(
    port: str | None = None,
    sampling_rate: int = 256,
    duration_s: float = 30.0,
    baudrate: int = 115200,
    signal_key: str = "ecg_or_ppg",
) -> SignalSource:
    """构造「从串口可穿戴设备读一段生理信号」的 async callable。

    产出形状对齐 ``make_synthetic_hrv_source`` / ``make_synthetic_eda_source``
    （``{signal_key: ndarray, "sampling_rate": int}``），故 HrvChannel / EdaChannel 无需改动。

    **线路协议约定**：设备按行输出 ASCII 十进制样本（每行一个数，如 ``"0.123\\n"``）。
    这是最小可用约定；私有二进制协议请另写适配器（本层允许多实现并存）。

    Args:
        port:          串口名（如 ``"COM3"`` / ``"/dev/ttyUSB0"``）；None = 自动探测第一个可用口。
        sampling_rate: 采样率 Hz。默认 256（ECG 常见）。
        duration_s:    单次采集时长（秒）。默认 30.0（HRV 需足够长才有可靠 RMSSD）。
        baudrate:      串口波特率。默认 115200。
        signal_key:    产出 dict 的信号键名；``"ecg_or_ppg"``（HrvChannel）
                       或 ``"eda"``（EdaChannel）。

    Returns:
        async callable ``() -> dict[str, Any] | None``：
        - 成功：``{signal_key: float64 ndarray, "sampling_rate": int}``。
        - pyserial 缺失 / 无可用串口 / 读取失败 / 样本不足：None（warning 后回退）。
    """
    expected = max(1, int(sampling_rate * duration_s))

    def _read() -> dict[str, Any] | None:
        """阻塞串口读取（走线程池）。缺库/无口/失败/样本不足 → None。"""
        try:
            import serial  # pyserial，延迟 import
            from serial.tools import list_ports
        except ImportError as exc:
            logger.warning("make_wearable_source: pyserial 不可用，本轮跳过: %s", exc)
            return None

        resolved = port
        if resolved is None:
            candidates = list(list_ports.comports())
            if not candidates:
                logger.warning("make_wearable_source: 未探测到任何串口，本轮跳过")
                return None
            resolved = candidates[0].device
            logger.debug("make_wearable_source: 自动选用串口 %s", resolved)

        samples: list[float] = []
        try:
            # 读超时留 2× 余量：设备偶发抖动不至于立刻判失败
            with serial.Serial(resolved, baudrate, timeout=duration_s * 2.0) as ser:
                while len(samples) < expected:
                    raw = ser.readline()
                    if not raw:  # 超时返回空字节 → 设备停止输出
                        break
                    try:
                        samples.append(float(raw.decode("ascii", errors="ignore").strip()))
                    except ValueError:
                        continue  # 跳过非数值行（设备握手/日志行）
        except Exception as exc:
            logger.warning("make_wearable_source: 串口读取失败（%s），本轮跳过: %s", resolved, exc)
            return None

        # 样本不足会让 HRV 的 R 峰检出不可靠 → 宁可判无证据，也不产垃圾先验
        if len(samples) < expected:
            logger.warning(
                "make_wearable_source: 样本不足（%d/%d），本轮跳过", len(samples), expected
            )
            return None
        return {signal_key: np.asarray(samples, dtype=float), "sampling_rate": int(sampling_rate)}

    async def _source() -> dict[str, Any] | None:
        """async：从串口读一段生理信号并返回信号 dict。"""
        return await asyncio.to_thread(_read)

    return _source
