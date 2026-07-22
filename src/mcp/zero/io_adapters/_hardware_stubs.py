"""硬件信号源桩（T3 占位）—— mic / camera / wearable 真设备接入占位。

调用任何工厂函数均抛 NotImplementedError（T3 桩，硬件未接入）。
硬件桩不默认导出（调用方需显式从 ``_hardware_stubs`` import）。

设计意图：
- 脱设备单测可验证桩行为（pytest.raises(NotImplementedError)）。
- 不 import sounddevice（未装硬件驱动，避免 ImportError）。
- 真设备接入在后续 Task 实现，替换桩体即可、签名不变。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Coroutine
from typing import Any

# 信号源 callable 的类型别名（与适配层其他工厂返回类型对齐）
SignalSource = Callable[[], Coroutine[Any, Any, Any]]


def make_mic_source(
    device: int | str | None = None,
    sample_rate: int = 16000,
    chunk_duration_s: float = 3.0,
) -> SignalSource:
    """麦克风实时采集信号源工厂（T3 桩，未实现）。

    Args:
        device:           音频输入设备 index 或名称（None = 系统默认）。
        sample_rate:      采样率（Hz）。默认 16000（AudioChannel 期望值）。
        chunk_duration_s: 每次采集时长（秒）。默认 3.0。

    Raises:
        NotImplementedError: 始终抛出，T3 桩未实现真设备接入。
    """
    raise NotImplementedError(
        "make_mic_source: 麦克风接入为 T3 桩，尚未实现真设备采集。"
        " 请使用 make_audio_file_source 作为替代，或在后续 Task 实现此桩。"
    )


def make_camera_source(
    device: int = 0,
    yunet_model_path: str | os.PathLike[str] | None = None,
) -> SignalSource:
    """摄像头实时采集信号源工厂（T3 桩，未实现）。

    Args:
        device:           摄像头设备 index（默认 0）。
        yunet_model_path: YuNet 人脸检测模型路径（None 则跳过检测）。

    Raises:
        NotImplementedError: 始终抛出，T3 桩未实现真设备接入。
    """
    raise NotImplementedError(
        "make_camera_source: 摄像头接入为 T3 桩，尚未实现真设备采集。"
        " 请使用 make_vision_file_source 作为替代，或在后续 Task 实现此桩。"
    )


def make_wearable_source(
    port: str | None = None,
    sampling_rate: int = 256,
) -> SignalSource:
    """可穿戴设备实时采集信号源工厂（T3 桩，未实现）。

    Args:
        port:          串口名称（None = 自动探测）。
        sampling_rate: 采样率（Hz）。默认 256（ECG 常见采样率）。

    Raises:
        NotImplementedError: 始终抛出，T3 桩未实现真设备接入。
    """
    raise NotImplementedError(
        "make_wearable_source: 可穿戴设备接入为 T3 桩，尚未实现真设备采集。"
        " 请使用 make_synthetic_eda_source / make_synthetic_hrv_source 作为替代，"
        " 或在后续 Task 实现此桩。"
    )
