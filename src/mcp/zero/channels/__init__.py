"""感知通道子包公开 API。

导出：
- CallablePerceptionChannel：将任意 async callable 包装为满足 PerceptionChannel Protocol
  的感知通道，可直接交给 PerceptionHub。
- EdaChannel / HrvChannel：生理感知通道（EDA/SC·HRV/RMSSD → ModalityPrior，真接入）。
- AudioChannel：语音感知通道（stub，真接入挂 Wav2Small ONNX）。
- VisionChannel：视觉面部感知通道（stub，真接入挂 EmotiEffLib）。
"""

from __future__ import annotations

from src.mcp.zero.channels.audio_channel import AudioChannel
from src.mcp.zero.channels.callable_channel import CallablePerceptionChannel
from src.mcp.zero.channels.physio_channel import EdaChannel, HrvChannel
from src.mcp.zero.channels.vision_channel import VisionChannel

__all__ = [
    "CallablePerceptionChannel",
    "EdaChannel",
    "HrvChannel",
    "AudioChannel",
    "VisionChannel",
]
