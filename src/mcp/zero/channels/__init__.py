"""感知通道子包公开 API。

导出：
- CallablePerceptionChannel：将任意 async callable 包装为满足 PerceptionChannel Protocol
  的感知通道，可直接交给 PerceptionHub。
- EdaChannel / HrvChannel：生理感知通道（EDA/SC·HRV/RMSSD → ModalityPrior，真接入 NeuroKit2）。
- AudioChannel：语音感知通道（真接入 audeering w2v2 维度 SER，走 ZERO_AUDIO_MODEL_PATH）。
- VisionChannel：视觉面部感知通道（真接入 EmotiEffLib ONNX 多任务 VA，走 ZERO_VISION_MODEL_NAME）。
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
