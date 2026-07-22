"""感知输入 I/O 适配层（T3）—— 文件/合成信号源工厂公开 API。

本包提供将本地文件或合成信号包装为 async callable 的工厂函数，
供 AudioChannel / VisionChannel / EdaChannel / HrvChannel 的
``signal_source`` 参数注入。

设计约束：
- 只依赖科学栈（librosa / cv2 / neurokit2）+ 标准库。
- 不反向依赖编排层 / 记忆层 / Zero（三层单向依赖守卫）。
- 不直接 import Zero 代码库（经 MCP 调用，见 CLAUDE.md 红线 5）。
- 路径 / 配置经参数传入，不读 env（配置由调用方持有）。

默认导出（4 个文件/合成工厂）：
- ``make_audio_file_source``：从本地音频文件异步加载 16kHz mono float32 帧。
- ``make_vision_file_source``：从本地图像文件异步读取 RGB 人脸裁剪帧。
- ``make_synthetic_eda_source``：生成合成 EDA 信号 dict（neurokit2）。
- ``make_synthetic_hrv_source``：生成合成 ECG/HRV 信号 dict（neurokit2）。

硬件桩（mic / camera / wearable）不默认导出，调用方需显式导入：
    from src.mcp.zero.io_adapters._hardware_stubs import make_mic_source
"""

from __future__ import annotations

from src.mcp.zero.io_adapters.audio_file_adapter import make_audio_file_source
from src.mcp.zero.io_adapters.physio_synthetic_adapter import (
    make_synthetic_eda_source,
    make_synthetic_hrv_source,
)
from src.mcp.zero.io_adapters.vision_file_adapter import make_vision_file_source

__all__ = [
    "make_audio_file_source",
    "make_vision_file_source",
    "make_synthetic_eda_source",
    "make_synthetic_hrv_source",
]
