"""感知输入 I/O 适配层（T3）—— 文件/合成/**真硬件**信号源工厂公开 API。

本包提供将本地文件、合成信号或**真实设备**包装为 async callable 的工厂函数，
供 AudioChannel / VisionChannel / EdaChannel / HrvChannel 的
``signal_source`` 参数注入。**Channel 核心一字不改**。

设计约束：
- 只依赖科学栈（librosa / cv2 / neurokit2）与硬件库（sounddevice / pyserial）+ 标准库。
- 不反向依赖编排层 / 记忆层 / Zero（三层单向依赖守卫）。
- 不直接 import Zero 代码库（经 MCP 调用，见 CLAUDE.md 红线 5）。
- 路径 / 设备 / 配置经参数传入，不读 env（配置由调用方持有）。
- 硬件库一律**延迟 import**；缺库 / 无设备 / 读取失败 → warning + 返回 None，
  由 Channel 走既有「本轮无证据」优雅回退。**故构造工厂在无硬件环境下永远安全**。

文件 / 合成源（零硬件依赖）：
- ``make_audio_file_source``：从本地音频文件异步加载 16kHz mono float32 帧。
- ``make_vision_file_source``：从本地图像文件异步读取 RGB 人脸裁剪帧。
- ``make_synthetic_eda_source``：生成合成 EDA 信号 dict（neurokit2）。
- ``make_synthetic_hrv_source``：生成合成 ECG/HRV 信号 dict（neurokit2）。

真硬件源（依赖为**可选 extra**，默认不装；缺库即优雅回退 None）：
- ``make_mic_source``：麦克风实时采集（sounddevice，``[hardware-audio]``）。
- ``make_camera_source``：摄像头抓帧 + 可选 YuNet 人脸裁剪（opencv）。
- ``make_wearable_source``：串口可穿戴设备读生理信号（pyserial，``[hardware-wearable]``）。

⚠ 真机验证边界：mic / wearable 路径在本仓环境（未装 sounddevice / pyserial、无可穿戴设备）
只经 mock 单测覆盖；camera 依赖 cv2（在环境内）但可用摄像头取决于运行机器。
真机端到端须在具备对应硬件的机器上跑 ``-m realenv`` 用例。
"""

from __future__ import annotations

from src.mcp.zero.io_adapters.audio_file_adapter import make_audio_file_source
from src.mcp.zero.io_adapters.hardware_adapters import (
    make_camera_source,
    make_mic_source,
    make_wearable_source,
)
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
    "make_mic_source",
    "make_camera_source",
    "make_wearable_source",
]
