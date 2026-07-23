"""音频文件 I/O 适配器 —— 从本地文件异步加载 16kHz mono float32 帧。

定位：感知输入 I/O 适配层（T3）。
- 只依赖科学栈（librosa）+ 标准库，不反向依赖编排/记忆/Zero。
- 路径经参数传入，不读 env（配置由调用方持有）。
- 阻塞 I/O 经 asyncio.to_thread 调度，不堵事件循环。
- 缺文件/解码失败 → logging.warning + 返回 None（不抛）。
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Coroutine
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def make_audio_file_source(
    path: str | os.PathLike[str],
) -> Callable[[], Coroutine[Any, Any, np.ndarray | None]]:
    """构造"从文件读取音频帧"的 async callable（工厂）。

    返回的 async callable 每次调用均从 ``path`` 重新加载音频，以
    ``asyncio.to_thread`` 异步调度阻塞的 librosa.load，不堵事件循环。

    Args:
        path: 音频文件路径（WAV/MP3/FLAC 等 librosa 支持的格式）。
              路径经参数持有，不读 env。

    Returns:
        async callable ``() -> np.ndarray | None``：
        - 成功：float32 1D ndarray，16kHz mono（librosa 直出）。
        - 文件不存在 / 解码失败：logging.warning + 返回 None，不抛异常。
    """
    resolved = os.fspath(path)

    async def _source() -> np.ndarray | None:
        """从 ``resolved`` 加载 16kHz mono float32 帧。"""
        try:
            import librosa  # 延迟 import：避免模块加载时因缺包崩溃

            # lambda 包装：使 mypy 从 to_thread 返回类型拿到确定的 tuple[ndarray, int]
            # （librosa.load 签名中 sr 返回 int | float；lambda 内显式解构消歧）
            def _load() -> tuple[np.ndarray, int]:
                arr, sr_out = librosa.load(resolved, sr=16000, mono=True)
                return arr, int(sr_out)

            y, _sr = await asyncio.to_thread(_load)
            return y.astype(np.float32)
        except FileNotFoundError:
            logger.warning("audio_file_source: 文件不存在，跳过: %s", resolved)
            return None
        except Exception as exc:  # noqa: BLE001 — 解码/格式错误等均退回 None
            logger.warning("audio_file_source: 音频加载失败，跳过 (%s): %s", resolved, exc)
            return None

    return _source
