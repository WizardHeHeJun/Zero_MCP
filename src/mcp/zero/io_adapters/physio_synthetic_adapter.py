"""生理信号合成 I/O 适配器 —— 用 neurokit2 生成一次合成 EDA/ECG 信号的工厂。

定位：感知输入 I/O 适配层（T3）。
- 只依赖科学栈（neurokit2/numpy）+ 标准库，不反向依赖编排/记忆/Zero。
- 同步工厂：构造时（延迟 import neurokit2）一次性生成 ndarray，
  返回捕获该 ndarray 的 async 闭包；每次调用零阻塞即返回。
- neurokit2 ImportError → 工厂内 warning，返回的 callable 每次 return None。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def make_synthetic_eda_source(
    duration: int = 60,
    sampling_rate: int = 4,
    scr_number: int = 5,
    random_state: int = 0,
) -> Callable[[], Coroutine[Any, Any, dict[str, Any] | None]]:
    """构造"返回合成 EDA 信号 dict"的 async callable（工厂，同步执行）。

    构造时调用 ``nk.eda_simulate(...)`` 生成一次 ndarray；返回的 async 闭包
    每次调用零阻塞返回相同信号（测试/离线调试用）。

    Args:
        duration:       模拟时长（秒）。默认 60。
        sampling_rate:  采样率（Hz）。默认 4（EdaChannel 默认采样率）。
        scr_number:     皮肤电反应（SCR）事件数。默认 5。
        random_state:   随机种子（保证可复现）。默认 0。

    Returns:
        async callable ``() -> {"eda": ndarray, "sampling_rate": int} | None``：
        - 成功：含 "eda"（float64 ndarray）与 "sampling_rate"（int）的 dict。
        - neurokit2 不可用：返回 None。
    """
    _data: dict[str, Any] | None

    try:
        import neurokit2 as nk  # 延迟 import：避免模块加载时因缺包崩溃

        arr: np.ndarray = nk.eda_simulate(
            duration=duration,
            sampling_rate=sampling_rate,
            scr_number=scr_number,
            random_state=random_state,
        )
        _data = {"eda": arr, "sampling_rate": int(sampling_rate)}
    except ImportError as exc:
        logger.warning("make_synthetic_eda_source: neurokit2 不可用，将返回 None: %s", exc)
        _data = None

    async def _source() -> dict[str, Any] | None:
        """返回预生成的合成 EDA 信号 dict（零阻塞）。"""
        return _data

    return _source


def make_synthetic_hrv_source(
    duration: int = 30,
    sampling_rate: int = 256,
    heart_rate: int = 70,
    random_state: int = 0,
) -> Callable[[], Coroutine[Any, Any, dict[str, Any] | None]]:
    """构造"返回合成 ECG/HRV 信号 dict"的 async callable（工厂，同步执行）。

    构造时调用 ``nk.ecg_simulate(...)`` 生成一次 ndarray；返回的 async 闭包
    每次调用零阻塞返回相同信号（测试/离线调试用）。

    Args:
        duration:       模拟时长（秒）。默认 30。
        sampling_rate:  采样率（Hz）。默认 256（ECG 常见采样率）。
        heart_rate:     心率（bpm）。默认 70。
        random_state:   随机种子（保证可复现）。默认 0。

    Returns:
        async callable ``() -> {"ecg_or_ppg": ndarray, "sampling_rate": int} | None``：
        - 成功：含 "ecg_or_ppg"（float64 ndarray）与 "sampling_rate"（int）的 dict。
        - neurokit2 不可用：返回 None。
    """
    _data: dict[str, Any] | None

    try:
        import neurokit2 as nk  # 延迟 import：避免模块加载时因缺包崩溃

        arr: np.ndarray = nk.ecg_simulate(
            duration=duration,
            sampling_rate=sampling_rate,
            heart_rate=heart_rate,
            random_state=random_state,
        )
        _data = {"ecg_or_ppg": arr, "sampling_rate": int(sampling_rate)}
    except ImportError as exc:
        logger.warning("make_synthetic_hrv_source: neurokit2 不可用，将返回 None: %s", exc)
        _data = None

    async def _source() -> dict[str, Any] | None:
        """返回预生成的合成 ECG 信号 dict（零阻塞）。"""
        return _data

    return _source
