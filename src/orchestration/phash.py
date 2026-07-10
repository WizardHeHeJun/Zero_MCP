"""感知哈希（average hash）统一工具（编排层共享）。

统一此前 action_guard（TOCTOU 验证）与 desktop_graph（停滞检测信号 A）各自的
phash 实现，消除「phash 双实现」技术债（原两份算法一致但 I/O 契约不同，无法共用）。

核心算法：图像转 8x8 灰度（cv2.INTER_AREA 缩放），阈值取 `> mean`（严格大于），
得 64 位感知哈希。两个消费点 I/O 契约不同——TOCTOU 用文件路径 + bool 向量 + 归一化
比率，停滞检测用图像字节 + "0/1" 字符串 + 位数——故按契约分层暴露多个 API，
但核心位计算（`_average_hash_bits`）只有一份。

层约束：编排层内部工具（src/orchestration/），只 import cv2/numpy 科学栈，
不 import 记忆/存储/agents 层。action_guard（orchestration/safety）与 desktop_graph
（orchestration）均同层下用本模块。
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_HASH_SIDE: int = 8  # 8x8 缩放
_HASH_BITS: int = _HASH_SIDE * _HASH_SIDE  # 64 位


def _average_hash_bits(gray_img: np.ndarray) -> np.ndarray:
    """核心：灰度图 → 64 位 average hash 布尔向量（阈值 `> mean`）。

    Args:
        gray_img: 单通道灰度图（cv2.IMREAD_GRAYSCALE / imdecode 结果）。

    Returns:
        shape=(64,) dtype=bool 的位向量。
    """
    resized = cv2.resize(gray_img, (_HASH_SIDE, _HASH_SIDE), interpolation=cv2.INTER_AREA)
    mean_val = float(resized.mean())
    bits: np.ndarray = (resized > mean_val).flatten()
    return bits


def average_hash_from_path(screenshot_path: str) -> np.ndarray:
    """从文件路径计算 average hash 位向量（TOCTOU 验证用）。

    Args:
        screenshot_path: 截图文件路径。

    Returns:
        shape=(64,) dtype=bool 的位向量。

    Raises:
        ValueError: 文件不存在或无法读取（TOCTOU 调用点据此降级放行）。
    """
    img = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取截图文件：{screenshot_path}")
    return _average_hash_bits(img)


def average_hash_from_bytes(image_bytes: bytes) -> str | None:
    """从图像字节计算 average hash 二进制字符串（停滞检测信号 A 用）。

    Args:
        image_bytes: PNG/JPEG 等图像字节数据。

    Returns:
        64 位 "0"/"1" 字符串（便于存入 state 的 str 字段）；解码失败返回 None（不抛出）。
    """
    try:
        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        bits = _average_hash_bits(img)
        return "".join("1" if b else "0" for b in bits)
    except Exception as exc:
        logger.debug("average_hash_from_bytes: 计算失败 %s", exc)
        return None


def hamming_ratio(bits_a: np.ndarray, bits_b: np.ndarray) -> float:
    """两个位向量的归一化汉明距离（0.0=完全相同，1.0=完全不同），TOCTOU 用。

    Args:
        bits_a: 第一个 bool 位向量。
        bits_b: 第二个 bool 位向量。

    Returns:
        归一化汉明距离 [0.0, 1.0]。
    """
    diff = int(np.sum(bits_a != bits_b))
    return diff / len(bits_a)


def hamming_bits(hash_a: str, hash_b: str) -> int:
    """两个等长二进制字符串的汉明距离（不同位数量），停滞检测用。

    Args:
        hash_a: 64 位 "0"/"1" 字符串。
        hash_b: 64 位 "0"/"1" 字符串。

    Returns:
        汉明距离；长度不等时返回 64（最大值）。
    """
    if len(hash_a) != len(hash_b):
        return _HASH_BITS
    return sum(a != b for a, b in zip(hash_a, hash_b, strict=True))
