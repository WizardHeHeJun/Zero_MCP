"""test_phash.py — phash 统一工具单测（重点：crop 裁剪语义，Task 12 新增）。

覆盖：
- crop 命中区域内变化 → 位向量不同；区域外变化 → 位向量相同
- crop 越界自动 clamp；clamp 后空区域回退整图
- 与整图 hash 的默认行为兼容（crop=None）
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.orchestration.phash import average_hash_from_path, hamming_ratio


def _write_img(path: Path, arr: np.ndarray) -> str:
    cv2.imwrite(str(path), arr)
    return str(path)


def _base_img() -> np.ndarray:
    """200x200 左黑右白基准图（产生非平凡 hash bits）。"""
    img = np.zeros((200, 200), dtype=np.uint8)
    img[:, 100:] = 255
    return img


def test_crop_ignores_change_outside_region(tmp_path: Path) -> None:
    """变化发生在裁剪区外（模拟窗口角落动画）→ 局部 hash 相同。"""
    img_a = _base_img()
    img_b = _base_img()
    img_b[0:40, 0:40] = 200  # 左上角"动画区"变化，远离目标邻域

    path_a = _write_img(tmp_path / "a.png", img_a)
    path_b = _write_img(tmp_path / "b.png", img_b)

    crop = (120, 120, 200, 200)  # 目标邻域在右下
    bits_a = average_hash_from_path(path_a, crop)
    bits_b = average_hash_from_path(path_b, crop)
    assert hamming_ratio(bits_a, bits_b) == 0.0

    # 对照：整图 hash 能看到该变化（证明裁剪确实屏蔽了区外噪声）
    full_a = average_hash_from_path(path_a)
    full_b = average_hash_from_path(path_b)
    assert hamming_ratio(full_a, full_b) > 0.0


def test_crop_detects_change_inside_region(tmp_path: Path) -> None:
    """变化发生在裁剪区内（目标被篡改）→ 局部 hash 不同。"""
    img_a = _base_img()
    img_b = _base_img()
    img_b[140:200, 140:200] = 0  # 右下目标邻域内白变黑

    path_a = _write_img(tmp_path / "a.png", img_a)
    path_b = _write_img(tmp_path / "b.png", img_b)

    crop = (120, 120, 200, 200)
    bits_a = average_hash_from_path(path_a, crop)
    bits_b = average_hash_from_path(path_b, crop)
    assert hamming_ratio(bits_a, bits_b) > 0.0


def test_crop_clamps_out_of_bounds(tmp_path: Path) -> None:
    """裁剪区部分越界 → clamp 到图内，不抛异常。"""
    path = _write_img(tmp_path / "a.png", _base_img())
    bits = average_hash_from_path(path, (-50, -50, 100, 100))
    assert bits.shape == (64,)


def test_crop_degenerate_falls_back_to_full_image(tmp_path: Path) -> None:
    """裁剪区完全在图外（clamp 后为空）→ 回退整图 hash。"""
    path = _write_img(tmp_path / "a.png", _base_img())
    bits_degenerate = average_hash_from_path(path, (500, 500, 700, 700))
    bits_full = average_hash_from_path(path)
    assert hamming_ratio(bits_degenerate, bits_full) == 0.0
