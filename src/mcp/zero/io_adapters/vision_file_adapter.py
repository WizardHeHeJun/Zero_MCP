"""视觉文件 I/O 适配器 —— 从本地图像文件异步读取并可选做人脸裁剪（RGB）。

定位：感知输入 I/O 适配层（T3）。
- 只依赖科学栈（cv2）+ 标准库，不反向依赖编排/记忆/Zero。
- 路径经参数传入，不读 env（配置由调用方持有）。
- 阻塞 I/O 经 asyncio.to_thread 调度，不堵事件循环。
- 缺文件/读取失败/无脸 → logging.warning + 按 fallback 返回（不抛）。

FaceDetectorYN 调用签名（opencv-python 4.8+/opencv5.0）：
  det = cv2.FaceDetectorYN.create(model_path, "", (w, h), score_threshold, nms_threshold, top_k)
  det.setInputSize((w, h))
  _, faces = det.detect(frame_bgr)
  faces: ndarray shape (N, 15) | None；前 4 列为 [x, y, w, h]。
（签名已与 tests/mcp/test_zero_vision_real.py 及 test_zero_perception_e2e.py::_face_crop 对齐核验）
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Coroutine
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


def make_vision_file_source(
    path: str | os.PathLike[str],
    yunet_model_path: str | os.PathLike[str] | None = None,
    fallback_on_no_face: Literal["whole_image", "none"] = "none",
    score_threshold: float = 0.6,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
) -> Callable[[], Coroutine[Any, Any, np.ndarray | None]]:
    """构造"从文件读取 RGB 人脸帧"的 async callable（工厂）。

    整流程在 asyncio.to_thread 内同步执行：
    1. cv2.imread(path) → BGR；失败 → warning + None。
    2. 若 yunet_model_path 可用且 cv2.FaceDetectorYN 存在：
       - 创建 FaceDetectorYN，detect → 有脸取第一个 bbox 裁剪 → BGR→RGB 返回。
       - 无脸：按 fallback_on_no_face 处理。
    3. yunet 不可用（None / cv2 无该属性 / cv2.error）：
       - fallback="whole_image" → 整图 BGR→RGB；
       - fallback="none" → None。

    Args:
        path:                 图像文件路径（PNG/JPG/BMP 等 cv2 支持的格式）。
        yunet_model_path:     YuNet ONNX 模型路径；None 则跳过人脸检测。
        fallback_on_no_face:  无脸/检测不可用时的行为：
                              ``"none"`` → 返回 None；
                              ``"whole_image"`` → 返回整张图的 RGB ndarray。
        score_threshold:      FaceDetectorYN 置信度阈值（默认 0.6）。
        nms_threshold:        FaceDetectorYN NMS 阈值（默认 0.3）。
        top_k:                FaceDetectorYN 候选框上限（默认 5000）。

    Returns:
        async callable ``() -> np.ndarray | None``：
        - 成功：RGB uint8 ndarray（人脸裁剪或整图）。
        - 失败/无脸/fallback="none"：返回 None，不抛异常。
    """
    resolved_path = os.fspath(path)
    resolved_yunet = os.fspath(yunet_model_path) if yunet_model_path is not None else None

    def _load_and_crop() -> np.ndarray | None:
        """同步执行：imread → 可选人脸检测 → RGB 输出。"""
        import cv2  # 延迟 import：避免模块加载时因缺包崩溃

        if not os.path.exists(resolved_path):
            logger.warning("vision_file_source: 文件不存在，跳过: %s", resolved_path)
            return None

        frame_bgr: np.ndarray | None = cv2.imread(resolved_path)
        if frame_bgr is None:
            logger.warning("vision_file_source: cv2.imread 返回 None，跳过: %s", resolved_path)
            return None

        h, w = frame_bgr.shape[:2]

        # --- 尝试人脸检测 ---
        if resolved_yunet is not None and hasattr(cv2, "FaceDetectorYN"):
            try:
                det = cv2.FaceDetectorYN.create(
                    resolved_yunet,
                    "",
                    (w, h),
                    score_threshold,
                    nms_threshold,
                    top_k,
                )
                det.setInputSize((w, h))
                _, faces = det.detect(frame_bgr)

                if faces is not None and len(faces) > 0:
                    # 取第一张脸的 bbox [x, y, bw, bh]
                    x, y, bw, bh = (int(v) for v in faces[0][:4])
                    x, y = max(0, x), max(0, y)
                    crop_bgr = frame_bgr[y : y + bh, x : x + bw]
                    if crop_bgr.size == 0:
                        logger.warning("vision_file_source: 裁剪区域为空，按 fallback 处理")
                        return _apply_fallback(frame_bgr, fallback_on_no_face)
                    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                else:
                    logger.warning("vision_file_source: 未检测到人脸，按 fallback 处理")
                    return _apply_fallback(frame_bgr, fallback_on_no_face)

            except cv2.error as exc:
                logger.warning("vision_file_source: FaceDetectorYN 异常，按 fallback 处理: %s", exc)
                return _apply_fallback(frame_bgr, fallback_on_no_face)
        else:
            # yunet 不可用（None 或 cv2 无 FaceDetectorYN）
            return _apply_fallback(frame_bgr, fallback_on_no_face)

    async def _source() -> np.ndarray | None:
        """从 ``resolved_path`` 读取 RGB 人脸帧（阻塞 I/O 走线程池）。"""
        try:
            return await asyncio.to_thread(_load_and_crop)
        except ImportError as exc:
            logger.warning("vision_file_source: cv2 不可用，跳过: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001 — 与 audio 侧对齐，未预期错误退回 None
            logger.warning("vision_file_source: 处理失败，跳过: %s", exc)
            return None

    return _source


def _apply_fallback(
    frame_bgr: np.ndarray,
    fallback: Literal["whole_image", "none"],
) -> np.ndarray | None:
    """无脸/检测不可用时按 fallback 策略返回结果。"""
    import cv2  # 已在调用栈上层 import，此处取缓存

    if fallback == "whole_image":
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return None
