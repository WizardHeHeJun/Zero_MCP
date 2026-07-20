"""视觉感知通道（真接入）—— face frame → ModalityPrior。

模型 = EmotiEffLib（EfficientNet-B0 多任务，ONNX 后端），直出连续 VA ∈ [-1,1]。
用 ONNX 后端**绕过 timm 版本约束**（gap-1）：emotiefflib 的 timm==0.9.* 仅 torch 后端需要，
ONNX 后端零 timm 依赖，故不碰共用 env 的 D:\\Zero 依赖。选型/核验依据见
notes/2026-07-20-zero-link-audio-vision-real-integration.md。

设计依据（文献门纪要 notes/2026-07-16-zero-link-perception-litreview.md）：
- [EmotiEffLib GitHub sb-ai-lab] · [PyPI emotiefflib] · [HSEmotion ABAW-6 arXiv:2403.11590]
  ``*_va_mtl`` 多任务模型输出 scores shape (1, n_emotion+2)，**末两列 = [valence, arousal]**
  （gap-2-vision 已现场核验：happy 脸 valence≈+0.7 vs angry≈-0.7，见上述 note）。
  值域已是 [-1,1]（无需 *2-1，与 audio 不同），越界处 clip 防御。
- face Πv=0.20 > Πa=0.12：EmotiEffLib MT-DDAMFN CCC(v)=0.729 > CCC(a)=0.643。
- gap-1（timm 版本约束）：ONNX 后端零 timm 依赖，已绕过；绝不 conda prune。
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any

import numpy as np

from src.agents.models.zero_affect import ModalityPrior
from src.mcp.zero.external_priors import ModalityKind, build_recommended_prior

logger = logging.getLogger(__name__)

# 默认 EmotiEffLib 模型名（必须是 ``*_va_mtl`` 多任务模型才有 VA 输出）。env 可覆盖。
_DEFAULT_MODEL_NAME: str = "enet_b0_8_va_mtl"


def _load_recognizer(model_name: str, model_dir: str, device: str) -> Any:
    """延迟 import emotiefflib 并构造 ONNX recognizer（阻塞，走线程池）。

    model_dir 非空时拼进 model_name（emotiefflib 会追加 ``.onnx`` 并直接从该绝对路径加载，
    离线/自定义缓存友好；基类按子串识别 img_size/类别，路径不影响）。空时走默认
    ``~/.emotiefflib`` 缓存（首次自动从 GitHub 下载）。

    参考：[EmotiEffLib GitHub sb-ai-lab]（EmotiEffLibRecognizer(engine, model_name, device)）。
    """
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer

    resolved = os.path.join(model_dir, model_name) if model_dir else model_name
    return EmotiEffLibRecognizer(engine="onnx", model_name=resolved, device=device)


class VisionChannel:
    """视觉面部感知通道（真接入 EmotiEffLib ONNX 多任务 VA）。

    **Protocol 兼容**：结构上满足 PerceptionChannel Protocol（无需继承）：
    - name: str 属性（"vision"）。
    - async sense(frame=None) -> ModalityPrior | None（frame 有默认值=无参可调）。

    模型经 ``ZERO_VISION_MODEL_NAME``（默认 ``enet_b0_8_va_mtl``，须为 ``*_va_mtl`` 模型）
    + 可选 ``ZERO_VISION_MODEL_DIR``（自定义/离线缓存目录）指定。推理阻塞（CPU ~60ms/帧）
    → 走 ``asyncio.to_thread``。缺库/缺模型/非 VA 模型/推理失败/无输入 → warning + None
    （feature flag 默认关，零回归）。

    ⚠ 输入约定 **RGB** 人脸裁剪帧（与 EmotiEffLib 训练一致）；人脸检测/裁剪、BGR→RGB
    属独立 I/O 适配层职责，不进 Channel 核心。

    Args:
        signal_source: async callable → numpy RGB 人脸帧 | None；
                       PerceptionHub 无参调 sense() 时由此获取帧。
        device: 推理设备（默认 "cpu"）。
    """

    name: str = "vision"

    def __init__(self, signal_source: Any | None = None, device: str = "cpu") -> None:
        self.signal_source = signal_source
        self.device = device
        # 延迟加载缓存（首次推理时加载，线程锁防并发 collect 双载）
        self.recognizer: Any | None = None
        # 已判定不可用（非 *_va_mtl 模型）的哨兵，避免每轮重复加载昂贵 ONNX session
        self.recognizer_unavailable = False
        self.model_load_lock = threading.Lock()

    async def sense(
        self,
        frame: Any | None = None,
    ) -> ModalityPrior | None:
        """async：从人脸帧推理情感 VA，产出一条 ModalityPrior；无证据/未配置则返回 None。

        Args:
            frame: numpy RGB 人脸裁剪帧。None 时使用构造注入的 signal_source。

        Returns:
            ModalityPrior(modality="vision", mu=(μv,μa), precision=(0.20,0.12)) 或 None。

        Raises:
            不抛：I/O/推理异常（OSError/TimeoutError/RuntimeError/ValueError/ImportError）
            均 warning+None 回退；编程错误（TypeError 等）上抛供 PerceptionHub 兜。
        """
        # 运行时读 env——感知构造后 env 变更即时生效（不在 __init__ 缓存）
        if os.getenv("ZERO_VISION_CHANNEL_ENABLED", "false").lower() != "true":
            return None

        raw: Any = frame
        if raw is None and self.signal_source is not None:
            try:
                raw = await self.signal_source()
            except (OSError, TimeoutError, RuntimeError, ValueError) as exc:
                logger.warning("VisionChannel signal_source 调用失败，本轮跳过: %s", exc)
                return None

        if raw is None:
            return None

        # 阻塞推理走线程池，不堵事件循环（python-code.md async 规则）
        return await asyncio.to_thread(self._infer, raw)

    def _ensure_recognizer(self) -> Any | None:
        """延迟加载并缓存 recognizer；非 VA（非 mtl）模型 → warning + None（缓存哨兵不重载）。"""
        if self.recognizer is not None:
            return self.recognizer
        if self.recognizer_unavailable:  # 已判定非 VA 模型，不重复加载昂贵 ONNX session
            return None
        model_name = os.getenv("ZERO_VISION_MODEL_NAME", _DEFAULT_MODEL_NAME).strip()
        model_dir = os.getenv("ZERO_VISION_MODEL_DIR", "").strip()
        with self.model_load_lock:
            if self.recognizer is None and not self.recognizer_unavailable:
                rec = _load_recognizer(model_name, model_dir, self.device)
                if not getattr(rec, "is_mtl", False):
                    logger.warning(
                        "VisionChannel: 模型 %r 非 *_va_mtl（无 VA 输出），跳过", model_name
                    )
                    self.recognizer_unavailable = True
                    return None
                self.recognizer = rec
        return self.recognizer

    def _infer(self, frame: Any) -> ModalityPrior | None:
        """真模型推理（阻塞，由 sense() 经 asyncio.to_thread 调度）。

        流程：recognizer.predict_emotions(RGB 帧) → scores (1, n+2)；
        gap-2-vision 已核验末两列 = [valence, arousal]：μv=scores[-2], μa=scores[-1]（已 [-1,1]）。
        """
        try:
            recognizer = self._ensure_recognizer()
            if recognizer is None:
                return None

            _labels, scores = recognizer.predict_emotions(frame, logits=True)
            row = np.asarray(scores)[0]
            if row.shape[0] < 2:
                logger.warning("VisionChannel: scores 维度不足（无 VA 列），跳过")
                return None

            # gap-2-vision：末两列 = [valence, arousal]，已是 [-1,1]，越界处 clip 防御
            mu_v = float(np.clip(row[-2], -1.0, 1.0))
            mu_a = float(np.clip(row[-1], -1.0, 1.0))
            return build_recommended_prior(
                modality=self.name,
                mu=(mu_v, mu_a),
                kind=ModalityKind.FACE,
            )
        except ImportError as exc:
            logger.warning("VisionChannel: emotiefflib 不可用，本轮跳过: %s", exc)
            return None
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("VisionChannel 推理失败，本轮跳过: %s", exc)
            return None
