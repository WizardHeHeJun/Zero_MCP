"""视觉感知通道（stub）—— face frame → ModalityPrior。

当前为 stub：flag 关或无输入均返回 None。
真接入挂载点见 _infer() 方法注释。

设计依据（文献门纪要 notes/2026-07-16-zero-link-perception-litreview.md）：
- [EmotiEffLib GitHub sb-ai-lab] · [PyPI emotiefflib] · [HSEmotion ABAW-6 arXiv:2403.11590]
  EfficientNet-B0 多任务，直出连续 VA ∈ [-1,1]（无需值域映射），CPU ~60ms。
- face Πv=0.20 > Πa=0.12：EmotiEffLib MT-DDAMFN CCC(v)=0.729 > CCC(a)=0.643。
- ⚠ gap-1：EmotiEffLib 依赖 timm，接线前须 uv 层实测版本约束（0.4.5 vs 0.9.x），
  绝对不能 conda prune（会破坏 D:\\Zero 共用环境）。
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.agents.models.zero_affect import ModalityPrior

logger = logging.getLogger(__name__)


class VisionChannel:
    """视觉面部感知通道（stub）。

    **Protocol 兼容**：结构上满足 PerceptionChannel Protocol（无需继承）：
    - name: str 属性（"vision"）。
    - async sense(frame=None) -> ModalityPrior | None（frame 有默认值=无参可调）。

    真接入（EmotiEffLib）挂载点：
        1. 装依赖：``uv pip install emotiefflib``；timm 版本约束须先实测（gap-1）。
           ⚠ 绝不 conda prune——共用 affective-expression 环境含 D:\\Zero 依赖。
        2. 模型权重目录写入 env ``ZERO_VISION_MODEL_DIR``。
        3. 在 _infer() 中加载 EmotiEffLib 模型（构造时缓存），接收 BGR/RGB numpy 帧，
           直接得到 (valence, arousal) ∈ [-1,1]（无需值域映射）：
               μv = float(out_va[0])
               μa = float(out_va[1])
        4. return build_recommended_prior(modality=self.name, mu=(μv, μa),
                                          kind=ModalityKind.FACE)
        ⚠ gap-1：timm 版本约束（emotiefflib 旧版需 timm==0.4.5）接线前必须
                 uv 层实测确认，不得推测；权重目录走 env ZERO_VISION_MODEL_DIR（不硬编码）。

    Args:
        signal_source: async callable → numpy BGR/RGB 帧 | None；
                       PerceptionHub 无参调 sense() 时由此获取帧。
    """

    name: str = "vision"

    def __init__(self, signal_source: Any | None = None) -> None:
        self.signal_source = signal_source
        self.model_dir: str = os.getenv("ZERO_VISION_MODEL_DIR", "")

    async def sense(
        self,
        frame: Any | None = None,
    ) -> ModalityPrior | None:
        """async：从人脸帧推理情感 VA，产出一条 ModalityPrior；stub 恒返回 None。

        Args:
            frame: numpy BGR/RGB 人脸帧。None 时使用构造注入的 signal_source。

        Returns:
            stub：None。真接入后返回 ModalityPrior(modality="vision", ...)。

        Raises:
            不抛：I/O 异常（OSError/TimeoutError/RuntimeError/ValueError）
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

        return self._infer(raw)

    def _infer(self, frame: Any) -> ModalityPrior | None:
        """真模型推理挂点（当前 stub，返回 None）。

        真接入步骤：
        1. ``import emotiefflib`` / ``from emotiefflib import EmotiEffLib``（延迟 import，
           ImportError → warning+None）。
        2. 加载 ``self.model_dir`` 路径下的模型（构造时缓存，此处仅推理）。
        3. 预处理 frame（resize/normalize，参考 EmotiEffLib 文档）。
        4. 推理得 (valence, arousal) ∈ [-1,1]（EmotiEffLib 直出，无需映射）：
               μv = float(out_va[0])
               μa = float(out_va[1])
        5. return build_recommended_prior(modality=self.name, mu=(μv, μa),
                                          kind=ModalityKind.FACE)
        ⚠ gap-1：emotiefflib 对 timm 版本有约束，接线前须 uv 层实测（timm 0.4.5 vs 0.9.x）；
                 权重目录走 env ZERO_VISION_MODEL_DIR（不硬编码）。
        ⚠ 阻塞警告：EmotiEffLib/ONNX 推理耗时 >1ms 时，须改为
                 ``await asyncio.to_thread(model.predict, frame)`` 在线程池执行，
                 不可在 async sense() 里塞同步阻塞推理（违反 python-code.md async 规则）。
        """
        # stub：真接入时替换此行
        return None
