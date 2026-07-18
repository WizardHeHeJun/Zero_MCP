"""语音感知通道（stub）—— audio → ModalityPrior。

当前为 stub：flag 关或无输入均返回 None。
真接入挂载点见 _infer() 方法注释。

设计依据（文献门纪要 notes/2026-07-16-zero-link-perception-litreview.md）：
- [Wav2Small arXiv:2408.13920] · [dkounadis/wav2small HuggingFace]
  输出字段序待核验（gap-2，接线前必须确认 [arousal,dominance,valence] 否则 μv/μa 反转）。
  输出值域 [0,1] 须映射 *2-1 到 [-1,1]。
- [Wagner 2023 arXiv:2203.07378] 纯声学 valence 内在弱 → audio Πv=0.10 < Πa=0.25。
- [audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim HuggingFace] 备选全精度模型。
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.agents.models.zero_affect import ModalityPrior

logger = logging.getLogger(__name__)


class AudioChannel:
    """语音感知通道（stub）。

    **Protocol 兼容**：结构上满足 PerceptionChannel Protocol（无需继承）：
    - name: str 属性（"audio"）。
    - async sense(signal=None) -> ModalityPrior | None（signal 有默认值=无参可调）。

    真接入（Wav2Small ONNX）挂载点：
        1. 装依赖：``uv pip install onnxruntime``（CPU）或 gpu-cuda/gpu-dml extras。
        2. 下载 Wav2Small ONNX 权重，路径写入 env ``ZERO_AUDIO_MODEL_PATH``。
        3. 在 _infer() 中加载 ONNX session，接收 float32 16kHz numpy 音频帧，
           输出 shape (3,) 或 (1,3)，字段序 **接线前必须核验**（gap-2）：
           若字段序为 [arousal, dominance, valence]，则 μv=out[2]*2-1, μa=out[0]*2-1。
        ⚠ gap-2：字段序确认前不要上线，否则 μv/μa 反转喂到 Zero 内核。
        4. 移除 _infer() 中的 stub return None，接入真推理结果。

    Args:
        signal_source: async callable → numpy 音频帧 (float32,16kHz) | None；
                       PerceptionHub 无参调 sense() 时由此获取信号。
    """

    name: str = "audio"

    def __init__(self, signal_source: Any | None = None) -> None:
        self.signal_source = signal_source
        self.model_path: str = os.getenv("ZERO_AUDIO_MODEL_PATH", "")

    async def sense(
        self,
        signal: Any | None = None,
    ) -> ModalityPrior | None:
        """async：从音频帧推理情感 VA，产出一条 ModalityPrior；stub 恒返回 None。

        Args:
            signal: float32 numpy 音频帧（16kHz）。None 时使用构造注入的 signal_source。

        Returns:
            stub：None。真接入后返回 ModalityPrior(modality="audio", ...)。

        Raises:
            不抛：I/O 异常（OSError/TimeoutError/RuntimeError/ValueError）
            均 warning+None 回退；编程错误（TypeError 等）上抛供 PerceptionHub 兜。
        """
        # 运行时读 env——感知构造后 env 变更即时生效（不在 __init__ 缓存）
        if os.getenv("ZERO_AUDIO_CHANNEL_ENABLED", "false").lower() != "true":
            return None

        raw: Any = signal
        if raw is None and self.signal_source is not None:
            try:
                raw = await self.signal_source()
            except (OSError, TimeoutError, RuntimeError, ValueError) as exc:
                logger.warning("AudioChannel signal_source 调用失败，本轮跳过: %s", exc)
                return None

        if raw is None:
            return None

        return self._infer(raw)

    def _infer(self, frame: Any) -> ModalityPrior | None:
        """真模型推理挂点（当前 stub，返回 None）。

        真接入步骤：
        1. ``import onnxruntime as ort``（延迟 import，ImportError → warning+None）。
        2. 加载 ``self.model_path`` ONNX session（构造时缓存，此处仅推理）。
        3. 预处理 frame（resample/pad/normalize to float32 16kHz）。
        4. 推理得 out shape (3,)；核验字段序（gap-2）：
               μv = float(out[2]) * 2.0 - 1.0   # valence [0,1]→[-1,1]
               μa = float(out[0]) * 2.0 - 1.0   # arousal [0,1]→[-1,1]
        5. return build_recommended_prior(modality=self.name, mu=(μv, μa),
                                          kind=ModalityKind.AUDIO)
        ⚠ gap-2：字段序 [arousal,dominance,valence] 接线前必须以真模型输出核验，
                 否则 μv/μa 反转；权重路径走 env ZERO_AUDIO_MODEL_PATH（不硬编码）。
        ⚠ 阻塞警告：ONNX/模型推理耗时 >1ms 时，须改为
                 ``await asyncio.to_thread(ort_session.run, ...)`` 在线程池执行，
                 不可在 async sense() 里塞同步阻塞推理（违反 python-code.md async 规则）。
        """
        # stub：真接入时替换此行
        return None
