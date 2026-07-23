"""语音感知通道（真接入）—— audio → ModalityPrior。

模型 = audeering ``wav2vec2-large-robust-12-ft-emotion-msp-dim``（维度 SER，Wav2Small 教师）。
文献门首选的 Wav2Small 蒸馏 ONNX（72K 参数）在 HF 上**无可下载权重**（dkounadis/wav2small
仅模型卡），故落回文献门 option B 已核验的 audeering 全精度模型（同字段序，见下 gap-2）。
选型与核验依据见 notes/2026-07-20-zero-link-audio-vision-real-integration.md。

设计依据（文献门纪要 notes/2026-07-16-zero-link-perception-litreview.md）：
- [audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim HuggingFace] ·
  [Wagner 2023 arXiv:2203.07378]
  模型输出 shape (1,3) = **[arousal, dominance, valence]**，值域约 [0,1]（gap-2 已现场核验，
  见模型卡 print 语句 `Arousal=pred[0,0] Dominance=pred[0,1] Valence=pred[0,2]`）。
  → μa = out[0]*2-1, μv = out[2]*2-1（clip 到 [0,1] 后再映射，防越界）。
- [Wagner 2023 arXiv:2203.07378] 纯声学 valence 内在弱 → audio Πv=0.10 < Πa=0.25。

真接入注意（gap 已闭环）：
- gap-2（字段序）：**已核验 [arousal, dominance, valence]**，μv/μa 不反转（见上）。
- transformers 5.x 兼容：模型卡代码用旧 API ``self.init_weights()`` 在 transformers 5.x
  下缺 ``all_tied_weights_keys`` 崩溃；改用 ``self.post_init()``（现场核验修复）。
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

# 模型期望的采样率（Hz）。Channel 契约约定输入即 16kHz 单声道 float32；
# 重采样/加载 wav 属独立 I/O 适配层职责，不进 Channel 核心（脱硬件可单测）。
_TARGET_SAMPLE_RATE: int = 16000


def _load_audeering_model(model_id: str, device: str) -> tuple[Any, Any]:
    """延迟 import transformers/torch，定义 audeering 回归头并加载模型（阻塞，走线程池）。

    audeering 维度 SER 模型 = Wav2Vec2 backbone + 自定义回归头（dense+tanh+out_proj），
    输出 (batch,3) = [arousal, dominance, valence]。模型卡的 ``self.init_weights()`` 在
    transformers 5.x 下会因缺 ``all_tied_weights_keys`` 崩溃 → 改 ``self.post_init()``。

    参考：[audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim HuggingFace]。
    """
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
    )

    class _RegressionHead(nn.Module):
        """audeering 情感回归头（dense → tanh → out_proj）。"""

        def __init__(self, config: Any) -> None:
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features: Any) -> Any:
            x = self.dropout(features)
            x = torch.tanh(self.dense(x))
            x = self.dropout(x)
            return self.out_proj(x)

    class _EmotionModel(Wav2Vec2PreTrainedModel):  # type: ignore[misc]
        """Wav2Vec2 维度情感回归模型（backbone 均值池化 → 回归头）。"""

        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = _RegressionHead(config)
            # transformers 5.x：post_init 注册 all_tied_weights_keys（旧卡 init_weights 会崩）
            self.post_init()

        def forward(self, input_values: Any) -> Any:
            hidden = torch.mean(self.wav2vec2(input_values)[0], dim=1)
            return self.classifier(hidden)

    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = _EmotionModel.from_pretrained(model_id).to(device).eval()
    return processor, model


class AudioChannel:
    """语音感知通道（真接入 audeering w2v2 维度 SER）。

    **Protocol 兼容**：结构上满足 PerceptionChannel Protocol（无需继承）：
    - name: str 属性（"audio"）。
    - async sense(signal=None) -> ModalityPrior | None（signal 有默认值=无参可调）。

    模型经 ``ZERO_AUDIO_MODEL_PATH`` 指定（HF 模型 id 或本地快照目录；空=不启用真推理，
    优雅回退 None）。推理阻塞（CPU ~200ms/句）→ 走 ``asyncio.to_thread``，不堵事件循环。
    缺库/缺模型/推理失败/无输入 → warning + None（feature flag 默认关，零回归）。

    Args:
        signal_source: async callable → numpy 音频帧 (float32,16kHz,mono) | None；
                       PerceptionHub 无参调 sense() 时由此获取信号。
        device: torch 设备（默认 "cpu"；本环境 torch 为 CPU 构建）。
    """

    name: str = "audio"

    def __init__(self, signal_source: Any | None = None, device: str = "cpu") -> None:
        self.signal_source = signal_source
        self.device = device
        # 延迟加载缓存（首次推理时加载，线程锁防并发 collect 双载）
        self.processor: Any | None = None
        self.model: Any | None = None
        self.model_load_lock = threading.Lock()

    async def sense(
        self,
        signal: Any | None = None,
    ) -> ModalityPrior | None:
        """async：从音频帧推理情感 VA，产出一条 ModalityPrior；无证据/未配置则返回 None。

        Args:
            signal: float32 numpy 音频帧（16kHz，单声道）。None 时使用构造注入的 signal_source。

        Returns:
            ModalityPrior(modality="audio", mu=(μv,μa), precision=(0.10,0.25)) 或 None。

        Raises:
            不抛：I/O/推理异常（OSError/TimeoutError/RuntimeError/ValueError/ImportError）
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

        # 阻塞推理走线程池，不堵事件循环（python-code.md async 规则）
        return await asyncio.to_thread(self._infer, raw)

    def _ensure_model(self) -> tuple[Any, Any] | None:
        """延迟加载并缓存模型；模型 id 走 env ``ZERO_AUDIO_MODEL_PATH``（空=未配置返回 None）。"""
        if self.model is not None and self.processor is not None:
            return self.processor, self.model
        model_id = os.getenv("ZERO_AUDIO_MODEL_PATH", "").strip()
        if not model_id:
            logger.warning("AudioChannel: 未配置 ZERO_AUDIO_MODEL_PATH，跳过真推理")
            return None
        with self.model_load_lock:
            if self.model is None or self.processor is None:
                self.processor, self.model = _load_audeering_model(model_id, self.device)
        return self.processor, self.model

    def _infer(self, frame: Any) -> ModalityPrior | None:
        """真模型推理（阻塞，由 sense() 经 asyncio.to_thread 调度）。

        流程：coerce → processor 归一 → model → out=[aro,dom,val]（~[0,1]）→ clip(0,1)*2-1。
        gap-2 已核验字段序 [arousal,dominance,valence]：μa=out[0], μv=out[2]（不反转）。
        """
        try:
            loaded = self._ensure_model()
            if loaded is None:
                return None
            processor, model = loaded
            import torch  # 延迟 import（与 _load_audeering_model 一致；ImportError 下方捕获）

            samples = self._coerce_signal(frame)
            if samples is None:
                return None

            inputs = processor(samples, sampling_rate=_TARGET_SAMPLE_RATE)["input_values"][0]
            tensor = torch.from_numpy(np.asarray(inputs, dtype=np.float32).reshape(1, -1))
            with torch.no_grad():
                out = model(tensor.to(self.device)).detach().cpu().numpy()[0]

            # gap-2：out = [arousal, dominance, valence]，值域 ~[0,1] → clip 后线性映射 [-1,1]
            arousal01 = float(np.clip(out[0], 0.0, 1.0))
            valence01 = float(np.clip(out[2], 0.0, 1.0))
            mu_a = arousal01 * 2.0 - 1.0
            mu_v = valence01 * 2.0 - 1.0
            return build_recommended_prior(
                modality=self.name,
                mu=(mu_v, mu_a),
                kind=ModalityKind.AUDIO,
            )
        except ImportError as exc:
            logger.warning("AudioChannel: transformers/torch 不可用，本轮跳过: %s", exc)
            return None
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("AudioChannel 推理失败，本轮跳过: %s", exc)
            return None

    @staticmethod
    def _coerce_signal(frame: Any) -> Any | None:
        """将输入规整为 1D float32 numpy（多声道取均值降为单声道）；非法则 None。"""
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=1) if arr.shape[1] <= arr.shape[0] else arr.mean(axis=0)
        if arr.ndim != 1 or arr.size == 0:
            logger.warning("AudioChannel: 音频帧形状非法（需 1D/2D 非空），跳过")
            return None
        return arr
