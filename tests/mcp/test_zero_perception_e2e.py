"""端到端联调：真三模态感知通道 → ModalityPrior → ZeroLinkClient.step → D:\\Zero 真内核。

标 `@pytest.mark.zerorepo`：需 D:\\Zero 的 `src/mcp_server` 在位，且在共用 conda 环境
affective-expression 内跑（server 子进程要 import Zero + mcp + langgraph）。

验证「真感知模型 → 真 ModalityPrior → external_priors 载荷（M3/M6 通过）→ Zero 竞争融合
→ expression」**整条闭环第一次真跑通**：
- physio（EdaChannel/HrvChannel + NeuroKit2 合成信号）**恒可跑**（importorskip neurokit2）。
- audio（AudioChannel + audeering w2v2）：模型已缓存才纳入，否则通道 graceful 回退 None、Hub 跳过。
- vision（VisionChannel + EmotiEffLib）：模型 + 人脸帧可得才纳入，否则 Hub 跳过。

与 test_zero_client_e2e.py 的区别：那里 priors 是手搓的；这里是**真感知模型跑出来的**，
经真实 PerceptionHub.collect() 汇聚，再走 client 注入真 server——闭环两端都真。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest

nk = pytest.importorskip("neurokit2")

from src.agents.models.zero_affect import (  # noqa: E402
    FACS_KEYS_EXT,
    TEXT_LABELS,
    AffectStimulus,
    ExpressionBundle,
)
from src.mcp.zero.channels.audio_channel import AudioChannel  # noqa: E402
from src.mcp.zero.channels.physio_channel import EdaChannel, HrvChannel  # noqa: E402
from src.mcp.zero.channels.vision_channel import VisionChannel  # noqa: E402
from src.mcp.zero.client import ZeroLinkClient, ZeroLinkConnectionError  # noqa: E402
from src.mcp.zero.external_priors import build_external_priors_override  # noqa: E402
from src.mcp.zero.perception import PerceptionHub  # noqa: E402

_ZERO_SERVER = Path("D:/Zero/src/mcp_server/server.py")
_AUDIO_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
_VALID_FACS = frozenset(FACS_KEYS_EXT)
_IMG_URL = (
    "https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/main/"
    "tests/test_images/20180720_174416.jpg"
)
_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)


def _set_stdio_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """设 stdio 联调 env（对齐 test_zero_client_e2e.py：本测试进程即在目标环境）。"""
    monkeypatch.setenv("ZERO_LINK_ENABLED", "true")
    monkeypatch.setenv("ZERO_LINK_TRANSPORT", "stdio")
    monkeypatch.setenv("ZERO_SERVER_COMMAND", sys.executable)
    monkeypatch.setenv("ZERO_SERVER_ARGS", '["-m", "src.mcp_server"]')
    monkeypatch.setenv("ZERO_SERVER_CWD", r"D:\Zero")


def _assert_valid_head(head: Any, ctx: str) -> None:
    """对抗核验 ExpressionHead：facs_au 键 ⊆ 全集、值 [0,1]、text_label 合法。"""
    unknown = set(head.facs_au) - _VALID_FACS
    assert not unknown, f"{ctx}: facs_au 含未知键 {unknown}"
    for key, val in head.facs_au.items():
        assert 0.0 <= val <= 1.0, f"{ctx}: facs_au[{key}]={val} 超出 [0,1]"
    assert head.text_label in TEXT_LABELS, f"{ctx}: text_label={head.text_label!r} 非法"


def _audio_cached() -> bool:
    """audeering 模型是否已在 HF 缓存（不触发下载）。"""
    if importlib.util.find_spec("transformers") is None:
        return False
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(_AUDIO_MODEL, allow_patterns=["config.json"], local_files_only=True)
        return True
    except Exception:  # noqa: BLE001 — 任何缺失即视为不可用
        return False


def _download(url: str, suffix: str) -> str | None:
    try:
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        urllib.request.urlretrieve(url, path)
        return path
    except OSError:
        return None


def _face_crop() -> np.ndarray | None:
    """尽力取一张 RGB 人脸裁剪帧（EmotiEffLib + cv2.FaceDetectorYN + 测试图）；失败 None。"""
    if importlib.util.find_spec("emotiefflib") is None:
        return None
    try:
        import cv2
    except ImportError:
        return None
    if not hasattr(cv2, "FaceDetectorYN"):
        return None
    img_path = _download(_IMG_URL, ".jpg")
    yunet_path = _download(_YUNET_URL, ".onnx")
    if img_path is None or yunet_path is None:
        return None
    frame_bgr = cv2.imread(img_path)
    if frame_bgr is None:
        return None
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_bgr.shape[:2]
    try:
        det = cv2.FaceDetectorYN.create(yunet_path, "", (w, h), 0.6, 0.3, 5000)
        det.setInputSize((w, h))
        _, faces = det.detect(frame_bgr)
    except cv2.error:
        return None
    if faces is None or len(faces) == 0:
        return None
    x, y, bw, bh = (int(v) for v in faces[0][:4])
    x, y = max(0, x), max(0, y)
    crop = frame_rgb[y : y + bh, x : x + bw, :]
    return crop if crop.size and bw >= 40 and bh >= 40 else None


@pytest.mark.zerorepo
class TestPerceptionToZeroE2E:
    """真三模态感知 → Zero 真内核端到端。缺 server / 非目标环境 → skip。"""

    async def test_real_multimodal_priors_to_zero_kernel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if not _ZERO_SERVER.is_file():
            pytest.skip(f"D:\\Zero MCP server 不存在（{_ZERO_SERVER}），跳过感知端到端")

        # --- 开各感知通道 flag（audio/vision 仅在模型可得时开）---
        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        audio_ok = _audio_cached()
        if audio_ok:
            monkeypatch.setenv("ZERO_AUDIO_CHANNEL_ENABLED", "true")
            monkeypatch.setenv("ZERO_AUDIO_MODEL_PATH", _AUDIO_MODEL)
        vision_crop = _face_crop()
        if vision_crop is not None:
            monkeypatch.setenv("ZERO_VISION_CHANNEL_ENABLED", "true")

        # --- 真信号（physio 用 nk 合成高唤醒；audio 用高能量噪声）---
        eda_signal = {
            "eda": nk.eda_simulate(duration=60, sampling_rate=4, scr_number=9, random_state=42),
            "sampling_rate": 4,
        }
        ecg_signal = {
            "ecg_or_ppg": nk.ecg_simulate(
                duration=30, sampling_rate=256, heart_rate=110, random_state=42
            ),
            "sampling_rate": 256,
        }
        loud_audio = (np.random.default_rng(0).standard_normal(16000 * 3) * 0.3).astype(np.float32)

        async def _eda_src() -> dict[str, Any]:
            return eda_signal

        async def _ecg_src() -> dict[str, Any]:
            return ecg_signal

        async def _audio_src() -> np.ndarray:
            return loud_audio

        async def _vision_src() -> np.ndarray | None:
            return vision_crop

        # --- 真 PerceptionHub.collect()：各通道跑真模型出真先验，不可用者 graceful 跳过 ---
        channels: list[Any] = [
            EdaChannel(sampling_rate=4, signal_source=_eda_src),
            HrvChannel(sampling_rate=256, signal_source=_ecg_src),
        ]
        if audio_ok:
            channels.append(AudioChannel(signal_source=_audio_src))
        if vision_crop is not None:
            channels.append(VisionChannel(signal_source=_vision_src))

        hub = PerceptionHub(channels)
        priors = await hub.collect()
        modalities = {p.modality for p in priors}

        prior_lines = "\n".join(
            f"  {p.modality:10s} μ=({p.mu[0]:+.3f},{p.mu[1]:+.3f}) Π={p.precision}" for p in priors
        )
        print(
            f"\n[感知→Zero E2E] 真收集先验 {len(priors)} 条：\n{prior_lines}"
            f"\n  audio_cached={audio_ok} vision_face={vision_crop is not None}"
        )

        # physio 两流恒在（真判别信号）；每条都是真模型输出的合法先验
        assert "eda/sc" in modalities, "EdaChannel 未产出真先验（physio 路径异常）"
        assert "hrv/rmssd" in modalities, "HrvChannel 未产出真先验"
        for p in priors:
            assert -1.0 <= p.mu[0] <= 1.0 and -1.0 <= p.mu[1] <= 1.0
            assert p.precision[0] > 0.0 and p.precision[1] > 0.0
        if audio_ok:
            assert "audio" in modalities, "audeering 已缓存但 AudioChannel 未产出先验"
        if vision_crop is not None:
            assert "vision" in modalities, "有人脸帧但 VisionChannel 未产出先验"

        # 真先验构造 external_priors 载荷：M3 精度上界 / M6 流数上界 必须通过（真值合法）
        payload = build_external_priors_override(priors)
        assert payload["external_priors"], "external_priors 载荷不应为空"
        assert len(payload["external_priors"]) == len(priors)

        # --- 经 client 注入 D:\Zero 真 server，验证内核消费真先验并出 expression ---
        _set_stdio_env(monkeypatch)
        client = ZeroLinkClient()
        try:
            await client.__aenter__()
        except ZeroLinkConnectionError as exc:
            pytest.skip(f"连不上 Zero server（是否在 affective-expression 环境？）：{exc}")
        try:
            sid = await client.open_session(persona="perception-e2e")
            assert isinstance(sid, str) and sid

            # 带真感知先验 step → Zero 竞争融合 → 合法 ExpressionBundle
            bundle = await client.step(sid, AffectStimulus(valence=0.2, arousal=0.3), priors=priors)
            assert isinstance(bundle, ExpressionBundle)
            assert len(bundle.valence_arousal) == 2
            _assert_valid_head(bundle.spontaneous, "感知E2E.spontaneous")
            _assert_valid_head(bundle.voluntary, "感知E2E.voluntary")
            assert bundle.prosody_scale in ("ratio", "normalized", None)
            fused_va = tuple(round(x, 3) for x in bundle.valence_arousal)
            print(
                f"[感知→Zero E2E] 内核融合后 (v,a)={fused_va} "
                f"text_label={bundle.voluntary.text_label!r}"
            )

            await client.close_session(sid)
        finally:
            await client.__aexit__(None, None, None)
