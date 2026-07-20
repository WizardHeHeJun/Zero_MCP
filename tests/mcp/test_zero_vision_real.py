"""真 EmotiEffLib 判别性 eval（对抗性核验 μv 判别力 + gap-2-vision 列序不反转）。

gate：importorskip emotiefflib + cv2.FaceDetectorYN 可用 + 测试图/检测器可下载；缺任一自动 skip。
核心（handoff「验行为对不对」教训）：happy 脸的 μv（valence）须显著高于 angry/负性脸；
若列序反转（μv 取成 arousal），angry 脸 arousal 反而更高 → 该断言失败 → 揪出 bug。

实测（deterministic）：happy μv≈+0.68 vs angry μv≈-0.72 → Δμv≈1.40；反转成 arousal 则方向翻转。
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
from typing import Any

import numpy as np
import pytest

pytest.importorskip("emotiefflib")
import cv2  # noqa: E402

from src.mcp.zero.channels.vision_channel import VisionChannel  # noqa: E402

_IMG_URL = (
    "https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/main/"
    "tests/test_images/20180720_174416.jpg"
)
_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
_NEG_LABELS = ("Anger", "Sadness", "Fear", "Disgust", "Contempt")


def _download(url: str, suffix: str) -> str | None:
    try:
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        urllib.request.urlretrieve(url, path)
        return path
    except OSError:  # URLError 是 OSError 子类，覆盖网络/写盘失败
        return None


@pytest.fixture(scope="module")
def _labeled_crops() -> dict[str, list[np.ndarray]]:
    """下载测试图 + YuNet，检测人脸并用 recognizer 打标；任一步失败则 skip。"""
    if not hasattr(cv2, "FaceDetectorYN"):
        pytest.skip("cv2 无 FaceDetectorYN（无法检测人脸），跳过真视觉 eval")

    img_path = _download(_IMG_URL, ".jpg")
    yunet_path = _download(_YUNET_URL, ".onnx")
    if img_path is None or yunet_path is None:
        pytest.skip("测试图 / YuNet 下载失败（无网络），跳过真视觉 eval")

    frame_bgr = cv2.imread(img_path)
    if frame_bgr is None:
        pytest.skip("测试图读取失败，跳过真视觉 eval")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_bgr.shape[:2]

    try:
        det = cv2.FaceDetectorYN.create(yunet_path, "", (w, h), 0.6, 0.3, 5000)
        det.setInputSize((w, h))
        _, faces = det.detect(frame_bgr)
    except cv2.error as exc:  # pragma: no cover - 检测器初始化失败
        pytest.skip(f"YuNet 检测失败：{exc}")

    if faces is None or len(faces) < 2:
        pytest.skip("检测到人脸不足 2 张，无法做判别性对比")

    from emotiefflib.facial_analysis import EmotiEffLibRecognizer

    try:
        rec: Any = EmotiEffLibRecognizer(engine="onnx", model_name="enet_b0_8_va_mtl", device="cpu")
    except Exception as exc:  # noqa: BLE001 — 模型不可下载/加载 → skip
        pytest.skip(f"EmotiEffLib 模型不可用：{exc}")

    crops: dict[str, list[np.ndarray]] = {}
    for f in faces:
        x, y, bw, bh = (int(v) for v in f[:4])
        x, y = max(0, x), max(0, y)
        crop = frame_rgb[y : y + bh, x : x + bw, :]
        if crop.size == 0 or bw < 40 or bh < 40:
            continue
        label = rec.predict_emotions(crop, logits=True)[0][0]
        crops.setdefault(label, []).append(crop)

    has_happy = "Happiness" in crops
    has_neg = any(k in crops for k in _NEG_LABELS)
    if not (has_happy and has_neg):
        pytest.skip(f"未同时检测到 happy 与负性脸（labels={list(crops)}），跳过判别性断言")
    return crops


@pytest.fixture(autouse=True)
def _enable_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZERO_VISION_CHANNEL_ENABLED", "true")


async def _mu_v_for(crops: list[np.ndarray]) -> list[float]:
    ch = VisionChannel()
    out: list[float] = []
    for crop in crops:
        prior = await ch.sense(frame=crop)
        assert prior is not None, "真 RGB 人脸帧未产出 ModalityPrior（通道路径异常）"
        assert -1.0 <= prior.mu[0] <= 1.0 and -1.0 <= prior.mu[1] <= 1.0
        assert prior.modality == "vision"
        assert prior.precision[0] > prior.precision[1]  # face Πv > Πa
        out.append(prior.mu[0])  # μv
    return out


class TestVisionChannelDiscriminability:
    """真模型：happy 脸 μv 显著高于负性脸（判别力 + 列序反转守卫）。"""

    async def test_happy_valence_gt_negative(
        self, _labeled_crops: dict[str, list[np.ndarray]]
    ) -> None:
        happy_mu_v = await _mu_v_for(_labeled_crops["Happiness"])
        neg_crops: list[np.ndarray] = []
        for k in _NEG_LABELS:
            neg_crops.extend(_labeled_crops.get(k, []))
        neg_mu_v = await _mu_v_for(neg_crops)

        happy_mean = float(np.mean(happy_mu_v))
        neg_mean = float(np.mean(neg_mu_v))
        print(
            f"\n[vision 判别性 eval]\n"
            f"  happy μv = {happy_mu_v} (mean {happy_mean:+.4f})\n"
            f"  neg   μv = {neg_mu_v} (mean {neg_mean:+.4f})\n"
            f"  Δμv = {happy_mean - neg_mean:+.4f}"
        )
        # 判别力 + gap-2-vision 列序守卫：happy μv 显著高于负性（真≈1.4；反转成 arousal 会翻向）
        assert happy_mean > neg_mean + 0.3, (
            f"μv 判别力不足：happy μv={happy_mean:.4f} 未显著大于 neg μv={neg_mean:.4f}。"
            "疑 VA 列序反转（μv 取成 arousal）或映射退化，回报审查。"
        )
