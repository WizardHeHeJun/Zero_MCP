"""VisionChannel 单测（mock recognizer，脱真库/权重依赖）。

核心守卫：VA 列序映射（EmotiEffLib va_mtl scores 末两列 = [valence, arousal]
→ μv=scores[-2], μa=scores[-1]，**不反转**；值域已 [-1,1] 无需 *2-1）。
覆盖：正常映射/clip、非 mtl 模型回退、flag 守卫、无帧/signal_source、缺库回退、
      Protocol 符合、模态命名/精度非对称（face Πv>Πa）、非生理流。
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.agents.models.zero_affect import ModalityPrior
from src.mcp.zero.channels.vision_channel import VisionChannel
from src.mcp.zero.external_priors import is_physio_stream
from src.mcp.zero.perception import PerceptionChannel, PerceptionHub

_LOGGER = "src.mcp.zero.channels.vision_channel"


def _make_fake_recognizer(valence: float, arousal: float, is_mtl: bool = True) -> MagicMock:
    """构造 recognizer 替身：predict_emotions → (labels, scores)，末两列 [valence, arousal]。"""
    rec = MagicMock()
    rec.is_mtl = is_mtl
    # 8 情感 logits + [valence, arousal]（模拟 enet_b0_8_va_mtl 的 (1,10) 输出）
    scores = np.array(
        [[0.1, 0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.3, valence, arousal]], dtype=np.float32
    )
    rec.predict_emotions.return_value = (["Happiness"], scores)
    return rec


def _make_loader(rec: MagicMock) -> Any:
    def _loader(_name: str, _dir: str, _device: str) -> MagicMock:
        return rec

    return _loader


def _rgb_face(size: int = 224) -> np.ndarray:
    return (np.random.default_rng(0).random((size, size, 3)) * 255).astype(np.uint8)


@pytest.fixture()
def _enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZERO_VISION_CHANNEL_ENABLED", "true")


# ---------------------------------------------------------------------------
# VA 列序核心守卫
# ---------------------------------------------------------------------------


class TestVisionChannelColumnOrder:
    """gap-2-vision：末两列 = [valence, arousal]，μv=scores[-2]、μa=scores[-1]，绝不反转。"""

    async def test_positive_valence_low_arousal(self, _enabled: None) -> None:
        """scores[-2]=+0.7(valence), scores[-1]=+0.2(arousal) → μv=+0.7, μa=+0.2。"""
        ch = VisionChannel()
        with patch(f"{_LOGGER}._load_recognizer", _make_loader(_make_fake_recognizer(0.7, 0.2))):
            result = await ch.sense(frame=_rgb_face())
        assert result is not None
        assert result.mu[0] == pytest.approx(0.7)  # μv from scores[-2]
        assert result.mu[1] == pytest.approx(0.2)  # μa from scores[-1]

    async def test_negative_valence_high_arousal(self, _enabled: None) -> None:
        """scores[-2]=-0.7(val), scores[-1]=+0.6(aro) → μv=-0.7, μa=+0.6（angry 语义）。"""
        ch = VisionChannel()
        with patch(f"{_LOGGER}._load_recognizer", _make_loader(_make_fake_recognizer(-0.7, 0.6))):
            result = await ch.sense(frame=_rgb_face())
        assert result is not None
        assert result.mu[0] == pytest.approx(-0.7)
        assert result.mu[1] == pytest.approx(0.6)

    async def test_out_of_range_clipped(self, _enabled: None) -> None:
        """越界 clip 到 [-1,1]：val=1.3→μv=1.0，aro=-1.4→μa=-1.0（不抛 ValidationError）。"""
        ch = VisionChannel()
        with patch(f"{_LOGGER}._load_recognizer", _make_loader(_make_fake_recognizer(1.3, -1.4))):
            result = await ch.sense(frame=_rgb_face())
        assert result is not None
        assert result.mu[0] == pytest.approx(1.0)
        assert result.mu[1] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# 契约：模态命名 / 精度非对称 / 非生理流 / kind=FACE
# ---------------------------------------------------------------------------


class TestVisionChannelContract:
    async def test_modality_and_precision(self, _enabled: None) -> None:
        """modality="vision"；精度 face Πv>Πa（0.20>0.12，效价信噪比高于唤醒）。"""
        ch = VisionChannel()
        with patch(f"{_LOGGER}._load_recognizer", _make_loader(_make_fake_recognizer(0.3, 0.1))):
            result = await ch.sense(frame=_rgb_face())
        assert result is not None
        assert result.modality == "vision"
        assert result.precision[0] > 0.0 and result.precision[1] > 0.0
        assert result.precision[0] > result.precision[1]  # Πv > Πa

    def test_vision_is_not_physio_stream(self) -> None:
        assert not is_physio_stream(VisionChannel.name)

    def test_satisfies_perception_channel(self) -> None:
        assert isinstance(VisionChannel(), PerceptionChannel)


# ---------------------------------------------------------------------------
# flag / 无帧 / 非 mtl 模型 / signal_source
# ---------------------------------------------------------------------------


class TestVisionChannelGuards:
    async def test_disabled_by_default_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ZERO_VISION_CHANNEL_ENABLED", raising=False)
        ch = VisionChannel()
        assert await ch.sense(frame=_rgb_face()) is None

    async def test_no_frame_no_source_returns_none(self, _enabled: None) -> None:
        ch = VisionChannel()
        assert await ch.sense(frame=None) is None

    async def test_non_mtl_model_returns_none(self, _enabled: None, caplog: Any) -> None:
        """非 *_va_mtl 模型（is_mtl=False）无 VA 输出 → warning + None。"""
        ch = VisionChannel()
        rec = _make_fake_recognizer(0.5, 0.5, is_mtl=False)
        with patch(f"{_LOGGER}._load_recognizer", _make_loader(rec)):
            with caplog.at_level(logging.WARNING, logger=_LOGGER):
                result = await ch.sense(frame=_rgb_face())
        assert result is None
        assert "va_mtl" in caplog.text or "VA" in caplog.text

    async def test_non_mtl_model_not_reloaded_each_call(self, _enabled: None) -> None:
        """非 *_va_mtl 判定不可用后缓存哨兵，后续 sense() 不重复加载昂贵模型（WARN 修复回归）。"""
        ch = VisionChannel()
        loader = MagicMock(return_value=_make_fake_recognizer(0.5, 0.5, is_mtl=False))
        with patch(f"{_LOGGER}._load_recognizer", loader):
            assert await ch.sense(frame=_rgb_face()) is None
            assert await ch.sense(frame=_rgb_face()) is None
        assert loader.call_count == 1  # 第二次走哨兵，不再加载

    async def test_signal_source_async_provides_frame(self, _enabled: None) -> None:
        source = AsyncMock(return_value=_rgb_face())
        ch = VisionChannel(signal_source=source)
        with patch(f"{_LOGGER}._load_recognizer", _make_loader(_make_fake_recognizer(0.4, 0.3))):
            result = await ch.sense()
        assert isinstance(result, ModalityPrior)
        source.assert_awaited_once()

    async def test_signal_source_exception_returns_none(self, _enabled: None, caplog: Any) -> None:
        source = AsyncMock(side_effect=RuntimeError("camera 超时"))
        ch = VisionChannel(signal_source=source)
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            result = await ch.sense()
        assert result is None
        assert "signal_source" in caplog.text


# ---------------------------------------------------------------------------
# 优雅回退：缺库 / 推理失败
# ---------------------------------------------------------------------------


class TestVisionChannelGracefulFallback:
    async def test_import_error_returns_none(self, _enabled: None, caplog: Any) -> None:
        ch = VisionChannel()
        with patch(f"{_LOGGER}._load_recognizer", side_effect=ImportError("no emotiefflib")):
            with caplog.at_level(logging.WARNING, logger=_LOGGER):
                result = await ch.sense(frame=_rgb_face())
        assert result is None
        assert "emotiefflib" in caplog.text.lower() or "import" in caplog.text.lower()

    async def test_runtime_error_returns_none(self, _enabled: None) -> None:
        ch = VisionChannel()
        with patch(f"{_LOGGER}._load_recognizer", side_effect=RuntimeError("bad onnx")):
            result = await ch.sense(frame=_rgb_face())
        assert result is None


# ---------------------------------------------------------------------------
# PerceptionHub 集成：异常通道跳过，vision 先验保留（AD-3）
# ---------------------------------------------------------------------------


class TestVisionChannelInHub:
    async def test_prior_preserved_when_other_channel_raises(self, _enabled: None) -> None:
        vision = VisionChannel(signal_source=AsyncMock(return_value=_rgb_face()))
        # spec 限定：裸 MagicMock 会自动伪造 prepare/reset 等可选协议方法，令 Hub 的鸭子类型
        # 检测误判（并 await 一个不可等待的 mock）。只暴露 Protocol 真实成员。
        bad: Any = MagicMock(spec=["name", "sense"])
        bad.name = "bad"
        bad.sense = AsyncMock(side_effect=RuntimeError("设备故障"))
        hub = PerceptionHub([bad, vision])
        with patch(f"{_LOGGER}._load_recognizer", _make_loader(_make_fake_recognizer(0.5, 0.4))):
            priors = await hub.collect()
        assert len(priors) == 1
        assert priors[0].modality == "vision"
