"""AudioChannel 单测（mock 模型，脱真库/权重依赖）。

核心守卫：gap-2 字段序映射（out=[arousal,dominance,valence] → μa=out[0], μv=out[2]，**不反转**）。
覆盖：正常路径映射/clip、flag 守卫、无信号/signal_source、缺库优雅回退、Protocol 符合、
      模态命名/精度非对称（audio Πa>Πv）、未配置权重回退。
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import torch

from src.agents.models.zero_affect import ModalityPrior
from src.mcp.zero.channels.audio_channel import AudioChannel
from src.mcp.zero.external_priors import is_physio_stream
from src.mcp.zero.perception import PerceptionChannel, PerceptionHub

_LOGGER = "src.mcp.zero.channels.audio_channel"


def _fake_processor(samples: Any, sampling_rate: int) -> dict[str, list[np.ndarray]]:
    """mock Wav2Vec2Processor：原样返回 input_values（不改数值）。"""
    return {"input_values": [np.asarray(samples, dtype=np.float32)]}


def _make_fake_loader(aro: float, dom: float, val: float) -> Any:
    """构造 _load_audeering_model 替身：model(tensor) → torch.tensor([[aro,dom,val]])。"""

    def _fake_model(_tensor: Any) -> Any:
        return torch.tensor([[aro, dom, val]], dtype=torch.float32)

    def _loader(_model_id: str, _device: str) -> tuple[Any, Any]:
        return _fake_processor, _fake_model

    return _loader


def _audio_signal(n: int = 16000) -> np.ndarray:
    return np.random.default_rng(0).standard_normal(n).astype(np.float32)


@pytest.fixture()
def _enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZERO_AUDIO_CHANNEL_ENABLED", "true")
    monkeypatch.setenv("ZERO_AUDIO_MODEL_PATH", "dummy/model-id")


# ---------------------------------------------------------------------------
# gap-2 映射核心守卫
# ---------------------------------------------------------------------------


class TestAudioChannelGap2Mapping:
    """gap-2：字段序 [arousal,dominance,valence]，μa=out[0]、μv=out[2]，绝不反转。"""

    async def test_high_arousal_low_valence(self, _enabled: None) -> None:
        """out=[aro=0.9,dom=0.5,val=0.1] → μa=+0.8, μv=-0.8（valence 取 out[2] 非 out[0]）。"""
        ch = AudioChannel()
        with patch(f"{_LOGGER}._load_audeering_model", _make_fake_loader(0.9, 0.5, 0.1)):
            result = await ch.sense(signal=_audio_signal())
        assert result is not None
        assert result.mu[1] == pytest.approx(0.8)  # μa from arousal(out[0])
        assert result.mu[0] == pytest.approx(-0.8)  # μv from valence(out[2])

    async def test_low_arousal_high_valence(self, _enabled: None) -> None:
        """out=[aro=0.1,dom=0.5,val=0.9] → μa=-0.8, μv=+0.8（反转会让此断言失败）。"""
        ch = AudioChannel()
        with patch(f"{_LOGGER}._load_audeering_model", _make_fake_loader(0.1, 0.5, 0.9)):
            result = await ch.sense(signal=_audio_signal())
        assert result is not None
        assert result.mu[1] == pytest.approx(-0.8)
        assert result.mu[0] == pytest.approx(0.8)

    async def test_out_of_range_clipped(self, _enabled: None) -> None:
        """越界值 clip(0,1) 后映射：aro=1.5→μa=1.0，val=-0.3→μv=-1.0（不抛 ValidationError）。"""
        ch = AudioChannel()
        with patch(f"{_LOGGER}._load_audeering_model", _make_fake_loader(1.5, 0.5, -0.3)):
            result = await ch.sense(signal=_audio_signal())
        assert result is not None
        assert result.mu[1] == pytest.approx(1.0)
        assert result.mu[0] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# 契约：模态命名 / 精度非对称 / 非生理流
# ---------------------------------------------------------------------------


class TestAudioChannelContract:
    async def test_modality_and_precision(self, _enabled: None) -> None:
        """modality="audio"；精度 audio Πa>Πv（0.25>0.10，唤醒信噪比高于效价）。"""
        ch = AudioChannel()
        with patch(f"{_LOGGER}._load_audeering_model", _make_fake_loader(0.5, 0.5, 0.5)):
            result = await ch.sense(signal=_audio_signal())
        assert result is not None
        assert result.modality == "audio"
        assert result.precision[0] > 0.0 and result.precision[1] > 0.0
        assert result.precision[1] > result.precision[0]  # Πa > Πv

    def test_audio_is_not_physio_stream(self) -> None:
        """audio 不是生理流（不触发 Zero M2 效价精度覆写）。"""
        assert not is_physio_stream(AudioChannel.name)

    def test_satisfies_perception_channel(self) -> None:
        assert isinstance(AudioChannel(), PerceptionChannel)


# ---------------------------------------------------------------------------
# flag / 无信号 / 未配置权重
# ---------------------------------------------------------------------------


class TestAudioChannelGuards:
    async def test_disabled_by_default_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ZERO_AUDIO_CHANNEL_ENABLED", raising=False)
        ch = AudioChannel()
        assert await ch.sense(signal=_audio_signal()) is None

    async def test_explicitly_disabled_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ZERO_AUDIO_CHANNEL_ENABLED", "false")
        ch = AudioChannel()
        assert await ch.sense(signal=_audio_signal()) is None

    async def test_no_signal_no_source_returns_none(self, _enabled: None) -> None:
        ch = AudioChannel()
        assert await ch.sense(signal=None) is None

    async def test_no_model_path_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """flag 开但 ZERO_AUDIO_MODEL_PATH 空 → warning + None（未配置权重优雅回退）。"""
        monkeypatch.setenv("ZERO_AUDIO_CHANNEL_ENABLED", "true")
        monkeypatch.delenv("ZERO_AUDIO_MODEL_PATH", raising=False)
        ch = AudioChannel()
        assert await ch.sense(signal=_audio_signal()) is None

    async def test_signal_source_async_provides_signal(self, _enabled: None) -> None:
        source = AsyncMock(return_value=_audio_signal())
        ch = AudioChannel(signal_source=source)
        with patch(f"{_LOGGER}._load_audeering_model", _make_fake_loader(0.5, 0.5, 0.5)):
            result = await ch.sense()
        assert isinstance(result, ModalityPrior)
        source.assert_awaited_once()

    async def test_signal_source_exception_returns_none(self, _enabled: None, caplog: Any) -> None:
        source = AsyncMock(side_effect=RuntimeError("mic 超时"))
        ch = AudioChannel(signal_source=source)
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            result = await ch.sense()
        assert result is None
        assert "signal_source" in caplog.text


# ---------------------------------------------------------------------------
# 优雅回退：缺库 / 推理失败
# ---------------------------------------------------------------------------


class TestAudioChannelGracefulFallback:
    async def test_import_error_returns_none(self, _enabled: None, caplog: Any) -> None:
        """加载抛 ImportError（缺 transformers/torch）→ None + warning。"""
        ch = AudioChannel()
        with patch(f"{_LOGGER}._load_audeering_model", side_effect=ImportError("no transformers")):
            with caplog.at_level(logging.WARNING, logger=_LOGGER):
                result = await ch.sense(signal=_audio_signal())
        assert result is None
        assert "transformers" in caplog.text.lower() or "import" in caplog.text.lower()

    async def test_runtime_error_returns_none(self, _enabled: None, caplog: Any) -> None:
        """推理抛 RuntimeError → None + warning（不拖垮 Hub）。"""
        ch = AudioChannel()
        with patch(f"{_LOGGER}._load_audeering_model", side_effect=RuntimeError("bad weights")):
            with caplog.at_level(logging.WARNING, logger=_LOGGER):
                result = await ch.sense(signal=_audio_signal())
        assert result is None

    async def test_bad_signal_shape_returns_none(self, _enabled: None) -> None:
        """3D 输入无法降为单声道 → None（不抛）。"""
        ch = AudioChannel()
        bad = np.zeros((2, 2, 2), dtype=np.float32)
        with patch(f"{_LOGGER}._load_audeering_model", _make_fake_loader(0.5, 0.5, 0.5)):
            result = await ch.sense(signal=bad)
        assert result is None


# ---------------------------------------------------------------------------
# PerceptionHub 集成：异常通道跳过，audio 先验保留（AD-3）
# ---------------------------------------------------------------------------


class TestAudioChannelInHub:
    async def test_prior_preserved_when_other_channel_raises(self, _enabled: None) -> None:
        from unittest.mock import MagicMock

        audio = AudioChannel(signal_source=AsyncMock(return_value=_audio_signal()))
        bad: Any = MagicMock()
        bad.name = "bad"
        bad.sense = AsyncMock(side_effect=RuntimeError("设备故障"))
        hub = PerceptionHub([bad, audio])
        with patch(f"{_LOGGER}._load_audeering_model", _make_fake_loader(0.6, 0.5, 0.4)):
            priors = await hub.collect()
        assert len(priors) == 1
        assert priors[0].modality == "audio"
