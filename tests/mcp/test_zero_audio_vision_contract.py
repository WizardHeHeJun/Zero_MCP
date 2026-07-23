"""AudioChannel / VisionChannel 基础契约单测（跨通道，脱模型）。

audio/vision 已从 stub 升级为真接入（audeering w2v2 / EmotiEffLib ONNX）；本文件只覆盖
**不依赖真模型**的基础契约：flag 守卫、Protocol 符合、name 属性、无输入回退、Hub 跳过。
真模型映射/判别性见 test_zero_audio_channel.py / test_zero_vision_channel.py（mock）与
test_zero_{audio,vision}_real.py（真模型判别性 eval）。

覆盖：
  1. flag 关（默认/显式 false）→ sense None（两通道）。
  2. isinstance(PerceptionChannel) True（runtime_checkable）。
  3. 无输入（signal/frame=None 且无 signal_source）→ sense None。
  4. name 属性正确（"audio" / "vision"）。
  5. 注册 PerceptionHub：flag 关 → collect 跳过，返回空列表。
"""

from __future__ import annotations

import pytest

from src.mcp.zero.channels.audio_channel import AudioChannel
from src.mcp.zero.channels.vision_channel import VisionChannel
from src.mcp.zero.perception import PerceptionChannel, PerceptionHub

# ---------------------------------------------------------------------------
# 1. flag 关（默认/显式 false）→ sense None
# ---------------------------------------------------------------------------


class TestChannelDisabledByDefault:
    """flag 未设或 false → sense 返回 None（两通道都覆盖，零回归守卫）。"""

    async def test_audio_disabled_by_default_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AudioChannel 未设 ZERO_AUDIO_CHANNEL_ENABLED → sense None。"""
        monkeypatch.delenv("ZERO_AUDIO_CHANNEL_ENABLED", raising=False)
        assert await AudioChannel().sense() is None

    async def test_audio_explicitly_disabled_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ZERO_AUDIO_CHANNEL_ENABLED=false → sense None。"""
        monkeypatch.setenv("ZERO_AUDIO_CHANNEL_ENABLED", "false")
        assert await AudioChannel().sense() is None

    async def test_vision_disabled_by_default_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """VisionChannel 未设 ZERO_VISION_CHANNEL_ENABLED → sense None。"""
        monkeypatch.delenv("ZERO_VISION_CHANNEL_ENABLED", raising=False)
        assert await VisionChannel().sense() is None

    async def test_vision_explicitly_disabled_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ZERO_VISION_CHANNEL_ENABLED=false → sense None。"""
        monkeypatch.setenv("ZERO_VISION_CHANNEL_ENABLED", "false")
        assert await VisionChannel().sense() is None


# ---------------------------------------------------------------------------
# 2. isinstance(PerceptionChannel) True（runtime_checkable 协议符合性）
# ---------------------------------------------------------------------------


class TestChannelProtocolCompliance:
    """AudioChannel / VisionChannel 满足 PerceptionChannel Protocol。"""

    def test_audio_channel_isinstance_perception_channel(self) -> None:
        assert isinstance(AudioChannel(), PerceptionChannel)

    def test_vision_channel_isinstance_perception_channel(self) -> None:
        assert isinstance(VisionChannel(), PerceptionChannel)

    def test_audio_channel_has_sense_method(self) -> None:
        assert callable(AudioChannel().sense)

    def test_vision_channel_has_sense_method(self) -> None:
        assert callable(VisionChannel().sense)


# ---------------------------------------------------------------------------
# 3. flag 开但无输入（signal/frame=None 且无 signal_source）→ sense None
# ---------------------------------------------------------------------------


class TestChannelEnabledNoInput:
    """flag 开启但无输入 → sense None（无证据本轮跳过，不触发模型加载）。"""

    async def test_audio_enabled_no_signal_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ZERO_AUDIO_CHANNEL_ENABLED", "true")
        assert await AudioChannel().sense(signal=None) is None

    async def test_vision_enabled_no_frame_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ZERO_VISION_CHANNEL_ENABLED", "true")
        assert await VisionChannel().sense(frame=None) is None

    async def test_audio_enabled_no_model_path_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """flag=true + 传入 signal，但未配置 ZERO_AUDIO_MODEL_PATH → None（未配置权重优雅回退）。"""
        monkeypatch.setenv("ZERO_AUDIO_CHANNEL_ENABLED", "true")
        monkeypatch.delenv("ZERO_AUDIO_MODEL_PATH", raising=False)
        import numpy as np

        assert await AudioChannel().sense(signal=np.zeros(16000, dtype=np.float32)) is None


# ---------------------------------------------------------------------------
# 4. name 属性正确
# ---------------------------------------------------------------------------


class TestChannelNameAttribute:
    """AudioChannel / VisionChannel name 属性按协议定义正确。"""

    def test_audio_channel_name(self) -> None:
        assert AudioChannel().name == "audio"

    def test_vision_channel_name(self) -> None:
        assert VisionChannel().name == "vision"


# ---------------------------------------------------------------------------
# 5. 注册 PerceptionHub → flag 关时 collect 跳过，返回空列表
# ---------------------------------------------------------------------------


class TestChannelInPerceptionHub:
    """flag 关的通道注册 PerceptionHub，sense 返回 None → collect 跳过，先验列表为空。"""

    async def test_audio_disabled_skipped_in_hub_collect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ZERO_AUDIO_CHANNEL_ENABLED", raising=False)
        priors = await PerceptionHub([AudioChannel()]).collect()
        assert priors == []

    async def test_vision_disabled_skipped_in_hub_collect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ZERO_VISION_CHANNEL_ENABLED", raising=False)
        priors = await PerceptionHub([VisionChannel()]).collect()
        assert priors == []

    async def test_both_disabled_in_hub_both_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ZERO_AUDIO_CHANNEL_ENABLED", raising=False)
        monkeypatch.delenv("ZERO_VISION_CHANNEL_ENABLED", raising=False)
        priors = await PerceptionHub([AudioChannel(), VisionChannel()]).collect()
        assert priors == []
