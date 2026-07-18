"""AudioChannel / VisionChannel stub 单测（Task I）。

覆盖：
  1. AudioChannel：flag 关（默认）→ sense None。
  2. VisionChannel：flag 关（默认）→ sense None。
  3. AudioChannel isinstance(PerceptionChannel) True（runtime_checkable）。
  4. VisionChannel isinstance(PerceptionChannel) True。
  5. 注册 PerceptionHub：stub 通道 sense 返回 None → collect 跳过，返回空列表。
  6. stub 通道 flag 开启但无输入 → sense None（stub _infer 恒 None）。
  7. AudioChannel / VisionChannel name 属性正确。
"""

from __future__ import annotations

import pytest

from src.mcp.zero.channels.audio_channel import AudioChannel
from src.mcp.zero.channels.vision_channel import VisionChannel
from src.mcp.zero.perception import PerceptionChannel, PerceptionHub

# ---------------------------------------------------------------------------
# 1-2. flag 关（默认）→ sense None
# ---------------------------------------------------------------------------


class TestStubChannelDisabledByDefault:
    """flag 未设或 false → sense 返回 None（两个 stub 通道都覆盖）。"""

    async def test_audio_disabled_by_default_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AudioChannel 未设 ZERO_AUDIO_CHANNEL_ENABLED → sense None。"""
        monkeypatch.delenv("ZERO_AUDIO_CHANNEL_ENABLED", raising=False)
        ch = AudioChannel()
        result = await ch.sense()
        assert result is None

    async def test_audio_explicitly_disabled_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ZERO_AUDIO_CHANNEL_ENABLED=false → sense None。"""
        monkeypatch.setenv("ZERO_AUDIO_CHANNEL_ENABLED", "false")
        ch = AudioChannel()
        result = await ch.sense()
        assert result is None

    async def test_vision_disabled_by_default_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """VisionChannel 未设 ZERO_VISION_CHANNEL_ENABLED → sense None。"""
        monkeypatch.delenv("ZERO_VISION_CHANNEL_ENABLED", raising=False)
        ch = VisionChannel()
        result = await ch.sense()
        assert result is None

    async def test_vision_explicitly_disabled_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ZERO_VISION_CHANNEL_ENABLED=false → sense None。"""
        monkeypatch.setenv("ZERO_VISION_CHANNEL_ENABLED", "false")
        ch = VisionChannel()
        result = await ch.sense()
        assert result is None


# ---------------------------------------------------------------------------
# 3-4. isinstance(PerceptionChannel) True（runtime_checkable 协议符合性）
# ---------------------------------------------------------------------------


class TestStubChannelProtocolCompliance:
    """AudioChannel / VisionChannel 满足 PerceptionChannel Protocol。"""

    def test_audio_channel_isinstance_perception_channel(self) -> None:
        """AudioChannel isinstance(PerceptionChannel) True。"""
        ch = AudioChannel()
        assert isinstance(ch, PerceptionChannel)

    def test_vision_channel_isinstance_perception_channel(self) -> None:
        """VisionChannel isinstance(PerceptionChannel) True。"""
        ch = VisionChannel()
        assert isinstance(ch, PerceptionChannel)

    def test_audio_channel_has_sense_method(self) -> None:
        """AudioChannel 具有 sense 方法。"""
        ch = AudioChannel()
        assert callable(ch.sense)

    def test_vision_channel_has_sense_method(self) -> None:
        """VisionChannel 具有 sense 方法。"""
        ch = VisionChannel()
        assert callable(ch.sense)


# ---------------------------------------------------------------------------
# 5. 注册 PerceptionHub → collect 跳过 None，返回空列表
# ---------------------------------------------------------------------------


class TestStubChannelInPerceptionHub:
    """stub 通道注册 PerceptionHub，sense 返回 None → collect 跳过，先验列表为空。"""

    async def test_audio_stub_skipped_in_hub_collect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AudioChannel（flag 关）注册 Hub → collect 结果不含任何先验。"""
        monkeypatch.delenv("ZERO_AUDIO_CHANNEL_ENABLED", raising=False)
        hub = PerceptionHub([AudioChannel()])
        priors = await hub.collect()
        assert priors == []

    async def test_vision_stub_skipped_in_hub_collect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """VisionChannel（flag 关）注册 Hub → collect 结果不含任何先验。"""
        monkeypatch.delenv("ZERO_VISION_CHANNEL_ENABLED", raising=False)
        hub = PerceptionHub([VisionChannel()])
        priors = await hub.collect()
        assert priors == []

    async def test_both_stubs_in_hub_both_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AudioChannel + VisionChannel 同时注册 Hub，全部跳过 → collect 返回空列表。"""
        monkeypatch.delenv("ZERO_AUDIO_CHANNEL_ENABLED", raising=False)
        monkeypatch.delenv("ZERO_VISION_CHANNEL_ENABLED", raising=False)
        hub = PerceptionHub([AudioChannel(), VisionChannel()])
        priors = await hub.collect()
        assert priors == []


# ---------------------------------------------------------------------------
# 6. flag 开但无输入 → stub _infer 恒 None → sense None
# ---------------------------------------------------------------------------


class TestStubChannelEnabledNoInput:
    """flag 开启但无输入（signal/frame=None 且无 signal_source）→ sense None。

    stub 通道 _infer 恒返回 None（真接入前的占位），即使 flag 开启也无产出。
    """

    async def test_audio_enabled_no_signal_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AudioChannel flag=true 但 signal=None 且无 signal_source → None。"""
        monkeypatch.setenv("ZERO_AUDIO_CHANNEL_ENABLED", "true")
        ch = AudioChannel()
        result = await ch.sense(signal=None)
        assert result is None

    async def test_vision_enabled_no_frame_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """VisionChannel flag=true 但 frame=None 且无 signal_source → None。"""
        monkeypatch.setenv("ZERO_VISION_CHANNEL_ENABLED", "true")
        ch = VisionChannel()
        result = await ch.sense(frame=None)
        assert result is None

    async def test_audio_enabled_with_signal_infer_still_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AudioChannel flag=true，传入 signal → _infer stub 返回 None（真接入前占位）。"""
        monkeypatch.setenv("ZERO_AUDIO_CHANNEL_ENABLED", "true")
        import numpy as np

        ch = AudioChannel()
        fake_frame = np.zeros(16000, dtype=np.float32)
        result = await ch.sense(signal=fake_frame)
        assert result is None

    async def test_vision_enabled_with_frame_infer_still_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """VisionChannel flag=true，传入 frame → _infer stub 返回 None（真接入前占位）。"""
        monkeypatch.setenv("ZERO_VISION_CHANNEL_ENABLED", "true")
        import numpy as np

        ch = VisionChannel()
        fake_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = await ch.sense(frame=fake_frame)
        assert result is None


# ---------------------------------------------------------------------------
# 7. name 属性正确
# ---------------------------------------------------------------------------


class TestStubChannelNameAttribute:
    """AudioChannel / VisionChannel name 属性按协议定义正确。"""

    def test_audio_channel_name(self) -> None:
        """AudioChannel.name == "audio"。"""
        ch = AudioChannel()
        assert ch.name == "audio"

    def test_vision_channel_name(self) -> None:
        """VisionChannel.name == "vision"。"""
        ch = VisionChannel()
        assert ch.name == "vision"
