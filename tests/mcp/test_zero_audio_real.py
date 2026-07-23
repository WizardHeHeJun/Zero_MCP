"""真 audeering w2v2 判别性 eval（对抗性核验 μa 判别力 + gap-2 不反转）。

gate：importorskip transformers/torch + 模型已缓存（local_files_only 探测）；缺任一自动 skip，
不阻断 CI。核心（handoff「验行为对不对」教训）：高能量音频的 μa 须显著高于安静音频；
若 μv/μa 反转（测成 valence），该判别 margin 不成立 → 断言失败 → 揪出 bug。

实测（deterministic，eval 模式无 dropout）：
  loud_white_noise μa≈+0.41 vs quiet_hum μa≈-0.24 → Δμa≈0.65；反转成 valence 仅 Δ≈0.23。
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")

from src.mcp.zero.channels.audio_channel import AudioChannel  # noqa: E402

_MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
_SR = 16000


def _model_cached() -> bool:
    """探测模型是否已在 HF 缓存（不触发网络下载）。"""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(_MODEL_ID, allow_patterns=["config.json"], local_files_only=True)
        return True
    except Exception:  # noqa: BLE001 — 任何缺失/网络问题都视为不可用 → skip
        return False


pytestmark = pytest.mark.skipif(
    not _model_cached(),
    reason=f"audeering 模型 {_MODEL_ID!r} 未在 HF 缓存（跳过真 eval，不阻断 CI）",
)


@pytest.fixture(autouse=True)
def _enable_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZERO_AUDIO_CHANNEL_ENABLED", "true")
    monkeypatch.setenv("ZERO_AUDIO_MODEL_PATH", _MODEL_ID)


def _loud_noise(dur: int = 3) -> np.ndarray:
    return (np.random.default_rng(0).standard_normal(_SR * dur) * 0.3).astype(np.float32)


def _quiet_hum(dur: int = 3) -> np.ndarray:
    t = np.arange(_SR * dur) / _SR
    return (0.02 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


class TestAudioChannelDiscriminability:
    """真模型：高能量音频 μa 显著高于安静音频（判别力 + gap-2 反转守卫）。"""

    async def test_loud_vs_quiet_arousal(self) -> None:
        ch = AudioChannel()
        loud = await ch.sense(signal=_loud_noise())
        quiet = await ch.sense(signal=_quiet_hum())

        assert loud is not None, "高能量信号未产出 ModalityPrior（模型路径异常）"
        assert quiet is not None, "安静信号未产出 ModalityPrior（模型路径异常）"

        print(
            f"\n[audio 判别性 eval]\n"
            f"  loud_noise  μv={loud.mu[0]:+.4f} μa={loud.mu[1]:+.4f}\n"
            f"  quiet_hum   μv={quiet.mu[0]:+.4f} μa={quiet.mu[1]:+.4f}\n"
            f"  Δμa = {loud.mu[1] - quiet.mu[1]:+.4f}"
        )

        # 合法性
        for r in (loud, quiet):
            assert -1.0 <= r.mu[0] <= 1.0 and -1.0 <= r.mu[1] <= 1.0
            assert r.modality == "audio"
            assert r.precision[1] > r.precision[0]  # audio Πa > Πv

        # 判别力 + gap-2 反转守卫：Δμa > 0.4（真≈0.65；若反转成 valence 仅≈0.23 → 失败）
        delta = loud.mu[1] - quiet.mu[1]
        assert delta > 0.4, (
            f"μa 判别力不足：loud μa={loud.mu[1]:.4f} 未显著大于 quiet μa={quiet.mu[1]:.4f}"
            f"（Δ={delta:.4f}≤0.4）。疑 μv/μa 反转（gap-2）或映射退化，回报审查。"
        )


class TestAudioChannelPriorValidity:
    """真信号经 AudioChannel → ModalityPrior 字段全部合法。"""

    async def test_real_signal_produces_valid_prior(self) -> None:
        ch = AudioChannel()
        result = await ch.sense(signal=_loud_noise())
        assert result is not None
        assert result.modality == "audio"
        assert -1.0 <= result.mu[0] <= 1.0
        assert -1.0 <= result.mu[1] <= 1.0
        assert result.precision[0] > 0.0 and result.precision[1] > 0.0
