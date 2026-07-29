"""io_adapters 单测（T3）—— 脱硬件、零真设备。

覆盖：
A1. audio_file_source：soundfile 造临时 WAV → callable 返回 float32 1D 非空。
A2. audio_file_source：删文件后调 → None。
A3. audio_file_source：不存在路径 → None（FileNotFoundError 优雅回退）。
V1. vision_file_source：PIL 造临时图；yunet=None, fallback="whole_image" → 非 None RGB ndarray。
V2. vision_file_source：yunet=None, fallback="none" → None。
V3. vision_file_source：不存在路径 → None。
V4. vision_file_source（FaceDetectorYN）：skipif 缺属性；真 yunet=None 路径可跑。
P1. make_synthetic_eda_source：importorskip neurokit2 → callable
    → dict 含 "eda" ndarray + "sampling_rate" int。
P2. make_synthetic_hrv_source：importorskip neurokit2 → callable
    → dict 含 "ecg_or_ppg" ndarray + "sampling_rate" int。
P3. 注入冒烟 EDA：make_synthetic_eda_source 喂 EdaChannel(signal_source=...)
    → sense() 非 None。
P4. 注入冒烟 audio：make_audio_file_source 喂 AudioChannel(signal_source=...)
    → sense() 非 None（mock librosa + _load_audeering_model）。
S1. 硬件桩：make_mic_source() → NotImplementedError。
S2. 硬件桩：make_camera_source() → NotImplementedError。
S3. 硬件桩：make_wearable_source() → NotImplementedError。
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import soundfile as sf

from src.mcp.zero.io_adapters import (
    make_audio_file_source,
    make_camera_source,
    make_mic_source,
    make_synthetic_eda_source,
    make_synthetic_hrv_source,
    make_vision_file_source,
    make_wearable_source,
)

# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

_LOGGER_AUDIO = "src.mcp.zero.io_adapters.audio_file_adapter"
_LOGGER_VISION = "src.mcp.zero.io_adapters.vision_file_adapter"


def _make_tmp_wav(n_samples: int = 16000, sr: int = 16000) -> str:
    """用 soundfile 造一个临时 WAV 文件，返回路径。"""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    data = np.zeros(n_samples, dtype=np.float32)
    sf.write(path, data, sr)
    return path


def _make_tmp_png(width: int = 64, height: int = 64) -> str:
    """用 PIL 造一个临时 RGB PNG 文件，返回路径。"""
    from PIL import Image

    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img = Image.fromarray(
        np.zeros((height, width, 3), dtype=np.uint8),
        mode="RGB",
    )
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# A1-A3：audio_file_source
# ---------------------------------------------------------------------------


class TestAudioFileSource:
    """audio_file_source：从临时 WAV 加载，脱真 librosa（使用 soundfile 造测试文件）。"""

    async def test_valid_wav_returns_float32_1d(self) -> None:
        """A1：soundfile 造临时 WAV → callable 返回 float32 1D 非空 ndarray。"""
        path = _make_tmp_wav()
        try:
            source = make_audio_file_source(path)
            result = await source()
        finally:
            os.unlink(path)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.dtype == np.float32
        assert result.size > 0

    async def test_deleted_file_returns_none(self, caplog: Any) -> None:
        """A2：删文件后调 → None（FileNotFoundError 优雅回退）。"""
        path = _make_tmp_wav()
        os.unlink(path)  # 先删掉

        import logging

        source = make_audio_file_source(path)
        with caplog.at_level(logging.WARNING, logger=_LOGGER_AUDIO):
            result = await source()

        assert result is None

    async def test_nonexistent_path_returns_none(self, caplog: Any) -> None:
        """A3：不存在路径 → None + warning。"""
        import logging

        source = make_audio_file_source("/nonexistent/path/audio.wav")
        with caplog.at_level(logging.WARNING, logger=_LOGGER_AUDIO):
            result = await source()

        assert result is None


# ---------------------------------------------------------------------------
# V1-V4：vision_file_source
# ---------------------------------------------------------------------------


class TestVisionFileSource:
    """vision_file_source：从临时 PNG 读取，yunet=None 模式脱真检测器。"""

    async def test_whole_image_fallback_returns_rgb_ndarray(self) -> None:
        """V1：yunet=None, fallback="whole_image" → 非 None RGB uint8 ndarray。"""
        path = _make_tmp_png(64, 64)
        try:
            source = make_vision_file_source(
                path, yunet_model_path=None, fallback_on_no_face="whole_image"
            )
            result = await source()
        finally:
            os.unlink(path)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.shape[2] == 3  # RGB
        assert result.dtype == np.uint8  # VisionChannel 期望 RGB uint8

    async def test_none_fallback_returns_none(self) -> None:
        """V2：yunet=None, fallback="none" → None。"""
        path = _make_tmp_png(64, 64)
        try:
            source = make_vision_file_source(
                path, yunet_model_path=None, fallback_on_no_face="none"
            )
            result = await source()
        finally:
            os.unlink(path)

        assert result is None

    async def test_nonexistent_path_returns_none(self, caplog: Any) -> None:
        """V3：不存在路径 → None + warning。"""
        import logging

        source = make_vision_file_source(
            "/nonexistent/path/image.png",
            yunet_model_path=None,
            fallback_on_no_face="whole_image",
        )
        with caplog.at_level(logging.WARNING, logger=_LOGGER_VISION):
            result = await source()

        assert result is None

    @pytest.mark.skipif(
        not hasattr(cv2, "FaceDetectorYN"),
        reason="cv2 无 FaceDetectorYN，跳过人脸检测路径测试",
    )
    async def test_yunet_path_none_still_works_with_facedetectoryn_available(self) -> None:
        """V4（FaceDetectorYN 可用时）：yunet=None 仍走 fallback 路径，不崩。"""
        path = _make_tmp_png(64, 64)
        try:
            # yunet=None → 跳过检测，走 fallback
            source = make_vision_file_source(
                path,
                yunet_model_path=None,
                fallback_on_no_face="whole_image",
            )
            result = await source()
        finally:
            os.unlink(path)

        assert result is not None
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# P1-P2：physio_synthetic_adapter（缺 neurokit2 仅跳过本组，不波及 audio/vision/桩，W2）
# ---------------------------------------------------------------------------

_SKIP_NO_NK = pytest.mark.skipif(
    importlib.util.find_spec("neurokit2") is None,
    reason="需 neurokit2（合成生理信号）",
)


@_SKIP_NO_NK
class TestSyntheticEdaSource:
    """make_synthetic_eda_source：需 neurokit2 → 验证输出 dict 形状。"""

    async def test_returns_dict_with_eda_and_sampling_rate(self) -> None:
        """P1：callable → dict 含 "eda" ndarray + "sampling_rate" int。"""
        source = make_synthetic_eda_source(
            duration=5, sampling_rate=4, scr_number=2, random_state=0
        )
        result = await source()

        assert result is not None
        assert isinstance(result, dict)
        assert "eda" in result
        assert "sampling_rate" in result
        assert isinstance(result["eda"], np.ndarray)
        assert result["eda"].size > 0
        assert isinstance(result["sampling_rate"], int)
        assert result["sampling_rate"] == 4

    async def test_multiple_calls_return_same_data(self) -> None:
        """合成工厂预生成一次 ndarray，多次调用返回相同对象。"""
        source = make_synthetic_eda_source(duration=5, sampling_rate=4, random_state=42)
        r1 = await source()
        r2 = await source()
        assert r1 is r2  # 同一 dict 对象（闭包捕获）


@_SKIP_NO_NK
class TestSyntheticHrvSource:
    """make_synthetic_hrv_source：需 neurokit2 → 验证输出 dict 形状。"""

    async def test_returns_dict_with_ecg_and_sampling_rate(self) -> None:
        """P2：callable → dict 含 "ecg_or_ppg" ndarray + "sampling_rate" int。"""
        source = make_synthetic_hrv_source(
            duration=10, sampling_rate=256, heart_rate=70, random_state=0
        )
        result = await source()

        assert result is not None
        assert isinstance(result, dict)
        assert "ecg_or_ppg" in result
        assert "sampling_rate" in result
        assert isinstance(result["ecg_or_ppg"], np.ndarray)
        assert result["ecg_or_ppg"].size > 0
        assert isinstance(result["sampling_rate"], int)
        assert result["sampling_rate"] == 256

    async def test_multiple_calls_return_same_data(self) -> None:
        """合成工厂预生成一次 ndarray，多次调用返回相同对象。"""
        source = make_synthetic_hrv_source(duration=10, sampling_rate=256, random_state=99)
        r1 = await source()
        r2 = await source()
        assert r1 is r2


# ---------------------------------------------------------------------------
# P3：注入冒烟 EDA —— make_synthetic_eda_source → EdaChannel → sense() 非 None
# ---------------------------------------------------------------------------


@_SKIP_NO_NK
class TestSyntheticEdaInjectionSmoke:
    """P3：make_synthetic_eda_source 喂 EdaChannel(signal_source=...) → sense() 非 None。

    mock neurokit2 复用 test_zero_physio_channel.py 的 _make_nk_mock 模式，
    monkeypatch ZERO_PHYSIO_CHANNEL_ENABLED=true。
    """

    async def test_eda_source_injected_into_eda_channel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """合成 EDA source 注入 EdaChannel，攒够基线后 sense() 产出 ModalityPrior 非 None。"""
        from src.mcp.zero.channels.physio_channel import EdaChannel

        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")

        # 构造合成 EDA source（工厂同步调 nk.eda_simulate 已在模块顶部 importorskip，可用）
        eda_source = make_synthetic_eda_source(
            duration=5, sampling_rate=4, scr_number=3, random_state=0
        )

        # 注入时钟：通道按**真实秒数**判冷启动，注入后无需真等待即可跨过双门
        now = [0.0]
        ch = EdaChannel(sampling_rate=4, signal_source=eda_source, clock=lambda: now[0])
        for _ in range(3):
            await ch.sense()
            now[0] += 200.0
        result = await ch.sense()

        assert result is not None
        assert result.modality == "eda/sc"
        assert -1.0 <= result.mu[1] <= 1.0

    async def test_eda_source_reaches_channel_on_cold_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """首窗返 None 是**冷启动**而非接线断——两者用基线历史区分。

        判别性：若接线断（source 未被调用/返回 None），`baseline_history` 会**保持为空**；
        接线通则首窗虽返 None，历史里已落一条。
        """
        from src.mcp.zero.channels.physio_channel import EdaChannel

        monkeypatch.setenv("ZERO_PHYSIO_CHANNEL_ENABLED", "true")
        eda_source = make_synthetic_eda_source(
            duration=5, sampling_rate=4, scr_number=3, random_state=0
        )
        ch = EdaChannel(sampling_rate=4, signal_source=eda_source)

        assert await ch.sense() is None  # 冷启动：无基线证据
        assert len(ch.baseline_history) == 1, "signal_source 未被取到（接线断），非冷启动"


# ---------------------------------------------------------------------------
# P4：注入冒烟 audio —— make_audio_file_source → AudioChannel → sense() 非 None
# ---------------------------------------------------------------------------


class TestAudioFileInjectionSmoke:
    """P4：make_audio_file_source 喂 AudioChannel → sense() 非 None（mock 链路）。

    mock librosa.load（绕过真文件 I/O）+ mock _load_audeering_model（绕过真模型）。
    monkeypatch ZERO_AUDIO_CHANNEL_ENABLED=true + ZERO_AUDIO_MODEL_PATH=dummy。
    """

    async def test_audio_source_injected_into_audio_channel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """audio_file_source 注入 AudioChannel，channel.sense() 产出 ModalityPrior 非 None。"""
        import torch

        from src.mcp.zero.channels.audio_channel import AudioChannel

        monkeypatch.setenv("ZERO_AUDIO_CHANNEL_ENABLED", "true")
        monkeypatch.setenv("ZERO_AUDIO_MODEL_PATH", "dummy/model-id")

        # 造临时 WAV（librosa 被 mock 拦截；文件须存在，否则 source 走 FileNotFoundError）
        path = _make_tmp_wav(n_samples=16000)

        def _fake_processor(samples: Any, sampling_rate: int) -> dict[str, list[np.ndarray]]:
            return {"input_values": [np.asarray(samples, dtype=np.float32)]}

        def _fake_model(_tensor: Any) -> Any:
            # 返回 [arousal=0.7, dominance=0.5, valence=0.6]
            return torch.tensor([[0.7, 0.5, 0.6]], dtype=torch.float32)

        def _fake_loader(_model_id: str, _device: str) -> tuple[Any, Any]:
            return _fake_processor, _fake_model

        # mock librosa.load → 直接返回零信号（不真读文件）
        fake_librosa = MagicMock()
        fake_librosa.load.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        try:
            audio_source = make_audio_file_source(path)

            with patch.dict("sys.modules", {"librosa": fake_librosa}):
                # 先调一次 source()，验证它能正常返回信号（librosa 被 mock）
                signal = await audio_source()

            assert signal is not None
            assert signal.dtype == np.float32

            # 再把 source 注入 AudioChannel，验证注入链路
            ch = AudioChannel(signal_source=audio_source)
            _audio_logger = "src.mcp.zero.channels.audio_channel"
            with patch(f"{_audio_logger}._load_audeering_model", _fake_loader):
                with patch.dict("sys.modules", {"librosa": fake_librosa}):
                    result = await ch.sense()

            assert result is not None
            assert result.modality == "audio"
            assert -1.0 <= result.mu[0] <= 1.0
            assert -1.0 <= result.mu[1] <= 1.0

        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# P5：注入冒烟 vision —— make_vision_file_source → VisionChannel → sense() 非 None
# ---------------------------------------------------------------------------


class TestVisionFileInjectionSmoke:
    """P5：make_vision_file_source 喂 VisionChannel(signal_source=...) → sense() 非 None。

    yunet=None + fallback="whole_image" → source 产整图 RGB 帧；mock _load_recognizer 返回
    is_mtl 识别器（predict_emotions 出末两列 [valence,arousal]）。monkeypatch
    ZERO_VISION_CHANNEL_ENABLED=true。**不依赖 neurokit2/FaceDetectorYN**（验 W2 修复：
    vision 冒烟应在缺 neurokit2 时也能跑）。
    """

    async def test_vision_source_injected_into_vision_channel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """合成 vision source 注入 VisionChannel，channel.sense() 产出 ModalityPrior 非 None。"""
        from src.mcp.zero.channels.vision_channel import VisionChannel

        monkeypatch.setenv("ZERO_VISION_CHANNEL_ENABLED", "true")

        path = _make_tmp_png(64, 64)
        try:
            vision_source = make_vision_file_source(
                path, yunet_model_path=None, fallback_on_no_face="whole_image"
            )
            # 先验证 source 产帧（整图 RGB 回退，无 FaceDetectorYN 依赖）
            frame = await vision_source()
            assert frame is not None

            # mock recognizer：is_mtl=True，scores 末两列 = [valence=0.6, arousal=0.5]
            fake_rec = MagicMock()
            fake_rec.is_mtl = True
            fake_rec.predict_emotions.return_value = (
                ["happy"],
                np.array([[0.1] * 8 + [0.6, 0.5]], dtype=np.float32),
            )

            def _fake_load_recognizer(_model_name: str, _model_dir: str, _device: str) -> Any:
                return fake_rec

            ch = VisionChannel(signal_source=vision_source)
            _vision_mod = "src.mcp.zero.channels.vision_channel"
            with patch(f"{_vision_mod}._load_recognizer", _fake_load_recognizer):
                result = await ch.sense()

            assert result is not None
            assert result.modality == "vision"
            assert -1.0 <= result.mu[0] <= 1.0
            assert -1.0 <= result.mu[1] <= 1.0
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# H1-H8：真硬件适配器（mic / camera / wearable）
#
# 核心契约（脱设备可测的关键）：**构造工厂永不抛**；缺库 / 无设备 / 读取失败一律由返回的
# callable 产 None，交 Channel 走既有「本轮无证据」优雅回退。故本组全部在无硬件环境下运行。
# ---------------------------------------------------------------------------


def _hide_module(name: str) -> Any:
    """patch.dict 上下文：让 `import <name>` 抛 ImportError（模拟缺库）。"""
    return patch.dict("sys.modules", {name: None})


class TestHardwareAdapters:
    """真硬件信号源工厂：构造不抛、缺库/失败优雅回退 None、成功路径形状正确。"""

    def test_factories_never_raise_on_construction(self) -> None:
        """H1：三个工厂在无任何硬件/库的环境下**构造均不抛**（替代旧的 NotImplementedError 桩）。"""
        assert callable(make_mic_source())
        assert callable(make_camera_source())
        assert callable(make_wearable_source())

    async def test_mic_missing_lib_returns_none(self) -> None:
        """H2：缺 sounddevice → 返回 None（不抛），Channel 侧按无证据处理。"""
        source = make_mic_source()
        with _hide_module("sounddevice"):
            assert await source() is None

    async def test_mic_success_shape(self) -> None:
        """H3：录音成功 → 1-D float32 帧，长度 = sample_rate × chunk_duration_s。"""
        fake_sd = MagicMock()
        fake_sd.rec.return_value = np.zeros((800, 1), dtype=np.float32)
        source = make_mic_source(sample_rate=400, chunk_duration_s=2.0)
        with patch.dict("sys.modules", {"sounddevice": fake_sd}):
            frame = await source()
        assert frame is not None
        assert frame.dtype == np.float32 and frame.ndim == 1 and frame.size == 800
        assert fake_sd.rec.call_args.args[0] == 800  # frames = 400×2.0
        fake_sd.wait.assert_called_once()  # 必须等录制完成，否则拿到未填充缓冲

    async def test_mic_device_error_returns_none(self) -> None:
        """H4：sounddevice 抛库内异常（无输入设备等）→ None，不外泄异常。"""
        fake_sd = MagicMock()
        fake_sd.rec.side_effect = RuntimeError("PortAudio: no input device")
        source = make_mic_source()
        with patch.dict("sys.modules", {"sounddevice": fake_sd}):
            assert await source() is None

    async def test_camera_cannot_open_returns_none(self) -> None:
        """H5：摄像头打不开 → None，且**必须 release**（否则设备被长期占用）。"""
        fake_cv2 = MagicMock()
        cap = MagicMock()
        cap.isOpened.return_value = False
        fake_cv2.VideoCapture.return_value = cap
        source = make_camera_source(device=3)
        with patch.dict("sys.modules", {"cv2": fake_cv2}):
            assert await source() is None
        cap.release.assert_called_once()

    async def test_camera_success_returns_rgb_uint8(self) -> None:
        """H6：抓帧成功 → RGB uint8 帧；无 YuNet 路径时返回整帧且已 BGR→RGB。"""
        fake_cv2 = MagicMock()
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, np.zeros((4, 5, 3), dtype=np.uint8))
        fake_cv2.VideoCapture.return_value = cap
        fake_cv2.cvtColor.return_value = np.ones((4, 5, 3), dtype=np.uint8)
        source = make_camera_source()
        with patch.dict("sys.modules", {"cv2": fake_cv2}):
            frame = await source()
        assert frame is not None and frame.dtype == np.uint8 and frame.shape == (4, 5, 3)
        fake_cv2.cvtColor.assert_called()  # BGR→RGB 转换必须发生
        cap.release.assert_called_once()

    async def test_wearable_missing_lib_returns_none(self) -> None:
        """H7：缺 pyserial → None（不抛）。"""
        source = make_wearable_source()
        with _hide_module("serial"):
            assert await source() is None

    async def test_wearable_insufficient_samples_returns_none(self) -> None:
        """H8：样本不足 → None（宁可判无证据，也不产会让 RMSSD 不可靠的短信号）。

        判别性：同一 mock 下把 duration 调到样本够用即应成功——证明 None 来自「样本不足」
        这一条判定，而非串口路径整体不通。
        """
        fake_serial = MagicMock()
        port = MagicMock()
        # 只吐 5 个样本后返回空字节（设备停止输出）
        port.readline.side_effect = [b"0.1\n", b"0.2\n", b"0.3\n", b"0.4\n", b"0.5\n", b""]
        fake_serial.Serial.return_value.__enter__.return_value = port
        fake_mod = MagicMock()
        fake_mod.Serial = fake_serial.Serial
        fake_tools = MagicMock()
        fake_tools.comports.return_value = [MagicMock(device="COM9")]

        mods = {
            "serial": fake_mod,
            "serial.tools": MagicMock(),
            "serial.tools.list_ports": fake_tools,
        }
        fake_mod.tools = MagicMock()

        # 需要 10 个样本但只有 5 个 → None
        source = make_wearable_source(port="COM9", sampling_rate=10, duration_s=1.0)
        with patch.dict("sys.modules", mods):
            with patch("serial.tools.list_ports", fake_tools, create=True):
                assert await source() is None

        # 判别性对照：需要 5 个样本 → 成功且形状正确
        port.readline.side_effect = [b"0.1\n", b"0.2\n", b"0.3\n", b"0.4\n", b"0.5\n"]
        source_ok = make_wearable_source(port="COM9", sampling_rate=5, duration_s=1.0)
        with patch.dict("sys.modules", mods):
            with patch("serial.tools.list_ports", fake_tools, create=True):
                result = await source_ok()
        assert result is not None
        assert result["sampling_rate"] == 5
        assert result["ecg_or_ppg"].shape == (5,)
