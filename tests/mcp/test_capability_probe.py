"""test_capability_probe.py — CapabilityFlags 探测逻辑单测。

覆盖：
- EP 列表 mock：CUDA/DML/CPU 三档 effective_device
- PERCEPTION_DEVICE=cpu 跳过 GPU 探测（omniparser=False、vram 不调用）
- rapidocr import 失败 → ocr=False
- mss import 失败 → mss_available=False
- 幂等缓存：probe_capabilities 两次调用返回同一对象
- cuda 强制但不可用 → 降级 auto → cpu
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as _numpy_preload  # noqa: F401  — 预加载 numpy，确保 patch.dict 不将其从 sys.modules 移除
import pytest

import src.mcp.desktop.capability_probe as probe_mod
from src.mcp.desktop.capability_probe import CapabilityFlags, probe_capabilities


def _reset_cache() -> None:
    """每个测试前清空模块级缓存，保证幂等测试可控。"""
    probe_mod._CACHED_FLAGS = None


@pytest.fixture(autouse=True)
def clear_probe_cache() -> None:
    """autouse fixture：每次测试前后均清空缓存。"""
    _reset_cache()
    yield
    _reset_cache()


def _fake_ort(providers: list[str]) -> MagicMock:
    """构造一个 fake onnxruntime 模块，get_available_providers 返回给定列表。"""
    fake = MagicMock()
    fake.get_available_providers.return_value = providers
    return fake


# ── EP 探测 ───────────────────────────────────────────────────────────────────


def test_auto_cuda_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    """auto 模式下 CUDA 优先于 DML。"""
    monkeypatch.setenv("PERCEPTION_DEVICE", "auto")
    monkeypatch.delenv("OMNIPARSER_MODEL_DIR", raising=False)
    fake_ort = _fake_ort(["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"])
    with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
        with patch.object(probe_mod, "_probe_gpu_vram_mb", return_value=None):
            flags = probe_mod._do_probe()
    assert flags.effective_device == "cuda"
    assert flags.cuda_accel is True
    assert flags.dml_accel is False


def test_auto_dml_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """auto 模式：无 CUDA 时选 DML。"""
    monkeypatch.setenv("PERCEPTION_DEVICE", "auto")
    monkeypatch.delenv("OMNIPARSER_MODEL_DIR", raising=False)
    fake_ort = _fake_ort(["DmlExecutionProvider", "CPUExecutionProvider"])
    with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
        with patch.object(probe_mod, "_probe_gpu_vram_mb", return_value=None):
            flags = probe_mod._do_probe()
    assert flags.effective_device == "dml"
    assert flags.cuda_accel is False
    assert flags.dml_accel is True


def test_auto_cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """auto 模式：仅 CPU EP 时 effective_device=cpu。"""
    monkeypatch.setenv("PERCEPTION_DEVICE", "auto")
    fake_ort = _fake_ort(["CPUExecutionProvider"])
    with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
        flags = probe_mod._do_probe()
    assert flags.effective_device == "cpu"
    assert flags.cuda_accel is False
    assert flags.dml_accel is False


def test_force_cpu_skips_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """PERCEPTION_DEVICE=cpu 时跳过 GPU 探测，omniparser=False。"""
    monkeypatch.setenv("PERCEPTION_DEVICE", "cpu")
    fake_ort = _fake_ort(["CUDAExecutionProvider", "CPUExecutionProvider"])
    with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
        with patch.object(probe_mod, "_probe_gpu_vram_mb") as mock_vram:
            flags = probe_mod._do_probe()
    # cpu 强制模式不调 _probe_gpu_vram_mb
    mock_vram.assert_not_called()
    assert flags.effective_device == "cpu"
    assert flags.cuda_accel is False
    assert flags.omniparser is False


def test_force_cuda_not_available_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """PERCEPTION_DEVICE=cuda 但 CUDA 不可用时，降级 auto → cpu。"""
    monkeypatch.setenv("PERCEPTION_DEVICE", "cuda")
    monkeypatch.delenv("OMNIPARSER_MODEL_DIR", raising=False)
    fake_ort = _fake_ort(["CPUExecutionProvider"])
    with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
        flags = probe_mod._do_probe()
    # 强制 cuda 但不可用 → 降级 auto → cpu（无 DML 也无 CUDA）
    assert flags.effective_device == "cpu"
    assert flags.cuda_accel is False


def test_onnxruntime_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """onnxruntime import 失败时全部 GPU 加速不可用，回退 cpu。"""
    monkeypatch.setenv("PERCEPTION_DEVICE", "auto")
    # 使 onnxruntime import 抛 ImportError
    with patch.dict(sys.modules, {"onnxruntime": None}):  # type: ignore[dict-item]
        flags = probe_mod._do_probe()
    assert flags.effective_device == "cpu"
    assert flags.cuda_accel is False
    assert flags.dml_accel is False


# ── OCR / mss 可用性 ──────────────────────────────────────────────────────────


def test_ocr_true_when_rapidocr_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """rapidocr_onnxruntime 可 import 时 ocr=True。"""
    monkeypatch.setenv("PERCEPTION_DEVICE", "cpu")
    fake_ort = _fake_ort(["CPUExecutionProvider"])
    fake_rapidocr = MagicMock()
    with patch.dict(sys.modules, {"onnxruntime": fake_ort, "rapidocr_onnxruntime": fake_rapidocr}):
        flags = probe_mod._do_probe()
    assert flags.ocr is True


def test_ocr_false_when_rapidocr_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """rapidocr_onnxruntime 不可 import 时 ocr=False。"""
    monkeypatch.setenv("PERCEPTION_DEVICE", "cpu")
    fake_ort = _fake_ort(["CPUExecutionProvider"])
    # 将 rapidocr_onnxruntime 从 sys.modules 中删掉并使 import 失败
    original = sys.modules.pop("rapidocr_onnxruntime", None)
    try:
        with patch.dict(sys.modules, {"onnxruntime": fake_ort, "rapidocr_onnxruntime": None}):  # type: ignore[dict-item]
            flags = probe_mod._do_probe()
        assert flags.ocr is False
    finally:
        if original is not None:
            sys.modules["rapidocr_onnxruntime"] = original


def test_mss_true_when_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """mss 可 import 时 mss_available=True。"""
    monkeypatch.setenv("PERCEPTION_DEVICE", "cpu")
    fake_ort = _fake_ort(["CPUExecutionProvider"])
    fake_mss = MagicMock()
    with patch.dict(sys.modules, {"onnxruntime": fake_ort, "mss": fake_mss}):
        flags = probe_mod._do_probe()
    assert flags.mss_available is True


def test_mss_false_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """mss 不可 import 时 mss_available=False。"""
    monkeypatch.setenv("PERCEPTION_DEVICE", "cpu")
    fake_ort = _fake_ort(["CPUExecutionProvider"])
    original = sys.modules.pop("mss", None)
    try:
        with patch.dict(sys.modules, {"onnxruntime": fake_ort, "mss": None}):  # type: ignore[dict-item]
            flags = probe_mod._do_probe()
        assert flags.mss_available is False
    finally:
        if original is not None:
            sys.modules["mss"] = original


# ── OmniParser ────────────────────────────────────────────────────────────────


def test_omniparser_false_without_model_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """OMNIPARSER_MODEL_DIR 未设时 omniparser=False（即便 GPU 可用）。"""
    monkeypatch.setenv("PERCEPTION_DEVICE", "auto")
    monkeypatch.delenv("OMNIPARSER_MODEL_DIR", raising=False)
    fake_ort = _fake_ort(["CUDAExecutionProvider", "CPUExecutionProvider"])
    with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
        with patch.object(probe_mod, "_probe_gpu_vram_mb", return_value=8192):
            flags = probe_mod._do_probe()
    assert flags.omniparser is False


def test_omniparser_false_on_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """CPU-only 机器 omniparser=False，即便 model_dir 存在。"""
    import pathlib

    model_dir = pathlib.Path(str(tmp_path)) / "omniparser_models"
    model_dir.mkdir()
    (model_dir / "model.pt").write_bytes(b"\x00")  # 非空目录

    monkeypatch.setenv("PERCEPTION_DEVICE", "cpu")
    monkeypatch.setenv("OMNIPARSER_MODEL_DIR", str(model_dir))
    fake_ort = _fake_ort(["CPUExecutionProvider"])
    with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
        flags = probe_mod._do_probe()
    assert flags.omniparser is False


# ── 幂等缓存 ──────────────────────────────────────────────────────────────────


def test_probe_capabilities_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """probe_capabilities() 多次调用返回完全相同的对象（缓存幂等）。"""
    monkeypatch.setenv("PERCEPTION_DEVICE", "cpu")
    fake_ort = _fake_ort(["CPUExecutionProvider"])
    with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
        first = probe_capabilities()
        second = probe_capabilities()
    assert first is second  # 同一对象引用


def test_probe_capabilities_cache_not_called_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_do_probe 只在第一次调用时执行，第二次直接走缓存。"""
    monkeypatch.setenv("PERCEPTION_DEVICE", "cpu")
    fake_ort = _fake_ort(["CPUExecutionProvider"])
    call_count = 0
    original_do_probe = probe_mod._do_probe

    def counting_do_probe() -> CapabilityFlags:
        nonlocal call_count
        call_count += 1
        return original_do_probe()

    with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
        with patch.object(probe_mod, "_do_probe", side_effect=counting_do_probe):
            probe_capabilities()
            probe_capabilities()
            probe_capabilities()

    assert call_count == 1, f"_do_probe 应只调用一次，实际 {call_count} 次"
