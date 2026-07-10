"""启动时硬件/库能力探测，产出 CapabilityFlags 并缓存（幂等）。

探测顺序（规格书 Task 3B）：
  1. 读 .env PERCEPTION_DEVICE=auto|cpu|dml|cuda
  2. onnxruntime EP 探测：auto 按 CUDA>DML>CPU 查 get_available_providers()，
     强制模式不满足则 log warning 降 auto；cpu 跳过 GPU 探测。
  3. GPU 显存探测（try/except，失败仅 log debug）。
  4. OmniParser 可用性：仅 GPU 且显存>=OMNIPARSER_MIN_VRAM_MB 且目录存在非空。
  5. RapidOCR import try/except。
  6. mss import try/except。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# module-level 幂等缓存
_CACHED_FLAGS: CapabilityFlags | None = None


@dataclass
class CapabilityFlags:
    """探测结果：各能力是否可用。

    Attributes:
        ocr: RapidOCR 可 import（不含推理测试）。
        omniparser: OmniParser 模型就绪（GPU + 显存 + 目录非空）。
        cuda_accel: onnxruntime CUDAExecutionProvider 可用。
        dml_accel: onnxruntime DmlExecutionProvider 可用。
        mss_available: mss 可 import。
        effective_device: 实际生效设备标识（"cuda"|"dml"|"cpu"）。
    """

    ocr: bool
    omniparser: bool
    cuda_accel: bool
    dml_accel: bool
    mss_available: bool
    effective_device: str


def _parse_bool_env(key: str, default: str = "false") -> bool:
    """读取布尔 env 变量，1/true/yes 为 True（不区分大小写）。"""
    return os.environ.get(key, default).lower() in {"1", "true", "yes"}


def _probe_gpu_vram_mb() -> int | None:
    """探测 GPU 可用显存（MB）。失败返回 None，仅 log debug。

    优先用 pynvml（CUDA GPU）；DML 无通用显存查询接口，暂不支持。
    """
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_mb: int = int(mem_info.free) // (1024 * 1024)
        pynvml.nvmlShutdown()
        logger.debug("GPU 显存探测：free=%d MB", vram_mb)
        return vram_mb
    except Exception as exc:
        logger.debug("GPU 显存探测失败（非致命）：%s", exc)
        return None


def probe_capabilities() -> CapabilityFlags:
    """探测并返回 CapabilityFlags；已探测则直接返回缓存（幂等）。

    Returns:
        CapabilityFlags 实例，反映当前运行环境的能力状态。
    """
    global _CACHED_FLAGS
    if _CACHED_FLAGS is not None:
        return _CACHED_FLAGS

    _CACHED_FLAGS = _do_probe()
    logger.info(
        "capability_probe 完成：device=%s cuda=%s dml=%s ocr=%s omniparser=%s mss=%s",
        _CACHED_FLAGS.effective_device,
        _CACHED_FLAGS.cuda_accel,
        _CACHED_FLAGS.dml_accel,
        _CACHED_FLAGS.ocr,
        _CACHED_FLAGS.omniparser,
        _CACHED_FLAGS.mss_available,
    )
    return _CACHED_FLAGS


def _do_probe() -> CapabilityFlags:
    """实际执行一次完整探测，返回 CapabilityFlags。"""

    # ── 步骤 1：读取 PERCEPTION_DEVICE 配置 ───────────────────────────────────
    perception_device = os.environ.get("PERCEPTION_DEVICE", "auto").lower().strip()
    if perception_device not in {"auto", "cpu", "dml", "cuda"}:
        logger.warning("PERCEPTION_DEVICE=%r 无效，回退 auto", perception_device)
        perception_device = "auto"
    logger.debug("PERCEPTION_DEVICE 配置值：%s", perception_device)

    # ── 步骤 2：onnxruntime EP 探测 ───────────────────────────────────────────
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]

        available = ort.get_available_providers()
    except Exception as exc:
        logger.warning("onnxruntime import 失败：%s；所有 GPU 加速不可用", exc)
        available = []

    cuda_available = "CUDAExecutionProvider" in available
    dml_available = "DmlExecutionProvider" in available

    # 强制模式不满足则 log warning 并降 auto
    if perception_device == "cuda" and not cuda_available:
        logger.warning("PERCEPTION_DEVICE=cuda 但 CUDAExecutionProvider 不可用，降级为 auto")
        perception_device = "auto"
    elif perception_device == "dml" and not dml_available:
        logger.warning("PERCEPTION_DEVICE=dml 但 DmlExecutionProvider 不可用，降级为 auto")
        perception_device = "auto"

    # 按 CUDA > DML > CPU 确定 effective_device
    if perception_device == "cpu":
        effective_device = "cpu"
        cuda_accel = False
        dml_accel = False
    elif perception_device == "cuda":
        effective_device = "cuda"
        cuda_accel = True
        dml_accel = False
    elif perception_device == "dml":
        effective_device = "dml"
        cuda_accel = False
        dml_accel = True
    else:
        # auto：CUDA > DML > CPU
        if cuda_available:
            effective_device = "cuda"
            cuda_accel = True
            dml_accel = False
        elif dml_available:
            effective_device = "dml"
            cuda_accel = False
            dml_accel = True
        else:
            effective_device = "cpu"
            cuda_accel = False
            dml_accel = False

    logger.debug(
        "EP 探测：available=%s effective_device=%s cuda_accel=%s dml_accel=%s",
        available,
        effective_device,
        cuda_accel,
        dml_accel,
    )

    # ── 步骤 3：GPU 显存探测（仅 GPU 模式；cpu 跳过）────────────────────────
    vram_mb: int | None = None
    if perception_device != "cpu" and effective_device in {"cuda", "dml"}:
        vram_mb = _probe_gpu_vram_mb()

    # ── 步骤 4：OmniParser 可用性 ─────────────────────────────────────────────
    # 条件：GPU + 显存 >= OMNIPARSER_MIN_VRAM_MB + 目录存在且非空
    omniparser_min_vram = int(os.environ.get("OMNIPARSER_MIN_VRAM_MB", "4096"))
    omniparser_model_dir = os.environ.get("OMNIPARSER_MODEL_DIR", "").strip()

    is_gpu = effective_device in {"cuda", "dml"}
    has_enough_vram = vram_mb is not None and vram_mb >= omniparser_min_vram
    has_model_dir = (
        bool(omniparser_model_dir)
        and Path(omniparser_model_dir).is_dir()
        and any(Path(omniparser_model_dir).iterdir())
    )
    omniparser = is_gpu and has_enough_vram and has_model_dir
    logger.debug(
        "OmniParser 探测：is_gpu=%s vram_mb=%s min_vram=%s has_model_dir=%s → %s",
        is_gpu,
        vram_mb,
        omniparser_min_vram,
        has_model_dir,
        omniparser,
    )

    # ── 步骤 5：RapidOCR 可用性 ───────────────────────────────────────────────
    try:
        import rapidocr_onnxruntime  # noqa: F401

        ocr = True
        logger.debug("RapidOCR import 成功")
    except Exception as exc:
        ocr = False
        logger.debug("RapidOCR import 失败：%s", exc)

    # ── 步骤 6：mss 可用性 ────────────────────────────────────────────────────
    try:
        import mss  # noqa: F401

        mss_available = True
        logger.debug("mss import 成功")
    except Exception as exc:
        mss_available = False
        logger.debug("mss import 失败：%s", exc)

    return CapabilityFlags(
        ocr=ocr,
        omniparser=omniparser,
        cuda_accel=cuda_accel,
        dml_accel=dml_accel,
        mss_available=mss_available,
        effective_device=effective_device,
    )
