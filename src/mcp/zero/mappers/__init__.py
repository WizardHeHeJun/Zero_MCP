"""韵律/FACS/生理 映射器子包公开 API。

导出：
- ProsodyParams / LinearProsodyMapper（引擎无关 TTS 韵律参数与线性映射实现）
- AU_TO_ARKIT / ArkitFacsMapper（ARKit blendshape FACS 映射表与映射器实现）
- PhysiologyParams / LinearPhysiologyMapper（引擎无关生理动画参数与线性映射实现）
"""

from __future__ import annotations

# -- FACS 映射器 --------------------------------------------------------------
from src.mcp.zero.mappers.facs import AU_TO_ARKIT, ArkitFacsMapper

# -- 生理映射器 ---------------------------------------------------------------
from src.mcp.zero.mappers.physiology import LinearPhysiologyMapper, PhysiologyParams

# -- 韵律映射器 ---------------------------------------------------------------
from src.mcp.zero.mappers.prosody import LinearProsodyMapper, ProsodyParams

__all__ = [
    # FACS 映射器
    "AU_TO_ARKIT",
    "ArkitFacsMapper",
    # 生理映射器
    "LinearPhysiologyMapper",
    "PhysiologyParams",
    # 韵律映射器
    "LinearProsodyMapper",
    "ProsodyParams",
]
