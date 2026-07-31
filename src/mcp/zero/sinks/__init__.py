"""表达半程渲染终端子包公开 API。

导出 RenderFrame（渲染指令数据模型）、RenderingExpressionSink（渲染终端实现）、
VtsExpressionSink（VTube Studio Live2D 渲染终端）与离散行为手势层
BehaviorOverlayEngine / OverlayFrame / EngineSnapshot / VOCABULARY
（手势合成引擎，蓝图 2026-07-31 T2）。
"""

from __future__ import annotations

from src.mcp.zero.sinks.behavior_overlay import (
    VOCABULARY,
    BehaviorOverlayEngine,
    EngineSnapshot,
    OverlayFrame,
)
from src.mcp.zero.sinks.rendering import RenderFrame, RenderingExpressionSink
from src.mcp.zero.sinks.vts import VtsExpressionSink

__all__ = [
    "VOCABULARY",
    "BehaviorOverlayEngine",
    "EngineSnapshot",
    "OverlayFrame",
    "RenderFrame",
    "RenderingExpressionSink",
    "VtsExpressionSink",
]
