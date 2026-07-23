"""表达半程渲染终端子包公开 API。

导出 RenderFrame（渲染指令数据模型）与 RenderingExpressionSink（渲染终端实现）。
"""

from __future__ import annotations

from src.mcp.zero.sinks.rendering import RenderFrame, RenderingExpressionSink

__all__ = [
    "RenderFrame",
    "RenderingExpressionSink",
]
