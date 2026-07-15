"""感知半程适配器子包公开 API。

导出 CallablePerceptionChannel：将任意 async callable 包装为
满足 PerceptionChannel Protocol 的感知通道，可直接交给 PerceptionHub。
"""

from __future__ import annotations

from src.mcp.zero.channels.callable_channel import CallablePerceptionChannel

__all__ = [
    "CallablePerceptionChannel",
]
