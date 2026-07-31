"""VTS 离散行为业务层子包（蓝图 2026-07-31-vts-discrete-behavior · T4）。

`BehaviorService` 是行为层业务的唯一入口：MCP server
（`src/mcp/vts_behavior_mcp_server.py`）只做薄转发——传输层零业务逻辑
（`rules/mcp-integration.md`）。sink 生命周期、热键 catalog、触发分发与
状态聚合全在本包。
"""

from __future__ import annotations

from src.mcp.behavior.service import BehaviorService

__all__ = ["BehaviorService"]
