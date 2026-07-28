"""记忆层：长期记忆的读写 API（显式 scope · 任务完成节流 · 时序失效）。

对上提供稳定读写 API，对下封装存储后端（当前 SQLite；Zep/Mem0/Graphiti 为扩展点），
**屏蔽存储细节**——上层 Agent 不得绕过本层直连存储/图谱。

分层纪律：只调下层（import ``src/storage/``），**不 import 编排层**。
四条硬约束见 ``.claude/rules/memory-rules.md`` 与 ``ScopedMemoryAPI`` docstring。
"""

from __future__ import annotations

from src.memory.api import ScopedMemoryAPI

__all__ = ["ScopedMemoryAPI"]
