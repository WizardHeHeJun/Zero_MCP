"""存储层（最底层）：运行态快照 + 长期记忆事实的持久化后端。

默认后端 **SQLite**（``aiosqlite``，零 infra、文件即库、支持 ``:memory:``）。
Postgres（运行态 Checkpointer）/ Neo4j·Graphiti（记忆图谱）作为平行后端在本层扩展，
上层经结构化 Protocol 消费、换后端无需改动。

分层纪律：本层**不 import 记忆层 / 编排层**；可 import ``src/agents/models/`` 的共享契约模型。
两类数据**物理分表**（``snapshots`` / ``memory_facts``），落实「运行态与长期记忆分离」。
"""

from __future__ import annotations

from src.storage.memory_store import MemoryFact, SqliteMemoryStore
from src.storage.snapshot_store import SqliteSnapshotStore
from src.storage.sqlite_backend import MEMORY_DB, init_schema, open_connection

__all__ = [
    "MEMORY_DB",
    "open_connection",
    "init_schema",
    "SqliteSnapshotStore",
    "SqliteMemoryStore",
    "MemoryFact",
]
