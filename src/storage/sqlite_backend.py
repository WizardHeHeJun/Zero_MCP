"""存储层（最底层）：SQLite 连接与 schema 管理。

选型理由（工程决策，非文献）：本仓需要一个**零 infra、可离线、可测**的默认后端——
Postgres/Neo4j/Redis 都要求外部服务在位，而本仓当前无常驻 infra。SQLite 经 ``aiosqlite``
提供 async 接口，文件即库（亦支持 ``:memory:``），满足「运行态与长期记忆分离」纪律下的
两类持久化需求，且可被单测完全覆盖。

**分层纪律（project-root.md）**：本层是最底层——
- **不 import 记忆层 / 编排层**（只被它们调用）。
- 可 import ``src/agents/models/`` 的**共享契约模型**（该目录是跨层契约唯一真相，
  被 Agent 层与下游共用，不算反向依赖）。

**扩展点**：Postgres（运行态 Checkpointer）、Neo4j/Graphiti（长期记忆图谱）在此层平行新增
后端模块即可，上层经 Protocol 结构化消费、无需改动（见 storage 扩展指南）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 内存库标识：SQLite 约定，进程内存活、进程退出即消失（测试默认）
MEMORY_DB: str = ":memory:"

# ── schema ────────────────────────────────────────────────────────────────────
# 两张表分别对应「运行态快照」与「长期记忆事实」，**物理分表**以落实
# memory-rules 第 3 条「运行态与长期记忆分离」——即便同库也不混表。

_SCHEMA_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
)
"""

# 长期记忆事实表。时序语义（memory-rules 第 4 条）：新事实**使旧事实失效**而非物理删除，
# 故设 invalidated_at 而非 DELETE；查询默认只取 invalidated_at IS NULL 的当前有效事实。
_SCHEMA_MEMORY_FACTS = """
CREATE TABLE IF NOT EXISTS memory_facts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    scope          TEXT NOT NULL,
    scope_key      TEXT NOT NULL,
    task_id        TEXT NOT NULL,
    content        TEXT NOT NULL,
    metadata       TEXT,
    created_at     TEXT NOT NULL,
    invalidated_at TEXT
)
"""

# 按「作用域 + 作用域键 + 是否有效」检索是最热路径（读当前有效记忆）
_SCHEMA_MEMORY_INDEX = """
CREATE INDEX IF NOT EXISTS idx_memory_scope
    ON memory_facts (scope, scope_key, invalidated_at)
"""

_ALL_SCHEMA: tuple[str, ...] = (_SCHEMA_SNAPSHOTS, _SCHEMA_MEMORY_FACTS, _SCHEMA_MEMORY_INDEX)


def _import_aiosqlite() -> Any:
    """延迟 import aiosqlite（缺库时给出可操作的错误信息）。"""
    try:
        import aiosqlite
    except ImportError as exc:  # pragma: no cover - 环境已装，保留可读报错
        raise RuntimeError(
            "存储层需要 aiosqlite（`uv pip install aiosqlite`）。"
            "本层是最底层依赖，缺库无法优雅回退——请安装后重试。"
        ) from exc
    return aiosqlite


async def open_connection(db_path: str | Path = MEMORY_DB) -> Any:
    """打开 SQLite 连接并**确保 schema 就位**（幂等）。

    Args:
        db_path: 数据库文件路径；``":memory:"``（默认）= 进程内内存库。
                 文件路径的父目录不存在时自动创建。

    Returns:
        已建表的 ``aiosqlite.Connection``（调用方负责 ``await conn.close()``）。
    """
    aiosqlite = _import_aiosqlite()
    if db_path != MEMORY_DB:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(db_path)
    await init_schema(conn)
    return conn


async def init_schema(conn: Any) -> None:
    """在给定连接上建表建索引（``IF NOT EXISTS``，可重复调用）。"""
    for statement in _ALL_SCHEMA:
        await conn.execute(statement)
    await conn.commit()
    logger.debug("storage: schema 就位（snapshots / memory_facts）")
