"""存储层：长期记忆事实的 SQLite 持久化（时序失效语义）。

本模块只做**持久化原语**——不做抽取/去重/节流（那是记忆层 `src/memory/` 的职责）。
分层：最底层，不 import 记忆层/编排层。

时序语义（memory-rules 第 4 条）：新事实**使旧事实失效而非物理删除**。故：
- 写入用 ``append_fact``（只追加，永不 UPDATE 内容）；
- 失效用 ``invalidate_facts``（打 ``invalidated_at`` 时间戳，行仍在）；
- 读取默认只返回**当前有效**事实；``include_invalidated=True`` 可取全历史做审计/回溯。
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

MemoryScope = Literal["session", "user", "group"]


@dataclass(frozen=True)
class MemoryFact:
    """一条长期记忆事实（存储层视角的行记录）。

    fact_id        : 自增主键。
    scope/scope_key: 作用域与其键（如 scope="user"、scope_key="u-42"）。
    task_id        : 产生该事实的任务 ID（对应 LangGraph thread_id）。
    content        : 事实文本。
    metadata       : 附加元数据（已反序列化；无则空 dict）。
    created_at     : 写入时间（ISO8601 UTC）。
    invalidated_at : 失效时间；None = **当前有效**。
    """

    fact_id: int
    scope: str
    scope_key: str
    task_id: str
    content: str
    metadata: dict[str, Any]
    created_at: str
    invalidated_at: str | None


class SqliteMemoryStore:
    """长期记忆事实的 SQLite 存储（只追加 + 时序失效）。

    连接由调用方注入并负责生命周期（同 ``SqliteSnapshotStore``）。

    Args:
        connection: 已建 schema 的 aiosqlite 连接（见 ``sqlite_backend.open_connection``）。
    """

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    async def append_fact(
        self,
        scope: MemoryScope,
        scope_key: str,
        task_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """追加一条事实，返回其 ``fact_id``。**只追加，不覆盖既有行。**"""
        created_at = dt.datetime.now(dt.UTC).isoformat()
        cursor = await self.connection.execute(
            "INSERT INTO memory_facts "
            "(scope, scope_key, task_id, content, metadata, created_at, invalidated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, NULL)",
            (scope, scope_key, task_id, content, json.dumps(metadata or {}), created_at),
        )
        await self.connection.commit()
        fact_id = int(cursor.lastrowid)
        logger.debug("SqliteMemoryStore.append_fact: id=%d scope=%s/%s", fact_id, scope, scope_key)
        return fact_id

    async def invalidate_facts(self, fact_ids: list[int]) -> int:
        """把给定事实标记为失效（**不删除行**），返回实际失效条数。

        已失效的行不重复打戳（``invalidated_at IS NULL`` 条件），保证首次失效时间稳定。
        """
        if not fact_ids:
            return 0
        stamp = dt.datetime.now(dt.UTC).isoformat()
        placeholders = ",".join("?" for _ in fact_ids)
        cursor = await self.connection.execute(
            f"UPDATE memory_facts SET invalidated_at = ? "  # noqa: S608 - 占位符由长度生成，非外部拼接
            f"WHERE id IN ({placeholders}) AND invalidated_at IS NULL",
            (stamp, *fact_ids),
        )
        await self.connection.commit()
        return int(cursor.rowcount)

    async def query_facts(
        self,
        scope: MemoryScope,
        scope_key: str,
        include_invalidated: bool = False,
        limit: int | None = None,
    ) -> list[MemoryFact]:
        """按作用域读取事实（默认只取**当前有效**的，按写入时间升序）。

        Args:
            scope/scope_key:     作用域与其键（**必须显式给**，本层不设默认）。
            include_invalidated: True 时连已失效事实一并返回（审计/时序回溯用）。
            limit:               最多返回条数；None = 不限。
        """
        sql = "SELECT id, scope, scope_key, task_id, content, metadata, created_at, invalidated_at "
        sql += "FROM memory_facts WHERE scope = ? AND scope_key = ?"
        if not include_invalidated:
            sql += " AND invalidated_at IS NULL"
        sql += " ORDER BY id ASC"
        params: tuple[Any, ...] = (scope, scope_key)
        if limit is not None:
            sql += " LIMIT ?"
            params = (*params, limit)

        async with self.connection.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [
            MemoryFact(
                fact_id=int(r[0]),
                scope=str(r[1]),
                scope_key=str(r[2]),
                task_id=str(r[3]),
                content=str(r[4]),
                metadata=json.loads(r[5]) if r[5] else {},
                created_at=str(r[6]),
                invalidated_at=r[7],
            )
            for r in rows
        ]
