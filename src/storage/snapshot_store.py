"""存储层：``SnapshotStore`` 的 SQLite 实现（运行态感知快照持久化）。

对应契约：``src/agents/protocols.py::SnapshotStore``（``save`` / ``load``）。
**结构化实现，不显式继承 Protocol**——与 `InMemorySnapshotStore` 打桩同款，避免运行期
继承耦合（mypy 结构子型即可校验）。

分层：本模块属存储层（最底层），只 import ``src/agents/models/`` 的共享契约模型
（跨层契约唯一真相，被 Agent 层与下游共用，不算反向依赖，见 project-root.md）。

**为何存快照进「运行态」而非记忆图谱**（memory-rules 第 3 条）：`ScreenSnapshot` 是运行态
证据（含屏幕原始文本，`is_untrusted=True`），不是长期语义记忆——混进图谱会污染事实抽取。
故本实现落 `snapshots` 表，与 `memory_facts` **物理分表**。
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from src.agents.models.screen_snapshot import ScreenSnapshot

logger = logging.getLogger(__name__)


class SqliteSnapshotStore:
    """SnapshotStore 的 SQLite 实现（结构化满足 Protocol）。

    连接由**调用方注入并负责生命周期**（本类不开不关连接）——便于多 store 共用一条连接、
    也便于测试用内存库。典型接线::

        conn = await open_connection("var/runtime.db")
        store = SqliteSnapshotStore(conn)
        agent = ScreenPerceptionAgent(snapshot_store=store)
        ...
        await conn.close()

    Args:
        connection: 已建 schema 的 aiosqlite 连接（见 ``sqlite_backend.open_connection``）。
    """

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    async def save(self, snapshot: ScreenSnapshot) -> str:
        """存储快照，返回 ``snapshot_id``。

        同 ID 重复保存做**覆盖**（``INSERT OR REPLACE``）：快照 ID 由感知侧生成且唯一，
        重放/重试写同一 ID 时应幂等，而非累积多行或抛主键冲突。

        Args:
            snapshot: 完整感知快照对象。

        Returns:
            ``snapshot.snapshot_id``（可作 state 引用；契约要求与入参 ID 一致）。
        """
        payload = snapshot.model_dump_json()
        created_at = dt.datetime.now(dt.UTC).isoformat()
        await self.connection.execute(
            "INSERT OR REPLACE INTO snapshots (snapshot_id, payload, created_at) VALUES (?, ?, ?)",
            (snapshot.snapshot_id, payload, created_at),
        )
        await self.connection.commit()
        logger.debug("SqliteSnapshotStore.save: %s", snapshot.snapshot_id)
        return snapshot.snapshot_id

    async def load(self, snapshot_id: str) -> ScreenSnapshot:
        """按 ID 加载快照。

        Args:
            snapshot_id: ``save()`` 返回的 ID。

        Returns:
            反序列化后的 ``ScreenSnapshot``。

        Raises:
            KeyError: 该 ID 不存在。**刻意抛而非返回 None**——契约返回类型不可空，
                与打桩实现 ``InMemorySnapshotStore``（dict 直取，缺失即 KeyError）行为一致。
        """
        async with self.connection.execute(
            "SELECT payload FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise KeyError(f"快照不存在：{snapshot_id!r}")
        return ScreenSnapshot.model_validate_json(row[0])
