"""编排层组装根：把 memory / storage 真实现接进图（env 未配则退打桩，零回归）。

**为什么放编排层**：选「用真后端还是打桩」是**组装决策**，不是记忆层或存储层的职责——
那两层只提供能力，不读 env、不决定自己被不被启用。编排层是唯一同时知道两者的地方
（依赖只能自上而下：orchestration → memory → storage）。

⚠ **与硬约束 7「不直连记忆/存储」不冲突**：该约束约束的是**节点 / Agent**——它们只能见到
``MemoryAPI`` / ``SnapshotStore`` Protocol，不得知道具体后端。本模块是**组装根**（composition
root），依赖注入里唯一被允许知道具体类型的地方；它把具体实现装配好后，节点拿到的仍是 Protocol
形状。判据很简单：本模块之外的编排/Agent 代码若出现 ``SqliteMemoryStore`` 之类的具体类型，
那才是违约。

**为什么是 async context manager 而非 `get_graph` 内部构造**：开连接是 async，而
``get_graph()`` 是同步工厂；且连接生命周期须覆盖整个图的运行期。故由调用方（真正的组装根）
持有本 CM，把产出的两个实现注入 ``get_graph``::

    async with persistent_stores() as p:
        graph = get_graph(client=client, memory_api=p.memory_api, snapshot_store=p.snapshot_store)
        await graph.ainvoke(...)

配置（全走 env，不硬编码）：

| env | 作用 |
| --- | --- |
| ``ZERO_MCP_PERSISTENCE_DB`` | SQLite 路径。**未设/空 = 关**（退打桩，零回归）；``:memory:`` 亦可 |
| ``ZERO_MCP_MEMORY_SCOPE_KEY`` | 记忆作用域键。开持久化时**必填**，缺失即 fail-fast |

⚠ **缺 scope_key 为何 fail-fast 而非退打桩**：静默退化会让接线方以为「记忆已开」而实际没写，
属本仓反复踩过的「绿灯不响」类故障。宁可启动即炸，也不要跑了一整轮才发现记忆是空的。
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from src.memory import ScopedMemoryAPI
from src.orchestration.protocols import MemoryAPI, NoopMemoryAPI
from src.storage import SqliteMemoryStore, SqliteSnapshotStore, open_connection

logger = logging.getLogger(__name__)

ENV_DB_PATH: str = "ZERO_MCP_PERSISTENCE_DB"
ENV_SCOPE_KEY: str = "ZERO_MCP_MEMORY_SCOPE_KEY"


@dataclass(frozen=True)
class PersistenceBundle:
    """组装产物：可直接注入 ``get_graph`` 的两个实现 + 是否真启用。

    memory_api     : 真实现（``ScopedMemoryAPI``）或打桩（``NoopMemoryAPI``）。
    snapshot_store : 真实现（``SqliteSnapshotStore``）或 ``None``
                     （``get_graph`` 对 None 的既有语义 = 跳过 phash 信号 A，保持不变）。
    enabled        : 是否接了真后端——供调用方日志/断言，避免「以为开了其实没开」。
    """

    memory_api: MemoryAPI
    snapshot_store: Any | None
    enabled: bool


@asynccontextmanager
async def persistent_stores(
    db_path: str | None = None,
    scope_key: str | None = None,
) -> AsyncIterator[PersistenceBundle]:
    """构造持久化实现并管理连接生命周期；env 未配则产打桩（零回归）。

    Args:
        db_path:   显式 SQLite 路径；None = 读 ``ZERO_MCP_PERSISTENCE_DB``。
                   最终为空 → **关闭持久化**，产打桩且不开连接。
        scope_key: 显式记忆作用域键；None = 读 ``ZERO_MCP_MEMORY_SCOPE_KEY``。
                   开持久化但最终为空 → ``ValueError`` fail-fast。

    Yields:
        ``PersistenceBundle``（退出时自动关闭连接，异常路径亦关）。

    Raises:
        ValueError: 开了持久化却没给 scope_key（作用域必须显式，memory-rules 第 2 条）。
    """
    resolved_db = db_path if db_path is not None else os.getenv(ENV_DB_PATH, "")
    if not resolved_db:
        logger.info("persistent_stores: 未配 %s，使用打桩（零回归）", ENV_DB_PATH)
        yield PersistenceBundle(memory_api=NoopMemoryAPI(), snapshot_store=None, enabled=False)
        return

    resolved_key = scope_key if scope_key is not None else os.getenv(ENV_SCOPE_KEY, "")
    if not resolved_key or not resolved_key.strip():
        raise ValueError(
            f"已配 {ENV_DB_PATH}={resolved_db!r} 开启持久化，但缺 {ENV_SCOPE_KEY}。"
            "记忆作用域由 (scope, scope_key) 共同确定，缺 scope_key 会跨用户/会话串味"
            "（memory-rules 第 2 条）。此处**刻意 fail-fast 而非退打桩**——静默退化会让接线方"
            "以为记忆已开、实则整轮没写。"
        )

    connection = await open_connection(resolved_db)
    try:
        bundle = PersistenceBundle(
            memory_api=ScopedMemoryAPI(SqliteMemoryStore(connection), scope_key=resolved_key),
            snapshot_store=SqliteSnapshotStore(connection),
            enabled=True,
        )
        logger.info(
            "persistent_stores: 已接真后端 db=%r scope_key=%r（记忆写入仍只在 memory_flush 节点）",
            resolved_db,
            resolved_key,
        )
        yield bundle
    finally:
        await connection.close()
        logger.debug("persistent_stores: 连接已关闭")
