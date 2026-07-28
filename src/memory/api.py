"""记忆层：``MemoryAPI`` 的落地实现（显式 scope · 任务完成节流 · 时序失效）。

对应契约：``src/orchestration/protocols.py::MemoryAPI``（``write_session_summary``）。
**结构化实现，不显式继承 Protocol**（与 `NoopMemoryAPI` 打桩同款）。

分层（project-root.md）：记忆层**只调下层**——本模块 import ``src/storage/``，
**不 import 编排层**；上层 Agent 不得绕过本 API 直连存储/图谱。

四条硬约束在此落实（memory-rules.md）：
1. **任务完成节流**——本 API 只提供「一次任务写一条摘要」的粒度；无逐步写入接口。
   在 Supervisor 的任务完成节点调用一次即可，本类另设**同 task_id 重复写守卫**。
2. **作用域必须显式**——``scope`` 是必填位置参数（契约如此），且 ``scope_key`` 必须非空；
   **本类不提供默认 user**，缺 scope_key 直接 ``ValueError`` fail-fast。
3. **运行态与长期记忆分离**——本类只写 ``memory_facts``；运行态快照走
   ``src/storage/snapshot_store.py``（物理分表），二者不互串。
4. **时序事实会失效**——同一 (scope, scope_key) 下写新摘要时，可将该 task_id 的旧摘要
   标记失效（``supersede_same_task=True``，默认开），实现「新事实使旧事实失效而非删除」。

**扩展点**（不在本轮范围）：Zep / Mem0 / Graphiti 作为**另一种 store 实现**接入——
本类只依赖「append / invalidate / query」三个原语，换后端不改本类；自动抽取（实体/关系）
与跨事实去重属抽取策略层，见 memory 扩展指南。
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from src.storage.memory_store import MemoryFact, SqliteMemoryStore

logger = logging.getLogger(__name__)

MemoryScope = Literal["session", "user", "group"]


class ScopedMemoryAPI:
    """MemoryAPI 的落地实现：显式作用域 + 任务完成节流 + 时序失效。

    Args:
        store:                 长期记忆存储（``SqliteMemoryStore`` 或任何同形后端）。
        scope_key:             作用域键（如 user id / session id / group id）。**必填非空**——
                               作用域语义由「scope + scope_key」共同确定，只给 scope 会让不同
                               用户/会话的记忆串味（memory-rules 第 2 条）。
        supersede_same_task:   写入时是否把**同 task_id** 的旧摘要标记失效。默认 True
                               （同一任务重跑/重试产生的新摘要应取代旧的，而非并存）。

    Raises:
        ValueError: ``scope_key`` 为空/空白。
    """

    def __init__(
        self,
        store: SqliteMemoryStore,
        scope_key: str,
        supersede_same_task: bool = True,
    ) -> None:
        if not scope_key or not scope_key.strip():
            raise ValueError(
                "scope_key 必须非空——记忆作用域由 (scope, scope_key) 共同确定；"
                "缺失会导致跨用户/会话记忆串味（memory-rules 第 2 条：作用域必须显式）"
            )
        self.store = store
        self.scope_key = scope_key
        self.supersede_same_task = supersede_same_task

    async def write_session_summary(
        self,
        task_id: str,
        scope: MemoryScope,
        summary: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """写入任务会话摘要（**只应在任务完成节点调用**，见 memory-rules 第 1 条）。

        Args:
            task_id:  任务唯一 ID（与 LangGraph thread_id 对应）。
            scope:    作用域，**必须显式指定**（session / user / group）。
            summary:  任务摘要文本；空白摘要视为无内容，跳过写入（不产空事实污染记忆）。
            metadata: 可选附加元数据（执行状态、步骤数等）。

        Raises:
            ValueError: ``scope`` 不是三个合法值之一（防拼写错误静默落库成新作用域）。
        """
        if scope not in ("session", "user", "group"):
            raise ValueError(
                f"非法记忆作用域 {scope!r}，必须是 'session' / 'user' / 'group' 之一"
                "（禁止默认 user；拼写错误若放行会静默产生新作用域、记忆再也读不回）"
            )
        if not summary or not summary.strip():
            logger.warning("ScopedMemoryAPI: task_id=%s 摘要为空，跳过写入（不产空事实）", task_id)
            return

        if self.supersede_same_task:
            stale = [
                fact.fact_id
                for fact in await self.store.query_facts(scope, self.scope_key)
                if fact.task_id == task_id
            ]
            if stale:
                count = await self.store.invalidate_facts(stale)
                logger.debug(
                    "ScopedMemoryAPI: task_id=%s 使 %d 条旧摘要失效（时序取代，非删除）",
                    task_id,
                    count,
                )

        await self.store.append_fact(
            scope=scope,
            scope_key=self.scope_key,
            task_id=task_id,
            content=summary,
            metadata=metadata,
        )
        logger.debug("ScopedMemoryAPI.write_session_summary: task=%s scope=%s", task_id, scope)

    async def read_current(
        self,
        scope: MemoryScope,
        limit: int | None = None,
    ) -> list[MemoryFact]:
        """读取本作用域下**当前有效**的事实（时间升序）。

        读也必须显式给 scope——与写同一纪律（memory-rules 第 2 条）。
        需要含已失效事实的全历史（审计/时序回溯）时直接用 store 的
        ``query_facts(include_invalidated=True)``。
        """
        if scope not in ("session", "user", "group"):
            raise ValueError(f"非法记忆作用域 {scope!r}，必须是 'session' / 'user' / 'group' 之一")
        return await self.store.query_facts(scope, self.scope_key, limit=limit)
