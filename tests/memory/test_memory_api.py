"""记忆层单测：ScopedMemoryAPI 的四条硬约束（memory-rules.md）。

覆盖：
  M1. 契约结构兼容：满足 MemoryAPI 的 write_session_summary 签名（与 NoopMemoryAPI 同形）。
  M2. 作用域必须显式：空 scope_key 构造期 fail-fast；非法 scope 写入期 fail-fast。
  M3. 作用域隔离：不同 scope / scope_key 写入互不可见。
  M4. 时序失效：同 task_id 重写使旧摘要**失效而非删除**（历史仍可回溯）。
  M5. 运行态与长期记忆分离：写记忆不落 snapshots 表。
  M6. 空摘要跳过（不产空事实污染记忆）。
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from src.memory import ScopedMemoryAPI
from src.orchestration.protocols import NoopMemoryAPI
from src.storage import MEMORY_DB, SqliteMemoryStore, open_connection


@pytest.fixture
async def store() -> Any:
    """内存库上的记忆事实存储（每例独立）。"""
    connection = await open_connection(MEMORY_DB)
    try:
        yield SqliteMemoryStore(connection)
    finally:
        await connection.close()


class TestContractCompatibility:
    def test_signature_matches_noop_stub(self) -> None:
        """M1：与打桩实现同签名——可直接替换注入点，无需改编排层。"""
        real = inspect.signature(ScopedMemoryAPI.write_session_summary)
        noop = inspect.signature(NoopMemoryAPI.write_session_summary)
        assert list(real.parameters) == list(noop.parameters)


class TestExplicitScope:
    async def test_empty_scope_key_fails_fast(self, store: Any) -> None:
        """M2：scope_key 空/空白 → 构造期 ValueError（不给「默认 user」的机会）。"""
        for bad in ("", "   "):
            with pytest.raises(ValueError, match="scope_key 必须非空"):
                ScopedMemoryAPI(store, scope_key=bad)

    async def test_illegal_scope_rejected(self, store: Any) -> None:
        """M2：非法 scope 写入期 ValueError——拼错若放行会静默造出读不回的新作用域。"""
        api = ScopedMemoryAPI(store, scope_key="u-1")
        with pytest.raises(ValueError, match="非法记忆作用域"):
            await api.write_session_summary("t-1", "users", "摘要")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="非法记忆作用域"):
            await api.read_current("USER")  # type: ignore[arg-type]

    async def test_all_three_scopes_accepted(self, store: Any) -> None:
        """M2：三个合法 scope 均可写读（判别性：证明上例的拒绝不是「全都拒」）。"""
        api = ScopedMemoryAPI(store, scope_key="k-1")
        for scope in ("session", "user", "group"):
            await api.write_session_summary(f"t-{scope}", scope, f"{scope} 摘要")  # type: ignore[arg-type]
            assert [f.content for f in await api.read_current(scope)] == [f"{scope} 摘要"]  # type: ignore[arg-type]


class TestScopeIsolation:
    async def test_different_scope_keys_isolated(self, store: Any) -> None:
        """M3：同 scope 不同 scope_key 互不可见（防跨用户串味）。"""
        api_a = ScopedMemoryAPI(store, scope_key="u-a")
        api_b = ScopedMemoryAPI(store, scope_key="u-b")
        await api_a.write_session_summary("t-1", "user", "A 的记忆")
        await api_b.write_session_summary("t-1", "user", "B 的记忆")
        assert [f.content for f in await api_a.read_current("user")] == ["A 的记忆"]
        assert [f.content for f in await api_b.read_current("user")] == ["B 的记忆"]


class TestTemporalInvalidation:
    async def test_same_task_supersedes_not_deletes(self, store: Any) -> None:
        """M4：同 task_id 重写 → 旧摘要失效但**行仍在**，历史可回溯。"""
        api = ScopedMemoryAPI(store, scope_key="u-1")
        await api.write_session_summary("t-1", "user", "第一版")
        await api.write_session_summary("t-1", "user", "修订版")

        current = await api.read_current("user")
        assert [f.content for f in current] == ["修订版"], "当前有效应只剩最新一条"

        history = await store.query_facts("user", "u-1", include_invalidated=True)
        assert [f.content for f in history] == ["第一版", "修订版"], "旧事实应保留（失效非删除）"
        assert history[0].invalidated_at is not None and history[1].invalidated_at is None

    async def test_different_tasks_coexist(self, store: Any) -> None:
        """M4 判别性：不同 task_id 的摘要**并存**——取代只发生在同一任务内。"""
        api = ScopedMemoryAPI(store, scope_key="u-1")
        await api.write_session_summary("t-1", "user", "任务一")
        await api.write_session_summary("t-2", "user", "任务二")
        assert [f.content for f in await api.read_current("user")] == ["任务一", "任务二"]

    async def test_supersede_can_be_disabled(self, store: Any) -> None:
        """M4：关掉取代开关则同 task 多条并存（可配，不写死策略）。"""
        api = ScopedMemoryAPI(store, scope_key="u-1", supersede_same_task=False)
        await api.write_session_summary("t-1", "user", "第一版")
        await api.write_session_summary("t-1", "user", "第二版")
        assert len(await api.read_current("user")) == 2


class TestRuntimeSeparation:
    async def test_memory_write_does_not_touch_snapshots(self, store: Any) -> None:
        """M5：写长期记忆不落运行态表（memory-rules 第 3 条，物理分表实证）。"""
        api = ScopedMemoryAPI(store, scope_key="u-1")
        await api.write_session_summary("t-1", "user", "摘要")
        async with store.connection.execute("SELECT COUNT(*) FROM snapshots") as cur:
            assert (await cur.fetchone())[0] == 0


class TestEmptySummary:
    async def test_blank_summary_skipped(self, store: Any) -> None:
        """M6：空/空白摘要跳过写入，不产空事实。"""
        api = ScopedMemoryAPI(store, scope_key="u-1")
        await api.write_session_summary("t-1", "user", "")
        await api.write_session_summary("t-2", "user", "   ")
        assert await api.read_current("user") == []
