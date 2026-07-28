"""存储层单测：SQLite 快照存储 + 长期记忆事实存储（内存库，零 infra）。

覆盖：
  S1. schema 幂等（重复 init 不炸）。
  S2. 快照 save→load round-trip；重复 save 同 ID 幂等覆盖；缺失 ID → KeyError。
  S3. 事实只追加、query 默认只回当前有效。
  S4. invalidate 打戳**不删行**（include_invalidated 仍能取回），重复失效不改首次时间戳。
  S5. 作用域隔离：不同 scope / scope_key 互不串。
"""

from __future__ import annotations

from typing import Any

import pytest

from src.agents.models.screen_snapshot import ScreenSnapshot
from src.storage import (
    MEMORY_DB,
    SqliteMemoryStore,
    SqliteSnapshotStore,
    init_schema,
    open_connection,
)


@pytest.fixture
async def conn() -> Any:
    """内存库连接（每例独立，用完即关）。"""
    connection = await open_connection(MEMORY_DB)
    try:
        yield connection
    finally:
        await connection.close()


def _make_snapshot(snapshot_id: str = "snap-1") -> ScreenSnapshot:
    """构造最小合法 ScreenSnapshot（填齐必填字段，可空项给 None/空列表）。"""
    return ScreenSnapshot(
        snapshot_id=snapshot_id,
        timestamp_ms=1_700_000_000_000,
        screen_width=1920,
        screen_height=1080,
        active_window_title=None,
        uia_elements=[],
        text_blocks=[],
        visual_objects=[],
        screenshot_path=None,
        perception_mode="uia_ocr",
        capability_flags={"ocr": True},
    )


class TestSchema:
    async def test_init_schema_is_idempotent(self, conn: Any) -> None:
        """S1：重复建表不抛（IF NOT EXISTS）。"""
        await init_schema(conn)
        await init_schema(conn)


class TestSqliteSnapshotStore:
    async def test_save_load_round_trip(self, conn: Any) -> None:
        """S2：save 返回入参 ID；load 还原等价对象。"""
        store = SqliteSnapshotStore(conn)
        snap = _make_snapshot("snap-abc")
        assert await store.save(snap) == "snap-abc"
        loaded = await store.load("snap-abc")
        assert loaded.snapshot_id == "snap-abc"
        assert loaded.is_untrusted is True  # 契约不变式随往返保持

    async def test_save_same_id_is_idempotent(self, conn: Any) -> None:
        """S2：同 ID 重复保存覆盖而非累积/抛主键冲突。"""
        store = SqliteSnapshotStore(conn)
        await store.save(_make_snapshot("dup"))
        await store.save(_make_snapshot("dup"))
        async with conn.execute("SELECT COUNT(*) FROM snapshots WHERE snapshot_id='dup'") as cur:
            assert (await cur.fetchone())[0] == 1

    async def test_load_missing_raises_key_error(self, conn: Any) -> None:
        """S2：缺失 ID → KeyError（与打桩 InMemorySnapshotStore 行为一致，非返回 None）。"""
        store = SqliteSnapshotStore(conn)
        with pytest.raises(KeyError):
            await store.load("nope")


class TestSqliteMemoryStore:
    async def test_append_and_query_current(self, conn: Any) -> None:
        """S3：追加后可按作用域读回，默认只回当前有效、按时间升序。"""
        store = SqliteMemoryStore(conn)
        await store.append_fact("user", "u-1", "t-1", "第一条")
        await store.append_fact("user", "u-1", "t-2", "第二条")
        facts = await store.query_facts("user", "u-1")
        assert [f.content for f in facts] == ["第一条", "第二条"]
        assert all(f.invalidated_at is None for f in facts)

    async def test_invalidate_marks_not_deletes(self, conn: Any) -> None:
        """S4：失效是**打戳不删行**——默认查不到，include_invalidated 仍能取回。"""
        store = SqliteMemoryStore(conn)
        fid = await store.append_fact("session", "s-1", "t-1", "旧事实")
        assert await store.invalidate_facts([fid]) == 1

        assert await store.query_facts("session", "s-1") == []
        history = await store.query_facts("session", "s-1", include_invalidated=True)
        assert len(history) == 1 and history[0].content == "旧事实"
        assert history[0].invalidated_at is not None, "行应保留并带失效时间戳（时序语义）"

    async def test_reinvalidate_keeps_first_timestamp(self, conn: Any) -> None:
        """S4：已失效的行不重复打戳（首次失效时间稳定，审计可信）。"""
        store = SqliteMemoryStore(conn)
        fid = await store.append_fact("group", "g-1", "t-1", "x")
        await store.invalidate_facts([fid])
        first = (await store.query_facts("group", "g-1", include_invalidated=True))[0]
        assert await store.invalidate_facts([fid]) == 0  # 第二次影响 0 行
        again = (await store.query_facts("group", "g-1", include_invalidated=True))[0]
        assert again.invalidated_at == first.invalidated_at

    async def test_empty_invalidate_is_noop(self, conn: Any) -> None:
        """S4：空列表不发 SQL、返回 0（防拼出非法 IN () 语句）。"""
        assert await SqliteMemoryStore(conn).invalidate_facts([]) == 0

    async def test_scopes_are_isolated(self, conn: Any) -> None:
        """S5：scope 与 scope_key 任一不同即互不可见（防记忆串味）。"""
        store = SqliteMemoryStore(conn)
        await store.append_fact("user", "u-1", "t", "属于 u-1")
        await store.append_fact("user", "u-2", "t", "属于 u-2")
        await store.append_fact("session", "u-1", "t", "同键不同 scope")

        assert [f.content for f in await store.query_facts("user", "u-1")] == ["属于 u-1"]
        assert [f.content for f in await store.query_facts("user", "u-2")] == ["属于 u-2"]
        assert [f.content for f in await store.query_facts("session", "u-1")] == ["同键不同 scope"]

    async def test_metadata_round_trip(self, conn: Any) -> None:
        """元数据 JSON 往返；未给时为空 dict（非 None，消费侧无需判空）。"""
        store = SqliteMemoryStore(conn)
        await store.append_fact("user", "u-1", "t-1", "带元数据", {"steps": 3, "ok": True})
        await store.append_fact("user", "u-1", "t-2", "无元数据")
        facts = await store.query_facts("user", "u-1")
        assert facts[0].metadata == {"steps": 3, "ok": True}
        assert facts[1].metadata == {}
