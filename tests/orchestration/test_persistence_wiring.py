"""编排层组装根单测：persistent_stores 的开关语义与端到端接线。

覆盖：
  W1. env 未配 → 打桩（NoopMemoryAPI + snapshot_store=None），**零回归**。
  W2. 配了 db 但缺 scope_key → **fail-fast**（不静默退打桩）。
  W3. 配齐 → 产真实现，且退出后连接已关。
  W4. 显式入参优先于 env。
  W5. 端到端：经组装根拿到的 memory_api 写入后，事实真落库且作用域正确。
  W6. 端到端：经组装根拿到的 snapshot_store 存取快照 round-trip。
  W7. 异常路径也关连接（不泄漏句柄）。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from src.agents.models.screen_snapshot import ScreenSnapshot
from src.memory import ScopedMemoryAPI
from src.orchestration.persistence import (
    ENV_DB_PATH,
    ENV_SCOPE_KEY,
    persistent_stores,
)
from src.orchestration.protocols import NoopMemoryAPI
from src.storage import SqliteMemoryStore, SqliteSnapshotStore


def _make_snapshot(snapshot_id: str = "snap-w") -> ScreenSnapshot:
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
        capability_flags={},
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """每例从干净 env 起（防开发机上真配了持久化污染测试）。"""
    monkeypatch.delenv(ENV_DB_PATH, raising=False)
    monkeypatch.delenv(ENV_SCOPE_KEY, raising=False)


class TestDisabledByDefault:
    async def test_env_unset_yields_stubs(self) -> None:
        """W1：env 未配 → 打桩且 enabled=False（零回归：既有调用行为不变）。"""
        async with persistent_stores() as bundle:
            assert bundle.enabled is False
            assert isinstance(bundle.memory_api, NoopMemoryAPI)
            assert bundle.snapshot_store is None
            # 打桩仍可调用（不抛），保证注入后图能跑
            await bundle.memory_api.write_session_summary("t", "session", "s")

    async def test_empty_db_path_treated_as_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """W1：空串等同未配（防 `ZERO_MCP_PERSISTENCE_DB=` 被误当成开启）。"""
        monkeypatch.setenv(ENV_DB_PATH, "")
        async with persistent_stores() as bundle:
            assert bundle.enabled is False


class TestFailFastOnMissingScopeKey:
    async def test_db_without_scope_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """W2：开了持久化却缺 scope_key → ValueError，**不静默退打桩**。

        这是本仓反复踩过的「绿灯不响」类故障的预防：静默退化会让接线方以为记忆已开、
        实则整轮没写。
        """
        monkeypatch.setenv(ENV_DB_PATH, ":memory:")
        with pytest.raises(ValueError, match=ENV_SCOPE_KEY):
            async with persistent_stores():
                pass

    async def test_blank_scope_key_also_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """W2：空白 scope_key 同样 fail-fast（不是只判 None）。"""
        monkeypatch.setenv(ENV_DB_PATH, ":memory:")
        monkeypatch.setenv(ENV_SCOPE_KEY, "   ")
        with pytest.raises(ValueError):
            async with persistent_stores():
                pass


class TestEnabledPath:
    async def test_yields_real_implementations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """W3：配齐 → 真实现（非打桩），enabled=True。"""
        monkeypatch.setenv(ENV_DB_PATH, ":memory:")
        monkeypatch.setenv(ENV_SCOPE_KEY, "u-1")
        async with persistent_stores() as bundle:
            assert bundle.enabled is True
            assert isinstance(bundle.memory_api, ScopedMemoryAPI)
            assert isinstance(bundle.snapshot_store, SqliteSnapshotStore)

    async def test_explicit_args_override_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """W4：显式入参优先于 env（便于测试/多租户接线）。"""
        monkeypatch.setenv(ENV_DB_PATH, "")  # env 说关
        async with persistent_stores(db_path=":memory:", scope_key="u-x") as bundle:
            assert bundle.enabled is True
            assert isinstance(bundle.memory_api, ScopedMemoryAPI)

    async def test_connection_closed_on_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """W3：退出后连接已关（再用即抛，证明不是「忘了关」）。"""
        monkeypatch.setenv(ENV_DB_PATH, ":memory:")
        monkeypatch.setenv(ENV_SCOPE_KEY, "u-1")
        async with persistent_stores() as bundle:
            store: Any = bundle.snapshot_store
            conn = store.connection
        # aiosqlite 对已关闭连接抛 ValueError("no active connection")——精确断言而非盲断 Exception，
        # 否则「测试写错导致的任意异常」也会让本例假绿。
        with pytest.raises(ValueError, match="no active connection"):
            await conn.execute("SELECT 1")

    async def test_connection_closed_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """W7：CM 体内抛异常也关连接（finally 路径，不泄漏句柄）。"""
        monkeypatch.setenv(ENV_DB_PATH, ":memory:")
        monkeypatch.setenv(ENV_SCOPE_KEY, "u-1")
        captured: Any = None
        with pytest.raises(RuntimeError, match="boom"):
            async with persistent_stores() as bundle:
                captured = bundle.snapshot_store.connection
                raise RuntimeError("boom")
        with pytest.raises(ValueError, match="no active connection"):
            await captured.execute("SELECT 1")


class TestEndToEndWiring:
    async def test_memory_write_lands_in_db(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """W5：经组装根的 memory_api 写入 → 事实真落库，作用域为传入的 scope_key。

        这条是「通电」的实证：不是断言对象类型，而是断言**数据真的到了存储层**。
        """
        monkeypatch.setenv(ENV_DB_PATH, ":memory:")
        monkeypatch.setenv(ENV_SCOPE_KEY, "u-wired")
        async with persistent_stores() as bundle:
            # 模拟 memory_flush_node 的调用形状（scope="session" 显式）
            await bundle.memory_api.write_session_summary(
                task_id="task-1",
                scope="session",
                summary="任务完成摘要",
                metadata={"step_count": 3},
            )
            store = SqliteMemoryStore(bundle.snapshot_store.connection)
            facts = await store.query_facts("session", "u-wired")

        assert [f.content for f in facts] == ["任务完成摘要"]
        assert facts[0].task_id == "task-1"
        assert facts[0].metadata == {"step_count": 3}

    async def test_snapshot_round_trip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """W6：经组装根的 snapshot_store 存取快照 round-trip（替换 InMemory 打桩后仍可用）。"""
        monkeypatch.setenv(ENV_DB_PATH, ":memory:")
        monkeypatch.setenv(ENV_SCOPE_KEY, "u-1")
        async with persistent_stores() as bundle:
            ref = await bundle.snapshot_store.save(_make_snapshot("snap-e2e"))
            loaded = await bundle.snapshot_store.load(ref)
        assert ref == "snap-e2e" and loaded.snapshot_id == "snap-e2e"

    async def test_file_backend_persists_across_sessions(self) -> None:
        """W5 补强：文件后端下**跨 CM 生命周期**数据仍在（证明真落盘，非内存假象）。"""
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "sub" / "p.db")  # 父目录不存在 → 应自动创建
            async with persistent_stores(db_path=db, scope_key="u-p") as first:
                await first.memory_api.write_session_summary("t-1", "user", "跨会话事实")
            async with persistent_stores(db_path=db, scope_key="u-p") as second:
                facts = await second.memory_api.read_current("user")
            assert [f.content for f in facts] == ["跨会话事实"]
