"""memory_flush_node 单元测试（Task 10BC）。

覆盖：
  - MemoryAPI mock 被调用，scope="session"（显式，不默认 user）
  - write_session_summary 参数正确（task_id / scope / summary / metadata）
  - StepArchive mock 被调用，步骤全量归档
  - 无 Neo4j/向量库依赖（纯 mock，不接真实存储）
  - memory_api 调用失败时优雅降级（不崩溃，返回 task_status）
  - 返回增量只含 task_status（唯一写记忆点，不修改其他字段）
  - scope 只接受 session/user/group（Literal 约束，memory-rules.md）
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from src.orchestration.desktop_graph import make_memory_flush_node
from src.orchestration.protocols import NoopMemoryAPI
from src.orchestration.state import DesktopTaskState, StepArchive, StepRecord, TaskStatus

# ── 辅助构造 ──────────────────────────────────────────────────────────────────


def _make_step(agent: str = "perceive", step_index: int = 0) -> StepRecord:
    """构造 StepRecord 测试实例。"""
    return StepRecord(
        step_index=step_index,
        agent=agent,
        instruction="测试指令",
        snapshot_ref=None,
        perception_summary=None,
        control_error=None,
        perception_error=None,
        task_status=TaskStatus.RUNNING,
    )


def _make_state(**kwargs: object) -> DesktopTaskState:
    """构造 DesktopTaskState，只覆盖指定字段。"""
    defaults: dict[str, object] = {
        "task_id": "task-mem-001",
        "task_description": "记忆刷新测试任务",
        "task_status": TaskStatus.DONE,
        "stall_count": 0,
        "perception_error": None,
        "control_error": None,
        "snapshot_ref": None,
        "perception_summary": None,
        "step_history": [],
    }
    defaults.update(kwargs)
    return DesktopTaskState(**defaults)


def _make_mock_memory_api() -> MagicMock:
    """构造 MemoryAPI mock（write_session_summary 为 AsyncMock）。"""
    mock = MagicMock()
    mock.write_session_summary = AsyncMock()
    return mock


def _make_mock_step_archive() -> MagicMock:
    """构造 StepArchive mock（archive 为 AsyncMock）。"""
    mock = MagicMock()
    mock.archive = AsyncMock()
    return mock


# ── 核心：scope=session 显式验证 ──────────────────────────────────────────────


class TestMemoryFlushScope:
    """memory_flush_node 必须以 scope='session' 调用 MemoryAPI（memory-rules.md）。"""

    async def test_write_called_with_scope_session(self) -> None:
        """write_session_summary 以 scope='session' 被调用（不默认 user）。"""
        mock_api = _make_mock_memory_api()
        state = _make_state(task_id="task-scope-test", task_status=TaskStatus.DONE)

        node = make_memory_flush_node(memory_api=mock_api)
        await node(state)

        mock_api.write_session_summary.assert_called_once()
        _, kwargs = mock_api.write_session_summary.call_args
        # scope 必须显式为 session，不得为 user 或 group
        assert kwargs.get("scope") == "session", (
            f"scope 应为 'session'，实际为 {kwargs.get('scope')!r}。"
            "违反 memory-rules.md：记忆读写必须显式指定 scope，禁止默认 user。"
        )

    async def test_scope_is_not_user(self) -> None:
        """scope 不得为 'user'（防止跨会话记忆泄漏）。"""
        mock_api = _make_mock_memory_api()
        state = _make_state()

        node = make_memory_flush_node(memory_api=mock_api)
        await node(state)

        _, kwargs = mock_api.write_session_summary.call_args
        assert kwargs.get("scope") != "user"

    async def test_scope_is_not_group(self) -> None:
        """scope 不得为 'group'（单任务 session 不应使用 group 作用域）。"""
        mock_api = _make_mock_memory_api()
        state = _make_state()

        node = make_memory_flush_node(memory_api=mock_api)
        await node(state)

        _, kwargs = mock_api.write_session_summary.call_args
        assert kwargs.get("scope") != "group"


# ── write_session_summary 参数验证 ────────────────────────────────────────────


class TestMemoryFlushWriteArguments:
    """write_session_summary 调用参数正确性测试。"""

    async def test_task_id_passed_correctly(self) -> None:
        """task_id 参数与 state.task_id 一致。"""
        mock_api = _make_mock_memory_api()
        state = _make_state(task_id="specific-task-123")

        node = make_memory_flush_node(memory_api=mock_api)
        await node(state)

        _, kwargs = mock_api.write_session_summary.call_args
        assert kwargs.get("task_id") == "specific-task-123"

    async def test_summary_is_non_empty_string(self) -> None:
        """summary 为非空字符串。"""
        mock_api = _make_mock_memory_api()
        state = _make_state(task_description="测试任务描述")

        node = make_memory_flush_node(memory_api=mock_api)
        await node(state)

        _, kwargs = mock_api.write_session_summary.call_args
        summary = kwargs.get("summary", "")
        assert isinstance(summary, str)
        assert len(summary) > 0

    async def test_summary_contains_task_id(self) -> None:
        """summary 包含 task_id 信息。"""
        mock_api = _make_mock_memory_api()
        state = _make_state(task_id="task-in-summary")

        node = make_memory_flush_node(memory_api=mock_api)
        await node(state)

        _, kwargs = mock_api.write_session_summary.call_args
        summary = kwargs.get("summary", "")
        assert "task-in-summary" in summary

    async def test_summary_contains_task_status(self) -> None:
        """summary 包含最终 task_status 信息。"""
        mock_api = _make_mock_memory_api()
        state = _make_state(task_status=TaskStatus.DONE)

        node = make_memory_flush_node(memory_api=mock_api)
        await node(state)

        _, kwargs = mock_api.write_session_summary.call_args
        summary = kwargs.get("summary", "")
        assert "DONE" in summary

    async def test_metadata_passed_as_dict(self) -> None:
        """metadata 参数为 dict（含 step_count / stall_count / task_status）。"""
        mock_api = _make_mock_memory_api()
        steps = [_make_step(step_index=i) for i in range(3)]
        state = _make_state(step_history=steps, stall_count=1)

        node = make_memory_flush_node(memory_api=mock_api)
        await node(state)

        _, kwargs = mock_api.write_session_summary.call_args
        metadata = kwargs.get("metadata")
        assert isinstance(metadata, dict)
        assert "step_count" in metadata
        assert metadata["step_count"] == 3
        assert "stall_count" in metadata
        assert metadata["stall_count"] == 1

    async def test_write_called_exactly_once(self) -> None:
        """write_session_summary 只调用一次（单次写入，不重复）。"""
        mock_api = _make_mock_memory_api()
        state = _make_state()

        node = make_memory_flush_node(memory_api=mock_api)
        await node(state)

        assert mock_api.write_session_summary.call_count == 1

    async def test_failed_status_also_writes_memory(self) -> None:
        """FAILED 状态也触发记忆写入（不只 DONE 才写）。"""
        mock_api = _make_mock_memory_api()
        state = _make_state(task_status=TaskStatus.FAILED, perception_error="连接失败")

        node = make_memory_flush_node(memory_api=mock_api)
        await node(state)

        mock_api.write_session_summary.assert_called_once()
        _, kwargs = mock_api.write_session_summary.call_args
        assert kwargs.get("scope") == "session"


# ── StepArchive 归档验证 ──────────────────────────────────────────────────────


class TestMemoryFlushArchive:
    """memory_flush_node 调用 StepArchive.archive 全量归档。"""

    async def test_archive_called_with_all_steps(self) -> None:
        """archive 以全量 step_history 被调用。"""
        mock_api = _make_mock_memory_api()
        mock_archive = _make_mock_step_archive()
        steps = [_make_step(step_index=i) for i in range(5)]
        state = _make_state(task_id="archive-task", step_history=steps)

        node = make_memory_flush_node(memory_api=mock_api, step_archive=mock_archive)
        await node(state)

        mock_archive.archive.assert_called_once()
        _, kwargs = mock_archive.archive.call_args
        assert kwargs.get("task_id") == "archive-task"
        archived_steps = kwargs.get("steps", [])
        assert len(archived_steps) == 5

    async def test_archive_called_even_with_empty_history(self) -> None:
        """step_history 为空时 archive 仍被调用（归档空列表）。"""
        mock_api = _make_mock_memory_api()
        mock_archive = _make_mock_step_archive()
        state = _make_state(step_history=[])

        node = make_memory_flush_node(memory_api=mock_api, step_archive=mock_archive)
        await node(state)

        mock_archive.archive.assert_called_once()

    async def test_archive_task_id_matches_state(self) -> None:
        """archive 的 task_id 与 state.task_id 一致。"""
        mock_api = _make_mock_memory_api()
        mock_archive = _make_mock_step_archive()
        state = _make_state(task_id="specific-archive-id")

        node = make_memory_flush_node(memory_api=mock_api, step_archive=mock_archive)
        await node(state)

        _, kwargs = mock_archive.archive.call_args
        assert kwargs.get("task_id") == "specific-archive-id"


# ── 无 Neo4j / 无真实存储依赖 ─────────────────────────────────────────────────


class TestMemoryFlushNoRealStorage:
    """验证 memory_flush_node 不直连 Neo4j/向量库（纯 Protocol 打桩）。"""

    async def test_noop_memory_api_does_not_raise(self) -> None:
        """NoopMemoryAPI（打桩）不抛出异常，正常完成写入。"""
        noop_api = NoopMemoryAPI()
        state = _make_state()

        node = make_memory_flush_node(memory_api=noop_api)
        result = await node(state)  # 不应抛出

        assert "task_status" in result

    async def test_noop_step_archive_does_not_raise(self) -> None:
        """StepArchive 打桩（无操作）不抛出异常。"""
        mock_api = _make_mock_memory_api()
        noop_archive = StepArchive()  # 默认打桩（无操作）
        state = _make_state(step_history=[_make_step()])

        node = make_memory_flush_node(memory_api=mock_api, step_archive=noop_archive)
        result = await node(state)

        assert "task_status" in result

    async def test_memory_written_via_protocol_interface(self) -> None:
        """节点只通过 MemoryAPI Protocol 接口写记忆，不直连底层驱动。

        验证方式：MemoryAPI mock 被正常调用，返回正确增量字段。
        节点不持有任何底层连接句柄（orchestration-rules 封装边界）。
        """
        mock_api = _make_mock_memory_api()
        state = _make_state()

        node = make_memory_flush_node(memory_api=mock_api)
        result = await node(state)

        # mock 被调用 = 经过 Protocol 接口（非底层驱动直调）
        mock_api.write_session_summary.assert_called_once()
        assert "task_status" in result


# ── 优雅降级（写入失败不崩溃） ───────────────────────────────────────────────


class TestMemoryFlushGracefulDegradation:
    """write_session_summary 或 archive 失败时优雅降级，不崩溃。"""

    async def test_memory_write_failure_does_not_raise(self) -> None:
        """MemoryAPI 抛出异常 → 节点 catch，返回正常增量，不崩溃。"""
        mock_api = MagicMock()
        mock_api.write_session_summary = AsyncMock(side_effect=RuntimeError("Neo4j 连接失败"))

        state = _make_state()
        node = make_memory_flush_node(memory_api=mock_api)

        # 不应抛出
        result = await node(state)
        assert "task_status" in result

    async def test_archive_failure_does_not_raise(self) -> None:
        """StepArchive.archive 抛出异常 → 节点 catch，返回正常增量，不崩溃。"""
        mock_api = _make_mock_memory_api()
        mock_archive = MagicMock()
        mock_archive.archive = AsyncMock(side_effect=RuntimeError("归档失败"))

        state = _make_state()
        node = make_memory_flush_node(memory_api=mock_api, step_archive=mock_archive)

        result = await node(state)
        assert "task_status" in result

    async def test_both_fail_still_returns_task_status(self) -> None:
        """memory_api 和 archive 都失败 → 仍返回正确 task_status 增量。"""
        mock_api = MagicMock()
        mock_api.write_session_summary = AsyncMock(side_effect=RuntimeError("记忆写入失败"))
        mock_archive = MagicMock()
        mock_archive.archive = AsyncMock(side_effect=RuntimeError("归档失败"))

        state = _make_state(task_status=TaskStatus.DONE)
        node = make_memory_flush_node(memory_api=mock_api, step_archive=mock_archive)

        result = await node(state)
        assert result["task_status"] == TaskStatus.DONE


# ── 返回增量验证 ──────────────────────────────────────────────────────────────


class TestMemoryFlushReturnIncrement:
    """memory_flush_node 只返回 task_status 增量，不修改其他字段。"""

    async def test_returns_only_task_status(self) -> None:
        """返回增量只包含 task_status 字段（节点签名约束）。"""
        mock_api = _make_mock_memory_api()
        state = _make_state(task_status=TaskStatus.DONE)

        node = make_memory_flush_node(memory_api=mock_api)
        result = await node(state)

        assert set(result.keys()) == {"task_status"}

    async def test_done_status_preserved_in_return(self) -> None:
        """DONE 状态原样返回（不改为 FAILED）。"""
        mock_api = _make_mock_memory_api()
        state = _make_state(task_status=TaskStatus.DONE)

        node = make_memory_flush_node(memory_api=mock_api)
        result = await node(state)

        assert result["task_status"] == TaskStatus.DONE

    async def test_failed_status_preserved_in_return(self) -> None:
        """FAILED 状态原样返回（唯一写记忆点，不修改终态）。"""
        mock_api = _make_mock_memory_api()
        state = _make_state(task_status=TaskStatus.FAILED)

        node = make_memory_flush_node(memory_api=mock_api)
        result = await node(state)

        assert result["task_status"] == TaskStatus.FAILED

    async def test_default_noop_when_no_api_injected(self) -> None:
        """未注入 memory_api 时使用 NoopMemoryAPI 打桩，正常返回。"""
        state = _make_state(task_status=TaskStatus.DONE)

        # 不传 memory_api → 使用 NoopMemoryAPI 打桩
        node = make_memory_flush_node()
        result = await node(state)

        assert result["task_status"] == TaskStatus.DONE
