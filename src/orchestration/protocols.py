"""编排层 Protocol 接口定义（Task 10BC）。

放独立文件以避免 desktop_graph.py 与 state.py 之间的循环 import。

Protocols:
  MemoryAPI   — 记忆写入（唯一写入点 memory_flush_node，scope=session 显式）
  SnapshotStore — 快照存取（re-export 自 src.agents.protocols，权威定义在 agents 层，
                 消除双定义技术债；此处 re-export 供 desktop_graph.py 使用）
  StepArchive — 历史步骤归档（Supervisor 截断时使用，定义在 state.py 移至此作别名）

层约束：
  - 三层单向依赖：orchestration → agents.models；Protocol 打桩不直连记忆/图谱层。
  - MemoryAPI.write_session_summary scope 参数类型约束为 Literal，禁止默认 user。
  - Agent 节点不得直接调用 MemoryAPI，只有 memory_flush_node 调用。
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from src.agents.protocols import SnapshotStore
from src.orchestration.state import StepRecord

# ── MemoryAPI Protocol ─────────────────────────────────────────────────────────


@runtime_checkable
class MemoryAPI(Protocol):
    """记忆写入接口（Protocol 打桩，实现由 eng-team / 记忆层接线）。

    设计约束（memory-rules.md）：
    - 记忆写入只在「任务完成」节点（memory_flush_node），不在每步写入。
    - scope 必须显式指定，禁止默认 user（防止记忆串味/泄漏）。
    - Agent 节点不得直接持有 MemoryAPI 引用。
    """

    async def write_session_summary(
        self,
        task_id: str,
        scope: Literal["session", "user", "group"],
        summary: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """写入任务会话摘要。

        Args:
            task_id: 任务唯一 ID（与 LangGraph thread_id 对应）。
            scope: 记忆作用域，必须显式指定（session/user/group），禁止默认 user。
            summary: 任务摘要文本。
            metadata: 可选附加元数据（执行状态、步骤数等）。
        """
        ...


class NoopMemoryAPI:
    """MemoryAPI 无操作打桩（测试/开发环境使用，不接真实记忆后端）。"""

    async def write_session_summary(
        self,
        task_id: str,
        scope: Literal["session", "user", "group"],
        summary: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """无操作（打桩），记录调用参数供测试断言。"""


# ── SnapshotStore Protocol（re-export，权威定义在 src.agents.protocols） ────────
# 双定义技术债已消除：SnapshotStore 权威定义收归 agents 层（src/agents/protocols.py），
# 此处 re-export 供 desktop_graph.py 停滞检测使用（orchestration → agents 下调允许）。
# 保留 __all__ 中的 "SnapshotStore" 以维持既有 import 路径不破。


# ── IncidentReporter Protocol ──────────────────────────────────────────────────


@runtime_checkable
class IncidentReporter(Protocol):
    """事件上报接口（Protocol 打桩，error_report_node 使用）。

    实现由 eng-team 接线（告警/日志聚合/监控系统等）。
    """

    async def report(
        self,
        task_id: str,
        stall_count: int,
        errors: dict[str, str | None],
        snapshot_ref: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """上报错误/停滞事件。

        Args:
            task_id: 任务唯一 ID。
            stall_count: 当前停滞计数。
            errors: 错误信息字典（perception_error / control_error 等）。
            snapshot_ref: 最新感知快照 ID（用于事后排查）。
            metadata: 可选附加信息。
        """
        ...


class NoopIncidentReporter:
    """IncidentReporter 无操作打桩（测试/开发环境使用）。"""

    async def report(
        self,
        task_id: str,
        stall_count: int,
        errors: dict[str, str | None],
        snapshot_ref: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """无操作（打桩），记录调用参数供测试断言。"""


# ── StepArchive 别名（唯一真相在 state.py） ───────────────────────────────────
# desktop_graph.py 可统一从此处 import，避免混乱。

__all__ = [
    "MemoryAPI",
    "NoopMemoryAPI",
    "SnapshotStore",
    "IncidentReporter",
    "NoopIncidentReporter",
    "StepRecord",
]
