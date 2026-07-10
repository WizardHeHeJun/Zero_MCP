"""agents 层 Protocol 接口定义与共享数据模型。

放此处以避免 agents 层反向依赖 orchestration 层（硬红线：三层单向依赖，
orchestration → agents → mcp client，禁止 agents → orchestration）。

内容：
  ActionGuardProtocol — DesktopControlAgent 持有的安全门接口（结构子类型）。
    实现类（ActionGuard）在 orchestration 层，图构建时注入，agents 层
    只持 Protocol 引用，不 import 具体实现类。
  ConfirmRequest     — interrupt 时向人工暴露的确认请求（规格书 §7.6）。
  ConfirmResponse    — 人工对 interrupt 的响应（规格书 §7.6）。
  SnapshotStore      — 快照存取接口（Agent 层不直连存储，经此 Protocol 打桩）。
    权威定义在此（agents 层）；screen_perception_agent.py 与
    orchestration/protocols.py 均从此处 import，消除双定义下 mypy 结构子型
    互不认的技术债。

层约束：本模块只 import 标准库 + pydantic + src.agents.models.screen_snapshot，
不 import 任何 orchestration / memory / storage 层。

orchestration/state.py 可从此处 import ConfirmRequest/ConfirmResponse（下调允许）；
orchestration/protocols.py re-export SnapshotStore（下调允许）。
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from src.agents.models.screen_snapshot import ActionRisk, ActionSpec, ScreenSnapshot

# ── ConfirmRequest / ConfirmResponse（规格书 §7.6） ───────────────────────────


class ConfirmRequest(BaseModel):
    """interrupt 时向人工暴露的确认请求（规格书 §7.6）。

    DESTRUCTIVE 动作在执行前触发 lg_interrupt(ConfirmRequest(...))，
    等待人工（或上游 Command(resume=ConfirmResponse(...))）响应。

    权威定义位置：src/agents/protocols.py（agents 层）。
    orchestration/state.py 从此处 re-export，供编排层使用。
    """

    action_id: str
    action_type: str
    risk_level: ActionRisk
    description: str
    coordinates: tuple[int, int] | None = None
    target_element_id: str | None = None


class ConfirmResponse(BaseModel):
    """人工（或自动化测试）对 interrupt 的响应（规格书 §7.6）。

    confirmed=True 时继续执行写操作；confirmed=False 时中止并设 FAILED。
    """

    confirmed: bool
    reason: str = ""


# ── ActionGuardProtocol ───────────────────────────────────────────────────────


@runtime_checkable
class ActionGuardProtocol(Protocol):
    """安全门接口（agents 层 Protocol，结构子类型）。

    DesktopControlAgent 只持此 Protocol 引用，不 import orchestration 层的
    ActionGuard 具体实现类。图构建时由 desktop_graph.py（orchestration 层）
    将 ActionGuard 实例注入——满足结构子类型，mypy 可验证。

    方法语义与 ActionGuard 一致（规格书 §1.2 / §1.5）：
      classify_risk  — 三级白名单风险判定，返回有效 ActionRisk。
      toctou_verify  — Pre-execution UI State Verification，返回 pass/abort。
    """

    async def classify_risk(self, action: ActionSpec) -> ActionRisk:
        """三级风险判定：白名单二次确认 + 声明风险取最高级。

        Args:
            action: 待判定的动作规格。

        Returns:
            ActionRisk（可能比 action.risk_level 更高）。
        """
        ...

    async def toctou_verify(
        self,
        action: ActionSpec,
        snapshot_before: ScreenSnapshot | None = None,
    ) -> Literal["pass", "abort"]:
        """TOCTOU 验证（Pre-execution UI State Verification）。

        Args:
            action: 待验证的动作规格。
            snapshot_before: 可选的执行前快照（已有截图则复用）。

        Returns:
            "pass"（界面稳定，可执行）或 "abort"（界面已变，拒绝执行）。
        """
        ...


# ── SnapshotStore Protocol（权威定义，agents 层） ─────────────────────────────


@runtime_checkable
class SnapshotStore(Protocol):
    """快照持久化接口（Protocol 打桩，实现由 eng-team 接线）。

    Agent 层只持接口引用，不直连 Postgres/磁盘。

    权威定义位置：src/agents/protocols.py。screen_perception_agent.py（感知节点
    使用）与 orchestration/protocols.py（desktop_graph 停滞检测使用）均从此处 import，
    避免同名 Protocol 双定义导致 mypy 结构子型互不认。
    """

    async def save(self, snapshot: ScreenSnapshot) -> str:
        """存储快照，返回快照 ID（snapshot_ref）。

        Args:
            snapshot: 完整感知快照对象。

        Returns:
            可用于 state 引用的快照 ID 字符串（等同 snapshot.snapshot_id）。
        """
        ...

    async def load(self, snapshot_id: str) -> ScreenSnapshot:
        """按 ID 加载快照。

        Args:
            snapshot_id: save() 返回的 ID。

        Returns:
            对应的 ScreenSnapshot 对象。
        """
        ...
