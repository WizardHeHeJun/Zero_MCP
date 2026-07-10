"""编排层 State 定义（Task 10A）。

DesktopTaskState 是贯穿整个桌面任务执行图的核心 state 容器，
遵循 LangGraph Pydantic BaseModel state 模式（SDK 核验 §1b）。

设计约束：
- step_history 用 LastValue（不加 operator.add reducer，R2 决策）：
  每个 Worker 节点返回新的完整 step_history，Supervisor 截断时直接
  返回截断后的完整 list。理由：operator.add 追加语义与「Supervisor
  截断需替换整个 list」冲突；LastValue 覆写语义更直白。
- snapshot_ref: str | None，大对象（ScreenSnapshot 本体）不进 state，
  经 SnapshotStore Protocol 外存（orchestration-rules）。
- 模型 ID 走 .env，不硬编码（agent-framework-rules）。
- ConfirmRequest / ConfirmResponse 权威定义在 src.agents.protocols（agents 层），
  本文件从那里 re-export，供编排层内部（graph.py 等）使用。
  orchestration 下调 agents.protocols 符合三层单向依赖方向。

层依赖：本文件 import pydantic / enum / src.agents.*（下调允许），
不 import orchestration 内其他模块（避免循环 import）。
"""

from __future__ import annotations

import os
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from src.agents.models.screen_snapshot import ActionSpec
from src.agents.protocols import ConfirmRequest, ConfirmResponse

# re-export：编排层内其他模块可从此处 import，无需知道权威定义在 agents.protocols
__all__ = [
    "ConfirmRequest",
    "ConfirmResponse",
    "DesktopTaskState",
    "StepArchive",
    "StepRecord",
    "TaskStatus",
]

# ── TaskStatus StrEnum ─────────────────────────────────────────────────────────


class TaskStatus(StrEnum):
    """桌面任务执行状态（Supervisor 输出 + 条件边判断依据）。

    RUNNING        — 任务进行中（默认初始状态）
    WAITING_CONFIRM — 等待人工确认高危动作（interrupt 挂起状态）
    STALLED        — 检测到停滞（stall_count 达阈值）
    DONE           — 任务成功完成
    FAILED         — 任务失败（错误/人工拒绝/停滞超限）
    """

    RUNNING = "RUNNING"
    WAITING_CONFIRM = "WAITING_CONFIRM"
    STALLED = "STALLED"
    DONE = "DONE"
    FAILED = "FAILED"


# ── ConfirmRequest / ConfirmResponse re-export（权威定义在 src.agents.protocols）
# 下游模块可继续 from src.orchestration.state import ConfirmRequest/ConfirmResponse。
# __all__ 中声明即为显式 re-export（PEP 484），无需重赋值。


# ── StepRecord ────────────────────────────────────────────────────────────────


class StepRecord(BaseModel):
    """单步执行记录，追加到 step_history。

    字段设计以「可序列化、不含大对象」为原则（orchestration-rules）。
    snapshot_ref 只存 ID，ScreenSnapshot 本体经 SnapshotStore 外存。
    """

    step_index: int
    agent: str  # "perceive" | "control" | "supervisor" | 等
    instruction: str  # 本步执行的指令摘要
    snapshot_ref: str | None  # 感知快照 ID（外存引用）
    perception_summary: str | None  # 感知摘要（可选，截断由 prompt_loader 处理）
    control_error: str | None  # 控制错误信息（None 表示成功）
    perception_error: str | None  # 感知错误信息（None 表示成功）
    task_status: str  # 执行后 task_status 快照
    metadata: dict[str, Any] = Field(default_factory=dict)  # 扩展字段


# ── StepArchive Protocol 打桩 ─────────────────────────────────────────────────
# 放此处以避免 graph.py 与 state.py 之间的循环 import。
# desktop_graph.py（Task 10BC）应从此处 import StepArchive。


class StepArchive:
    """历史步骤归档接口打桩（Protocol，实现由 eng-team 接线）。

    Supervisor 截断 step_history 时将超出 STATE_STEP_KEEP 的步骤归档，
    避免 state 无限增长。打桩实现（无操作）用于测试环境。

    工程假设：归档存储由 eng-team 接线（Postgres / 文件），
    不直连存储层（Protocol 打桩）。
    """

    async def archive(
        self,
        task_id: str,
        steps: list[StepRecord],
    ) -> None:
        """将超出保留窗口的步骤归档（打桩：无操作）。

        Args:
            task_id: 任务 ID（归档分组依据）。
            steps: 待归档的 StepRecord 列表（已超出保留窗口的老步骤）。
        """


# ── DesktopTaskState ──────────────────────────────────────────────────────────

# 环境配置
STATE_STEP_KEEP: int = int(os.environ.get("STATE_STEP_KEEP", "20"))
"""step_history 最多保留的最近步数（超出后截断，旧步骤经 StepArchive 归档）。"""


class DesktopTaskState(BaseModel):
    """桌面任务执行图的核心 State（蓝图 §5.1 全字段，Task 10A 落地）。

    LangGraph Pydantic BaseModel state（SDK 核验 §1b）：
    - 无 Annotated reducer 字段 → 全部 LastValue（覆写语义）。
    - step_history: list[StepRecord]（R2 决策：LastValue，不加 operator.add）。
      每个节点返回更新后的完整 step_history；Supervisor 截断时直接覆写整个 list。
    - snapshot_ref: str | None（只存 ID，大对象不进 state）。

    字段说明（按蓝图 §5.1）：
      task_id           — 任务唯一 ID，与 LangGraph thread_id 对应。
      task_description  — 原始任务描述（自然语言）。
      task_status       — 当前任务状态（TaskStatus）。
      current_instruction — Supervisor 下发给当前 Worker 的指令。
      next_agent        — 下一步路由目标节点名（Supervisor 输出）。
      step_history      — 步骤执行历史（LastValue，Supervisor 截断替换整个 list）。
      snapshot_ref      — 最新感知快照 ID（外存引用）。
      perception_summary — 最新感知摘要（原始，截断由 prompt_loader 处理）。
      perception_error  — 最新感知错误（None=无错误）。
      pending_action    — 待执行动作（Supervisor 分配给控制 Worker）。
      control_error     — 最新控制错误（None=无错误）。
      stall_count       — 连续停滞计数（stall_detect_node 累加）。
      last_screen_hash  — 上次感知的屏幕 phash（停滞检测用，字符串序列化 bits）。
      uia_hollow        — 当前目标窗口是否 UIA 空洞（Task 1 实测，影响提示词）。
      capability_flags  — 能力协商结果（来自 DesktopMCPClient 缓存）。
    """

    # 任务基础信息
    task_id: str = ""
    task_description: str = ""
    task_status: str = TaskStatus.RUNNING

    # Supervisor 输出（每轮 plan 后更新）
    current_instruction: str = ""
    next_agent: str = ""

    # 步骤历史（LastValue，R2 决策：不加 reducer，截断时直接覆写整个 list）
    step_history: list[StepRecord] = Field(default_factory=list)

    # 感知层输出（perceive_node 更新）
    snapshot_ref: str | None = None
    perception_summary: str | None = None
    perception_error: str | None = None

    # 控制层输出（control_node 更新）
    pending_action: ActionSpec | None = None
    control_error: str | None = None

    # 停滞检测（stall_detect_node 更新）
    stall_count: int = 0
    last_screen_hash: str | None = None

    # UIA 空洞标记（Task 1 实测）
    uia_hollow: bool = False

    # 能力协商结果（来自 DesktopMCPClient 缓存）
    capability_flags: dict[str, bool] = Field(default_factory=dict)
