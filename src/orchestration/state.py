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
- StepRecord / append_step 权威定义在 src.agents.models.step_record（code-review
  F2 根治：原先本体在此文件、被 src/agents/*.py 反向 import 违反三层单向依赖，
  现挪到 agents 与 orchestration 共同下调的契约层，本文件从那里 re-export）。
  既有 `from src.orchestration.state import StepRecord, append_step` 消费面
  （15 处调用点）无需改动，import 路径保持可用。

层依赖：本文件 import pydantic / enum / src.agents.*（下调允许），
不 import orchestration 内其他模块（避免循环 import）。
"""

from __future__ import annotations

import os
from enum import StrEnum

from pydantic import BaseModel, Field

from src.agents.models.screen_snapshot import ActionSpec
from src.agents.models.step_record import StepRecord, append_step
from src.agents.protocols import ConfirmRequest, ConfirmResponse

# re-export：编排层内其他模块可从此处 import，无需知道权威定义在
# agents.protocols / agents.models.step_record
__all__ = [
    "ConfirmRequest",
    "ConfirmResponse",
    "DesktopTaskState",
    "StepArchive",
    "StepRecord",
    "TaskStatus",
    "append_step",
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

# ── StepRecord / append_step re-export（权威定义在 src.agents.models.step_record）
# 下游模块可继续 from src.orchestration.state import StepRecord/append_step。
# __all__ 中声明即为显式 re-export（PEP 484），无需重赋值。


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
      stall_count       — **连续**停滞计数（stall_detect_node 维护：本轮有信号则
                          累加，无任何信号则归零——K5 ② 连续语义）。
      last_screen_hash  — 上次感知的屏幕 phash（停滞检测用，字符串序列化 bits）。
      counted_error_fingerprint — 信号 C 已计数的错误指纹（K5 ① 去重）。
      uia_hollow        — 当前目标窗口是否 UIA 空洞（Task 1 实测，影响提示词；
                          由 perceive 增量随快照刷新，感知失败时保留旧值）。
      target_window_handle — 定向感知目标窗口 HWND（K6，None=前台窗口）。
      capability_flags  — 能力协商结果（来自 DesktopMCPClient 缓存）。

    step_history 由 Worker 节点经 `append_step` 纯函数追加（K4：perceive /
    control 增量各自带回追加后的完整 list），Supervisor 只做截断归档。
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
    # K5 ①：信号 C 去重——上次已计入 stall_count 的错误指纹
    # （(perception_error, control_error) 的序列化形态）。错误文本在 LastValue
    # state 里会跨节点残留（perceive 成功不清 control_error），按指纹去重保证
    # 同一错误只计一次；错误清空后指纹归 None，同一错误再现视为新停滞事件。
    counted_error_fingerprint: str | None = None

    # UIA 空洞标记（Task 1 实测；K4 起由 perceive 增量随快照刷新）
    uia_hollow: bool = False

    # K6：定向感知目标窗口 HWND（None=前台窗口，现状口径零回归）。
    # ⚠ HWND 生命周期风险：跨 checkpoint resume 后句柄可能已失效（窗口关闭/
    #   重建），失效表现为 screen_snapshot 调用失败，经既有 perception_error
    #   路径回报，不需新错误通道。
    # ⚠ 任务中途切换 target 会使下一次 stall 信号 A 比对失真一轮：新旧窗口
    #   画面必然不同 → 误判「有进展」（保守方向——只会少计停滞，不会误杀）。
    target_window_handle: int | None = None

    # 能力协商结果（来自 DesktopMCPClient 缓存）
    capability_flags: dict[str, bool] = Field(default_factory=dict)
