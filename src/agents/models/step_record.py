"""单步执行记录契约（`src/agents/models/` 共享契约层，project-root.md 已认证）。

契约判据（决定本文件放这里而非 `src/orchestration/state.py`）：
- **纯数据形状 + 无 I/O 的窗口序号 append 辅助**——`StepRecord` 是 pydantic
  数据模型，`append_step` 是不做任何网络/磁盘 I/O 的纯函数（只读 `prev` 列表、
  返回新列表），不依赖 orchestration 层的图/节点/Checkpointer 语义。
- **无上层依赖**：本文件不 import `src.orchestration.*` 或 `src.memory.*`，
  可被 `src/agents/*`（Worker Agent）与 `src/orchestration/*`（State 容器）
  双向共用而不构成跨层反向依赖——被 agents 层与 orchestration 层共同消费，
  权威定义放在两者共同的下调目标 `src/agents/models/` 才不产生
  agents → orchestration 的反向 import（code-review F2 根治方案）。

step_index 语义警示（原 `src/orchestration/state.py` docstring 原文保留）：
`step_index` 取 `len(prev)`，是「当前保留窗口内」的序号——Supervisor 截断
`step_history` 后序号从截断处重新计数（窗口索引语义，**非全局单调步号**），
消费方不应把 `step_index` 当跨截断周期的唯一键使用。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = ["StepRecord", "append_step"]


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


def append_step(
    prev: list[StepRecord],
    *,
    agent: str,
    instruction: str,
    increment: dict[str, Any],
    task_status: str,
) -> list[StepRecord]:
    """将本步执行记录追加到 step_history，返回**新 list**（K4 ①，纯函数）。

    不 mutate `prev`——interrupt 重放时节点整体重跑，若原地 append 会在重放中
    重复追加同一步；返回新 list（LastValue 覆写语义）保证重放确定性安全。

    step_index 取 `len(prev)`：是「当前保留窗口内」的序号——Supervisor 截断
    step_history 后序号从截断处重新计数（窗口索引语义，非全局单调步号）。

    Args:
        prev: 追加前的 step_history（不被修改）。
        agent: 本步执行的 Worker 名（"perceive" / "control" 等）。
        instruction: 本步执行的指令摘要（通常取 state.current_instruction）。
        increment: 本步节点即将返回的 state 增量——从中提取 snapshot_ref /
            perception_summary / perception_error / control_error 四个记录字段
            （缺失键按 None 记录）。
        task_status: 本步执行后的 task_status 快照（增量未改状态时传当前值）。

    Returns:
        追加了本步 StepRecord 的新 list（长度 = len(prev) + 1）。
    """
    record = StepRecord(
        step_index=len(prev),
        agent=agent,
        instruction=instruction,
        snapshot_ref=increment.get("snapshot_ref"),
        perception_summary=increment.get("perception_summary"),
        control_error=increment.get("control_error"),
        perception_error=increment.get("perception_error"),
        task_status=str(task_status),
    )
    return [*prev, record]
