"""桌面任务 Supervisor Agent（Task 10A）。

职责：读 state → 调 LLM（Anthropic Claude）→ 解析 plan 输出
（next_agent / current_instruction / task_status）→ 截断 step_history
→ 返回 state 增量（仅 3 个字段 + 截断后的 step_history）。

设计约束：
- Supervisor 无业务路由判断（is_browser_task 等路由归条件边，Task 10BC）。
- 模型 ID 走 os.environ["DESKTOP_SUPERVISOR_MODEL"]，默认 claude-opus-4-8。
- 缺 ANTHROPIC_API_KEY 时 log warning + 返回 FAILED，不崩溃。
- step_history 超 STATE_STEP_KEEP 时调 StepArchive 打桩归档，返回最近 K 步完整 list。
- 节点签名 (state) -> dict，只返回增量。
- prompt_loader 依赖注入（抽象接口，不 import 具体 loader），Task 11B 实现后接入。
- I/O 全 async，不阻塞（LLM 调用用 anthropic Python SDK async 接口）。
- 公开接口完整类型注解。

层依赖：
  orchestration → agents.models（允许）；不反向 import 记忆/存储层。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Protocol

from src.orchestration.state import (
    STATE_STEP_KEEP,
    DesktopTaskState,
    StepArchive,
    StepRecord,
    TaskStatus,
)

logger = logging.getLogger(__name__)

# ── 模型 ID（走 .env，不硬编码，agent-framework-rules）────────────────────────

_DEFAULT_MODEL = "claude-opus-4-8"
DESKTOP_SUPERVISOR_MODEL: str = os.environ.get("DESKTOP_SUPERVISOR_MODEL", _DEFAULT_MODEL)

# ── 回路硬上限（K4 紧后 §3.3）──────────────────────────────────────────────────
# 设计输入 notes/2026-08-05-llm-integration-survey-k3k4-actionspec.md §3.3：
# env 可配硬上限，命中返回可区分的 failure_reason="max_iterations_exceeded"，
# 不与 LLM 判定失败混用（先例=Agent SDK agent-loop 的 error_max_turns 专属
# 子类型；computer-use-demo 的裸 while True 是反面）。
# 默认 30 为工程假设：一轮 ≈ supervisor+worker+stall 三个超步，须配合调用方
# recursion_limit ≥ 3×上限+裕量（停滞收口本身要 ~8 轮，30 覆盖其 3 倍余量）。
DESKTOP_MAX_ITERATIONS: int = int(os.environ.get("DESKTOP_MAX_ITERATIONS", "30"))

MAX_ITERATIONS_EXCEEDED: str = "max_iterations_exceeded"
"""failure_reason 机读值：回路硬上限命中（与 LLM 判定 FAILED 可区分）。"""

# ── PromptLoader Protocol（依赖注入接口，不 import 具体实现）────────────────────


class PromptLoader(Protocol):
    """Supervisor 提示词加载接口（Protocol，实现由 Task 11B 完成）。

    依赖注入而非直接 import 具体 loader，使 supervisor 可独立测试（mock loader）。
    """

    def render_supervisor(
        self,
        state: DesktopTaskState,
    ) -> tuple[str, str]:
        """渲染 Supervisor 提示词（system + user）。

        Args:
            state: 当前任务 state。

        Returns:
            (system_prompt, user_prompt) 两个字符串。
        """
        ...


class _FallbackPromptLoader:
    """PromptLoader 占位实现（Task 11B 完成前的最小可用版本）。

    不依赖 Jinja2 文件，直接组装基础提示词以保证 supervisor 可独立测试。
    Task 11B 完成后，图构建时注入真实 PromptLoader 替换此占位实现。

    ⚠ 行为分叉提醒（PR #26 审查 WARN②）：本占位不含 last_step_outcome
    三态引导（K4 紧后 §3.2 只落在真 PromptLoader + jinja2 模板一侧），错误
    只平铺进历史文本让 LLM 自己猜——恰是三态设计要规避的行为。仅供
    「直接构造 DesktopSupervisorAgent 而未接线 loader」的兜底场景；生产
    默认装配（get_graph）注入真 PromptLoader，不走此路径。
    """

    def render_supervisor(
        self,
        state: DesktopTaskState,
    ) -> tuple[str, str]:
        """渲染基础 Supervisor 提示词（占位版本，不含完整上下文截断）。

        Args:
            state: 当前任务 state。

        Returns:
            (system_prompt, user_prompt)。
        """
        system = (
            "你是桌面任务编排器（Supervisor）。根据任务描述与执行历史，"
            "决定下一步动作。\n"
            "以 JSON 格式回复，必须包含以下字段：\n"
            '  "next_agent": "perceive" | "control" | "playwright" | "done" | "error"\n'
            '  "current_instruction": "<下发给 Worker 的指令>"\n'
            '  "task_status": "RUNNING" | "WAITING_CONFIRM" | "DONE" | "FAILED"\n'
            "只返回 JSON，不加任何解释。"
        )
        # 取最近 N 步历史（截断在 prompt_loader 层，此处做基础截断）
        recent_steps = state.step_history[-10:] if state.step_history else []
        history_text = (
            "\n".join(
                f"[step {s.step_index}] agent={s.agent} status={s.task_status}"
                f" instruction={s.instruction!r}"
                + (f" control_error={s.control_error!r}" if s.control_error else "")
                + (f" perception_error={s.perception_error!r}" if s.perception_error else "")
                for s in recent_steps
            )
            or "(无历史步骤)"
        )
        user = (
            f"任务描述：{state.task_description}\n"
            f"当前状态：{state.task_status}\n"
            f"感知摘要：{state.perception_summary or '(无)'}\n"
            f"感知错误：{state.perception_error or '无'}\n"
            f"控制错误：{state.control_error or '无'}\n"
            f"停滞计数：{state.stall_count}\n"
            f"UIA 空洞：{state.uia_hollow}\n"
            f"历史步骤：\n{history_text}\n"
            "请给出下一步决策（JSON 格式）："
        )
        return system, user


# ── LLM 响应解析 ───────────────────────────────────────────────────────────────

_VALID_TASK_STATUSES: frozenset[str] = frozenset(s.value for s in TaskStatus)

_VALID_NEXT_AGENTS: frozenset[str] = frozenset(
    ["perceive", "control", "playwright", "memory_flush", "error_report", "done", "error"]
)


def _parse_plan_response(raw_text: str) -> dict[str, str]:
    """解析 LLM 返回的 JSON plan 响应，提取 3 个增量字段。

    容错：若 LLM 返回包裹在 markdown code fence 里的 JSON，自动剥除。
    字段缺失或值非法时使用安全默认值（不崩溃）。

    Args:
        raw_text: LLM 输出的原始文本。

    Returns:
        包含 next_agent / current_instruction / task_status 的字典。
    """
    text = raw_text.strip()

    # 剥除 markdown code fence（```json ... ``` 或 ``` ... ```）
    if text.startswith("```"):
        lines = text.splitlines()
        # 去掉首行（```json 或 ```）和尾行（```）
        inner_lines = lines[1:]
        if inner_lines and inner_lines[-1].strip() == "```":
            inner_lines = inner_lines[:-1]
        text = "\n".join(inner_lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("_parse_plan_response: JSON 解析失败（%s）raw=%r", exc, raw_text[:200])
        return {
            "next_agent": "error_report",
            "current_instruction": f"LLM 响应解析失败: {exc}",
            "task_status": TaskStatus.FAILED,
        }

    if not isinstance(data, dict):
        logger.warning("_parse_plan_response: 期望 dict，得到 %r", type(data))
        return {
            "next_agent": "error_report",
            "current_instruction": "LLM 响应非 dict",
            "task_status": TaskStatus.FAILED,
        }

    next_agent = str(data.get("next_agent", "error_report"))
    if next_agent not in _VALID_NEXT_AGENTS:
        logger.warning(
            "_parse_plan_response: next_agent=%r 不在有效集合，降级 error_report", next_agent
        )
        next_agent = "error_report"

    current_instruction = str(data.get("current_instruction", ""))

    task_status = str(data.get("task_status", TaskStatus.RUNNING))
    if task_status not in _VALID_TASK_STATUSES:
        logger.warning("_parse_plan_response: task_status=%r 不合法，降级 RUNNING", task_status)
        task_status = TaskStatus.RUNNING

    return {
        "next_agent": next_agent,
        "current_instruction": current_instruction,
        "task_status": task_status,
    }


# ── step_history 截断 ──────────────────────────────────────────────────────────


async def _truncate_step_history(
    task_id: str,
    step_history: list[StepRecord],
    keep: int,
    archive: StepArchive,
) -> list[StepRecord]:
    """截断 step_history，将超出保留窗口的老步骤归档。

    R2 决策：返回最近 keep 步的完整 list（LastValue 覆写整个字段）。

    Args:
        task_id: 任务 ID（归档分组依据）。
        step_history: 当前完整 step_history。
        keep: 保留最近 K 步（STATE_STEP_KEEP）。
        archive: StepArchive 打桩接口。

    Returns:
        截断后的完整 list（最多 keep 步）。
    """
    if len(step_history) <= keep:
        return step_history

    overflow = step_history[:-keep]
    retained = step_history[-keep:]

    logger.info(
        "_truncate_step_history: task_id=%r 归档 %d 步，保留 %d 步",
        task_id,
        len(overflow),
        len(retained),
    )

    try:
        await archive.archive(task_id=task_id, steps=overflow)
    except Exception as exc:
        # 归档失败不阻断主流程，只记 warning
        logger.warning("_truncate_step_history: StepArchive.archive 失败（%s）", exc)

    return retained


# ── DesktopSupervisorAgent ─────────────────────────────────────────────────────


class DesktopSupervisorAgent:
    """桌面任务 Supervisor Agent。

    职责单一：读 state → render 提示词 → 调 LLM → 解析 plan → 截断 step_history
    → 返回 3 增量字段。不含任何业务路由判断（is_browser_task 等路由在条件边函数）。

    依赖注入：
        llm_client — Anthropic AsyncAnthropic 客户端（可 mock）。
        prompt_loader — PromptLoader Protocol 实现（可 mock，Task 11B 前用占位）。
        step_archive — StepArchive 打桩（可 mock）。

    模型 ID：os.environ["DESKTOP_SUPERVISOR_MODEL"]，默认 claude-opus-4-8。
    缺 ANTHROPIC_API_KEY：log warning + 返回 FAILED，不崩溃。
    """

    def __init__(
        self,
        llm_client: Any,
        prompt_loader: PromptLoader | None = None,
        step_archive: StepArchive | None = None,
        model: str | None = None,
    ) -> None:
        """初始化 DesktopSupervisorAgent。

        Args:
            llm_client: Anthropic AsyncAnthropic 客户端实例（依赖注入，可 mock）。
                        图构建时传入；缺 ANTHROPIC_API_KEY 时可传 None（优雅回退）。
            prompt_loader: PromptLoader Protocol 实现；None 时使用占位实现。
            step_archive: StepArchive 打桩实例；None 时使用默认打桩（无操作）。
            model: 模型 ID 覆写；None 时读 DESKTOP_SUPERVISOR_MODEL 环境变量。
        """
        self.llm_client = llm_client
        self.prompt_loader: PromptLoader = prompt_loader or _FallbackPromptLoader()
        self.step_archive: StepArchive = step_archive or StepArchive()
        self.model: str = model or DESKTOP_SUPERVISOR_MODEL

    async def plan(
        self,
        state: DesktopTaskState,
        rendered_prompt: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        """调用 LLM 生成下一步 plan，返回 3 个增量字段。

        不含业务路由判断（next_agent 直接来自 LLM，条件边函数负责解释路由）。

        Args:
            state: 当前任务 state。
            rendered_prompt: 可选的预渲染提示词 (system, user)；
                             None 时调 self.prompt_loader.render_supervisor(state)。

        Returns:
            dict，含 next_agent / current_instruction / task_status 3 个字段。
            缺 ANTHROPIC_API_KEY 或 LLM 调用失败时返回 task_status=FAILED。
        """
        # 检查 API key
        if not os.environ.get("ANTHROPIC_API_KEY"):
            logger.warning("DesktopSupervisorAgent.plan: 缺 ANTHROPIC_API_KEY，任务 FAILED")
            return {
                "next_agent": "error_report",
                "current_instruction": "缺 ANTHROPIC_API_KEY，无法调用 LLM",
                "task_status": TaskStatus.FAILED,
            }

        if self.llm_client is None:
            logger.warning("DesktopSupervisorAgent.plan: llm_client 为 None，任务 FAILED")
            return {
                "next_agent": "error_report",
                "current_instruction": "llm_client 未初始化",
                "task_status": TaskStatus.FAILED,
            }

        # 渲染提示词
        if rendered_prompt is not None:
            system_prompt, user_prompt = rendered_prompt
        else:
            system_prompt, user_prompt = self.prompt_loader.render_supervisor(state)

        # 调用 LLM
        try:
            response = await self.llm_client.messages.create(
                model=self.model,
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw_text: str = response.content[0].text
            logger.debug(
                "DesktopSupervisorAgent.plan: LLM 返回 raw=%r",
                raw_text[:200],
            )
        except Exception as exc:
            logger.warning(
                "DesktopSupervisorAgent.plan: LLM 调用失败（%s）",
                exc,
            )
            return {
                "next_agent": "error_report",
                "current_instruction": f"LLM 调用失败: {exc}",
                "task_status": TaskStatus.FAILED,
            }

        return _parse_plan_response(raw_text)


# ── supervisor_node 节点函数工厂 ───────────────────────────────────────────────


def make_supervisor_node(agent: DesktopSupervisorAgent) -> Any:
    """生成 supervisor_node 节点函数（闭包注入 agent）。

    节点签名 (state: DesktopTaskState) -> dict，只返回增量。
    增量字段：next_agent / current_instruction / task_status / step_history（截断后）。

    Args:
        agent: 已构造的 DesktopSupervisorAgent 实例。

    Returns:
        符合 LangGraph 节点签名的异步函数。
    """

    async def supervisor_node(state: DesktopTaskState) -> dict[str, Any]:
        """LangGraph Supervisor 节点。

        执行流程：
          0. 终态单调守卫（K5 ③）：task_status 已是 DONE/FAILED 时不调 LLM，
             直接返回空增量 {}——route_after_supervisor 规则 1 会导向 memory_flush。
             防两类回退：终态被 LLM 新 plan 覆写回 RUNNING；人工拒绝（FAILED）后
             同一 pending_action 再次路由进 control 重复 interrupt。
          0b. 回路硬上限（K4 紧后 §3.3）：本轮为第 iteration_count+1 轮，超过
             DESKTOP_MAX_ITERATIONS 时不调 LLM，返回
             failure_reason="max_iterations_exceeded" + next_agent=error_report
             ——终态由 error_report_node 统一落（FAILED + 现场包），此处不设
             task_status，保证与 LLM 判定失败可区分。
          1. 调 agent.plan(state) 获取 3 个增量字段。
          2. 截断 step_history（超 STATE_STEP_KEEP 时调 StepArchive 归档）。
          3. 返回增量 dict（plan 3 字段 + step_history + iteration_count 递增）。

        注意：无任何业务路由判断（is_browser_task / uia_hollow 等），
              路由逻辑统一在 Task 10BC 的条件边函数中。

        Args:
            state: 当前 DesktopTaskState。

        Returns:
            state 增量字典，含 next_agent / current_instruction / task_status /
            step_history（截断后完整 list）；终态时为空 dict（无更新）。
        """
        # 0. 终态单调守卫（K5 ③）：DONE/FAILED 不再调 LLM 重新规划
        if state.task_status in (TaskStatus.DONE, TaskStatus.FAILED):
            logger.info(
                "supervisor_node: task_status=%r 已终态，跳过 LLM 直接返回空增量",
                state.task_status,
            )
            return {}

        # 0b. 回路硬上限（K4 紧后 §3.3）：命中时不调 LLM，failure_reason 可区分
        this_iteration = state.iteration_count + 1
        if this_iteration > DESKTOP_MAX_ITERATIONS:
            logger.error(
                "supervisor_node: 迭代轮次 %d 超过硬上限 DESKTOP_MAX_ITERATIONS=%d，"
                "failure_reason=%s → error_report",
                this_iteration,
                DESKTOP_MAX_ITERATIONS,
                MAX_ITERATIONS_EXCEEDED,
            )
            return {
                "next_agent": "error_report",
                "current_instruction": (
                    f"回路硬上限命中：已规划 {state.iteration_count} 轮，"
                    f"上限 {DESKTOP_MAX_ITERATIONS}（DESKTOP_MAX_ITERATIONS）"
                ),
                "failure_reason": MAX_ITERATIONS_EXCEEDED,
                "iteration_count": this_iteration,
            }

        # 1. 获取 plan 增量（3 字段）
        plan_increment = await agent.plan(state)

        # 2. 截断 step_history（R2：LastValue，返回完整 list）
        truncated_history = await _truncate_step_history(
            task_id=state.task_id,
            step_history=state.step_history,
            keep=STATE_STEP_KEEP,
            archive=agent.step_archive,
        )

        return {
            "next_agent": plan_increment["next_agent"],
            "current_instruction": plan_increment["current_instruction"],
            "task_status": plan_increment["task_status"],
            "step_history": truncated_history,
            "iteration_count": this_iteration,
        }

    return supervisor_node


# ── 顶层占位节点（图构建时需用 make_supervisor_node 注入 agent）───────────────


async def supervisor_node(state: DesktopTaskState) -> dict[str, Any]:
    """顶层 supervisor_node 占位（图构建时需用 make_supervisor_node 注入 agent）。

    此函数为 import 便利保留，不应直接注册到图（无 agent 注入会 raise RuntimeError）。

    Args:
        state: 编排 state。

    Raises:
        RuntimeError: 始终抛出，提示使用 make_supervisor_node。
    """
    raise RuntimeError(
        "supervisor_node 未注入 agent——请使用 make_supervisor_node(agent) 生成节点函数，"
        "再注册到 StateGraph。"
    )
