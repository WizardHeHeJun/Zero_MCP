"""桌面任务执行图（Task 10BC）。

图连线（R3 决策，per 蓝图主研批注）：
    START → supervisor
    supervisor →route_after_supervisor→ perceive | control | playwright
                                        | memory_flush | error_report
    perceive → stall_detect            ← R3：感知失败也经统一停滞节点
    control →route_after_control→ stall_detect | supervisor
    stall_detect →route_after_stall→ supervisor | error_report
    error_report → memory_flush
    memory_flush → END

节点：
    supervisor_node  — 调 DesktopSupervisorAgent.plan，无业务路由判断。
    perceive_node    — 调 ScreenPerceptionAgent.perceive，返回感知增量。
    control_node     — 调 DesktopControlAgent.execute，含 interrupt/resume 分区。
    stall_detect_node— 三信号停滞检测（phash 不变 / 步骤重复 / 错误指纹去重计数），
                       无信号轮 stall_count 归零（K5 连续语义）。
    error_report_node— 记录错误 + incident_reporter 打桩，设 FAILED。
    memory_flush_node— 唯一记忆写入点，scope=session 显式，经 MemoryAPI Protocol。

Protocols（在 src/orchestration/protocols.py）：
    MemoryAPI / SnapshotStore / IncidentReporter

工厂：
    get_graph(checkpointer=None) → 编译好的 StateGraph（默认 InMemorySaver）。

设计约束（红线）：
    - 三层单向依赖：orchestration → agents → mcp client；不反向 import 记忆/存储层。
    - 记忆写入只在 memory_flush_node（唯一写入点，scope=session 显式）。
    - Agent 节点不直连图谱/向量库（MemoryAPI Protocol 打桩）。
    - 大对象不进 state（snapshot_ref: str，ScreenSnapshot 本体经 SnapshotStore 外存）。
    - 模型 ID 走 os.environ，不硬编码（DesktopSupervisorAgent 内处理）。
    - 节点签名 (state) -> dict，只返回增量。
    - Supervisor 无业务路由判断（is_browser_task 等在条件边函数）。
    - 条件边函数带 -> Literal[...] 注解。
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.desktop_control_agent import DesktopControlAgent, make_control_node
from src.agents.screen_perception_agent import ScreenPerceptionAgent, make_perceive_node
from src.mcp.desktop_mcp_client import DesktopMCPClient
from src.orchestration.desktop_supervisor import DesktopSupervisorAgent, make_supervisor_node
from src.orchestration.phash import average_hash_from_bytes as _compute_average_hash
from src.orchestration.phash import hamming_bits as _hamming_distance
from src.orchestration.prompt_loader import PromptLoader
from src.orchestration.protocols import (
    IncidentReporter,
    MemoryAPI,
    NoopIncidentReporter,
    NoopMemoryAPI,
    SnapshotStore,
)
from src.orchestration.safety.action_guard import ActionGuard
from src.orchestration.safety.incident_reporter import FileIncidentReporter
from src.orchestration.state import DesktopTaskState, StepArchive, TaskStatus

logger = logging.getLogger(__name__)

# ── 环境配置（工程假设，Task 12 标定） ────────────────────────────────────────

STALL_MAX_STEPS: int = int(os.environ.get("STALL_MAX_STEPS", "5"))
"""连续同 Worker 步骤超过此数触发停滞信号（步骤重复信号）。"""

STALL_THRESHOLD: int = int(os.environ.get("STALL_THRESHOLD", "3"))
"""stall_count 达到此阈值时路由至 error_report（触发停滞处理）。"""

PHASH_UNCHANGED_THRESHOLD: int = int(os.environ.get("PHASH_UNCHANGED_THRESHOLD", "10"))
"""画面 phash 汉明距离小于此值视为「画面未变」（phash 不变信号）。
工程假设：取 8x8 = 64 bits，阈值 10 ≈ 15.6% 不同位。"""

INCIDENT_STEP_WINDOW: int = int(os.environ.get("INCIDENT_STEP_WINDOW", "10"))
"""异常上报时附带的最近步骤窗口大小（error_report_node metadata.recent_steps）。"""

INCIDENT_SUMMARY_MAX_CHARS: int = 2000
"""现场包 recent_steps 中 perception_summary 的逐步截断上限（字符）。
对齐 PERCEPTION_SUMMARY_MAX_TOKENS 的字符近似口径；全文可经 snapshot_ref 回查。"""


def _truncate_step_for_incident(step: dict[str, Any]) -> dict[str, Any]:
    """截断 StepRecord dump 中的长文本字段，保证现场包单包体积有界。

    Args:
        step: StepRecord.model_dump(mode="json") 结果。

    Returns:
        原字典（perception_summary 超限时替换为截断副本 + 截断标记）。
    """
    summary = step.get("perception_summary")
    if isinstance(summary, str) and len(summary) > INCIDENT_SUMMARY_MAX_CHARS:
        step["perception_summary"] = (
            summary[:INCIDENT_SUMMARY_MAX_CHARS]
            + f"…[截断 {len(summary) - INCIDENT_SUMMARY_MAX_CHARS} 字符]"
        )
    return step


# ── 辅助函数：感知哈希 ────────────────────────────────────────────────────────
# phash 计算（average hash）已统一至 src.orchestration.phash（消除双实现技术债）。
# 本模块通过别名 import 保留 _compute_average_hash / _hamming_distance 名字，
# 供 stall_detect_node 信号 A 调用与既有测试引用，底层实现单一。


# ── 辅助函数：is_browser_task（条件边路由辅助，纯函数） ──────────────────────


def is_browser_task(instruction: str) -> bool:
    """判断指令是否为浏览器任务（独立纯函数，路由归此，不在 Supervisor 内）。

    工程假设：通过关键词匹配判断（http/https/浏览器/browser/chrome/firefox/edge）。
    Task 12 可替换为 LLM 二次分类。

    Args:
        instruction: Supervisor 下发的当前指令文本。

    Returns:
        True 表示浏览器任务，应路由至 playwright 节点。
    """
    lower = instruction.lower()
    browser_patterns = [
        r"\bhttps?://",
        r"\bbrowser\b",
        r"\bchrome\b",
        r"\bfirefox\b",
        r"\bedge\b",
        r"\bplaywri",
        r"浏览器",
        r"打开网页",
        r"访问网址",
    ]
    return any(re.search(p, lower) for p in browser_patterns)


# ── 条件边路由函数（全部带 -> Literal[...] 注解，独立可单测） ─────────────────


def route_after_supervisor(
    state: DesktopTaskState,
) -> Literal["perceive", "control", "playwright", "memory_flush", "error_report"]:
    """Supervisor 节点后的路由函数。

    优先级（从高到低）：
      1. task_status in (DONE, FAILED) → memory_flush（任务结束，写记忆）
      2. failure_reason 非 None → error_report（专属失败原因，如
         max_iterations_exceeded——K4 紧后 §3.3 回路硬上限）
      3. stall_count >= STALL_THRESHOLD → error_report（停滞超阈值）
      4. next_agent == "perceive" → perceive
      5. next_agent == "control" → control
      6. is_browser_task(current_instruction) → playwright
      7. 默认 → error_report（未识别的 next_agent）

    注意：感知失败停滞路径（perception_error 非 None）由 stall_detect_node 累加
    stall_count，再由此函数的规则 2 路由至 error_report（R3 决策）。

    Args:
        state: 当前 DesktopTaskState。

    Returns:
        目标节点名（Literal 类型）。
    """
    task_status = state.task_status
    stall_count = state.stall_count
    next_agent = state.next_agent
    instruction = state.current_instruction

    # 规则 1：任务终态 → memory_flush
    if task_status in (TaskStatus.DONE, TaskStatus.FAILED):
        logger.debug(
            "route_after_supervisor: task_status=%r → memory_flush",
            task_status,
        )
        return "memory_flush"

    # 规则 2：专属失败原因（回路硬上限等）→ error_report
    if state.failure_reason is not None:
        logger.debug(
            "route_after_supervisor: failure_reason=%r → error_report",
            state.failure_reason,
        )
        return "error_report"

    # 规则 3：停滞超阈值 → error_report
    if stall_count >= STALL_THRESHOLD:
        logger.debug(
            "route_after_supervisor: stall_count=%d >= STALL_THRESHOLD=%d → error_report",
            stall_count,
            STALL_THRESHOLD,
        )
        return "error_report"

    # 规则 4-5：按 next_agent 路由
    if next_agent == "perceive":
        return "perceive"
    if next_agent == "control":
        return "control"

    # 规则 6：浏览器任务 → playwright
    if is_browser_task(instruction):
        logger.debug("route_after_supervisor: is_browser_task=True → playwright")
        return "playwright"

    # 默认：未识别 → error_report
    logger.warning(
        "route_after_supervisor: 未识别 next_agent=%r → error_report",
        next_agent,
    )
    return "error_report"


def route_after_control(
    state: DesktopTaskState,
) -> Literal["stall_detect", "supervisor"]:
    """control_node 后的路由函数。

    - control_error 非 None 或 task_status=FAILED → stall_detect（累加停滞计数）
    - 正常（control_error=None）→ supervisor

    Args:
        state: 当前 DesktopTaskState。

    Returns:
        目标节点名（Literal 类型）。
    """
    if state.control_error is not None or state.task_status == TaskStatus.FAILED:
        logger.debug(
            "route_after_control: control_error=%r task_status=%r → stall_detect",
            state.control_error,
            state.task_status,
        )
        return "stall_detect"
    return "supervisor"


def route_after_stall(
    state: DesktopTaskState,
) -> Literal["supervisor", "error_report"]:
    """stall_detect_node 后的路由函数。

    - stall_count >= STALL_THRESHOLD → error_report
    - 否则 → supervisor

    Args:
        state: 当前 DesktopTaskState。

    Returns:
        目标节点名（Literal 类型）。
    """
    if state.stall_count >= STALL_THRESHOLD:
        logger.debug(
            "route_after_stall: stall_count=%d >= STALL_THRESHOLD=%d → error_report",
            state.stall_count,
            STALL_THRESHOLD,
        )
        return "error_report"
    return "supervisor"


# ── stall_detect_node（三信号停滞检测）────────────────────────────────────────


def _error_fingerprint(
    perception_error: str | None,
    control_error: str | None,
) -> str | None:
    """(perception_error, control_error) 的错误指纹（K5 ①，纯函数）。

    两者皆 None 时返回 None（无错误无指纹）；否则返回确定性的序列化字符串，
    供 stall_detect_node 与 state.counted_error_fingerprint 比对去重。

    Args:
        perception_error: 当前 state 的感知错误文本（None=无）。
        control_error: 当前 state 的控制错误文本（None=无）。

    Returns:
        指纹字符串，或 None（无错误）。
    """
    if perception_error is None and control_error is None:
        return None
    return f"p={perception_error!r}|c={control_error!r}"


def make_stall_detect_node(
    snapshot_store: SnapshotStore | None = None,
) -> Any:
    """生成 stall_detect_node 节点函数。

    三停滞信号（任一触发则 stall_count 累加）：
      信号 A — 画面 phash 不变：当前 snapshot_ref 对应图像与 last_screen_hash 汉明距离
               < PHASH_UNCHANGED_THRESHOLD，视为画面未变化。
               注：snapshot_ref 只存 ID，需通过 snapshot_store 加载图像；
               snapshot_store=None 时跳过此信号（测试/无存储环境）。
      信号 B — 步骤重复：len(step_history) > STALL_MAX_STEPS 且最近 N 步均为同一 Worker。
      信号 C — 错误指纹去重计数（K5 ①）：(perception_error, control_error) 指纹
               相对「上次已计数指纹」（state.counted_error_fingerprint）**新产生**时
               +1。错误文本在 LastValue state 里会跨节点残留（perceive 成功不清
               control_error），按指纹去重保证同一错误只计一次；持续同 Worker
               重试造成的停滞由信号 B 兜住。错误清空后指纹归 None，同一错误
               再现视为新停滞事件（可再计）。

    stall_count **连续语义**（K5 ②）：本轮无任何停滞信号（increment==0）时归零。
    归零逻辑必须在本节点内——control 成功路径绕过 stall_detect 直回 supervisor，
    放在别处会漏掉「经过本节点但无信号」的清零时机。

    返回增量：{"stall_count", "last_screen_hash", "counted_error_fingerprint"}
    （只更新这三个字段，LastValue 覆写；错误文本不清——supervisor 的 prompt
    仍需 perception_error/control_error 原文）。

    Args:
        snapshot_store: 快照存取接口（用于信号 A phash 比对）；None 时跳过 A 信号。

    Returns:
        符合 LangGraph 节点签名 `(state) -> dict` 的异步函数。
    """

    async def stall_detect_node(state: DesktopTaskState) -> dict[str, Any]:
        """停滞检测节点：三信号累加 stall_count，无信号轮归零（连续语义）。

        Args:
            state: 当前 DesktopTaskState。

        Returns:
            state 增量字典，含 stall_count / last_screen_hash /
            counted_error_fingerprint。
        """
        stall_increment = 0
        new_hash: str | None = state.last_screen_hash

        # 信号 A：画面 phash 不变（需要 snapshot_store 加载图像）
        if snapshot_store is not None and state.snapshot_ref is not None:
            try:
                snapshot = await snapshot_store.load(state.snapshot_ref)
                current_hash: str | None = None

                # 从 screenshot_path 读取图像字节计算 phash
                # 文件 I/O 走 asyncio.to_thread，不阻塞事件循环（python-code.md）
                if snapshot.screenshot_path is not None:
                    try:
                        img_bytes = await asyncio.to_thread(
                            Path(snapshot.screenshot_path).read_bytes
                        )
                        current_hash = _compute_average_hash(img_bytes)
                    except OSError as exc:
                        logger.debug(
                            "stall_detect_node: 无法读取截图文件 %s: %s",
                            snapshot.screenshot_path,
                            exc,
                        )

                if current_hash is not None:
                    if state.last_screen_hash is not None:
                        dist = _hamming_distance(current_hash, state.last_screen_hash)
                        if dist < PHASH_UNCHANGED_THRESHOLD:
                            logger.info(
                                "stall_detect_node: 信号A 画面未变化 hamming_dist=%d < %d",
                                dist,
                                PHASH_UNCHANGED_THRESHOLD,
                            )
                            stall_increment += 1
                    new_hash = current_hash
            except Exception as exc:
                logger.debug("stall_detect_node: 信号A 计算失败（%s），跳过", exc)

        # 信号 B：步骤重复（最近 STALL_MAX_STEPS+1 步都是同一 Worker）
        history = state.step_history
        if len(history) > STALL_MAX_STEPS:
            recent = history[-(STALL_MAX_STEPS + 1) :]
            agents_in_recent = {step.agent for step in recent}
            if len(agents_in_recent) == 1:
                repeated_agent = next(iter(agents_in_recent))
                logger.info(
                    "stall_detect_node: 信号B 步骤重复 最近 %d 步均为 agent=%r",
                    len(recent),
                    repeated_agent,
                )
                stall_increment += 1

        # 信号 C：错误指纹去重计数（K5 ①）——只在 (perception_error, control_error)
        # 指纹相对「上次已计数指纹」新产生时 +1。错误文本不在此清除（supervisor
        # 的 prompt 仍需原文），去重靠 counted_error_fingerprint 比对。
        error_fp = _error_fingerprint(state.perception_error, state.control_error)
        new_counted_fp: str | None = state.counted_error_fingerprint
        if error_fp is not None:
            if error_fp != state.counted_error_fingerprint:
                logger.info(
                    "stall_detect_node: 信号C 新错误指纹 %s",
                    error_fp,
                )
                stall_increment += 1
                new_counted_fp = error_fp
        else:
            # 本轮无错误：清计数指纹——同一错误此后再现视为新停滞事件（可再计）
            new_counted_fp = None

        # K5 ②：stall_count 连续语义——本轮无任何停滞信号即归零。
        # 归零必须放本节点内（control 成功路径绕过 stall_detect 直回 supervisor，
        # 放别处会漏掉「经过本节点但无信号」的清零时机）。
        if stall_increment > 0:
            new_stall_count = state.stall_count + stall_increment
        else:
            new_stall_count = 0

        logger.info(
            "stall_detect_node: stall_increment=%d new_stall_count=%d "
            "(threshold=%d) signals triggered=%d",
            stall_increment,
            new_stall_count,
            STALL_THRESHOLD,
            stall_increment,
        )

        return {
            "stall_count": new_stall_count,
            "last_screen_hash": new_hash,
            "counted_error_fingerprint": new_counted_fp,
        }

    return stall_detect_node


# ── error_report_node ─────────────────────────────────────────────────────────


def make_error_report_node(
    incident_reporter: IncidentReporter | None = None,
) -> Any:
    """生成 error_report_node 节点函数。

    职责：
      - log error（task_id / stall_count / errors / snapshot_ref）。
      - 调 incident_reporter.report 打桩（上报告警/监控系统）。
      - 设 task_status=FAILED，返回增量。

    Args:
        incident_reporter: 事件上报接口；None 时使用 NoopIncidentReporter 打桩。

    Returns:
        符合 LangGraph 节点签名 `(state) -> dict` 的异步函数。
    """
    reporter: IncidentReporter = incident_reporter or NoopIncidentReporter()

    async def error_report_node(state: DesktopTaskState) -> dict[str, Any]:
        """错误上报节点：记录错误信息 + incident_reporter 打桩 + 设 FAILED。

        Args:
            state: 当前 DesktopTaskState。

        Returns:
            state 增量字典，含 task_status="FAILED"。
        """
        errors: dict[str, str | None] = {
            "perception_error": state.perception_error,
            "control_error": state.control_error,
        }

        logger.error(
            "error_report_node: task_id=%r stall_count=%d failure_reason=%r "
            "perception_error=%r control_error=%r snapshot_ref=%r",
            state.task_id,
            state.stall_count,
            state.failure_reason,
            state.perception_error,
            state.control_error,
            state.snapshot_ref,
        )

        try:
            await reporter.report(
                task_id=state.task_id,
                stall_count=state.stall_count,
                errors=errors,
                snapshot_ref=state.snapshot_ref,
                metadata={
                    "task_description": state.task_description,
                    "task_status": state.task_status,
                    # K4 紧后 §3.3：专属失败原因（如 max_iterations_exceeded），
                    # None=经停滞/错误路径进入（非硬上限）
                    "failure_reason": state.failure_reason,
                    "iteration_count": state.iteration_count,
                    "step_count": len(state.step_history),
                    # 前 N 步流程（Task 14：异常现场含步骤历史，窗口 INCIDENT_STEP_WINDOW）。
                    # perception_summary 逐步截断（对齐 PERCEPTION_SUMMARY_MAX_TOKENS 的
                    # 2000 字符近似口径）——现场包单包有界，长文本全文可经 snapshot_ref 回查
                    "recent_steps": [
                        _truncate_step_for_incident(s.model_dump(mode="json"))
                        for s in state.step_history[-INCIDENT_STEP_WINDOW:]
                    ],
                },
            )
        except Exception as exc:
            logger.warning("error_report_node: incident_reporter.report 失败（%s）", exc)

        return {"task_status": TaskStatus.FAILED}

    return error_report_node


# ── memory_flush_node（唯一记忆写入点）────────────────────────────────────────


def make_memory_flush_node(
    memory_api: MemoryAPI | None = None,
    step_archive: StepArchive | None = None,
) -> Any:
    """生成 memory_flush_node 节点函数。

    唯一记忆写入点（memory-rules.md）：
      - 调 memory_api.write_session_summary(scope="session"，显式不默认 user）。
      - 调 step_archive.archive 全量归档当前 step_history。
      - 不直连 Neo4j/向量库/图谱（经 MemoryAPI Protocol 打桩）。

    Args:
        memory_api: 记忆写入接口；None 时使用 NoopMemoryAPI 打桩。
        step_archive: 步骤归档接口；None 时使用 StepArchive 打桩（无操作）。

    Returns:
        符合 LangGraph 节点签名 `(state) -> dict` 的异步函数。
    """
    mem_api: MemoryAPI = memory_api or NoopMemoryAPI()
    archive: StepArchive = step_archive or StepArchive()

    async def memory_flush_node(state: DesktopTaskState) -> dict[str, Any]:
        """记忆写入节点：任务完成后唯一写记忆点，scope=session 显式。

        执行流程：
          1. 组装任务摘要（task_id / task_status / 步骤数 / 最终感知摘要）。
          2. 调 memory_api.write_session_summary（scope="session"，显式）。
          3. 调 step_archive.archive 全量归档 step_history。

        Args:
            state: 当前 DesktopTaskState。

        Returns:
            state 增量字典（返回 task_status 以确认终态，不改变状态）。
        """
        # 组装摘要
        summary_lines = [
            f"任务ID: {state.task_id}",
            f"任务描述: {state.task_description}",
            f"最终状态: {state.task_status}",
            f"执行步骤数: {len(state.step_history)}",
            f"停滞计数: {state.stall_count}",
        ]
        if state.failure_reason:
            summary_lines.append(f"失败原因: {state.failure_reason}")
        if state.perception_summary:
            summary_lines.append(f"最终感知摘要: {state.perception_summary[:500]}")
        if state.perception_error:
            summary_lines.append(f"最终感知错误: {state.perception_error}")
        if state.control_error:
            summary_lines.append(f"最终控制错误: {state.control_error}")
        summary = "\n".join(summary_lines)

        metadata: dict[str, Any] = {
            "step_count": len(state.step_history),
            "stall_count": state.stall_count,
            "task_status": state.task_status,
            "failure_reason": state.failure_reason,
        }

        # 写入记忆（scope="session"，显式不默认 user）
        try:
            await mem_api.write_session_summary(
                task_id=state.task_id,
                scope="session",  # 显式指定，禁止默认 user（memory-rules.md）
                summary=summary,
                metadata=metadata,
            )
            logger.info(
                "memory_flush_node: 记忆写入完成 task_id=%r scope=session step_count=%d",
                state.task_id,
                len(state.step_history),
            )
        except Exception as exc:
            logger.warning("memory_flush_node: memory_api.write_session_summary 失败（%s）", exc)

        # 全量归档 step_history
        try:
            await archive.archive(task_id=state.task_id, steps=state.step_history)
            logger.info(
                "memory_flush_node: step_history 归档完成 task_id=%r steps=%d",
                state.task_id,
                len(state.step_history),
            )
        except Exception as exc:
            logger.warning("memory_flush_node: step_archive.archive 失败（%s）", exc)

        # 返回 task_status 增量（保持终态，不改变）
        return {"task_status": state.task_status}

    return memory_flush_node


# ── playwright 占位节点（Task 10BC 不实现，仅占位防图编译报错）────────────────


async def _playwright_placeholder_node(state: DesktopTaskState) -> dict[str, Any]:
    """Playwright 浏览器任务占位节点（Task 10BC 范围外，仅防图编译报错）。

    工程假设：浏览器任务由独立 PlaywrightAgent 处理（eng-team 未来接线）。
    当前实现直接返回 error_report（未实现提示），不崩溃。

    Args:
        state: 当前 DesktopTaskState。

    Returns:
        state 增量，路由到 error_report 处理。
    """
    logger.warning(
        "_playwright_placeholder_node: Playwright Agent 未实现，task_id=%r → error_report",
        state.task_id,
    )
    return {
        "control_error": "Playwright Agent 尚未实现（占位节点）",
        "task_status": TaskStatus.FAILED,
    }


# ── incident_reporter feature-flag 接线（Task 14，默认关零回归）──────────────


def _resolve_incident_reporter(
    incident_reporter: IncidentReporter | None,
    snapshot_store: SnapshotStore | None,
) -> IncidentReporter | None:
    """解析 get_graph 使用的 IncidentReporter（feature-flag 纪律：默认关零回归）。

    优先级：
      1. 显式注入的 incident_reporter → 原样返回。
      2. env INCIDENT_DIR 已设（非空）→ FileIncidentReporter（现场包落盘，
         注入 snapshot_store 以便附截图）。
      3. 否则 → None（make_error_report_node 内落 NoopIncidentReporter，
         现有测试全部不设 env，零回归）。

    Args:
        incident_reporter: get_graph 调用方显式注入的实现（None 表示未注入）。
        snapshot_store: get_graph 传入的快照存取接口（透传给 FileIncidentReporter）。

    Returns:
        IncidentReporter 实现或 None（Noop 兜底在 make_error_report_node）。
    """
    if incident_reporter is not None:
        return incident_reporter
    incident_dir = os.environ.get("INCIDENT_DIR")
    if incident_dir:
        logger.info("get_graph: INCIDENT_DIR=%r 已设，启用 FileIncidentReporter", incident_dir)
        return FileIncidentReporter(incident_dir=incident_dir, snapshot_store=snapshot_store)
    return None


# ── get_graph 工厂 ─────────────────────────────────────────────────────────────


def get_graph(
    client: DesktopMCPClient | None = None,
    supervisor_agent: DesktopSupervisorAgent | None = None,
    perception_agent: ScreenPerceptionAgent | None = None,
    control_agent: DesktopControlAgent | None = None,
    memory_api: MemoryAPI | None = None,
    incident_reporter: IncidentReporter | None = None,
    snapshot_store: SnapshotStore | None = None,
    step_archive: StepArchive | None = None,
    checkpointer: Any = None,
) -> Any:
    """构建并编译桌面任务执行图（工厂函数）。

    默认使用 InMemorySaver（测试用，免装 Postgres）。
    生产环境由 eng-team 注入 AsyncPostgresSaver（独立包，首用需 .setup()）。

    图连线（R3 决策）：
        START → supervisor
        supervisor → perceive | control | playwright | memory_flush | error_report
        perceive → stall_detect
        control → stall_detect | supervisor
        stall_detect → supervisor | error_report
        error_report → memory_flush
        memory_flush → END

    playwright 节点当前为占位，直接返回 FAILED（eng-team 未来实现后替换）。

    Args:
        client: DesktopMCPClient 实例（async with 块内注入）。
                None 时各 Agent 节点需已通过 *_agent 参数注入，否则图运行时失败。
        supervisor_agent: DesktopSupervisorAgent 实例；None 时创建默认实例（K3 ②③）：
                注入真 PromptLoader（jinja2 模板）；env ANTHROPIC_API_KEY 已设且
                anthropic 包可用时自动构造 AsyncAnthropic（模型 ID 走 env
                DESKTOP_SUPERVISOR_MODEL，见 desktop_supervisor.py），否则
                llm_client=None——注意这**不是**任务可继续的静默回退：plan 会返回
                FAILED 增量，经 error_report → memory_flush 收口（「优雅回退」＝
                不崩溃、有终态，而非任务照常执行）。
        perception_agent: ScreenPerceptionAgent 实例；None 且 client 非 None 时自动创建，
                并与 stall/incident 共用同一 snapshot_store（K3 ①：store 分裂会让
                信号 A 加载不到 perceive 存的快照而静默失效）。
        control_agent: DesktopControlAgent 实例；None 且 client 非 None 时自动创建。
        memory_api: MemoryAPI 实现；None 时使用 NoopMemoryAPI 打桩。
        incident_reporter: IncidentReporter 实现；None 时看 env INCIDENT_DIR——
                已设则启用 FileIncidentReporter（异常现场落盘，Task 14），
                未设则 NoopIncidentReporter 打桩（默认关零回归）。
        snapshot_store: SnapshotStore 实现；None 时跳过 phash 信号 A 与 control
                节点的 TOCTOU 快照复用（K1 ⑤，均为可选优化，零回归）。
        step_archive: StepArchive 实现；None 时使用无操作打桩。
        checkpointer: LangGraph Checkpointer；None 时使用 InMemorySaver（测试默认）。

    Returns:
        编译好的 CompiledGraph（可调用 .invoke / .astream_events / .get_state 等）。
    """
    # 默认 Checkpointer（InMemorySaver，测试用）
    if checkpointer is None:
        checkpointer = InMemorySaver()

    # 默认 Supervisor（K3 ②③：真 PromptLoader；有 key 且 anthropic 可用时接真 LLM）
    if supervisor_agent is None:
        llm_client: Any = None
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                # 函数内 try-import：anthropic 为可选依赖，无包时不阻断图构建
                from anthropic import AsyncAnthropic

                llm_client = AsyncAnthropic(api_key=api_key)
            except ImportError:
                logger.warning(
                    "get_graph: 检测到 ANTHROPIC_API_KEY 但 anthropic 包不可用，"
                    "Supervisor 无 LLM 客户端（plan 将返回 FAILED 经 error_report 收口）"
                )
        else:
            logger.info(
                "get_graph: 未设 ANTHROPIC_API_KEY，Supervisor 无 LLM 客户端"
                "（任务将经 plan 的 FAILED 增量 → error_report → memory_flush 收口）"
            )
        supervisor_agent = DesktopSupervisorAgent(
            llm_client=llm_client,
            prompt_loader=PromptLoader(),
            step_archive=step_archive or StepArchive(),
        )

    # 自动创建 PerceptionAgent（需要 client）
    if perception_agent is None and client is not None:
        # K3 ①：与 stall/incident 共用同一 snapshot_store——agent 私有 InMemory
        # store 与注入 store 分裂会让信号 A 永远加载不到 perceive 存的快照
        # （静默失效）。snapshot_store=None 时 agent 内部仍落 InMemory 打桩（现状）。
        perception_agent = ScreenPerceptionAgent(
            client=client,
            snapshot_store=snapshot_store,
        )

    # 自动创建 ControlAgent（需要 client）
    if control_agent is None and client is not None:
        guard = ActionGuard(client=client)
        control_agent = DesktopControlAgent(client=client, guard=guard)

    # 构造节点函数
    sup_node = make_supervisor_node(supervisor_agent)

    if perception_agent is not None:
        perceive_node = make_perceive_node(perception_agent)
    else:

        async def perceive_node(state: DesktopTaskState) -> dict[str, Any]:
            """感知节点占位（无 PerceptionAgent 注入）。"""
            logger.warning("perceive_node: PerceptionAgent 未注入")
            return {"perception_error": "PerceptionAgent 未注入"}

    if control_agent is not None:
        # K1 ⑤：注入 snapshot_store（同 stall 节点先例），control 节点复用
        # state.snapshot_ref 快照做 TOCTOU 第一张基线（过新鲜度门，省一次 RPC）
        ctrl_node = make_control_node(control_agent, snapshot_store=snapshot_store)
    else:

        async def ctrl_node(state: DesktopTaskState) -> dict[str, Any]:
            """控制节点占位（无 ControlAgent 注入）。"""
            logger.warning("control_node: ControlAgent 未注入")
            return {"control_error": "ControlAgent 未注入"}

    stall_node = make_stall_detect_node(snapshot_store=snapshot_store)
    err_node = make_error_report_node(
        incident_reporter=_resolve_incident_reporter(incident_reporter, snapshot_store)
    )
    mem_node = make_memory_flush_node(
        memory_api=memory_api,
        step_archive=step_archive,
    )

    # 构建图
    builder = StateGraph(DesktopTaskState)

    # 注册节点
    builder.add_node("supervisor", sup_node)
    builder.add_node("perceive", perceive_node)
    builder.add_node("control", ctrl_node)
    builder.add_node("stall_detect", stall_node)
    builder.add_node("error_report", err_node)
    builder.add_node("memory_flush", mem_node)
    builder.add_node("playwright", _playwright_placeholder_node)

    # 图连线（R3 决策）
    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "perceive": "perceive",
            "control": "control",
            "playwright": "playwright",
            "memory_flush": "memory_flush",
            "error_report": "error_report",
        },
    )

    # perceive → stall_detect（R3：感知失败也经统一停滞节点）
    builder.add_edge("perceive", "stall_detect")

    builder.add_conditional_edges(
        "control",
        route_after_control,
        {
            "stall_detect": "stall_detect",
            "supervisor": "supervisor",
        },
    )

    builder.add_conditional_edges(
        "stall_detect",
        route_after_stall,
        {
            "supervisor": "supervisor",
            "error_report": "error_report",
        },
    )

    # error_report → memory_flush → END
    builder.add_edge("error_report", "memory_flush")
    builder.add_edge("memory_flush", END)

    # playwright 占位 → error_report
    builder.add_edge("playwright", "error_report")

    return builder.compile(checkpointer=checkpointer)
