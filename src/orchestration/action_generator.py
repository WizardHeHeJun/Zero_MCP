"""instruction→ActionSpec 生成层 Agent（蓝图任务 8）。

`ActionGeneratorAgent` 是独立编排节点（蓝图决策 A：`generate_action`，插在
`supervisor →(next_agent=="control")→ generate_action → control` 之间），职责
单一：把 Supervisor 下发的 `current_instruction` 翻译为**恰好一个**结构化
`ActionSpec`，写入 `state.pending_action`，交给已有的 `control` 节点执行
（interrupt 分区协议不变——生成层永远在 control 之前落 pending_action，
Checkpointer 持久化后重放读同一值，不受 LLM 非确定性影响）。

放编排层而非 agents 层（蓝图决策 A续，工程假设）：本 Agent 需要
`PromptLoader`（渲染提示词）与 `SnapshotStore.load`（加载感知快照）两个
编排层能力，agents → orchestration 反向 import 违三层单向依赖红线；与
`DesktopSupervisorAgent` 同构，不新开例外。

生成策略（蓝图决策 F）：tools + strict:true + tool_choice（5 个自定义工具，
工具名即判别式，`disable_parallel_tool_use` 保证恰一次调用），不用
`messages.parse`（5 类异构 schema 非单一固定 schema）。主备单次切换复用
`llm_fallback.call_with_single_fallback`（同 Supervisor 语义）。

grounding 解析（蓝图决策 C）：LLM 只能引用 prompt 中列出的紧凑 id
（`target_element_id`），服务端按 `prompt_loader.build_grounding_table` 查
bbox 中心覆写坐标——不信 LLM 自报坐标防像素幻觉；`uia_hollow` 场景 LLM 只给
`coordinate` 时沿用（OCR 兜底通道，不覆写）。

失败处理（蓝图决策 D）：单条丢弃 + 结构化记录进下一轮 planner，不建 JSON
自修复循环。失败复用 `control_error` 字段 + 机读令牌
`ACTION_GENERATION_FAILED_TOKEN`（位置无关，消费侧 `re.search` 提取，同构
`[desk:toctou_degraded]` 既有约定），本节点自行 `append_step`（agent=
"generate_action"），经 `route_after_generate_action` 复用既有 stall_detect
通路。

层依赖：本文件 import agents.models（下调允许）与 orchestration 内其他模块
（同层），不反向 import agents 层的具体 Agent 实现。
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Protocol

from pydantic import ValidationError

from src.agents.models.action_generation_schema import (
    ActionGenerationBase,
    ClickActionInput,
    GroundingCoordinate,
    KeyActionInput,
    TypeActionInput,
    WaitActionInput,
    WindowCloseActionInput,
)
from src.agents.models.screen_snapshot import ActionSpec, BBox, ScreenSnapshot
from src.agents.models.step_record import append_step
from src.orchestration.desktop_supervisor import (
    DESKTOP_SUPERVISOR_MODEL,
    DESKTOP_SUPERVISOR_MODEL_FALLBACK,
)
from src.orchestration.llm_fallback import LLMFallbackError, call_with_single_fallback
from src.orchestration.prompt_loader import (
    PERCEPTION_SUMMARY_MAX_TOKENS,
    PromptLoader,
    build_grounding_table,
)
from src.orchestration.protocols import SnapshotStore
from src.orchestration.state import DesktopTaskState

logger = logging.getLogger(__name__)

# ── 模型 ID（走 .env，不硬编码，agent-framework-rules）────────────────────────
# 工程假设（蓝图「工程假设清单」）：默认取同 Supervisor 族模型——是否该用更快/
# 更便宜的模型专司生成留待实机标定。

DESKTOP_ACTION_GENERATOR_MODEL: str = os.environ.get(
    "DESKTOP_ACTION_GENERATOR_MODEL", DESKTOP_SUPERVISOR_MODEL
)
DESKTOP_ACTION_GENERATOR_MODEL_FALLBACK: str | None = (
    os.environ.get("DESKTOP_ACTION_GENERATOR_MODEL_FALLBACK") or DESKTOP_SUPERVISOR_MODEL_FALLBACK
)

# ── 失败机读令牌（同构 [desk:toctou_degraded] 既有约定）─────────────────────────

ACTION_GENERATION_FAILED_TOKEN: str = "[desk:action_generation_failed]"
"""生成失败时 control_error 携带的机读令牌，位置无关，消费侧用 re.search 提取。"""

# ── 5 类生成子模型 → action_type 映射（工具名即判别式） ─────────────────────────

_ACTION_MODELS: dict[str, type[ActionGenerationBase]] = {
    "click": ClickActionInput,
    "type": TypeActionInput,
    "key": KeyActionInput,
    "window_close": WindowCloseActionInput,
    "wait": WaitActionInput,
}

_TOOL_DESCRIPTIONS: dict[str, str] = {
    "click": "点击一个已定位的元素或坐标。",
    "type": "向已定位的输入焦点输入一段文本。",
    "key": "发送一次按键或组合键（如 enter、ctrl+c）。",
    "window_close": "关闭指定窗口（高危动作，执行前需人工确认）。",
    "wait": "什么都不做，等待若干毫秒（用于容忍界面过渡）。",
}


def _build_tools() -> list[dict[str, Any]]:
    """构造 5 个 strict:true 自定义工具定义（蓝图决策 F）。

    复核①（2026-08-31 现场核验）：官方 computer/browser **工具集条目**不接受
    strict:true，但本仓 5 个均为自定义 tool，不受此限制。

    Returns:
        5 个工具定义 dict 列表，`input_schema` 取自对应生成子模型
        `model_json_schema()`（已在 PR-α 现场核验：无 prefixItems / 无超出
        strict 兼容范围的数组约束 / 无数值-字符串约束，`{x,y}` 坐标对象方案）。
    """
    return [
        {
            "name": action_type,
            "description": _TOOL_DESCRIPTIONS[action_type],
            "input_schema": model_cls.model_json_schema(),
            "strict": True,
        }
        for action_type, model_cls in _ACTION_MODELS.items()
    ]


def _extract_tool_use(response: Any) -> Any | None:
    """从 LLM 响应中取首个 tool_use 内容块（disable_parallel_tool_use 保证至多一个）。

    Args:
        response: `messages.create` 返回的响应对象（或兼容 mock）。

    Returns:
        tool_use 内容块（含 `.name` / `.input`），未找到时返回 None。
    """
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "tool_use":
            return block
    return None


# ── PromptLoader Protocol（依赖注入接口，同 desktop_supervisor.py 先例）────────


class ActionPromptLoader(Protocol):
    """ActionGeneratorAgent 提示词加载接口（结构子类型，真实现=PromptLoader）。"""

    def render_action_generation(
        self,
        state: DesktopTaskState,
        snapshot: ScreenSnapshot,
    ) -> tuple[str, str]:
        """渲染 ActionGenerator 提示词（system + user）。"""
        ...


# ── ActionGeneratorAgent ───────────────────────────────────────────────────────


class ActionGeneratorAgent:
    """instruction→ActionSpec 生成层 Agent（蓝图任务 8）。

    依赖注入：
        llm_client — Anthropic AsyncAnthropic 客户端（可 mock）。
        prompt_loader — ActionPromptLoader Protocol 实现；None 时用真 PromptLoader。
        snapshot_store — SnapshotStore Protocol 实现；None 时 generate() 直接失败
            （无法加载 state.snapshot_ref 对应的快照，见 generate() docstring①）。
        model / fallback_model — 模型 ID 覆写；None 时读 env（默认同 Supervisor 族）。
    """

    def __init__(
        self,
        llm_client: Any,
        prompt_loader: ActionPromptLoader | None = None,
        snapshot_store: SnapshotStore | None = None,
        model: str | None = None,
        fallback_model: str | None = None,
    ) -> None:
        """初始化 ActionGeneratorAgent。

        Args:
            llm_client: Anthropic AsyncAnthropic 客户端实例（依赖注入，可 mock）；
                缺 ANTHROPIC_API_KEY 时可传 None（优雅回退）。
            prompt_loader: ActionPromptLoader Protocol 实现；None 时使用真
                `PromptLoader()`（Task 11B 模板已就绪，无需占位实现）。
            snapshot_store: 快照存取接口；None 时 generate() 直接失败（不调 LLM）。
            model: 模型 ID 覆写；None 时读 DESKTOP_ACTION_GENERATOR_MODEL 环境变量
                （默认同 DESKTOP_SUPERVISOR_MODEL）。
            fallback_model: 备用模型 ID 覆写；None 时读
                DESKTOP_ACTION_GENERATOR_MODEL_FALLBACK 环境变量（默认同
                DESKTOP_SUPERVISOR_MODEL_FALLBACK）。
        """
        self.llm_client = llm_client
        self.prompt_loader: ActionPromptLoader = prompt_loader or PromptLoader()
        self.snapshot_store: SnapshotStore | None = snapshot_store
        self.model: str = model or DESKTOP_ACTION_GENERATOR_MODEL
        self.fallback_model: str | None = fallback_model or DESKTOP_ACTION_GENERATOR_MODEL_FALLBACK

    async def generate(self, state: DesktopTaskState) -> dict[str, Any]:
        """生成恰好一个动作，返回 state 增量。

        失败路径（① 缺 key / llm_client None / 无 snapshot_ref / 快照加载失败
        ② LLM 调用失败（含主备均败）③ 无 tool_use 响应 ④ 未知工具名
        ⑤ 子模型校验失败 ⑥ target_element_id 不在本次可用元素表中）统一走
        `_fail`：不调 LLM（① 类）或调用/解析失败即返回 control_error（带机读
        令牌）+ pending_action=None + append_step，经 route_after_generate_action
        → stall_detect（蓝图决策 D，单条丢弃不做自修复循环）。

        每次调用都重新生成，不做隐藏缓存——即使 state.pending_action 已非
        None（正常情况下不会，因为 control 成功/拒绝均清空），也会覆盖。

        Args:
            state: 当前 DesktopTaskState（取 current_instruction / snapshot_ref /
                step_history / task_status）。

        Returns:
            成功：`{"pending_action": ActionSpec, "step_history": [...]}`。
            失败：`{"control_error": "...[desk:action_generation_failed]...",
            "pending_action": None, "step_history": [...]}`。
        """
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return self._fail(state, "缺 ANTHROPIC_API_KEY，无法调用 LLM")

        if self.llm_client is None:
            return self._fail(state, "llm_client 未初始化")

        if state.snapshot_ref is None:
            return self._fail(state, "state 缺 snapshot_ref，尚未感知，无法生成动作")

        if self.snapshot_store is None:
            return self._fail(state, "snapshot_store 未注入，无法加载快照")

        try:
            snapshot = await self.snapshot_store.load(state.snapshot_ref)
        except Exception as exc:
            return self._fail(state, f"快照加载失败: {exc}")

        system_prompt, user_prompt = self.prompt_loader.render_action_generation(state, snapshot)

        try:
            response = await call_with_single_fallback(
                self.model,
                self.fallback_model,
                lambda model: self._call_llm(model, system_prompt, user_prompt),
            )
        except LLMFallbackError as exc:
            return self._fail(state, f"LLM 调用失败: {exc}")

        tool_use = _extract_tool_use(response)
        if tool_use is None:
            return self._fail(state, "LLM 响应未含工具调用")

        model_cls = _ACTION_MODELS.get(tool_use.name)
        if model_cls is None:
            return self._fail(state, f"未知工具名: {tool_use.name!r}")

        try:
            parsed = model_cls.model_validate(tool_use.input)
        except ValidationError as exc:
            return self._fail(state, f"工具输入校验失败: {exc}")

        # id 存在性核验 + 服务端坐标解析（蓝图决策 C：不信 LLM 自报坐标）
        grounding_error = self._resolve_grounding(parsed, snapshot)
        if grounding_error is not None:
            return self._fail(state, grounding_error)

        action_id = str(uuid.uuid4())
        spec: ActionSpec = parsed.to_action_spec(action_id)

        logger.info(
            "ActionGeneratorAgent.generate: task_id=%r action_type=%r action_id=%r",
            state.task_id,
            spec.action_type,
            action_id,
        )

        step_history = append_step(
            state.step_history,
            agent="generate_action",
            instruction=state.current_instruction,
            increment={},
            task_status=str(state.task_status),
        )
        return {"pending_action": spec, "step_history": step_history}

    def _resolve_grounding(
        self,
        parsed: ActionGenerationBase,
        snapshot: ScreenSnapshot,
    ) -> str | None:
        """核验 target_element_id 存在性，并用 bbox 中心覆写坐标（蓝图决策 C）。

        只对 `ClickActionInput` / `TypeActionInput` 生效（唯二含
        target_element_id/coordinate 定位字段的子模型）；其余子模型（key/
        window_close/wait）无定位字段，原样放行。

        Args:
            parsed: 已通过 pydantic 校验的生成子模型实例（就地修改 .coordinate）。
            snapshot: 本轮感知快照（与渲染 prompt 用的是同一份，保证 id 表一致）。

        Returns:
            None=通过（或不适用）；否则为失败原因文本。
        """
        if not isinstance(parsed, (ClickActionInput, TypeActionInput)):
            return None

        if parsed.target_element_id is not None:
            grounding_table: dict[str, BBox] = build_grounding_table(
                snapshot, PERCEPTION_SUMMARY_MAX_TOKENS
            )
            bbox = grounding_table.get(parsed.target_element_id)
            if bbox is None:
                return (
                    f"target_element_id={parsed.target_element_id!r} "
                    "不在本次可用元素表中（LLM 引用了未列出的 id）"
                )
            center_x = bbox.x + bbox.width // 2
            center_y = bbox.y + bbox.height // 2
            parsed.coordinate = GroundingCoordinate(x=center_x, y=center_y)
            return None

        if parsed.coordinate is None:
            # model_validator 已在 pydantic 校验阶段挡掉此情形，此处兜底不可达
            return "click/type 动作缺 target_element_id 与 coordinate"

        # LLM 只给 coordinate（uia_hollow 兜底通道）：沿用，不覆写
        return None

    def _fail(self, state: DesktopTaskState, reason: str) -> dict[str, Any]:
        """构造失败增量（蓝图决策 D：单条丢弃 + 结构化记录进下一轮 planner）。

        Args:
            state: 当前 DesktopTaskState。
            reason: 人读失败原因（散文），与机读令牌拼接进 control_error。

        Returns:
            state 增量字典，含 control_error（带机读令牌）/ pending_action=None /
            step_history（append_step 追加本步记录）。
        """
        control_error = f"{reason} {ACTION_GENERATION_FAILED_TOKEN}"
        logger.warning(
            "ActionGeneratorAgent.generate: 失败 task_id=%r reason=%s",
            state.task_id,
            reason,
        )
        step_history = append_step(
            state.step_history,
            agent="generate_action",
            instruction=state.current_instruction,
            increment={"control_error": control_error},
            task_status=str(state.task_status),
        )
        return {
            "control_error": control_error,
            "pending_action": None,
            "step_history": step_history,
        }

    async def _call_llm(self, model: str, system_prompt: str, user_prompt: str) -> Any:
        """单次 LLM 调用（主/备共用，tools + strict + tool_choice）。

        Args:
            model: 本次调用使用的模型 ID。
            system_prompt: 系统提示词。
            user_prompt: 用户提示词。

        Returns:
            `messages.create` 原始响应对象（异常向上抛，由
            `call_with_single_fallback` 分级处理）。
        """
        response = await self.llm_client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=_build_tools(),
            tool_choice={"type": "any", "disable_parallel_tool_use": True},
        )
        return response


# ── generate_action_node 节点函数工厂 ───────────────────────────────────────────


def make_generate_action_node(agent: ActionGeneratorAgent) -> Any:
    """生成 generate_action_node 节点函数（闭包注入 agent）。

    节点签名 (state: DesktopTaskState) -> dict，只返回增量（`ActionGeneratorAgent
    .generate` 已构造完整增量字典，本工厂只做闭包封装，同
    `make_supervisor_node`/`make_control_node` 既有模式）。

    Args:
        agent: 已构造的 ActionGeneratorAgent 实例。

    Returns:
        符合 LangGraph 节点签名的异步函数。
    """

    async def generate_action_node(state: DesktopTaskState) -> dict[str, Any]:
        """LangGraph 动作生成节点。"""
        return await agent.generate(state)

    return generate_action_node
