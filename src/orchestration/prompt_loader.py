"""Supervisor / ActionGenerator 提示词加载与上下文截断（Task 11B + 蓝图 PR-β 任务 7）。

PromptLoader 负责：
  1. 从 Path(__file__).parent/"prompts" 用 Jinja2 FileSystemLoader 加载模板
     （路径固定，不依赖 cwd，不走 .env）。
  2. 截断 step_history：只取最近 CONTEXT_STEP_WINDOW 步（默认 10），
     超出部分在此层丢弃（不写归档——归档由 Supervisor 节点的 StepArchive 负责）。
  3. 截断 perception_summary / 元素定位表：字符数超过 PERCEPTION_SUMMARY_MAX_TOKENS
     （默认 2000）时在此截断。
     注释：当前用字符数近似 token 预算（留 20% 余量），后期可替换为
     tiktoken.encoding_for_model("cl100k_base").encode() 精确计量（R6 工程假设）。
  4. 渲染 system / user 两个提示词，返回 tuple[str, str]。
  5. `render_action_generation`（蓝图决策 C）：把 ScreenSnapshot 的 uia_elements/
     text_blocks/visual_objects 三类元素合并为统一紧凑 id 查找表注入 ActionGenerator
     的 user 提示词——grounding 解析（LLM 引用 id → 服务端查 bbox）放生成层，
     不放感知层。

设计约束（agent-framework-rules / orchestration-rules）：
  - prompt 模板与代码分离（可版本化、可测）。
  - 上下文截断统一在此层，不在 Agent 节点内、不在模板内。
  - 公开接口完整类型注解。
  - 不含 I/O，render_* 均为同步方法（模板渲染是 CPU 运算，无网络/磁盘等待）。
  - 只下调 agents.models（ScreenSnapshot/BBox 契约模型，project-root.md 已认证
    的跨层共享契约，不算反向依赖）与 orchestration.state，不 import 任何其他
    上层模块。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from jinja2 import Environment, FileSystemLoader

from src.agents.models.screen_snapshot import BBox, ScreenSnapshot
from src.orchestration.state import DesktopTaskState, StepRecord

logger = logging.getLogger(__name__)

# ── 上下文预算常量 ────────────────────────────────────────────────────────────

CONTEXT_STEP_WINDOW: int = 10
"""step_history 注入模板时保留的最近步数（默认 10）。
超出部分在 prompt 层丢弃；state 层归档由 Supervisor 节点的 StepArchive 负责。
"""

PERCEPTION_SUMMARY_MAX_TOKENS: int = 2000
"""perception_summary 注入模板前的字符数上限（默认 2000）。
注释：当前用字符数近似 token 预算（留 20% 余量），后期可替换为
tiktoken.encoding_for_model("cl100k_base").encode() 精确计量（R6 工程假设）。
"""

# ── 模板目录（不走 .env，不依赖 cwd）─────────────────────────────────────────

_PROMPTS_DIR: Path = Path(__file__).parent / "prompts"

# ── 内部辅助：截断函数 ────────────────────────────────────────────────────────


def _truncate_step_window(
    step_history: list[StepRecord],
    window: int,
) -> list[dict[str, object]]:
    """取最近 window 步并转为 dict 列表供模板消费。

    模板只消费 dict 键（step_index / agent / instruction / task_status /
    control_error / perception_error），不依赖 StepRecord 对象，
    保持模板与 Pydantic 模型的解耦。

    Args:
        step_history: 完整 step_history（DesktopTaskState.step_history）。
        window: 保留的最近步数。

    Returns:
        最多 window 个步骤的 dict 列表（按时间升序，最新在最后）。
    """
    recent: list[StepRecord] = step_history[-window:] if step_history else []
    return [
        {
            "step_index": s.step_index,
            "agent": s.agent,
            "instruction": s.instruction,
            "task_status": s.task_status,
            "control_error": s.control_error,
            "perception_error": s.perception_error,
        }
        for s in recent
    ]


def _truncate_perception_summary(
    summary: str | None,
    max_chars: int,
) -> str | None:
    """字符数截断 perception_summary。

    超过 max_chars 时截断并记录 warning，否则原样返回。
    None 值原样返回（模板侧有 None 处理）。

    注释：当前为字符数近似，后期替换为 tiktoken 精确计量（R6 工程假设）。

    Args:
        summary: 原始感知摘要（可为 None）。
        max_chars: 字符数上限。

    Returns:
        截断后的摘要（或 None）。
    """
    if summary is None:
        return None
    if len(summary) <= max_chars:
        return summary
    logger.warning(
        "_truncate_perception_summary: 摘要 %d 字符超出上限 %d，截断。"
        "（当前为字符数近似，后期替换为 tiktoken——R6 工程假设）",
        len(summary),
        max_chars,
    )
    return summary[:max_chars]


GroundingEntry = tuple[str, str, str, BBox]
"""(compact_id, role_or_label, display_text, bbox) 四元组，见 `_iter_grounding_entries`。"""


def _iter_grounding_entries(snapshot: ScreenSnapshot) -> list[GroundingEntry]:
    """按 uia → ocr → vis 顺序遍历三类感知元素，赋序列化层紧凑 id（蓝图决策 C）。

    compact_id 格式 `"{type_prefix}:{index}"`（如 `"uia:3"`），序列化层新赋、
    与感知层 element_id/block_id/object_id 本体无关——现场核查：
    `UIAElement.element_id` 已含 `"uia_"` 前缀、`TextBlock.block_id` 已含
    `"ocr_"` 前缀（见 src/mcp/desktop/tools/perception.py），但两者格式不一
    （前者含 hwnd，后者含 snapshot_id）且 `VisualObject.object_id` 尚无生成
    约定；本函数不改感知层 id 本体（勿误改清单），改用紧凑序号 id 统一三类
    元素的引用格式，同时比原始 id 更省 token。

    Args:
        snapshot: 完整感知快照。

    Returns:
        (compact_id, role/label, display_text, bbox) 四元组列表，按类型分组、
        组内保持原始顺序。
    """
    entries: list[GroundingEntry] = []
    for i, el in enumerate(snapshot.uia_elements):
        entries.append((f"uia:{i}", el.control_type, el.name, el.bbox))
    for i, tb in enumerate(snapshot.text_blocks):
        entries.append((f"ocr:{i}", "text", tb.text, tb.bbox))
    for i, vo in enumerate(snapshot.visual_objects):
        entries.append((f"vis:{i}", vo.label, vo.label, vo.bbox))
    return entries


def _render_grounding_line(compact_id: str, role: str, text: str, bbox: BBox) -> str:
    """渲染单个元素为紧凑 HTML 类行（Skyvern 序列化实测 -11.4% token 的口径）。

    Args:
        compact_id: `_iter_grounding_entries` 赋的紧凑 id。
        role: 控件类型 / 标签（UIA control_type、OCR 固定 "text"、视觉 label）。
        text: 展示文本（UIA name、OCR 文本、视觉 label）。
        bbox: 元素边界框（物理像素）。

    Returns:
        单行字符串，如 `<el id="uia:3" role="button" text="确定" bbox="100,200,180,240"/>`。
    """
    safe_text = (text or "").replace('"', "'").replace("\n", " ")
    return (
        f'<el id="{compact_id}" role="{role}" text="{safe_text}" '
        f'bbox="{bbox.x},{bbox.y},{bbox.width},{bbox.height}"/>'
    )


def _build_grounding_lines(
    snapshot: ScreenSnapshot,
    max_chars: int,
) -> tuple[list[str], dict[str, BBox]]:
    """构造截断后的渲染行与 id→bbox 查找表（同一次截断，两者严格一致）。

    截断口径同 `_truncate_perception_summary`（PERCEPTION_SUMMARY_MAX_TOKENS
    字符数近似），超限的元素**同时不进渲染文本与查找表**——保证「LLM 只能
    引用它实际看到的 id」：查找表若含渲染文本未展示的元素，会让 id 存在性
    核验对着 LLM 看不到的数据做判断，语义不一致。

    Args:
        snapshot: 完整感知快照。
        max_chars: 渲染文本字符数预算。

    Returns:
        (渲染行列表, compact_id -> BBox 查找表)。
    """
    entries = _iter_grounding_entries(snapshot)
    lines: list[str] = []
    table: dict[str, BBox] = {}
    total_chars = 0
    dropped = 0
    for compact_id, role, text, bbox in entries:
        line = _render_grounding_line(compact_id, role, text, bbox)
        if total_chars + len(line) + 1 > max_chars:
            dropped += 1
            continue
        lines.append(line)
        table[compact_id] = bbox
        total_chars += len(line) + 1

    if dropped:
        logger.warning(
            "_build_grounding_lines: 元素表截断，%d/%d 个元素超出 %d 字符预算，已丢弃",
            dropped,
            len(entries),
            max_chars,
        )
    return lines, table


def _serialize_snapshot_compact(
    snapshot: ScreenSnapshot,
    max_chars: int = PERCEPTION_SUMMARY_MAX_TOKENS,
) -> str:
    """纯函数：把三类感知元素序列化为紧凑 HTML 类文本，供 user 提示词注入（蓝图任务 7）。

    Args:
        snapshot: 完整感知快照。
        max_chars: 渲染文本字符数预算（默认 PERCEPTION_SUMMARY_MAX_TOKENS，与
            perception_summary 截断同构口径）。

    Returns:
        逐行 `<el .../>` 拼接的字符串；无可定位元素时返回空字符串。
    """
    lines, _ = _build_grounding_lines(snapshot, max_chars)
    return "\n".join(lines)


def derive_last_step_outcome(
    step_history: list[StepRecord],
) -> Literal["initial", "succeeded", "failed"]:
    """派生上一步结果三态（K4 紧后 §3.2，纯函数）。

    设计输入 notes/2026-08-05-llm-integration-survey-k3k4-actionspec.md §3.2：
    prompt 模板用 last_step_outcome 显式驱动不同段落（Agent-S2 manager.py 的
    failed_subtask 三态拆分先例），不让 LLM 从平铺 history 里自己猜上一步成败。

    三态判据：
      - "initial"   — step_history 为空（任务第一步）。
      - "failed"    — 最后一步带 control_error 或 perception_error。
      - "succeeded" — 其余（最后一步无错误）。

    注：设计输入的失败三分类（可恢复/不可恢复/用户拒绝）中，「用户拒绝」路径
    （control_error="人工拒绝执行…" + task_status=FAILED）经 supervisor 终态
    守卫（K5 ③）根本不会再进 LLM 规划——拒绝即任务收口，无需在此分类；
    可恢复/不可恢复的判断权留给 LLM（模板 failed 分支给出指导性文案）。

    Args:
        step_history: 完整 step_history（未截断亦可，只看最后一步）。

    Returns:
        "initial" / "succeeded" / "failed"。
    """
    if not step_history:
        return "initial"
    last = step_history[-1]
    if last.control_error is not None or last.perception_error is not None:
        return "failed"
    return "succeeded"


# ── PromptLoader ──────────────────────────────────────────────────────────────


class PromptLoader:
    """Supervisor 提示词加载器（Task 11B）。

    上下文截断（step_history 取最近 CONTEXT_STEP_WINDOW 步、
    perception_summary ≤ PERCEPTION_SUMMARY_MAX_TOKENS 字符）统一在此层完成，
    不在 Agent 节点内、不在 Jinja2 模板内。

    模板路径：Path(__file__).parent / "prompts"（不走 .env，不依赖 cwd）。
    加载方式：Jinja2 FileSystemLoader，实例构造时一次性建立 Environment，
              后续 render 复用（无 I/O 开销）。

    实例可直接注入 DesktopSupervisorAgent（满足 desktop_supervisor.PromptLoader Protocol）。

    示例用法::

        loader = PromptLoader()
        system, user = loader.render_supervisor(state)
    """

    def __init__(
        self,
        step_window: int = CONTEXT_STEP_WINDOW,
        summary_max_chars: int = PERCEPTION_SUMMARY_MAX_TOKENS,
        prompts_dir: Path | None = None,
    ) -> None:
        """初始化 PromptLoader，构建 Jinja2 Environment。

        Args:
            step_window: step_history 注入模板前保留的最近步数
                （默认 CONTEXT_STEP_WINDOW=10）。
            summary_max_chars: perception_summary 字符数上限
                （默认 PERCEPTION_SUMMARY_MAX_TOKENS=2000）。
            prompts_dir: 模板目录覆写（默认 Path(__file__).parent/"prompts"，
                供测试注入自定义目录，正常使用不需传此参数）。
        """
        self.step_window: int = step_window
        self.summary_max_chars: int = summary_max_chars
        self.prompts_dir: Path = prompts_dir if prompts_dir is not None else _PROMPTS_DIR
        self.env: Environment = Environment(
            loader=FileSystemLoader(str(self.prompts_dir)),
            autoescape=False,
            keep_trailing_newline=True,
        )

    def render_supervisor(
        self,
        state: DesktopTaskState,
    ) -> tuple[str, str]:
        """渲染 Supervisor 的 system + user 提示词。

        执行流程：
          1. 截断 step_history（取最近 step_window 步，转 dict 列表供模板消费）。
          2. 截断 perception_summary（超 summary_max_chars 字符时截断，注释标 tiktoken 后期）。
          3. 组装模板变量（与 supervisor_user.jinja2 变量约定对齐）。
          4. 渲染 supervisor_system.jinja2（系统提示词，无变量）。
          5. 渲染 supervisor_user.jinja2（用户提示词，注入运行时上下文）。
          6. 返回 (system_prompt, user_prompt)。

        Args:
            state: 当前 DesktopTaskState（包含 task_description / step_history /
                   perception_summary / perception_error / control_error /
                   stall_count / uia_hollow / capability_flags）。

        Returns:
            (system_prompt, user_prompt) 两个非空字符串。
            模板语法错误会向上抛出 jinja2.TemplateError（由调用方决定如何降级）。
        """
        # 1. step_history 截断（最近 step_window 步，转 dict 供模板消费）
        step_history_window: list[dict[str, object]] = _truncate_step_window(
            state.step_history,
            self.step_window,
        )

        # 2. perception_summary 字符数截断
        # 注释：当前用字符数近似 token 预算，后期替换为 tiktoken 精确计量（R6 工程假设）
        perception_summary: str | None = _truncate_perception_summary(
            state.perception_summary,
            self.summary_max_chars,
        )

        # 3. 组装模板变量（与 prompts/__init__.py 变量约定对齐）
        # K4 紧后 §3.2：last_step_outcome 三态显式驱动模板分支，
        # 不让 LLM 从平铺 history 里猜上一步成败
        user_vars: dict[str, object] = {
            "task_description": state.task_description,
            "step_history_window": step_history_window,
            "perception_summary": perception_summary,
            "last_step_outcome": derive_last_step_outcome(state.step_history),
            "errors": {
                "perception_error": state.perception_error,
                "control_error": state.control_error,
            },
            "capability_flags": state.capability_flags,
            "stall_count": state.stall_count,
            "uia_hollow": state.uia_hollow,
        }

        # 4. 渲染 system 提示词（supervisor_system.jinja2，无变量注入）
        system_template = self.env.get_template("supervisor_system.jinja2")
        system_prompt: str = system_template.render()

        # 5. 渲染 user 提示词（supervisor_user.jinja2，注入运行时上下文）
        user_template = self.env.get_template("supervisor_user.jinja2")
        user_prompt: str = user_template.render(**user_vars)

        logger.debug(
            "PromptLoader.render_supervisor: system=%d chars, user=%d chars, "
            "step_window=%d/%d, summary=%s",
            len(system_prompt),
            len(user_prompt),
            len(step_history_window),
            self.step_window,
            f"{len(perception_summary)} chars" if perception_summary is not None else "None",
        )

        return system_prompt, user_prompt

    def render_action_generation(
        self,
        state: DesktopTaskState,
        snapshot: ScreenSnapshot,
    ) -> tuple[str, str, dict[str, BBox]]:
        """渲染 ActionGeneratorAgent 的 system + user 提示词 + grounding 查找表。

        与 `render_supervisor` 的区别：不含任务分解/路由上下文，只含「翻译当前
        指令为一次动作调用」所需的最小上下文——当前指令、上一步结果三态、
        错误原文、三类感知元素合并后的紧凑 id 表。

        **grounding 表与渲染文本同一次构建**（PR #29 审查 BLOCK 收口）：查找表
        随渲染结果一并返回，消费方（ActionGeneratorAgent）不得自行重建——两次
        独立构建若预算不一致，LLM 可引用它没看到的 id 被放行（坐标级安全缺口，
        审查已实证）。原 `build_grounding_table` 公开入口因此删除。

        Args:
            state: 当前 DesktopTaskState（取 current_instruction / step_history /
                perception_error / control_error / uia_hollow）。
            snapshot: 本轮感知快照（须是 state.snapshot_ref 对应的最新快照，
                由调用方经 SnapshotStore 加载后传入——本方法不做快照加载 I/O）。

        Returns:
            (system_prompt, user_prompt, grounding_table)：前两者为非空字符串，
            grounding_table 是与 user_prompt 中元素表**严格一致**的
            compact_id -> BBox 查找表。
            模板语法错误会向上抛出 jinja2.TemplateError（由调用方决定如何降级）。
        """
        grounding_lines, grounding_table = _build_grounding_lines(
            snapshot,
            self.summary_max_chars,
        )
        grounding_table_text: str = "\n".join(grounding_lines)

        user_vars: dict[str, object] = {
            "current_instruction": state.current_instruction,
            "last_step_outcome": derive_last_step_outcome(state.step_history),
            "errors": {
                "perception_error": state.perception_error,
                "control_error": state.control_error,
            },
            "uia_hollow": state.uia_hollow,
            "grounding_table": grounding_table_text,
        }

        system_template = self.env.get_template("action_generation_system.jinja2")
        system_prompt: str = system_template.render()

        user_template = self.env.get_template("action_generation_user.jinja2")
        user_prompt: str = user_template.render(**user_vars)

        logger.debug(
            "PromptLoader.render_action_generation: system=%d chars, user=%d chars, "
            "grounding_table=%d chars / %d entries",
            len(system_prompt),
            len(user_prompt),
            len(grounding_table_text),
            len(grounding_table),
        )

        return system_prompt, user_prompt, grounding_table
