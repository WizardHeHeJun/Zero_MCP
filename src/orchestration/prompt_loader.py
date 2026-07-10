"""Supervisor 提示词加载与上下文截断（Task 11B）。

PromptLoader 负责：
  1. 从 Path(__file__).parent/"prompts" 用 Jinja2 FileSystemLoader 加载模板
     （路径固定，不依赖 cwd，不走 .env）。
  2. 截断 step_history：只取最近 CONTEXT_STEP_WINDOW 步（默认 10），
     超出部分在此层丢弃（不写归档——归档由 Supervisor 节点的 StepArchive 负责）。
  3. 截断 perception_summary：字符数超过 PERCEPTION_SUMMARY_MAX_TOKENS（默认 2000）
     时在此截断。
     注释：当前用字符数近似 token 预算（留 20% 余量），后期可替换为
     tiktoken.encoding_for_model("cl100k_base").encode() 精确计量（R6 工程假设）。
  4. 渲染 system / user 两个提示词，返回 tuple[str, str]。

设计约束（agent-framework-rules / orchestration-rules）：
  - prompt 模板与代码分离（可版本化、可测）。
  - 上下文截断统一在此层，不在 Agent 节点内、不在模板内。
  - 公开接口完整类型注解。
  - 不含 I/O，render_supervisor 为同步方法（模板渲染是 CPU 运算，无网络/磁盘等待）。
  - 不 import 任何 src.agents.* 以外的上层模块（本文件在 orchestration 层，
    只 import orchestration.state 和 orchestration.prompts）。
"""

from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

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
        user_vars: dict[str, object] = {
            "task_description": state.task_description,
            "step_history_window": step_history_window,
            "perception_summary": perception_summary,
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
