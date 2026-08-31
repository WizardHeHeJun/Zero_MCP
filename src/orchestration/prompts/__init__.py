"""Supervisor 提示词模板包（Task 11A）。

包含两个 Jinja2 模板文件：
  - supervisor_system.jinja2  : 系统提示词（角色 + 任务分解 + 输出 JSON schema + 安全提醒）
  - supervisor_user.jinja2    : 用户提示词（注入运行时上下文变量）

设计原则：
  - 模板与代码分离：提示词可版本化、可测，不在业务逻辑里拼字符串（agent-framework-rules）。
  - 模板不含 Python 逻辑：条件判断仅用 Jinja2 原生语法（{% if %}），不调用 Python 函数。
  - 截断不在模板内：step_history 截断、perception_summary 长度限制由 prompt_loader.py 处理。
  - 上下文预算显式管理：大对象不进模板变量（传截断后的文本，不传对象引用）。

模板路径由 PromptLoader（Task 11B）通过 Jinja2 FileSystemLoader 加载，
路径固定为 Path(__file__).parent（不依赖 cwd，不走 .env）。

变量约定（在 supervisor_user.jinja2 使用）：
  task_description         : str  — 原始任务描述
  step_history_window      : list[dict] — 截断后的步骤历史（每项含 step_index/agent/
                             instruction/task_status/control_error/perception_error）
  perception_summary       : str | None — 最新感知摘要（已截断）
  last_step_outcome        : str  — 上一步结果三态 "initial"|"succeeded"|"failed"
                             （K4 紧后 §3.2，prompt_loader.derive_last_step_outcome
                             派生，显式驱动模板分支，不让 LLM 从 history 猜）
  errors                   : dict  — {"perception_error": ..., "control_error": ...}
  capability_flags         : dict[str, bool] — 能力协商结果
  stall_count              : int   — 连续停滞计数
  uia_hollow               : bool  — 当前窗口是否 UIA 空洞（影响提示词）
"""

from pathlib import Path

#: 模板目录绝对路径（供 PromptLoader 的 FileSystemLoader 使用）
TEMPLATES_DIR: Path = Path(__file__).parent

__all__ = ["TEMPLATES_DIR"]
