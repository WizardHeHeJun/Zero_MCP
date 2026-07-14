"""安全门模块（编排层）。

ActionGuard 实现三级风险判定、TOCTOU 验证与屏幕文本过滤。
位于编排层（src/orchestration/safety/），不依赖记忆层或存储层。

依据：
- VPI-Bench (arXiv:2506.02456)：护栏在编排层比系统提示防御更有效。
- TOCTOU (arXiv:2604.18860)：坐标点击是 notification hijacking 主命中点。
"""

from src.agents.text_filter import sanitize_screen_text
from src.orchestration.safety.action_guard import ActionGuard
from src.orchestration.safety.incident_reporter import FileIncidentReporter

__all__ = ["ActionGuard", "FileIncidentReporter", "sanitize_screen_text"]
