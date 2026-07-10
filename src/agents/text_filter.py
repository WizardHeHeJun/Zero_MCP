"""屏幕文本注入过滤（agents 层共享工具，纯函数）。

`sanitize_screen_text` 对屏幕 OCR/UIA 文本执行三层注入过滤，屏蔽
提示词注入攻击（Prompt Injection）。

设计依据：
- 注入过滤三层 (arXiv:2506.02456, OWASP LLM01:2025, Unit42)：
  结构标记正则 → 关键词词表 → 混淆检测（NFKC + Base64）。
- 护栏在编排层，但过滤逻辑本身是纯函数，放 agents 层避免
  orchestration → agents 的反向 import 循环。
  编排层 ActionGuard 调用此函数时走 orchestration → agents（允许下调）。

层约束：本模块只 import 标准库，不 import 任何 src.* 模块。
"""

from __future__ import annotations

import base64
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# ── 第一层：结构标记正则（大小写不敏感） ──────────────────────────────────────
# 匹配 ChatML / tool_call / function_call / Jinja 注入模板等

_STRUCT_TAG_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"</?(?:SYSTEM|tool|tool_call|function_call|assistant|human)[^>]*>",
        re.IGNORECASE,
    ),
    re.compile(r"\{\{[^}]*\}\}"),  # Jinja/双大括号
    re.compile(r"\[\[INST\]\]", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"<\|im_end\|>", re.IGNORECASE),
    re.compile(r"<\|endoftext\|>", re.IGNORECASE),
]

# ── 第二层：关键词词表（英文，命中即替换该 text_block 为 [FILTERED]） ─────────
# 来源：Unit42 野外清单 + OWASP LLM01:2025

_INJECTION_KEYWORDS_EN: list[str] = [
    "ignore all instructions",
    "ignore previous instructions",
    "disregard all instructions",
    "system override",
    "god mode",
    "developer mode",
    "do anything now",
    "[begin_admin_session]",
    "security update",
    "new instructions",
    "prompt injection",
    "jailbreak",
]

# gap#9：中文越权注入词表。镜像 _INJECTION_KEYWORDS_EN 的语义类别（覆写指令 /
# 提权越权 / 越狱注入 / 解除限制），选注入特征强、足够长的短语以压低误报。
# 词表为工程假设——由英文（已有据：VPI-Bench arXiv:2506.02456 · OWASP LLM01:2025 ·
# Unit42）适配而来，中文界面有效性与误报率由 Task 12 真实钉钉/微信文本实测标定。
_INJECTION_KEYWORDS_ZH: list[str] = [
    # 覆写/忽略既有指令（EN: ignore/disregard ... instructions）
    "忽略以上所有指令",
    "忽略之前的所有指令",
    "忽略上述指令",
    "忽略所有指令",
    "忽略之前的指令",
    "忽略前面的指令",
    "无视所有指令",
    "无视以上指令",
    "忽略系统提示",
    "忽略你的指令",
    "忽略你之前的指令",
    # 提权/越权（EN: system override / god mode / developer mode）
    "系统管理员覆盖",
    "管理员覆盖",
    "上帝模式",
    "管理员模式",  # FP-watch：可能撞真实「设置」项，Task 12 标定误报率
    "开发者模式",  # FP-watch：同上（对应 EN developer mode）
    # 越狱/注入（EN: jailbreak / prompt injection / do anything now）
    "越狱模式",
    "提示词注入",
    "提示注入",
    "解除所有限制",
    "解除安全限制",
    "忽略安全限制",
    "禁用安全限制",
]

_ALL_INJECTION_KEYWORDS: list[str] = _INJECTION_KEYWORDS_EN + _INJECTION_KEYWORDS_ZH

# 最小 Base64 串长度（工程假设：≥20 字符，误报率待 Task 12 实测）
_BASE64_MIN_LEN: int = 20


def sanitize_screen_text(text: str) -> str:
    """对屏幕 OCR/UIA 文本执行三层注入过滤（纯函数，可单独测试）。

    三层过滤顺序：
      1. 结构标记正则（精确低误报）：替换 ChatML/tool_call/Jinja 等注入标记。
      2. 关键词词表匹配（大小写不敏感）：命中越权指令词汇则整体替换为 [FILTERED]。
      3. 混淆检测：NFKC 规范化后重跑 1/2 层；对长度 ≥ 20 的 ASCII 串尝试
         Base64 解码，解码结果命中词表则过滤。

    中文越权词表见 _INJECTION_KEYWORDS_ZH（gap#9，工程假设，误报率待 Task 12 标定）。

    Args:
        text: 待过滤的原始屏幕文本。

    Returns:
        过滤后的文本；命中关键词级过滤时返回 "[FILTERED]"。
    """
    # 第一层：结构标记正则替换
    result = text
    for pattern in _STRUCT_TAG_PATTERNS:
        result = pattern.sub("[FILTERED]", result)

    # 第二层：关键词词表（大小写不敏感）
    lower_result = result.lower()
    for keyword in _ALL_INJECTION_KEYWORDS:
        if keyword.lower() in lower_result:
            logger.warning("sanitize_screen_text: 命中注入关键词 %r，整体过滤", keyword)
            return "[FILTERED]"

    # 第三层：混淆检测（NFKC 规范化 + Base64 解混淆）
    normalized = unicodedata.normalize("NFKC", result)
    if normalized != result:
        # 规范化后重跑第一层
        for pattern in _STRUCT_TAG_PATTERNS:
            normalized = pattern.sub("[FILTERED]", normalized)
        # 规范化后重跑第二层
        lower_normalized = normalized.lower()
        for keyword in _ALL_INJECTION_KEYWORDS:
            if keyword.lower() in lower_normalized:
                logger.warning(
                    "sanitize_screen_text: NFKC 规范化后命中注入关键词 %r，整体过滤",
                    keyword,
                )
                return "[FILTERED]"
        result = normalized

    # Base64 解混淆检测
    # 提取所有长度 ≥ 20 的纯 ASCII 串（Base64 字符集子集）
    ascii_candidates = re.findall(rf"[A-Za-z0-9+/=]{{{_BASE64_MIN_LEN},}}", result)
    for candidate in ascii_candidates:
        try:
            # 补齐 padding
            padding = (4 - len(candidate) % 4) % 4
            decoded_bytes = base64.b64decode(candidate + "=" * padding)
            decoded_str = decoded_bytes.decode("utf-8", errors="ignore").lower()
            for keyword in _ALL_INJECTION_KEYWORDS:
                if keyword.lower() in decoded_str:
                    logger.warning(
                        "sanitize_screen_text: Base64 解码后命中注入关键词 %r，整体过滤",
                        keyword,
                    )
                    return "[FILTERED]"
        except Exception as exc:
            # Base64 解码失败，非注入，跳过；记录 debug 日志供排查
            logger.debug("sanitize_screen_text: Base64 解码失败（非注入），跳过: %s", exc)

    return result
