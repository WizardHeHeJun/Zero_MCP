"""主备 LLM 模型单次切换调用（蓝图决策 F，从 `DesktopSupervisorAgent.plan` 抽取）。

设计输入 notes/2026-08-05-llm-integration-survey-k3k4-actionspec.md §2.2：
主模型调用异常时自动切备用重试**一次**，该次不再回退（防主备互踢死循环）——
仿 UFO llm_call.py get_completion(use_backup_engine) 递归一次后置 False。

抽取原因（蓝图决策 F）：`ActionGeneratorAgent`（`src/orchestration/action_generator.py`）
需要同款主备单次切换语义，但两个调用方（Supervisor / 生成层）的失败文案与增量
形状不同——本模块只负责「调用哪个模型、切不切备用」，不代管失败文案；调用方
捕获 `LLMFallbackError` 后按各自 state 增量契约格式化。

层依赖：本文件不 import agents/memory/storage 层，纯编排层工具函数。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


class LLMFallbackError(Exception):
    """主模型调用失败、且（无备用 或 备用同样失败）时抛出。

    携带两次尝试的原始异常，供调用方按自身 state 增量契约格式化失败文案
    （K3 §2.2 单次切换不递归：本异常最多在两次调用尝试后才抛出）。

    Attributes:
        primary_error: 主模型调用异常。
        fallback_model: 本次是否尝试了备用模型；None=未配置备用（未尝试）。
        fallback_error: 备用模型调用异常；fallback_model 为 None 时恒为 None。
    """

    def __init__(
        self,
        primary_error: Exception,
        fallback_model: str | None,
        fallback_error: Exception | None = None,
    ) -> None:
        self.primary_error = primary_error
        self.fallback_model = fallback_model
        self.fallback_error = fallback_error
        super().__init__(self._format())

    def _format(self) -> str:
        if self.fallback_model is None:
            return f"主模型调用失败（无备用模型）: {self.primary_error}"
        return (
            f"主模型调用失败: {self.primary_error}；"
            f"备用模型 {self.fallback_model!r} 调用失败: {self.fallback_error}"
        )


async def call_with_single_fallback[T](
    primary_model: str,
    fallback_model: str | None,
    call_fn: Callable[[str], Awaitable[T]],
) -> T:
    """主备模型单次切换调用。

    主模型调用异常时，若配置了备用模型，自动切换重试**恰一次**（不再递归/
    回退，防主备互踢死循环）；备用同样失败或未配置备用时抛出
    `LLMFallbackError`（携带两次尝试的原始异常，供调用方格式化失败增量）。

    Args:
        primary_model: 主模型 ID。
        fallback_model: 备用模型 ID；None=不配置备用（主模型异常即判定终态失败）。
        call_fn: 单次调用逻辑（模型 ID -> awaitable 结果）；调用方以闭包捕获
            system/user prompt 等调用上下文，本函数不关心调用内容。

    Returns:
        call_fn 的成功返回值（主模型或备用模型任一次成功即返回，不再继续尝试）。

    Raises:
        LLMFallbackError: 主模型失败且（无备用 或 备用也失败）。
    """
    try:
        return await call_fn(primary_model)
    except Exception as primary_exc:
        if fallback_model is None:
            logger.warning(
                "call_with_single_fallback: 主模型 %r 调用失败（%s），无备用模型",
                primary_model,
                primary_exc,
            )
            raise LLMFallbackError(primary_exc, fallback_model=None) from primary_exc

        logger.warning(
            "call_with_single_fallback: 主模型 %r 调用失败（%s），切备用模型 %r 重试一次",
            primary_model,
            primary_exc,
            fallback_model,
        )
        try:
            return await call_fn(fallback_model)
        except Exception as fallback_exc:
            logger.warning(
                "call_with_single_fallback: 备用模型 %r 也失败（%s），不再回退（防互踢死循环）",
                fallback_model,
                fallback_exc,
            )
            raise LLMFallbackError(primary_exc, fallback_model, fallback_exc) from fallback_exc
