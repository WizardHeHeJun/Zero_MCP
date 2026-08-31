"""llm_fallback 单测（ActionSpec 生成层蓝图 PR-β 任务 3/10）。

覆盖 `call_with_single_fallback` 四态 + 单次不递归不变式：
  1. 主模型成功 → 不碰备用，恰一次调用。
  2. 主模型异常 + 有备用 → 切备用重试一次成功。
  3. 主备均异常 → 抛 LLMFallbackError，恰两次调用（单次切换不递归）。
  4. 无备用模型 → 主模型异常即抛 LLMFallbackError，恰一次调用。
  5. LLMFallbackError 携带的 primary_error/fallback_model/fallback_error 字段正确。
"""

from __future__ import annotations

import pytest

from src.orchestration.llm_fallback import LLMFallbackError, call_with_single_fallback


async def test_primary_success_fallback_not_used() -> None:
    """主模型成功时不碰备用：恰一次调用，返回主模型结果。"""
    calls: list[str] = []

    async def call_fn(model: str) -> str:
        calls.append(model)
        return f"result-from-{model}"

    result = await call_with_single_fallback("primary", "backup", call_fn)

    assert result == "result-from-primary"
    assert calls == ["primary"]


async def test_primary_fails_fallback_succeeds() -> None:
    """主模型异常 + 配了备用 → 切备用重试一次成功。"""
    calls: list[str] = []

    async def call_fn(model: str) -> str:
        calls.append(model)
        if model == "primary":
            raise RuntimeError("primary boom")
        return "backup-ok"

    result = await call_with_single_fallback("primary", "backup", call_fn)

    assert result == "backup-ok"
    assert calls == ["primary", "backup"]


async def test_both_models_fail_raises_with_two_calls() -> None:
    """主备均异常 → 抛 LLMFallbackError，恰两次调用（单次切换不递归）。"""
    calls: list[str] = []

    async def call_fn(model: str) -> str:
        calls.append(model)
        raise RuntimeError(f"{model} boom")

    with pytest.raises(LLMFallbackError) as exc_info:
        await call_with_single_fallback("primary", "backup", call_fn)

    assert calls == ["primary", "backup"], "备用失败后不得再回退主模型（防互踢死循环）"
    err = exc_info.value
    assert err.fallback_model == "backup"
    assert "primary boom" in str(err.primary_error)
    assert err.fallback_error is not None
    assert "backup boom" in str(err.fallback_error)


async def test_no_fallback_configured_raises_after_one_call() -> None:
    """无备用模型 → 主模型异常即抛 LLMFallbackError，恰一次调用。"""
    calls: list[str] = []

    async def call_fn(model: str) -> str:
        calls.append(model)
        raise RuntimeError("primary boom")

    with pytest.raises(LLMFallbackError) as exc_info:
        await call_with_single_fallback("primary", None, call_fn)

    assert calls == ["primary"]
    err = exc_info.value
    assert err.fallback_model is None
    assert err.fallback_error is None
    assert "primary boom" in str(err.primary_error)


def test_llm_fallback_error_str_no_fallback() -> None:
    """LLMFallbackError.__str__ 无备用时不含具体备用模型 ID（只提示"无备用"）。"""
    err = LLMFallbackError(RuntimeError("boom"), fallback_model=None)
    assert "boom" in str(err)
    assert "无备用模型" in str(err)


def test_llm_fallback_error_str_with_fallback() -> None:
    """LLMFallbackError.__str__ 有备用时同时含主备两条异常信息。"""
    err = LLMFallbackError(
        RuntimeError("primary boom"),
        fallback_model="backup-model",
        fallback_error=RuntimeError("backup boom"),
    )
    text = str(err)
    assert "primary boom" in text
    assert "backup-model" in text
    assert "backup boom" in text
