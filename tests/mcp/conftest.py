"""tests/mcp 共享 pytest 配置：`zerorepo` 跨仓用例的**覆盖归零守卫**。

`zerorepo` 用例真起 D:\\Zero server 子进程 / 真读 Zero 源码，前置缺失（Zero 不在位、缺 torch、
缺权重、缺 aiosqlite、端口被占、连不上）时一律 `pytest.skip` 不拖红——日常开发这是对的，
但**跨仓覆盖会静默归零**：Zero `.gitignore` 忽略 `artifacts/`（其 `git ls-files artifacts` 为空），
一次 `git clean` / 新克隆 / 换机器，就让「T4 真 prosody 接线」这类**唯一的 live 用例**变成绿色的
skip——看板全绿，而跨仓契约实际无人验证。默认输出还不打印 skip 原因（本仓已补 `-ra`）。

**用法**：联调清单 / 跨仓对齐时设 `ZERO_LINK_E2E_STRICT=1` 跑一次——`zerorepo` 用例的**任何 skip
转 fail**，把「本该真跑却没跑」暴露成红。默认不设 = 行为逐字不变（零回归）。

    # 平时（缺前置即跳过，不拖红）
    pytest -m "not realenv"
    # 跨仓对齐/发版前（要求 zerorepo 全部真跑）
    ZERO_LINK_E2E_STRICT=1 pytest -m zerorepo
"""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest

STRICT_ENV = "ZERO_LINK_E2E_STRICT"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def zerorepo_strict_enabled() -> bool:
    """`ZERO_LINK_E2E_STRICT` 是否开启（**运行时**读 env，便于 monkeypatch 与逐次调整）。"""
    return os.environ.get(STRICT_ENV, "").strip().lower() in _TRUTHY


def should_convert_skip(*, report_skipped: bool, has_zerorepo_marker: bool) -> bool:
    """是否把本条 skip 报告改判为 fail（纯函数，便于判别性单测）。

    三者同时成立才转：报告确为 skipped · 用例带 `zerorepo` 标记 · STRICT 开启。
    非 zerorepo 的 skip（如 realenv、缺可选依赖的普通用例）**不受影响**。
    """
    return report_skipped and has_zerorepo_marker and zerorepo_strict_enabled()


def strict_failure_message(nodeid: str, reason: str) -> str:
    """转 fail 时的报告文案——直指「本该真跑」而非让人以为是用例本身坏了。"""
    return (
        f"[{STRICT_ENV}] zerorepo 用例被 skip，但 STRICT 模式要求它真跑：\n"
        f"  用例：{nodeid}\n"
        f"  skip 原因：{reason}\n"
        f"跨仓 live 覆盖在此条上等于零。请补齐前置（D:\\Zero 在位 / conda env "
        f"affective-expression / Zero artifacts 权重在盘 / 端口空闲），或确认本条确实不该真跑后，"
        f"不设 {STRICT_ENV} 重跑。"
    )


def _skip_reason(report: pytest.TestReport) -> str:
    """从 skipped 报告里取原因；longrepr 形状为 (path, lineno, reason) 或已是字符串。"""
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr) if longrepr else "(未提供原因)"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """STRICT 开启时把 `zerorepo` 用例的 skip 报告改判为 fail。

    用 makereport 钩子集中处理，而非在 70+ 个 `pytest.skip()` 调用点逐个加分支——调用点零改动，
    且覆盖 setup/call/teardown 三阶段的所有 skip（含 fixture 内跳过）。
    """
    report = yield
    if should_convert_skip(
        report_skipped=report.skipped,
        has_zerorepo_marker=item.get_closest_marker("zerorepo") is not None,
    ):
        report.outcome = "failed"
        report.longrepr = strict_failure_message(item.nodeid, _skip_reason(report))
    return report
