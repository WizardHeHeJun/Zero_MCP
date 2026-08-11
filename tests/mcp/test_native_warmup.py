"""原生扩展预热的行为契约 + 两个 stdio server 入口的结构守卫。

被测：``src/mcp/native_warmup.py``，以及 ``vts_behavior_mcp_server`` /
``desktop_mcp_server`` 的 ``__main__`` 入口是否在 ``mcp.run()`` **之前**预热。

为什么要有结构守卫：本修复的全部要害就是「import 发生在事件循环之前」这个
**时序**（判据与最小复现见 native_warmup 模块 docstring）。预热调用被删掉、
或被挪到 ``mcp.run()`` 之后，功能测试一律照绿——真 wire 上却退回无限期挂起
（Zero 2026-08-11 通报的原始症状）。故守卫按 **AST** 检查调用点与相对次序，
不用文本 find：docstring/注释里正当地写着 ``mcp.run()`` 字样，文本锚点会被撞飞
（`ai-docs/pitfalls.md` ⑦⑧）。

不标 ``zerorepo``（不读 D:\\Zero）；无原生依赖（预热目标用纯 Python 模块）。
"""

from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path

import pytest

from src.mcp.native_warmup import warm_native_extensions

SERVER_PATHS = [
    Path("src/mcp/vts_behavior_mcp_server.py"),
    Path("src/mcp/desktop_mcp_server.py"),
]


# ── 行为契约 ─────────────────────────────────────────────────────────────────


def test_warm_returns_successfully_imported_modules() -> None:
    """预热成功的模块名原样返回，且确实进了 sys.modules。"""
    warmed = warm_native_extensions(("zoneinfo", "decimal"))
    assert warmed == ["zoneinfo", "decimal"]
    assert "zoneinfo" in sys.modules
    assert "decimal" in sys.modules


def test_missing_module_does_not_block_startup(caplog: pytest.LogCaptureFixture) -> None:
    """缺包只记 warning 不抛——server 仍须能起（工具体各自优雅回退）。"""
    with caplog.at_level(logging.WARNING, logger="src.mcp.native_warmup"):
        warmed = warm_native_extensions(("zoneinfo", "definitely_not_a_real_module_xyz"))
    assert warmed == ["zoneinfo"]
    assert "definitely_not_a_real_module_xyz" in caplog.text


def test_empty_sequence_is_noop() -> None:
    """空清单不报错、返回空表（调用方无需自己判空）。"""
    assert warm_native_extensions(()) == []


# ── 结构守卫：预热调用必须在 mcp.run() 之前 ──────────────────────────────────


def _main_block(path: Path) -> ast.If:
    """取模块里的 ``if __name__ == "__main__":`` 块（AST，非文本匹配）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
        ):
            return node
    pytest.fail(f"{path} 没有 __main__ 入口块")


def _call_line(block: ast.If, func_name: str) -> int:
    """返回块内**最靠前**的 ``func_name(...)`` 调用行号；找不到则 fail。

    取 min 而非「walk 到的第一个」：``ast.walk`` 是广度优先，产出次序与源码
    行序无关，拿首个命中会让本守卫的次序断言变成看运气。
    """
    lines = [
        node.lineno
        for node in ast.walk(block)
        if isinstance(node, ast.Call)
        and (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else getattr(node.func, "id", None)
        )
        == func_name
    ]
    if not lines:
        pytest.fail(f"入口块内未找到 {func_name}(...) 调用")
    return min(lines)


@pytest.mark.parametrize("path", SERVER_PATHS, ids=lambda p: p.stem)
def test_warmup_precedes_event_loop(path: Path) -> None:
    """两个 stdio server 都在 ``mcp.run()`` 之前调 ``warm_native_extensions``。

    次序是本修复的全部要害：反过来等于没修（见模块 docstring）。
    """
    block = _main_block(path)
    warm_line = _call_line(block, "warm_native_extensions")
    run_line = _call_line(block, "run")
    assert warm_line < run_line, (
        f"{path}：warm_native_extensions 在第 {warm_line} 行、mcp.run 在第 {run_line} 行——"
        "预热必须早于事件循环启动，否则首次 numpy 系 import 会无限期死锁。"
    )


@pytest.mark.parametrize("path", SERVER_PATHS, ids=lambda p: p.stem)
def test_warmup_is_gated_by_feature_flag(path: Path) -> None:
    """预热在 **``enabled`` 那个** flag 分支内——flag 关时不拉依赖（零回归不变式）。

    断言绑到 ``if enabled:`` 这个具体条件，而非「被某个 ``if`` 包住」：后者在
    ``if 1 == 2:`` 这类与 flag 毫无关系、甚至恒假的包裹下照样判绿（审查门
    2026-08-11 现场变异实证），兜不住「条件被换成看似合理但错的判断」——
    而那正是这条守卫存在的全部理由。``enabled`` 由两 server 各自的
    ``_is_enabled()`` 赋值，是它们唯一的 feature flag 出口。
    """
    block = _main_block(path)
    warm_line = _call_line(block, "warm_native_extensions")
    gated = [
        inner
        for inner in ast.walk(block)
        if isinstance(inner, ast.If)
        and inner is not block
        and isinstance(inner.test, ast.Name)
        and inner.test.id == "enabled"
        and any(
            isinstance(n, ast.Call) and getattr(n.func, "id", None) == "warm_native_extensions"
            for n in ast.walk(inner)
        )
    ]
    assert gated, (
        f"{path}：第 {warm_line} 行的预热不在 `if enabled:` 分支内——"
        "flag 关时不得拉业务/原生依赖（零回归不变式）。"
    )


@pytest.mark.parametrize("path", SERVER_PATHS, ids=lambda p: p.stem)
def test_flag_off_tools_gate_before_any_lazy_import(path: Path) -> None:
    """每个 ``@mcp.tool`` 体里 ``_require_enabled()`` 严格早于任何惰性 import。

    这条守的是**上面三条守不到的那一半**：预热在 ``if enabled:`` 内 ⇒ flag 关的
    部署**没有**预热保护，那时唯一挡住「事件循环内首次 import numpy」的，就是
    工具一进来先在门上抛 ToolError、根本走不到下面那行延迟 import。

    ⚠ 它为什么必须单独钉：把 ``flags = _get_flags()`` / ``import ... perception``
    提到门之前，读起来完全无害，而**全部现有测试照绿**——flag 关的用例只断言
    「抛 ToolError」，提前 import 之后那个 ToolError 照样抛，只是在真 wire 上
    先无限期卡在 numpy 扩展加载里（Zero 2026-08-11 §3.5 就此发问）。
    也就是说这条性质一旦破，症状只在 flag 关的真部署上出现，测试面全盲。

    惰性入口按 AST 取两类：函数体内的 ``import`` 语句，以及两 server 各自的
    惰性服务/能力获取器（它们内部再做 import）。
    """
    lazy_getters = {"_get_service", "_get_flags"}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tools = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "tool"
            for d in node.decorator_list
        )
    ]
    assert tools, f"{path}：没解析出任何 @mcp.tool——本守卫已失去锚点"

    for tool in tools:
        gates = [
            n.lineno
            for n in ast.walk(tool)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_require_enabled"
        ]
        lazies = [
            n.lineno
            for n in ast.walk(tool)
            if isinstance(n, (ast.Import, ast.ImportFrom))
            or (isinstance(n, ast.Call) and getattr(n.func, "id", None) in lazy_getters)
        ]
        assert gates, f"{path}::{tool.name} 未调 _require_enabled()——flag 关时该工具不设防"
        if not lazies:
            continue
        assert min(gates) < min(lazies), (
            f"{path}::{tool.name}：惰性 import 在第 {min(lazies)} 行、门在第 {min(gates)} 行——"
            "flag 关的部署没有预热保护，门之前的 import 会在事件循环内首次触达 numpy 而死锁。"
        )


@pytest.mark.parametrize("path", SERVER_PATHS, ids=lambda p: p.stem)
def test_enabled_comes_from_feature_flag_helper(path: Path) -> None:
    """``enabled`` 确由 ``_is_enabled()`` 赋值——否则上一条守卫盯的是个空壳名字。

    上一条把语义绑在 ``enabled`` 这个**名字**上；若有人把它改成别处来的值
    （或删掉赋值），那条断言就退化成「名字对得上」而不再代表 feature flag。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assigned = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "enabled" for t in node.targets)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "_is_enabled"
    ]
    assert assigned, f"{path}：`enabled` 不是由 `_is_enabled()` 赋值——flag 守卫已失去语义锚点"
