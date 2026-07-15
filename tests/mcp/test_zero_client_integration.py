"""test_zero_client_integration.py — ZeroLinkClient 集成/可靠性测试。

验证目标（mock 测不到的真实失败路径）：
1. 真 stdio spawn 失败 → ZeroLinkConnectionError（不 hang、不裸 crash、不泄漏子进程）。
2. 进程起来但 initialize 失败 → ZeroLinkConnectionError（不 hang、优雅回退）。
3. 连接失败后调用方 catch ZeroLinkConnectionError → 流程继续（不崩进程）。
4. AsyncExitStack 失败清理：__aenter__ 抛异常后 exit_stack is None 且 session is None。
5. 默认关零回归：import src.mcp.zero 无副作用（无连接建立）。
6. ZERO_LINK_ENABLED=false 时 __aenter__ 抛 ZeroLinkDisabledError（快速路径，不尝试 spawn）。

标记说明：
- @pytest.mark.realenv：涉及真子进程 spawn（不依赖 D:/Zero，但需要 Python 可执行文件）。
  本文件的真子进程测试本地必须跑绿；CI 如需跳过可加 -m "not realenv"。
- @pytest.mark.zerorepo：不涉及 D:/Zero 源码，此文件不用该标记。

测试不依赖任何外部 server 或网络，全部用「不存在的命令」或「立即退出」的假命令
触发真实失败路径。
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from src.mcp.zero.client import (
    ZeroLinkClient,
    ZeroLinkConnectionError,
    ZeroLinkDisabledError,
)

# ---------------------------------------------------------------------------
# 场景 1：真 stdio spawn 失败 → ZeroLinkConnectionError
# ---------------------------------------------------------------------------


@pytest.mark.realenv
async def test_stdio_spawn_nonexistent_command_raises_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ZERO_SERVER_COMMAND 指向不存在的命令时，__aenter__ 快速抛 ZeroLinkConnectionError。

    验证：
    - 不 hang（pytest-asyncio 超时守卫为 10s，实际应 <2s 失败）
    - 抛出 ZeroLinkConnectionError（不是裸 OSError / FileNotFoundError）
    - exit_stack 和 session 均为 None（无泄漏）
    """
    monkeypatch.setenv("ZERO_LINK_ENABLED", "true")
    monkeypatch.setenv("ZERO_LINK_TRANSPORT", "stdio")
    monkeypatch.setenv("ZERO_SERVER_COMMAND", "definitely_not_a_real_command_xyz_12345")
    monkeypatch.setenv("ZERO_SERVER_ARGS", "[]")
    monkeypatch.setenv("ZERO_SERVER_CWD", ".")

    client = ZeroLinkClient()

    with pytest.raises(ZeroLinkConnectionError):
        await asyncio.wait_for(client.__aenter__(), timeout=10.0)

    # 失败后 exit_stack / session 均为 None（已清理，无泄漏）
    assert client.exit_stack is None, "spawn 失败后 exit_stack 应为 None（已 aclose）"
    assert client.session is None, "spawn 失败后 session 应为 None"


# ---------------------------------------------------------------------------
# 场景 2：进程起来但 initialize 失败 → ZeroLinkConnectionError
# ---------------------------------------------------------------------------


@pytest.mark.realenv
async def test_stdio_process_exits_immediately_raises_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ZERO_SERVER_COMMAND 指向立即退出（sys.exit(1)）的 Python 脚本时，
    stdio_client 能建连但 initialize() 因 EOF/broken pipe 失败，
    __aenter__ 抛 ZeroLinkConnectionError。

    验证：
    - 不 hang（超时守卫 15s；initialize 读到 EOF 应快速失败）
    - 抛出 ZeroLinkConnectionError
    - exit_stack / session 均为 None
    """
    monkeypatch.setenv("ZERO_LINK_ENABLED", "true")
    monkeypatch.setenv("ZERO_LINK_TRANSPORT", "stdio")
    # Python -c "import sys; sys.exit(1)" 会立即退出、不发送任何 MCP 帧
    # 注意：ZERO_SERVER_ARGS 为 JSON 列表字符串
    monkeypatch.setenv("ZERO_SERVER_COMMAND", sys.executable)
    monkeypatch.setenv("ZERO_SERVER_ARGS", '["-c", "import sys; sys.exit(1)"]')
    monkeypatch.setenv("ZERO_SERVER_CWD", ".")

    client = ZeroLinkClient()

    with pytest.raises(ZeroLinkConnectionError):
        await asyncio.wait_for(client.__aenter__(), timeout=15.0)

    assert client.exit_stack is None, "initialize 失败后 exit_stack 应为 None（已 aclose）"
    assert client.session is None, "initialize 失败后 session 应为 None"


# ---------------------------------------------------------------------------
# 场景 3：连接失败后调用方 catch → 流程继续（graceful 语义）
# ---------------------------------------------------------------------------


@pytest.mark.realenv
async def test_connection_error_catchable_process_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """__aenter__ 抛 ZeroLinkConnectionError，调用方 catch 后流程不崩。

    验证「无 server 时优雅回退」的真实生产保证：
    - async with 抛 ZeroLinkConnectionError → 调用方 catch
    - catch 后后续代码正常执行（fallback_triggered=True）
    - 整个 coroutine 不 hang、不崩进程
    """
    monkeypatch.setenv("ZERO_LINK_ENABLED", "true")
    monkeypatch.setenv("ZERO_LINK_TRANSPORT", "stdio")
    monkeypatch.setenv("ZERO_SERVER_COMMAND", "definitely_not_a_real_command_xyz_12345")
    monkeypatch.setenv("ZERO_SERVER_ARGS", "[]")
    monkeypatch.setenv("ZERO_SERVER_CWD", ".")

    fallback_triggered = False

    try:
        async with ZeroLinkClient():
            # 不应到达这里
            pass  # pragma: no cover
    except ZeroLinkConnectionError:
        # 模拟编排层的降级处理
        fallback_triggered = True

    assert fallback_triggered, "ZeroLinkConnectionError 应被调用方 catch，fallback 路径触发"


# ---------------------------------------------------------------------------
# 场景 4：AsyncExitStack 失败清理（__aenter__ 异常后状态断言）
# ---------------------------------------------------------------------------


@pytest.mark.realenv
async def test_aenter_failure_cleans_exit_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """__aenter__ 失败（真 spawn 失败）后：
    - client.exit_stack is None（stack 已 aclose，未泄漏）
    - client.session is None
    - 同一 client 实例可安全 GC（无悬空资源句柄）
    """
    monkeypatch.setenv("ZERO_LINK_ENABLED", "true")
    monkeypatch.setenv("ZERO_LINK_TRANSPORT", "stdio")
    monkeypatch.setenv("ZERO_SERVER_COMMAND", "definitely_not_a_real_command_xyz_12345")
    monkeypatch.setenv("ZERO_SERVER_ARGS", "[]")
    monkeypatch.setenv("ZERO_SERVER_CWD", ".")

    client = ZeroLinkClient()

    # 初始状态
    assert client.exit_stack is None
    assert client.session is None

    with pytest.raises(ZeroLinkConnectionError):
        await asyncio.wait_for(client.__aenter__(), timeout=10.0)

    # 失败后状态：exit_stack 已清理，session 未赋值
    assert client.exit_stack is None, (
        "__aenter__ 失败时 exit_stack 应为 None（self.exit_stack = stack 仅在成功时执行）"
    )
    assert client.session is None, (
        "__aenter__ 失败时 session 应为 None（self.session = session 仅在成功时执行）"
    )


# ---------------------------------------------------------------------------
# 场景 5：默认关零回归 — import 无副作用
# ---------------------------------------------------------------------------


def test_import_zero_mcp_package_no_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """import src.mcp.zero 及子模块无副作用：不建连接、不读 env、不起子进程。

    验证：
    - 模块 import 后 ZeroLinkClient 实例 exit_stack / session 均为 None
    - _is_enabled() 在 ZERO_LINK_ENABLED 未设时返回 False（默认关）

    注意：不做 importlib.reload（会重建类对象，破坏已有 isinstance 引用）。
    import 无副作用通过「构造实例无 I/O」+ flag 默认关来验证。
    """
    from src.mcp.zero.client import _is_enabled

    # 未设 ZERO_LINK_ENABLED → 默认 false（monkeypatch 保证测试隔离）
    monkeypatch.delenv("ZERO_LINK_ENABLED", raising=False)
    assert _is_enabled() is False, "未设 ZERO_LINK_ENABLED 时 _is_enabled() 应返回 False"

    # 构造 client 实例不触发任何 I/O（无网络、无子进程、无 env 写入）
    client = ZeroLinkClient()
    assert client.exit_stack is None, "ZeroLinkClient() 构造时 exit_stack 应为 None"
    assert client.session is None, "ZeroLinkClient() 构造时 session 应为 None"


# ---------------------------------------------------------------------------
# 场景 6：ZERO_LINK_ENABLED=false → ZeroLinkDisabledError（快速路径，不尝试 spawn）
# ---------------------------------------------------------------------------


async def test_disabled_flag_raises_disabled_error_not_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ZERO_LINK_ENABLED=false 时 __aenter__ 抛 ZeroLinkDisabledError（非 ConnectionError）。

    区别于连接失败：disabled 是配置层拒绝，不尝试建立任何 transport 连接。
    此测试无需真子进程（不标 realenv）。
    """
    monkeypatch.setenv("ZERO_LINK_ENABLED", "false")
    # 即使 COMMAND 不存在也无关——flag 检查在 transport 之前
    monkeypatch.setenv("ZERO_SERVER_COMMAND", "definitely_not_a_real_command_xyz_12345")

    client = ZeroLinkClient()

    with pytest.raises(ZeroLinkDisabledError) as exc_info:
        await client.__aenter__()

    # 确认是 Disabled 而非 Connection 错误
    assert not isinstance(exc_info.value, ZeroLinkConnectionError), (
        "disabled flag 应抛 ZeroLinkDisabledError，不应是 ZeroLinkConnectionError"
    )
    # exit_stack / session 均未动（flag 检查提前返回）
    assert client.exit_stack is None
    assert client.session is None


# ---------------------------------------------------------------------------
# 场景 7：ZERO_LINK_ENABLED 未设 → __aenter__ 抛 ZeroLinkDisabledError（默认关）
# ---------------------------------------------------------------------------


async def test_unset_flag_raises_disabled_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ZERO_LINK_ENABLED 未设时（等同 false）__aenter__ 抛 ZeroLinkDisabledError。

    验证「默认关」不会静默通过或抛不同类型的异常。
    """
    monkeypatch.delenv("ZERO_LINK_ENABLED", raising=False)

    client = ZeroLinkClient()

    with pytest.raises(ZeroLinkDisabledError):
        await client.__aenter__()

    assert client.exit_stack is None
    assert client.session is None
