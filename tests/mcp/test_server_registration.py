"""test_server_registration.py — MCP server 工具注册与 feature flag 零回归测试。

覆盖：
- 10 个工具全部注册
- close_window destructiveHint=True，其余写操作 destructiveHint=False
- readOnlyHint 标注正确
- SCREEN_CAPABILITY_ENABLED=false（默认）时各工具调用 raise ToolError（零回归）
- SCREEN_CAPABILITY_ENABLED=true 时 _require_enabled 不抛
"""

from __future__ import annotations

import pytest
from mcp.server.fastmcp.exceptions import ToolError

import src.mcp.desktop_mcp_server as server_mod
from src.mcp.desktop_mcp_server import (
    _is_enabled,
    _require_enabled,
    mcp,
)

# ── 工具注册 ──────────────────────────────────────────────────────────────────

EXPECTED_TOOLS = {
    "screen_snapshot",
    "get_uia_tree",
    "ocr_region",
    "click_element",
    "type_text",
    "send_key",
    "window_list",
    "focus_window",
    "close_window",
    "get_capability_flags",
}

READ_ONLY_TOOLS = {
    "screen_snapshot",
    "get_uia_tree",
    "ocr_region",
    "window_list",
    "get_capability_flags",
}

DESTRUCTIVE_TOOLS = {"close_window"}


def _get_tool(name: str) -> object:
    return mcp._tool_manager._tools[name]


def test_all_10_tools_registered() -> None:
    """10 个工具全部已注册到 FastMCP。"""
    registered = set(mcp._tool_manager._tools.keys())
    assert registered == EXPECTED_TOOLS


def test_read_only_hint_correct() -> None:
    """readOnlyHint=True 的工具集合正确。"""
    for name, tool in mcp._tool_manager._tools.items():
        ann = tool.annotations
        if name in READ_ONLY_TOOLS:
            assert ann is not None and ann.readOnlyHint is True, (
                f"{name} 应为 readOnlyHint=True，实际 {ann}"
            )
        else:
            assert ann is None or ann.readOnlyHint is not True, (
                f"{name} 不应为 readOnlyHint=True，实际 {ann}"
            )


def test_close_window_destructive_hint() -> None:
    """close_window 必须标注 destructiveHint=True。"""
    tool = _get_tool("close_window")
    ann = tool.annotations  # type: ignore[union-attr]
    assert ann is not None
    assert ann.destructiveHint is True


def test_non_close_window_not_destructive() -> None:
    """除 close_window 外，所有工具 destructiveHint 不为 True。"""
    for name, tool in mcp._tool_manager._tools.items():
        if name == "close_window":
            continue
        ann = tool.annotations
        assert ann is None or ann.destructiveHint is not True, f"{name} 不应为 destructiveHint=True"


# ── feature flag ──────────────────────────────────────────────────────────────


def test_is_enabled_false_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """未设 SCREEN_CAPABILITY_ENABLED 时 _is_enabled() 返回 False。"""
    monkeypatch.delenv("SCREEN_CAPABILITY_ENABLED", raising=False)
    assert _is_enabled() is False


def test_is_enabled_true_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCREEN_CAPABILITY_ENABLED=true/1/yes（大小写不敏感）时返回 True。"""
    for val in ("true", "True", "TRUE", "1", "yes", "YES"):
        monkeypatch.setenv("SCREEN_CAPABILITY_ENABLED", val)
        assert _is_enabled() is True, f"期望 True，值={val!r}"


def test_require_enabled_raises_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """flag=false 时 _require_enabled() 抛 ToolError。"""
    monkeypatch.delenv("SCREEN_CAPABILITY_ENABLED", raising=False)
    with pytest.raises(ToolError):
        _require_enabled()


def test_require_enabled_ok_when_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """flag=true 时 _require_enabled() 不抛。"""
    monkeypatch.setenv("SCREEN_CAPABILITY_ENABLED", "true")
    _require_enabled()  # should not raise


# ── 工具体运行时 flag 零回归（mock 不走真实感知） ─────────────────────────────


@pytest.mark.asyncio
async def test_screen_snapshot_raises_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """flag=false 时 screen_snapshot 工具调用 raise ToolError。"""
    monkeypatch.delenv("SCREEN_CAPABILITY_ENABLED", raising=False)
    with pytest.raises(ToolError):
        await server_mod.screen_snapshot()


@pytest.mark.asyncio
async def test_get_uia_tree_raises_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SCREEN_CAPABILITY_ENABLED", raising=False)
    with pytest.raises(ToolError):
        await server_mod.get_uia_tree()


@pytest.mark.asyncio
async def test_ocr_region_raises_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SCREEN_CAPABILITY_ENABLED", raising=False)
    with pytest.raises(ToolError):
        await server_mod.ocr_region(
            bbox={"x": 0, "y": 0, "width": 100, "height": 100}, screenshot_path="/fake.png"
        )


@pytest.mark.asyncio
async def test_click_element_raises_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SCREEN_CAPABILITY_ENABLED", raising=False)
    with pytest.raises(ToolError):
        await server_mod.click_element()


@pytest.mark.asyncio
async def test_type_text_raises_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SCREEN_CAPABILITY_ENABLED", raising=False)
    with pytest.raises(ToolError):
        await server_mod.type_text(text="hello")


@pytest.mark.asyncio
async def test_send_key_raises_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SCREEN_CAPABILITY_ENABLED", raising=False)
    with pytest.raises(ToolError):
        await server_mod.send_key(key_combo="ctrl+c")


@pytest.mark.asyncio
async def test_window_list_raises_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SCREEN_CAPABILITY_ENABLED", raising=False)
    with pytest.raises(ToolError):
        await server_mod.window_list()


@pytest.mark.asyncio
async def test_focus_window_raises_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SCREEN_CAPABILITY_ENABLED", raising=False)
    with pytest.raises(ToolError):
        await server_mod.focus_window(window_handle=12345)


@pytest.mark.asyncio
async def test_close_window_raises_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SCREEN_CAPABILITY_ENABLED", raising=False)
    with pytest.raises(ToolError):
        await server_mod.close_window(window_handle=12345)


@pytest.mark.asyncio
async def test_get_capability_flags_raises_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SCREEN_CAPABILITY_ENABLED", raising=False)
    with pytest.raises(ToolError):
        await server_mod.get_capability_flags()


# ── flag=true + mock 感知层：验证工具体能正常转发 ────────────────────────────


@pytest.mark.asyncio
async def test_get_capability_flags_enabled_returns_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """flag=true 时 get_capability_flags 返回合法 JSON 字符串。"""
    import json

    from src.mcp.desktop.capability_probe import CapabilityFlags

    monkeypatch.setenv("SCREEN_CAPABILITY_ENABLED", "true")
    fake_flags = CapabilityFlags(
        ocr=True,
        omniparser=False,
        cuda_accel=False,
        dml_accel=False,
        mss_available=True,
        effective_device="cpu",
    )
    monkeypatch.setattr(server_mod, "_CAPABILITY_FLAGS", fake_flags)

    result = await server_mod.get_capability_flags()
    parsed = json.loads(result)
    assert parsed["ocr"] is True
    assert parsed["omniparser"] is False
    assert parsed["effective_device"] == "cpu"


# ── feat/desktop-hardening：新工具参数（K7 expected_root_hwnd / K8 pin_topmost）──


def test_click_element_signature_has_expected_root_hwnd() -> None:
    """click_element 工具面暴露 expected_root_hwnd，默认 None（不核验零回归）。"""
    import inspect

    param = inspect.signature(server_mod.click_element).parameters["expected_root_hwnd"]
    assert param.default is None


def test_focus_window_signature_has_pin_topmost() -> None:
    """focus_window 工具面暴露 pin_topmost，默认 False。"""
    import inspect

    param = inspect.signature(server_mod.focus_window).parameters["pin_topmost"]
    assert param.default is False


@pytest.mark.asyncio
async def test_click_element_forwards_expected_root_hwnd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """server 工具体把 expected_root_hwnd 转发到 control.do_click_element（K7 透传）。"""
    from src.agents.models.screen_snapshot import ActionResult

    monkeypatch.setenv("SCREEN_CAPABILITY_ENABLED", "true")
    seen_kwargs: dict[str, object] = {}

    async def fake_do_click(**kwargs: object) -> ActionResult:
        seen_kwargs.update(kwargs)
        return ActionResult(action_id="a", success=True, error_message=None, ui_changed=False)

    monkeypatch.setattr("src.mcp.desktop.tools.control.do_click_element", fake_do_click)

    await server_mod.click_element(
        coordinates=(10, 20), method="coordinate", expected_root_hwnd=0xAAA
    )
    assert seen_kwargs["expected_root_hwnd"] == 0xAAA


@pytest.mark.asyncio
async def test_focus_window_forwards_pin_topmost(monkeypatch: pytest.MonkeyPatch) -> None:
    """server 工具体把 pin_topmost 转发到 control.do_focus_window（K8 透传）。"""
    from src.agents.models.screen_snapshot import ActionResult

    monkeypatch.setenv("SCREEN_CAPABILITY_ENABLED", "true")
    seen_kwargs: dict[str, object] = {}

    async def fake_do_focus(**kwargs: object) -> ActionResult:
        seen_kwargs.update(kwargs)
        return ActionResult(action_id="a", success=True, error_message=None, ui_changed=True)

    monkeypatch.setattr("src.mcp.desktop.tools.control.do_focus_window", fake_do_focus)

    await server_mod.focus_window(window_handle=0x5678, pin_topmost=True)
    assert seen_kwargs["pin_topmost"] is True
