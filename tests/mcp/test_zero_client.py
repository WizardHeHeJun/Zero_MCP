"""test_zero_client.py — ZeroLinkClient 单元测试（AsyncMock，无真子进程，Task 5）。

覆盖矩阵（monkeypatch.setenv 控 flag，直接注入 client.session = AsyncMock() 绕传输）：
- disabled flag → ZeroLinkDisabledError
- require_session 在 context 外 → ZeroLinkConnectionError
- call_tool isError=True → ZeroLinkCallError（.tool 字段一致）
- call_tool 抛 McpError → 包成 ZeroLinkCallError
- open_session round-trip → 返回 session_id 字符串
- step priors=None → arguments 无 external_priors 键
- step priors 非空 → external_priors 每条是 list（无 tuple）+ ExpressionBundle 正确解析
- step priors 超 M6 (>5) → ValueError 透传（fail-fast）
- close_session → call_tool 以正确参数调一次
- graceful_step flag 关 → None
- graceful_step session_id=None → None
- graceful_step ZeroLinkCallError → None
- graceful_step ValueError (M6) → 透传（不吞）
- __aexit__ 清理 session / exit_stack
- _build_transport_params stdio 分支
- _build_transport_params http 分支
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, ErrorData, TextContent

from src.agents.models.zero_affect import AffectStimulus, ExpressionBundle, ModalityPrior
from src.mcp.zero.client import (
    ZeroLinkCallError,
    ZeroLinkClient,
    ZeroLinkConnectionError,
    ZeroLinkDisabledError,
    ZeroLinkUnknownSessionError,
    _build_http_client,
    _build_transport_params,
    _is_enabled,
    _is_unknown_session_text,
    generate_session_id,
)

# ---------------------------------------------------------------------------
# 辅助构造函数
# ---------------------------------------------------------------------------


def _make_call_result(text: str, *, is_error: bool = False) -> CallToolResult:
    """构造真实 CallToolResult（非 MagicMock，与 client._extract_text 兼容）。"""
    tc = TextContent(type="text", text=text)
    return CallToolResult(content=[tc], isError=is_error)


def _make_expression_json(valence: float = 0.3, arousal: float = 0.5) -> str:
    """构造合法 ExpressionBundle JSON（step 工具返回体）。"""
    head = {
        "facs_au": {"AU12": 0.8, "AU06": 0.6, "intensity": 0.7},
        "text_label": "content",
        "physiology": {"heart_rate_bpm": 80.0, "skin_conductance": 0.5, "pupil_mm": 4.0},
        "prosody": {"speech_rate": 1.0, "pitch": 1.0, "energy": 0.7},
    }
    data = {
        "expression": {
            "valence_arousal": [valence, arousal],
            "spontaneous": head,
            "voluntary": head,
        }
    }
    return json.dumps(data)


def _build_client_with_session(monkeypatch: pytest.MonkeyPatch) -> tuple[ZeroLinkClient, AsyncMock]:
    """启用 flag，构造 ZeroLinkClient 并注入 mock session（绕传输层）。"""
    monkeypatch.setenv("ZERO_LINK_ENABLED", "true")
    mock_session = AsyncMock()
    client = ZeroLinkClient()
    client.session = mock_session
    return client, mock_session


def _set_tool_return(mock_session: AsyncMock, text: str, *, is_error: bool = False) -> None:
    """设置 session.call_tool 返回指定文本内容。"""
    mock_session.call_tool.return_value = _make_call_result(text, is_error=is_error)


# ---------------------------------------------------------------------------
# feature flag
# ---------------------------------------------------------------------------


def test_is_enabled_false_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZERO_LINK_ENABLED", raising=False)
    assert _is_enabled() is False


def test_is_enabled_true_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    for val in ("true", "1", "yes", "TRUE", "YES"):
        monkeypatch.setenv("ZERO_LINK_ENABLED", val)
        assert _is_enabled() is True, f"应为 True，实际 False（env={val!r}）"


# ---------------------------------------------------------------------------
# Task 5.1 disabled flag → ZeroLinkDisabledError
# ---------------------------------------------------------------------------


async def test_disabled_flag_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """ZERO_LINK_ENABLED 未设或为 false 时，__aenter__ 抛 ZeroLinkDisabledError。"""
    monkeypatch.delenv("ZERO_LINK_ENABLED", raising=False)
    client = ZeroLinkClient()
    with pytest.raises(ZeroLinkDisabledError):
        await client.__aenter__()


async def test_disabled_flag_false_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """ZERO_LINK_ENABLED=false 时，__aenter__ 抛 ZeroLinkDisabledError。"""
    monkeypatch.setenv("ZERO_LINK_ENABLED", "false")
    client = ZeroLinkClient()
    with pytest.raises(ZeroLinkDisabledError):
        await client.__aenter__()


# ---------------------------------------------------------------------------
# Task 5.2 require_session 在 context 外 → ZeroLinkConnectionError
# ---------------------------------------------------------------------------


def test_require_session_raises_outside_context() -> None:
    """session=None 时直接调 _require_session() 抛 ZeroLinkConnectionError。"""
    client = ZeroLinkClient()
    assert client.session is None
    with pytest.raises(ZeroLinkConnectionError):
        client._require_session()


# ---------------------------------------------------------------------------
# Task 5.3 call_tool isError=True → ZeroLinkCallError（.tool 字段一致）
# ---------------------------------------------------------------------------


async def test_call_tool_is_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """mock session.call_tool 返回 isError=True 时 _call_tool 抛 ZeroLinkCallError，
    且 .tool 字段 == 工具名。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, "server error msg", is_error=True)

    with pytest.raises(ZeroLinkCallError) as exc_info:
        await client._call_tool("zero.open_session", {})

    assert exc_info.value.tool == "zero.open_session"
    assert "server error msg" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Task 5.4 call_tool 抛 McpError → 包成 ZeroLinkCallError
# ---------------------------------------------------------------------------


async def test_call_tool_mcp_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """mock session.call_tool 抛 McpError 时，_call_tool 将其包成 ZeroLinkCallError。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    mock_session.call_tool.side_effect = McpError(
        ErrorData(code=-32603, message="internal error", data=None)
    )

    with pytest.raises(ZeroLinkCallError) as exc_info:
        await client._call_tool("zero.step", {})

    assert exc_info.value.tool == "zero.step"
    assert "internal error" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Task 5.5 open_session round-trip → 返回 session_id
# ---------------------------------------------------------------------------


async def test_open_session_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """mock 返回 session_id=sid-1 的 JSON，open_session 返回 'sid-1'，
    且 call_tool 以 'zero.open_session' 调。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, json.dumps({"session_id": "sid-1"}))

    result = await client.open_session()

    assert result == "sid-1"
    mock_session.call_tool.assert_called_once_with("zero.open_session", {})


async def test_open_session_with_persona(monkeypatch: pytest.MonkeyPatch) -> None:
    """open_session(persona='default') 把 persona 传给 call_tool arguments。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, json.dumps({"session_id": "sid-2"}))

    await client.open_session(persona="default")

    mock_session.call_tool.assert_called_once_with("zero.open_session", {"persona": "default"})


# ---------------------------------------------------------------------------
# Task 5.6 step priors=None → arguments 无 external_priors 键
# ---------------------------------------------------------------------------


async def test_step_no_priors(monkeypatch: pytest.MonkeyPatch) -> None:
    """priors=None 时 step 的 call_tool arguments 无 'external_priors' 键，
    返回正确的 ExpressionBundle 实例。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, _make_expression_json())

    stimulus = AffectStimulus(valence=0.3, arousal=0.5)
    bundle = await client.step("sid-1", stimulus, priors=None)

    assert isinstance(bundle, ExpressionBundle)

    # 断言 call_tool 收到的 arguments 不含 external_priors
    call_args = mock_session.call_tool.call_args
    tool_name, arguments = call_args[0]
    assert tool_name == "zero.step"
    assert "external_priors" not in arguments
    assert arguments["session_id"] == "sid-1"


# ---------------------------------------------------------------------------
# Task 5.7 step priors 非空 → external_priors 每条是 list（无 tuple）
# ---------------------------------------------------------------------------


async def test_step_with_priors_tuple_to_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """priors 非空时，call_tool arguments['external_priors'] 每条是 list（name str +
    mu list + precision list，无 tuple），ExpressionBundle 正确解析。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, _make_expression_json())

    priors = [
        ModalityPrior(modality="vision", mu=(0.4, 0.3), precision=(0.2, 0.15)),
        ModalityPrior(modality="audio", mu=(-0.1, 0.6), precision=(0.1, 0.25)),
    ]
    stimulus = AffectStimulus(valence=0.3, arousal=0.5)
    bundle = await client.step("sid-1", stimulus, priors=priors)

    assert isinstance(bundle, ExpressionBundle)

    call_args = mock_session.call_tool.call_args
    _, arguments = call_args[0]
    ext_priors = arguments["external_priors"]

    assert isinstance(ext_priors, list)
    assert len(ext_priors) == 2

    for item in ext_priors:
        assert isinstance(item, list), f"外部先验条目应为 list，实际为 {type(item)}"
        name, mu, precision = item
        assert isinstance(name, str)
        assert isinstance(mu, list), f"mu 应为 list，实际为 {type(mu)}"
        assert isinstance(precision, list), f"precision 应为 list，实际为 {type(precision)}"
        # 无 tuple
        assert not isinstance(mu, tuple)
        assert not isinstance(precision, tuple)


# ---------------------------------------------------------------------------
# Task 5.8 step priors 超 M6 (>5) → ValueError 透传
# ---------------------------------------------------------------------------


async def test_step_m6_fail_fast_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """priors 超 M6（>5 条）时，build_external_priors_override 抛 ValueError，
    client.step 不吞，pytest.raises 捕获。"""
    client, mock_session = _build_client_with_session(monkeypatch)

    # 构造 6 条先验（默认 max_streams=5，触发 M6）
    priors = [
        ModalityPrior(modality=f"vision_{i}", mu=(0.1, 0.1), precision=(0.1, 0.1)) for i in range(6)
    ]
    stimulus = AffectStimulus(valence=0.0, arousal=0.0)

    with pytest.raises(ValueError, match="M6"):
        await client.step("sid-1", stimulus, priors=priors)

    # call_tool 不应被调用（fail-fast 在 call_tool 之前）
    mock_session.call_tool.assert_not_called()


# ---------------------------------------------------------------------------
# Task 5.9 close_session → call_tool 正确参数
# ---------------------------------------------------------------------------


async def test_close_session_calls_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """close_session('sid-1') 调 call_tool('zero.close_session', {'session_id':'sid-1'}) 一次。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, json.dumps({"ok": True}))

    await client.close_session("sid-1")

    mock_session.call_tool.assert_called_once_with("zero.close_session", {"session_id": "sid-1"})


# ---------------------------------------------------------------------------
# Task 5.10 graceful_step flag 关 → None
# ---------------------------------------------------------------------------


async def test_graceful_step_disabled_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """ZERO_LINK_ENABLED=false 时 graceful_step 返回 None（不 raise）。"""
    monkeypatch.delenv("ZERO_LINK_ENABLED", raising=False)
    client = ZeroLinkClient()  # session=None，flag 关
    stimulus = AffectStimulus(valence=0.0, arousal=0.0)

    result = await client.graceful_step("sid-1", stimulus)

    assert result is None


# ---------------------------------------------------------------------------
# Task 5.11 graceful_step session_id=None → None
# ---------------------------------------------------------------------------


async def test_graceful_step_none_session_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """session_id=None 时 graceful_step 返回 None。"""
    monkeypatch.setenv("ZERO_LINK_ENABLED", "true")
    client = ZeroLinkClient()
    # 不注入 session，session_id=None 走提前返回路径
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    result = await client.graceful_step(None, stimulus)

    assert result is None


# ---------------------------------------------------------------------------
# Task 5.12 graceful_step ZeroLinkCallError → None
# ---------------------------------------------------------------------------


async def test_graceful_step_call_error_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """step 抛 ZeroLinkCallError 时 graceful_step 降级返回 None。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    # 令 step 抛 ZeroLinkCallError
    _set_tool_return(mock_session, "boom", is_error=True)

    stimulus = AffectStimulus(valence=0.1, arousal=0.2)
    result = await client.graceful_step("sid-1", stimulus)

    assert result is None


# ---------------------------------------------------------------------------
# Task 5.13 graceful_step ValueError (M6) → 透传（不吞）
# ---------------------------------------------------------------------------


async def test_graceful_step_value_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """priors 超 M6 时 graceful_step 透传 ValueError（不 graceful 吞掉调用方参数错误）。"""
    client, mock_session = _build_client_with_session(monkeypatch)

    priors = [
        ModalityPrior(modality=f"vision_{i}", mu=(0.1, 0.1), precision=(0.1, 0.1)) for i in range(6)
    ]
    stimulus = AffectStimulus(valence=0.0, arousal=0.0)

    with pytest.raises(ValueError, match="M6"):
        await client.graceful_step("sid-1", stimulus, priors=priors)


# ---------------------------------------------------------------------------
# Task 5.14 __aexit__ 清理 session / exit_stack
# ---------------------------------------------------------------------------


async def test_aexit_clears_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """__aexit__ 后 session=None 且 exit_stack.__aexit__ 被调。"""
    monkeypatch.setenv("ZERO_LINK_ENABLED", "true")
    client = ZeroLinkClient()

    # 注入 mock exit_stack 和 session
    mock_stack = AsyncMock()
    mock_session = AsyncMock()
    client.exit_stack = mock_stack
    client.session = mock_session

    await client.__aexit__(None, None, None)

    assert client.session is None
    assert client.exit_stack is None
    mock_stack.__aexit__.assert_called_once_with(None, None, None)


# ---------------------------------------------------------------------------
# Task 5.15 _build_transport_params 各分支（纯函数）
# ---------------------------------------------------------------------------


def test_build_transport_params_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    """stdio 分支：返回 ('stdio', StdioServerParameters)，字段来自 env。"""
    from mcp.client.stdio import StdioServerParameters

    monkeypatch.setenv("ZERO_LINK_TRANSPORT", "stdio")
    monkeypatch.setenv("ZERO_SERVER_COMMAND", "python")
    monkeypatch.setenv("ZERO_SERVER_ARGS", '["-m", "zero.server"]')
    monkeypatch.setenv("ZERO_SERVER_CWD", r"C:\fake_zero")

    kind, params = _build_transport_params()

    assert kind == "stdio"
    assert isinstance(params, StdioServerParameters)
    assert params.command == "python"
    assert params.args == ["-m", "zero.server"]
    assert params.cwd == r"C:\fake_zero"


def test_build_transport_params_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """http 分支：返回 ('http', (endpoint, token))。"""
    monkeypatch.setenv("ZERO_LINK_TRANSPORT", "http")
    monkeypatch.setenv("ZERO_HTTP_ENDPOINT", "http://localhost:8080/mcp")
    monkeypatch.setenv("ZERO_HTTP_TOKEN", "test-token-abc")

    kind, params = _build_transport_params()

    assert kind == "http"
    endpoint, token = params
    assert endpoint == "http://localhost:8080/mcp"
    assert token == "test-token-abc"


def test_build_transport_params_http_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """http 分支无 token 时 token 为空字符串。"""
    monkeypatch.setenv("ZERO_LINK_TRANSPORT", "http")
    monkeypatch.setenv("ZERO_HTTP_ENDPOINT", "http://localhost:9090/mcp")
    monkeypatch.delenv("ZERO_HTTP_TOKEN", raising=False)

    kind, params = _build_transport_params()

    assert kind == "http"
    endpoint, token = params
    assert endpoint == "http://localhost:9090/mcp"
    assert token == ""


# ---------------------------------------------------------------------------
# T5 HTTP 鉴权：_build_http_client 构造 Bearer 头（客户端侧已就绪，锁定行为）
# ---------------------------------------------------------------------------


async def test_build_http_client_sets_bearer_header() -> None:
    """有 token → httpx.AsyncClient 预置 `Authorization: Bearer <token>`（RFC 6750）。"""
    client = _build_http_client("test-token-abc")
    assert client is not None
    try:
        assert client.headers["Authorization"] == "Bearer test-token-abc"
    finally:
        await client.aclose()


def test_build_http_client_no_token_returns_none() -> None:
    """无 token（空串）→ 返回 None（不鉴权，默认本地场景零回归）。"""
    assert _build_http_client("") is None


# ---------------------------------------------------------------------------
# Task 5.16 响应解析防御（code-review W2）：畸形响应封装为 ZeroLinkCallError
# ---------------------------------------------------------------------------


async def test_open_session_non_json_raises_call_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """open_session 收到非 JSON 响应 → ZeroLinkCallError（不穿透 JSONDecodeError）。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, "not json {{{")

    with pytest.raises(ZeroLinkCallError) as exc_info:
        await client.open_session()
    assert exc_info.value.tool == "zero.open_session"


async def test_open_session_missing_key_raises_call_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """open_session 响应缺 session_id 键 → ZeroLinkCallError（不穿透 KeyError）。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, json.dumps({"wrong_key": "x"}))

    with pytest.raises(ZeroLinkCallError):
        await client.open_session()


async def test_step_non_json_raises_call_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """step 收到非 JSON 响应 → ZeroLinkCallError。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, "garbage}{")
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    with pytest.raises(ZeroLinkCallError) as exc_info:
        await client.step("sid-1", stimulus)
    assert exc_info.value.tool == "zero.step"


async def test_step_invalid_expression_raises_call_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """step 响应结构不合 ExpressionBundle 契约（ValidationError）→ ZeroLinkCallError。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    # 合法 JSON 但 expression 缺必填字段 → from_step_output 抛 ValidationError
    _set_tool_return(mock_session, json.dumps({"expression": {"valence_arousal": [0.1, 0.2]}}))
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    with pytest.raises(ZeroLinkCallError):
        await client.step("sid-1", stimulus)


async def test_graceful_step_malformed_response_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """graceful_step 兜住畸形响应（封装后的 ZeroLinkCallError）降级返回 None（零回归契约完整）。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, "not json {{{")
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    result = await client.graceful_step("sid-1", stimulus)

    assert result is None


# ---------------------------------------------------------------------------
# Task 5.17 stim 序列化（code-review W3）：exclude_none 略去 None 可选字段
# ---------------------------------------------------------------------------


async def test_step_stim_excludes_none_coping(monkeypatch: pytest.MonkeyPatch) -> None:
    """coping_potential=None 时 stim 载荷不含该键（exclude_none，最小合法载荷）。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, _make_expression_json())
    stimulus = AffectStimulus(valence=0.3, arousal=0.5)  # coping_potential 默认 None

    await client.step("sid-1", stimulus)

    _tool, arguments = mock_session.call_tool.call_args.args
    assert "coping_potential" not in arguments["stim"]
    assert arguments["stim"] == {"valence": 0.3, "arousal": 0.5}


async def test_step_stim_includes_coping_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """coping_potential 非 None 时保留在 stim 载荷（含 0.0 边界不被 exclude_none 误删）。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, _make_expression_json())
    stimulus = AffectStimulus(valence=0.3, arousal=0.5, coping_potential=0.0)

    await client.step("sid-1", stimulus)

    _tool, arguments = mock_session.call_tool.call_args.args
    assert arguments["stim"]["coping_potential"] == 0.0


# ---------------------------------------------------------------------------
# Task 5.18 unknown-session 机读标记（zero-link T6·②）
#
# Zero step 命中未知/过期 session_id → ToolError 文本以机读前缀 "unknown-session:" 打头
# （Zero server `_UNKNOWN_SESSION_MARKER`）。本仓据此前缀抛更精确的 ZeroLinkUnknownSessionError
# 子类，供上层区分「可 resume 续会话」态。判别性重点：**只认前缀**，不误判 Zero 的其它含
# "session_id" 中文错误（如 open_session 的 config 校验）→ 证明是机读标记而非脆弱文本匹配。
# ---------------------------------------------------------------------------

# Zero server 真实抛出的 unknown-session 文本（逐字对齐 server.py step 分支，session_id='sid-1'）。
# ⚠ 冒号后中文仅供可读性——判定只看机读前缀 `unknown-session:`，Zero 若改中文措辞不影响本仓判定。
_ZERO_UNKNOWN_SESSION_TEXT = (
    "unknown-session: 未知 session_id='sid-1'；请先调 zero.open_session（可用同 id resume 续会话）"
)


def test_is_unknown_session_text_matches_zero_marker() -> None:
    """机读前缀命中：Zero 真实 unknown-session 文本 + 前导空白包裹变体 → True。"""
    assert _is_unknown_session_text(_ZERO_UNKNOWN_SESSION_TEXT) is True
    # 容忍未来包裹换行/缩进（lstrip 后仍以前缀打头）
    assert _is_unknown_session_text("\n  unknown-session: 未知 session_id='x'") is True


def test_is_unknown_session_text_rejects_non_marker() -> None:
    """判别性：非 unknown-session 消息一律 False，避免误判。

    覆盖三类易误判：(a) 通用错误；(b) Zero 其它含「session_id」的中文错误（open_session
    的 config 校验，不该被当成 unknown-session）；(c) marker 出现在**中间**而非前缀。
    """
    assert _is_unknown_session_text("boom") is False
    assert _is_unknown_session_text("server 返回 isError=True") is False
    # Zero open_session 的 config 校验错误——含 "session_id" 中文，但非 unknown-session
    assert _is_unknown_session_text("session_id 须为非空字符串，实际为 ''") is False
    # marker 只出现在中间（非前缀）→ 不算，防子串误判
    assert _is_unknown_session_text("error: unknown-session happened downstream") is False


async def test_step_unknown_session_raises_subclass(monkeypatch: pytest.MonkeyPatch) -> None:
    """step 命中 unknown-session isError → 抛 ZeroLinkUnknownSessionError，
    且是 ZeroLinkCallError 子类、`.tool == 'zero.step'`（向后兼容 + 精确化）。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, _ZERO_UNKNOWN_SESSION_TEXT, is_error=True)
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    with pytest.raises(ZeroLinkUnknownSessionError) as exc_info:
        await client.step("sid-1", stimulus)

    assert isinstance(exc_info.value, ZeroLinkCallError)  # 子类关系（graceful_step 仍兜住）
    assert exc_info.value.tool == "zero.step"


async def test_step_generic_error_not_unknown_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """判别性：普通 isError（非 marker）→ 抛基类，**不是** unknown-session 子类。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, "boom", is_error=True)
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    with pytest.raises(ZeroLinkCallError) as exc_info:
        await client.step("sid-1", stimulus)

    assert not isinstance(exc_info.value, ZeroLinkUnknownSessionError)


async def test_step_config_error_not_unknown_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """判别性：含「session_id」中文的 Zero 其它错误（如 config 校验）不被误判为 unknown-session。

    证明区分走机读前缀而非脆弱中文匹配——回执明言「靠字符串匹配脆弱」正是要规避的。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, "session_id 须为非空字符串，实际为 ''", is_error=True)
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    with pytest.raises(ZeroLinkCallError) as exc_info:
        await client.step("sid-1", stimulus)

    assert not isinstance(exc_info.value, ZeroLinkUnknownSessionError)


async def test_graceful_step_resume_retry_unknown_again_no_recursion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不递归守卫（code-review W4）：resume 后重试 step **再次** unknown-session → 降级 None。

    序列：step(unknown-session) → open_session(ok) → step(unknown-session 再现)。重试的 step 抛的
    ZeroLinkUnknownSessionError 是 ZeroLinkCallError 子类 → 被内层 except (ZeroLinkCallError,…) 兜住
    → None，**不再触发第二次 resume**（至多重试一次）。断言 call_count==3 锁定不递归。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    mock_session.call_tool.side_effect = [
        _make_call_result(_ZERO_UNKNOWN_SESSION_TEXT, is_error=True),
        _make_call_result(json.dumps({"session_id": "sid-1"})),
        _make_call_result(_ZERO_UNKNOWN_SESSION_TEXT, is_error=True),
    ]
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    result = await client.graceful_step("sid-1", stimulus)

    assert result is None
    # step → open_session → step，恰好三次；不因第二次 unknown-session 再 resume（不递归）
    assert mock_session.call_tool.call_count == 3
    assert [c.args[0] for c in mock_session.call_tool.call_args_list] == [
        "zero.step",
        "zero.open_session",
        "zero.step",
    ]


async def test_unknown_session_only_for_step_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """判别性（code-review W1）：unknown-session 子类语义**只对 zero.step 成立**。

    open_session 即便返回带机读前缀的 isError（未来 Zero 内部路由变化的假想场景），也只抛基类
    ZeroLinkCallError，不误升级为 ZeroLinkUnknownSessionError——保子类语义与触发路径严格对齐。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, _ZERO_UNKNOWN_SESSION_TEXT, is_error=True)

    with pytest.raises(ZeroLinkCallError) as exc_info:
        await client._call_tool("zero.open_session", {})

    assert not isinstance(exc_info.value, ZeroLinkUnknownSessionError)
    assert exc_info.value.tool == "zero.open_session"


# ---------------------------------------------------------------------------
# Task 5.19 T6 resume-by-id（zero-link T6·④）
#
# open_session 加可选 session_id（resume 入口）；graceful_step 命中 unknown-session → 用同 id
# 重开(+再供 config)后重试一次；generate_session_id 产不可猜 id（运行态访问凭据）。
# ---------------------------------------------------------------------------


async def test_open_session_passes_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """open_session(session_id=…) → 载荷含 session_id 键（resume 入口）+ config 一并透传。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, json.dumps({"session_id": "resumed-sid"}))

    sid = await client.open_session(session_id="resumed-sid", config={"x": 1})

    assert sid == "resumed-sid"
    _tool, arguments = mock_session.call_tool.call_args.args
    assert arguments["session_id"] == "resumed-sid"
    assert arguments["config"] == {"x": 1}


async def test_open_session_no_session_id_omits_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """不传 session_id → 载荷无该键（Zero 新铸 uuid4，零回归）。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, json.dumps({"session_id": "fresh"}))

    await client.open_session()

    _tool, arguments = mock_session.call_tool.call_args.args
    assert "session_id" not in arguments


async def test_graceful_step_resume_retry_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """unknown-session → 用同 id 重开(+再供 config)后重试 step 成功 → 返回 ExpressionBundle。

    调用序列：step(unknown-session) → open_session(ok) → step(ok)。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    mock_session.call_tool.side_effect = [
        _make_call_result(_ZERO_UNKNOWN_SESSION_TEXT, is_error=True),
        _make_call_result(json.dumps({"session_id": "sid-1"})),
        _make_call_result(_make_expression_json()),
    ]
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    result = await client.graceful_step("sid-1", stimulus, resume_config={"a": 1})

    assert isinstance(result, ExpressionBundle)
    calls = mock_session.call_tool.call_args_list
    assert [c.args[0] for c in calls] == ["zero.step", "zero.open_session", "zero.step"]
    # resume 用同 id 重开 + 再供 config
    assert calls[1].args[1]["session_id"] == "sid-1"
    assert calls[1].args[1]["config"] == {"a": 1}


async def test_graceful_step_resume_retry_fails_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """unknown-session → 重开成功但重试 step 再失败 → 降级 None（只重试一次，不递归）。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    mock_session.call_tool.side_effect = [
        _make_call_result(_ZERO_UNKNOWN_SESSION_TEXT, is_error=True),
        _make_call_result(json.dumps({"session_id": "sid-1"})),
        _make_call_result("boom", is_error=True),
    ]
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    result = await client.graceful_step("sid-1", stimulus)

    assert result is None
    # step → open_session → step，恰好三次（不再递归重试）
    assert mock_session.call_tool.call_count == 3


async def test_graceful_step_generic_error_no_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    """判别性：普通 isError（非 unknown-session）→ 不触发 resume，直接降级 None（不调 open）。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, "boom", is_error=True)
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    result = await client.graceful_step("sid-1", stimulus)

    assert result is None
    assert mock_session.call_tool.call_count == 1  # 只调 step，无 resume 的 open_session
    assert mock_session.call_tool.call_args.args[0] == "zero.step"


def test_generate_session_id_unguessable_and_unique() -> None:
    """generate_session_id 产 128-bit hex（32 字符）不可猜 id，两次调用不同（唯一）。"""
    a = generate_session_id()
    b = generate_session_id()

    assert a != b  # 唯一（CSPRNG）
    assert len(a) == 32  # 16 字节 → 32 hex 字符
    assert all(ch in "0123456789abcdef" for ch in a)  # 纯十六进制、无歧义


# ---------------------------------------------------------------------------
# Task 5.20 T5 连接层 CancelledError → ZeroLinkConnectionError（HTTP 401 传输层拒绝）
#
# streamable-http 401 时传输内部 anyio task group 取消 → 抛 CancelledError（BaseException，
# 非 Exception，__aenter__ 的 except Exception 接不住）。__aenter__ 用 Task.cancelling() 区分
# 外部取消（重抛）vs 传输内部取消（转 ZeroLinkConnectionError·连接层，符合回执 T5）。
# ---------------------------------------------------------------------------


async def test_aenter_transport_cancelled_becomes_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """传输建立期抛 CancelledError（非外部取消）→ 转 ZeroLinkConnectionError（T5 401）。"""
    monkeypatch.setenv("ZERO_LINK_ENABLED", "true")
    monkeypatch.setenv("ZERO_LINK_TRANSPORT", "stdio")

    class _CancelOnEnter:
        """模拟传输 CM：__aenter__ 抛 CancelledError（如 streamable-http 401 内部取消）。"""

        async def __aenter__(self) -> object:
            raise asyncio.CancelledError("模拟传输内部因连接被拒取消（401）")

        async def __aexit__(self, *exc: object) -> bool:
            return False

    # patch stdio_client 返回上述 CM（本任务未被外部 cancel，cancelling()==0 → 应转连接错）
    monkeypatch.setattr("src.mcp.zero.client.stdio_client", lambda *a, **k: _CancelOnEnter())

    with pytest.raises(ZeroLinkConnectionError) as exc_info:
        await ZeroLinkClient().__aenter__()

    # W2：原始 CancelledError 保留为 __cause__（诊断非鉴权类内部取消根因）
    assert isinstance(exc_info.value.__cause__, asyncio.CancelledError)
