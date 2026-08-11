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
- 机读错误码：令牌 `[zero:<code>]` 位置无关提取 + 六码归类 + 旧裸前缀过渡兼容
  （⚠ 本组夹具一律经 `_wire()` 加壳，见 Task 5.18 段头「夹具必须加壳」）
- graceful_step 不可降级族（config-incompatible / caller-fault / deploy-env）→ 上抛不吞
- __aexit__ 清理 session / exit_stack
- _build_transport_params stdio 分支
- _build_transport_params http 分支
"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock

import anyio
import pytest
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, ErrorData, TextContent

from src.agents.models.zero_affect import AffectStimulus, ExpressionBundle, ModalityPrior
from src.mcp.zero.client import (
    LOG_MARKER_INTERRUPTED_ON_OPEN,
    LOG_MARKER_INTERRUPTED_REFUSED,
    LOG_MARKER_INTERRUPTED_REFUSED_PURGING,
    LOG_MARKER_PROBE_FAILED_ON_OPEN,
    LOG_MARKER_PROBE_FAILED_REFUSED,
    LOG_MARKER_PROBE_MALFORMED,
    LOG_MARKER_PROBE_NOT_PROBED_UNDECIDABLE,
    LOG_MARKER_PROBE_STATE_MISMATCH,
    LOG_MARKER_PROBE_UNDECIDABLE,
    LOG_MARKER_PROBE_UNRECOGNIZED_ON_OPEN,
    LOG_MARKER_PROBE_UNRECOGNIZED_REFUSED,
    ZERO_ERROR_CODE_CONFIG_INVALID,
    ZERO_ERROR_CODE_DEPLOY_ENV_INVALID,
    ZERO_ERROR_CODE_PAYLOAD_INVALID,
    ZERO_ERROR_CODE_UNKNOWN_SESSION,
    ZERO_ERROR_CODES,
    ZeroInterruptProbe,
    ZeroLinkCallerFaultError,
    ZeroLinkCallError,
    ZeroLinkClient,
    ZeroLinkConfigIncompatibleError,
    ZeroLinkConnectionError,
    ZeroLinkDeployEnvError,
    ZeroLinkDisabledError,
    ZeroLinkLockTimeoutError,
    ZeroLinkMotionDisabledError,
    ZeroLinkNonDegradableError,
    ZeroLinkStepTimeoutError,
    ZeroLinkUnknownSessionError,
    ZeroOpenSessionInfo,
    _build_http_client,
    _build_transport_params,
    _is_enabled,
    classify_zero_error,
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
# Task 5.18 Zero 机读错误码（zero-link T6·② → 2026-07-29 位置无关令牌换代）
#
# 🛑 **夹具必须加壳**：FastMCP 在工具层把 ToolError 统一包成
#     "Error executing tool <tool_name>: <原文>"
# （`mcp/server/fastmcp/tools/base.py::Tool.run` 的 `except Exception` 分支；ToolError 自己
# 也继承 Exception，照样被再包一层）。**wire 上的文本永远带这层外壳**。
# 旧夹具直接喂未加壳原文，于是「`startswith(marker)` 判定」在单测里绿、对真 server 恒 False
# ——T6·④ 的 resume 重试通路长期是**生产死码**。本仓 stdio 直连 D:\Zero 实测：
#     "Error executing tool zero.step: [zero:unknown-session] 未知 session_id='bogus-sid-xyz'；…"
#     .lstrip().startswith("unknown-session") -> False
# ⇒ 本组用例一律经 `_wire()` 构造夹具；新增夹具**不得**绕过它，否则又变成假绿。
# 判别性重点：位置无关的令牌提取（`re.search`），且不误判其它含 "session_id" 中文的错误。
# ---------------------------------------------------------------------------


def _wire(tool: str, inner: str) -> str:
    """把 Zero 侧 ToolError 原文包成**真 wire 形态**（FastMCP 工具层加壳后的样子）。

    外壳文案逐字对齐 SDK 源码 `f"Error executing tool {self.name}: {e}"`，
    并与 stdio 直连 D:\\Zero 的实测输出一致。
    """
    return f"Error executing tool {tool}: {inner}"


# Zero server 真实抛出的 unknown-session 原文（`_tool_error(ZERO_ERROR_CODE_UNKNOWN_SESSION, …)`）。
# ⚠ 令牌后中文仅供人读——判定只看 `[zero:unknown-session]`，Zero 改中文措辞不影响本仓判定。
_ZERO_UNKNOWN_SESSION_INNER = (
    "[zero:unknown-session] 未知 session_id='sid-1'；"
    "请先调 zero.open_session（可用同 id resume 续会话）"
)
_ZERO_UNKNOWN_SESSION_TEXT = _wire("zero.step", _ZERO_UNKNOWN_SESSION_INNER)

# 老部署（Zero 令牌换代**之前**）的裸前缀原文——过渡兼容路径的夹具。
_LEGACY_UNKNOWN_SESSION_INNER = (
    "unknown-session: 未知 session_id='sid-1'；请先调 zero.open_session（可用同 id resume 续会话）"
)
_LEGACY_UNKNOWN_SESSION_TEXT = _wire("zero.step", _LEGACY_UNKNOWN_SESSION_INNER)

# config-incompatible：Zero step 的内核执行失败分支（活跃会话 config 不可变 → 须以新配置重开）。
_ZERO_CONFIG_INCOMPATIBLE_TEXT = _wire(
    "zero.step",
    "[zero:config-incompatible] 内核执行失败（**非** external_priors 传参问题，改传参无效）："
    "boom；多为会话级配置组合不兼容，须以新配置重开会话",
)


def test_wire_fixture_defeats_old_startswith_judgement() -> None:
    """**死码复现守卫**：真 wire 文本上，旧判据（裸前缀 startswith）恒 False，新判据 True。

    这是本轮修复的存在理由，也是防回退的锚：谁把判定改回 `startswith`／把夹具改回未加壳，
    本条即红。三个断言缺一不可——
    ① 加壳确实发生（否则夹具没判别力，等于回到旧假绿）；
    ② 旧判据在该文本上确实 False（死码复现）；
    ③ 新判据（位置无关令牌）确实提取到码。
    """
    assert _ZERO_UNKNOWN_SESSION_TEXT.startswith("Error executing tool zero.step: ")  # ①
    assert _ZERO_UNKNOWN_SESSION_TEXT.lstrip().startswith("unknown-session") is False  # ②
    assert classify_zero_error(_ZERO_UNKNOWN_SESSION_TEXT) == ZERO_ERROR_CODE_UNKNOWN_SESSION  # ③
    # 旧格式同理：裸前缀经加壳后也不在位置 0（说明这不是「换个码值」能修的，是格式契约要换）
    assert _LEGACY_UNKNOWN_SESSION_TEXT.lstrip().startswith("unknown-session") is False


def test_classify_zero_error_extracts_token_anywhere() -> None:
    """令牌提取**位置无关**：开头/中部/末尾都能取到，且**全表**逐码覆盖。

    覆盖面遍历 `ZERO_ERROR_CODES` 而非写死码数——写死会在每次跟随对方加码时
    变成需要手改的噪音，且改漏了也只是数字不符、不指向真问题。
    """
    assert classify_zero_error("[zero:payload-invalid] 开头") == ZERO_ERROR_CODE_PAYLOAD_INVALID
    # ⑤ 令牌出现在**中间**（真 wire 形态即如此）
    assert classify_zero_error(_wire("zero.open_session", "[zero:config-invalid] x")) == (
        ZERO_ERROR_CODE_CONFIG_INVALID
    )
    # 令牌在末尾（假想的其它包装方式，同样要认）
    assert classify_zero_error("something failed [zero:deploy-env-invalid]") == (
        ZERO_ERROR_CODE_DEPLOY_ENV_INVALID
    )
    # 全表逐个可提取（不是只有 unknown-session 一条通路）
    for code in ZERO_ERROR_CODES:
        assert classify_zero_error(_wire("zero.step", f"[zero:{code}] 文案")) == code


def test_classify_zero_error_legacy_bare_prefix_still_recognized() -> None:
    """② 过渡兼容：老部署的**裸前缀**格式（无令牌）仍被识别为 unknown-session。

    覆盖两态：加壳后的（真 wire）与未加壳的（位置 0，历史夹具口径）。
    ⏳ 撤除条件见 client `_LEGACY_UNKNOWN_SESSION_RE` 注释——本条是该兼容层的**唯一**理由，
    撤兼容时连同本条一起删。
    """
    assert classify_zero_error(_LEGACY_UNKNOWN_SESSION_TEXT) == ZERO_ERROR_CODE_UNKNOWN_SESSION
    assert classify_zero_error(_LEGACY_UNKNOWN_SESSION_INNER) == ZERO_ERROR_CODE_UNKNOWN_SESSION


def test_classify_zero_error_rejects_non_marker() -> None:
    """④ 判别性：无令牌、无旧前缀的普通错误一律 None（不误判、不硬套码）。

    覆盖四类易误判：(a) 通用错误；(b) Zero 其它含「session_id」中文错误（open_session 的
    payload 校验的**人读部分**）；(c) 含 "unknown-session" 但无冒号的偶然子串；
    (d) 形似但不合语法的令牌（大写 / 前导数字 / 缺方括号）。
    """
    assert classify_zero_error("boom") is None
    assert classify_zero_error("server 返回 isError=True") is None
    assert classify_zero_error("session_id 须为非空字符串，实际为 ''") is None
    assert classify_zero_error("error: unknown-session happened downstream") is None
    assert classify_zero_error("[zero:UNKNOWN-SESSION] 大写不合 kebab-case") is None
    assert classify_zero_error("[zero:1bad] 首字符须字母") is None
    assert classify_zero_error("zero:payload-invalid 缺方括号") is None


def test_classify_zero_error_unregistered_code_returned_as_is() -> None:
    """Zero 先行加码、本仓未跟：`classify` 如实返回该码（不吞不炸），由查表层降级处理。"""
    code = classify_zero_error(_wire("zero.step", "[zero:brand-new-code] 未来码"))
    assert code == "brand-new-code"
    assert code not in ZERO_ERROR_CODES


async def test_step_unknown_session_raises_subclass(monkeypatch: pytest.MonkeyPatch) -> None:
    """① 加壳的 unknown-session isError → 抛 ZeroLinkUnknownSessionError，
    且是 ZeroLinkCallError 子类、`.tool == 'zero.step'`（向后兼容 + 精确化）。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, _ZERO_UNKNOWN_SESSION_TEXT, is_error=True)
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    with pytest.raises(ZeroLinkUnknownSessionError) as exc_info:
        await client.step("sid-1", stimulus)

    assert isinstance(exc_info.value, ZeroLinkCallError)  # 子类关系（既有调用点零回归）
    assert exc_info.value.tool == "zero.step"
    # 不可降级族与它无关：unknown-session 是**可自愈**的，graceful_step 要 resume 而非上抛
    assert not isinstance(exc_info.value, ZeroLinkNonDegradableError)


async def test_step_legacy_unknown_session_raises_subclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """② 老部署裸前缀（加壳后）同样抛 ZeroLinkUnknownSessionError（过渡兼容不留缺口）。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, _LEGACY_UNKNOWN_SESSION_TEXT, is_error=True)
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    with pytest.raises(ZeroLinkUnknownSessionError):
        await client.step("sid-1", stimulus)


async def test_step_config_incompatible_raises_non_degradable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """③ config-incompatible → ZeroLinkConfigIncompatibleError（不可降级族）。

    判别性：它**不是** unknown-session 子类——不该走 resume 自愈路径。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, _ZERO_CONFIG_INCOMPATIBLE_TEXT, is_error=True)
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    with pytest.raises(ZeroLinkConfigIncompatibleError) as exc_info:
        await client.step("sid-1", stimulus)

    assert isinstance(exc_info.value, ZeroLinkNonDegradableError)
    assert isinstance(exc_info.value, ZeroLinkCallError)  # 既有 except 调用点零回归
    assert not isinstance(exc_info.value, ZeroLinkUnknownSessionError)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("payload-invalid", ZeroLinkCallerFaultError),
        ("external-prior-invalid", ZeroLinkCallerFaultError),
        ("config-invalid", ZeroLinkCallerFaultError),
        ("deploy-env-invalid", ZeroLinkDeployEnvError),
        # 动作通道未开：与 deploy-env-invalid **分类**（前者=env 值不合法该报警，
        # 后者=合法的默认关闭态，调用方该停止再调而非每轮报警），但同属不可降级。
        ("motion-disabled", ZeroLinkMotionDisabledError),
    ],
)
async def test_step_other_codes_map_to_expected_class(
    monkeypatch: pytest.MonkeyPatch, code: str, expected: type[ZeroLinkCallError]
) -> None:
    """其余五码的归类：调用方传参错 → CallerFault；部署端 env 错 → DeployEnv；
    动作通道未开 → MotionDisabled；均不可降级。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, _wire("zero.step", f"[zero:{code}] 文案"), is_error=True)
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    with pytest.raises(expected) as exc_info:
        await client.step("sid-1", stimulus)

    assert isinstance(exc_info.value, ZeroLinkNonDegradableError)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("timeout-lock", ZeroLinkLockTimeoutError),
        ("timeout-step", ZeroLinkStepTimeoutError),
    ],
)
async def test_step_timeout_codes_map_to_degradable_subclasses(
    monkeypatch: pytest.MonkeyPatch, code: str, expected: type[ZeroLinkCallError]
) -> None:
    """超时两码各挂独立子类且属**可降级**族（本仓第二轮回件 §2.5 承诺的消费侧落地）。

    判别性三连：不是 NonDegradable（graceful_step 不上抛）、不是 unknown-session
    （不触发 resume 自愈）、两码互不为对方子类（重试语义相反，正是分码的理由——
    编排层 catch `ZeroLinkLockTimeoutError` 退避重试时绝不能把 timeout-step 也捞进来）。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, _wire("zero.step", f"[zero:{code}] 文案"), is_error=True)
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    with pytest.raises(expected) as exc_info:
        await client.step("sid-1", stimulus)

    assert isinstance(exc_info.value, ZeroLinkCallError)  # 既有 except 调用点零回归
    assert not isinstance(exc_info.value, ZeroLinkNonDegradableError)
    assert not isinstance(exc_info.value, ZeroLinkUnknownSessionError)
    other = (
        ZeroLinkStepTimeoutError
        if expected is ZeroLinkLockTimeoutError
        else (ZeroLinkLockTimeoutError)
    )
    assert not isinstance(exc_info.value, other)


async def test_step_generic_error_not_unknown_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """④ 判别性：无令牌的普通 isError → 抛基类，既非 unknown-session 也非不可降级族。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, _wire("zero.step", "boom"), is_error=True)
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    with pytest.raises(ZeroLinkCallError) as exc_info:
        await client.step("sid-1", stimulus)

    assert not isinstance(exc_info.value, ZeroLinkUnknownSessionError)
    assert not isinstance(exc_info.value, ZeroLinkNonDegradableError)


async def test_step_unregistered_code_falls_back_to_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero 单边加新码 → 落基类 ZeroLinkCallError（不炸、不误升级），跨仓单边升级零回归。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, _wire("zero.step", "[zero:brand-new-code] x"), is_error=True)
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    with pytest.raises(ZeroLinkCallError) as exc_info:
        await client.step("sid-1", stimulus)

    assert type(exc_info.value) is ZeroLinkCallError


async def test_step_config_error_not_unknown_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """判别性：含「session_id」中文但**无令牌**的错误不被误判为 unknown-session。

    证明区分走机读令牌而非脆弱中文匹配——回执明言「靠字符串匹配脆弱」正是要规避的。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(
        mock_session, _wire("zero.step", "session_id 须为非空字符串，实际为 ''"), is_error=True
    )
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

    open_session 即便返回带 `[zero:unknown-session]` 令牌的 isError（未来 Zero 内部路由变化的
    假想场景），也只抛基类 ZeroLinkCallError，不误升级为 ZeroLinkUnknownSessionError
    ——保「子类 ⇒ resume 通路可走」严格成立。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(
        mock_session, _wire("zero.open_session", _ZERO_UNKNOWN_SESSION_INNER), is_error=True
    )

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
    """④ 判别性：无令牌的普通 isError → 不触发 resume，直接降级 None（不调 open）。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, _wire("zero.step", "boom"), is_error=True)
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    result = await client.graceful_step("sid-1", stimulus)

    assert result is None
    assert mock_session.call_tool.call_count == 1  # 只调 step，无 resume 的 open_session
    assert mock_session.call_tool.call_args.args[0] == "zero.step"


# ---------------------------------------------------------------------------
# Task 5.21 graceful_step 的「不可降级」分界（Zero §4.4-9）
#
# config-incompatible 意味活跃会话 config 不可变、须以新配置重开会话：静默 return None 会让
# **每一轮**都悄悄丢一次 step，且与偶发抖动在观测上不可区分 → 必须上抛。其余码同理归类。
# ---------------------------------------------------------------------------


async def test_graceful_step_config_incompatible_raises_not_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """③ config-incompatible → graceful_step **上抛**（不静默 None）、不触发 resume。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, _ZERO_CONFIG_INCOMPATIBLE_TEXT, is_error=True)
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    with pytest.raises(ZeroLinkConfigIncompatibleError):
        await client.graceful_step("sid-1", stimulus)

    # 不可自愈 → 不该白白多打一次 open_session
    assert mock_session.call_tool.call_count == 1


@pytest.mark.parametrize(
    "code",
    [
        "payload-invalid",
        "external-prior-invalid",
        "config-invalid",
        "deploy-env-invalid",
        # 动作通道未开：归类测试之外还要有这条**端到端**覆盖——归类对不代表
        # graceful_step 的 except 排序也对（新子类可能被更早的通用分支抢先兜住，
        # 那样就退回「每轮静默 return None」，正是本码归不可降级要防的形态）。
        "motion-disabled",
    ],
)
async def test_graceful_step_other_non_degradable_codes_raise(
    monkeypatch: pytest.MonkeyPatch, code: str
) -> None:
    """调用方传参错 / 部署端 env 错 / 动作通道未开同样上抛——静默每轮丢帧会把根因埋掉。"""
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, _wire("zero.step", f"[zero:{code}] 文案"), is_error=True)
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    with pytest.raises(ZeroLinkNonDegradableError):
        await client.graceful_step("sid-1", stimulus)


async def test_graceful_step_resume_path_non_degradable_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resume 路径上的不可降级错误同样上抛（顺序守卫）。

    序列：step(unknown-session) → open_session([zero:config-invalid]，resume_config 不合法)。
    若 `except ZeroLinkNonDegradableError: raise` 写在通用分支**之后**，它会被
    `except (ZeroLinkCallError, …)` 先兜住 → 又静默 None；本条即红。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    mock_session.call_tool.side_effect = [
        _make_call_result(_ZERO_UNKNOWN_SESSION_TEXT, is_error=True),
        _make_call_result(
            _wire("zero.open_session", "[zero:config-invalid] config 不合法：bad"), is_error=True
        ),
    ]
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    with pytest.raises(ZeroLinkCallerFaultError):
        await client.graceful_step("sid-1", stimulus, resume_config={"bad": 1})

    assert mock_session.call_tool.call_count == 2


async def test_graceful_step_lock_timeout_degrades_without_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """timeout-lock → 降级 None、**不触发 resume**（会话仍在，只是锁竞争，重开纯属浪费）。

    退避重试是关键路径编排层的事（catch `ZeroLinkLockTimeoutError` 自己做）；
    graceful_step 的契约是非关键路径丢一帧，不在兜底层里内置重试。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(
        mock_session,
        _wire("zero.step", "[zero:timeout-lock] 等待会话锁超时（5.0s）"),
        is_error=True,
    )
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    result = await client.graceful_step("sid-1", stimulus)

    assert result is None
    assert mock_session.call_tool.call_count == 1  # 只调 step，无 open_session、无重试
    assert mock_session.call_tool.call_args.args[0] == "zero.step"


async def test_graceful_step_step_timeout_no_retry_error_log(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """timeout-step → §2.5 承诺三件套：**不重试** + **ERROR** 级日志 + 降级 None。

    不重试是硬约束（半截运行态，原样重试会节点重跑/reducer 双重累加）；ERROR 级
    （非通用分支的 warning）让「内核慢」与偶发抖动在观测上分开。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(
        mock_session, _wire("zero.step", "[zero:timeout-step] 内核执行超时"), is_error=True
    )
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    with caplog.at_level(logging.ERROR, logger="src.mcp.zero.client"):
        result = await client.graceful_step("sid-1", stimulus)

    assert result is None
    assert mock_session.call_tool.call_count == 1  # 绝不原样重试
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("timeout-step" in r.getMessage() for r in error_records), (
        "timeout-step 降级必须留 ERROR 级日志（§2.5），否则慢内核在观测上与抖动不可分"
    )


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


# ---------------------------------------------------------------------------
# Task A（2026-07-29 止血）：消费 open_session 的 `resumed` / `interrupted_at`
#
# Zero 换代后 `zero.open_session` 返回 `{session_id, resumed}`，resume 且探测到上一轮被中途
# 取消时另带 `{interrupted_at: [待执行节点名]}`。两条硬约束：
#   ① **缺键即回落**——老部署只回 `{session_id}` 时行为逐字不变（现网零回归）；
#   ② **半截运行态不续跑**——graceful_step 的 unknown-session 自愈分支重开后**先看返回体**，
#      带非空 interrupted_at 就不重试 step（ERROR 日志 + 降级 None）。
# ---------------------------------------------------------------------------


def _client_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """只取本模块 logger 的记录（滤掉 pytest/其它库的噪声，避免断言被污染）。"""
    return [r for r in caplog.records if r.name == "src.mcp.zero.client"]


async def test_open_session_without_new_keys_is_zero_regression(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """🛑 **缺键即回落**：老部署只回 `{session_id}` → 返回值/日志/异常与换代前逐字一致。

    三个断言分别钉住零回归的三个面：
    ① 返回值仍是裸 session_id（不抛、不变形）；
    ② 两条观测量都是 `None`——「读不到」而非塌缩成 `False`/`[]`（否则调用方会把老部署
       误读成「Zero 说它没 resume」）；
    ③ **不打任何新日志**——缺键是老部署的正常态，不是异常，刷 warning 等于制造噪声。
    删掉 `_parse_*` 里的 `not in data` 回落分支即红（KeyError 穿透 / 打出伪日志）。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, json.dumps({"session_id": "sid-old"}))

    with caplog.at_level(logging.DEBUG, logger="src.mcp.zero.client"):
        sid = await client.open_session()

    assert sid == "sid-old"  # ①
    assert isinstance(client.last_open_session, ZeroOpenSessionInfo)
    assert client.last_open_session.resumed is None  # ②
    assert client.last_open_session.interrupted_at is None
    assert _client_records(caplog) == []  # ③


async def test_open_session_collects_resumed_and_interrupted_at(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """带两个新键 → 收下并挂在 client 上；`resumed` 落 INFO、`interrupted_at` 落 **WARNING**。

    级别不是随意选的：`interrupted_at` 非空 ⇒ 运行态停在 super-step 边界，是「后续帧不可全信」
    的信号，必须与普通 resume 在观测上分开。把 warning 改成 info 即红。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(
        mock_session,
        json.dumps(
            {"session_id": "sid-1", "resumed": True, "interrupted_at": ["affect_update", "render"]}
        ),
    )

    with caplog.at_level(logging.DEBUG, logger="src.mcp.zero.client"):
        sid = await client.open_session(session_id="sid-1")

    assert sid == "sid-1"
    info = client.last_open_session
    assert info is not None
    assert info.resumed is True
    assert info.interrupted_at == ("affect_update", "render")  # 元组化、保序
    records = _client_records(caplog)
    info_msgs = [r.getMessage() for r in records if r.levelno == logging.INFO]
    warn_msgs = [r.getMessage() for r in records if r.levelno == logging.WARNING]
    assert any("resumed=True" in m for m in info_msgs), f"resumed 须落 INFO，实得 {records!r}"
    assert any("affect_update" in m for m in warn_msgs), (
        f"interrupted_at 须落 WARNING 且带节点名，实得 {records!r}"
    )


async def test_open_session_resumed_false_is_not_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """判别性：`resumed: false` 必须读成 `False`（**不是** `None`）——与「缺键」严格可分。

    若实现用 `data.get("resumed") or None` 这类真值写法，本条即红（False 被塌缩成 None）。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, json.dumps({"session_id": "fresh", "resumed": False}))

    await client.open_session()

    assert client.last_open_session is not None
    assert client.last_open_session.resumed is False
    assert client.last_open_session.interrupted_at is None  # 未带该键


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"session_id": "s", "resumed": "true"}, "resumed"),  # 字符串伪真值
        ({"session_id": "s", "resumed": 1}, "resumed"),  # int（bool 的父类，须拒）
        ({"session_id": "s", "resumed": None}, "resumed"),
        ({"session_id": "s", "interrupted_at": {"node": 1}}, "interrupted_at"),  # dict
        ({"session_id": "s", "interrupted_at": None}, "interrupted_at"),
        ({"session_id": "s", "interrupted_at": "affect_update"}, "interrupted_at"),  # 裸 str
        ({"session_id": "s", "interrupted_at": ["ok", 7]}, "interrupted_at"),  # 含非 str
    ],
)
async def test_open_session_malformed_new_keys_do_not_raise(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    payload: dict[str, object],
    field: str,
) -> None:
    """形状防御：新键形状异常一律「读不到就当没有」+ 一条 warning，**绝不炸**。

    会话生命周期不能因为一条观测量的类型不对就打不开（open_session 的失败会一路上抛）。
    注意 `resumed: 1`：`isinstance(1, bool)` 为 False，故 int 也被拒——JSON 里的 `1` 不是 bool，
    把它当 True 用就是在猜对方意图。裸 str `"affect_update"` 亦须拒（可迭代但不是 list，
    逐字符拆开会得到一串假节点名）。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, json.dumps(payload))

    with caplog.at_level(logging.DEBUG, logger="src.mcp.zero.client"):
        sid = await client.open_session()

    assert sid == "s"  # 不炸、照常返回
    info = client.last_open_session
    assert info is not None
    assert getattr(info, field) is None, f"{field} 形状异常须回落 None，实得 {info!r}"
    warn_msgs = [r.getMessage() for r in _client_records(caplog) if r.levelno == logging.WARNING]
    assert any(field in m for m in warn_msgs), f"形状异常须留一条 warning，实得 {caplog.records!r}"


async def test_graceful_step_resume_interrupted_does_not_retry_step(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """🛑 **主修守卫**：自愈分支重开后拿到 `interrupted_at` → **不重试 step**，ERROR + 降级 None。

    序列：step(unknown-session) → open_session({...interrupted_at:[...]})，到此为止。
    还原成旧行为（重开后无条件 `return await self.step(...)`）即红：call_count 会变成 3，
    且第三次调用是 `zero.step`——那正是「在半截运行态上静默续跑」。
    三个断言各钉一面：不续跑（调用序列）、不静默（ERROR + 节点名）、不上抛（返回 None）。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    mock_session.call_tool.side_effect = [
        _make_call_result(_ZERO_UNKNOWN_SESSION_TEXT, is_error=True),
        _make_call_result(
            json.dumps(
                {"session_id": "sid-1", "resumed": True, "interrupted_at": ["affect_update"]}
            )
        ),
        # 第三个返回体故意备着：若实现仍无条件续跑，它会被消费掉 → 调用序列断言变红
        _make_call_result(_make_expression_json()),
    ]
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    with caplog.at_level(logging.DEBUG, logger="src.mcp.zero.client"):
        result = await client.graceful_step("sid-1", stimulus)

    assert result is None  # (b) 仍降级不上抛
    assert [c.args[0] for c in mock_session.call_tool.call_args_list] == [
        "zero.step",
        "zero.open_session",
    ], "重开后不得再发 step —— 那是在半截运行态上叠加新刺激"
    error_msgs = [r.getMessage() for r in _client_records(caplog) if r.levelno == logging.ERROR]
    assert any("affect_update" in m for m in error_msgs), (  # (a) 不静默且带节点名
        f"半截态拒绝续跑必须留 ERROR 级日志且带节点名，实得 {caplog.records!r}"
    )


@pytest.mark.parametrize(
    "open_payload",
    [
        {"session_id": "sid-1"},  # 老部署：两个新键都没有
        {"session_id": "sid-1", "resumed": True},  # 换代后但未探测到中断
        {"session_id": "sid-1", "resumed": True, "interrupted_at": []},  # 空表 = 无待执行节点
        {"session_id": "sid-1", "interrupted_at": {"bad": 1}},  # 形状异常 → 读不到就当没有
    ],
)
async def test_graceful_step_resume_retries_when_not_interrupted(
    monkeypatch: pytest.MonkeyPatch, open_payload: dict[str, object]
) -> None:
    """零回归：**没有**非空 `interrupted_at` 的四种返回体，自愈重试通路一律照旧走完。

    这是主修的另一半——「不续跑」只许在真有半截态时生效。若把判定写成
    `if info.interrupted_at is not None`（空表/形状异常也算中断）或干脆无条件拒绝，本条即红。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    mock_session.call_tool.side_effect = [
        _make_call_result(_ZERO_UNKNOWN_SESSION_TEXT, is_error=True),
        _make_call_result(json.dumps(open_payload)),
        _make_call_result(_make_expression_json()),
    ]
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    result = await client.graceful_step("sid-1", stimulus)

    assert isinstance(result, ExpressionBundle)
    assert [c.args[0] for c in mock_session.call_tool.call_args_list] == [
        "zero.step",
        "zero.open_session",
        "zero.step",
    ]


async def test_graceful_step_resume_retry_non_degradable_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """顺序守卫（新增分支不得破坏 `except ZeroLinkNonDegradableError: raise` 的排位）。

    序列：step(unknown-session) → open_session(ok，无 interrupted_at) → step(config-incompatible)。
    重开成功后**继续走到重试 step**，其抛的不可降级错误必须穿过内层 except 组上抛；若
    `except ZeroLinkNonDegradableError` 被挪到可降级元组之后，它会被先兜住 → 静默 None，本条即红。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    mock_session.call_tool.side_effect = [
        _make_call_result(_ZERO_UNKNOWN_SESSION_TEXT, is_error=True),
        _make_call_result(json.dumps({"session_id": "sid-1", "resumed": True})),
        _make_call_result(_ZERO_CONFIG_INCOMPATIBLE_TEXT, is_error=True),
    ]
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    with pytest.raises(ZeroLinkConfigIncompatibleError):
        await client.graceful_step("sid-1", stimulus)

    assert mock_session.call_tool.call_count == 3


# ---------------------------------------------------------------------------
# Task A·B1（2026-07-29 复核订正）：`interrupted_at` 的**四态**判读 + 残留缺口特征化
#
# 上一版把「键缺席」一律读成「未中断，可安全续跑」，被 Zero `daecce1` 源码否证：
# 缺席有四义（未探测·新建 / 未探测·活跃幂等重开 / 探测失败 / 探测成功且干净），其中「探测失败」
# 与我方要防的半截态**故障相关**（探测读的正是那份可能半写的 checkpoint）。
# 解析层因此改为四态；自愈分支逐格处置。
#
# 同时把「本帧拒绝」的真实收益特征化下来：它**没有**避免污染，只把污染推迟一帧。
#
# 🛑 **本段所有日志断言一律锚 `LOG_MARKER_*`，不锚中文文案**（2026-07-29 复审的实测教训）：
#    上一版锚的是计数词 `"三义"`，而该计数词本身正在被订正（三→四）。三条断言里**两条是
#    否定式**（`assert not any(...)`），文案一改就会**静默变成空真**——肯定式那条会红提醒你，
#    否定式那两条不会，守卫看着仍绿、判别力已归零。marker 是 ASCII 常量、由 client 导出、
#    改文案不改它 ⇒ 断言与措辞解耦（pitfalls ⑦ 脆弱锚点）。
# ---------------------------------------------------------------------------


_INTERRUPTED_OPEN_PAYLOAD = json.dumps(
    {"session_id": "sid-1", "resumed": True, "interrupted_at": ["affect_update"]}
)


def _error_msgs(caplog: pytest.LogCaptureFixture) -> list[str]:
    """本模块 logger 的 ERROR 文案。"""
    return [r.getMessage() for r in _client_records(caplog) if r.levelno == logging.ERROR]


def _warning_msgs(caplog: pytest.LogCaptureFixture) -> list[str]:
    """本模块 logger 的 WARNING 文案。"""
    return [r.getMessage() for r in _client_records(caplog) if r.levelno == logging.WARNING]


def _marked_records(
    caplog: pytest.LogCaptureFixture, marker: str, level: int
) -> list[logging.LogRecord]:
    """取带指定 `LOG_MARKER_*` 的日志记录 —— 锚 `record.args[0]`，**不锚渲染后的中文**。

    client 侧所有带 marker 的日志一律把 marker 作为**第一个** `%s` 参数，故 `args[0]` 是
    结构化锚点：文案怎么改都不影响它，连 `%` 渲染都不必发生。这是本轮把守卫从中文计数词
    （`"三义"`）迁走后的稳定标识（pitfalls ⑦）。
    """
    return [
        r
        for r in _client_records(caplog)
        if r.levelno == level and isinstance(r.args, tuple) and r.args[:1] == (marker,)
    ]


@pytest.mark.parametrize(
    ("payload", "expected_probe", "expected_nodes"),
    [
        ({"session_id": "s"}, ZeroInterruptProbe.ABSENT, None),
        ({"session_id": "s", "interrupted_at": []}, ZeroInterruptProbe.CLEAN, ()),
        ({"session_id": "s", "interrupted_at": ["n1"]}, ZeroInterruptProbe.INTERRUPTED, ("n1",)),
        ({"session_id": "s", "interrupted_at": "n1"}, ZeroInterruptProbe.MALFORMED, None),
    ],
)
async def test_interrupt_probe_distinguishes_four_states(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
    expected_probe: ZeroInterruptProbe,
    expected_nodes: tuple[str, ...] | None,
) -> None:
    """🛑 解析层四态可分 —— 「键缺席」≠「键在但为空」≠「形状坏」。

    信息损失的具体位置（本轮修的）：`ABSENT` 与 `MALFORMED` 在 `interrupted_at` 上都是
    ``None``，只看那一个字段永远分不开「对方没说」与「对方说了但契约漂移」。判定必须读
    `interrupt_probe`。
    把 `_parse_open_session_interrupted_at` 退回「统一返回 tuple|None」即红（前两行与后两行
    分别塌缩成同一格）；把空表也判成 `INTERRUPTED`（漏掉 `if nodes else`）第二行即红。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, json.dumps(payload))

    await client.open_session()

    info = client.last_open_session
    assert info is not None
    assert info.interrupt_probe is expected_probe
    assert info.interrupted_at == expected_nodes


async def test_interrupt_probe_four_states_are_pairwise_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """判别性总闸：四种返回体判出的 probe **两两不同**（任意两格塌缩即红）。

    与上一条互补——上一条钉「每格取到哪个值」，本条钉「没有两格取到同一个值」。
    只看 `interrupted_at` 时 ABSENT/MALFORMED 都是 ``None``，本条即是那处塌缩的守卫。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    probes: list[ZeroInterruptProbe] = []
    for extra in ({}, {"interrupted_at": []}, {"interrupted_at": ["n1"]}, {"interrupted_at": 7}):
        _set_tool_return(mock_session, json.dumps({"session_id": "s", **extra}))
        await client.open_session()
        assert client.last_open_session is not None
        probes.append(client.last_open_session.interrupt_probe)

    assert len(set(probes)) == 4, f"四态必须两两可分，实得 {probes!r}"


async def test_graceful_step_resume_absent_key_old_deployment_logs_nothing_new(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """三态①**老部署**（`resumed` 键都不发）：照常续跑，且**不打**「不可判」WARNING（零回归）。

    判别位是 `resumed`：新 Zero **无条件**回它（两条 return 路径都带），老部署根本不发
    ⇒ `resumed is None` ⇔「对方不是会发中断观测量的那一代」，此时缺 `interrupted_at`
    是正常态、不是信号。若把「不可判」WARNING 的条件从 `info.resumed is True` 放宽成
    `info.interrupt_probe is ABSENT`，本条即红——老部署每一次自愈都会被刷一条无意义告警。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    mock_session.call_tool.side_effect = [
        _make_call_result(_ZERO_UNKNOWN_SESSION_TEXT, is_error=True),
        _make_call_result(json.dumps({"session_id": "sid-1"})),  # 老部署：两个新键都没有
        _make_call_result(_make_expression_json()),
    ]

    with caplog.at_level(logging.DEBUG, logger="src.mcp.zero.client"):
        result = await client.graceful_step("sid-1", AffectStimulus(valence=0.1, arousal=0.2))

    assert isinstance(result, ExpressionBundle)  # 照常续跑
    assert not any(LOG_MARKER_PROBE_UNDECIDABLE in m for m in _warning_msgs(caplog)), (
        f"老部署不得被刷「不可判」告警，实得 {caplog.records!r}"
    )
    assert _error_msgs(caplog) == []


async def test_graceful_step_resume_absent_key_with_resumed_true_warns_undecidable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """三态②**新 Zero 但不可判**（`resumed: true` + 键缺席）：照常续跑 + 一条**可区分**的 WARNING。

    这一格覆盖 Zero 侧「中断探测失败」（宽 `except Exception` 只记日志）——它与我方要防的
    半截态是故障相关的，但从返回体上与「探测干净」完全同形，我方**无法判定**。
    处置论证见 client 分支内注释：保守拒绝会误伤 100% 的健康 resume（探测干净同样不发该键），
    等于废掉整个自愈能力；故照常续跑，但必须留下可与 Zero 侧日志做时间对齐的可区分 WARNING。
    删掉该 WARNING 分支即红（三态②与③在观测上重新变得不可区分）。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    mock_session.call_tool.side_effect = [
        _make_call_result(_ZERO_UNKNOWN_SESSION_TEXT, is_error=True),
        _make_call_result(json.dumps({"session_id": "sid-1", "resumed": True})),
        _make_call_result(_make_expression_json()),
    ]

    with caplog.at_level(logging.DEBUG, logger="src.mcp.zero.client"):
        result = await client.graceful_step("sid-1", AffectStimulus(valence=0.1, arousal=0.2))

    assert isinstance(result, ExpressionBundle)  # 不保守拒绝
    assert [c.args[0] for c in mock_session.call_tool.call_args_list] == [
        "zero.step",
        "zero.open_session",
        "zero.step",
    ]
    assert any(LOG_MARKER_PROBE_UNDECIDABLE in m for m in _warning_msgs(caplog)), (
        f"不可判的一格必须留可区分 WARNING，实得 {caplog.records!r}"
    )


@pytest.mark.parametrize(
    ("open_payload", "why"),
    [
        ({"session_id": "sid-1", "resumed": True, "interrupted_at": []}, "CLEAN：对方明确说干净"),
        ({"session_id": "sid-1", "resumed": False}, "新建会话：没有旧运行态可污染"),
    ],
)
async def test_graceful_step_resume_decidable_states_do_not_warn(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    open_payload: dict[str, object],
    why: str,
) -> None:
    """三态③**可判且安全**：照常续跑且**不打**「不可判」WARNING。

    两格都必须与②严格分开：
    · `interrupted_at: []` = 对方**明确**探测过且无待执行节点（证据强度高于「没说」）；
    · `resumed: false` = 对方新铸了会话，压根没有旧运行态可被污染。
    若把 WARNING 条件写成「只要没有非空 interrupted_at 就告警」，本条两格全红。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    mock_session.call_tool.side_effect = [
        _make_call_result(_ZERO_UNKNOWN_SESSION_TEXT, is_error=True),
        _make_call_result(json.dumps(open_payload)),
        _make_call_result(_make_expression_json()),
    ]

    with caplog.at_level(logging.DEBUG, logger="src.mcp.zero.client"):
        result = await client.graceful_step("sid-1", AffectStimulus(valence=0.1, arousal=0.2))

    assert isinstance(result, ExpressionBundle), why
    assert not any(LOG_MARKER_PROBE_UNDECIDABLE in m for m in _warning_msgs(caplog)), (
        f"{why} → 不得打「不可判」告警，实得 {caplog.records!r}"
    )


async def test_graceful_step_resume_malformed_probe_errors_but_continues(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`MALFORMED`：**照常续跑**（不可判，不能因对方一个类型笔误就废掉自愈）+ **ERROR**。

    级别选 ERROR 不是随意：形状坏是跨语言契约破裂的直接证据，比任何单帧数据都重要。
    把该分支并回 `ABSENT`（只 warning 或不打）即红；把它并进「拒绝续跑」也红（result 变 None）。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    mock_session.call_tool.side_effect = [
        _make_call_result(_ZERO_UNKNOWN_SESSION_TEXT, is_error=True),
        _make_call_result(
            json.dumps({"session_id": "sid-1", "resumed": True, "interrupted_at": 7})
        ),
        _make_call_result(_make_expression_json()),
    ]

    with caplog.at_level(logging.DEBUG, logger="src.mcp.zero.client"):
        result = await client.graceful_step("sid-1", AffectStimulus(valence=0.1, arousal=0.2))

    assert isinstance(result, ExpressionBundle)
    assert any(LOG_MARKER_PROBE_MALFORMED in m for m in _error_msgs(caplog)), (
        f"形状漂移须落 ERROR，实得 {caplog.records!r}"
    )


# ── 残留缺口的**特征化**（不是缺陷断言，是把已知缺口做成可执行记录）─────────────


async def test_next_frame_after_interrupted_refusal_runs_normal_step(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """🔖 **特征化·已知缺口**：本帧拒绝**没有**避免污染，只把它推迟一帧，且此后不可观测。

    钉住的跨仓事实（Zero `daecce1` 现场核验，2026-07-29）：
      (a) 止血判定只能在**重开之后**做，而重开已让该会话在 Zero registry 变活跃
          ⇒ 下一帧不再报 unknown-session ⇒ 走正常 `zero.step` 路径。
          ⚠ 口径订正（Zero `667e923`）：其 `step` 现有**每轮事后中断检查**，但**只记 WARNING、
          不改返回体、不拒帧** ⇒ 对**我方侧**仍不可观测（返回值与健康帧逐位同形），
          本用例记录的现象不变；排障时可按 sid 去对方日志对齐。
          旧表述「Zero 的 step 完全不做中断检查」已被该提交推翻，勿再引用；
      (b) Zero 的 `interrupted_at()` 只在「resume 且**不活跃**」时探测（活跃分支提前 return）
          ⇒ 这条 ERROR 对同一 session **一生只出现一次**。
    本用例记录的正是「第二帧长得和一个健康帧一模一样」这件事：无重开、无探测、无日志、
    返回值是正常 `ExpressionBundle`，我方也**不记账**（脏会话在 client 侧无记忆）。

    🛑 将来若 Zero 让 `zero.step` 也检测中断（或在活跃 resume 上继续探测），第二帧就不再是
    一次干净的 step —— 断言 ① / ③ 会变红。**那时请勿直接改绿**：先更新跨仓认知
    （client docstring 里的因果链、给 Zero 的回件），再重写本用例。

    🛑 **对称提醒：我方自己动手时本用例同样会红**（2026-07-29 复审补——上一版只警告了
    「Zero 若改则勿直接改绿」，把这个缺口写得像是单侧无解，措辞过头）。缺口是「**第一帧**
    躲不掉」，不是「后续帧无解」：client 侧记一笔脏 session（`session_id → INTERRUPTED`），
    下一帧开头即可拒绝或再报一次 ERROR —— 那是**可实现的单侧缓解**，本轮**有意不做**
    （代价见 client `graceful_step` docstring：会给无状态句柄引入会话级记账、多实例/多进程
    下各记各的仍不完整、且清账时机无契约可依）。⇒ 真去做记账时，①③（无重开的干净 step）
    与 ②（第二帧零 ERROR）**会变红，那是预期的**，届时按新语义重写本用例即可，
    不必把它当回归。

    变异实证：把第二帧也做成拒绝（client 记住脏 session）→ ①③ 红；
    在第二帧补打一条 ERROR → ② 红。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    mock_session.call_tool.side_effect = [
        _make_call_result(_ZERO_UNKNOWN_SESSION_TEXT, is_error=True),  # 帧1：step
        _make_call_result(_INTERRUPTED_OPEN_PAYLOAD),  # 帧1：重开 → 检出半截
        _make_call_result(_make_expression_json()),  # 帧2：step 直接成功
    ]
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    with caplog.at_level(logging.DEBUG, logger="src.mcp.zero.client"):
        first = await client.graceful_step("sid-1", stimulus)
        errors_after_first = len(_error_msgs(caplog))
        calls_after_first = len(mock_session.call_tool.call_args_list)
        second = await client.graceful_step("sid-1", stimulus)

    assert first is None  # 帧1 拒绝
    assert errors_after_first == 1  # 帧1 恰一条 ERROR
    # ① 帧2 只发了一次 zero.step：不重开、不探测 —— 与一个健康帧的调用序列**逐字相同**
    assert [c.args[0] for c in mock_session.call_tool.call_args_list[calls_after_first:]] == [
        "zero.step"
    ]
    # ② 帧2 一条新 ERROR 都没有 —— 「一生只出现一次」，此后污染不可观测
    assert len(_error_msgs(caplog)) == errors_after_first
    # ③ 帧2 拿回的是形态完全正常的返回值（静默续跑在半截运行态上）
    assert isinstance(second, ExpressionBundle)
    # ④ client 侧唯一的残渣：last_open_session 仍记着帧1 的 INTERRUPTED，但**无人消费**
    assert client.last_open_session is not None
    assert client.last_open_session.interrupt_probe is ZeroInterruptProbe.INTERRUPTED


async def test_normal_resume_path_has_no_interrupt_guard(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """🔖 **特征化·已知缺口（本轮新披露）**：拒绝/purge **只挂在 unknown-session 自愈分支**，
    **常规 resume 路径完全无守卫**。

    序列即调用方的正常用法：`open_session(session_id=…)` 拿到
    `{resumed: true, interrupted_at: [...]}` → 直接 `step()` → 再走 `graceful_step` 的
    **正常**路径（首次 step 就成功，压根不进 except）。实测：
      · 两次调用全部照常返回 `ExpressionBundle`；
      · 全程**一条 ERROR 都没有**；
      · 唯一信号是 `open_session` 里那条 WARNING（`LOG_MARKER_INTERRUPTED_ON_OPEN`）。
    ⚠ Zero 侧的兜底口径已变（`667e923`）：其 `zero.step` 现有**每轮事后中断检查**，
    但**只记 WARNING、不改返回体、不拒帧** ⇒ 上面三条实测结论**逐条仍成立**
    （我方这侧看到的仍是「正常返回、无 ERROR、只有 open 那条 WARNING」）。
    早先「`zero.step` 不做任何中断检查（`daecce1` 全函数 grep 零命中）」的表述已过期，勿再引用
    ——现在的准确说法是「**我方侧**不可观测，对方侧每帧有一条 WARNING」。

    本轮**有意不在正常路径加拦截**（论证见 client `graceful_step` docstring「常规 resume 路径」
    段：处置权在调用方、误伤面比自愈路径大得多、正确形态是给一个显式 API 而非在降级路径里
    偷改语义）。⇒ 本用例不是缺陷断言，是把这条缺口做成**可执行记录**：
    将来任一侧补上守卫（我方拦截 / Zero 让 step 也检中断），断言 ①②③ 会红，
    **届时勿直接改绿**，先更新跨仓认知再重写。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    mock_session.call_tool.side_effect = [
        _make_call_result(_INTERRUPTED_OPEN_PAYLOAD),  # 调用方自己 resume，直接看到半截态
        _make_call_result(_make_expression_json()),  # 裸 step：无守卫
        _make_call_result(_make_expression_json()),  # graceful_step 正常路径：同样无守卫
    ]
    stimulus = AffectStimulus(valence=0.1, arousal=0.2)

    with caplog.at_level(logging.DEBUG, logger="src.mcp.zero.client"):
        sid = await client.open_session(session_id="sid-1")
        bare = await client.step(sid, stimulus)
        graceful = await client.graceful_step(sid, stimulus)

    assert sid == "sid-1"
    # ① 裸 step 照常返回（无任何拦截）
    assert isinstance(bare, ExpressionBundle)
    # ② graceful_step 的正常路径同样照常返回 —— 半截态对它完全不可见
    assert isinstance(graceful, ExpressionBundle)
    # ③ 全程零 ERROR：唯一信号是 open_session 的那条 WARNING
    assert _error_msgs(caplog) == [], f"常规路径今天不产生 ERROR，实得 {caplog.records!r}"
    assert len(_marked_records(caplog, LOG_MARKER_INTERRUPTED_ON_OPEN, logging.WARNING)) == 1
    # ④ 探针信息**拿得到**（调用方有据可依，缺的只是我方不替它决策）
    assert client.last_open_session is not None
    assert client.last_open_session.interrupt_probe is ZeroInterruptProbe.INTERRUPTED


# ── 破坏性恢复动作（zero.purge_session）：默认关，显式开 ─────────────────────────


async def test_graceful_step_interrupted_does_not_purge_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """🛑 **破坏性动作默认不执行**：检出半截态**不**自动调 `zero.purge_session`。

    默认 False 的论证（见 graceful_step docstring）：purge 不可逆且过度杀伤（删该 thread
    **全部** checkpoint 历史，含干净祖先）；判据 `next` 非空有良性同形（图停在 interrupt/
    断点也是 `next` 非空）；且让全系统最可降级的路径去做最不可逆的动作，层级是反的。
    把默认值改成 True 即红。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    mock_session.call_tool.side_effect = [
        _make_call_result(_ZERO_UNKNOWN_SESSION_TEXT, is_error=True),
        _make_call_result(_INTERRUPTED_OPEN_PAYLOAD),
    ]

    result = await client.graceful_step("sid-1", AffectStimulus(valence=0.1, arousal=0.2))

    assert result is None
    assert [c.args[0] for c in mock_session.call_tool.call_args_list] == [
        "zero.step",
        "zero.open_session",
    ], "默认路径上不得出现 zero.purge_session"


async def test_graceful_step_interrupted_purges_when_opted_in(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """显式开关打开 → 拒绝续跑**之后**调一次 `zero.purge_session`，仍降级 None（不重试本帧）。

    本帧不重试是有意的：purge 已把会话摘牌 + 删态，同帧再 step 只会又撞 unknown-session；
    留给**下一帧**用同 id 重开出一个空运行态的新会话，那才是真正的止血闭环。
    去掉 `if purge_on_interrupted:` 分支即红（调用序列少一项）。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    mock_session.call_tool.side_effect = [
        _make_call_result(_ZERO_UNKNOWN_SESSION_TEXT, is_error=True),
        _make_call_result(_INTERRUPTED_OPEN_PAYLOAD),
        _make_call_result(json.dumps({"ok": True, "purged": True})),
    ]

    with caplog.at_level(logging.DEBUG, logger="src.mcp.zero.client"):
        result = await client.graceful_step(
            "sid-1", AffectStimulus(valence=0.1, arousal=0.2), purge_on_interrupted=True
        )

    assert result is None  # 本帧仍降级，不重试
    assert [c.args[0] for c in mock_session.call_tool.call_args_list] == [
        "zero.step",
        "zero.open_session",
        "zero.purge_session",
    ]
    assert mock_session.call_tool.call_args_list[-1].args[1] == {"session_id": "sid-1"}
    assert any("purged=True" in m for m in _warning_msgs(caplog)), (
        f"破坏性动作执行后必须留痕，实得 {caplog.records!r}"
    )


def test_refusal_markers_are_not_substrings_of_each_other() -> None:
    """反空真前置守卫：两个拒绝 marker 必须**互不为子串**，否则下一条的「禁出现」断言恒真。

    这正是上一轮踩的那类坑的一般形式（pitfalls ⑥ 恒真式）：`assert marker_a not in msg`
    只有在 `marker_a` 确实可能出现时才有判别力。若将来有人把 marker 改名成
    `"[zl:interrupted-refused]"` / `"[zl:interrupted-refused-x]"` 之外的互含形式，
    先在这里红，而不是让下一条守卫悄悄失效。
    """
    assert LOG_MARKER_INTERRUPTED_REFUSED not in LOG_MARKER_INTERRUPTED_REFUSED_PURGING
    assert LOG_MARKER_INTERRUPTED_REFUSED_PURGING not in LOG_MARKER_INTERRUPTED_REFUSED


@pytest.mark.parametrize(
    ("purge_on", "expected_marker", "forbidden_marker"),
    [
        (False, LOG_MARKER_INTERRUPTED_REFUSED, LOG_MARKER_INTERRUPTED_REFUSED_PURGING),
        (True, LOG_MARKER_INTERRUPTED_REFUSED_PURGING, LOG_MARKER_INTERRUPTED_REFUSED),
    ],
)
async def test_interrupted_refusal_error_text_is_branch_specific(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    purge_on: bool,
    expected_marker: str,
    forbidden_marker: str,
) -> None:
    """🛑 **文案必须按 `purge_on_interrupted` 分支出**（2026-07-29 复审实测的自相矛盾）。

    上一版无条件先打同一条 ERROR、再看开关，于是 `purge_on_interrupted=True` 时输出：
      ERROR   「…脏运行态仍在，**下一帧**将走正常 step 路径在其上续跑…要真正止血请以
               `purge_on_interrupted=True` 调用」
      WARNING 「已清除该会话**全部**持久运行态（purged=True）」
    —— 状态刚被删掉，ERROR 却断言下一帧会在其上续跑，并劝调用方去开一个**已经开着**的开关。
    两条文案自相矛盾，且后者会把运维引向错误动作。

    还原成「无条件打同一条 ERROR」即红：`purge_on=True` 那一行会既缺 `_PURGING` marker、
    又出现被禁的普通 marker。断言全部锚 `record.args[0]`（见 `_marked_records`），
    与中文措辞解耦。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    mock_session.call_tool.side_effect = [
        _make_call_result(_ZERO_UNKNOWN_SESSION_TEXT, is_error=True),
        _make_call_result(_INTERRUPTED_OPEN_PAYLOAD),
        _make_call_result(json.dumps({"ok": True, "purged": True})),  # 仅 purge_on=True 会消费
    ]

    with caplog.at_level(logging.DEBUG, logger="src.mcp.zero.client"):
        result = await client.graceful_step(
            "sid-1",
            AffectStimulus(valence=0.1, arousal=0.2),
            purge_on_interrupted=purge_on,
        )

    assert result is None
    hit = _marked_records(caplog, expected_marker, logging.ERROR)
    assert len(hit) == 1, f"本分支须恰有一条对应 ERROR，实得 {caplog.records!r}"
    assert _marked_records(caplog, forbidden_marker, logging.ERROR) == [], (
        f"另一分支的文案不得出现（正是上一版的自相矛盾来源），实得 {caplog.records!r}"
    )
    # 节点名仍须在场（两条文案都带），否则「响亮的 ERROR」名不副实
    assert "affect_update" in hit[0].getMessage()
    # purge 分支：ERROR 只能说「已请求 + 结果见随后日志」，真正的结论由随后的 WARNING 给出
    purged_warned = any("purged=True" in m for m in _warning_msgs(caplog))
    assert purged_warned is purge_on, (
        f"purge 结论只应在开关打开时出现，purge_on={purge_on}，实得 {caplog.records!r}"
    )


async def test_graceful_step_purge_failure_is_swallowed_even_if_non_degradable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """善后失败**永不上抛**——连不可降级码也吞（本方法只在「本帧已判定降级」之后跑）。

    理由：让一次**清理**失败把一条已判定为可降级的调用变成 raise，等于用更坏的失败模式
    替换较轻的那个。这是 `_purge_after_interrupted` 刻意宽于 `graceful_step` 主路径的一处，
    故单独钉住。把 `_purge_after_interrupted` 的 except 去掉即红（异常穿透成 raise）。
    同时钉「脏运行态仍在」必须落 ERROR：清理没成功而调用方以为成功，比不清理更危险。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    mock_session.call_tool.side_effect = [
        _make_call_result(_ZERO_UNKNOWN_SESSION_TEXT, is_error=True),
        _make_call_result(_INTERRUPTED_OPEN_PAYLOAD),
        _make_call_result(
            _wire("zero.purge_session", "[zero:config-incompatible] boom"), is_error=True
        ),
    ]

    with caplog.at_level(logging.DEBUG, logger="src.mcp.zero.client"):
        result = await client.graceful_step(
            "sid-1", AffectStimulus(valence=0.1, arousal=0.2), purge_on_interrupted=True
        )

    assert result is None  # 不上抛
    assert any("脏运行态仍在" in m for m in _error_msgs(caplog)), (
        f"清理失败必须显式说明状态未恢复，实得 {caplog.records!r}"
    )


@pytest.mark.parametrize(
    "transport_exc",
    [anyio.BrokenResourceError(), anyio.ClosedResourceError(), anyio.EndOfStream()],
    ids=["broken-resource", "closed-resource", "end-of-stream"],
)
async def test_purge_swallows_raw_transport_exceptions(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, transport_exc: Exception
) -> None:
    """🛑 善后路径必须兜住**底层传输异常**——它们不属 OSError 家族，旧 except 元组接不住。

    现场核验（本机 anyio，2026-07-29）：
        anyio.BrokenResourceError.__mro__ == (BrokenResourceError, Exception, BaseException, object)
        issubclass(anyio.BrokenResourceError, OSError) is False
    而 `ZeroLinkConnectionError` 是 **OSError** 子类 ⇒ 旧的
    `except (ZeroLinkCallError, ZeroLinkConnectionError, McpError)` 对这三个**一个都接不住**。
    偏偏 purge 发生在「刚撞过 unknown-session（多半是对端重启）」之后，正是 stdio 管道最易碎
    的时刻 —— 于是一条**已判定降级 None** 的调用会变成 raise，正是该机制要避免的失败升级。
    ⇒ docstring 的「永不上抛」措辞也随之订正为「吞 `Exception` 全族，`BaseException` 仍穿透」。

    把 except 退回原来的三元组即红（异常穿透 → `pytest.raises` 之外的 raise）。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    mock_session.call_tool.side_effect = [
        _make_call_result(_ZERO_UNKNOWN_SESSION_TEXT, is_error=True),
        _make_call_result(_INTERRUPTED_OPEN_PAYLOAD),
        transport_exc,  # zero.purge_session 时管道已断
    ]

    with caplog.at_level(logging.DEBUG, logger="src.mcp.zero.client"):
        result = await client.graceful_step(
            "sid-1", AffectStimulus(valence=0.1, arousal=0.2), purge_on_interrupted=True
        )

    assert result is None  # 不上抛：降级契约保住
    # 放宽 except **不制造静默面**：失败分支用 logger.exception ⇒ 必带完整 traceback。
    failures = [
        r for r in _client_records(caplog) if r.levelno == logging.ERROR and r.exc_info is not None
    ]
    assert len(failures) == 1, f"清理失败须留一条带 traceback 的 ERROR，实得 {caplog.records!r}"
    assert type(transport_exc).__name__ in failures[0].getMessage()


async def test_purge_does_not_swallow_base_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """`BaseException` **仍原样穿透**——放宽 except 的边界必须钉死，否则「吞一切」变成新静默面。

    `asyncio.CancelledError` 是 BaseException：善后不是压住取消语义的理由（同 `__aenter__`
    里对 CancelledError 的处置口径）。把 `except Exception` 写成 `except BaseException` 即红。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    mock_session.call_tool.side_effect = [
        _make_call_result(_ZERO_UNKNOWN_SESSION_TEXT, is_error=True),
        _make_call_result(_INTERRUPTED_OPEN_PAYLOAD),
        asyncio.CancelledError(),
    ]

    with pytest.raises(asyncio.CancelledError):
        await client.graceful_step(
            "sid-1", AffectStimulus(valence=0.1, arousal=0.2), purge_on_interrupted=True
        )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"ok": True, "purged": True}, True),
        ({"ok": True, "purged": False}, False),
        ({"ok": True}, False),  # 缺键 → 不猜
        ({"ok": True, "purged": 1}, False),  # int 不是 bool → 不猜
    ],
)
async def test_purge_session_returns_purged_flag(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, object], expected: bool
) -> None:
    """`purge_session` 只如实转述 Zero 的 `purged`；缺键/非 bool → False（**不猜**）。

    ⚠ 该值语义是「删除通路可用」而非「确有数据被删」：Zero 侧 `purged` 只在 checkpointer
    没有 `adelete_thread` 时才为 False，对不存在的 thread 调它是 no-op 也照样回 True。
    把回落值写成 `bool(data.get("purged"))` 即红（`1` 会被读成 True）。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, json.dumps(payload))

    assert await client.purge_session("sid-1") is expected
    assert mock_session.call_tool.call_args.args[1] == {"session_id": "sid-1"}


async def test_purge_session_malformed_response_becomes_call_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """畸形响应（非 JSON / 非 object）统一封装成 `ZeroLinkCallError`，不让原始异常穿透边界。"""
    client, mock_session = _build_client_with_session(monkeypatch)

    _set_tool_return(mock_session, "not-json")
    with pytest.raises(ZeroLinkCallError):
        await client.purge_session("sid-1")

    _set_tool_return(mock_session, json.dumps([1, 2]))  # 合法 JSON 但不是 object
    with pytest.raises(ZeroLinkCallError):
        await client.purge_session("sid-1")


# ---------------------------------------------------------------------------
# Task（2026-07-30）：接 Zero 的**显式** `interrupt_probe` 四态
#
# 对方在 `667e923` 把「`interrupted_at` 缺席」拆成一个**恒存在**的 `interrupt_probe`：
#   not_probed / clean / interrupted / probe_failed
# 其中 `probe_failed`（对方探测**自己抛了**）正是本仓当初索要显式化的那一格 —— 此前它与
# 「探测干净」在返回体上完全同形，我方把它当安全 ⇒ 止血在最该生效时静默失效。
#
# 本段守卫的四条主线（与验收一一对应）：
#   ① 四态各自的处置分支都有用例，摘掉任一格的处置对应用例即红；
#   ② 老部署（无该键）回退 + **正控**证明新部署确实走了新分支（否则「不炸」是恒真的）；
#   ③ 未知第五态按最坏情况处置；
#   ④ 🛑 `probe_failed` **绝不能**被当 clean —— 专门的对照变异格。
#
# 🛑 **夹具不得只造一态**（pitfalls 新增条）：既有夹具（`_INTERRUPTED_OPEN_PAYLOAD` 等）一律
#    不带 `interrupt_probe` 键，若沿用它们，本段测的全是老轨、四态里另外几格一个都碰不到。
#    故下面每一态都用 `_probe_payload()` **显式构造**，且刻意让「载荷」与「令牌」可独立设置。
# ---------------------------------------------------------------------------


def _probe_payload(
    token: object,
    *,
    resumed: bool | None = True,
    nodes: object | None = None,
    with_nodes: bool = False,
    session_id: str = "sid-1",
) -> str:
    """构造**新部署**形态的 open_session 返回体（恒带 `interrupt_probe`）。

    `with_nodes=True` 时才写 `interrupted_at` 键（`nodes` 即其值）——Zero 只在
    `interrupted` 那一态发它，故「令牌 vs 载荷」必须能分别设置，否则自洽性那两格测不到。
    """
    payload: dict[str, object] = {"session_id": session_id, "interrupt_probe": token}
    if resumed is not None:
        payload["resumed"] = resumed
    if with_nodes:
        payload["interrupted_at"] = nodes
    return json.dumps(payload)


def _legacy_payload(*, resumed: bool | None = True, session_id: str = "sid-1") -> str:
    """构造**老部署**形态的返回体：**没有** `interrupt_probe` 键（走缺席推断老轨）。"""
    payload: dict[str, object] = {"session_id": session_id}
    if resumed is not None:
        payload["resumed"] = resumed
    return json.dumps(payload)


def _self_heal_calls(mock_session: AsyncMock) -> list[str]:
    """本次 graceful_step 实际发出的工具名序列（自愈路径的调用序列即行为判据）。"""
    return [c.args[0] for c in mock_session.call_tool.call_args_list]


async def _run_self_heal(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    open_payload: str,
    *,
    purge_on_interrupted: bool = False,
) -> tuple[ExpressionBundle | None, AsyncMock, ZeroLinkClient]:
    """跑一次 unknown-session 自愈：step 报未知 → 重开（给定返回体）→ 视判读决定是否重试。

    第三个返回体**故意备着**：实现若在该态下仍续跑，它会被消费掉 ⇒ 调用序列断言当场变红。
    也回 client 本体，供断言判读结果（`last_open_session`）而**不必**另造一遍流程。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    mock_session.call_tool.side_effect = [
        _make_call_result(_ZERO_UNKNOWN_SESSION_TEXT, is_error=True),
        _make_call_result(open_payload),
        _make_call_result(_make_expression_json()),
    ]
    with caplog.at_level(logging.DEBUG, logger="src.mcp.zero.client"):
        result = await client.graceful_step(
            "sid-1",
            AffectStimulus(valence=0.1, arousal=0.2),
            purge_on_interrupted=purge_on_interrupted,
        )
    return result, mock_session, client


# ── ① 解析层：四态逐格映射 + 原始令牌保留 ─────────────────────────────────────


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("not_probed", ZeroInterruptProbe.NOT_PROBED),
        ("clean", ZeroInterruptProbe.CLEAN),
        ("probe_failed", ZeroInterruptProbe.PROBE_FAILED),
        ("interrupted", ZeroInterruptProbe.INTERRUPTED),  # 令牌即判据，载荷缺席也算
    ],
)
async def test_explicit_probe_token_maps_to_state(
    monkeypatch: pytest.MonkeyPatch,
    token: str,
    expected: ZeroInterruptProbe,
) -> None:
    """🛑 对方的四个令牌**逐格**映射到我方四个态，且原始令牌原样留在 `interrupt_probe_raw`。

    删掉 `_ZERO_PROBE_TOKEN_TO_STATE` 里任一条目 → 该行落到 `UNRECOGNIZED`，本条即红。
    把判读改回「只看 `interrupted_at` 缺席」→ 四行全部塌缩成 `ABSENT`，四行齐红。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, _probe_payload(token))

    await client.open_session(session_id="sid-1")

    info = client.last_open_session
    assert info is not None
    assert info.interrupt_probe is expected
    assert info.interrupt_probe_raw == token, "原始令牌须原样保留（归因日志要打出来给人看）"


async def test_probe_failed_is_not_collapsed_into_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """🛑🛑 **本任务的核心判别式**：`probe_failed` 与 `clean` 在解析层必须判成**不同**的态。

    这一格就是我方向对方索要显式化的目标：改动前两者在返回体上同形（都不带 `interrupted_at`）
    ⇒ 我方只能把「探测失败」读成「可安全续跑」。若谁把 `"probe_failed"` 映到
    `ZeroInterruptProbe.CLEAN`（或让它落回 ABSENT 再按老口径续跑），本条当场红。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    states: dict[str, ZeroInterruptProbe] = {}
    for token in ("clean", "probe_failed"):
        _set_tool_return(mock_session, _probe_payload(token))
        await client.open_session(session_id="sid-1")
        assert client.last_open_session is not None
        states[token] = client.last_open_session.interrupt_probe

    assert states["probe_failed"] is not states["clean"], (
        f"probe_failed 与 clean 判成了同一个态（{states!r}）—— 那正是本轮要消灭的信息损失"
    )
    assert states["probe_failed"] is ZeroInterruptProbe.PROBE_FAILED
    assert states["clean"] is ZeroInterruptProbe.CLEAN


async def test_all_seven_states_are_reachable_and_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """判别性总闸：双轨七态**两两不同**，且每一态都有真实可达的返回体。

    夹具遮路的守卫（pitfalls 新增条）：若某态没有任何返回体能造出来，它的处置分支就永远
    不会被走到 —— 本条把「七态都可达」做成可执行断言，任意两格塌缩即红。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    cases: dict[ZeroInterruptProbe, str] = {
        # 新轨四态
        ZeroInterruptProbe.NOT_PROBED: _probe_payload("not_probed"),
        ZeroInterruptProbe.CLEAN: _probe_payload("clean"),
        ZeroInterruptProbe.PROBE_FAILED: _probe_payload("probe_failed"),
        ZeroInterruptProbe.INTERRUPTED: _probe_payload("interrupted"),
        ZeroInterruptProbe.UNRECOGNIZED: _probe_payload("probe_skipped_by_config"),
        # 老轨专属两态
        ZeroInterruptProbe.ABSENT: _legacy_payload(),
        ZeroInterruptProbe.MALFORMED: json.dumps(
            {"session_id": "sid-1", "resumed": True, "interrupted_at": 7}
        ),
    }
    seen: dict[ZeroInterruptProbe, ZeroInterruptProbe] = {}
    for expected, payload in cases.items():
        _set_tool_return(mock_session, payload)
        await client.open_session(session_id="sid-1")
        assert client.last_open_session is not None
        seen[expected] = client.last_open_session.interrupt_probe

    assert seen == {k: k for k in cases}, f"逐格映射错位：{seen!r}"
    assert len(set(seen.values())) == 7, f"七态必须两两可分，实得 {sorted(set(seen.values()))}"


# ── ③ 未知第五态 / 值形状坏 → UNRECOGNIZED（最坏情况 + 告警）───────────────────


async def test_unknown_fifth_state_becomes_unrecognized_with_raw_kept(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """对方按其 bump 纪律②新增第五态 → `UNRECOGNIZED` + 告警**点名该取值**，且**不炸**。

    三个断言各钉一面：不炸（open_session 照常返回 sid）、判成 UNRECOGNIZED（不当 clean）、
    原始令牌进日志（否则运维看不出对方新增了什么，只能去猜）。
    把未知取值兜到 `CLEAN`/`ABSENT` 即红；改成抛异常 → 第一个断言红（会话打不开）。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, _probe_payload("probe_skipped_by_config"))

    with caplog.at_level(logging.DEBUG, logger="src.mcp.zero.client"):
        sid = await client.open_session(session_id="sid-1")

    assert sid == "sid-1"
    info = client.last_open_session
    assert info is not None
    assert info.interrupt_probe is ZeroInterruptProbe.UNRECOGNIZED
    assert info.interrupt_probe_raw == "probe_skipped_by_config"
    warns = _warning_msgs(caplog)
    assert any("probe_skipped_by_config" in m for m in warns), (
        f"未知取值必须被点名打出来，实得 {caplog.records!r}"
    )
    assert len(_marked_records(caplog, LOG_MARKER_PROBE_UNRECOGNIZED_ON_OPEN, logging.WARNING)) >= 1


@pytest.mark.parametrize("bad", [7, None, True, {"state": "clean"}, ["clean"]])
async def test_malformed_probe_value_is_worst_case_not_clean(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, bad: object
) -> None:
    """`interrupt_probe` 的值**非 str** → `UNRECOGNIZED`（最坏情况），**绝不**当 clean、不炸。

    `True` 那一格特意留着：`isinstance(True, int)` 为真而不是 str，若实现用 `str(raw)` 兜底，
    它会变成 `"True"` 再落 UNRECOGNIZED —— 结论虽同，但 raw 会被污染成一个对方从未发过的
    字符串，故这里同时断言 `interrupt_probe_raw is None`（「值非 str」与「键缺席」由态区分）。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, _probe_payload(bad))

    with caplog.at_level(logging.DEBUG, logger="src.mcp.zero.client"):
        sid = await client.open_session(session_id="sid-1")

    assert sid == "sid-1"  # 形状坏不得让会话打不开
    info = client.last_open_session
    assert info is not None
    assert info.interrupt_probe is ZeroInterruptProbe.UNRECOGNIZED
    assert info.interrupt_probe_raw is None, "非 str 的取值不得被强转成字符串塞进 raw"
    assert len(_marked_records(caplog, LOG_MARKER_PROBE_UNRECOGNIZED_ON_OPEN, logging.WARNING)) >= 1


# ── 自洽性：正证据优先 ────────────────────────────────────────────────────────


async def test_nonempty_nodes_override_non_interrupted_token(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """令牌说 `clean` 却带**非空**待执行节点 → 取 `INTERRUPTED`（正证据优先）+ 矛盾告警。

    这一格今天对方不会发（其两处同源），但那是**对方的实现细节**。若反过来信令牌，我方就会
    在「明知有待执行节点」的情况下放行 —— 方向错了。把覆盖逻辑删掉即红（态变回 CLEAN）。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(
        mock_session, _probe_payload("clean", with_nodes=True, nodes=["affect_update"])
    )

    with caplog.at_level(logging.DEBUG, logger="src.mcp.zero.client"):
        await client.open_session(session_id="sid-1")

    info = client.last_open_session
    assert info is not None
    assert info.interrupt_probe is ZeroInterruptProbe.INTERRUPTED
    assert info.interrupted_at == ("affect_update",)
    assert len(_marked_records(caplog, LOG_MARKER_PROBE_STATE_MISMATCH, logging.WARNING)) == 1


async def test_interrupted_token_without_nodes_stays_interrupted(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """令牌说 `interrupted` 但载荷缺席 → **仍取 INTERRUPTED**（令牌是判据）+ 矛盾告警。

    反向的一格：若实现把「态」挂在载荷非空上（`if info.interrupted_at:`），这一格会被
    静默读成安全并照常续跑 —— 那正是「判据回退到载荷」的老病。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, _probe_payload("interrupted"))

    with caplog.at_level(logging.DEBUG, logger="src.mcp.zero.client"):
        await client.open_session(session_id="sid-1")

    info = client.last_open_session
    assert info is not None
    assert info.interrupt_probe is ZeroInterruptProbe.INTERRUPTED
    assert len(_marked_records(caplog, LOG_MARKER_PROBE_STATE_MISMATCH, logging.WARNING)) == 1
    # open 面那条 INTERRUPTED 告警也必须照发（最该响的时候不能没声）
    assert len(_marked_records(caplog, LOG_MARKER_INTERRUPTED_ON_OPEN, logging.WARNING)) == 1


# ── ② 老部署回退 + **正控**（证明新部署真走了新分支）──────────────────────────


async def test_legacy_deployment_without_probe_key_falls_back_to_old_track(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """老部署（无 `interrupt_probe`）→ 走缺席推断老轨，且**不打**任何新轨告警（零回归）。

    ⚠ 单独看本条**近乎恒真**（老部署路径本来就什么都不做）——判别力来自它与下一条正控
    构成的对照：同一个自愈流程，仅在返回体里**加上** `interrupt_probe: probe_failed`，
    行为就必须从「续跑」翻转成「拒绝」。缺了正控，「老部署不炸」这句话谁写都能过。
    """
    result, mock_session, client = await _run_self_heal(monkeypatch, caplog, _legacy_payload())

    assert isinstance(result, ExpressionBundle)  # 照常续跑（逐字等于改动前）
    assert _self_heal_calls(mock_session) == ["zero.step", "zero.open_session", "zero.step"]
    info = client.last_open_session
    assert info is not None
    assert info.interrupt_probe is ZeroInterruptProbe.ABSENT, "老部署须落老轨的 ABSENT"
    assert info.interrupt_probe_raw is None, "老部署不得凭空造出 raw 令牌"
    # 老轨的「不可判」告警照旧（resumed=True + 缺席）
    assert len(_marked_records(caplog, LOG_MARKER_PROBE_UNDECIDABLE, logging.WARNING)) == 1
    # 新轨的四个 marker 一个都不该出现
    for marker in (
        LOG_MARKER_PROBE_FAILED_ON_OPEN,
        LOG_MARKER_PROBE_FAILED_REFUSED,
        LOG_MARKER_PROBE_UNRECOGNIZED_ON_OPEN,
        LOG_MARKER_PROBE_NOT_PROBED_UNDECIDABLE,
    ):
        assert not any(marker in m for m in _warning_msgs(caplog) + _error_msgs(caplog)), (
            f"老部署不得触发新轨 marker {marker}"
        )


async def test_new_deployment_probe_failed_takes_the_new_branch(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """🛑 **上一条的正控**：同一自愈流程，返回体仅多一个 `interrupt_probe: probe_failed`
    ⇒ 行为**翻转**为拒绝本帧，且归因 marker 与老部署那条**不同**。

    这条同时兜住验收 ①（probe_failed 的处置分支）与 ④（不得当 clean）：
    · 摘掉 client 里 `PROBE_FAILED` 那个 `if` 分支 → 走到续跑，调用序列出现第三次 step，本条红；
    · 把 `"probe_failed"` 映射到 CLEAN → 同样续跑，本条红；
    · 归因断言还钉住「老部署不可判」与「新部署探测失败」**日志分得开**（两个不同 marker）。
    """
    result, mock_session, _client = await _run_self_heal(
        monkeypatch, caplog, _probe_payload("probe_failed")
    )

    assert result is None, "probe_failed 属不可判且故障相关 ⇒ 必须按最坏情况拒绝本帧"
    assert _self_heal_calls(mock_session) == ["zero.step", "zero.open_session"], (
        "拒绝本帧 = 重开后不得再发 step（第三个备好的返回体不该被消费）"
    )
    assert len(_marked_records(caplog, LOG_MARKER_PROBE_FAILED_REFUSED, logging.ERROR)) == 1
    # 归因分离：不得错打成「老部署不可判」
    assert not any(LOG_MARKER_PROBE_UNDECIDABLE in m for m in _warning_msgs(caplog)), (
        "新部署的探测失败不得复用老部署「不可判」的 marker —— 两者归因不同"
    )


async def test_probe_failed_and_clean_differ_in_behaviour(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """🛑🛑 验收④的对照变异格：`clean` 续跑、`probe_failed` 拒绝 —— **行为必须不同**。

    只要有人把 probe_failed「当 clean」（映射合并、分支删除、条件写成 `is not INTERRUPTED`
    就续跑…… 任一种），两侧的返回值与调用序列就会变得一样，本条当场红。
    """
    clean_result, clean_mock, _c1 = await _run_self_heal(
        monkeypatch, caplog, _probe_payload("clean")
    )
    failed_result, failed_mock, _c2 = await _run_self_heal(
        monkeypatch, caplog, _probe_payload("probe_failed")
    )

    assert isinstance(clean_result, ExpressionBundle), "明确干净 ⇒ 照常续跑"
    assert failed_result is None, "探测失败 ⇒ 拒绝本帧"
    assert _self_heal_calls(clean_mock) == ["zero.step", "zero.open_session", "zero.step"]
    assert _self_heal_calls(failed_mock) == ["zero.step", "zero.open_session"]
    assert _self_heal_calls(clean_mock) != _self_heal_calls(failed_mock), (
        "两态的调用序列相同 ⇒ probe_failed 被当成了 clean，这正是本轮要消灭的那一格"
    )


# ── ① 逐格处置：graceful_step 决策表 ─────────────────────────────────────────


async def test_unrecognized_state_refuses_frame(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """未知第五态在决策点同样**按最坏情况拒绝**，marker 与 probe_failed **分开**。

    摘掉 `UNRECOGNIZED` 那个 `if` → 落到「续跑」（第三次 step 被消费），本条红。
    marker 分开的理由：两者都拒帧，但归因不同（对方探测失败 vs 对方说了我方读不懂）。
    """
    result, mock_session, _client = await _run_self_heal(
        monkeypatch, caplog, _probe_payload("probe_skipped_by_config")
    )

    assert result is None
    assert _self_heal_calls(mock_session) == ["zero.step", "zero.open_session"]
    assert len(_marked_records(caplog, LOG_MARKER_PROBE_UNRECOGNIZED_REFUSED, logging.ERROR)) == 1
    assert not any(LOG_MARKER_PROBE_FAILED_REFUSED in m for m in _error_msgs(caplog))


async def test_not_probed_with_resumed_true_continues_with_own_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`not_probed` + `resumed=True`（活跃幂等重开）→ **照常续跑** + 自己的可区分 WARNING。

    为什么不拒绝（与 probe_failed 的取舍差别，见 client 分支注释）：本格不是故障相关的，
    它是对方按设计跳过探测（会话仍活跃），触发条件是并发/幂等重开这类正常控制流。
    摘掉这条 WARNING → 「对方明确没探测」与「对方说干净」在观测上重新塌缩，本条红；
    把它改成拒绝 → 第一个断言红（并发场景下的正常业务被打成丢帧）。
    """
    result, mock_session, _client = await _run_self_heal(
        monkeypatch, caplog, _probe_payload("not_probed")
    )

    assert isinstance(result, ExpressionBundle)
    assert _self_heal_calls(mock_session) == ["zero.step", "zero.open_session", "zero.step"]
    assert (
        len(_marked_records(caplog, LOG_MARKER_PROBE_NOT_PROBED_UNDECIDABLE, logging.WARNING)) == 1
    )
    # 归因分离：这不是老部署的「缺席不可判」
    assert not any(LOG_MARKER_PROBE_UNDECIDABLE in m for m in _warning_msgs(caplog))
    assert _error_msgs(caplog) == []


async def test_not_probed_with_resumed_false_is_silent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`not_probed` + `resumed=False`（新铸会话）→ 续跑且**不打**告警（真安全，没有旧运行态）。

    正控：没有这一格，上一条的 WARNING 可能是「只要 not_probed 就响」的无条件噪音。
    把 WARNING 条件里的 `info.resumed is True` 删掉即红。
    """
    result, _mock, _client = await _run_self_heal(
        monkeypatch, caplog, _probe_payload("not_probed", resumed=False)
    )

    assert isinstance(result, ExpressionBundle)
    assert not any(LOG_MARKER_PROBE_NOT_PROBED_UNDECIDABLE in m for m in _warning_msgs(caplog)), (
        f"新铸会话没有旧运行态可污染，不该告警，实得 {caplog.records!r}"
    )


async def test_clean_token_is_silent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`clean`（对方明确说干净）→ 续跑且不打任何不可判/拒绝日志（否则告警变噪音）。"""
    result, mock_session, _client = await _run_self_heal(
        monkeypatch, caplog, _probe_payload("clean")
    )

    assert isinstance(result, ExpressionBundle)
    assert _self_heal_calls(mock_session) == ["zero.step", "zero.open_session", "zero.step"]
    assert _error_msgs(caplog) == []
    for marker in (
        LOG_MARKER_PROBE_UNDECIDABLE,
        LOG_MARKER_PROBE_NOT_PROBED_UNDECIDABLE,
        LOG_MARKER_PROBE_UNRECOGNIZED_ON_OPEN,
        LOG_MARKER_PROBE_FAILED_ON_OPEN,
    ):
        assert not any(marker in m for m in _warning_msgs(caplog)), f"clean 不该触发 {marker}"


async def test_explicit_interrupted_token_refuses_frame(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """新轨 `interrupted` 令牌（**载荷也在**）→ 拒绝本帧，与老轨同一条处置。"""
    result, mock_session, _client = await _run_self_heal(
        monkeypatch,
        caplog,
        _probe_payload("interrupted", with_nodes=True, nodes=["affect_update"]),
    )

    assert result is None
    assert _self_heal_calls(mock_session) == ["zero.step", "zero.open_session"]
    assert len(_marked_records(caplog, LOG_MARKER_INTERRUPTED_REFUSED, logging.ERROR)) == 1
    assert any("affect_update" in m for m in _error_msgs(caplog))


# ── 破坏性动作的边界：不可判 ≠ 确定半截 ⇒ 不 purge ──────────────────────────


@pytest.mark.parametrize("token", ["probe_failed", "probe_skipped_by_config"])
async def test_undecidable_states_never_purge_even_when_opted_in(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, token: str
) -> None:
    """🛑 `purge_on_interrupted=True` 也**不得** purge 不可判的两格（判据是「读不出来」）。

    `purge_on_interrupted` 承诺的是「检出**半截**就清」；把它扩张成「不可判也清」= 让调用方
    在不知情下对一个可能完全健康的会话执行不可逆删除（一次探测抛异常完全可能出在干净会话上）。
    若谁把这两格并进 INTERRUPTED 分支，`zero.purge_session` 会出现在调用序列里，本条即红。
    """
    result, mock_session, _client = await _run_self_heal(
        monkeypatch, caplog, _probe_payload(token), purge_on_interrupted=True
    )

    assert result is None  # 仍拒绝本帧
    calls = _self_heal_calls(mock_session)
    assert "zero.purge_session" not in calls, f"不可判的一格不得 purge，实得调用序列 {calls}"
    assert calls == ["zero.step", "zero.open_session"]


async def test_interrupted_still_purges_when_opted_in(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """正控：**确定**半截那一格在 opt-in 下仍照旧 purge（上一条不能把 purge 能力整体废掉）。

    没有这一格，「不可判不 purge」可能是「任何情况都不 purge」的退化实现（purge 通路全死）。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    mock_session.call_tool.side_effect = [
        _make_call_result(_ZERO_UNKNOWN_SESSION_TEXT, is_error=True),
        _make_call_result(_probe_payload("interrupted", with_nodes=True, nodes=["render"])),
        _make_call_result(json.dumps({"ok": True, "purged": True})),
    ]

    with caplog.at_level(logging.DEBUG, logger="src.mcp.zero.client"):
        result = await client.graceful_step(
            "sid-1",
            AffectStimulus(valence=0.1, arousal=0.2),
            purge_on_interrupted=True,
        )

    assert result is None
    assert _self_heal_calls(mock_session) == [
        "zero.step",
        "zero.open_session",
        "zero.purge_session",
    ]
    assert len(_marked_records(caplog, LOG_MARKER_INTERRUPTED_REFUSED_PURGING, logging.ERROR)) == 1


# ── open 面（公开 open_session）的可观测性：probe_failed 不能只在自愈路径可见 ──


async def test_public_open_session_surfaces_probe_failed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """公开 `open_session()` 拿到 `probe_failed` 也要留一条 WARNING（该路径无守卫）。

    调用方若自己 resume 而不走 `graceful_step`，决策层那条 ERROR 压根不会发生 ⇒ 若 open 面
    也不报，「对方探测失败」在这条路径上**完全不可观测**。
    ⚠ marker 与决策层的 `*_REFUSED` **刻意不同名**：两者落在同一个 caplog 里，同名会让
    「决策层真拒绝了」的断言被本条日志喂成恒真（pitfalls ⑥ 同类）。
    """
    client, mock_session = _build_client_with_session(monkeypatch)
    _set_tool_return(mock_session, _probe_payload("probe_failed"))

    with caplog.at_level(logging.DEBUG, logger="src.mcp.zero.client"):
        sid = await client.open_session(session_id="sid-1")

    assert sid == "sid-1"  # open 面只报告、不处置（处置权在调用方）
    assert len(_marked_records(caplog, LOG_MARKER_PROBE_FAILED_ON_OPEN, logging.WARNING)) == 1
    assert not any(LOG_MARKER_PROBE_FAILED_REFUSED in m for m in _error_msgs(caplog)), (
        "open 面不得复用决策层的 marker，否则拒绝断言会被喂成恒真"
    )
