"""`zero.describe_config` 回读面接入的单测（AsyncMock 注入 session，不起真 server）。

覆盖矩阵（每一格都配**正控**，避免「对方没这工具时不报错」这类恒真式，pitfalls ⑥）：

探测四态
- OK：调通 + JSON object → `probe=OK`、版本/键集解析正确
- NOT_REGISTERED：调用失败 **且** `list_tools` 里确实没有该工具 → 优雅回退、**不抛**
  · 正控①：同样的调用失败，但 `list_tools` **有**该工具 → `CALL_FAILED`（≠ NOT_REGISTERED）
  · 正控②：`list_tools` 自己抛 → `CALL_FAILED`（问不到 ≠ 不在册）
  · 正控③：NOT_REGISTERED 会**负缓存**并短路后续 RTT；CALL_FAILED **不缓存**
- MALFORMED：非 JSON / JSON 非 object → 判 MALFORMED、不缓存

缓存
- 不传 sid（部署端默认）第二次调用**不发 RTT**
- 传 sid 且 `resolved_for_session=True` → 缓存；`False`（对方不认识该 id）→ **不缓存**
- 失效点：`open_session` / `close_session` / `purge_session` / `force_refresh` / `__aexit__`

① 错误码表运行期核对（一律 warn，不 raise）
- 一致 / 对方多 / 本仓多 / 两侧同时非空（疑似改名）/ 键缺席 / 回读面不可用

② 发流前自检
- schema 版本不一致 + 版本可信 + strict → **raise**（唯一的硬失败）
- 同上但 strict=False / 版本不认识 → 只 warn（在不可信观测量上不 raise）
- cap/max 与本机不同 → 只报告；这批 priors 按对方阈值会被拒 → 只报告
- **本机阈值 env 写坏 ≠ 会被拒**：没传 priors 时 `would_be_rejected` 仍为 False、
  原因落在 `local_env_error` 且点名 env；同时传会被拒的 priors 时两格各自置位（两通路独立）

版本演进
- 认识的版本正常用；不认识 → 降级但不炸 + `LOG_MARKER_DESCRIBE_VERSION_UNKNOWN`
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from src.agents.models.zero_affect import ModalityPrior
from src.mcp.zero.client import (
    DESCRIBE_CONFIG_EXPECTED_KEYS,
    DESCRIBE_CONFIG_OPTIONAL_KEYS,
    KNOWN_DESCRIBE_CONFIG_VERSIONS,
    LOG_MARKER_DESCRIBE_CALL_FAILED,
    LOG_MARKER_DESCRIBE_FIELDS_DRIFT,
    LOG_MARKER_DESCRIBE_NOT_REGISTERED,
    LOG_MARKER_DESCRIBE_VERSION_UNKNOWN,
    LOG_MARKER_ERROR_CODE_TABLE_DRIFT,
    LOG_MARKER_EXTERNAL_PRIOR_PREFLIGHT,
    ZERO_ERROR_CODES,
    ZeroConfigProbe,
    ZeroDeployConfig,
    ZeroLinkCallError,
    ZeroLinkClient,
    ZeroLinkDeployEnvError,
    ZeroLinkNonDegradableError,
    ZeroLinkSchemaIncompatibleError,
    check_external_prior_limits,
    diff_error_codes,
)
from src.mcp.zero.external_priors import (
    EXTERNAL_PRIOR_SCHEMA_VERSION,
    ZERO_EXTERNAL_PRIOR_PRECISION_CAP_DEFAULT,
    ZERO_MAX_EXTERNAL_STREAMS_DEFAULT,
)

_TOOL = "zero.describe_config"

# ---------------------------------------------------------------------------
# 夹具构造
# ---------------------------------------------------------------------------


def _payload(**overrides: Any) -> dict[str, Any]:
    """构造一份**形状完整**的 describe_config 返回体（21 键，逐键取自 Zero 真实实现）。

    默认值取 Zero 部署端默认；`overrides` 覆盖单键，`None` 值的键仍**保留**
    （Zero 明言不可知项显式回 null、不省略键）。删键请用 `_payload_without`。
    """
    data: dict[str, Any] = {
        "describe_config_version": 1,
        "session_id": None,
        "resolved_for_session": False,
        "workspace_enabled": True,
        "gate_fusion": False,
        "exclude_physio_fusion": True,
        "precision_commensurable": False,
        "ignition_beta": 1.0,
        "coping_potential_enabled": True,
        "text_coping_enabled": False,
        "fear_domain_enabled": False,
        "canonical_physiology": False,
        "facs_extended": True,
        "external_prior_precision_cap": ZERO_EXTERNAL_PRIOR_PRECISION_CAP_DEFAULT,
        "max_external_streams": ZERO_MAX_EXTERNAL_STREAMS_DEFAULT,
        "external_prior_schema_version": EXTERNAL_PRIOR_SCHEMA_VERSION,
        "governance_gated_flags": ["gate_fusion", "ignition_beta"],
        "error_codes": sorted(ZERO_ERROR_CODES),
        "sample_sigma_cap": None,
        "affect_readout": None,
        "weights_version": None,
    }
    data.update(overrides)
    return data


def _payload_without(*keys: str) -> dict[str, Any]:
    """删掉指定键的返回体（用于「缺键」判读）。"""
    data = _payload()
    for key in keys:
        data.pop(key, None)
    return data


def _ok_result(data: Any) -> CallToolResult:
    text = data if isinstance(data, str) else json.dumps(data)
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=False)


def _tools_result(*names: str) -> ListToolsResult:
    return ListToolsResult(tools=[Tool(name=n, inputSchema={"type": "object"}) for n in names])


def _client(monkeypatch: pytest.MonkeyPatch) -> tuple[ZeroLinkClient, AsyncMock]:
    monkeypatch.setenv("ZERO_LINK_ENABLED", "true")
    # 阈值 env 显式清掉：本文件多处比对「对方值 vs 本机默认」，留着外部 env 会让断言随环境漂移。
    monkeypatch.delenv("ZERO_EXTERNAL_PRIOR_PRECISION_CAP", raising=False)
    monkeypatch.delenv("ZERO_MAX_EXTERNAL_STREAMS", raising=False)
    session = AsyncMock()
    session.list_tools.return_value = _tools_result(_TOOL, "zero.step")
    client = ZeroLinkClient()
    client.session = session
    return client, session


def _serve(session: AsyncMock, data: Any) -> None:
    session.call_tool.return_value = _ok_result(data)


def _cfg_ok(**overrides: Any) -> ZeroDeployConfig:
    """直接构造一个 OK 态 `ZeroDeployConfig`（供纯判读函数单测，不经 I/O）。"""
    import types as _types

    data = _payload(**overrides)
    return ZeroDeployConfig(
        probe=ZeroConfigProbe.OK,
        version=data["describe_config_version"],
        resolved_for_session=data["resolved_for_session"] is True,
        fields=_types.MappingProxyType(data),
    )


# ---------------------------------------------------------------------------
# 探测：OK 态
# ---------------------------------------------------------------------------


async def test_describe_config_ok_parses_version_and_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session = _client(monkeypatch)
    _serve(session, _payload())

    cfg = await client.describe_config()

    assert cfg.probe is ZeroConfigProbe.OK
    assert cfg.available is True
    assert cfg.version == 1
    assert cfg.version_known is True
    assert cfg.enforceable is True
    assert cfg.missing_keys == ()
    assert cfg.unexpected_keys == ()
    session.call_tool.assert_awaited_once_with(_TOOL, {})


async def test_describe_config_passes_session_id_and_reads_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session = _client(monkeypatch)
    _serve(session, _payload(session_id="sid-1", resolved_for_session=True, gate_fusion=True))

    cfg = await client.describe_config(session_id="sid-1")

    assert cfg.resolved_for_session is True
    assert cfg.fields["gate_fusion"] is True
    session.call_tool.assert_awaited_once_with(_TOOL, {"session_id": "sid-1"})


async def test_resolved_for_session_only_accepts_real_bool(monkeypatch: pytest.MonkeyPatch) -> None:
    """`resolved_for_session` 只认真 bool：JSON 里写成 1/"true" 的伪真值一律按 False。"""
    client, session = _client(monkeypatch)
    _serve(session, _payload(session_id="sid-1", resolved_for_session=1))

    cfg = await client.describe_config(session_id="sid-1")

    assert cfg.resolved_for_session is False


# ---------------------------------------------------------------------------
# 探测：老部署（NOT_REGISTERED）+ 三个正控
# ---------------------------------------------------------------------------


def _fail_call(session: AsyncMock, exc: Exception | None = None) -> None:
    session.call_tool.side_effect = exc or ZeroLinkCallError(
        _TOOL, "Unknown tool: zero.describe_config"
    )


async def test_old_deployment_without_tool_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """④ 对方没有该工具 → 优雅回退：不抛、probe=NOT_REGISTERED、能力显式降级。"""
    client, session = _client(monkeypatch)
    _fail_call(session)
    session.list_tools.return_value = _tools_result("zero.open_session", "zero.step")

    with caplog.at_level(logging.WARNING):
        cfg = await client.describe_config()

    assert cfg.probe is ZeroConfigProbe.NOT_REGISTERED
    assert cfg.available is False
    assert cfg.enforceable is False
    assert LOG_MARKER_DESCRIBE_NOT_REGISTERED in caplog.text
    # 能力**显式降级**（而不是静默「检查通过」）：两条判读都回 checked=False + 原因串
    diff = diff_error_codes(cfg)
    assert diff.checked is False and diff.in_sync is False
    assert "不可用" in diff.reason
    report = check_external_prior_limits(cfg)
    assert report.checked is False


async def test_positive_control_tool_registered_is_call_failed_not_old_deployment(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """正控①：同样的调用失败，但工具**在册** → CALL_FAILED，**不得**被判成老部署。

    没有这一格，上一条就是恒真式（任何失败都回 NOT_REGISTERED 也能全绿）。
    """
    client, session = _client(monkeypatch)
    _fail_call(session)
    session.list_tools.return_value = _tools_result(_TOOL, "zero.step")

    with caplog.at_level(logging.WARNING):
        cfg = await client.describe_config()

    assert cfg.probe is ZeroConfigProbe.CALL_FAILED
    assert LOG_MARKER_DESCRIBE_CALL_FAILED in caplog.text
    assert LOG_MARKER_DESCRIBE_NOT_REGISTERED not in caplog.text
    assert "工具在册=True" in cfg.detail


async def test_positive_control_list_tools_failure_is_undecided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正控②：连能力面都问不到 → CALL_FAILED（问不到 ≠ 不在册），且**不下负缓存**。"""
    client, session = _client(monkeypatch)
    _fail_call(session)
    session.list_tools.side_effect = RuntimeError("transport gone")

    cfg = await client.describe_config()

    assert cfg.probe is ZeroConfigProbe.CALL_FAILED
    assert "工具在册=None" in cfg.detail
    assert client.describe_config_absent is None


async def test_positive_control_not_registered_is_cached_but_call_failed_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正控③：NOT_REGISTERED 负缓存短路后续 RTT；CALL_FAILED 每次都重探。"""
    client, session = _client(monkeypatch)
    _fail_call(session)
    session.list_tools.return_value = _tools_result("zero.step")

    await client.describe_config()
    await client.describe_config(session_id="sid-1")
    assert session.call_tool.await_count == 1, "负缓存未生效：老部署上仍在反复付 RTT"

    client2, session2 = _client(monkeypatch)
    _fail_call(session2)
    session2.list_tools.return_value = _tools_result(_TOOL)
    await client2.describe_config()
    await client2.describe_config()
    assert session2.call_tool.await_count == 2, "CALL_FAILED 被缓存了——不确定的事不该下定论"


async def test_non_degradable_error_from_probe_is_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """只读探测面即便撞上不可降级族错误也**不上抛**——它不该有权炸掉调用方。"""
    client, session = _client(monkeypatch)
    _fail_call(session, ZeroLinkDeployEnvError(_TOOL, "[zero:deploy-env-invalid] 坏 env"))
    session.list_tools.return_value = _tools_result(_TOOL)

    cfg = await client.describe_config()

    assert cfg.probe is ZeroConfigProbe.CALL_FAILED
    assert "ZeroLinkDeployEnvError" in cfg.detail


# ---------------------------------------------------------------------------
# 探测：MALFORMED
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "hint"),
    [("not json at all", "非合法 JSON"), (json.dumps([1, 2]), "不是 JSON object")],
)
async def test_malformed_response_is_flagged_not_crashed(
    monkeypatch: pytest.MonkeyPatch, body: str, hint: str
) -> None:
    client, session = _client(monkeypatch)
    _serve(session, body)

    cfg = await client.describe_config()

    assert cfg.probe is ZeroConfigProbe.MALFORMED
    assert hint in cfg.detail
    assert client.describe_config_cache == {}, "畸形返回体不该被缓存"


# ---------------------------------------------------------------------------
# 缓存与失效边界（⑤）
# ---------------------------------------------------------------------------


async def test_deployment_default_is_cached_second_call_has_no_rtt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⑤ 缓存真的生效：第二次调用**不再发 RTT**。"""
    client, session = _client(monkeypatch)
    _serve(session, _payload())

    first = await client.describe_config()
    second = await client.describe_config()

    assert session.call_tool.await_count == 1
    assert first is second


async def test_session_scoped_value_is_cached_only_when_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """会话级值：`resolved_for_session=True` 才缓存；False（对方不认识该 id）**不缓存**。

    后者若缓存 = 把「部署端默认伪装成会话值」的答案钉死在 sid 键上，等会话真开出来仍在服务它。
    """
    client, session = _client(monkeypatch)
    _serve(session, _payload(session_id="sid-1", resolved_for_session=True))
    await client.describe_config(session_id="sid-1")
    await client.describe_config(session_id="sid-1")
    assert session.call_tool.await_count == 1

    client2, session2 = _client(monkeypatch)
    _serve(session2, _payload(session_id="ghost", resolved_for_session=False))
    await client2.describe_config(session_id="ghost")
    await client2.describe_config(session_id="ghost")
    assert session2.call_tool.await_count == 2, "未知 id 的部署端默认被缓存到 sid 键下了"


async def test_open_session_invalidates_session_scoped_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """失效点：resume 重开会让 Zero 重建 config。

    SessionConfig 不进 checkpoint ⇒ 未再供 config 时回落 env 默认（本仓 R11 提过的那条）。
    """
    client, session = _client(monkeypatch)
    _serve(session, _payload(session_id="sid-1", resolved_for_session=True, gate_fusion=False))
    first = await client.describe_config(session_id="sid-1")
    assert first.fields["gate_fusion"] is False

    _serve(session, {"session_id": "sid-1", "resumed": True})
    await client.open_session(session_id="sid-1")

    _serve(session, _payload(session_id="sid-1", resolved_for_session=True, gate_fusion=True))
    second = await client.describe_config(session_id="sid-1")
    assert second.fields["gate_fusion"] is True, "resume 后仍在服务旧会话的门控"
    # 部署端默认那一条**不受影响**（它由 server 进程 env 决定，与会话无关）
    assert None not in client.describe_config_cache


@pytest.mark.parametrize("closer", ["close_session", "purge_session"])
async def test_close_and_purge_invalidate_session_scoped_cache(
    monkeypatch: pytest.MonkeyPatch, closer: str
) -> None:
    client, session = _client(monkeypatch)
    _serve(session, _payload(session_id="sid-1", resolved_for_session=True))
    await client.describe_config(session_id="sid-1")
    assert "sid-1" in client.describe_config_cache

    _serve(session, {"ok": True, "purged": True})
    await getattr(client, closer)("sid-1")

    assert "sid-1" not in client.describe_config_cache


async def test_force_refresh_bypasses_both_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    client, session = _client(monkeypatch)
    _fail_call(session)
    session.list_tools.return_value = _tools_result("zero.step")
    assert (await client.describe_config()).probe is ZeroConfigProbe.NOT_REGISTERED

    session.call_tool.side_effect = None
    _serve(session, _payload())
    cfg = await client.describe_config(force_refresh=True)

    assert cfg.probe is ZeroConfigProbe.OK, "force_refresh 没能清掉负缓存"


async def test_aexit_clears_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """缓存作用域 = 一次连接：下一次 `async with` 很可能是另一个 server 进程。"""
    client, session = _client(monkeypatch)
    _serve(session, _payload())
    await client.describe_config()
    assert client.describe_config_cache

    await client.__aexit__(None, None, None)

    assert client.describe_config_cache == {}
    assert client.describe_config_absent is None


# ---------------------------------------------------------------------------
# ① 错误码表运行期核对
# ---------------------------------------------------------------------------


def test_error_codes_in_sync() -> None:
    diff = diff_error_codes(_cfg_ok())
    assert diff.checked is True
    assert diff.in_sync is True
    assert diff.rename_suspected is False


async def test_zero_has_extra_code_warns_not_raises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """判别力①：对方码表多一个码 → warn（不炸），且原因串指出「切分会掏空既有归类」。"""
    client, session = _client(monkeypatch)
    _serve(session, _payload(error_codes=sorted({*ZERO_ERROR_CODES, "stim-invalid"})))

    with caplog.at_level(logging.WARNING):
        diff = await client.check_error_codes()

    assert diff.checked is True
    assert diff.zero_only == ("stim-invalid",)
    assert diff.client_only == ()
    assert diff.rename_suspected is False
    assert LOG_MARKER_ERROR_CODE_TABLE_DRIFT in caplog.text
    assert "对方多" in diff.reason and "掏空" in diff.reason


async def test_client_has_extra_code_warns_not_raises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """判别力②：对方少一个我方有的码 → warn，且文案点名「本仓拿着过期认知 → 归责降级」。"""
    dropped = "timeout-step"
    client, session = _client(monkeypatch)
    _serve(session, _payload(error_codes=sorted(ZERO_ERROR_CODES - {dropped})))

    with caplog.at_level(logging.WARNING):
        diff = await client.check_error_codes()

    assert diff.client_only == (dropped,)
    assert diff.zero_only == ()
    assert LOG_MARKER_ERROR_CODE_TABLE_DRIFT in caplog.text
    assert "过期认知" in diff.reason


def test_both_sides_nonempty_reads_as_rename() -> None:
    """码被**改名**时运行期只看得见「一删一增」——须作为联合信号呈现，别当两件事查。"""
    renamed = sorted((ZERO_ERROR_CODES - {"timeout-lock"}) | {"timeout-acquire"})
    diff = diff_error_codes(_cfg_ok(error_codes=renamed))

    assert diff.rename_suspected is True
    assert diff.zero_only == ("timeout-acquire",)
    assert diff.client_only == ("timeout-lock",)
    assert "疑似改名" in diff.reason


@pytest.mark.parametrize("bad", [None, "unknown-session", ["ok", 3]])
def test_error_codes_shape_drift_is_not_checked(bad: Any) -> None:
    """形状坏 → `checked=False`（整条丢弃，不逐元素过滤），**不是**「一致」。"""
    diff = diff_error_codes(_cfg_ok(error_codes=bad))
    assert diff.checked is False
    assert diff.in_sync is False
    assert "形状非 list[str]" in diff.reason


def test_error_codes_key_absent_is_not_checked() -> None:
    import types as _types

    cfg = ZeroDeployConfig(
        probe=ZeroConfigProbe.OK,
        version=1,
        fields=_types.MappingProxyType(_payload_without("error_codes")),
    )
    assert diff_error_codes(cfg).checked is False


def test_error_codes_still_compared_on_unknown_version() -> None:
    """版本不认识时**仍比对**（最强动作只是 warn，不比对等于主动丢信号），但结论串带存疑标注。"""
    unknown = max(KNOWN_DESCRIBE_CONFIG_VERSIONS) + 99
    diff = diff_error_codes(
        _cfg_ok(describe_config_version=unknown, error_codes=sorted({*ZERO_ERROR_CODES, "x-new"}))
    )
    assert diff.checked is True
    assert diff.zero_only == ("x-new",)
    assert "本仓不认识" in diff.reason


# ---------------------------------------------------------------------------
# ② 发流前自检
# ---------------------------------------------------------------------------


async def test_preflight_passes_on_matching_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    client, session = _client(monkeypatch)
    _serve(session, _payload())

    report = await client.preflight_external_priors()

    assert report.checked is True
    assert report.schema_mismatch is False
    assert report.limits_differ is False
    assert report.would_be_rejected is False
    assert "自检通过" in report.reason


async def test_schema_version_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """判别力③：schema 版本不一致 = **契约不兼容** → 上抛（与 ① 的 warn 处置有意不同）。"""
    client, session = _client(monkeypatch)
    _serve(session, _payload(external_prior_schema_version=EXTERNAL_PRIOR_SCHEMA_VERSION + 1))

    with pytest.raises(ZeroLinkSchemaIncompatibleError) as exc_info:
        await client.preflight_external_priors()

    assert "契约不兼容" in str(exc_info.value)
    assert isinstance(exc_info.value, ZeroLinkNonDegradableError), (
        "须属不可降级族，否则 graceful_step 的 except 元组会把它吞成静默 None"
    )


async def test_schema_mismatch_reports_without_raising_when_not_strict(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client, session = _client(monkeypatch)
    _serve(session, _payload(external_prior_schema_version=EXTERNAL_PRIOR_SCHEMA_VERSION + 1))

    with caplog.at_level(logging.WARNING):
        report = await client.preflight_external_priors(strict=False)

    assert report.schema_mismatch is True
    assert LOG_MARKER_EXTERNAL_PRIOR_PREFLIGHT in caplog.text


async def test_schema_mismatch_on_unknown_version_warns_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """版本不认识 ⇒ 我方对该键的读法本身不可信 ⇒ **不在不可信观测量上 raise**，降级 warn。"""
    unknown = max(KNOWN_DESCRIBE_CONFIG_VERSIONS) + 99
    client, session = _client(monkeypatch)
    _serve(
        session,
        _payload(
            describe_config_version=unknown,
            external_prior_schema_version=EXTERNAL_PRIOR_SCHEMA_VERSION + 1,
        ),
    )

    with caplog.at_level(logging.WARNING):
        report = await client.preflight_external_priors()  # strict 默认 True 也不抛

    assert report.schema_mismatch is True
    assert report.version_known is False
    assert LOG_MARKER_DESCRIBE_VERSION_UNKNOWN in caplog.text
    assert "降级为告警不上抛" in report.reason


async def test_unreadable_schema_version_is_not_a_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """对方那一位读不到（缺键）时**不算不一致**——在读不到的位上硬失败等于把老部署一律判死。"""
    client, session = _client(monkeypatch)
    _serve(session, _payload_without("external_prior_schema_version"))

    report = await client.preflight_external_priors()

    assert report.zero_schema_version is None
    assert report.schema_mismatch is False
    assert "不可读" in report.reason


async def test_limit_drift_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """阈值不同不是错误（对方更严我方就该按对方来）→ 只报告。"""
    client, session = _client(monkeypatch)
    _serve(session, _payload(external_prior_precision_cap=0.4, max_external_streams=2))

    with caplog.at_level(logging.WARNING):
        report = await client.preflight_external_priors()

    assert report.limits_differ is True
    assert report.zero_precision_cap == 0.4
    assert report.client_precision_cap == ZERO_EXTERNAL_PRIOR_PRECISION_CAP_DEFAULT
    assert LOG_MARKER_EXTERNAL_PRIOR_PREFLIGHT in caplog.text


async def test_preflight_detects_priors_that_zero_would_reject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """撞 ExternalPriorError 之前就知道：按**对方的** max_streams 干跑本仓校验。"""
    client, session = _client(monkeypatch)
    _serve(session, _payload(max_external_streams=1))
    priors = [
        ModalityPrior(modality="vision", mu=(0.5, 0.3), precision=(0.2, 0.12)),
        ModalityPrior(modality="audio", mu=(-0.2, 0.6), precision=(0.1, 0.25)),
    ]

    report = await client.preflight_external_priors(priors)

    assert report.would_be_rejected is True
    assert "M6" in report.rejection
    assert "会被拒" in report.reason


async def test_preflight_accepts_priors_within_zero_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正控：同一批 priors 在对方阈值放宽后**不再**被判会拒（否则上一条是恒真式）。"""
    client, session = _client(monkeypatch)
    _serve(session, _payload(max_external_streams=5))
    priors = [
        ModalityPrior(modality="vision", mu=(0.5, 0.3), precision=(0.2, 0.12)),
        ModalityPrior(modality="audio", mu=(-0.2, 0.6), precision=(0.1, 0.25)),
    ]

    report = await client.preflight_external_priors(priors)

    assert report.would_be_rejected is False
    assert report.rejection == ""


# --- 本机 env 坏掉 ≠ 这批 priors 会被拒（2026-07-30 审查 BLOCK 修）------------------


_BROKEN_CAP_ENV = "ZERO_EXTERNAL_PRIOR_PRECISION_CAP"


def _two_streams() -> list[ModalityPrior]:
    """两条非 physio 流（不会被 merge_physio 合并）—— 对方 max_streams=1 时必触发 M6。"""
    return [
        ModalityPrior(modality="vision", mu=(0.5, 0.3), precision=(0.2, 0.12)),
        ModalityPrior(modality="audio", mu=(-0.2, 0.6), precision=(0.1, 0.25)),
    ]


async def test_broken_local_env_does_not_masquerade_as_rejection(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """🛑 本机阈值 env 写坏 ≠ 「这批 priors 会被拒」——两件无关的事不得共用一格。

    此前二者混用 `rejection`：本机 env 一坏，**一条 priors 都没传**时 `would_be_rejected`
    也变 True，与 `ZeroExternalPriorPreflight` 自己的契约（「空串 = 不会被拒或没传 priors」）
    直接矛盾；调用方据此放弃发流 = 本机一个配置笔误把自己关在门外。
    此前漏测的成因是夹具 `_client` 显式清空了这两个 env，故本条**显式设一个非法值**。
    """
    client, session = _client(monkeypatch)
    monkeypatch.setenv(_BROKEN_CAP_ENV, "abc")  # 非法：无法解析为 float
    _serve(session, _payload())

    with caplog.at_level(logging.WARNING):
        report = await client.preflight_external_priors()  # 注意：**没传 priors**

    assert report.checked is True
    assert report.would_be_rejected is False, "没传 priors 却被判「会被拒」——两格混用的原病灶"
    assert report.rejection == ""
    assert report.local_env_error is not None, "本机 env 坏了却没有任何一格记下来 = 信息丢了"
    assert _BROKEN_CAP_ENV in report.local_env_error, (
        f"原因串没点名是哪个 env（实得 {report.local_env_error!r}），排障时等于没说"
    )
    # 读不到就是读不到：不许拿本仓默认值冒充「本机生效值」，也不许据此宣称阈值不同
    assert report.client_precision_cap is None
    assert report.client_max_streams is None
    assert report.limits_differ is False, "一侧不可读时不得判成「阈值不同」"
    # 不许宣称通过，且仍留在告警面（拆格前它混在 rejection 里是能见的，拆格不能把它降噪）
    assert "自检通过" not in report.reason
    assert LOG_MARKER_EXTERNAL_PRIOR_PREFLIGHT in caplog.text


async def test_broken_local_env_still_detects_priors_that_zero_would_reject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """判别力正控：本机 env 坏着，「对方会拒这批 priors」这条通路仍须照常判出。

    证明拆分后两条通路各自独立：干跑显式传对方的 cap/max，`_resolve_*` 拿到显式值就直接返回、
    根本不读 env ⇒ 本机 env 坏不坏与这条判读无关。若把拆分做成「env 一坏整条短路」，
    本条即红——那正是旧实现的行为（`if not rejection_reason and priors ...`）。
    """
    client, session = _client(monkeypatch)
    monkeypatch.setenv(_BROKEN_CAP_ENV, "abc")
    _serve(session, _payload(max_external_streams=1))

    report = await client.preflight_external_priors(_two_streams())

    assert report.local_env_error is not None, "前提没成立：本机 env 未被判非法，本条失去意义"
    assert report.would_be_rejected is True, "本机 env 坏掉把对方阈值这条真信号一起丢了"
    assert "M6" in report.rejection, f"拒绝原因不是 M6 流数超限：{report.rejection!r}"
    assert "会被拒" in report.reason and _BROKEN_CAP_ENV in report.reason, (
        "两件事都发生时，结论串须同时说清（一格一句），不是二选一"
    )


async def test_healthy_local_env_leaves_env_slot_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """正控：本机 env 正常时 `local_env_error` 必须是 None（否则新字段是恒真的噪音格）。"""
    client, session = _client(monkeypatch)
    monkeypatch.setenv(_BROKEN_CAP_ENV, "0.8")  # 合法值：显式设而非删，钉住「能解析就没事」
    _serve(session, _payload())

    report = await client.preflight_external_priors()

    assert report.local_env_error is None
    assert report.client_precision_cap == 0.8
    assert "自检通过" in report.reason


async def test_preflight_on_old_deployment_neither_raises_nor_claims_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """老部署：不抛、也**不宣称通过** —— checked=False 才是诚实回报。"""
    client, session = _client(monkeypatch)
    _fail_call(session)
    session.list_tools.return_value = _tools_result("zero.step")

    report = await client.preflight_external_priors(strict=True)

    assert report.checked is False
    assert report.would_be_rejected is False
    assert "不可用" in report.reason


# ---------------------------------------------------------------------------
# 版本与键集形状
# ---------------------------------------------------------------------------


async def test_unknown_version_degrades_but_does_not_crash(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    unknown = max(KNOWN_DESCRIBE_CONFIG_VERSIONS) + 99
    client, session = _client(monkeypatch)
    _serve(session, _payload(describe_config_version=unknown))

    with caplog.at_level(logging.WARNING):
        cfg = await client.describe_config()

    assert cfg.available is True, "版本不认识**不**等于不可用——代际判别与版本取值无关"
    assert cfg.version_known is False
    assert cfg.enforceable is False
    assert LOG_MARKER_DESCRIBE_VERSION_UNKNOWN in caplog.text


@pytest.mark.parametrize("bad_version", ["1", None, True, 1.5])
async def test_non_int_version_reads_as_unknown(
    monkeypatch: pytest.MonkeyPatch, bad_version: Any
) -> None:
    """版本位形状坏 → 按「不认识」处置（`True` 也不行：bool 是 int 子类，须显式排除）。"""
    client, session = _client(monkeypatch)
    _serve(session, _payload(describe_config_version=bad_version))

    cfg = await client.describe_config()

    assert cfg.version_known is False
    assert cfg.available is True


async def test_missing_and_extra_keys_are_reported(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client, session = _client(monkeypatch)
    data = _payload_without("affect_readout")
    data["brand_new_gate"] = True
    _serve(session, data)

    with caplog.at_level(logging.WARNING):
        cfg = await client.describe_config()

    assert cfg.missing_keys == ("affect_readout",)
    assert cfg.unexpected_keys == ("brand_new_gate",)
    assert LOG_MARKER_DESCRIBE_FIELDS_DRIFT in caplog.text
    assert cfg.available is True, "键集漂移不该让整个回读面不可用"


def test_expected_key_set_is_21_keys() -> None:
    """本仓**必需**键集的规模 pin —— 与 Zero 真实返回体的一致性由 zerorepo 跨仓守卫负责。

    可选键（对方 v3 起新增）**不计入**这个数：混进来就分不清「必需集被人改了」与
    「对方又加了一个新版本键」，而两者的处置完全不同。
    """
    assert len(DESCRIBE_CONFIG_EXPECTED_KEYS) == 21
    assert set(_payload()) == DESCRIBE_CONFIG_EXPECTED_KEYS
    # 两个集合不得相交：同一个键既必需又可选是自相矛盾的登记，会让 missing/absent 双报。
    assert not (DESCRIBE_CONFIG_EXPECTED_KEYS & DESCRIBE_CONFIG_OPTIONAL_KEYS)


def test_optional_keys_are_absent_from_v1_payload_without_being_reported_missing() -> None:
    """可选键缺席**不算缺**——这是分层的全部意义所在（v1/v2 部署只有 21 键）。

    判别力：若把可选键并进必需集，本用例第二条断言立刻红（missing 会变成那两个键）。
    """
    cfg = _cfg_ok()  # v1 形态载荷：21 键，7 个可选键一个都没有
    assert cfg.available is True
    assert cfg.missing_keys == ()  # 必需键齐全 ⇒ 不报缺
    assert cfg.absent_optional_keys == (  # 但如实记账（v3 两键 + v4 三键 + v5 两键）
        "checkpointer_impl",
        "memory_store_impl",
        "motion_backend",
        "motion_enabled",
        "semantic_store_impl",
        "stateless_http",
        "transport",
    )
    assert cfg.unexpected_keys == ()  # 且不会反过来被当成「多了键」


def test_v3_payload_reports_neither_missing_nor_unexpected() -> None:
    """v3 形态（23 键）**必需/多余**两个方向都不报 —— 对方正常演进不该触发告警。

    ⚠ `absent_optional_keys` 这一栏**有意非空**：v3 确实没有 v4 才引入的后端回读三键，
    如实记账正是分层的职责（「对方版本旧」≠「契约漂移」）。把它断言成空等于要求
    v3 部署长出 v4 的键。
    """
    cfg = _cfg_ok(transport="stdio", stateless_http=False)
    assert cfg.missing_keys == ()
    assert cfg.absent_optional_keys == (
        "checkpointer_impl",
        "memory_store_impl",
        "motion_backend",
        "motion_enabled",
        "semantic_store_impl",
    )
    assert cfg.unexpected_keys == ()


def test_v4_payload_reports_only_v5_keys_absent() -> None:
    """v4 形态（26 键）只缺 v5 的动作通道两键——分层按代际逐档记账，不是「有/没有」两态。

    值取对方真实语义：不传 sid 时后端三键回 `null`（不可知），`semantic_store_impl`
    另有 `"disabled"` 态；本用例只管键集，值语义的判别见
    `test_semantic_store_disabled_is_distinguishable_from_unknown`。
    """
    cfg = _cfg_ok(
        transport="stdio",
        stateless_http=False,
        checkpointer_impl=None,
        memory_store_impl=None,
        semantic_store_impl=None,
    )
    assert cfg.missing_keys == ()
    assert cfg.absent_optional_keys == ("motion_backend", "motion_enabled")
    assert cfg.unexpected_keys == ()


def test_v5_payload_reports_nothing_absent() -> None:
    """v5 形态（28 键，= 今天所连部署 Zero `e70787a`）三个方向全空 —— 已认全对方现役键集。

    `motion_enabled` 取 `False`（对方默认关）：本键的价值恰在「关」这一态——它是我方将来
    接动作通道时的**调用前判据**，免得靠先调一次吃 `[zero:motion-disabled]` 来探能力
    （那等于把正常分支建成错误分支）。
    """
    cfg = _cfg_ok(
        transport="stdio",
        stateless_http=False,
        checkpointer_impl=None,
        memory_store_impl=None,
        semantic_store_impl=None,
        motion_enabled=False,
        motion_backend="synth",
    )
    assert cfg.missing_keys == ()
    assert cfg.absent_optional_keys == ()
    assert cfg.unexpected_keys == ()
    # 值原样透传可读（OPTIONAL 分层只影响记账，不吞值）——将来接动作通道即读这一位。
    assert cfg.fields.get("motion_enabled") is False
    assert cfg.fields.get("motion_backend") == "synth"


def test_semantic_store_disabled_is_distinguishable_from_unknown() -> None:
    """`semantic_store_impl` 三态可分：`null`(不可知) / `"disabled"`(关) / 类名(开)。

    🛑 这一格盯的是**判别力本身**：Zero 2026-08-11 回件 §3.2 明确「『关闭』与『不可知』
    若都回 null 就不可区分」。我方任何把这两态并成一态的读法（如 `or ""`、`if not v`）
    都会让「语义后端到底开没开」这个事实在消费侧丢失。
    """
    unknown = _cfg_ok(semantic_store_impl=None)
    disabled = _cfg_ok(semantic_store_impl="disabled")
    enabled = _cfg_ok(semantic_store_impl="SqliteVecSemanticStore")
    values = [
        unknown.fields.get("semantic_store_impl"),
        disabled.fields.get("semantic_store_impl"),
        enabled.fields.get("semantic_store_impl"),
    ]
    assert values == [None, "disabled", "SqliteVecSemanticStore"]
    assert len({repr(v) for v in values}) == 3, "三态必须两两可分"


def test_genuinely_unknown_key_still_surfaces_as_unexpected() -> None:
    """正控：分层**没有**削弱守卫——我方尚不知道的键仍落进 unexpected。

    这一格是分层设计的兜底证明：若 `unexpected_keys` 被写成「减去所有见过的键」之类的宽判据，
    对方加第 24 个键时我方就再也不会被提醒「漏读了新能力」。
    """
    cfg = _cfg_ok(transport="stdio", stateless_http=False, brand_new=1)
    assert cfg.unexpected_keys == ("brand_new",)
    assert cfg.missing_keys == ()
