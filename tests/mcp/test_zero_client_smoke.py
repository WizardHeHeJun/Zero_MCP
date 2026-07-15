"""test_zero_client_smoke.py — ZeroLinkClient in-memory 集成测试（Task 6）。

**无 marker**：纯 in-memory FastMCP mock server，不依赖 D:\\Zero 亦不需真实桌面，
任何环境都应跑（故不标 zerorepo/realenv，避免 CI 的 `-m "not zerorepo"` 误剔除这些高价值集成用例）。

技术决策（现场核验 mcp 1.28.1）：
- FastMCP.tool(name="zero.open_session") 接受带点工具名，注册成功。
  结论：无需改契约，工具名可直接使用 "zero.*" 点分格式。
- create_connected_server_and_client_session 返回 AsyncGenerator[ClientSession, None]，
  即 async context manager，用 async with 进入取真 ClientSession。
- 测试策略：构造 FastMCP mock server，通过 create_connected_server_and_client_session
  取真 ClientSession，注入 client.session（绕 __aenter__ 传输层），跑完整工具链路。

覆盖：
1. open_session → step（带 priors）→ close_session 完整链：ExpressionBundle 正确解析。
2. external_priors 以 list-of-list 透传到 server 侧（无 tuple）。
3. server 返回 isError=True 时 client 抛 ZeroLinkCallError（真链路）。
"""

from __future__ import annotations

from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from src.agents.models.zero_affect import AffectStimulus, ExpressionBundle, ModalityPrior
from src.mcp.zero.client import ZeroLinkCallError, ZeroLinkClient

# ---------------------------------------------------------------------------
# FastMCP 点名工具核验注释
# ---------------------------------------------------------------------------
# 现场核验（mcp 1.28.1）：FastMCP.tool(name="zero.open_session") 接受带点工具名。
# 三个工具均以 "zero.*" 注册成功，无需改为下划线名或其他形式。
# 结论：工具名契约与 Zero server 侧对齐，无需 eng-lead 介入。
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# server 侧接收到的 external_priors 存储（供断言检查）
# ---------------------------------------------------------------------------
_received_step_args: dict[str, Any] = {}


def _make_mock_server() -> FastMCP:
    """构造带三工具的 FastMCP mock server。"""
    server = FastMCP("zero-mock-server")

    @server.tool(name="zero.open_session")
    def open_session_tool(persona: str | None = None, config: dict | None = None) -> dict:
        """mock zero.open_session → 返回固定 session_id。"""
        return {"session_id": "smoke-sid-1"}

    @server.tool(name="zero.step")
    def step_tool(
        session_id: str,
        stim: dict,
        external_priors: list | None = None,
    ) -> dict:
        """mock zero.step → 存储 external_priors 供断言，返回合法 expression dict。"""
        _received_step_args["session_id"] = session_id
        _received_step_args["external_priors"] = external_priors
        head = {
            "facs_au": {"AU12": 0.8, "AU06": 0.6, "intensity": 0.7},
            "text_label": "content",
            "physiology": {
                "heart_rate_bpm": 80.0,
                "skin_conductance": 0.5,
                "pupil_mm": 4.0,
            },
            "prosody": {"speech_rate": 1.0, "pitch": 1.0, "energy": 0.7},
        }
        return {
            "expression": {
                "valence_arousal": [
                    float(stim.get("valence", 0.3)),
                    float(stim.get("arousal", 0.5)),
                ],
                "spontaneous": head,
                "voluntary": head,
            }
        }

    @server.tool(name="zero.close_session")
    def close_session_tool(session_id: str) -> dict:
        """mock zero.close_session → 返回 ok。"""
        return {"ok": True}

    return server


def _make_error_server() -> FastMCP:
    """构造返回 isError=True 的 FastMCP mock server（覆盖场景 3）。"""
    from mcp.server.fastmcp.exceptions import ToolError

    server = FastMCP("zero-error-server")

    @server.tool(name="zero.open_session")
    def open_session_tool() -> dict:
        raise ToolError("mock server error: session rejected")

    return server


# ---------------------------------------------------------------------------
# 场景 1 & 2：完整链路 + external_priors list-of-list 透传
# ---------------------------------------------------------------------------


async def test_full_chain_open_step_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """open_session → step(带 priors) → close_session 完整链，
    ExpressionBundle 正确解析，external_priors 以 list-of-list 透传到 server。"""
    monkeypatch.setenv("ZERO_LINK_ENABLED", "true")
    _received_step_args.clear()

    mock_server = _make_mock_server()

    async with create_connected_server_and_client_session(mock_server) as real_session:
        # 绕 __aenter__ 传输层，直接注入真 ClientSession
        client = ZeroLinkClient()
        client.session = real_session

        # 1. open_session
        sid = await client.open_session()
        assert sid == "smoke-sid-1"

        # 2. step（带 priors）
        priors = [
            ModalityPrior(modality="vision", mu=(0.4, 0.3), precision=(0.2, 0.15)),
            ModalityPrior(modality="audio", mu=(-0.1, 0.6), precision=(0.1, 0.25)),
        ]
        stimulus = AffectStimulus(valence=0.3, arousal=0.5)
        bundle = await client.step(sid, stimulus, priors=priors)

        # ExpressionBundle 正确解析
        assert isinstance(bundle, ExpressionBundle)
        assert bundle.valence_arousal == pytest.approx((0.3, 0.5))
        assert bundle.spontaneous.text_label == "content"
        assert bundle.voluntary.text_label == "content"

        # 3. external_priors 在 server 侧为 list-of-list（无 tuple）
        ep = _received_step_args.get("external_priors")
        assert ep is not None, "server 侧应收到 external_priors"
        assert isinstance(ep, list)
        assert len(ep) == 2
        for item in ep:
            assert isinstance(item, list), f"每条应为 list，实际 {type(item)}"
            name, mu, precision = item
            assert isinstance(name, str)
            assert isinstance(mu, list)
            assert isinstance(precision, list)

        # 4. close_session（无异常即通过）
        await client.close_session(sid)


async def test_step_without_priors_no_external_priors_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """priors=None 时 server 侧收到的 external_priors 参数为 None（默认值）。"""
    monkeypatch.setenv("ZERO_LINK_ENABLED", "true")
    _received_step_args.clear()

    mock_server = _make_mock_server()

    async with create_connected_server_and_client_session(mock_server) as real_session:
        client = ZeroLinkClient()
        client.session = real_session

        sid = await client.open_session()
        stimulus = AffectStimulus(valence=0.1, arousal=0.2)
        bundle = await client.step(sid, stimulus, priors=None)

        assert isinstance(bundle, ExpressionBundle)
        # priors=None → client 不传 external_priors → server 侧 default=None
        ep = _received_step_args.get("external_priors")
        assert ep is None


# ---------------------------------------------------------------------------
# 场景 3：server 返回 isError=True → ZeroLinkCallError（真链路）
# ---------------------------------------------------------------------------


async def test_server_error_raises_zero_link_call_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """server 侧工具抛 ToolError（isError=True）时，client 抛 ZeroLinkCallError（真链路）。"""
    monkeypatch.setenv("ZERO_LINK_ENABLED", "true")

    error_server = _make_error_server()

    async with create_connected_server_and_client_session(error_server) as real_session:
        client = ZeroLinkClient()
        client.session = real_session

        with pytest.raises(ZeroLinkCallError) as exc_info:
            await client.open_session()

        assert exc_info.value.tool == "zero.open_session"
