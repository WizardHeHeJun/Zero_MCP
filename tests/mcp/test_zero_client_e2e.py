"""端到端联调测试：ZeroLinkClient 接 D:\\Zero 真 MCP server（`src.mcp_server`）跑全契约。

标记 `@pytest.mark.zerorepo`：需 D:\\Zero 的 `src/mcp_server` 在位，且**在共用 conda 环境
affective-expression 内跑**（server 子进程要 import Zero + mcp + langgraph）。
- D:\\Zero server 未建/缺失 → skip（不拖红）。
- 连接失败（如未在 affective-expression 环境、`sys.executable` 非该环境 python）→ skip 并提示。

server 启动方式对齐 Zero 回执（`notes/2026-07-16-zero-answers-client-ready.md` §二）：
`command`=conda env python（此处用 sys.executable）· `args=["-m","src.mcp_server"]` ·
`cwd=D:\\Zero`。会话门控 server 侧默认开（workspace/coping），故 external_priors/coping 生效。

覆盖（单会话一趟跑完，避免重复起 server 子进程）：
open → step(裸) → step(带 external_priors 含 physio 流) → step(带 coping) → 同会话跨轮累积
→ 未知 session（server ToolError→isError→ZeroLinkCallError）→ graceful_step 降级 None
→ close → close 后 step 失败。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.agents.models.zero_affect import (
    FACS_KEYS_EXT,
    TEXT_LABELS,
    AffectStimulus,
    ExpressionBundle,
    ModalityPrior,
)
from src.mcp.zero.client import ZeroLinkCallError, ZeroLinkClient, ZeroLinkConnectionError

_ZERO_SERVER = Path("D:/Zero/src/mcp_server/server.py")
_VALID_FACS = frozenset(FACS_KEYS_EXT)


def _set_stdio_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """设 stdio 联调 env：本测试进程即在 affective-expression 环境，故 sys.executable 即目标。"""
    monkeypatch.setenv("ZERO_LINK_ENABLED", "true")
    monkeypatch.setenv("ZERO_LINK_TRANSPORT", "stdio")
    monkeypatch.setenv("ZERO_SERVER_COMMAND", sys.executable)
    monkeypatch.setenv("ZERO_SERVER_ARGS", '["-m", "src.mcp_server"]')
    monkeypatch.setenv("ZERO_SERVER_CWD", r"D:\Zero")


def _assert_valid_head(head: object, ctx: str) -> None:
    """对抗核验单个 ExpressionHead：facs_au 键 ⊆ 全集、值 [0,1]、text_label 合法。"""
    facs_au: dict[str, float] = head.facs_au  # type: ignore[attr-defined]
    unknown = set(facs_au) - _VALID_FACS
    assert not unknown, f"{ctx}: facs_au 含未知键 {unknown}"
    for key, val in facs_au.items():
        assert 0.0 <= val <= 1.0, f"{ctx}: facs_au[{key}]={val} 超出 [0,1]"
    assert head.text_label in TEXT_LABELS, f"{ctx}: text_label={head.text_label!r} 不在 TEXT_LABELS"  # type: ignore[attr-defined]


@pytest.mark.zerorepo
class TestZeroClientE2E:
    """接 D:\\Zero 真 server 的端到端全契约回归。缺 server 或非目标环境 → skip。"""

    async def test_full_contract_against_real_server(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if not _ZERO_SERVER.is_file():
            pytest.skip(f"D:\\Zero MCP server 不存在（{_ZERO_SERVER}），跳过端到端联调")
        _set_stdio_env(monkeypatch)

        client = ZeroLinkClient()
        try:
            await client.__aenter__()
        except ZeroLinkConnectionError as exc:
            pytest.skip(
                f"连不上 Zero server（是否在 affective-expression 环境跑？"
                f"server 需 import langgraph/mcp）：{exc}"
            )

        try:
            # 1. open_session → 非空 session_id
            sid = await client.open_session(persona="e2e")
            assert isinstance(sid, str) and sid, "session_id 应为非空 str"

            # 2. step 裸（无 priors/coping）→ ExpressionBundle 逐字段核验
            bundle = await client.step(sid, AffectStimulus(valence=0.6, arousal=0.4))
            assert isinstance(bundle, ExpressionBundle)
            assert len(bundle.valence_arousal) == 2
            _assert_valid_head(bundle.spontaneous, "step裸.spontaneous")
            _assert_valid_head(bundle.voluntary, "step裸.voluntary")
            assert bundle.prosody_scale in ("ratio", "normalized", None)

            # 3. step 带 external_priors（含 physio eda/sc 流 → server M2 覆写 Πv）→ 被接受
            priors = [
                ModalityPrior(modality="vision", mu=(0.5, 0.3), precision=(0.20, 0.12)),
                ModalityPrior(modality="audio", mu=(-0.2, 0.6), precision=(0.10, 0.25)),
                ModalityPrior(modality="eda/sc", mu=(0.0, 0.7), precision=(0.001, 0.18)),
            ]
            bundle2 = await client.step(
                sid, AffectStimulus(valence=-0.3, arousal=0.5), priors=priors
            )
            assert isinstance(bundle2, ExpressionBundle)
            _assert_valid_head(bundle2.voluntary, "step带先验.voluntary")

            # 4. step 带 coping（显式值非 None → 保留）→ 解析正常
            bundle3 = await client.step(
                sid, AffectStimulus(valence=-0.5, arousal=0.7, coping_potential=0.8)
            )
            _assert_valid_head(bundle3.voluntary, "step带coping.voluntary")

            # 5. 同会话第 4 步（跨轮累积不报错）
            bundle4 = await client.step(sid, AffectStimulus(valence=0.2, arousal=0.1))
            assert len(bundle4.valence_arousal) == 2

            # 6. 未知 session_id → server ToolError → ZeroLinkCallError（tool 字段正确）
            with pytest.raises(ZeroLinkCallError) as exc_info:
                await client.step("bogus-sid-xyz", AffectStimulus(valence=0.0, arousal=0.0))
            assert exc_info.value.tool == "zero.step"

            # 7. graceful_step 未知 session → None（优雅降级）
            degraded = await client.graceful_step(
                "bogus-sid-xyz", AffectStimulus(valence=0.0, arousal=0.0)
            )
            assert degraded is None

            # 8. close_session → 无错
            await client.close_session(sid)

            # 9. close 后 step → ZeroLinkCallError（会话已释放）
            with pytest.raises(ZeroLinkCallError):
                await client.step(sid, AffectStimulus(valence=0.0, arousal=0.0))
        finally:
            await client.__aexit__(None, None, None)
