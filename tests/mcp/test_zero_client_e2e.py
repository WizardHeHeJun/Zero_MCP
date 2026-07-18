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

import importlib.util
import os
import socket
import subprocess
import sys
import time
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
# 真 13-AU 权重（Zero artifacts；缺失/无 torch → 真权重用例 skip，占位路径不受影响）
_FACS_WEIGHT_V2 = Path("D:/Zero/artifacts/facs_decoder_ext_v2.pt")
_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_UVICORN_AVAILABLE = importlib.util.find_spec("uvicorn") is not None

# HTTP（streamable-http）联调 endpoint（对齐 Zero 回执 §一默认；server 子进程由测试起/拆）
_HTTP_HOST = "127.0.0.1"
_HTTP_PORT = 8000
_HTTP_PATH = "/mcp"
_HTTP_ENDPOINT = f"http://{_HTTP_HOST}:{_HTTP_PORT}{_HTTP_PATH}"


def _port_is_free(host: str, port: int) -> bool:
    """检测端口空闲（避免撞到其它进程占用的 8000 导致误连/flaky）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) != 0


def _wait_port(host: str, port: int, *, timeout: float = 40.0) -> bool:
    """轮询端口就绪（server 起来后接受连接）；超时返回 False。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.5)
    return False


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

    async def test_real_facs_weights_13_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """设 ZERO_FACS_MODEL_PATH → server 出**全 13 键 facs_au**（值∈[0,1]、python float）。

        对抗核验 Zero 的量纲更正（`2026-07-16-zero-followup-http-and-realweights.md` §三）：
        设 FACS 权重**不翻** prosody_scale——仍为 "ratio"（FACS 模型不覆盖 prosody，
        normalized 只在另接真 prosody 模型时出现）。torch/权重缺失即 skip。
        """
        if not _ZERO_SERVER.is_file():
            pytest.skip(f"D:\\Zero MCP server 不存在（{_ZERO_SERVER}）")
        if not _TORCH_AVAILABLE:
            pytest.skip("torch 未安装，跳过真 13-AU 权重路径")
        if not _FACS_WEIGHT_V2.is_file():
            pytest.skip(f"真 13-AU 权重缺失（{_FACS_WEIGHT_V2}）")
        _set_stdio_env(monkeypatch)
        monkeypatch.setenv("ZERO_FACS_MODEL_PATH", str(_FACS_WEIGHT_V2))
        monkeypatch.setenv("ZERO_FACS_EXTENDED", "true")

        client = ZeroLinkClient()
        try:
            await client.__aenter__()
        except ZeroLinkConnectionError as exc:
            pytest.skip(f"连不上 Zero server（真权重路径 import torch/加载失败？）：{exc}")
        try:
            sid = await client.open_session()
            bundle = await client.step(sid, AffectStimulus(valence=0.6, arousal=0.4))
            for head_name in ("spontaneous", "voluntary"):
                head = getattr(bundle, head_name)
                assert set(head.facs_au) == _VALID_FACS, (
                    f"{head_name}: 真权重应出全 13 键，实际 {sorted(head.facs_au)}"
                )
                for key, val in head.facs_au.items():
                    assert isinstance(val, float), (
                        f"{head_name}.facs_au[{key}] 非 float: {type(val)}"
                    )
                    assert 0.0 <= val <= 1.0, f"{head_name}.facs_au[{key}]={val} 超出 [0,1]"
            # 对抗核验 Zero 更正：FACS 权重不翻 prosody_scale（仍 ratio）
            assert bundle.prosody_scale == "ratio", (
                f"设 FACS 权重后 prosody_scale 应仍 'ratio'（Zero 更正：normalized 只在接真 "
                f"prosody 模型时出现），实际 {bundle.prosody_scale!r}"
            )
            await client.close_session(sid)
        finally:
            await client.__aexit__(None, None, None)

    async def test_http_transport_full_contract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """HTTP（streamable-http）传输端到端：测试起 Zero 真 http server（uvicorn 子进程），
        client 的 http 分支连 endpoint 跑全契约，验证与 stdio 逐字一致。

        对齐 Zero 回执 `2026-07-16-zero-followup-http-and-realweights.md` §一
        （`ZERO_MCP_TRANSPORT=http` + host/port/path）。server 缺失/uvicorn 缺失/端口占用/
        起不来 → skip（不拖红）。⚠ 本轮 HTTP 无鉴权，仅 127.0.0.1 本机联调。
        """
        if not _ZERO_SERVER.is_file():
            pytest.skip(f"D:\\Zero MCP server 不存在（{_ZERO_SERVER}）")
        if not _UVICORN_AVAILABLE:
            pytest.skip("uvicorn 未安装，跳过 HTTP 传输端到端")
        if not _port_is_free(_HTTP_HOST, _HTTP_PORT):
            pytest.skip(f"端口 {_HTTP_PORT} 被占用，跳过（避免误连/flaky）")

        # 起 Zero http server 子进程（env：ZERO_MCP_TRANSPORT=http + host/port/path）
        server_env = dict(os.environ)
        server_env.update(
            {
                "ZERO_MCP_TRANSPORT": "http",
                "ZERO_MCP_HTTP_HOST": _HTTP_HOST,
                "ZERO_MCP_HTTP_PORT": str(_HTTP_PORT),
                "ZERO_MCP_HTTP_PATH": _HTTP_PATH,
            }
        )
        proc = subprocess.Popen(
            [sys.executable, "-m", "src.mcp_server"],
            cwd=r"D:\Zero",
            env=server_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            if not _wait_port(_HTTP_HOST, _HTTP_PORT, timeout=40.0):
                pytest.skip(f"Zero http server 端口 {_HTTP_PORT} 未在 40s 内就绪，跳过")

            # client http 分支
            monkeypatch.setenv("ZERO_LINK_ENABLED", "true")
            monkeypatch.setenv("ZERO_LINK_TRANSPORT", "http")
            monkeypatch.setenv("ZERO_HTTP_ENDPOINT", _HTTP_ENDPOINT)
            monkeypatch.delenv("ZERO_HTTP_TOKEN", raising=False)  # 本轮 server 无鉴权

            client = ZeroLinkClient()
            try:
                await client.__aenter__()
            except ZeroLinkConnectionError as exc:
                pytest.skip(f"http client 连不上 {_HTTP_ENDPOINT}：{exc}")
            try:
                # 全契约（与 stdio 用例同结构，验证契约逐字一致）
                sid = await client.open_session(persona="http-e2e")
                assert isinstance(sid, str) and sid
                bundle = await client.step(sid, AffectStimulus(valence=0.6, arousal=0.4))
                assert isinstance(bundle, ExpressionBundle)
                _assert_valid_head(bundle.voluntary, "http.step裸.voluntary")
                assert bundle.prosody_scale in ("ratio", "normalized", None)
                # coping 路径
                b2 = await client.step(
                    sid, AffectStimulus(valence=-0.5, arousal=0.7, coping_potential=0.8)
                )
                _assert_valid_head(b2.voluntary, "http.step_coping.voluntary")
                # 未知 session → ZeroLinkCallError（HTTP 错误路径）
                with pytest.raises(ZeroLinkCallError):
                    await client.step("bogus-http-sid", AffectStimulus(valence=0.0, arousal=0.0))
                await client.close_session(sid)
            finally:
                await client.__aexit__(None, None, None)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
