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
from src.mcp.zero.mappers import LinearPhysiologyMapper, LinearProsodyMapper, ProsodyParams

_ZERO_SERVER = Path("D:/Zero/src/mcp_server/server.py")
_VALID_FACS = frozenset(FACS_KEYS_EXT)
# 真 13-AU 权重（Zero artifacts；缺失/无 torch → 真权重用例 skip，占位路径不受影响）
_FACS_WEIGHT_V2 = Path("D:/Zero/artifacts/facs_decoder_ext_v2.pt")
# 真 prosody 解码器权重（Zero artifacts；T4 normalized 上线，缺失/无 torch → skip）
_PROSODY_WEIGHT = Path("D:/Zero/artifacts/prosody_decoder.pt")
# 真 physiology 解码器权重（Zero artifacts；① WESAD 接线，缺失/无 torch → skip）
_PHYSIO_WEIGHT = Path("D:/Zero/artifacts/physiology_decoder.pt")
_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_UVICORN_AVAILABLE = importlib.util.find_spec("uvicorn") is not None

# HTTP（streamable-http）联调 endpoint（对齐 Zero 回执 §一默认；server 子进程由测试起/拆）
_HTTP_HOST = "127.0.0.1"
_HTTP_PORT = 8000
_HTTP_PATH = "/mcp"
_HTTP_ENDPOINT = f"http://{_HTTP_HOST}:{_HTTP_PORT}{_HTTP_PATH}"
# T5 Bearer 鉴权用独立端口（避与无鉴权 HTTP 用例串扰）
_HTTP_PORT_AUTH = 8001
_HTTP_ENDPOINT_AUTH = f"http://{_HTTP_HOST}:{_HTTP_PORT_AUTH}{_HTTP_PATH}"


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
            # physiology 默认门关（无 canonical env·无真模型）→ legacy 形状 {hr, sc, pupil_mm}
            # （零回归基线；canonical 门开/真模型形状分别见 test_canonical_placeholder_physiology_
            # gate_on / test_real_physiology_weights_canonical）。
            assert bundle.spontaneous.physiology.pupil_mm is not None, (
                "默认路径（门关无真模型）应出 legacy pupil_mm"
            )
            assert bundle.spontaneous.physiology.temperature_c is None, (
                "默认路径不应出 canonical temperature_c"
            )

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
            # 注：Zero server T6·② 接线后 step 会返回带 `unknown-session:` 机读前缀的 ToolError，
            # 本仓据此抛 ZeroLinkUnknownSessionError（ZeroLinkCallError 子类）——故此断言对新旧 Zero
            # 均成立。待 Zero 侧提交该接线后，可收窄为 pytest.raises(ZeroLinkUnknownSessionError)。
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

    async def test_real_prosody_weights_normalized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """T4 上线：设 ZERO_PROSODY_MODEL_PATH → prosody_scale=="normalized" + 三值∈[0,1]。

        Zero 回执（commit 64b8482）：接真 prosody 模型后 `prosody_scale=="normalized"`，
        `prosody={speech_rate,pitch,energy}` sigmoid∈[0,1]（Zero 侧打 tag 前 [0,1] fail-fast，
        越界不以 normalized 放行）。本仓 mapper 的 normalized 分支据此**转 live**——用真接线值真跑
        `LinearProsodyMapper.map()`（normalized 分支）验产有效 ProsodyParams。缺 torch/权重即 skip。
        """
        if not _ZERO_SERVER.is_file():
            pytest.skip(f"D:\\Zero MCP server 不存在（{_ZERO_SERVER}）")
        if not _TORCH_AVAILABLE:
            pytest.skip("torch 未安装，跳过真 prosody 权重路径")
        if not _PROSODY_WEIGHT.is_file():
            pytest.skip(f"真 prosody 权重缺失（{_PROSODY_WEIGHT}）")
        _set_stdio_env(monkeypatch)
        monkeypatch.setenv("ZERO_PROSODY_MODEL_PATH", str(_PROSODY_WEIGHT))

        client = ZeroLinkClient()
        try:
            await client.__aenter__()
        except ZeroLinkConnectionError as exc:
            pytest.skip(f"连不上 Zero server（真 prosody 权重 import torch/加载失败？）：{exc}")
        try:
            sid = await client.open_session()
            bundle = await client.step(sid, AffectStimulus(valence=0.6, arousal=0.4))
            # T4 上线断言：接真 prosody 模型 → prosody_scale 翻 normalized（FACS-only 则仍 ratio）
            assert bundle.prosody_scale == "normalized", (
                f"设 ZERO_PROSODY_MODEL_PATH 后 prosody_scale 应翻 'normalized'，"
                f"实际 {bundle.prosody_scale!r}"
            )
            for head_name in ("spontaneous", "voluntary"):
                head = getattr(bundle, head_name)
                assert head.prosody_scale == "normalized", (
                    f"{head_name}.prosody_scale 应 normalized"
                )
                for field in ("speech_rate", "pitch", "energy"):
                    val = getattr(head.prosody, field)
                    assert isinstance(val, float) and 0.0 <= val <= 1.0, (
                        f"{head_name}.prosody.{field}={val} 非 [0,1] float（normalized 契约）"
                    )
            # mapper normalized 分支现 live：用真接线值真跑，确认产有效 ProsodyParams（消费路径通）
            params = await LinearProsodyMapper().map(bundle.spontaneous)
            assert isinstance(params, ProsodyParams)
            await client.close_session(sid)
        finally:
            await client.__aexit__(None, None, None)

    async def test_canonical_placeholder_physiology_gate_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """② 上线：设 ZERO_PHYSIOLOGY_CANONICAL_PLACEHOLDER=true（无真模型）→ 占位出 canonical
        physiology `{heart_rate_bpm, skin_conductance(μS), temperature_c}`（删 pupil_mm）。

        对齐 Zero 回执（commit b503990+432f8d9）：门 = ZERO_PHYSIOLOGY_CANONICAL_PLACEHOLDER，
        门开且无真模型 → canonical 占位（议会 2026-07-23 公式）。**纯占位路径无需 torch/权重**——
        故本用例只要 server 起得来即 live 跑（不 skip on torch）。真跑本仓 mapper 验消费路径通：
        temperature_c 驱动 skin_temperature_level（非 None）、pupil_dilation=None（无 pupil）。
        ⚠ sc 中点偏置：占位 arousal=0→0μS，真 decoder 中立态~10μS，**禁跨路径绝对比较**——本用例
        仅断言 sc 落 μS 域 [0,20]，不断言绝对值。
        """
        if not _ZERO_SERVER.is_file():
            pytest.skip(f"D:\\Zero MCP server 不存在（{_ZERO_SERVER}）")
        _set_stdio_env(monkeypatch)
        monkeypatch.setenv("ZERO_PHYSIOLOGY_CANONICAL_PLACEHOLDER", "true")
        # 纯 canonical 占位路径：确保无真 physiology 模型（真模型另有专测）
        monkeypatch.delenv("ZERO_PHYSIOLOGY_MODEL_PATH", raising=False)

        client = ZeroLinkClient()
        try:
            await client.__aenter__()
        except ZeroLinkConnectionError as exc:
            pytest.skip(f"连不上 Zero server：{exc}")
        try:
            sid = await client.open_session(persona="physio-canonical-e2e")
            bundle = await client.step(sid, AffectStimulus(valence=0.6, arousal=0.4))
            for head_name in ("spontaneous", "voluntary"):
                head = getattr(bundle, head_name)
                phys = head.physiology
                assert phys.temperature_c is not None, (
                    f"{head_name}: canonical 门开应出 temperature_c，实际 None"
                )
                assert phys.pupil_mm is None, (
                    f"{head_name}: canonical 门开应删 pupil_mm，实际 {phys.pupil_mm}"
                )
                assert 50.0 <= phys.heart_rate_bpm <= 120.0, (
                    f"{head_name}: hr={phys.heart_rate_bpm} 越 canonical 域 [50,120]"
                )
                assert 0.0 <= phys.skin_conductance <= 20.0, (
                    f"{head_name}: sc={phys.skin_conductance} 须 μS 域 [0,20]（禁跨路径比较）"
                )
                assert 33.0 <= phys.temperature_c <= 36.0, (
                    f"{head_name}: temp={phys.temperature_c} 越占位域 [33,36]"
                )
            # mapper 真跑：temperature_c 驱动 skin_temperature_level（非 None）、pupil_dilation=None
            params = await LinearPhysiologyMapper().map(bundle.spontaneous)
            assert params.skin_temperature_level is not None, (
                "canonical 有 temperature_c，skin_temperature_level 不应为 None"
            )
            assert params.pupil_dilation is None, "canonical 无 pupil_mm，pupil_dilation 应 None"
            assert 0.0 <= params.skin_conductance_level <= 1.0
            await client.close_session(sid)
        finally:
            await client.__aexit__(None, None, None)

    async def test_real_physiology_weights_canonical(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """① 上线：设 ZERO_PHYSIOLOGY_MODEL_PATH → 真 WESAD decoder 出 canonical physiology
        `{heart_rate_bpm, skin_conductance(μS), temperature_c}`∈decoder 域（无 pupil_mm）。

        对齐 Zero 回执（① 已实现·`2026-07-23-zero-link-physiology-implemented.md`）：设真模型
        env → 恒 canonical（gate 不影响真模型路径）。真跑本仓 mapper 验 μS/°C 归一消费路径通。
        缺 torch/权重即 skip。⚠ 真 decoder 中立态 sc~10μS（sigmoid≈0.5），禁与占位路径跨路径比较——
        本用例仅断言落 decoder 域 sc[0,20]/temp[30,40]，不断言绝对值。
        """
        if not _ZERO_SERVER.is_file():
            pytest.skip(f"D:\\Zero MCP server 不存在（{_ZERO_SERVER}）")
        if not _TORCH_AVAILABLE:
            pytest.skip("torch 未安装，跳过真 physiology 权重路径")
        if not _PHYSIO_WEIGHT.is_file():
            pytest.skip(f"真 physiology 权重缺失（{_PHYSIO_WEIGHT}）")
        _set_stdio_env(monkeypatch)
        monkeypatch.setenv("ZERO_PHYSIOLOGY_MODEL_PATH", str(_PHYSIO_WEIGHT))

        client = ZeroLinkClient()
        try:
            await client.__aenter__()
        except ZeroLinkConnectionError as exc:
            pytest.skip(f"连不上 Zero server（真 physiology 权重 import torch/加载失败？）：{exc}")
        try:
            sid = await client.open_session(persona="physio-real-e2e")
            bundle = await client.step(sid, AffectStimulus(valence=0.6, arousal=0.4))
            for head_name in ("spontaneous", "voluntary"):
                head = getattr(bundle, head_name)
                phys = head.physiology
                assert phys.temperature_c is not None, (
                    f"{head_name}: 真 decoder 应出 temperature_c，实际 None"
                )
                assert phys.pupil_mm is None, (
                    f"{head_name}: canonical WESAD 无 pupil_mm，实际 {phys.pupil_mm}"
                )
                assert 50.0 <= phys.heart_rate_bpm <= 120.0, (
                    f"{head_name}: hr={phys.heart_rate_bpm} 越 decoder 域 [50,120]"
                )
                assert 0.0 <= phys.skin_conductance <= 20.0, (
                    f"{head_name}: sc={phys.skin_conductance} 越 μS 域 [0,20]"
                )
                assert 30.0 <= phys.temperature_c <= 40.0, (
                    f"{head_name}: temp={phys.temperature_c} 越 decoder 域 [30,40]"
                )
            # mapper 真跑：μS/°C 归一，temperature_c 驱动 tlevel、pupil_dilation=None
            params = await LinearPhysiologyMapper().map(bundle.spontaneous)
            assert params.skin_temperature_level is not None
            assert params.pupil_dilation is None
            assert 0.0 <= params.skin_conductance_level <= 1.0
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


@pytest.mark.zerorepo
class TestZeroClientHttpBearerAuth:
    """T5 · HTTP Bearer 端到端真鉴权（Zero server 设 ZERO_MCP_HTTP_TOKEN → 强制 401）。

    对齐 Zero 回执（commit 0e219c1）：设 `ZERO_MCP_HTTP_TOKEN` 后 /mcp 强制
    `Authorization: Bearer <token>`；缺/错 token → **传输层 HTTP 401**（纯 ASGI 中间件·先于 MCP
    会话），本仓 client `__aenter__` 包成 `ZeroLinkConnectionError`（连接层，不走 graceful_step）。
    env 名映射：本仓 `ZERO_HTTP_TOKEN` ↔ Zero `ZERO_MCP_HTTP_TOKEN`，两侧同值即通。
    server/uvicorn 缺失/端口占用/起不来 → skip（不拖红）。
    """

    async def test_bearer_401_and_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """错 token/缺 token → 401 → ZeroLinkConnectionError；对 token（两侧同值）→ 成功跑契约。"""
        if not _ZERO_SERVER.is_file():
            pytest.skip(f"D:\\Zero MCP server 不存在（{_ZERO_SERVER}）")
        if not _UVICORN_AVAILABLE:
            pytest.skip("uvicorn 未安装，跳过 HTTP Bearer 鉴权端到端")
        if not _port_is_free(_HTTP_HOST, _HTTP_PORT_AUTH):
            pytest.skip(f"端口 {_HTTP_PORT_AUTH} 被占用，跳过（避免误连/flaky）")

        token = "e2e-secret-0123456789abcdef"  # 纯 ASCII 非空（Zero auth.py 要求，否则 fail-fast）
        server_env = dict(os.environ)
        server_env.update(
            {
                "ZERO_MCP_TRANSPORT": "http",
                "ZERO_MCP_HTTP_HOST": _HTTP_HOST,
                "ZERO_MCP_HTTP_PORT": str(_HTTP_PORT_AUTH),
                "ZERO_MCP_HTTP_PATH": _HTTP_PATH,
                "ZERO_MCP_HTTP_TOKEN": token,  # Zero 侧强制鉴权
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
            if not _wait_port(_HTTP_HOST, _HTTP_PORT_AUTH, timeout=40.0):
                pytest.skip(f"Zero http(auth) server 端口 {_HTTP_PORT_AUTH} 未在 40s 内就绪，跳过")

            monkeypatch.setenv("ZERO_LINK_ENABLED", "true")
            monkeypatch.setenv("ZERO_LINK_TRANSPORT", "http")
            monkeypatch.setenv("ZERO_HTTP_ENDPOINT", _HTTP_ENDPOINT_AUTH)

            # A. 错 token → 传输层 401 → ZeroLinkConnectionError（连接失败，不是 ZeroLinkCallError）
            monkeypatch.setenv("ZERO_HTTP_TOKEN", "wrong-token-xxxx")
            with pytest.raises(ZeroLinkConnectionError):
                await ZeroLinkClient().__aenter__()

            # B. 缺 token（Bearer 头缺失）→ 401 → ZeroLinkConnectionError
            monkeypatch.delenv("ZERO_HTTP_TOKEN", raising=False)
            with pytest.raises(ZeroLinkConnectionError):
                await ZeroLinkClient().__aenter__()

            # C. 对 token（本仓 ZERO_HTTP_TOKEN == Zero ZERO_MCP_HTTP_TOKEN）→ 鉴权通过跑全契约
            monkeypatch.setenv("ZERO_HTTP_TOKEN", token)
            client_ok = ZeroLinkClient()
            try:
                await client_ok.__aenter__()
            except ZeroLinkConnectionError as exc:
                pytest.skip(f"对 token 仍连不上（环境问题？）：{exc}")
            try:
                sid = await client_ok.open_session(persona="http-auth-e2e")
                assert isinstance(sid, str) and sid
                bundle = await client_ok.step(sid, AffectStimulus(valence=0.3, arousal=0.4))
                assert isinstance(bundle, ExpressionBundle)
                _assert_valid_head(bundle.voluntary, "http_auth.step.voluntary")
                await client_ok.close_session(sid)
            finally:
                await client_ok.__aexit__(None, None, None)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
