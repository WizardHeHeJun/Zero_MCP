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
from src.mcp.zero.client import (
    ZeroLinkCallError,
    ZeroLinkClient,
    ZeroLinkConnectionError,
    generate_session_id,
)
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
# aiosqlite 是 Zero sqlite checkpointer（AsyncSqliteSaver）的连接依赖：缺它 Zero 会静默回退
# InMemorySaver → 跨重启不恢复态 → resume transparency 断言会**硬红而非 skip**（易误判 flaky）。
# server 子进程与本测试进程同环境（sys.executable），故本进程 find_spec 即 server 侧可用性代理。
_AIOSQLITE_AVAILABLE = importlib.util.find_spec("aiosqlite") is not None

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


# ── T6 跨重启 resume 观测量（探针实证：见 notes/2026-07-24-zero-link-resume-e2e-probe.md）──
# affect_readout="map"（确定性后验均值读出，无采样）下，重复强刺激的 valence_arousal 随会话历史
# **单调漂移**（arousal 逐步降）：实测 step1 arousal≈0.5967 → step5≈0.5885（Δ≈8.2e-3）。
# 驱动是 Zero `value_table`（ValueAgent TD 在线学习，键恒 "mcp-step"）跨轮经 checkpoint 累积
# ——**非 mood**（MCP 默认门控 mood 关、MoodAgent no-op；对抗核验 wf_f94b368f-cd3 订正此归因）。
# 故「连续第 N 步」≠「全新第 1 步」，可判别 resume 是否真从 checkpoint 恢复运行态。
_RESUME_STIMULUS = AffectStimulus(valence=0.9, arousal=0.8)
_RESUME_CONFIG: dict[str, object] = {"affect_readout": "map"}
_RESUME_PRE_STEPS = 4  # 重启前步数；比较第 _RESUME_PRE_STEPS+1 步（实测漂移在此已达 8.2e-3）
_RESUME_TRANSPARENCY_TOL = 1e-6  # 确定性 map 读出跨进程 bit 级一致（实测 0.0），仅防意外末位差
_RESUME_STATE_MARGIN = 3e-3  # 观测量随历史漂移下界（实测 8.2e-3，取 ~1/2.7 作阈；判别性守卫）


def _va_maxdiff(a: tuple[float, float], b: tuple[float, float]) -> float:
    """valence_arousal 两分量的最大绝对差。"""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


async def _run_strong_steps(n: int) -> tuple[float, float]:
    """独立会话（单一 server 子进程）跑 n 步强刺激，返回末步 valence_arousal。"""
    async with ZeroLinkClient() as client:
        sid = await client.open_session(config=_RESUME_CONFIG)
        va = (0.0, 0.0)
        for _ in range(n):
            bundle = await client.step(sid, _RESUME_STIMULUS)
            va = (bundle.valence_arousal[0], bundle.valence_arousal[1])
        await client.close_session(sid)
    return va


async def _run_resume_across_restart(session_id: str) -> tuple[float, float]:
    """跨 server 重启：_RESUME_PRE_STEPS 步 → 出 CM（杀 server 子进程）→ 新 CM（重启 server）
    以同 id resume → 再 step 一次（第 PRE+1 步），返回该步 valence_arousal。

    stdio 传输下每次 __aenter__ 起新 server 子进程、__aexit__ 杀掉——天然模拟 server 重启；
    两次子进程共享同一 ZERO_CHECKPOINT_DB sqlite 文件 → resume-by-id 按 thread_id 恢复运行态。
    """
    async with ZeroLinkClient() as client_a:  # server 子进程 #1
        await client_a.open_session(config=_RESUME_CONFIG, session_id=session_id)
        for _ in range(_RESUME_PRE_STEPS):
            await client_a.step(session_id, _RESUME_STIMULUS)
    # 出 CM → server #1 被杀；sqlite 文件保留
    async with ZeroLinkClient() as client_b:  # server 子进程 #2（重启）
        await client_b.open_session(config=_RESUME_CONFIG, session_id=session_id)  # resume-by-id
        bundle = await client_b.step(session_id, _RESUME_STIMULUS)  # 第 PRE+1 步
        va = (bundle.valence_arousal[0], bundle.valence_arousal[1])
        await client_b.close_session(session_id)
    return va


async def _assert_readout_deterministic(va_reference: tuple[float, float]) -> None:
    """确定性 canary：同 `_RESUME_CONFIG` 再跑一次全新 1 步，应与 `va_reference` **逐字一致**。

    分离两种失败：**读出非确定** vs **resume 失效**。`affect_readout="map"` 是 `SessionConfig`
    字段，经 `open_session(config=)` 传入；Zero `server.py:97-104` 对**未命中允许键的 override
    静默丢弃、零日志**（非门控字段也走同一过滤器），一旦键名/白名单漂移就回落默认 `"sample"` +
    `rng_seed=None`（`runner.py:49-51`）→ 每步新建 `random.Random()` 采样，噪声 σ≈O(0.1) 远大于
    `_RESUME_TRANSPARENCY_TOL`。彼时下方 transparency 断言会以「运行态未跨重启恢复（sqlite/
    resume-by-id 失效？）」的文案报红，把排障整轮带偏——故先用本 canary 报出真正的病因。
    """
    va_again = await _run_strong_steps(1)
    assert _va_maxdiff(va_again, va_reference) < _RESUME_TRANSPARENCY_TOL, (
        f"确定性 canary 失败：同 config 两次全新 1 步 {va_reference} vs {va_again}"
        f"（差 {_va_maxdiff(va_again, va_reference):.2e}，应 < {_RESUME_TRANSPARENCY_TOL:.0e}）。"
        f"读出已非确定 → config override {_RESUME_CONFIG!r} 很可能被 Zero server 的 config "
        f"过滤器**静默丢弃**（回落 'sample' 随机读出）。**先查 Zero 允许键白名单，再怀疑 "
        f"resume/sqlite**——下方 transparency/state-margin 断言在此前提下均不可信。"
    )


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


@pytest.mark.zerorepo
class TestZeroClientResumeAcrossRestart:
    """T6 · 跨 server 重启会话持久（`ZERO_CHECKPOINT_BACKEND=sqlite` resume-by-id）端到端。

    对齐 Zero `server.py`（open_session 传 session_id 走 resume；运行态是否真续取决持久后端）与
    `storage/checkpointer.py`（sqlite=AsyncSqliteSaver 懒建表·`ZERO_CHECKPOINT_DB` 指库文件）。
    补此前缺口：resume 只有 mock 单测（`test_zero_client.py`），无「真起 server→step→杀 server→
    重启→同 id 续 step→验状态连续」的端到端。stdio 传输下每次 __aenter__ 起新 server 子进程、
    __aexit__ 杀掉——天然模拟重启；两次子进程共享同一 sqlite 文件即触发跨重启恢复。

    观测量确定性与判别裕度经探针实证（notes 2026-07-24）：map 读出 valence_arousal 随历史单调漂移
    Δ≈8.2e-3 >> 阈值。⚠ `ZERO_CHECKPOINT_DB` 必设**绝对临时路径**（server cwd=D:\\Zero，
    默认相对 `data/checkpoints.sqlite3` 会污染 Zero 仓）。缺 server/非目标环境 → skip。
    """

    async def test_sqlite_resume_recovers_state_across_restart(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """sqlite 后端：重启后同 id resume → 第 N 步 == 连续第 N 步（精确恢复）、≠ 全新第 1 步。"""
        if not _ZERO_SERVER.is_file():
            pytest.skip(f"D:\\Zero MCP server 不存在（{_ZERO_SERVER}）")
        if not _AIOSQLITE_AVAILABLE:
            # 缺 aiosqlite → Zero 回退 InMemory → 不恢复态 → transparency 硬红误判 flaky，故 skip
            pytest.skip(
                "aiosqlite 未安装，Zero sqlite checkpointer 会回退 InMemory，跳过跨重启 resume"
            )
        _set_stdio_env(monkeypatch)
        # 显式钉住漂移驱动门控（默认即开；防宿主 env 关掉致观测量不漂移、state-margin 守卫误触发）
        monkeypatch.setenv("ZERO_MCP_WORKSPACE_ENABLED", "true")
        monkeypatch.setenv("ZERO_CHECKPOINT_BACKEND", "sqlite")
        monkeypatch.setenv("ZERO_CHECKPOINT_DB", str(tmp_path / "resume.sqlite3"))

        # 连续基线 + 全新第 1 步（各独立会话，同 sqlite 但不重启）
        try:
            va_cont = await _run_strong_steps(_RESUME_PRE_STEPS + 1)
        except ZeroLinkConnectionError as exc:
            pytest.skip(f"连不上 Zero server（是否在 affective-expression 环境跑？）：{exc}")
        va_fresh1 = await _run_strong_steps(1)
        # 确定性 canary：先排除「config override 被静默丢弃 → 随机读出」，再谈 resume 是否失效
        await _assert_readout_deterministic(va_fresh1)

        # 跨重启 split：PRE 步 → 杀 server → 重启 resume → 第 PRE+1 步
        va_resumed = await _run_resume_across_restart(generate_session_id())

        # 判别性守卫：观测量确随历史漂移，否则下方 transparency 断言 vacuous（fail loud）
        assert _va_maxdiff(va_cont, va_fresh1) > _RESUME_STATE_MARGIN, (
            f"连续第 {_RESUME_PRE_STEPS + 1} 步 {va_cont} 与全新第 1 步 {va_fresh1} 漂移 "
            f"{_va_maxdiff(va_cont, va_fresh1):.2e} ≤ {_RESUME_STATE_MARGIN:.0e}：观测量不再判别，"
            f"resume 测试将 vacuous（增大 _RESUME_PRE_STEPS 或换观测量）"
        )
        # transparency：resume 精确恢复运行态 → 第 PRE+1 步与连续第 PRE+1 步逐字一致
        assert _va_maxdiff(va_resumed, va_cont) < _RESUME_TRANSPARENCY_TOL, (
            f"sqlite resume 后第 {_RESUME_PRE_STEPS + 1} 步 {va_resumed} 应 == 连续第 "
            f"{_RESUME_PRE_STEPS + 1} 步 {va_cont}（差 {_va_maxdiff(va_resumed, va_cont):.2e}）——"
            f"运行态未跨重启恢复（sqlite 后端/resume-by-id 失效？）"
        )
        # 显式：resumed 不是全新起（若 resume 退化为新会话则会 ≈ va_fresh1）
        assert _va_maxdiff(va_resumed, va_fresh1) > _RESUME_STATE_MARGIN, (
            f"sqlite resume 后 {va_resumed} 不应 == 全新第 1 步 {va_fresh1}（退化为新会话？）"
        )

    async def test_memory_backend_resume_starts_fresh(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """负对照：`memory` 后端下同 id 重开 = **全新会话**（不报错但不恢复态）。

        锁定 Zero `server.py:219` 语义「memory 后端重开=全新会话」，并证明上面 sqlite 用例的状态
        连续性是**后端特有**（非 resume-by-id 本身平凡成立）——判别性负对照。
        """
        if not _ZERO_SERVER.is_file():
            pytest.skip(f"D:\\Zero MCP server 不存在（{_ZERO_SERVER}）")
        _set_stdio_env(monkeypatch)
        monkeypatch.setenv("ZERO_MCP_WORKSPACE_ENABLED", "true")  # 与 sqlite 用例同门控口径
        monkeypatch.setenv("ZERO_CHECKPOINT_BACKEND", "memory")
        # memory 后端不落盘；设 DB 仅防意外相对路径写入 Zero 仓（memory 路径不读它）
        monkeypatch.setenv("ZERO_CHECKPOINT_DB", str(tmp_path / "unused.sqlite3"))

        try:
            va_fresh1 = await _run_strong_steps(1)
        except ZeroLinkConnectionError as exc:
            pytest.skip(f"连不上 Zero server（是否在 affective-expression 环境跑？）：{exc}")
        # 同 canary：本用例的等值断言（tol 1e-6）同样只有在读出确定时才有意义
        await _assert_readout_deterministic(va_fresh1)
        va_mem_resumed = await _run_resume_across_restart(generate_session_id())

        # memory 后端：重启后 resume 拿不到旧 checkpoint → 第 PRE+1 步退化为全新第 1 步
        assert _va_maxdiff(va_mem_resumed, va_fresh1) < _RESUME_TRANSPARENCY_TOL, (
            f"memory resume 后 {va_mem_resumed} 应 == 全新第 1 步 {va_fresh1}"
            f"（差 {_va_maxdiff(va_mem_resumed, va_fresh1):.2e}）：memory 后端不应跨重启恢复态"
        )
