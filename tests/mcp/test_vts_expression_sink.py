"""VtsExpressionSink 单测（离线：不连真 VTS，协议/合成/状态机/配置门全覆盖）。

覆盖：
  1. head_to_params 合成语义：AU12 上行 / AU15·AU04 负向（语义静息给出下拉空间）/
     空 AU 回静息 / intensity 增益两态 / expressiveness 放大 / decorate 两态 / clamp。
  2. BlinkMachine 时序（注入 now 确定性驱动）与高唤醒缩短间隔。
  3. VtsApiClient 应答匹配（requestID 过滤、APIError 抛 VtsApiError）。
  4. from_env 配置门：默认关返回 None（零回归）；开启时各键映射正确。
  5. render() 未连接优雅 no-op；已连接时三档 HeadPolicy 的 target/leak 语义。
  6. ExpressionSink Protocol 符合性。
"""

from __future__ import annotations

import json
import random
from typing import Any

import pytest

from src.agents.models.zero_affect import ExpressionBundle
from src.mcp.zero.expression_sink import ExpressionSink, HeadPolicy
from src.mcp.zero.sinks.vts import (
    GOVERNED_PARAMS,
    LEAK_PARAMS,
    SEMANTIC_REST,
    VTS_ERROR_AUTH_IN_PROGRESS,
    BlinkMachine,
    OuNoise,
    VtsApiClient,
    VtsApiError,
    VtsExpressionSink,
    head_to_params,
)


class _ApiStub:
    """只为 `_request_new_token` 造一个必抛指定异常的 api 替身。"""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise self.error


# ---------------------------------------------------------------------------
# 辅助构造
# ---------------------------------------------------------------------------

RANGES: dict[str, tuple[float, float, float]] = {
    "MouthSmile": (0.0, 1.0, 0.0),
    "MouthOpen": (0.0, 1.0, 0.0),
    "Brows": (0.0, 1.0, 0.0),
    "BrowLeftY": (0.0, 1.0, 0.0),
    "BrowRightY": (0.0, 1.0, 0.0),
    "EyeOpenLeft": (0.0, 1.0, 0.0),
    "EyeOpenRight": (0.0, 1.0, 0.0),
    "FaceAngleX": (-30.0, 30.0, 0.0),
    "FaceAngleY": (-30.0, 30.0, 0.0),
    "FaceAngleZ": (-90.0, 90.0, 0.0),
}
"""实测 VTS 1.35.10 回读形状：[0,1] 表情参数 defaultValue 全 0（坑源），FaceAngle ±30/±90。"""


def _head_dict(facs_au: dict[str, float], text_label: str = "content") -> dict[str, Any]:
    return {
        "facs_au": facs_au,
        "text_label": text_label,
        "physiology": {"heart_rate_bpm": 70.0, "skin_conductance": 0.3},
        "prosody": {"speech_rate": 1.0, "pitch": 1.0, "energy": 0.5},
    }


def _bundle(
    vol_au: dict[str, float],
    spont_au: dict[str, float],
    valence: float = 0.3,
    arousal: float = 0.4,
) -> ExpressionBundle:
    return ExpressionBundle.from_step_output(
        {
            "valence_arousal": [valence, arousal],
            "voluntary": _head_dict(vol_au),
            "spontaneous": _head_dict(spont_au),
        }
    )


def _head(facs_au: dict[str, float]):
    return _bundle(facs_au, facs_au).voluntary


# ---------------------------------------------------------------------------
# 1. head_to_params 合成语义
# ---------------------------------------------------------------------------


class TestHeadToParams:
    def test_au12_raises_mouth_smile_above_semantic_rest(self) -> None:
        params = head_to_params(_head({"AU12": 0.8}), 0.0, 0.0, RANGES, decorate=False)
        assert params["MouthSmile"] > SEMANTIC_REST["MouthSmile"]

    def test_au15_pulls_mouth_smile_below_semantic_rest(self) -> None:
        """负向表情依赖语义静息基准——若按 defaultValue=0 作基准，此断言必失败。"""
        params = head_to_params(_head({"AU15": 0.8}), 0.0, 0.0, RANGES, decorate=False)
        assert params["MouthSmile"] < SEMANTIC_REST["MouthSmile"]

    def test_au04_lowers_brows(self) -> None:
        params = head_to_params(_head({"AU04": 0.8}), 0.0, 0.0, RANGES, decorate=False)
        assert params["Brows"] < SEMANTIC_REST["Brows"]
        assert params["BrowLeftY"] == params["BrowRightY"] == params["Brows"]

    def test_empty_au_returns_semantic_rest_pose(self) -> None:
        params = head_to_params(_head({}), 0.0, 0.0, RANGES, decorate=False)
        for name, rest in SEMANTIC_REST.items():
            assert params[name] == pytest.approx(rest)
        assert params["EyeOpenLeft"] == pytest.approx(1.0)  # 睁眼，非 defaultValue=0

    def test_intensity_gain_attenuates_when_applied(self) -> None:
        strong = head_to_params(
            _head({"AU12": 0.8, "intensity": 1.0}), 0.0, 0.0, RANGES, decorate=False
        )
        damped = head_to_params(
            _head({"AU12": 0.8, "intensity": 0.5}), 0.0, 0.0, RANGES, decorate=False
        )
        ignored = head_to_params(
            _head({"AU12": 0.8, "intensity": 0.5}),
            0.0,
            0.0,
            RANGES,
            apply_intensity=False,
            decorate=False,
        )
        assert damped["MouthSmile"] < strong["MouthSmile"]
        assert ignored["MouthSmile"] == pytest.approx(strong["MouthSmile"])

    def test_expressiveness_amplifies_au(self) -> None:
        base = head_to_params(_head({"AU12": 0.4}), 0.0, 0.0, RANGES, decorate=False)
        amplified = head_to_params(
            _head({"AU12": 0.4}), 0.0, 0.0, RANGES, expressiveness=2.0, decorate=False
        )
        assert amplified["MouthSmile"] > base["MouthSmile"]

    def test_decorate_opens_mouth_with_arousal_and_tilts_head(self) -> None:
        plain = head_to_params(_head({}), 0.6, 0.8, RANGES, decorate=False)
        decorated = head_to_params(_head({}), 0.6, 0.8, RANGES, decorate=True)
        assert plain["MouthOpen"] == pytest.approx(0.0)
        assert plain["FaceAngleZ"] == pytest.approx(0.0)
        assert decorated["MouthOpen"] > 0.0
        assert decorated["FaceAngleZ"] > 0.0
        assert decorated["FaceAngleY"] > 0.0

    def test_all_values_within_ranges_under_extreme_au(self) -> None:
        extreme = dict.fromkeys(
            (
                "AU01",
                "AU02",
                "AU04",
                "AU05",
                "AU06",
                "AU07",
                "AU12",
                "AU15",
                "AU17",
                "AU20",
                "AU23",
                "AU26",
            ),
            1.0,
        )
        params = head_to_params(_head(extreme), 1.0, 1.0, RANGES, expressiveness=2.0)
        for name in GOVERNED_PARAMS:
            lo, hi, _ = RANGES[name]
            assert lo <= params[name] <= hi, name


# ---------------------------------------------------------------------------
# 2. BlinkMachine / OuNoise
# ---------------------------------------------------------------------------


class TestBlinkMachine:
    def test_open_then_blink_then_reopen(self) -> None:
        random.seed(7)
        bm = BlinkMachine(now=0.0)
        assert bm.factor(0.1, 0.0) == 1.0
        t_blink = bm.next_at + 0.01
        assert bm.factor(t_blink, 0.0) == 0.0  # 进入闭合
        assert bm.factor(t_blink + 0.05, 0.0) == 0.0  # 闭合期内
        assert bm.factor(t_blink + BlinkMachine.CLOSE_S + 0.01, 0.0) == 0.0  # 收尾帧
        assert bm.factor(t_blink + BlinkMachine.CLOSE_S + 0.02, 0.0) == 1.0
        assert bm.next_at > t_blink  # 下次眨眼已重排

    def test_high_arousal_shortens_interval(self) -> None:
        random.seed(11)
        calm_intervals, excited_intervals = [], []
        for arousal, out in ((0.0, calm_intervals), (1.0, excited_intervals)):
            for _ in range(200):
                bm = BlinkMachine(now=0.0)
                t = bm.next_at + 0.01
                bm.factor(t, arousal)  # 触发闭合
                bm.factor(t + BlinkMachine.CLOSE_S + 0.01, arousal)  # 收尾并重排
                out.append(bm.next_at - (t + BlinkMachine.CLOSE_S + 0.01))
        assert sum(excited_intervals) / len(excited_intervals) < sum(calm_intervals) / len(
            calm_intervals
        )


class TestOuNoise:
    def test_mean_reverts_and_stays_bounded(self) -> None:
        random.seed(3)
        noise = OuNoise(sigma=0.05)
        values = [noise.step(0.05) for _ in range(2000)]
        assert max(abs(v) for v in values) < 1.0
        assert abs(sum(values) / len(values)) < 0.05


# ---------------------------------------------------------------------------
# 3. VtsApiClient 应答匹配
# ---------------------------------------------------------------------------


class FakeWs:
    """send 收集 / recv 出队 的假 WebSocket。"""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.to_recv: list[dict[str, Any]] = []

    async def send(self, text: str) -> None:
        self.sent.append(json.loads(text))

    async def recv(self) -> str:
        import asyncio

        while not self.to_recv:  # 等测试端填充响应（避免与 request 协程的调度竞态）
            await asyncio.sleep(0)
        return json.dumps(self.to_recv.pop(0))


class TestVtsApiClient:
    async def test_request_skips_unrelated_and_returns_matching_data(self) -> None:
        ws = FakeWs()
        client = VtsApiClient(ws)  # type: ignore[arg-type]

        async def run() -> dict[str, Any]:
            return await client.request("APIStateRequest")

        task = run()
        # 先安排响应队列：一条无关 requestID + 一条匹配的
        import asyncio

        t = asyncio.ensure_future(task)
        await asyncio.sleep(0)  # 让 send 先执行
        req_id = ws.sent[0]["requestID"]
        ws.to_recv = [
            {"requestID": "unrelated", "messageType": "APIStateResponse", "data": {"x": 1}},
            {"requestID": req_id, "messageType": "APIStateResponse", "data": {"active": True}},
        ]
        data = await t
        assert data == {"active": True}
        assert ws.sent[0]["messageType"] == "APIStateRequest"

    async def test_api_error_raises(self) -> None:
        ws = FakeWs()
        client = VtsApiClient(ws)  # type: ignore[arg-type]
        import asyncio

        t = asyncio.ensure_future(client.request("ModelLoadRequest", {"modelID": "x"}))
        await asyncio.sleep(0)
        req_id = ws.sent[0]["requestID"]
        ws.to_recv = [{"requestID": req_id, "messageType": "APIError", "data": {"errorID": 1}}]
        with pytest.raises(VtsApiError) as exc_info:
            await t
        # errorID 结构化携带：判定走字段，不回头从人读文案里正则抠
        assert exc_info.value.error_id == 1

    async def test_api_error_without_error_id_keeps_none(self) -> None:
        """畸形/无 errorID 的 APIError ⇒ error_id 回落 None，不炸也不硬套一个数。"""
        ws = FakeWs()
        client = VtsApiClient(ws)  # type: ignore[arg-type]
        import asyncio

        t = asyncio.ensure_future(client.request("APIStateRequest"))
        await asyncio.sleep(0)
        req_id = ws.sent[0]["requestID"]
        ws.to_recv = [{"requestID": req_id, "messageType": "APIError", "data": "不是 dict"}]
        with pytest.raises(VtsApiError) as exc_info:
            await t
        assert exc_info.value.error_id is None


class TestAuthPendingWindowHint:
    """errorID 51（授权流程已在进行中）单列一支，文案直接给出处置。

    为什么值得单列：这一支与其它 APIError 的差别不在「失败」而在**留下了什么**——
    VTS 里正挂着一个授权窗等人点。2026-08-11 排查时三小时内撞了三次，每次都以为
    是新问题；且它与「上一个进程占着连接」不同，`Get-NetTCPConnection` 查不出来。
    """

    async def test_error_51_message_tells_operator_what_to_do(self) -> None:
        sink = VtsExpressionSink(token_path=None)
        sink.api = _ApiStub(VtsApiError("原始英文", error_id=VTS_ERROR_AUTH_IN_PROGRESS))

        with pytest.raises(VtsApiError) as exc_info:
            await sink._request_new_token({"pluginName": "p", "pluginDeveloper": "d"})

        text = str(exc_info.value)
        assert "VTube Studio" in text and "点掉" in text, "须点名到哪去做什么"
        assert "Get-NetTCPConnection" in text, "须点破「端口检查查不出这一种」"
        assert "原始英文" in text, "原文须保留，便于按 VTS 官方文档回查"
        assert exc_info.value.error_id == VTS_ERROR_AUTH_IN_PROGRESS

    async def test_other_api_errors_pass_through_untouched(self) -> None:
        """判别力：非 51 的 APIError **原样上抛**，不被这支加壳。

        没有这一条，「把所有 APIError 都换成授权窗文案」也会让上一条全绿——
        那会把真正的失败原因（如模型不存在）改写成一句误导。
        """
        original = VtsApiError("ModelLoadRequest -> APIError: {'errorID': 8}", error_id=8)
        sink = VtsExpressionSink(token_path=None)
        sink.api = _ApiStub(original)

        with pytest.raises(VtsApiError) as exc_info:
            await sink._request_new_token({"pluginName": "p", "pluginDeveloper": "d"})

        assert exc_info.value is original, "非 51 必须是原对象上抛，不重新包装"
        assert "点掉" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# 4. from_env 配置门（默认关零回归）
# ---------------------------------------------------------------------------


class TestFromEnv:
    def test_disabled_by_default_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VTS_SINK_ENABLED", raising=False)
        assert VtsExpressionSink.from_env() is None

    def test_explicit_false_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VTS_SINK_ENABLED", "false")
        assert VtsExpressionSink.from_env() is None

    def test_enabled_maps_env_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VTS_SINK_ENABLED", "true")
        monkeypatch.setenv("VTS_API_URL", "ws://127.0.0.1:9001")
        monkeypatch.setenv("VTS_SINK_MODEL", "Hiyori_A")
        monkeypatch.setenv("VTS_SINK_EXPRESSIVENESS", "1.8")
        monkeypatch.setenv("VTS_SINK_APPLY_INTENSITY", "false")
        monkeypatch.setenv("VTS_SINK_AMBIENT_MOTION", "false")
        sink = VtsExpressionSink.from_env()
        assert sink is not None
        assert sink.url == "ws://127.0.0.1:9001"
        assert sink.model_name == "Hiyori_A"
        assert sink.expressiveness == pytest.approx(1.8)
        assert sink.apply_intensity is False
        assert sink.ambient_motion is False


# ---------------------------------------------------------------------------
# 5. render()：未连接优雅 / 三档 HeadPolicy 语义
# ---------------------------------------------------------------------------


def _connected_sink(**kwargs: Any) -> VtsExpressionSink:
    """构造"已连接"状态的 sink（不真连：手工置 running/api/ranges）。"""
    sink = VtsExpressionSink(decorate=False, **kwargs)
    sink.running = True
    sink.api = object()  # render() 只判非 None，不实际调用
    sink.ranges = dict(RANGES)
    sink.target = {k: SEMANTIC_REST.get(k, RANGES[k][2]) for k in GOVERNED_PARAMS}
    return sink


class TestRender:
    async def test_not_connected_is_noop_and_warns_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        sink = VtsExpressionSink()
        bundle = _bundle({"AU12": 0.8}, {"AU12": 0.8})
        with caplog.at_level("WARNING"):
            await sink.render(bundle, policy=HeadPolicy.VOLUNTARY_ONLY)
            await sink.render(bundle, policy=HeadPolicy.VOLUNTARY_ONLY)
        warnings = [r for r in caplog.records if "未连接" in r.getMessage()]
        assert len(warnings) == 1

    async def test_voluntary_only_uses_voluntary_head(self) -> None:
        sink = _connected_sink()
        bundle = _bundle(vol_au={"AU12": 0.8}, spont_au={"AU15": 0.8})
        await sink.render(bundle, policy=HeadPolicy.VOLUNTARY_ONLY)
        assert sink.target["MouthSmile"] > 0.5  # voluntary 的笑，而非 spontaneous 的垂
        assert sink.leak == {}

    async def test_spontaneous_only_uses_spontaneous_head(self) -> None:
        sink = _connected_sink()
        bundle = _bundle(vol_au={"AU12": 0.8}, spont_au={"AU15": 0.8})
        await sink.render(bundle, policy=HeadPolicy.SPONTANEOUS_ONLY)
        assert sink.target["MouthSmile"] < 0.5

    async def test_dual_leaks_spont_minus_vol_on_leak_params_only(self) -> None:
        sink = _connected_sink()
        # voluntary 无 AU06；spontaneous 眯眼（AU06 拉低 EyeOpen）→ 眼周泄漏为负
        bundle = _bundle(vol_au={"AU12": 0.8}, spont_au={"AU12": 0.8, "AU06": 0.8})
        await sink.render(bundle, policy=HeadPolicy.DUAL)
        assert set(sink.leak) == set(LEAK_PARAMS)
        assert sink.leak["EyeOpenLeft"] < 0.0
        assert "MouthSmile" not in sink.leak  # 泄漏不碰主表情参数
        assert sink.target["MouthSmile"] > 0.5  # 主帧仍是 voluntary

    async def test_render_updates_arousal_abs(self) -> None:
        sink = _connected_sink()
        bundle = _bundle({"AU12": 0.5}, {"AU12": 0.5}, arousal=-0.7)
        await sink.render(bundle, policy=HeadPolicy.VOLUNTARY_ONLY)
        assert sink.arousal == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# 6. Protocol 符合性
# ---------------------------------------------------------------------------


def test_vts_sink_satisfies_expression_sink_protocol() -> None:
    assert isinstance(VtsExpressionSink(), ExpressionSink)


# ---------------------------------------------------------------------------
# 7. 生命周期端到端（假 VTS server：__aenter__/渲染循环/故障可观测/__aexit__ 清理）
# ---------------------------------------------------------------------------


class FakeVtsServer:
    """自动应答的假 VTS WebSocket：send 解析请求即生成响应，recv 出队。"""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.responses: list[dict[str, Any]] = []
        self.closed = False
        self.inject_error = False  # InjectParameterDataRequest 回 APIError
        self.revoke_first_auth = False  # 第一次 AuthenticationRequest 判失效

    def _respond(self, req: dict[str, Any]) -> dict[str, Any]:
        mt, rid = req["messageType"], req["requestID"]
        if mt == "AuthenticationTokenRequest":
            data: dict[str, Any] = {"authenticationToken": "fake-token-123"}
        elif mt == "AuthenticationRequest":
            if self.revoke_first_auth:
                self.revoke_first_auth = False
                data = {"authenticated": False, "reason": "token revoked"}
            else:
                data = {"authenticated": True, "reason": "ok"}
        elif mt == "InputParameterListRequest":
            data = {
                "defaultParameters": [
                    {"name": n, "min": lo, "max": hi, "defaultValue": rest}
                    for n, (lo, hi, rest) in RANGES.items()
                ]
            }
        elif mt == "InjectParameterDataRequest":
            if self.inject_error:
                return {"requestID": rid, "messageType": "APIError", "data": {"errorID": 1}}
            data = {}
        else:
            data = {}
        return {
            "requestID": rid,
            "messageType": f"{mt.removesuffix('Request')}Response",
            "data": data,
        }

    async def send(self, text: str) -> None:
        req = json.loads(text)
        self.sent.append(req)
        self.responses.append(self._respond(req))

    async def recv(self) -> str:
        import asyncio

        while not self.responses:
            await asyncio.sleep(0)
        return json.dumps(self.responses.pop(0))

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_vts(monkeypatch: pytest.MonkeyPatch) -> FakeVtsServer:
    import src.mcp.zero.sinks.vts as vts_mod

    fake = FakeVtsServer()

    async def fake_connect(url: str) -> FakeVtsServer:
        return fake

    monkeypatch.setattr(vts_mod, "_connect", fake_connect)
    return fake


class TestLifecycle:
    async def test_enter_injects_frames_and_exit_closes_ws(
        self, fake_vts: FakeVtsServer, tmp_path: Any
    ) -> None:
        import asyncio

        token_file = tmp_path / "tok"
        sink = VtsExpressionSink(token_path=token_file, render_hz=100.0)
        async with sink:
            assert sink.healthy
            await asyncio.sleep(0.05)  # 100Hz 下若干帧
        injects = [m for m in fake_vts.sent if m["messageType"] == "InjectParameterDataRequest"]
        assert injects, "渲染循环应至少注入一帧"
        ids = {p["id"] for p in injects[0]["data"]["parameterValues"]}
        assert ids == set(GOVERNED_PARAMS)  # 白名单全量注入
        assert fake_vts.closed  # __aexit__ 释放连接
        assert token_file.read_text(encoding="utf-8") == "fake-token-123"  # 弹窗路径落盘

    async def test_loop_failure_is_observable_and_cleanup_still_runs(
        self, fake_vts: FakeVtsServer, tmp_path: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        import asyncio

        fake_vts.inject_error = True
        sink = VtsExpressionSink(token_path=tmp_path / "tok", render_hz=100.0)
        bundle = _bundle({"AU12": 0.8}, {"AU12": 0.8})
        async with sink:
            await asyncio.sleep(0.05)  # 首帧注入即 APIError → 循环退出
            assert sink.running is False
            assert isinstance(sink.last_error, VtsApiError)
            assert not sink.healthy
            with caplog.at_level("WARNING"):
                await sink.render(bundle, policy=HeadPolicy.VOLUNTARY_ONLY)
            assert any("丢帧" in r.getMessage() for r in caplog.records)
        assert fake_vts.closed  # 循环带错结束也不泄漏 ws

    async def test_cached_token_reused_and_revoked_token_reauths(
        self, fake_vts: FakeVtsServer, tmp_path: Any
    ) -> None:
        token_file = tmp_path / "tok"
        token_file.write_text("stale-token", encoding="utf-8")
        fake_vts.revoke_first_auth = True  # 缓存 token 已被撤销
        sink = VtsExpressionSink(token_path=token_file, render_hz=100.0)
        async with sink:
            pass
        auth_reqs = [m for m in fake_vts.sent if m["messageType"] == "AuthenticationRequest"]
        token_reqs = [m for m in fake_vts.sent if m["messageType"] == "AuthenticationTokenRequest"]
        assert auth_reqs[0]["data"]["authenticationToken"] == "stale-token"  # 先试缓存
        assert len(token_reqs) == 1  # 失效后自动重走一次弹窗授权
        assert auth_reqs[1]["data"]["authenticationToken"] == "fake-token-123"
        assert token_file.read_text(encoding="utf-8") == "fake-token-123"  # 缓存已刷新


# ---------------------------------------------------------------------------
# 8. VtsApiClient 并发串行化（蓝图 AD-10 请求锁）
# ---------------------------------------------------------------------------


class TestVtsApiClientConcurrency:
    """两协程并发 request 各自拿到匹配响应、无超时（AD-10）。

    无锁时的失败形态：两协程同时 ``ws.recv()``，先醒者消费并永久丢弃
    （``continue`` 分支）对方的响应，使对方超时。本用例判别力已按
    「先证能红」纪律实证：临时把 ``request`` 的 ``async with self.lock``
    换成每次新建的一次性锁（等效去锁）后，本用例以 TimeoutError 稳定变红。
    """

    async def test_concurrent_requests_each_get_matching_response(self) -> None:
        import asyncio

        ws = FakeWs()
        client = VtsApiClient(ws, timeout=1.0)  # 短超时：判别失败时 fail-fast

        def _response_for(req: dict[str, Any]) -> dict[str, Any]:
            mt = req["messageType"]
            return {
                "requestID": req["requestID"],
                "messageType": f"{mt.removesuffix('Request')}Response",
                "data": {"echo": mt},
            }

        async def respond_adversarially() -> None:
            # 对抗式应答：尽量等到两个请求同时在场，再按**逆序**投递响应。
            # 有锁（串行化）时任意时刻至多一个请求在场，逆序退化为顺序，
            # 客户端各取所需；无锁时两请求同时在场，先醒的协程会先撞上
            # 对方的响应并丢弃，暴露撕响应。
            answered = 0
            while answered < 2:
                for _ in range(200):  # 让出调度，给第二个请求入场窗口
                    if len(ws.sent) - answered >= 2:
                        break
                    await asyncio.sleep(0)
                pending = ws.sent[answered:]
                if not pending:
                    await asyncio.sleep(0)
                    continue
                for req in reversed(pending):
                    ws.to_recv.append(_response_for(req))
                answered += len(pending)

        responder = asyncio.ensure_future(respond_adversarially())
        try:
            first, second = await asyncio.gather(
                client.request("APIStateRequest"),
                client.request("StatisticsRequest"),
            )
        finally:
            responder.cancel()
            try:
                await responder
            except asyncio.CancelledError:
                pass
        assert first == {"echo": "APIStateRequest"}
        assert second == {"echo": "StatisticsRequest"}
