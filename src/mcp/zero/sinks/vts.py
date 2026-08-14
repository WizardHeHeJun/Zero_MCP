"""VTube Studio Live2D 表达渲染终端——VtsExpressionSink。

结构上满足 ExpressionSink Protocol（无需继承）：经 VTS Public API（WebSocket）
把 ExpressionBundle 的 FACS AU 合成为 VTS 默认输入参数并以渲染循环连续注入，
驱动 Live2D 皮套做出表情。生命周期用 ``async with``（连接/认证/读值域/起渲染
循环，退出时停循环并交还参数控制权）。

配置全走 .env（`from_env` 工厂）；``VTS_SINK_ENABLED`` 默认关——零回归。
依赖 ``websockets`` 走 optional extra ``sinks-vts``（延迟 import：默认关的能力
默认不装依赖，模块加载不因缺包崩溃——对齐 io_adapters 惯例）。

VTS API 行为依据（官方 README 现场核验 + VTS 1.35.10 实测，2026-07-30）：
- 注入参数须 **≥1Hz 重发**，停发 1s 判 lost 回落默认值（故用渲染循环而非逐帧发）。
- 授权 token 一次弹窗获取后可跨会话复用（落盘 token 文件；用户可在 VTS 内撤销，
  撤销后本 sink 会删除失效缓存并自动重走一次弹窗授权）。
- ``InputParameterListRequest`` 回的 defaultValue **全 0**（含 EyeOpen 0=闭眼、
  MouthSmile/Brows 0=最低端）——直接作静息基准会闭眼且负向表情（AU15 垂嘴角/
  AU04 皱眉）无下拉空间，故 [0,1] 表情参数改用**语义静息基准**（Live2D 标准
  参数惯例：嘴形/眉 0.5 中性、睁眼 1.0；工程假设，见 SEMANTIC_REST）。
- ``ModelLoadRequest`` 有全局 2s 冷却。

失败可观测性（审查门修订）：渲染循环广谱兜底 ``Exception``——任何异常都会置
``running=False`` 并记入 ``last_error``，使后续 ``render()`` 走"未连接"告警路径
而非静默丢帧；``ExpressionRouter`` 的"单 sink 失败不拖垮其他"依赖 render() 抛错
才可见，而本 sink 的失败面在后台循环，故用 ``healthy`` 属性补一个显式探测点。

微表情层（ambient_motion，可关）：眨眼状态机（高唤醒更频）、呼吸正弦
（频率随唤醒 0.25→0.4Hz）、眉/头 OU 微噪声（左右独立→自然不对称）、
目标一阶低通收敛（表情"浮现"而非线性滑入）。

DUAL 策略语义（对齐 expression_sink.HeadPolicy 文档）：voluntary 为主表情，
(spontaneous − voluntary) 的参数差按 leak_weight 泄漏到眉/眼参数。
⚠ 实测（2026-07-30 所连 Zero 部署）双头输出逐值相同、泄漏恒 0——通道保留，
上游分化后自动生效。
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.agents.models.zero_affect import ExpressionBundle, ExpressionHead
from src.mcp.zero.expression_sink import HeadPolicy

if TYPE_CHECKING:  # 仅类型标注用——运行时不导入，避免与 behavior_overlay 环导
    from src.mcp.zero.sinks.behavior_overlay import BehaviorOverlayEngine
    from src.mcp.zero.sinks.trajectory import TrajectoryPlayer

logger = logging.getLogger(__name__)

# ── VTS 输入参数 ─────────────────────────────────────────────────────────────

GOVERNED_PARAMS: tuple[str, ...] = (
    "MouthSmile",
    "MouthOpen",
    "Brows",
    "BrowLeftY",
    "BrowRightY",
    "EyeOpenLeft",
    "EyeOpenRight",
    "FaceAngleX",
    "FaceAngleY",
    "FaceAngleZ",
)
"""本 sink 治理的 VTS 默认输入参数白名单——每帧全量注入（缺 AU 回语义静息）。"""

OPTIONAL_OVERLAY_PARAMS: tuple[str, ...] = (
    "BodyAngleX",
    "BodyAngleY",
    "BodyAngleZ",
    "EyeLeftX",
    "EyeLeftY",
    "EyeRightX",
    "EyeRightY",
)
"""行为叠加层（蓝图 AD-5）的可选注入参数——**只收不 raise**：连接时在场的并入
``ranges``，缺席的记入 ``unavailable_params``（对应离散行为降级，如 body→head）。
不进 ``GOVERNED_PARAMS``（其"缺参即 raise"语义被现有测试锁定）；仅在对应行为
活跃期按需注入，release 收束到 0 后停发，借 VTS 1s lost 回收机制自然交还控制权。"""

LEAK_PARAMS: frozenset[str] = frozenset(
    {"Brows", "BrowLeftY", "BrowRightY", "EyeOpenLeft", "EyeOpenRight"}
)
"""DUAL 微表情泄漏只作用的参数集（眉/眼周——对齐 HeadPolicy.DUAL 文档语义）。"""

SEMANTIC_REST: dict[str, float] = {
    "MouthSmile": 0.5,
    "MouthOpen": 0.0,
    "Brows": 0.5,
    "BrowLeftY": 0.5,
    "BrowRightY": 0.5,
    "EyeOpenLeft": 1.0,
    "EyeOpenRight": 1.0,
}
"""[0,1] 表情参数的语义静息基准（工程假设，依据见模块 docstring）。"""

REQUEST_TIMEOUT_S = 10.0
"""单次 VTS API 请求超时（秒）。VTS 卡死但连接未断时 fail-fast 而非无限期挂起；
超时按连接层故障处理（渲染循环退出置 running=False）。授权请求除外，见
``_authenticate``（等用户在 VTS 弹窗里点允许，无上限）。"""


async def _connect(url: str) -> Any:
    """建立 WebSocket 连接（延迟 import websockets；独立函数=单测可替换的缝）。"""
    from websockets.asyncio.client import connect  # 延迟 import：缺包不崩模块加载

    return await connect(url, max_size=2**23)


def _clamp(lo: float, hi: float, x: float) -> float:
    return max(lo, min(hi, x))


def _clamp01(x: float) -> float:
    return _clamp(0.0, 1.0, x)


def head_to_params(
    head: ExpressionHead,
    valence: float,
    arousal: float,
    ranges: dict[str, tuple[float, float, float]],
    *,
    expressiveness: float = 1.0,
    apply_intensity: bool = True,
    decorate: bool = True,
) -> dict[str, float]:
    """单表达头的 FACS AU 合成为 VTS 输入参数帧（纯函数，值均已 clamp 到值域）。

    合成权重与 AU→参数对应关系经文献核验（FACS AU 解剖动作 ↔ VTS 参数语义 ↔
    Live2D 标准参数范围），多 AU 驱动同一参数时线性加权后 clamp。

    Args:
        head:           表达头（facs_au 键 ⊆ 13 AU 子集 + "intensity"）。
        valence:        本轮 (v,a) 中的 v，仅用于装饰性头部姿态。
        arousal:        本轮 (v,a) 中的 a，用于装饰性张嘴/头部姿态。
        ranges:         参数名 → (min, max, defaultValue)，运行时读回，不硬编码。
        expressiveness: 表现力增益（乘 AU 后 clamp01）。1.0=忠实 AU 幅度；
                        实测产品幅度叠 VTS 平滑后肉眼偏淡，演示建议 1.5~2.0。
        apply_intensity: 是否按 facs_au["intensity"] 作全局增益（与
                        ArkitFacsMapper.apply_intensity 同语义）。
        decorate:       是否叠加装饰项（高唤醒微张嘴——所连 Zero 通路不出 AU26，
                        否则嘴全程闭死；valence 倾头/arousal 抬头微姿态）。

    Returns:
        GOVERNED_PARAMS 全键的参数帧 dict。
    """
    gain = _clamp01(head.facs_au.get("intensity", 1.0)) if apply_intensity else 1.0
    a = {
        k: _clamp01(v * gain * expressiveness) for k, v in head.facs_au.items() if k != "intensity"
    }

    def au(key: str) -> float:
        return a.get(key, 0.0)

    def out(name: str, s_up: float, s_down: float) -> float:
        lo, hi, _ = ranges[name]
        rest = SEMANTIC_REST.get(name, ranges[name][2])
        return _clamp(lo, hi, rest + s_up * (hi - rest) - s_down * (rest - lo))

    brow_raise = _clamp01(max(au("AU01"), au("AU02")) + 0.3 * min(au("AU01"), au("AU02")))
    brow_up, brow_down = 0.6 * brow_raise, 1.0 * au("AU04")
    smile_up = _clamp01(1.0 * au("AU12") + 0.35 * au("AU06") + 0.3 * au("AU20"))
    smile_down = _clamp01(0.9 * au("AU15") + 0.25 * au("AU23") + 0.2 * au("AU17"))
    open_up = _clamp01(1.0 * au("AU26") + 0.15 * au("AU20"))
    open_down = _clamp01(0.5 * au("AU23") + 0.4 * au("AU17"))
    eye_up = 0.6 * au("AU05")
    eye_down = _clamp01(0.55 * au("AU07") + 0.25 * au("AU06"))

    params = {
        "MouthSmile": out("MouthSmile", smile_up, smile_down),
        "MouthOpen": out("MouthOpen", open_up, open_down),
        "Brows": out("Brows", brow_up, brow_down),
        "BrowLeftY": out("BrowLeftY", brow_up, brow_down),
        "BrowRightY": out("BrowRightY", brow_up, brow_down),
        "EyeOpenLeft": out("EyeOpenLeft", eye_up, eye_down),
        "EyeOpenRight": out("EyeOpenRight", eye_up, eye_down),
        "FaceAngleX": ranges["FaceAngleX"][2],
        "FaceAngleY": ranges["FaceAngleY"][2],
        "FaceAngleZ": ranges["FaceAngleZ"][2],
    }
    if decorate:
        lo_m, hi_m, _ = ranges["MouthOpen"]
        params["MouthOpen"] = _clamp(lo_m, hi_m, params["MouthOpen"] + 0.35 * max(0.0, arousal))
        for name, frac in (
            ("FaceAngleZ", 0.30 * valence),
            ("FaceAngleY", 0.15 * arousal),
        ):
            lo, hi, rest = ranges[name]
            params[name] = _clamp(lo, hi, rest + frac * (hi - lo) / 2.0)
    return params


class OuNoise:
    """Ornstein-Uhlenbeck 微噪声——回归 0 的随机游走（微表情的自然颤动源）。"""

    def __init__(self, sigma: float, theta: float = 1.5) -> None:
        self.x = 0.0
        self.sigma = sigma
        self.theta = theta

    def step(self, dt: float) -> float:
        """推进 dt 秒并返回当前值。"""
        self.x += -self.theta * self.x * dt + self.sigma * random.gauss(0.0, 1.0) * math.sqrt(dt)
        return self.x


class BlinkMachine:
    """眨眼状态机：随机间隔快闭快开；高唤醒时间隔缩短（生理事实）。

    时间由调用方注入（now 单调秒），便于单测确定性驱动。
    """

    CLOSE_S = 0.13
    """单次闭眼时长（秒）。"""

    def __init__(self, now: float = 0.0) -> None:
        self.next_at = now + random.uniform(1.5, 3.0)
        self.phase_until = 0.0
        self.closing = False

    def factor(self, now: float, arousal: float) -> float:
        """返回眼开度乘子：1=正常，0=闭合。"""
        if self.closing:
            if now >= self.phase_until:
                self.closing = False
                interval = random.uniform(2.0, 5.0) * (1.0 - 0.4 * _clamp01(arousal))
                self.next_at = now + interval
            return 0.0
        if now >= self.next_at:
            self.closing = True
            self.phase_until = now + self.CLOSE_S
            return 0.0
        return 1.0


class VtsApiError(RuntimeError):
    """VTS Public API 返回 APIError 或协议异常。

    ``error_id`` 携带 VTS 的机读 ``errorID``（协议异常/无该字段时为 None）——
    判定一律走本字段，**不要回头从 message 文本里正则抠**：那份文案里既有 VTS 的
    英文原文也有我方中文，是给人读的，不是判据。
    """

    def __init__(self, message: str, *, error_id: int | None = None) -> None:
        super().__init__(message)
        self.error_id = error_id


VTS_ERROR_AUTH_IN_PROGRESS = 51
"""VTS ``errorID`` 51：授权流程已在进行中（VTS 里有一个挂起的授权窗）。

⚠ 这个态**极易由我方自己制造**：一次被中途放弃的 ``vts_connect``（客户端超时/取消）
会在 VTS 里留下挂起授权窗，而 VTS **不随请求方断开而收掉它** ⇒ 此后每次连接都撞 51，
直到有人手动点掉。且它与「上一个进程没退干净占着连接」**不是同一种残留态**——
后者 ``Get-NetTCPConnection -LocalPort 8001`` 查得出，本条查不出（连接确已断，
挂起的是 VTS 进程里的 UI 状态）。见 ``ai-docs/pitfalls.md`` 同名条目。
"""


_UNSET: Any = object()
"""``VtsApiClient.request`` 的 timeout 未传参哨兵（区别于显式 None=无限等待）。"""


class VtsApiClient:
    """最小 VTS Public API 客户端（应答式：一发一收，按 requestID 匹配）。"""

    def __init__(self, ws: Any, timeout: float = REQUEST_TIMEOUT_S) -> None:
        self.ws = ws
        self.timeout = timeout
        self.lock = asyncio.Lock()

    async def request(
        self,
        message_type: str,
        data: dict[str, Any] | None = None,
        *,
        timeout: float | None = _UNSET,
    ) -> dict[str, Any]:
        """发送一个请求并等待其响应的 data 段。

        APIError 抛 ``VtsApiError``；超过 timeout（默认实例级
        ``REQUEST_TIMEOUT_S``）抛 ``TimeoutError``——VTS 卡死但连接未断时
        fail-fast，由调用方按连接层故障处置。``timeout=None`` 显式传入表示
        无限等待（仅授权弹窗场景）。

        收发全程持 ``self.lock`` 串行化（蓝图 AD-10）：渲染循环与热键触发/枚举
        共用本 client，并发调用时两个协程同时 ``ws.recv()`` 会撕响应——一方消费
        并永久丢弃（``continue`` 分支）另一方的帧使其超时（websockets 库亦禁止
        并发 recv）。锁放 client 内而非调用方约定，防未来第三个调用方再踩。
        """
        req_id = uuid.uuid4().hex[:16]
        payload: dict[str, Any] = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": req_id,
            "messageType": message_type,
        }
        if data is not None:
            payload["data"] = data

        async def _roundtrip() -> dict[str, Any]:
            async with self.lock:
                await self.ws.send(json.dumps(payload))
                while True:
                    resp = json.loads(await self.ws.recv())
                    if resp.get("requestID") != req_id:
                        continue
                    if resp.get("messageType") == "APIError":
                        # ⚠ 变量名不得叫 payload：那会把外层闭包变量变成本函数局部变量，
                        # 上面 send(json.dumps(payload)) 当场 UnboundLocalError。
                        err_data = resp.get("data")
                        raw_id = err_data.get("errorID") if isinstance(err_data, dict) else None
                        raise VtsApiError(
                            f"{message_type} -> APIError: {err_data}",
                            error_id=raw_id if isinstance(raw_id, int) else None,
                        )
                    # 显式标注：resp 来自 json.loads ⇒ Any，直接 return 会让
                    # strict mypy 判 [no-any-return]（本行是 main 上既有的唯一
                    # mypy 红点，顺手清掉——同 `_dump_model` 的类型安全包装惯例）。
                    result: dict[str, Any] = resp.get("data", {})
                    return result

        wait = self.timeout if timeout is _UNSET else timeout
        if wait is None:
            return await _roundtrip()
        return await asyncio.wait_for(_roundtrip(), timeout=wait)


def bool_env(key: str, default: str) -> bool:
    """读布尔型 env（真值集 {"1", "true", "yes"}，与仓内先例一致）。

    公开（无下划线前缀，INFO1 修订）：本仓多处跨模块复用（`from_env`、行为层
    ``src/mcp/behavior/service.py`` 的 ``VTS_BEHAVIOR_HOTKEYS`` 门控），
    前导下划线曾暗示"仅本模块私有"，与实际被跨模块 import 的事实不符。
    """
    return os.environ.get(key, default).lower() in {"1", "true", "yes"}


def kwargs_from_env() -> dict[str, Any]:
    """从 ``VTS_*`` env 组装 ``VtsExpressionSink`` 构造 kwargs（不含开关门控）。

    公开（无下划线前缀，INFO1 修订，理由同 ``bool_env``）：供 ``from_env``
    （``VTS_SINK_ENABLED`` 门）与行为层 service
    （``VTS_BEHAVIOR_ENABLED`` 门，蓝图 AD-9）共用同一套连接/模型/表现力配置。

    Raises:
        ValueError: ``VTS_SINK_EXPRESSIVENESS`` 配了非数值（fail-fast，
            错误信息带 env 键名）。
    """
    raw_gain = os.environ.get("VTS_SINK_EXPRESSIVENESS", "1.0")
    try:
        gain = float(raw_gain)
    except ValueError as exc:
        raise ValueError(f"VTS_SINK_EXPRESSIVENESS={raw_gain!r} 不是合法数值") from exc
    token_file = os.environ.get("VTS_TOKEN_FILE", ".vts_token")
    model = os.environ.get("VTS_SINK_MODEL", "").strip() or None
    return {
        "url": os.environ.get("VTS_API_URL", "ws://127.0.0.1:8001"),
        "token_path": Path(token_file) if token_file else None,
        "model_name": model,
        "expressiveness": gain,
        "apply_intensity": bool_env("VTS_SINK_APPLY_INTENSITY", "true"),
        "ambient_motion": bool_env("VTS_SINK_AMBIENT_MOTION", "true"),
    }


class VtsExpressionSink:
    """VTube Studio 渲染终端（满足 ExpressionSink Protocol）。

    用法::

        sink = VtsExpressionSink.from_env()   # VTS_SINK_ENABLED=false 时为 None
        if sink is not None:
            async with sink:
                router = ExpressionRouter([sink], policy=HeadPolicy.DUAL)
                await router.route(step_out)   # 之后表情由渲染循环持续注入

    render() 只更新目标状态；实际注入由 __aenter__ 起的渲染循环以 render_hz
    连续执行（满足 VTS ≥1Hz 心跳）。未进入 async with（或渲染循环已因故障
    退出，见 ``healthy``/``last_error``）时 render() no-op 并 warning 一次。
    """

    def __init__(
        self,
        *,
        url: str = "ws://127.0.0.1:8001",
        token_path: Path | None = None,
        plugin_name: str = "Zero-MCP Expression Bridge",
        plugin_developer: str = "Zero_MCP",
        model_name: str | None = None,
        expressiveness: float = 1.0,
        apply_intensity: bool = True,
        ambient_motion: bool = True,
        decorate: bool = True,
        render_hz: float = 20.0,
        leak_weight: float = 0.35,
        smooth_tau: float = 0.35,
    ) -> None:
        self.url = url
        self.token_path = token_path
        self.plugin_name = plugin_name
        self.plugin_developer = plugin_developer
        self.model_name = model_name
        self.expressiveness = expressiveness
        self.apply_intensity = apply_intensity
        self.ambient_motion = ambient_motion
        self.decorate = decorate
        self.render_hz = render_hz
        self.leak_weight = leak_weight
        self.smooth_tau = smooth_tau

        self.ws: Any = None
        self.api: VtsApiClient | None = None
        self.ranges: dict[str, tuple[float, float, float]] = {}
        self.unavailable_params: list[str] = []
        self.all_params: dict[str, tuple[float, float, float]] = {}
        self.trajectory: TrajectoryPlayer | None = None
        # 行为叠加引擎（蓝图 AD-4）：由行为 service 连接后挂上；None=无手势叠加（零回归）
        self.behavior_overlay: BehaviorOverlayEngine | None = None
        # 语音口型独占层（speech-play 蓝图 2026-08-14 AD-5）：由行为 service 连接后
        # 挂上（与 self.trajectory 是不同实例的同类对象）；None=无语音播放（零回归）。
        # 混合顺序上它排在 self.trajectory 之后，对播放期 mouth_track 涉及的键有
        # 最终话语权——"最后应用者赢"是既有排序规则天然给出的独占语义。
        self.speech_mouth: TrajectoryPlayer | None = None
        self.render_task: asyncio.Task[None] | None = None
        self.running = False
        self.last_error: BaseException | None = None
        self.warned_not_connected = False
        self.warned_overlay_failure = False
        # render() 与渲染循环间的共享目标状态
        self.target: dict[str, float] = {}
        self.leak: dict[str, float] = {}
        self.arousal = 0.0

    @property
    def healthy(self) -> bool:
        """渲染循环存活且无故障——调用方的显式健康探测点。

        本 sink 的失败面在后台循环（render() 本身不抛），ExpressionRouter 的
        "单 sink 失败不拖垮其他"看不见它，须由调用方轮询本属性或检查
        ``last_error``。
        """
        return self.running and self.last_error is None

    # ── 配置工厂 ─────────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> VtsExpressionSink | None:
        """按 .env 构造；``VTS_SINK_ENABLED`` 非真值（默认）返回 None——零回归。

        kwargs 组装在模块级 ``kwargs_from_env``（与行为层 service 共用）。

        Raises:
            ValueError: ``VTS_SINK_EXPRESSIVENESS`` 配了非数值（fail-fast，
                错误信息带 env 键名）。
        """
        if not bool_env("VTS_SINK_ENABLED", "false"):
            return None
        return cls(**kwargs_from_env())

    # ── 生命周期 ─────────────────────────────────────────────────────────────

    async def __aenter__(self) -> VtsExpressionSink:
        self.last_error = None
        self.warned_not_connected = False
        self.warned_overlay_failure = False
        self.ws = await _connect(self.url)
        try:
            self.api = VtsApiClient(self.ws)
            await self._authenticate()
            if self.model_name is not None:
                await self._ensure_model(self.model_name)
            await self._read_ranges()
            self.target = self._rest_pose()
            self.leak = {}
            self.running = True
            self.render_task = asyncio.create_task(self._render_loop())
        except BaseException:
            await self.ws.close()
            self.ws = None
            self.api = None
            raise
        logger.info("VtsExpressionSink 已连接：%s（render_hz=%s）", self.url, self.render_hz)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.running = False
        try:
            if self.render_task is not None:
                try:
                    await self.render_task
                except Exception as exc:
                    # 循环内部已广谱兜底，这里只防御性兜住收尾竞态，
                    # 不让后台任务的异常顶替语句体正在传播的异常。
                    logger.warning("渲染循环收尾异常：%s", exc)
                self.render_task = None
        finally:
            if self.ws is not None:
                await self.ws.close()
                self.ws = None
            self.api = None
        logger.info("VtsExpressionSink 已断开——停止注入，1s 后 VTS 收回参数控制权。")

    # ── ExpressionSink Protocol ──────────────────────────────────────────────

    async def render(self, bundle: ExpressionBundle, *, policy: HeadPolicy) -> None:
        """按 policy 更新渲染目标（实际注入由渲染循环连续执行）。

        DUAL：voluntary 为主帧，(spontaneous − voluntary) 差异按 leak_weight
        泄漏到 LEAK_PARAMS（眉/眼周）。未连接（含渲染循环已故障退出）时
        no-op + warning 一次。
        """
        if not self.running or self.api is None:
            if not self.warned_not_connected:
                logger.warning(
                    "VtsExpressionSink.render 在未连接状态被调用——丢帧（仅告警一次）。"
                    "未进入 async with，或渲染循环已故障退出（last_error=%r）。",
                    self.last_error,
                )
                self.warned_not_connected = True
            return

        va = bundle.valence_arousal
        if policy is HeadPolicy.SPONTANEOUS_ONLY:
            main_head = bundle.spontaneous
        else:
            main_head = bundle.voluntary
        main = head_to_params(
            main_head,
            va[0],
            va[1],
            self.ranges,
            expressiveness=self.expressiveness,
            apply_intensity=self.apply_intensity,
            decorate=self.decorate,
        )
        leak: dict[str, float] = {}
        if policy is HeadPolicy.DUAL:
            spont = head_to_params(
                bundle.spontaneous,
                va[0],
                va[1],
                self.ranges,
                expressiveness=self.expressiveness,
                apply_intensity=self.apply_intensity,
                decorate=self.decorate,
            )
            leak = {k: spont[k] - main[k] for k in LEAK_PARAMS}
        self.target = main
        self.leak = leak
        self.arousal = _clamp01(abs(va[1]))

    # ── 内部 ─────────────────────────────────────────────────────────────────

    def _rest_pose(self) -> dict[str, float]:
        return {k: SEMANTIC_REST.get(k, self.ranges[k][2]) for k in GOVERNED_PARAMS}

    async def _authenticate(self) -> None:
        assert self.api is not None
        plugin = {
            "pluginName": self.plugin_name,
            "pluginDeveloper": self.plugin_developer,
        }
        token = await self._load_cached_token()
        from_cache = token is not None
        if token is None:
            token = await self._request_new_token(plugin)
        auth = await self.api.request(
            "AuthenticationRequest", {**plugin, "authenticationToken": token}
        )
        if auth.get("authenticated") is not True and from_cache:
            # 缓存 token 已被用户在 VTS 内撤销：删失效缓存，自动重走一次弹窗授权。
            logger.info("VTS 缓存 token 失效（%s），重新请求授权……", auth.get("reason"))
            if self.token_path is not None:
                await asyncio.to_thread(self.token_path.unlink, missing_ok=True)
            token = await self._request_new_token(plugin)
            auth = await self.api.request(
                "AuthenticationRequest", {**plugin, "authenticationToken": token}
            )
        if auth.get("authenticated") is not True:
            raise VtsApiError(f"VTS 认证失败：{auth.get('reason')}")

    async def _load_cached_token(self) -> str | None:
        if self.token_path is None:
            return None
        path = self.token_path

        def _read() -> str | None:
            if not path.exists():
                return None
            return path.read_text(encoding="utf-8").strip() or None

        return await asyncio.to_thread(_read)

    async def _request_new_token(self, plugin: dict[str, str]) -> str:
        assert self.api is not None
        logger.info("VTS 首次接入：等待用户在 VTube Studio 弹窗中允许插件……")
        # timeout=None：这一步在等人点弹窗，不设上限。
        try:
            data = await self.api.request("AuthenticationTokenRequest", dict(plugin), timeout=None)
        except VtsApiError as exc:
            if exc.error_id != VTS_ERROR_AUTH_IN_PROGRESS:
                raise
            # 单列这一支：它与其它 APIError 的差别不在「失败」而在**留下了什么**——
            # VTS 里正挂着一个授权窗等人处置。混在通用文案里等于把这条线索丢掉，
            # 而调用方（含跨仓消费方）拿到原始英文 errorID 51 未必知道要去点它。
            raise VtsApiError(
                f"{VTS_ERROR_AUTH_IN_PROGRESS} 授权流程已在进行中：VTube Studio 里有一个"
                "挂起的授权窗未处置（多半是上一次被中途放弃的连接留下的——VTS 不会随"
                "请求方断开而收掉它）。处置：到 VTS 界面点掉/允许那个弹窗后重试。"
                f"⚠ 该残留态用 Get-NetTCPConnection 查不出（连接确已断）。原文：{exc}",
                error_id=exc.error_id,
            ) from exc
        token = str(data["authenticationToken"])
        if self.token_path is not None:
            await asyncio.to_thread(self.token_path.write_text, token, encoding="utf-8")
            logger.info("VTS 授权 token 已落盘 %s（勿提交，走 .gitignore）", self.token_path)
        return token

    async def _ensure_model(self, model_name: str) -> None:
        assert self.api is not None
        data = await self.api.request("AvailableModelsRequest")
        models = data.get("availableModels", [])
        target = next((m for m in models if m.get("modelName") == model_name), None)
        if target is None:
            names = [m.get("modelName") for m in models]
            raise VtsApiError(f"VTS 未找到模型 {model_name!r}，可用：{names}")
        if target.get("modelLoaded") is True:
            return
        await self.api.request("ModelLoadRequest", {"modelID": target["modelID"]})
        await asyncio.sleep(3.0)  # 官方 2s 全局冷却 + 加载动画余量

    async def _read_ranges(self) -> None:
        """读回参数值域表：``GOVERNED_PARAMS`` 缺参即 raise（现有语义不变）；
        ``OPTIONAL_OVERLAY_PARAMS`` 只收不 raise（蓝图 AD-5）——官方对 custom
        参数所在数组未明示，``defaultParameters`` 与 ``customParameters`` 两个
        数组都查（防御性兼顾）；缺席的记 ``unavailable_params``。

        缺可选参数的告警级别按 ``self.behavior_overlay`` 是否已挂载分级
        （INFO2 修订）：已挂（行为层用户，经 ``BehaviorService.connect()`` 在
        ``__aenter__`` **之前**挂上——见其 docstring）时该信息与降级执行直接
        相关，保持 WARNING；未挂（纯表情通路用户，未启用行为层）时是无关
        噪音，降为 DEBUG 且措辞中性化（不预设读者已知道"行为层"概念）。
        """
        assert self.api is not None
        data = await self.api.request("InputParameterListRequest")
        listed = [*data.get("defaultParameters", []), *data.get("customParameters", [])]
        # 全量参数表（轨迹通道的作用空间，2026-07-31 二期）：动作模型可驱动任意
        # 所连部署实际存在的输入参数，不限于 GOVERNED/OPTIONAL 白名单。
        self.all_params = {
            p["name"]: (float(p["min"]), float(p["max"]), float(p["defaultValue"])) for p in listed
        }
        table = {
            p["name"]: (
                float(p["min"]),
                float(p["max"]),
                float(p["defaultValue"]),
            )
            for p in data.get("defaultParameters", [])
            if p["name"] in GOVERNED_PARAMS
        }
        missing = [g for g in GOVERNED_PARAMS if g not in table]
        if missing:
            raise VtsApiError(f"所连 VTS 部署缺输入参数：{missing}")
        optional = {
            p["name"]: (
                float(p["min"]),
                float(p["max"]),
                float(p["defaultValue"]),
            )
            for p in listed
            if p["name"] in OPTIONAL_OVERLAY_PARAMS
        }
        self.unavailable_params = [p for p in OPTIONAL_OVERLAY_PARAMS if p not in optional]
        if self.unavailable_params:
            if self.behavior_overlay is not None:
                logger.warning(
                    "所连 VTS 部署缺可选叠加参数 %s——对应离散行为将降级执行"
                    "（如 body→head 微量近似，见蓝图 AD-5）。",
                    self.unavailable_params,
                )
            else:
                logger.debug(
                    "所连 VTS 部署缺可选参数 %s（当前未启用，不影响表情渲染）。",
                    self.unavailable_params,
                )
        table.update(optional)
        self.ranges = table

    async def _render_loop(self) -> None:
        """render_hz 连续合成注入。

        广谱兜底 ``Exception``（保留 CancelledError 语义）：任何异常（连接层、
        APIError——如用户中途换模型致参数不存在、畸形响应）都置 running=False
        并记入 last_error，使故障对 render() 与 ``healthy`` 可见；不自动重连。

        混合顺序（ambient → behavior_overlay → trajectory → speech_mouth）：

        - 行为叠加（蓝图 AD-4）：ambient 块之后、逐参 clamp 之前把
          ``behavior_overlay`` 的手势帧合入——offsets 加性合入（可选参数以
          defaultValue 为基线、仅活跃期按需注入，AD-5），eye_gate 乘到
          EyeOpenLeft/Right（与 ambient 眨眼同为乘法链）；
        - 轨迹回放（2026-07-31 二期）合于 behavior_overlay 之后；
        - 语音口型独占层（speech-play 蓝图 2026-08-14 AD-5）合于 trajectory
          **之后**——结构与 trajectory 段完全同构（absolute 按 strength 混合、
          offset 加性叠加；speech 只用 absolute，代码保持同构简单）。这一
          排序即嘴部独占的结构性保证：``speech_mouth`` 涉及的键是本帧最后
          被写入的，不被 ambient/overlay/trajectory 的同键取值覆盖。

        该段**单独兜 Exception**：引擎/回放器异常只整帧丢弃对应层叠加 +
        warning 一次，不杀 sink——否则 bug 会沿本循环的广谱兜底杀死整条
        表情通道且不自动重连（故障隔离/可观测性理由，对齐模块 docstring
        的失败可观测性约定）。
        """
        assert self.api is not None
        dt = 1.0 / self.render_hz
        current = dict(self.target)
        blink = BlinkMachine(time.monotonic())
        noise = {
            "BrowLeftY": OuNoise(0.05),
            "BrowRightY": OuNoise(0.05),
            "FaceAngleZ": OuNoise(1.4),
            "FaceAngleY": OuNoise(0.9),
        }
        t0 = time.monotonic()
        alpha = 1.0 - math.exp(-dt / self.smooth_tau)

        try:
            while self.running:
                now = time.monotonic()
                for k in GOVERNED_PARAMS:
                    goal = self.target[k] + self.leak_weight * self.leak.get(k, 0.0)
                    current[k] += alpha * (goal - current[k])
                frame = dict(current)
                if self.ambient_motion:
                    breath_hz = 0.25 + 0.15 * self.arousal
                    frame["FaceAngleX"] += 0.9 * math.sin(2.0 * math.pi * breath_hz * (now - t0))
                    for k, n in noise.items():
                        frame[k] += n.step(dt)
                    bf = blink.factor(now, self.arousal)
                    frame["EyeOpenLeft"] *= bf
                    frame["EyeOpenRight"] *= bf
                if (
                    self.behavior_overlay is not None
                    or self.trajectory is not None
                    or self.speech_mouth is not None
                ):
                    # 局部防御（AD-4 故障面）：先在副本上合成，成功才替换 frame
                    # ——引擎/回放器异常时整帧丢弃叠加（无半应用态），表情注入照常。
                    try:
                        merged = dict(frame)
                        if self.behavior_overlay is not None:
                            overlay = self.behavior_overlay.apply(now)
                            for k, delta in overlay.offsets.items():
                                if k in merged:
                                    merged[k] += delta
                                elif k in self.ranges:
                                    # 可选参数仅活跃期进 frame，基线取 defaultValue（AD-5）
                                    merged[k] = self.ranges[k][2] + delta
                            merged["EyeOpenLeft"] *= overlay.eye_gate
                            merged["EyeOpenRight"] *= overlay.eye_gate
                        if self.trajectory is not None:
                            # 轨迹回放（2026-07-31 二期）最后应用=对其参数有最终话语权：
                            # absolute 按 takeover 强度向目标值混合，offset 加性叠加。
                            tf = self.trajectory.apply(now)
                            if tf is not None:
                                for k, v in tf.values.items():
                                    table = self.ranges.get(k) or self.all_params.get(k)
                                    if table is None:
                                        continue
                                    base = merged.get(k, table[2])
                                    if tf.mode == "offset":
                                        merged[k] = base + v * tf.strength
                                    else:
                                        merged[k] = base + (v - base) * tf.strength
                        if self.speech_mouth is not None:
                            # 语音口型独占层（speech-play 蓝图 AD-5）：结构与上面
                            # trajectory 段完全同构，但**排在其后**——对 mouth_track
                            # 涉及的键有最终话语权（嘴部独占的结构性保证，见类
                            # docstring 混合顺序说明）。
                            sf = self.speech_mouth.apply(now)
                            if sf is not None:
                                for k, v in sf.values.items():
                                    table = self.ranges.get(k) or self.all_params.get(k)
                                    if table is None:
                                        continue
                                    base = merged.get(k, table[2])
                                    if sf.mode == "offset":
                                        merged[k] = base + v * sf.strength
                                    else:
                                        merged[k] = base + (v - base) * sf.strength
                        frame = merged
                    except Exception as exc:
                        if not self.warned_overlay_failure:
                            logger.warning(
                                "行为叠加/轨迹回放层异常——丢弃本帧叠加（仅告警一次），"
                                "表情注入不受影响：%r",
                                exc,
                            )
                            self.warned_overlay_failure = True
                vals = []
                for k in GOVERNED_PARAMS:
                    lo, hi, _ = self.ranges[k]
                    vals.append({"id": k, "value": _clamp(lo, hi, frame[k])})
                for k in frame:
                    if k in GOVERNED_PARAMS:
                        continue
                    table = self.ranges.get(k) or self.all_params.get(k)
                    if table is None:
                        continue
                    lo, hi, _ = table
                    vals.append({"id": k, "value": _clamp(lo, hi, frame[k])})
                await self.api.request(
                    "InjectParameterDataRequest",
                    {"faceFound": True, "mode": "set", "parameterValues": vals},
                )
                await asyncio.sleep(dt)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.running = False
            self.last_error = exc
            logger.warning("VTS 渲染循环故障退出（不自动重连），后续 render() 将丢帧：%r", exc)
