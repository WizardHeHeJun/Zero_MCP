"""VTS 离散行为业务层——BehaviorService（蓝图 2026-07-31 §4/§5/§6 · T4）。

server（`src/mcp/vts_behavior_mcp_server.py`）薄转发本类；sink 生命周期、
热键 catalog、触发分发与状态聚合全在此层（AD-9）——传输层零业务逻辑
（`rules/mcp-integration.md`）。

## 生命周期（AD-9：惰性持有 sink、显式 connect/disconnect、幂等）

构造接受注入的 ``VtsExpressionSink``（未来编排层同进程共存场景：同一 sink、
同一连接、同一插件身份，表情流调 ``render()``、本层调 engine）；standalone
场景（v1）在首次 ``connect()`` 时经 ``vts.kwargs_from_env()`` 自建——读同一套
``VTS_*`` env（``VTS_BEHAVIOR_ENABLED`` 门控在 server 层）。connect/disconnect
均幂等；渲染循环故障后（``healthy=False``）再次 ``connect()`` = 显式重连路径
（sink 不自动重连的补偿）。

## 触发分发（AD-6 / AD-7）

- 行为词 → ``BehaviorOverlayEngine.trigger``。时间由本层取 ``time.monotonic()``
  注入——引擎不自取时钟（AD-3 可测性约定）；
- ``hotkey:<hotkeyID>`` → 我方保守冷却预拦（``HOTKEY_COOLDOWN_S``）后纯协议
  转发 ``HotkeyTriggerRequest``（不经 overlay 引擎），VTS 侧热键错误码按
  ``HOTKEY_ERROR_RECEIPT_CODES`` 映射为业务性拒绝回执；表外错误码透传
  ``VtsApiError`` 由 server 映射 ``[vtsb:vts_error]``（AD-11）。

业务性拒绝进回执 ``code`` 字段不抛异常；协议性失败（未连接/循环故障）抛
``ToolError``（带 ``[vtsb:not_connected]`` 位置无关令牌），server 侧
``except ToolError: raise`` 原样透传。

并发纪律（蓝图 §6，本次审查新增 W1）：``connect()``/``disconnect()`` 整体持
``self.lifecycle_lock``（``asyncio.Lock``）串行化——MCP SDK 对每条 JSON-RPC
请求 ``tg.start_soon`` 并发派发（审查员已在已安装 SDK 源码核验），未加锁时
双 ``connect()`` 交错会在**同一** sink 实例上并发进入 ``__aenter__``：两次
调用共享的 ``self.sink`` 在首次赋值后就已固定（赋值发生在两者的同步前缀，
早于各自第一个真正的挂起点），但各自的 ``__aenter__`` 会互相覆盖
``self.ws``/``self.api``/``self.render_task`` 等字段——较晚完成的一次会
覆盖较早一次持有的 ``render_task`` 句柄，使其永不被 ``__aexit__`` 等待/
取消而泄漏，且两次并发的 ``VtsApiClient.request`` 走的是各自不同的锁实例，
AD-10 的请求锁保护不到这里。防御对象与 AD-10（``VtsApiClient`` 内部请求锁）
同思路，但锁的粒度补在上一层——生命周期状态转移本身。``trigger``/
``interrupt``/``status`` 等其余方法与渲染循环运行在同一事件循环，engine
调用单步完成天然原子、无需锁；对 VTS 的请求经 ``VtsApiClient`` 内置串行锁
（AD-10）与渲染循环互不撕响应。
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid

from mcp.server.fastmcp.exceptions import ToolError

from src.agents.models.vts_behavior import (
    VTSB_COOLDOWN,
    VTSB_HOTKEY_UNAVAILABLE,
    VTSB_INVALID_PARAMS,
    VTSB_NOT_CONNECTED,
    VTSB_THROTTLED,
    BehaviorCatalog,
    BehaviorInfo,
    BehaviorReceipt,
    BehaviorRequest,
    BehaviorStatus,
    HotkeyInfo,
    ParamCatalog,
    ParamInfo,
    TrajectoryReceipt,
    TrajectoryRequest,
)
from src.mcp.zero.sinks.behavior_overlay import (
    VOCABULARY,
    BehaviorOverlayEngine,
    BehaviorSpec,
    Ranges,
    resolve_degradation,
)
from src.mcp.zero.sinks.trajectory import TrajectoryPlayer
from src.mcp.zero.sinks.vts import (
    GOVERNED_PARAMS,
    VtsApiError,
    VtsExpressionSink,
    bool_env,
    kwargs_from_env,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 热键通路常量（AD-7）
# ---------------------------------------------------------------------------

HOTKEY_PREFIX: str = "hotkey:"
"""热键命名空间前缀（AD-7）：热键与程序化词表统一进 ``behavior_trigger``——
对调用方是同一张清单、同一个 trigger 工具、同一套回执。"""

HOTKEY_COOLDOWN_S: float = 5.0
"""同一热键的保守冷却窗口（秒）。官方文档自相矛盾（README『每 5 帧』vs
ErrorID.cs 注释『5 秒』，矛盾本身已核验）——保守按 5s 在我方仲裁层预拦、
不去撞 VTS 侧 203；接真 VTS 实测后可放宽 [工程假设]（AD-7）。"""

HOTKEY_TYPES: frozenset[str] = frozenset({"TriggerAnimation", "ToggleExpression"})
"""枚举暴露的热键类型白名单（AD-7）：动画触发与表情开关；其余类型
（换模型/移动模型等）不进 catalog。"""

HOTKEY_KIND_BY_TYPE: dict[str, str] = {
    "TriggerAnimation": "animation",
    "ToggleExpression": "expression",
}
"""VTS 原生 type → 我方粗分类 kind（契约 ``HotkeyInfo.kind``，供 Zero 侧 LLM
免解析 VTS 原生枚举；白名单内 type 与 file 后缀一一对应，type 即可定 kind）。"""

HOTKEY_ERROR_RECEIPT_CODES: dict[int, str] = {
    200: VTSB_THROTTLED,  # HotkeyQueueFull：VTS 侧热键队列满
    201: VTSB_HOTKEY_UNAVAILABLE,  # NoModelLoaded：无模型加载
    202: VTSB_HOTKEY_UNAVAILABLE,  # IDNotFound：无此 ID / 换模型后失效（AD-7 显式映射）
    203: VTSB_COOLDOWN,  # CooldownNotOver：正常应被我方 5s 预拦，撞到即如实回执
}
"""VTS 热键错误码 → 业务性拒绝回执 code（AD-7/AD-11，码号依 ErrorID.cs
200–208 段）。表外错误码（DataInvalid 等）视为协议性失败：透传 ``VtsApiError``
由 server 映射 ``[vtsb:vts_error]``。"""

_VTS_ERROR_ID_RE: re.Pattern[str] = re.compile(r"['\"]errorID['\"]:\s*(\d+)")
"""从 ``VtsApiError`` 文本提取 VTS errorID（异常消息含 APIError data 的 repr，
见 ``VtsApiClient.request``）。提不到按表外错误处理（协议性失败）。"""


def _extract_vts_error_id(exc: VtsApiError) -> int | None:
    """从 VtsApiError 消息提取 errorID；提不到返回 None（按协议性失败处理）。"""
    match = _VTS_ERROR_ID_RE.search(str(exc))
    return int(match.group(1)) if match else None


def _rejected(code: str, detail: str) -> BehaviorReceipt:
    """装配业务性拒绝回执（AD-11：进 ``code`` 字段，不抛异常）。"""
    return BehaviorReceipt(
        status="rejected",
        behavior_id=uuid.uuid4().hex[:16],
        channels=[],
        estimated_duration_ms=0,
        code=code,
        detail=detail,
    )


def _direction_candidates(spec: BehaviorSpec) -> tuple[str | None, ...]:
    """该词全部合法 direction——无方向词（``spec.directions is None``）退化为
    单一 ``(None,)`` 情形（W2：catalog 逐 direction 判定的迭代对象）。"""
    return spec.directions if spec.directions is not None else (None,)


def _behavior_info(spec: BehaviorSpec, ranges: Ranges | None) -> BehaviorInfo:
    """装配 catalog 单词条目：available/degraded 按所连部署实际参数集计算（AD-5）。

    ranges=None（未连接）时按静态知识乐观返回（available=True、无降级说明）——
    词表是静态知识，未连接也完整交付；引擎触发时会按当时 ranges 再判一次
    （降级触发时定死），这里只作 catalog 呈现。

    降级判定复用引擎的单一真相 ``behavior_overlay.resolve_degradation``
    （W2/W5，code-review 修订——修复此前两处口径各自重复实现导致的矛盾：
    catalog 曾未按 direction 过滤即对 ``spec.primary`` 全部轨道判缺参，与
    引擎「先按 direction 过滤再判定」不一致，复现场景见 `resolve_degradation`
    docstring）：逐个该词全部合法 direction 调用判定，取**最坏情形**汇总——
    任一 direction 不可执行则整体 ``available=False``；否则只要有 direction
    降级即整体标 ``degraded``，并在说明文本里点名哪些 direction 降级、
    哪些完整（避免「只有部分方向缺参」时 catalog 报 degraded 而引擎对未受
    影响的 direction 实际 accepted 无降级的矛盾）。
    """
    if ranges is None:
        return BehaviorInfo(
            name=spec.name,
            definition=spec.definition,
            params_schema=dict(spec.params_schema),
            typical_duration_ms=spec.typical_duration_ms,
            cooldown_s=spec.cooldown_s,
            channels=list(spec.channels),
            available=True,
            degraded=None,
        )

    resolutions = [(d, resolve_degradation(spec, ranges, d)) for d in _direction_candidates(spec)]
    unresolvable = [d for d, r in resolutions if not r.resolvable]
    degraded_directions = [d for d, r in resolutions if r.resolvable and r.degraded_channels]
    full_directions = [d for d, r in resolutions if r.resolvable and not r.degraded_channels]

    degraded: str | None = None
    if unresolvable:
        if spec.directions is None:
            missing = resolutions[0][1].missing
            degraded = spec.degraded_note or f"所需参数缺席且无可用降级：{list(missing)}"
        else:
            degraded = spec.degraded_note or f"以下方向所需参数缺席且无可用降级：{unresolvable}"
    elif degraded_directions:
        if spec.directions is None:
            degraded = spec.degraded_note
        else:
            tail = f"；完整方向：{full_directions}" if full_directions else ""
            note = spec.degraded_note or "参数降级"
            degraded = f"{note}（降级方向：{degraded_directions}{tail}）"
    return BehaviorInfo(
        name=spec.name,
        definition=spec.definition,
        params_schema=dict(spec.params_schema),
        typical_duration_ms=spec.typical_duration_ms,
        cooldown_s=spec.cooldown_s,
        channels=list(spec.channels),
        available=not unresolvable,
        degraded=degraded,
    )


class BehaviorService:
    """行为层业务门面：sink 生命周期 · 热键 catalog · 触发分发 · 状态聚合。

    用法（server 薄转发）::

        service = BehaviorService()        # standalone：connect 时按 env 自建 sink
        await service.connect()            # 幂等；健康时重复调用为 no-op
        receipt = await service.trigger(BehaviorRequest(name="nod"))
        status = service.status()          # 纯状态读取（sync，无 I/O）
        await service.disconnect()         # 幂等

    时间来源：本层取 ``time.monotonic()`` 注入 engine（引擎不自取时钟，AD-3）。
    """

    def __init__(self, sink: VtsExpressionSink | None = None) -> None:
        self.sink = sink
        self.engine = BehaviorOverlayEngine()
        self.trajectory_player = TrajectoryPlayer()
        self.connected = False
        self.hotkeys: list[HotkeyInfo] | None = None
        self.hotkey_cooldown_until: dict[str, float] = {}
        self.model_id: str | None = None
        # 生命周期状态转移锁（W1）：connect()/disconnect() 整体持有，防御 MCP SDK
        # 对并发 JSON-RPC 请求的 tg.start_soon 并发派发（见类 docstring「并发纪律」）。
        # 命名对齐 VtsApiClient.lock（AD-10 同类场景的既有命名先例）。
        self.lifecycle_lock = asyncio.Lock()

    # ── 生命周期（AD-9 / AD-10 同思路的上层补丁 W1） ────────────────────────────

    async def connect(self) -> BehaviorStatus:
        """连接 VTS 并挂载手势引擎（幂等）。

        整体持 ``self.lifecycle_lock``（W1）：防御并发 ``connect()``/
        ``disconnect()`` 交错——MCP SDK 对每条 JSON-RPC 请求并发派发，未加锁
        时两次 ``connect()`` 会在同一 sink 上并发进入 ``__aenter__``，互相
        覆盖 ``self.ws``/``self.api``/``self.render_task`` 造成句柄泄漏（详见
        类 docstring）。

        - 已连接且健康：no-op，直接返回状态；
        - 已连接但渲染循环故障（``healthy=False``）：先收尾旧连接再重连
          （显式重连路径——sink 不自动重连）；
        - standalone（构造未注入 sink）：首连时经 ``kwargs_from_env()`` 自建；
        - ``behavior_overlay`` 在 ``__aenter__`` **之前**挂上（vts.py
          ``_read_ranges`` 按其是否已挂选择告警级别——INFO2 修订）；
        - 热键枚举（``VTS_BEHAVIOR_HOTKEYS``，默认开）与 model_id 读取失败
          不拖垮连接——均为可选增强，优雅回退（AD-7）。
        """
        async with self.lifecycle_lock:
            if self.connected and self.sink is not None and self.sink.healthy:
                return self.status()
            if self.connected and self.sink is not None:
                logger.info("VTS 连接已故障（last_error=%r），显式重连……", self.sink.last_error)
                await self.sink.__aexit__(None, None, None)
                self.connected = False
                self.hotkeys = None
                self.model_id = None
            if self.sink is None:
                self.sink = VtsExpressionSink(**kwargs_from_env())
            self.sink.behavior_overlay = self.engine
            self.sink.trajectory = self.trajectory_player
            if not self.sink.running:
                await self.sink.__aenter__()
            self.connected = True
            self.model_id = await self._read_model_id()
            if bool_env("VTS_BEHAVIOR_HOTKEYS", "true"):
                self.hotkeys = await self._enumerate_hotkeys()
            else:
                self.hotkeys = []
                logger.info("VTS_BEHAVIOR_HOTKEYS=false：不枚举热键，hotkey: 触发将被拒绝。")
            logger.info(
                "行为层已连接：model_id=%s，热键 %d 个，缺席可选参数 %s。",
                self.model_id,
                len(self.hotkeys),
                self.sink.unavailable_params,
            )
            return self.status()

    async def disconnect(self) -> BehaviorStatus:
        """断开 VTS（幂等）：停渲染循环、关 ws——1s 后 VTS 收回参数控制权。

        整体持 ``self.lifecycle_lock``（W1，同 ``connect()`` 的并发防御理由）。

        引擎的活跃包络/冷却表保留（按时间自然过期，不随连接清零）；
        hotkeys/model_id 回到未连接态（None，契约语义：尚未枚举）。
        """
        async with self.lifecycle_lock:
            sink = self.sink
            if sink is not None and (self.connected or sink.running):
                await sink.__aexit__(None, None, None)
            self.connected = False
            self.hotkeys = None
            self.model_id = None
            return self.status()

    # ── 触发 / 打断（AD-6 / AD-7） ───────────────────────────────────────────

    async def trigger(self, request: BehaviorRequest) -> BehaviorReceipt:
        """触发行为词或热键（``hotkey:<hotkeyID>``），返回三态回执。

        业务性拒绝（冷却/节流/占用/未知词/热键不可用）进回执 ``code``；
        协议性失败抛异常：未连接/循环故障 → ``ToolError``（``[vtsb:not_connected]``），
        VTS 表外错误 → ``VtsApiError`` 透传给 server 映射 ``[vtsb:vts_error]``。
        """
        sink = self._require_connected()
        now = time.monotonic()
        if request.name.startswith(HOTKEY_PREFIX):
            return await self._trigger_hotkey(sink, request.name, now)
        return self.engine.trigger(request, now=now, ranges=sink.ranges)

    def interrupt(self, channel: str | None = None) -> BehaviorReceipt:
        """打断活跃行为（channel=None 清全部）——交叉淡出回语义静息基准（AD-6）。

        只清手势层，不触碰表情 target；无匹配活跃行为也返回 accepted（幂等）。
        """
        self._require_connected()
        return self.engine.interrupt(channel, now=time.monotonic())

    # ── 裸参数轨迹通道（2026-07-31 二期：Zero 侧动作模型直驱） ────────────────

    def animate(self, request: TrajectoryRequest) -> TrajectoryReceipt:
        """轨迹投喂：动作模型输出的关键帧段进回放队列（纯状态操作，无 I/O）。

        业务性拒绝（未知 mode/键集不一致/参数全缺席/队列满）进回执 ``code``；
        未连接/循环故障走 ``_require_connected`` 抛 ToolError（AD-11 分界不变）。
        """
        sink = self._require_connected()
        known = frozenset(sink.ranges) | frozenset(sink.all_params)
        result = self.trajectory_player.feed(
            [(kf.t_ms / 1000.0, kf.params) for kf in request.keyframes],
            mode=request.mode,
            append=request.append,
            now=time.monotonic(),
            known_params=known,
        )
        status = "accepted" if result.ok else "rejected"
        return TrajectoryReceipt(
            status=status,
            duration_ms=result.duration_ms,
            dropped_params=result.dropped_params,
            queue_depth=result.queue_depth,
            code=result.code,
            detail=result.detail,
        )

    def clear_params(self) -> TrajectoryReceipt:
        """清除轨迹队列并交还参数控制权（幂等；250ms 缓出无跳变）。"""
        self._require_connected()
        self.trajectory_player.clear(time.monotonic())
        return TrajectoryReceipt(status="accepted", detail="轨迹已清除，参数交还中（缓出）。")

    def list_params(self) -> ParamCatalog:
        """所连 VTS 部署的全量输入参数表（动作模型的作用空间）；未连接时 params=None。"""
        sink = self.sink
        if not self.connected or sink is None or not sink.all_params:
            return ParamCatalog(params=None, connected=False)
        governed = set(GOVERNED_PARAMS)
        params = [
            ParamInfo(name=name, min=lo, max=hi, default_value=dv, governed=name in governed)
            for name, (lo, hi, dv) in sorted(sink.all_params.items())
        ]
        return ParamCatalog(params=params, connected=True)

    # ── 清单 / 状态（AD-7 / §5） ─────────────────────────────────────────────

    async def list_catalog(self, refresh: bool = False) -> BehaviorCatalog:
        """程序化词表 + 已发现热键的**同一张清单**（AD-7）。

        词表是静态知识：未连接时仍完整返回（``hotkeys=None`` 表示尚未枚举）；
        已连接时每词 available/degraded 按 ``sink.ranges`` 实际键集计算。
        ``refresh=True`` 且已连接时重枚举热键（用户中途换模型场景——只刷热键
        不刷 ranges，蓝图 open questions）。
        """
        sink = self.sink
        connected = self.connected and sink is not None
        if connected and refresh and bool_env("VTS_BEHAVIOR_HOTKEYS", "true"):
            self.hotkeys = await self._enumerate_hotkeys()
        ranges = sink.ranges if connected and sink is not None else None
        behaviors = [_behavior_info(spec, ranges) for spec in VOCABULARY.values()]
        return BehaviorCatalog(
            behaviors=behaviors,
            hotkeys=self.hotkeys if connected else None,
            connected=connected,
        )

    def status(self) -> BehaviorStatus:
        """聚合状态快照：连接/健康态 + 活跃行为 + 全部冷却（引擎词 + 热键）。

        sink 的失败面在后台渲染循环（异常即停、不自动重连）——本方法即
        显式探测点：``healthy=False`` 时 ``last_error`` 带循环最后错误，
        恢复路径 = 再次 ``connect()``。纯状态读取，无 I/O（sync）。
        """
        now = time.monotonic()
        snap = self.engine.snapshot(now)
        cooldowns = dict(snap.cooldowns)
        for name, until in list(self.hotkey_cooldown_until.items()):
            remaining = int(round((until - now) * 1000.0))
            if remaining > 0:
                cooldowns[name] = remaining
            else:
                del self.hotkey_cooldown_until[name]
        sink = self.sink
        connected = self.connected and sink is not None
        healthy = False
        last_error: str | None = None
        unavailable: list[str] = []
        if sink is not None:
            healthy = connected and sink.healthy
            if sink.last_error is not None:
                last_error = repr(sink.last_error)
            unavailable = list(sink.unavailable_params)
        traj_active, traj_remaining = self.trajectory_player.snapshot(now)
        return BehaviorStatus(
            connected=connected,
            healthy=healthy,
            last_error=last_error,
            active=snap.active,
            cooldowns=cooldowns,
            unavailable_params=unavailable,
            hotkey_count=len(self.hotkeys) if self.hotkeys is not None else None,
            model_id=self.model_id,
            trajectory_active=traj_active,
            trajectory_remaining_ms=traj_remaining,
        )

    # ── 内部 ─────────────────────────────────────────────────────────────────

    def _require_connected(self) -> VtsExpressionSink:
        """协议性前置检查：未连接/循环故障抛 ``ToolError``（``[vtsb:not_connected]``），
        server 侧 ``except ToolError: raise`` 原样透传（AD-11）。"""
        sink = self.sink
        if not self.connected or sink is None or sink.api is None:
            raise ToolError(f"{VTSB_NOT_CONNECTED} 尚未连接 VTS——先调用 vts_connect。")
        if not sink.healthy:
            raise ToolError(
                f"{VTSB_NOT_CONNECTED} 渲染循环已故障"
                f"（last_error={sink.last_error!r}）——再次 vts_connect 显式重连。"
            )
        return sink

    async def _trigger_hotkey(
        self, sink: VtsExpressionSink, name: str, now: float
    ) -> BehaviorReceipt:
        """热键触发：保守冷却预拦 → 纯协议转发（不经 overlay 引擎，AD-7）。

        与程序化行为共享回执语义：接受/拒绝均走 ``BehaviorReceipt``；热键不占
        仲裁通道（channels 空），时长未知（VTS 侧异步执行）。
        """
        if not bool_env("VTS_BEHAVIOR_HOTKEYS", "true"):
            return _rejected(
                VTSB_HOTKEY_UNAVAILABLE,
                "热键开关关（VTS_BEHAVIOR_HOTKEYS=false），hotkey: 触发不可用",
            )
        hotkey_id = name[len(HOTKEY_PREFIX) :].strip()
        if not hotkey_id:
            return _rejected(VTSB_INVALID_PARAMS, "hotkey: 后须带 hotkeyID（见 behavior_list）")
        until = self.hotkey_cooldown_until.get(name)
        if until is not None and now < until:
            remaining = max(1, int(round((until - now) * 1000.0)))
            return _rejected(
                VTSB_COOLDOWN,
                f"热键冷却未过（保守 {HOTKEY_COOLDOWN_S:g}s 预拦），剩余 {remaining}ms",
            )
        assert sink.api is not None  # _require_connected 已保证
        try:
            await sink.api.request("HotkeyTriggerRequest", {"hotkeyID": hotkey_id})
        except VtsApiError as exc:
            error_id = _extract_vts_error_id(exc)
            code = HOTKEY_ERROR_RECEIPT_CODES.get(error_id) if error_id is not None else None
            if code is None:
                raise  # 表外错误码：协议性失败，由 server 映射 [vtsb:vts_error]
            return _rejected(code, f"VTS 拒绝热键触发（errorID={error_id}）：{exc}")
        self.hotkey_cooldown_until[name] = now + HOTKEY_COOLDOWN_S
        return BehaviorReceipt(
            status="accepted",
            behavior_id=uuid.uuid4().hex[:16],
            channels=[],
            estimated_duration_ms=0,
            detail=f"热键 {hotkey_id} 已触发（VTS 侧异步执行，时长未知）",
        )

    async def _read_model_id(self) -> str | None:
        """尽力读取当前加载的模型 ID（取不到置 None，不拖垮 connect）。

        先 ``CurrentModelRequest``；失败回退 ``AvailableModelsRequest``
        （筛 ``modelLoaded``，响应形状对齐 ``vts._ensure_model`` 的既有消费）。
        """
        assert self.sink is not None and self.sink.api is not None
        api = self.sink.api
        try:
            data = await api.request("CurrentModelRequest")
        except (VtsApiError, TimeoutError) as exc:
            logger.warning("CurrentModelRequest 失败（回退 AvailableModels）：%r", exc)
        else:
            model_id = data.get("modelID")
            if data.get("modelLoaded") is True and model_id:
                return str(model_id)
            return None
        try:
            data = await api.request("AvailableModelsRequest")
        except (VtsApiError, TimeoutError) as exc:
            logger.warning("AvailableModelsRequest 失败——model_id 置 None：%r", exc)
            return None
        loaded = next(
            (m for m in data.get("availableModels", []) if m.get("modelLoaded") is True),
            None,
        )
        model_id = loaded.get("modelID") if loaded is not None else None
        return str(model_id) if model_id else None

    async def _enumerate_hotkeys(self) -> list[HotkeyInfo]:
        """枚举当前模型热键并按类型白名单过滤（AD-7）。

        枚举失败按无热键处理（warning + 空表，不拖垮连接——热键是可选增强）；
        可经 ``list_catalog(refresh=True)`` 重试。
        """
        assert self.sink is not None and self.sink.api is not None
        try:
            data = await self.sink.api.request("HotkeysInCurrentModelRequest")
        except (VtsApiError, TimeoutError) as exc:
            logger.warning(
                "热键枚举失败——按无热键处理（可 behavior_list(refresh=True) 重试）：%r",
                exc,
            )
            return []
        raw = data.get("availableHotkeys", [])
        hotkeys: list[HotkeyInfo] = []
        for item in raw:
            hk_type = str(item.get("type", ""))
            if hk_type not in HOTKEY_TYPES:
                continue
            hotkeys.append(
                HotkeyInfo(
                    hotkey_id=str(item.get("hotkeyID", "")),
                    name=str(item.get("name", "")),
                    type=hk_type,
                    file=str(item.get("file", "")),
                    kind=HOTKEY_KIND_BY_TYPE[hk_type],
                )
            )
        logger.info("热键枚举完成：VTS 报 %d 个，类型白名单过滤后 %d 个。", len(raw), len(hotkeys))
        return hotkeys
