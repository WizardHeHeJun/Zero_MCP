"""VTS 离散行为 MCP Server（FastMCP stdio · 蓝图 2026-07-31 §5/§6 · T5；
语音播放 speech_play 见 speech-play 蓝图 2026-08-14 §T5）。

feature flag：VTS_BEHAVIOR_ENABLED（默认 false）；speech_play 额外要求
VTS_SPEECH_ENABLED（默认 false）——双 flag 复合门，见 `_require_speech_enabled`。
传输：stdio（供 Zero 侧经 MCP client spawn 子进程）。

设计约束（AD-8，逐条对照 desktop_mcp_server.py 先例）：
- 传输层零业务逻辑：工具体只做参数转换 + 转发 + 错误映射。
- 业务全在 src/mcp/behavior/service.py（模块级惰性 global + 延迟 import）。
- VTS_BEHAVIOR_ENABLED=false 时始终注册工具、运行时首行 raise（更易测）。
- 机读错误码走位置无关令牌 [vtsb:*]（AD-11，符号唯一真相在
  src/agents/models/vts_behavior.py）：业务性拒绝在回执 code 字段（正常返回），
  协议性失败才抛 ToolError（服务层的 ToolError 原样透传，其余异常映射
  [vtsb:vts_error]）。speech_play 是例外（AD-4）：全部失败路径统一走
  ToolError，见 vts_behavior.py 模块 docstring 码表 speech_play 例外条。
- ⚠ stdout 是 JSON-RPC 线路，绝不可写——日志一律 stderr（configure_logging）。
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import ValidationError

from src.agents.models.vts_behavior import (
    INTENSITY_DEFAULT,
    VTSB_DISABLED,
    VTSB_INVALID_PARAMS,
    VTSB_SPEECH_DISABLED,
    VTSB_VTS_ERROR,
    BehaviorRequest,
    SpeechRequest,
    TrajectoryKeyframe,
    TrajectoryRequest,
)

if TYPE_CHECKING:  # 仅类型标注用——运行时经 _get_service() 延迟 import
    from src.mcp.behavior.service import BehaviorService

logger = logging.getLogger(__name__)

# ── 全局状态 ──────────────────────────────────────────────────────────────────

_SERVICE: BehaviorService | None = None

mcp = FastMCP(
    name="vts-behavior",
    instructions=(
        "VTube Studio 离散行为执行层：以 12 个离散行为词驱动 Live2D 皮套做"
        "点头/摇头/歪头/瞥视/眨眼/扬眉/皱眉/瞪眼/微笑/前倾/后撤/摇摆等瞬态动作，"
        "并可触发 VTS 侧已配置的热键动画。仅 VTS_BEHAVIOR_ENABLED=true 时生效。\n"
        "用法：① 先 vts_connect 建立连接（幂等；渲染循环故障后再次调用即显式重连）。"
        "② behavior_list 取词表——每词含定义文本/参数 schema/典型时长/冷却/降级态，"
        "已发现的 VTS 热键在同一张清单（经 behavior_trigger 以 name='hotkey:<hotkeyID>' "
        "触发）。③ behavior_trigger 触发行为并读回执：status=accepted/replaced 表示已"
        "执行，rejected 是正常业务回执（code 带 [vtsb:*] 机读令牌，如冷却/节流/通道"
        "占用），不是错误——行为不排队，被拒后不必立即重试。④ 话锋突转时用 "
        "behavior_interrupt 打断活跃行为；behavior_status 探测连接/健康态。"
        "⑤ speech_play(wav_path, mouth_track, fps) 播放同机绝对路径的 wav"
        "（PCM 16-bit / mono / 44100Hz）并同步注入口型（mouth_track 形状同 "
        "params_animate 的 keyframes，只含嘴部参数键）：完成调度即返 "
        "{accepted, duration_ms}（不等播完，按 duration_ms 节流下一次调用）；"
        "重叠调用按 FIFO 排队顺序播放（不打断在播）；播放期嘴部参数独占，播完/失败"
        "即释放。仅 VTS_BEHAVIOR_ENABLED 与 VTS_SPEECH_ENABLED 均为 true 时可用，"
        "失败一律 ToolError（无 rejected 业务回执）。\n"
        "⚠ 跨进程共存警告：本 server 与另一进程的表情流 sink（VTS_SINK_ENABLED）"
        "同时连 VTS 会形成两个插件对同一参数的 set 模式独占冲突（VTS 454）——"
        "同一时刻只允许一个进程持有 VTS 注入连接；同进程共存请注入同一 sink 实例。"
    ),
)


def _is_enabled() -> bool:
    """检查 VTS_BEHAVIOR_ENABLED feature flag（真值集与仓内先例一致）。"""
    return os.environ.get("VTS_BEHAVIOR_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
    }


def _require_enabled() -> None:
    """feature flag 未开时 raise ToolError（带 [vtsb:disabled] 机读令牌）。"""
    if not _is_enabled():
        raise ToolError(
            f"{VTSB_DISABLED} VTS 行为能力未启用（VTS_BEHAVIOR_ENABLED=false）。"
            "请在 .env 中设置 VTS_BEHAVIOR_ENABLED=true 后重启 server。"
        )


def _is_speech_enabled() -> bool:
    """检查 VTS_SPEECH_ENABLED feature flag（真值集与仓内先例一致）。"""
    return os.environ.get("VTS_SPEECH_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
    }


def _require_speech_enabled() -> None:
    """speech_play 专属第二道门：VTS_SPEECH_ENABLED 未开时 raise ToolError
    （带 [vtsb:speech_disabled] 机读令牌）。

    speech_play 需要 ``VTS_BEHAVIOR_ENABLED`` 与 ``VTS_SPEECH_ENABLED`` 两个
    flag 皆开（speech-play 蓝图 AD-9：第一句既有 `_require_enabled()` 不变 +
    第二句本函数，**顺序复合门**，不合并成一个新函数）——工具体里必须作为
    **第二条**裸调用紧跟 `_require_enabled()`，前面不可插入任何其它可执行
    语句：这是 `_require_enabled` 那道 AST 硬化守卫
    （`test_flag_off_tools_gate_is_first_executable_statement`）覆盖不到的
    那一半，需要一条平行守卫钉住"第二条也是裸门"（不修改既有守卫本身）。
    """
    if not _is_speech_enabled():
        raise ToolError(
            f"{VTSB_SPEECH_DISABLED} 语音播放能力未启用（VTS_SPEECH_ENABLED=false）。"
            "请在 .env 中设置 VTS_SPEECH_ENABLED=true（并确保 VTS_BEHAVIOR_ENABLED=true）"
            "后重启 server。"
        )


def _get_service() -> BehaviorService:
    """获取模块级惰性 BehaviorService 单例（延迟 import：flag 关时不拉业务依赖）。"""
    global _SERVICE
    if _SERVICE is None:
        from src.mcp.behavior.service import BehaviorService  # noqa: PLC0415

        _SERVICE = BehaviorService()
    return _SERVICE


def _dump_model(obj: Any) -> str:
    """将 pydantic 模型序列化为 JSON 字符串（类型安全包装）。"""
    result: str = obj.model_dump_json()
    return result


# ── 行为工具 ──────────────────────────────────────────────────────────────────


@mcp.tool(
    name="behavior_list",
    description=(
        "获取离散行为词表 + 已发现 VTS 热键的同一张清单（BehaviorCatalog JSON）。"
        "词表是静态知识，未连接也完整返回（hotkeys=null 表示尚未枚举）；"
        "refresh=true 且已连接时重枚举热键（用户中途换模型场景）。"
        "示例词：nod（肯定点头）、eyes_widen（震惊瞪眼）、body_sway（愉悦摇摆）。"
    ),
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
)
async def behavior_list(refresh: bool = False) -> str:
    """列出行为词表与热键。

    Args:
        refresh: 已连接时重枚举热键（只刷热键不刷参数值域）。

    Returns:
        BehaviorCatalog 序列化 JSON。
    """
    _require_enabled()
    service = _get_service()
    try:
        catalog = await service.list_catalog(refresh=refresh)
        return _dump_model(catalog)
    except ToolError:
        raise
    except Exception as exc:
        logger.error("behavior_list 失败：%s", exc, exc_info=True)
        raise ToolError(f"{VTSB_VTS_ERROR} behavior_list 执行失败：{exc}") from exc


@mcp.tool(
    name="behavior_trigger",
    description=(
        "触发一个离散行为词（词表见 behavior_list）或 VTS 热键"
        "（name='hotkey:<hotkeyID>'），返回 BehaviorReceipt JSON 三态回执："
        "accepted/replaced=已执行，rejected=正常业务拒绝（code 带 [vtsb:*] 机读令牌，"
        "非错误）。示例：name='nod' repeat=2（连点两下头）；"
        "name='glance' direction='left'（向左瞥一眼）。"
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
)
async def behavior_trigger(
    name: str,
    intensity: float = INTENSITY_DEFAULT,
    repeat: int = 1,
    duration_ms: int | None = None,
    direction: str | None = None,
) -> str:
    """触发离散行为或热键。

    Args:
        name: 行为词（如 "nod"）或 "hotkey:<hotkeyID>"。
        intensity: 幅度 [0, 1]（缺省 0.5）。
        repeat: stroke 周期数 [1, 8]（仅节律行为有意义）。
        duration_ms: 覆盖词表典型时长（None=典型值，上限 10000）。
        direction: 方向词（head_tilt: left|right；glance: left|right|up|down）。

    Returns:
        BehaviorReceipt 序列化 JSON。
    """
    _require_enabled()
    service = _get_service()
    try:
        request = BehaviorRequest(
            name=name,
            intensity=intensity,
            repeat=repeat,
            duration_ms=duration_ms,
            direction=direction,
        )
    except ValidationError as exc:
        raise ToolError(f"{VTSB_INVALID_PARAMS} 触发参数不合法：{exc}") from exc
    try:
        receipt = await service.trigger(request)
        return _dump_model(receipt)
    except ToolError:
        raise
    except Exception as exc:
        logger.error("behavior_trigger 失败：%s", exc, exc_info=True)
        raise ToolError(f"{VTSB_VTS_ERROR} behavior_trigger 执行失败：{exc}") from exc


@mcp.tool(
    name="behavior_interrupt",
    description=(
        "打断活跃行为（交叉淡出回语义静息基准），话锋突转时用。channel=null 清全部，"
        "也可只清指定通道（head/gaze/eyelid/brows/mouth/body）。只清手势层，"
        "不触碰表情通路。返回 BehaviorReceipt JSON。"
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
)
async def behavior_interrupt(channel: str | None = None) -> str:
    """打断活跃行为。

    Args:
        channel: 目标通道（None=全部；合法值 head/gaze/eyelid/brows/mouth/body）。

    Returns:
        BehaviorReceipt 序列化 JSON（幂等：无活跃行为也返回 accepted）。
    """
    _require_enabled()
    service = _get_service()
    try:
        receipt = service.interrupt(channel)
        return _dump_model(receipt)
    except ToolError:
        raise
    except Exception as exc:
        logger.error("behavior_interrupt 失败：%s", exc, exc_info=True)
        raise ToolError(f"{VTSB_VTS_ERROR} behavior_interrupt 执行失败：{exc}") from exc


@mcp.tool(
    name="behavior_status",
    description=(
        "获取行为层健康/状态快照（BehaviorStatus JSON）：connected/healthy/last_error/"
        "活跃行为/冷却剩余/缺席可选参数/热键数/model_id/trajectory_active/"
        "trajectory_remaining_ms（轨迹回放态，含交还缓出窗口）/speech_active/"
        "speech_queue_depth/speech_last_error（语音播放态）。healthy=false 表示"
        "渲染循环已故障——恢复路径为再次 vts_connect（显式重连）。"
    ),
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
)
async def behavior_status() -> str:
    """获取行为层状态快照。

    Returns:
        BehaviorStatus 序列化 JSON。
    """
    _require_enabled()
    service = _get_service()
    try:
        return _dump_model(service.status())
    except ToolError:
        raise
    except Exception as exc:
        logger.error("behavior_status 失败：%s", exc, exc_info=True)
        raise ToolError(f"{VTSB_VTS_ERROR} behavior_status 执行失败：{exc}") from exc


# ── 裸参数轨迹工具（2026-07-31 二期：Zero 侧动作模型直驱） ────────────────────


@mcp.tool(
    name="params_list",
    description=(
        "获取所连 VTS 部署的全量输入参数表（ParamCatalog JSON：name/min/max/"
        "default_value/governed）——动作模型的作用空间。governed=true 的参数由表情"
        "通路恒定注入（轨迹接管时按强度混合），其余参数按需注入、停喂 1s 后 VTS "
        "自动收回。未连接时 params=null。"
    ),
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
)
async def params_list() -> str:
    """列出全量输入参数量程表。

    Returns:
        ParamCatalog 序列化 JSON。
    """
    _require_enabled()
    service = _get_service()
    try:
        return _dump_model(service.list_params())
    except ToolError:
        raise
    except Exception as exc:
        logger.error("params_list 失败：%s", exc, exc_info=True)
        raise ToolError(f"{VTSB_VTS_ERROR} params_list 执行失败：{exc}") from exc


@mcp.tool(
    name="params_animate",
    description=(
        "投喂一段参数关键帧轨迹（动作模型输出），渲染循环按时间轴插值回放注入 VTS。"
        "keyframes=[{t_ms, params:{参数名:值}}]（t_ms 严格升序、同段键集一致、"
        "单段≤10s）；mode='absolute'（按值接管，渐入渐出无跳变）或 'offset'（在表情"
        "基线上加性叠加）；append=true 排队无缝续接（流式投喂），false 清队即刻接管。"
        "返回 TrajectoryReceipt JSON：rejected 为正常业务拒绝（code 带 [vtsb:*]，"
        "queue_depth 供背压退避），非错误。"
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
)
async def params_animate(
    keyframes: list[TrajectoryKeyframe],
    mode: str = "absolute",
    append: bool = True,
) -> str:
    """投喂轨迹段。

    Args:
        keyframes: 关键帧列表（t_ms 相对段起点，params 为该时刻各参数值）。
        mode: absolute（接管）| offset（叠加）。
        append: True=队尾续接；False=清队即刻接管。

    Returns:
        TrajectoryReceipt 序列化 JSON。
    """
    _require_enabled()
    service = _get_service()
    try:
        request = TrajectoryRequest(keyframes=keyframes, mode=mode, append=append)
    except ValidationError as exc:
        raise ToolError(f"{VTSB_INVALID_PARAMS} 轨迹参数不合法：{exc}") from exc
    try:
        return _dump_model(service.animate(request))
    except ToolError:
        raise
    except Exception as exc:
        logger.error("params_animate 失败：%s", exc, exc_info=True)
        raise ToolError(f"{VTSB_VTS_ERROR} params_animate 执行失败：{exc}") from exc


@mcp.tool(
    name="params_clear",
    description=(
        "清除轨迹队列并交还参数控制权（幂等；250ms 缓出无跳变，非 governed 参数"
        "停喂 1s 后由 VTS 收回）。返回 TrajectoryReceipt JSON。"
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
)
async def params_clear() -> str:
    """清除轨迹并交还参数。

    Returns:
        TrajectoryReceipt 序列化 JSON。
    """
    _require_enabled()
    service = _get_service()
    try:
        return _dump_model(service.clear_params())
    except ToolError:
        raise
    except Exception as exc:
        logger.error("params_clear 失败：%s", exc, exc_info=True)
        raise ToolError(f"{VTSB_VTS_ERROR} params_clear 执行失败：{exc}") from exc


# ── 语音播放工具（speech-play 蓝图 2026-08-14 §T5） ───────────────────────────


@mcp.tool(
    name="speech_play",
    description=(
        "播放一段语音并同步注入口型（Zero 侧语音合成 → 我方渲染端播放）。"
        "wav_path 为同机绝对路径（PCM 16-bit / mono / 44100Hz，不重采样）；"
        "mouth_track 形状同 params_animate 的 keyframes"
        "（[{t_ms, params:{参数名:值}}]，t=0 对齐音频首采样，只含嘴部参数键）；"
        "fps 为采样率提示（默认 20）。返回 {accepted, duration_ms}——完成调度即返，"
        "不等播完，按 duration_ms 节流下一次调用；重叠调用按 FIFO 排队顺序播放"
        "（不打断在播）。播放期间 mouth_track 涉及的键独占，不被行为/表情层覆盖，"
        "播完/失败即释放。失败一律 ToolError（机读令牌区分文件/格式/设备/未连接"
        "四类，本工具没有 rejected 业务回执态）；仅 VTS_BEHAVIOR_ENABLED 与 "
        "VTS_SPEECH_ENABLED 均为 true 时可用。"
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
)
async def speech_play(
    wav_path: str,
    mouth_track: list[TrajectoryKeyframe],
    fps: float = 20.0,
) -> str:
    """播放语音并注入口型。

    Args:
        wav_path: 同机绝对路径的 wav 文件（PCM 16-bit / mono / 44100Hz）。
        mouth_track: 口型关键帧轨迹（形状同 params_animate 的 keyframes）。
        fps: 轨迹采样率提示（默认 20，不强制校验帧间隔）。

    Returns:
        SpeechReceipt 序列化 JSON（``{"accepted": true, "duration_ms": <float>}``）。
    """
    _require_enabled()
    _require_speech_enabled()
    service = _get_service()
    try:
        request = SpeechRequest(wav_path=wav_path, mouth_track=mouth_track, fps=fps)
    except ValidationError as exc:
        raise ToolError(f"{VTSB_INVALID_PARAMS} 语音播放参数不合法：{exc}") from exc
    try:
        receipt = await service.speech_play(request)
        return _dump_model(receipt)
    except ToolError:
        raise
    except Exception as exc:
        logger.error("speech_play 失败：%s", exc, exc_info=True)
        raise ToolError(f"{VTSB_VTS_ERROR} speech_play 执行失败：{exc}") from exc


# ── 连接管理工具 ──────────────────────────────────────────────────────────────


@mcp.tool(
    name="vts_connect",
    description=(
        "连接 VTube Studio 并挂载手势引擎（幂等）：连接/认证/读参数值域/起渲染循环 + "
        "热键枚举。渲染循环故障后（behavior_status.healthy=false）再次调用即显式重连。"
        "返回 BehaviorStatus JSON。"
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
)
async def vts_connect() -> str:
    """连接 VTS（幂等；失败抛 ToolError [vtsb:vts_error]）。

    Returns:
        BehaviorStatus 序列化 JSON。
    """
    _require_enabled()
    service = _get_service()
    try:
        return _dump_model(await service.connect())
    except ToolError:
        raise
    except Exception as exc:
        logger.error("vts_connect 失败：%s", exc, exc_info=True)
        raise ToolError(f"{VTSB_VTS_ERROR} vts_connect 执行失败：{exc}") from exc


@mcp.tool(
    name="vts_disconnect",
    description=(
        "断开 VTube Studio（幂等）：停渲染循环、关 WebSocket，1s 后 VTS 收回参数"
        "控制权。返回 BehaviorStatus JSON。"
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
)
async def vts_disconnect() -> str:
    """断开 VTS（幂等）。

    Returns:
        BehaviorStatus 序列化 JSON。
    """
    _require_enabled()
    service = _get_service()
    try:
        return _dump_model(await service.disconnect())
    except ToolError:
        raise
    except Exception as exc:
        logger.error("vts_disconnect 失败：%s", exc, exc_info=True)
        raise ToolError(f"{VTSB_VTS_ERROR} vts_disconnect 执行失败：{exc}") from exc


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.logging_config import configure_logging  # noqa: PLC0415
    from src.mcp.native_warmup import warm_native_extensions  # noqa: PLC0415

    # 接管 root：FastMCP 构造时 SDK 已抢注 RichHandler(stderr)——configure_logging
    # 摘掉抢注 handler，统一为 LOG_FORMAT + stderr（stdout 是 JSON-RPC 线路，绝不可写；
    # 不接管会 stderr 双份日志）。ZERO_MCP_LOG_LEVEL / ZERO_MCP_LOG_FILE 可调级别与落盘，
    # 见 .env.example。
    configure_logging()

    enabled = _is_enabled()
    speech_enabled = _is_speech_enabled()
    logger.info(
        "vts_behavior_mcp_server 启动：VTS_BEHAVIOR_ENABLED=%s，VTS_SPEECH_ENABLED=%s",
        enabled,
        speech_enabled,
    )

    if enabled:
        # 原生扩展预热（必须在 mcp.run() 之前）：_get_service() 的延迟 import 会
        # 经 src.mcp.zero 链路首次拉起 numpy，而事件循环起来之后再首次 import
        # numpy 会无限期死锁——Zero 2026-08-11 通报的 vts_connect 挂起即此。
        # 判据与最小复现见 src/mcp/native_warmup.py 模块 docstring。
        # 仍在 flag 内：VTS_BEHAVIOR_ENABLED=false 时不拉业务依赖（零回归不变）。
        warm_native_extensions(("src.mcp.behavior.service",))

        if speech_enabled:
            # speech_playback 整模块预热（speech-play 蓝图 AD-11）：sounddevice
            # 延迟 import 于该模块内，同一死锁判据（"是否首次触达"而非"触达的是
            # 谁"）。嵌套在 `if enabled:` 内层——ast.walk 递归找嵌套调用，既有
            # test_warmup_is_gated_by_feature_flag 不需要改。VTS_SPEECH_ENABLED=false
            # 时不拉 sounddevice（零回归不变）。
            warm_native_extensions(("src.mcp.behavior.speech_playback",))

        # 惰性持有 sink（AD-9）：不在启动时连 VTS——VTS 未启动/未开 API 也能起
        # server，连接由 vts_connect 工具显式建立（连接失败只在该工具报错）。
        logger.info("行为层就绪：经 vts_connect 显式建立 VTS 连接（惰性持有 sink）。")
    else:
        logger.warning(
            "VTS_BEHAVIOR_ENABLED=false：所有工具调用将被拒绝（运行时检查），"
            "设置为 true 并重启以启用 VTS 行为能力。"
        )

    mcp.run(transport="stdio")
