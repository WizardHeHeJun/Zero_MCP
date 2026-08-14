"""VTS 离散行为层契约数据模型（蓝图 2026-07-31-vts-discrete-behavior §2/§5 · T1）。

契约的**唯一真相**——MCP 行为层（`src/mcp/behavior/service.py`、
`src/mcp/vts_behavior_mcp_server.py`、`src/mcp/zero/sinks/behavior_overlay.py`）与
未来编排层共同 import 此模块，不得在其他地方重复定义这些数据形状
（共享契约层豁免判据见 `rules/project-root.md`：无业务逻辑、无上层依赖）。

## 机读错误码表（AD-11，按本模块符号名 pin）

码值一律为**位置无关令牌** `[vtsb:<code>]`：FastMCP 会把 ToolError 加壳为
``"Error executing tool <name>: <原文>"``，位置 0 的裸前缀判据在真 wire 上恒 False
（`rules/mcp-integration.md` 教训）。消费方一律 `re.search` 提取——判据的唯一真相是
本模块 `VTSB_CODE_RE` / `extract_vtsb_code`，测试夹具必须用加壳后的真 wire 形态。

| 符号 | 语义 | 载体 |
| --- | --- | --- |
| `VTSB_DISABLED` | feature flag（`VTS_BEHAVIOR_ENABLED`）未开 | ToolError |
| `VTSB_NOT_CONNECTED` | 未 `vts_connect` 或渲染循环已故障（detail 带 last_error） | ToolError |
| `VTSB_UNKNOWN_BEHAVIOR` | 词表外行为名 | 回执 rejected |
| `VTSB_INVALID_PARAMS` | 数值越界 / 非法 direction（两种载体，见下方） | ToolError / rejected |
| `VTSB_COOLDOWN` | per-behavior / 热键冷却未过（detail 带剩余 ms） | 回执 rejected |
| `VTSB_THROTTLED` | 全局节流（两次 accepted 最小间隔）拒绝 | 回执 rejected |
| `VTSB_CHANNEL_BUSY` | 同通道被更高优先级行为占用 | 回执 rejected |
| `VTSB_HOTKEY_UNAVAILABLE` | 热键失效 / 无此 ID / 热键开关关 | 回执 rejected |
| `VTSB_VTS_ERROR` | VTS APIError 透传（detail 带原始错误） | ToolError |
| `VTSB_SPEECH_DISABLED` | feature flag（`VTS_SPEECH_ENABLED`）未开 | ToolError |
| `VTSB_SPEECH_FILE_ERROR` | wav 路径不存在 / 非绝对路径 / 不可读 | ToolError |
| `VTSB_SPEECH_FORMAT_ERROR` | wav 非 PCM16/mono/44100，或文件损坏 | ToolError |
| `VTSB_SPEECH_DEVICE_ERROR` | 播放设备探测/起播失败 | ToolError |

**业务性拒绝进 `BehaviorReceipt.code` 字段而非 ToolError**（业务拒绝是正常回执不是
协议错误，AD-11）；协议性失败（disabled / not_connected / vts_error）才抛 ToolError。

`speech_play` 是本表的**例外**（speech-play 蓝图 2026-08-14 AD-4）：其全部失败路径
（含队列满 `VTSB_THROTTLED`）统一走 ToolError——`SpeechReceipt` 的成功形状是
Zero 已锁定的字面 `{accepted, duration_ms}`，没有 `code` 字段容纳业务性拒绝，
四类失败（文件/格式/未连接/设备）与队列满同属"调用方需先改数据/等资源才可能
重试"，与 `VTSB_INVALID_PARAMS` 数值越界走 ToolError 是同一原则的延伸而非违反。
"""

from __future__ import annotations

import math
import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# 机读错误码（AD-11）——消费方按本符号名引用，不得手写字符串字面量
# ---------------------------------------------------------------------------

VTSB_DISABLED: str = "[vtsb:disabled]"
"""feature flag（VTS_BEHAVIOR_ENABLED）未开（ToolError 载体）。"""

VTSB_NOT_CONNECTED: str = "[vtsb:not_connected]"
"""未 vts_connect 或渲染循环已故障（ToolError 载体，detail 带 last_error）。"""

VTSB_UNKNOWN_BEHAVIOR: str = "[vtsb:unknown_behavior]"
"""词表外行为名（回执 rejected 载体——不是解析层 422，见 BehaviorRequest.name）。"""

VTSB_INVALID_PARAMS: str = "[vtsb:invalid_params]"
"""参数不合法——**两种载体**（code-review W4 修订：显式区分，不拆码，令牌
保持稳定，两种情形共用同一个消费方判据）：

① **数值越界**（`intensity`/`repeat`/`duration_ms` 超出 `BehaviorRequest` 声明的
   合法区间，定义域见该模型 `model_validator`）——解析层（pydantic）拒收，
   `behavior_trigger` 工具体捕获 `ValidationError` 转 **ToolError**（server 侧，
   蓝图 §5 AD-8）。

② **未知/非法 direction**（合法值集随行为词而异，如 `head_tilt` 只认
   left/right、`glance` 认 4 向——契约层 `BehaviorRequest` 有意不管此项）——
   执行层 `BehaviorOverlayEngine.trigger` 语义校验后落**回执 rejected**
   （`code` 字段，业务性拒绝不是协议错误，AD-11）。
"""

VTSB_COOLDOWN: str = "[vtsb:cooldown]"
"""per-behavior / 热键冷却未过（回执 rejected 载体，detail 带剩余 ms）。"""

VTSB_THROTTLED: str = "[vtsb:throttled]"
"""全局节流拒绝（回执 rejected 载体）。"""

VTSB_CHANNEL_BUSY: str = "[vtsb:channel_busy]"
"""同通道被更高优先级活跃行为占用（回执 rejected 载体）。"""

VTSB_HOTKEY_UNAVAILABLE: str = "[vtsb:hotkey_unavailable]"
"""热键失效 / 无此 ID / VTS_BEHAVIOR_HOTKEYS 关（回执 rejected 载体）。"""

VTSB_VTS_ERROR: str = "[vtsb:vts_error]"
"""VTS APIError 透传（ToolError 载体，detail 带原始错误）。"""

VTSB_SPEECH_DISABLED: str = "[vtsb:speech_disabled]"
"""feature flag（VTS_SPEECH_ENABLED）未开（ToolError 载体）——speech_play 独有的
第二道门，见 vts_behavior_mcp_server._require_speech_enabled（speech-play 蓝图 AD-9）。"""

VTSB_SPEECH_FILE_ERROR: str = "[vtsb:speech_file_error]"
"""wav 路径不存在 / 非绝对路径 / 不可读（ToolError 载体）。"""

VTSB_SPEECH_FORMAT_ERROR: str = "[vtsb:speech_format_error]"
"""wav 非 PCM16/mono/44100Hz，或文件头/帧数据损坏（ToolError 载体）。"""

VTSB_SPEECH_DEVICE_ERROR: str = "[vtsb:speech_device_error]"
"""播放设备探测（check_output_settings）或起播失败（ToolError 载体）。"""

VTSB_CODE_RE: re.Pattern[str] = re.compile(r"\[vtsb:[a-z_]+\]")
"""位置无关令牌判据的唯一真相——消费方/测试一律用它 search，不用 startswith。"""


def extract_vtsb_code(text: str) -> str | None:
    """从任意 wire 文本（含 FastMCP 加壳形态）提取首个完整 `[vtsb:*]` 令牌。

    返回值可直接与本模块 `VTSB_*` 常量比对；无令牌返回 None。
    纯函数、位置无关——`"Error executing tool x: [vtsb:disabled] ..."` 同样可提取。
    """
    match = VTSB_CODE_RE.search(text)
    return match.group(0) if match else None


# ---------------------------------------------------------------------------
# 契约范围常量（蓝图 §2 参数三件套的边界，按符号名 pin）
# ---------------------------------------------------------------------------

INTENSITY_DEFAULT: float = 0.5
"""intensity 缺省值（BML `amount` 惯例的中点）。"""

REPEAT_MIN: int = 1
REPEAT_MAX: int = 8
"""repeat（stroke 周期数，BML `repetition` 惯例）合法区间 [1, 8]。"""

DURATION_MS_MAX: int = 10_000
"""duration_ms 覆盖值上限（防单次行为长期霸占通道）。"""

RECEIPT_STATUSES: frozenset[str] = frozenset({"accepted", "replaced", "rejected"})
"""回执三态（AD-6）。本仓自产自校（非跨仓拒收面，加态是本仓同步改动，无部署错位风险）。"""

# ---------------------------------------------------------------------------
# 语音播放契约范围常量（speech-play 蓝图 2026-08-14 §T1）
# ---------------------------------------------------------------------------

SPEECH_WAV_SAMPLE_RATE: int = 44100
"""wav 规格：采样率（Zero 侧 Bert-VITS2 原生输出，我方不重采样）。"""

SPEECH_WAV_CHANNELS: int = 1
"""wav 规格：声道数（mono）。"""

SPEECH_WAV_SAMPLE_WIDTH: int = 2
"""wav 规格：采样位宽（字节，2=PCM 16-bit）。"""

SPEECH_MAX_QUEUE: int = 5
"""语音播放队列限深（我方自加防御，Zero 规范未提）——满时 ToolError
`VTSB_THROTTLED`（speech-play 蓝图 AD-4：本工具无 rejected 回执容器容纳它）。"""

SPEECH_TRACK_MAX_SEGMENT_MS: int = 60_000
"""mouth_track 单段时长上限（speech 专属，2026-08-14 应 Zero 回件放宽）：
Zero 实测 0.17s/字、chat 回复 50–150 字 ⇒ 8.5–25.5s 常态超原 10s 继承限
（`TRAJECTORY_MAX_SEGMENT_MS` 只管 params_animate，不变）。取定值 60s 而非
「wav 时长派生」：后者把校验拖进 I/O 层且令 fps 从纯提示字段变语义量——
Zero 二选一里两者皆可（其回件明示定值 ≥60s 可接受）。"""

SPEECH_TRACK_MAX_KEYFRAMES: int = 1200
"""mouth_track 单段帧数上限（speech 专属）：20Hz × 60s，与时长上限同源配比
（Zero 回件同款算术）。防超大 payload 的职责不变，只是量程换成语音整句。"""


# ---------------------------------------------------------------------------
# 触发方向（Zero 侧 LLM → MCP）
# ---------------------------------------------------------------------------


class BehaviorRequest(BaseModel):
    """行为触发请求（Zero 侧 LLM → MCP 方向，`behavior_trigger` 工具输入）。

    - name: 行为词（蓝图 §2 词表 12 词）或 `hotkey:<hotkeyID>`（AD-7 统一命名空间）。
      **有意不用 `Literal` 硬拒**（延续 zero_affect 量纲兄弟键教训：拒收面在「对方先发、
      我方后收」的部署错位方向上会把 additive 演进变成 breaking change）——词表加词后
      旧解析层不得炸；未知名由执行侧返回 rejected 回执 + `VTSB_UNKNOWN_BEHAVIOR`
      （可观测、可降级），而非解析层 422。
    - intensity: 幅度 [0, 1]（BML `amount` 惯例），缺省 `INTENSITY_DEFAULT`。
    - repeat: stroke 周期数 [`REPEAT_MIN`, `REPEAT_MAX`]（BML `repetition` 惯例）。
    - duration_ms: 覆盖词表典型时长；None = 用典型值；上限 `DURATION_MS_MAX`。
    - direction: 方向词（head_tilt: left/right；glance: 4 向）。合法值集**随行为词而异**，
      故不在契约层校验——未知值由执行侧回 rejected + `VTSB_INVALID_PARAMS`。
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    intensity: float = INTENSITY_DEFAULT
    repeat: int = 1
    duration_ms: int | None = None
    direction: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> BehaviorRequest:
        if not self.name:
            raise ValueError("name 不得为空")
        # ⚠ 必须用 isfinite 显式判 NaN：比较式对 NaN 恒 False，NaN 幅度会静默通过
        # 并一路进入包络合成（zero_affect.ModalityPrior 同款纪律）。
        if not math.isfinite(self.intensity):
            raise ValueError("intensity 必须为有限值（NaN/inf 拒收）")
        if not (0.0 <= self.intensity <= 1.0):
            raise ValueError(f"intensity={self.intensity} 超出 [0, 1]")
        if not (REPEAT_MIN <= self.repeat <= REPEAT_MAX):
            raise ValueError(f"repeat={self.repeat} 超出 [{REPEAT_MIN}, {REPEAT_MAX}]")
        if self.duration_ms is not None and not (1 <= self.duration_ms <= DURATION_MS_MAX):
            raise ValueError(f"duration_ms={self.duration_ms} 超出 [1, {DURATION_MS_MAX}]")
        return self


# ---------------------------------------------------------------------------
# 回执方向（MCP → Zero 侧 LLM）
# ---------------------------------------------------------------------------


class BehaviorReceipt(BaseModel):
    """行为触发/中断回执（MCP → Zero 侧 LLM 方向，AD-6 三态）。

    - accepted: 已接受并开始执行；
    - replaced: 已接受，且按同通道优先级抢占了活跃行为（旧包络交叉淡出）；
    - rejected: 业务性拒绝——`code` 带机读令牌（见模块 docstring 码表）、`detail`
      带人读原因。**业务性拒绝是正常回执不是协议错误，不抛 ToolError**（AD-11）。

    channels 为触发时实际声明的通道集（降级会改变它，如 glance 借 head）；
    degraded_channels 非空表示发生了参数降级（如所连部署缺 BodyAngle，body→head 近似）。
    """

    model_config = ConfigDict(extra="forbid")

    status: str
    behavior_id: str
    channels: list[str] = Field(default_factory=list)
    estimated_duration_ms: int = 0
    degraded_channels: list[str] = Field(default_factory=list)
    code: str | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> BehaviorReceipt:
        if self.status not in RECEIPT_STATUSES:
            raise ValueError(f"status {self.status!r} 不在 {sorted(RECEIPT_STATUSES)} 内")
        return self


# ---------------------------------------------------------------------------
# 清单（behavior_list 输出）
# ---------------------------------------------------------------------------


class BehaviorInfo(BaseModel):
    """程序化词表单条（`behavior_list` 输出；definition 即交付 Zero 侧 LLM 的 prompt 素材）。

    - params_schema: 参数名 → 人读摘要（如 `{"direction": "left|right"}`）——
      按 Torshizi et al. 2025「定义 + 少量示例」形态交付，示例放工具 description。
    - available/degraded: 按所连部署的实际参数集计算——body 词缺 BodyAngle 时
      available 仍 True 但 degraded 说明降级方式；仅当完全无法执行才 False。
    - typical_duration_ms/cooldown_s: 词表典型值（v1 全部为工程假设，标定后回写）。
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    definition: str
    params_schema: dict[str, str] = Field(default_factory=dict)
    typical_duration_ms: int
    cooldown_s: float
    channels: list[str]
    available: bool = True
    degraded: str | None = None


class HotkeyInfo(BaseModel):
    """已发现的 VTS 热键单条（AD-7 可选增强，与程序化词表同一张清单呈现）。

    - hotkey_id/name/type/file: 来自 `HotkeysInCurrentModelRequest` 响应原样字段；
      type 为 VTS 原生枚举（TriggerAnimation/ToggleExpression），**str 不 Literal**
      （同 BehaviorRequest.name 的拒收面理由——VTS 加类型不得炸我方解析层）。
    - kind: 我方推导的粗分类 `animation` | `expression`（由 type + file 后缀），
      供 LLM 侧免解析 VTS 原生枚举。
    """

    model_config = ConfigDict(extra="forbid")

    hotkey_id: str
    name: str
    type: str
    file: str
    kind: str


class BehaviorCatalog(BaseModel):
    """`behavior_list` 输出：程序化词表 + 已发现热键的**同一张清单**（AD-7）。

    - behaviors: 词表静态知识，**未连接时仍完整返回**；
    - hotkeys: None = 未连接（尚未枚举）；[] = 已连接但无热键（或热键开关关）；
    - connected: 当前 sink 连接态。
    """

    model_config = ConfigDict(extra="forbid")

    behaviors: list[BehaviorInfo]
    hotkeys: list[HotkeyInfo] | None = None
    connected: bool = False


# ---------------------------------------------------------------------------
# 状态（behavior_status / vts_connect / vts_disconnect 输出）
# ---------------------------------------------------------------------------


class ActiveBehavior(BaseModel):
    """活跃行为快照单条。phase 为包络相位（attack/sustain/release，str 不 Literal——
    相位集是引擎内部演进面，快照只作可观测性呈现）。"""

    model_config = ConfigDict(extra="forbid")

    name: str
    channels: list[str]
    phase: str
    remaining_ms: int


class BehaviorStatus(BaseModel):
    """行为层健康/状态快照（`behavior_status` / `vts_connect` / `vts_disconnect` 输出）。

    sink 的失败面在后台渲染循环（异常即停、**不自动重连**）——本模型即 C 报告要求的
    显式探测点：healthy=False 时 last_error 带循环最后错误，恢复路径 = 再次
    `vts_connect`（显式重连）。

    - cooldowns: 行为词 → 冷却剩余 ms（仅列冷却中的词）；
    - unavailable_params: 所连部署缺席的可选参数（对应行为已降级，AD-5）；
    - hotkey_count/model_id: 未连接时为 None。
    - speech_active/speech_queue_depth/speech_last_error: 语音播放可观测性
      （speech-play 蓝图 AD-12，向后兼容加法字段）——未播放过/未启用时全取默认值。
    """

    model_config = ConfigDict(extra="forbid")

    connected: bool
    healthy: bool
    last_error: str | None = None
    active: list[ActiveBehavior] = Field(default_factory=list)
    cooldowns: dict[str, int] = Field(default_factory=dict)
    unavailable_params: list[str] = Field(default_factory=list)
    hotkey_count: int | None = None
    model_id: str | None = None
    trajectory_active: bool = False
    trajectory_remaining_ms: int = 0
    speech_active: bool = False
    speech_queue_depth: int = 0
    speech_last_error: str | None = None


# ---------------------------------------------------------------------------
# 裸参数轨迹通道（2026-07-31 二期）——Zero 侧动作模型 → 参数级直驱
# ---------------------------------------------------------------------------

TRAJECTORY_MODES: frozenset[str] = frozenset({"absolute", "offset"})
"""轨迹回放模式：absolute=按值接管（takeover 强度渐变，无跳变）；
offset=在表情/微表情基线上加性叠加。"""

TRAJECTORY_MAX_SEGMENT_MS: int = 10_000
"""单段轨迹时长上限（防单次投喂长期霸占参数；流式续接用 append 分块）。"""

TRAJECTORY_MAX_KEYFRAMES: int = 600
"""单段关键帧数上限（20Hz × 10s + 裕量；防超大 payload）。"""

TRAJECTORY_MAX_QUEUE: int = 5
"""轨迹段队列限深——满时新段回 rejected `[vtsb:throttled]`（背压：按回执退避重发）。"""


class TrajectoryKeyframe(BaseModel):
    """轨迹单关键帧：``t_ms`` 为相对段起点的毫秒时刻，``params`` 为该时刻各参数值。

    同一段内**所有关键帧的参数键集必须一致**（动作模型输出天然是稠密统一帧；
    键集不一致在执行侧回 rejected + `VTSB_INVALID_PARAMS`，不做隐式补值猜测）。
    """

    model_config = ConfigDict(extra="forbid")

    t_ms: int
    params: dict[str, float]

    @model_validator(mode="after")
    def _validate(self) -> TrajectoryKeyframe:
        if self.t_ms < 0:
            raise ValueError(f"t_ms={self.t_ms} 不得为负")
        if not self.params:
            raise ValueError("params 不得为空")
        for name, value in self.params.items():
            if not math.isfinite(value):
                raise ValueError(f"params[{name!r}]={value} 必须为有限值（NaN/inf 拒收）")
        return self


class TrajectoryRequest(BaseModel):
    """轨迹投喂请求（Zero 侧动作模型 → MCP，`params_animate` 工具输入）。

    - keyframes: 按 ``t_ms`` 严格升序；帧间线性插值（动作模型输出稠密帧，
      插值形态不做花活）；首帧建议 t_ms=0。⚠ **单帧段 = 短暂闪现**（时长 0，
      仅经交还缓出持住约 250ms）——要「持续保持某姿态」请用首尾同值的双帧段
      或持续流式投喂，勿依赖单帧。
    - mode: 见 ``TRAJECTORY_MODES``（str 不 Literal——同 BehaviorRequest.name 的
      演进理由；未知模式执行侧回 rejected）。
    - append: True=排到当前队列末尾无缝续接（流式投喂形态）；False=清队即刻
      接管（旧轨迹淡出）。
    """

    model_config = ConfigDict(extra="forbid")

    keyframes: list[TrajectoryKeyframe]
    mode: str = "absolute"
    append: bool = True

    @model_validator(mode="after")
    def _validate(self) -> TrajectoryRequest:
        if not self.keyframes:
            raise ValueError("keyframes 不得为空")
        if len(self.keyframes) > TRAJECTORY_MAX_KEYFRAMES:
            raise ValueError(
                f"keyframes 数 {len(self.keyframes)} 超上限 {TRAJECTORY_MAX_KEYFRAMES}"
            )
        times = [kf.t_ms for kf in self.keyframes]
        if any(b <= a for a, b in zip(times, times[1:], strict=False)):
            raise ValueError("keyframes 的 t_ms 必须严格升序")
        if times[-1] > TRAJECTORY_MAX_SEGMENT_MS:
            raise ValueError(f"单段时长 {times[-1]}ms 超上限 {TRAJECTORY_MAX_SEGMENT_MS}ms")
        return self


class TrajectoryReceipt(BaseModel):
    """轨迹投喂/清除回执。三态语义同 BehaviorReceipt（AD-6/AD-11）：
    业务性拒绝（未知模式/键集不一致/未知参数占比过高/队列满）进 ``code``，
    协议性失败（未连接等）才抛 ToolError。

    - dropped_params: 所连部署不存在、已被静默丢弃的参数名（其余照常回放）；
      **全部参数都不存在**才整段 rejected。
    - queue_depth: 回执后的队列深度（含当前播放段），供投喂方做背压。
    """

    model_config = ConfigDict(extra="forbid")

    status: str
    duration_ms: int = 0
    dropped_params: list[str] = Field(default_factory=list)
    queue_depth: int = 0
    code: str | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> TrajectoryReceipt:
        if self.status not in RECEIPT_STATUSES:
            raise ValueError(f"status {self.status!r} 不在 {sorted(RECEIPT_STATUSES)} 内")
        return self


# ---------------------------------------------------------------------------
# 语音播放 + 口型同步（speech-play 蓝图 2026-08-14 §T1）——Zero 侧语音合成 →
# 我方渲染端同步播放音频与注入口型轨迹
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    """语音播放请求（Zero 侧 → MCP，`speech_play` 工具输入）。

    - wav_path: **同机绝对路径**（不传音频字节：stdio JSON 传兆级 base64 不现实，
      跨仓拍板③无路径白名单约束）。契约层只校验非空字符串——存在性/可读性/
      格式校验留 I/O 层（`speech_playback.read_wav_meta`），理由：那些校验本身
      是阻塞 I/O，不属于纯校验的契约层职责。
    - mouth_track: 口型关键帧轨迹（跨仓拍板①，沿用 `params_animate` 的
      `TrajectoryKeyframe` 形状——Zero 按 `generate_dual` 输出同构生成，我方
      零新解析代码）。`t_ms` 时基相对段起点，`t=0` 对齐音频首采样。
    - fps: 轨迹采样率提示（默认 20，与现行 `params_animate` 投递一致）；
      仅供参考不强制校验帧间隔，实际回放仍按各帧 `t_ms` 插值。

    聚合校验（升序/非空/单段上限）用 **speech 专属上限**直接实现（2026-08-14
    应 Zero 回件放宽，替代蓝图 AD-2 的 scratch `TrajectoryRequest` 复用——那正是
    10s/600 帧继承限的来源，语音整句常态超 10s）：上限走
    `SPEECH_TRACK_MAX_SEGMENT_MS`/`SPEECH_TRACK_MAX_KEYFRAMES`（60s/1200 帧），
    逐帧校验（t_ms≥0/params 非空/有限值）仍由 `TrajectoryKeyframe` 自身承担；
    `params_animate` 的 `TRAJECTORY_MAX_*` 一字不动（零回归）。
    """

    model_config = ConfigDict(extra="forbid")

    wav_path: str
    mouth_track: list[TrajectoryKeyframe]
    fps: float = 20.0

    @model_validator(mode="after")
    def _validate(self) -> SpeechRequest:
        if not self.wav_path:
            raise ValueError("wav_path 不得为空")
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError(f"fps={self.fps} 必须为正的有限值")
        if not self.mouth_track:
            raise ValueError("mouth_track 不合法：keyframes 不得为空")
        if len(self.mouth_track) > SPEECH_TRACK_MAX_KEYFRAMES:
            raise ValueError(
                f"mouth_track 不合法：keyframes 数 {len(self.mouth_track)} "
                f"超上限 {SPEECH_TRACK_MAX_KEYFRAMES}"
            )
        times = [kf.t_ms for kf in self.mouth_track]
        if any(b <= a for a, b in zip(times, times[1:], strict=False)):
            raise ValueError("mouth_track 不合法：keyframes 的 t_ms 必须严格升序")
        if times[-1] > SPEECH_TRACK_MAX_SEGMENT_MS:
            raise ValueError(
                f"mouth_track 不合法：单段时长 {times[-1]}ms 超上限 {SPEECH_TRACK_MAX_SEGMENT_MS}ms"
            )
        return self


class SpeechReceipt(BaseModel):
    """语音播放回执（MCP → Zero 侧，`speech_play` 成功输出）。

    ⚠ **恰好** `accepted` + `duration_ms` 两字段——**故意偏离**本模块
    `BehaviorReceipt`/`TrajectoryReceipt` 的 `status`/`code` 三态惯例
    （speech-play 蓝图 AD-3）：Zero 已按这个字面形状写了 fake-session 单测
    （`{"accepted": true, "duration_ms": 3210.0}`），先于我方实现合入；改成
    我方内部惯例会在对方毫不知情的情况下悄悄破坏其已合入的解析代码。
    审查时**不要**把这处「不一致」当 bug 拉回统一。

    - accepted: 恒为 True（本工具全部失败路径走 ToolError，见模块 docstring
      码表 speech_play 例外条——没有 rejected 业务回执可容纳失败态，AD-4）。
    - duration_ms: 音频时长（wav 帧数 / SPEECH_WAV_SAMPLE_RATE * 1000）；
      **返回时机 = 完成调度即返，不等播完**——Zero 据此节流下一次调用。
    """

    model_config = ConfigDict(extra="forbid")

    accepted: bool
    duration_ms: float


class ParamInfo(BaseModel):
    """单个 VTS 输入参数的量程信息（`params_list` 输出——动作模型的作用空间）。

    - governed: 是否属于表情通路恒定注入的 GOVERNED_PARAMS（轨迹接管这些参数时
      与表情通路按 takeover 强度混合；非 governed 参数按需注入、停喂 1s 后 VTS 回收）。
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    min: float
    max: float
    default_value: float
    governed: bool = False


class ParamCatalog(BaseModel):
    """`params_list` 输出：所连 VTS 部署的全量输入参数表。未连接时 params=None。"""

    model_config = ConfigDict(extra="forbid")

    params: list[ParamInfo] | None = None
    connected: bool = False
