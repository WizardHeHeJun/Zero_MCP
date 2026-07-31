"""VTS 离散行为手势合成引擎——BehaviorOverlayEngine（蓝图 2026-07-31 §2/§3/§4 · T2）。

纯同步、无 I/O、无 await：时间一律由调用方注入 ``now``（单调秒，对齐
``BlinkMachine`` 可测性约定）——引擎内部**绝不** ``time.monotonic()``。包络是
``(触发时刻, 定标参数) -> f(now)`` 的纯函数，单测注入 now 序列即可确定性驱动
（AD-3，无随机相位）。trigger/interrupt/apply/snapshot 均单步完成，与渲染循环
同一事件循环内交错调用天然原子，无需锁（蓝图 §6 并发纪律）。

## 合成范式（AD-3）

每行为一条 attack–sustain–release（ADSR 式）参数包络 + 余弦缓动；节律行为
（nod/shake/blink/body_sway）走正弦 stroke，``repeat`` = stroke 周期数
（nod/blink=半周期一拍，shake/body_sway=整周期一拍）。多行为异通道线性叠加；
同通道抢占走 ``CROSSFADE_S`` 交叉淡化——旧包络淡出、新包络淡入，期间两包络共存。

输出 ``apply(now) -> OverlayFrame``：

- ``offsets``：参数名 → 加性偏移（角度/眉/嘴参数，以及 eyes_widen 对 EyeOpen
  的上推）。**包络结束的键从 offsets 消失**——sink 据此停发对应可选参数，1s 后
  VTS 判 lost 自动交还控制权（AD-5）；release 末端偏移严格收敛到 0，无观感跳变。
- ``eye_gate``：眼睑乘法门（blink 行为专用，1=不干预、0=全闭）——与 ambient
  眨眼乘子同为乘法链（闭×闭=闭，语义安全，AD-4）；淡化作用于「闭合深度」，
  淡出中的 blink 门收敛到 1（中性）而非 0（闭死）。

## 仲裁（AD-6——「事件驱动节流」的执行侧落地）

按序四层，业务性拒绝一律走回执 ``code`` 字段（不是 ToolError，AD-11）：

1. per-behavior 冷却 → ``[vtsb:cooldown]``（detail 带剩余 ms）；
2. 全局节流（两次 accepted 最小间隔 ``GLOBAL_THROTTLE_S``）→ ``[vtsb:throttled]``；
3. 同通道优先级：``new.priority >= active.priority`` → **replaced**（交叉淡化
   抢占）；否则 ``[vtsb:channel_busy]``。异通道 MERGE 并行叠加；**不排队**——
   行为是对「当下」的反应，队列里的过期行为比丢弃更糟 [工程假设]；
4. ``interrupt``：清指定通道/全部包络（``CROSSFADE_S`` 淡出回语义静息基准）。

``hotkey:<id>`` 命名空间不在本引擎（AD-7：热键是协议转发，由 service 层预拦
冷却后直发 VTS）；本引擎只认 ``VOCABULARY`` 的 12 个程序化行为词。

## 幅度定标与降级（AD-5，触发时决定）

幅度全部按触发时注入的 ``ranges``（参数名 → (min, max, defaultValue)）比例
推导，不硬编码度数：角度参数峰值 = ``intensity × 系数 × (max-min)/2``；
[0,1] 参数（眉/嘴/眼）幅度 = ``intensity × 系数`` 直乘。缺 BodyAngle* →
body 三词降级 FaceAngle 微量近似（约 ``DEGRADED_BODY_RATIO`` 幅度）；缺眼球
参数 → glance 借 FaceAngleX/Y 微偏。降级在触发时定死（包络存续期不再改判），
并经回执 ``degraded_channels`` 报告。缺参/降级判定的单一真相是模块级公开函数
``resolve_degradation``（按 direction 过滤后再判缺参）——``_resolve_tracks``
与 ``src/mcp/behavior/service.py`` 的 catalog 装配（`_behavior_info`）共用它，
不各自重复实现（W2/W5，code-review 修订）。

⚠ 本模块全部时长/冷却/优先级/幅度/淡化/节流常量均为
[工程假设，待 Hiyori_A 皮套标定]——标定后回写常量值与蓝图（蓝图 open questions）。
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

from src.agents.models.vts_behavior import (
    VTSB_CHANNEL_BUSY,
    VTSB_COOLDOWN,
    VTSB_INVALID_PARAMS,
    VTSB_THROTTLED,
    VTSB_UNKNOWN_BEHAVIOR,
    ActiveBehavior,
    BehaviorReceipt,
    BehaviorRequest,
)

Ranges = dict[str, tuple[float, float, float]]
"""触发时注入的参数值域表：参数名 → (min, max, defaultValue)（运行时读回，不硬编码）。"""

# ---------------------------------------------------------------------------
# 通道与优先级（蓝图 §2 / AD-6）
# ---------------------------------------------------------------------------

CHANNELS: frozenset[str] = frozenset({"head", "gaze", "eyelid", "brows", "mouth", "body"})
"""仲裁的最小单元——六通道全集（蓝图 §2）。interrupt 的 channel 参数以此校验。"""

PRIORITY_REACTIVE: int = 3
"""reactive 档（eyes_widen/brow_raise）——对刺激的即时反应，可抢占一切 [工程假设]。"""

PRIORITY_DELIBERATE: int = 2
"""deliberate 档（nod/shake/head_tilt/glance/blink/smile/brow_furrow）[工程假设]。"""

PRIORITY_POSTURE: int = 1
"""posture 档（lean_in/lean_back/body_sway）——慢姿态，最易被抢占 [工程假设]。"""

# ---------------------------------------------------------------------------
# 仲裁常量（AD-6）[工程假设，待 Hiyori_A 标定]
# ---------------------------------------------------------------------------

GLOBAL_THROTTLE_S: float = 0.25
"""任意两次 accepted 触发的最小间隔（秒）——防 LLM 侧连珠炮 [工程假设]。"""

CROSSFADE_S: float = 0.15
"""同通道 REPLACE / interrupt 的交叉淡化窗口（秒），SmartBody 范式 [工程假设]。"""

# ---------------------------------------------------------------------------
# 包络形态常量（AD-3）[工程假设，待 Hiyori_A 标定]
# ---------------------------------------------------------------------------

ATTACK_FRACTION: float = 0.18
"""hold 型包络 attack 段占总时长比例（余弦缓入）[2026-07-31 Hiyori_A 标定：0.25→0.18，
用户裁定「可以更主动」——更快的起势]。"""

RELEASE_FRACTION: float = 0.35
"""hold 型包络 release 段占总时长比例（余弦缓出，末端收敛到 0）[工程假设]。"""

# ---------------------------------------------------------------------------
# 幅度系数（AD-5）——角度参数乘 (max-min)/2 定标，[0,1] 参数直乘 intensity。
# [2026-07-31 Hiyori_A 实机标定]：标定证据 = 逐词峰值像素差（窗口 600x950，
# ambient 关，静息噪声底 2.0）+ Live2D 输出参数读回（全词单调、0.5→1.0 精确 ×2，
# 引擎无误）+ 用户裁定「整体太淡、只有歪头(30+/35)清晰可见、要更主动」。
# 标定策略：以 head_tilt 的视觉能量为基准上调其余词；俯仰/偏航在该皮上像素响应
# 弱于侧倾（roll 带动整头+头发），故 nod/shake 上调最多；眉/嘴推近量程天花板
# （brow@1.0 原已达 ±1 量程的 ~0.79）。未标注「标定」的仍为 [工程假设]。
# ---------------------------------------------------------------------------

NOD_SCALE: float = 0.72
"""nod：FaceAngleY 峰值 = intensity × 本系数 × 半量程（低头方向）
[2026-07-31 标定：0.35→0.62→0.72，俯仰像素响应弱（原 @0.5 仅 -6.8°/像素差 5.9）；
二轮用户裁定「晃头类幅度再大点」]。"""

SHAKE_SCALE: float = 0.65
"""shake：FaceAngleX 摆头幅度系数 [2026-07-31 标定：0.30→0.50→0.65（二轮加幅）]。"""

HEAD_TILT_SCALE: float = 0.42
"""head_tilt：FaceAngleZ 侧倾幅度系数 [2026-07-31 标定：0.30→0.35→0.42（二轮加幅）]。"""

GLANCE_EYE_SCALE: float = 0.85
"""glance：EyeLeft/RightX/Y 眼球偏转幅度系数 [2026-07-31 标定：0.60→0.85，
原 @0.5 眼球仅偏 0.30（±1 量程），眼部小面积需更大摆幅才可读]。"""

GLANCE_HEAD_SCALE: float = 0.18
"""glance 降级（缺眼球参数借 head）：FaceAngleX/Y 微偏幅度系数
[2026-07-31 随主系数等比上调：0.12→0.18]。"""

BLINK_DEPTH_SCALE: float = 1.6
"""blink：闭合深度 = intensity × 本系数，>1 为有意过驱动
[2026-07-31 标定：1.0→1.6——原 @0.5 只闭 74% 读作眯眼而非眨眼；过驱动后
@0.5≈80% 闭合、@1.0 饱和为平底全闭（自然的重眨），乘法门下限 0 由 clamp 保证]。"""

BROW_RAISE_SCALE: float = 0.50
"""brow_raise：BrowLeftY/BrowRightY 上扬偏移 = intensity × 本系数
[2026-07-31 标定：0.40→0.50，@1.0 恰用满静息上行半程;该皮眉被刘海遮挡，
数值已到位（读回 ±0.79），视觉含蓄属美术现实]。"""

BROW_FURROW_SCALE: float = 0.45
"""brow_furrow：BrowLeftY/BrowRightY 下压偏移系数 [2026-07-31 标定：0.35→0.45]。"""

BROW_FURROW_CENTER_SCALE: float = 0.30
"""brow_furrow：Brows（综合眉参数）伴随下压系数 [2026-07-31 标定：0.25→0.30]。"""

EYES_WIDEN_EYE_SCALE: float = 0.50
"""eyes_widen：EyeOpenLeft/Right 加性上推系数（走加法非乘法门，AD-4）[工程假设——
⚠ 实测 Hiyori_A 静息 ParamEyeLOpen=1.9 已满开，此分量在该类皮上被 clamp 吃掉
（读回位移恒 0）；保留供静息非满开的皮套，该词可读性由眉+仰头分量承担]。"""

EYES_WIDEN_BROW_SCALE: float = 0.40
"""eyes_widen：BrowLeftY/BrowRightY 伴随微扬系数 [2026-07-31 标定：0.25→0.40，
EyeOpen 分量在满开皮上失效后眉是主要可见分量]。"""

EYES_WIDEN_HEAD_SCALE: float = 0.15
"""eyes_widen：FaceAngleY 仰头后缩系数（startle 的头部构成）[2026-07-31 新增：
EyeOpen 天花板皮套上该词原本近乎不可见，补经典 startle 后缩使其在任意皮上可读]。"""

SMILE_SCALE: float = 0.50
"""smile：MouthSmile 上扬偏移 = intensity × 本系数
[2026-07-31 标定：0.45→0.50，@1.0 恰用满上行半程]。"""

LEAN_BODY_SCALE: float = 0.30
"""lean_in/lean_back：BodyAngleY 前倾/后撤幅度系数 [工程假设——Hiyori_A 无
BodyAngle，真身体轴幅度未经实机验证]。"""

DEGRADED_BODY_RATIO: float = 0.70
"""body 三词降级 head 近似时的幅度衰减比
[2026-07-31 标定：1/3→0.55→0.70（二轮加幅），蓝图 AD-5 的 1/3 在实机偏淡
（lean @0.5 像素差 3.4-6.9），用户两轮裁定加力；该皮无 BodyAngle，降级路径即事实主路径]。"""

LEAN_HEAD_SCALE: float = LEAN_BODY_SCALE * DEGRADED_BODY_RATIO
"""lean_in/lean_back 降级：FaceAngleY 微量近似系数（= body 系数 × 降级比）。"""

LEAN_BACK_TILT_SCALE: float = 0.08
"""lean_back 降级：FaceAngleZ 伴随微倾系数 [2026-07-31 标定：0.05→0.08]。"""

SWAY_BODY_X_SCALE: float = 0.25
"""body_sway：BodyAngleX 主摆幅度系数 [工程假设——同 LEAN_BODY_SCALE，未经实机]。"""

SWAY_BODY_Z_SCALE: float = 0.15
"""body_sway：BodyAngleZ 伴随摆动系数 [工程假设]。"""

SWAY_HEAD_SCALE: float = SWAY_BODY_X_SCALE * DEGRADED_BODY_RATIO
"""body_sway 降级：FaceAngleZ 微量近似系数（= 主摆系数 × 降级比）。"""

# ---------------------------------------------------------------------------
# 去僵硬层（2026-07-31 二轮标定新增）——用户裁定「动作僵硬、不如待机动画俏皮」。
# 两味药：①拍间衰减（第一拍最重、后拍渐轻，打破节拍器等幅感——手工动画惯例）；
# ②多轴伴随轨道（真人头动从不单轴：点头带微滚转、摇头带反相侧倾、歪头带低颌、
# 摇摆走交叉轴 8 字）。均为确定性(无随机)，不破引擎可测性。
# ---------------------------------------------------------------------------

STROKE_BEAT_DECAY: float = 0.82
"""stroke 型逐拍幅度衰减比（第 k 拍幅度 × 本系数^k）[2026-07-31 标定新增]。
⚠ 只作用于 offset 轨道（_shape）；blink 的乘法门轨道**豁免**（gate_at——闭合深度
逐拍变轻会读作「没闭上」，与词义背离，审查 WARN-1）。"""

NOD_ROLL_SCALE: float = NOD_SCALE * 0.16
"""nod 伴随：FaceAngleZ 随拍微滚转系数 [2026-07-31 标定新增]。"""

SHAKE_ROLL_SCALE: float = SHAKE_SCALE * 0.32
"""shake 伴随：FaceAngleZ 反相侧倾系数（钟摆弧线感）[2026-07-31 标定新增]。"""

TILT_DIP_SCALE: float = 0.10
"""head_tilt 伴随：FaceAngleY 低颌系数（好奇歪头带颌部下沉）[2026-07-31 标定新增]。"""

SWAY_CROSS_SCALE: float = SWAY_HEAD_SCALE * 0.45
"""body_sway 降级伴随：FaceAngleX 交叉轴反相微摆（8 字轨迹感）[2026-07-31 标定新增]。"""

# ---------------------------------------------------------------------------
# 波形 / 包络类型 / 方向符号
# ---------------------------------------------------------------------------

WAVE_ADSR: str = "adsr"
"""hold 型轨道波形：attack–sustain–release + 余弦缓动。"""

WAVE_HALF_SINE: str = "half_sine"
"""半周期正弦 stroke（nod：低头-回位为一拍，同号重复）。"""

WAVE_FULL_SINE: str = "full_sine"
"""整周期正弦 stroke（shake/body_sway：一去一回为一拍，端点为 0 天然无突跳）。"""

WAVE_GATE: str = "gate"
"""眼睑乘法门轨道（blink 专用）：不进 offsets，经 OverlayFrame.eye_gate 输出。"""

KIND_HOLD: str = "hold"
"""hold 型行为：ADSR 包络（attack 到峰值 → sustain 保持 → release 回 0）。"""

KIND_STROKE: str = "stroke"
"""stroke 型行为：正弦节律，repeat = 拍数，波形自身保证端点为 0。"""

EYE_GATE_TRACK: str = "eye_gate"
"""gate 轨道的占位参数名（**非 VTS 参数**，不参与 ranges 定标与在场判定）。"""

DIRECTION_SIGNS: dict[str, float] = {"left": -1.0, "right": 1.0, "up": 1.0, "down": -1.0}
"""方向 → 偏移符号：left/right 作用于 X 轴参数，up/down 作用于 Y 轴参数。
符号取向（left=负、up=正）为 [工程假设，待 Hiyori_A 标定]；FaceAngleY 正向=抬头
有仓内先例佐证（vts.head_to_params 的 arousal 抬头装饰项）。"""


# ---------------------------------------------------------------------------
# 包络原语（AD-3，纯函数）
# ---------------------------------------------------------------------------


def cosine_ease01(u: float) -> float:
    """余弦缓动：u∈[0,1] → [0,1]，端点导数为 0（无突跳起停）；越界输入自动夹取。"""
    u = _clamp01(u)
    return 0.5 * (1.0 - math.cos(math.pi * u))


def adsr_envelope(t: float, attack_s: float, sustain_s: float, release_s: float) -> float:
    """ADSR 式包络（无 decay 段）：返回 [0,1] 幅度系数。

    attack/release 用余弦缓动进出，sustain 恒 1；t 越界（<0 或超总长）返回 0——
    **release 末端严格收敛到 0**（AD-3：键从 offsets 消失前无残余偏移）。
    """
    if t < 0.0:
        return 0.0
    if t < attack_s:
        return cosine_ease01(t / attack_s)
    t -= attack_s
    if t < sustain_s:
        return 1.0
    t -= sustain_s
    if t < release_s:
        return 1.0 - cosine_ease01(t / release_s)
    return 0.0


def half_sine_strokes(u: float, repeat: int, decay: float = 1.0) -> float:
    """半周期正弦 stroke：归一化进度 u∈[0,1) 内做 repeat 拍（每拍 0→1→0，同号）。

    nod（低头-回位）与 blink 闭合深度共用；u 越界返回 0（端点无残余）。
    decay：逐拍幅度衰减比（第 k 拍 × decay^k，缺省 1.0 = 等幅；确定性，
    2026-07-31 去僵硬标定引入，见 STROKE_BEAT_DECAY）。
    """
    if not 0.0 <= u < 1.0:
        return 0.0
    v = u * repeat
    beat = math.floor(v)
    return (decay**beat) * math.sin(math.pi * (v - beat))


def full_sine_strokes(u: float, repeat: int, decay: float = 1.0) -> float:
    """整周期正弦 stroke：u∈[0,1) 内做 repeat 拍（每拍一去一回），端点为 0。

    decay 语义同 ``half_sine_strokes``（逐拍衰减，每拍相位归零后乘 decay^k，
    与无衰减时的连续正弦逐点一致——sin(2π(v−k)) ≡ sin(2πv)）。
    """
    if not 0.0 <= u < 1.0:
        return 0.0
    v = u * repeat
    beat = math.floor(v)
    return (decay**beat) * math.sin(2.0 * math.pi * (v - beat))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ---------------------------------------------------------------------------
# 词表规格（蓝图 §2 表的代码形态）
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class TrackPlan:
    """词表内单参数轨道计划（未定标）——触发时按 ranges × intensity 折算成 Track。

    - scale: 幅度系数；angle_scaled=True 时再乘 ``(max-min)/2``（角度参数），
      False 时直乘 intensity（[0,1] 参数与 gate 深度）。
    - sign: 静态符号（如 nod 低头 = FaceAngleY 负向）。
    - signed_by_direction: 再乘 ``DIRECTION_SIGNS[direction]``。
    - direction_filter: 非 None 时仅当解析后的 direction 命中才启用本轨道
      （glance 按方向选 X/Y 轴参数）。
    """

    param: str
    waveform: str
    scale: float
    angle_scaled: bool = True
    sign: float = 1.0
    signed_by_direction: bool = False
    direction_filter: tuple[str, ...] | None = None


@dataclass(frozen=True, kw_only=True)
class BehaviorSpec:
    """单个行为词的完整规格（蓝图 §2 表的一行）。

    definition/params_schema/typical_duration_ms/cooldown_s/channels 供
    ``behavior_list`` catalog 直出（definition 即交付 Zero 侧 LLM 的 prompt 素材）；
    其余字段为引擎合成/仲裁所需。repeat 仅对 stroke 型行为有意义（hold 型忽略，
    params_schema 不列即不建议传）。
    """

    name: str
    definition: str
    channels: tuple[str, ...]
    priority: int
    kind: str
    base_duration_ms: int
    repeat_scales_duration: bool
    cooldown_s: float
    params_schema: dict[str, str]
    primary: tuple[TrackPlan, ...]
    default_repeat: int = 1
    directions: tuple[str, ...] | None = None
    default_direction: str | None = None
    fallback: tuple[TrackPlan, ...] = ()
    fallback_channels: tuple[str, ...] = ()
    degraded_channels: tuple[str, ...] = ()
    degraded_note: str | None = None

    @property
    def typical_duration_ms(self) -> int:
        """catalog 展示的典型时长：stroke 型 = 单拍时长 × 建议拍数，hold 型 = 总时长。"""
        if self.repeat_scales_duration:
            return self.base_duration_ms * self.default_repeat
        return self.base_duration_ms


VOCABULARY: dict[str, BehaviorSpec] = {
    "nod": BehaviorSpec(
        name="nod",
        definition="肯定/附和：低头-回位的点头；repeat 控制点头次数。",
        channels=("head",),
        priority=PRIORITY_DELIBERATE,
        kind=KIND_STROKE,
        base_duration_ms=700,
        repeat_scales_duration=True,
        cooldown_s=1.5,
        params_schema={"intensity": "点头幅度 0-1", "repeat": "点头次数 1-8"},
        primary=(
            TrackPlan(param="FaceAngleY", waveform=WAVE_HALF_SINE, scale=NOD_SCALE, sign=-1.0),
            # 伴随微滚转（2026-07-31 去僵硬标定）：随拍同步的小幅 Z 侧倾，破单轴机械感。
            TrackPlan(param="FaceAngleZ", waveform=WAVE_HALF_SINE, scale=NOD_ROLL_SCALE),
        ),
    ),
    "shake": BehaviorSpec(
        name="shake",
        definition="否定/拒绝：左右摆头；repeat 为往复周期数（建议 2）。",
        channels=("head",),
        priority=PRIORITY_DELIBERATE,
        kind=KIND_STROKE,
        base_duration_ms=600,
        repeat_scales_duration=True,
        default_repeat=2,
        cooldown_s=1.5,
        params_schema={"intensity": "摆头幅度 0-1", "repeat": "往复周期数 1-8（建议 2）"},
        primary=(
            TrackPlan(param="FaceAngleX", waveform=WAVE_FULL_SINE, scale=SHAKE_SCALE),
            # 反相侧倾（2026-07-31 去僵硬标定）：头向左摆时向右微倾，钟摆弧线感。
            TrackPlan(
                param="FaceAngleZ", waveform=WAVE_FULL_SINE, scale=SHAKE_ROLL_SCALE, sign=-1.0
            ),
        ),
    ),
    "head_tilt": BehaviorSpec(
        name="head_tilt",
        definition="疑惑/俏皮：头向一侧倾斜保持片刻后回正；direction 选 left/right。",
        channels=("head",),
        priority=PRIORITY_DELIBERATE,
        kind=KIND_HOLD,
        base_duration_ms=2000,
        repeat_scales_duration=False,
        cooldown_s=2.0,
        directions=("left", "right"),
        default_direction="left",
        params_schema={"intensity": "侧倾幅度 0-1", "direction": "left|right（缺省 left）"},
        primary=(
            TrackPlan(
                param="FaceAngleZ",
                waveform=WAVE_ADSR,
                scale=HEAD_TILT_SCALE,
                signed_by_direction=True,
            ),
            # 低颌伴随（2026-07-31 去僵硬标定）：好奇歪头带颌部微沉，方向无关。
            TrackPlan(param="FaceAngleY", waveform=WAVE_ADSR, scale=TILT_DIP_SCALE, sign=-1.0),
        ),
    ),
    "glance": BehaviorSpec(
        name="glance",
        definition="回避/示意：目光瞥向一侧再收回；direction 选 left/right/up/down。",
        channels=("gaze",),
        priority=PRIORITY_DELIBERATE,
        kind=KIND_HOLD,
        base_duration_ms=1200,
        repeat_scales_duration=False,
        cooldown_s=1.0,
        directions=("left", "right", "up", "down"),
        default_direction="left",
        params_schema={
            "intensity": "瞥视幅度 0-1",
            "direction": "left|right|up|down（缺省 left）",
        },
        primary=(
            TrackPlan(
                param="EyeLeftX",
                waveform=WAVE_ADSR,
                scale=GLANCE_EYE_SCALE,
                signed_by_direction=True,
                direction_filter=("left", "right"),
            ),
            TrackPlan(
                param="EyeRightX",
                waveform=WAVE_ADSR,
                scale=GLANCE_EYE_SCALE,
                signed_by_direction=True,
                direction_filter=("left", "right"),
            ),
            TrackPlan(
                param="EyeLeftY",
                waveform=WAVE_ADSR,
                scale=GLANCE_EYE_SCALE,
                signed_by_direction=True,
                direction_filter=("up", "down"),
            ),
            TrackPlan(
                param="EyeRightY",
                waveform=WAVE_ADSR,
                scale=GLANCE_EYE_SCALE,
                signed_by_direction=True,
                direction_filter=("up", "down"),
            ),
        ),
        fallback=(
            TrackPlan(
                param="FaceAngleX",
                waveform=WAVE_ADSR,
                scale=GLANCE_HEAD_SCALE,
                signed_by_direction=True,
                direction_filter=("left", "right"),
            ),
            TrackPlan(
                param="FaceAngleY",
                waveform=WAVE_ADSR,
                scale=GLANCE_HEAD_SCALE,
                signed_by_direction=True,
                direction_filter=("up", "down"),
            ),
        ),
        fallback_channels=("gaze", "head"),
        degraded_channels=("gaze",),
        degraded_note="所连部署缺眼球参数（EyeLeft/RightX/Y），借 FaceAngleX/Y 微偏近似。",
    ),
    "blink": BehaviorSpec(
        name="blink",
        definition="强调/俏皮：单次或多次刻意眨眼；intensity 为闭合深度（1=全闭），"
        "repeat 为眨眼次数。",
        channels=("eyelid",),
        priority=PRIORITY_DELIBERATE,
        kind=KIND_STROKE,
        base_duration_ms=220,
        repeat_scales_duration=True,
        cooldown_s=1.0,
        params_schema={"intensity": "闭合深度 0-1（1=全闭）", "repeat": "眨眼次数 1-8"},
        primary=(
            TrackPlan(
                param=EYE_GATE_TRACK,
                waveform=WAVE_GATE,
                scale=BLINK_DEPTH_SCALE,
                angle_scaled=False,
            ),
        ),
    ),
    "brow_raise": BehaviorSpec(
        name="brow_raise",
        definition="惊讶/兴趣：双眉上扬后回落。",
        channels=("brows",),
        priority=PRIORITY_REACTIVE,
        kind=KIND_HOLD,
        base_duration_ms=900,
        repeat_scales_duration=False,
        cooldown_s=1.0,
        params_schema={"intensity": "扬眉幅度 0-1"},
        primary=(
            TrackPlan(
                param="BrowLeftY", waveform=WAVE_ADSR, scale=BROW_RAISE_SCALE, angle_scaled=False
            ),
            TrackPlan(
                param="BrowRightY", waveform=WAVE_ADSR, scale=BROW_RAISE_SCALE, angle_scaled=False
            ),
        ),
    ),
    "brow_furrow": BehaviorSpec(
        name="brow_furrow",
        definition="不悦/困惑：双眉下压（皱眉）后回落。",
        channels=("brows",),
        priority=PRIORITY_DELIBERATE,
        kind=KIND_HOLD,
        base_duration_ms=1200,
        repeat_scales_duration=False,
        cooldown_s=1.0,
        params_schema={"intensity": "压眉幅度 0-1"},
        primary=(
            TrackPlan(
                param="BrowLeftY",
                waveform=WAVE_ADSR,
                scale=BROW_FURROW_SCALE,
                angle_scaled=False,
                sign=-1.0,
            ),
            TrackPlan(
                param="BrowRightY",
                waveform=WAVE_ADSR,
                scale=BROW_FURROW_SCALE,
                angle_scaled=False,
                sign=-1.0,
            ),
            TrackPlan(
                param="Brows",
                waveform=WAVE_ADSR,
                scale=BROW_FURROW_CENTER_SCALE,
                angle_scaled=False,
                sign=-1.0,
            ),
        ),
    ),
    "eyes_widen": BehaviorSpec(
        name="eyes_widen",
        definition="震惊：睁大双眼、扬眉并微微仰头后缩。",
        # ⚠ 仲裁连带（2026-07-31 标定加仰头轨道扩入 head 通道的显式后果）：本词为
        # reactive 档（最高优先级），现在会抢占任何在播的头部行为（nod/shake/
        # head_tilt 及 body 三词降级态）——语义上成立（惊吓打断有意动作），
        # 已由 TestArbitration 显式锁定。
        channels=("eyelid", "brows", "head"),
        priority=PRIORITY_REACTIVE,
        kind=KIND_HOLD,
        base_duration_ms=800,
        repeat_scales_duration=False,
        cooldown_s=1.5,
        params_schema={"intensity": "震惊幅度 0-1"},
        primary=(
            TrackPlan(
                param="EyeOpenLeft",
                waveform=WAVE_ADSR,
                scale=EYES_WIDEN_EYE_SCALE,
                angle_scaled=False,
            ),
            TrackPlan(
                param="EyeOpenRight",
                waveform=WAVE_ADSR,
                scale=EYES_WIDEN_EYE_SCALE,
                angle_scaled=False,
            ),
            TrackPlan(
                param="BrowLeftY",
                waveform=WAVE_ADSR,
                scale=EYES_WIDEN_BROW_SCALE,
                angle_scaled=False,
            ),
            TrackPlan(
                param="BrowRightY",
                waveform=WAVE_ADSR,
                scale=EYES_WIDEN_BROW_SCALE,
                angle_scaled=False,
            ),
            # 仰头后缩（startle 头部构成，2026-07-31 标定新增）：EyeOpen 分量在
            # 静息满开的皮套上被 clamp 吃掉（Hiyori_A 读回位移恒 0），无此分量
            # 该词近乎不可见；方向与 nod 相反（正号 = 抬头）。
            TrackPlan(
                param="FaceAngleY",
                waveform=WAVE_ADSR,
                scale=EYES_WIDEN_HEAD_SCALE,
                sign=1.0,
            ),
        ),
    ),
    "smile": BehaviorSpec(
        name="smile",
        definition="无声好感：嘴角上扬保持片刻后回落。",
        channels=("mouth",),
        priority=PRIORITY_DELIBERATE,
        kind=KIND_HOLD,
        base_duration_ms=2000,
        repeat_scales_duration=False,
        cooldown_s=2.0,
        params_schema={"intensity": "上扬幅度 0-1"},
        primary=(
            TrackPlan(
                param="MouthSmile", waveform=WAVE_ADSR, scale=SMILE_SCALE, angle_scaled=False
            ),
        ),
    ),
    "lean_in": BehaviorSpec(
        name="lean_in",
        definition="兴趣/亲密：身体前倾靠近后回正。",
        channels=("body",),
        priority=PRIORITY_POSTURE,
        kind=KIND_HOLD,
        base_duration_ms=2500,
        repeat_scales_duration=False,
        cooldown_s=4.0,
        params_schema={"intensity": "前倾幅度 0-1"},
        primary=(
            TrackPlan(param="BodyAngleY", waveform=WAVE_ADSR, scale=LEAN_BODY_SCALE, sign=-1.0),
        ),
        fallback=(
            TrackPlan(param="FaceAngleY", waveform=WAVE_ADSR, scale=LEAN_HEAD_SCALE, sign=-1.0),
        ),
        fallback_channels=("body", "head"),
        degraded_channels=("body",),
        degraded_note="所连部署缺 BodyAngleY，借 FaceAngleY 微量低头近似。",
    ),
    "lean_back": BehaviorSpec(
        name="lean_back",
        definition="惊讶/嫌弃/放松：身体后撤再回正。",
        channels=("body",),
        priority=PRIORITY_POSTURE,
        kind=KIND_HOLD,
        base_duration_ms=2500,
        repeat_scales_duration=False,
        cooldown_s=4.0,
        params_schema={"intensity": "后撤幅度 0-1"},
        primary=(TrackPlan(param="BodyAngleY", waveform=WAVE_ADSR, scale=LEAN_BODY_SCALE),),
        fallback=(
            TrackPlan(param="FaceAngleY", waveform=WAVE_ADSR, scale=LEAN_HEAD_SCALE),
            TrackPlan(param="FaceAngleZ", waveform=WAVE_ADSR, scale=LEAN_BACK_TILT_SCALE),
        ),
        fallback_channels=("body", "head"),
        degraded_channels=("body",),
        degraded_note="所连部署缺 BodyAngleY，借 FaceAngleY 微量抬头 + FaceAngleZ 微倾近似。",
    ),
    "body_sway": BehaviorSpec(
        name="body_sway",
        definition="愉悦/哼歌感：身体节拍性左右轻摆；repeat 为拍数。",
        channels=("body",),
        priority=PRIORITY_POSTURE,
        kind=KIND_STROKE,
        base_duration_ms=1000,
        repeat_scales_duration=True,
        cooldown_s=3.0,
        params_schema={"intensity": "摆动幅度 0-1", "repeat": "拍数 1-8"},
        primary=(
            TrackPlan(param="BodyAngleX", waveform=WAVE_FULL_SINE, scale=SWAY_BODY_X_SCALE),
            TrackPlan(param="BodyAngleZ", waveform=WAVE_FULL_SINE, scale=SWAY_BODY_Z_SCALE),
        ),
        fallback=(
            TrackPlan(param="FaceAngleZ", waveform=WAVE_FULL_SINE, scale=SWAY_HEAD_SCALE),
            # 交叉轴反相微摆（2026-07-31 去僵硬标定）：Z 侧倾 + X 反相微偏 → 8 字轨迹感。
            TrackPlan(
                param="FaceAngleX", waveform=WAVE_FULL_SINE, scale=SWAY_CROSS_SCALE, sign=-1.0
            ),
        ),
        fallback_channels=("body", "head"),
        degraded_channels=("body",),
        degraded_note="所连部署缺 BodyAngleX/Z，借 FaceAngleZ 微摆 + FaceAngleX 交叉微偏近似。",
    ),
}
"""行为词表 v1（AD-2）：12 词的**唯一真相**——catalog、引擎、测试均以此为准。
幅度系数/包络形态已经 2026-07-31 Hiyori_A 实机标定（两轮：可见度 + 去僵硬，证据见
各常量 docstring）；典型时长/冷却/优先级仍为 [工程假设]（蓝图 §2 约定）。"""


# ---------------------------------------------------------------------------
# 运行期数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Track:
    """触发时定标完成的单参数轨道（幅度已按 ranges × intensity × 方向折算，带符号）。"""

    param: str
    waveform: str
    amplitude: float


@dataclass(frozen=True, kw_only=True)
class OverlayFrame:
    """一帧手势叠加输出（AD-3/AD-4）。

    - offsets: 参数名 → 加性偏移，由 sink 合入渲染帧后统一 clamp；**键消失 =
      该参数无活跃包络**，sink 据此停发可选参数（VTS 1s 后自动回收）。
    - eye_gate: 眼睑乘法门 [0,1]，乘到 EyeOpenLeft/Right（1=不干预）。
    """

    offsets: dict[str, float]
    eye_gate: float = 1.0


@dataclass(frozen=True, kw_only=True)
class EngineSnapshot:
    """``snapshot(now)`` 输出——供 ``BehaviorStatus.active/cooldowns`` 直接装配。"""

    active: list[ActiveBehavior]
    cooldowns: dict[str, int]


@dataclass(kw_only=True)
class ActiveEnvelope:
    """一次已接受触发的活跃包络：``(触发时刻, 定标参数) -> f(now)`` 纯函数（AD-3）。

    fade_in / fade_out_at 承载同通道 REPLACE 与 interrupt 的交叉淡化（AD-6）：
    被替换/打断的旧包络置 ``fade_out_at`` 后仍参与 apply 求和，至淡出窗口结束
    才被剪除——期间新旧两包络共存，各自独立淡入/淡出。
    """

    name: str
    behavior_id: str
    channels: tuple[str, ...]
    degraded_channels: tuple[str, ...]
    priority: int
    kind: str
    started_at: float
    duration_s: float
    repeat: int
    attack_s: float
    sustain_s: float
    release_s: float
    tracks: tuple[Track, ...]
    fade_in: bool = False
    fade_out_at: float | None = None

    def expired(self, now: float) -> bool:
        """包络是否已彻底结束（自然到期或淡出完成，取先到者）——结束即可剪除。"""
        end = self.started_at + self.duration_s
        if self.fade_out_at is not None:
            end = min(end, self.fade_out_at + CROSSFADE_S)
        return now >= end

    def offsets_at(self, now: float) -> dict[str, float]:
        """当前时刻各参数加性偏移（gate 轨道不在其中）。

        包络活跃期键恒在场（stroke 过零点也不消失，防 sink 停发/复发抖动）；
        结束后由引擎剪除整条包络，键随之消失（AD-5 交还语义）。
        """
        fade = self._fade(now)
        t = now - self.started_at
        u = t / self.duration_s
        out: dict[str, float] = {}
        for track in self.tracks:
            if track.waveform == WAVE_GATE:
                continue
            out[track.param] = track.amplitude * self._shape(track.waveform, t, u) * fade
        return out

    def gate_at(self, now: float) -> float:
        """眼睑乘法门值（1=不干预）。淡化作用于「闭合深度」而非门值本身——
        淡出中的 blink 门收敛到 1（中性）而不是 0（闭死）。"""
        gate = 1.0
        fade = self._fade(now)
        u = (now - self.started_at) / self.duration_s
        for track in self.tracks:
            if track.waveform != WAVE_GATE:
                continue
            # gate 轨道**豁免拍间衰减**（审查 WARN-1）：offset 轨道拍间变轻仍读得出
            # 是动作，闭合深度变轻则直接读作「没闭上」——实测 intensity=0.5 repeat=8
            # 末拍仅闭 19.9%，与 blink 词义背离。眨眼逐拍等深。
            depth = track.amplitude * half_sine_strokes(u, self.repeat) * fade
            gate *= _clamp01(1.0 - depth)
        return gate

    def phase(self, now: float) -> str:
        """包络相位（快照可观测性用）：hold 型 attack/sustain/release、stroke 型
        stroke、交叉淡出中 fading。相位集是引擎内部演进面（契约侧 str 不 Literal）。"""
        if self.fade_out_at is not None:
            return "fading"
        if self.kind == KIND_STROKE:
            return "stroke"
        t = now - self.started_at
        if t < self.attack_s:
            return "attack"
        if t < self.attack_s + self.sustain_s:
            return "sustain"
        return "release"

    def remaining_ms(self, now: float) -> int:
        """距包络彻底结束的剩余毫秒（淡出中按淡出窗口计）。"""
        end = self.started_at + self.duration_s
        if self.fade_out_at is not None:
            end = min(end, self.fade_out_at + CROSSFADE_S)
        return max(0, int(round((end - now) * 1000.0)))

    def _fade(self, now: float) -> float:
        factor = 1.0
        if self.fade_in:
            factor *= cosine_ease01((now - self.started_at) / CROSSFADE_S)
        if self.fade_out_at is not None:
            factor *= 1.0 - cosine_ease01((now - self.fade_out_at) / CROSSFADE_S)
        return factor

    def _shape(self, waveform: str, t: float, u: float) -> float:
        if waveform == WAVE_HALF_SINE:
            return half_sine_strokes(u, self.repeat, STROKE_BEAT_DECAY)
        if waveform == WAVE_FULL_SINE:
            return full_sine_strokes(u, self.repeat, STROKE_BEAT_DECAY)
        return adsr_envelope(t, self.attack_s, self.sustain_s, self.release_s)


class UnresolvableParamsError(Exception):
    """触发时定标失败：主/降级轨道所需参数均缺席（引擎内部信号，trigger 捕获
    后转 rejected 回执——正常部署不应出现：降级目标全为 GOVERNED_PARAMS，
    sink 连接期保证在场）。"""


# ---------------------------------------------------------------------------
# 触发时定标辅助（AD-5，单一真相——W2/W5：修复 catalog 与引擎口径不一致）
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Degradation:
    """按 direction 过滤后的轨道缺参/降级判定结果（``resolve_degradation`` 输出）。

    - plans: 实际生效的轨道计划（主计划或降级计划，已按 direction 过滤）；
    - channels: 对应通道集（主 ``spec.channels`` 或 ``spec.fallback_channels``）；
    - degraded_channels: 非空表示走了降级路径；
    - resolvable: False 表示主/降级计划所需参数均缺席（无法执行）；
    - missing: ``resolvable=False`` 时缺席的参数名（供拒绝回执/催告展示）。
    """

    plans: tuple[TrackPlan, ...]
    channels: tuple[str, ...]
    degraded_channels: tuple[str, ...]
    resolvable: bool
    missing: tuple[str, ...] = ()


def resolve_degradation(spec: BehaviorSpec, ranges: Ranges, direction: str | None) -> Degradation:
    """按 direction 过滤后的轨道缺参/降级判定（AD-5）——**单一真相函数**。

    ``BehaviorOverlayEngine._resolve_tracks``（触发时定标）与
    ``BehaviorService._behavior_info``（catalog 呈现）共用本函数，不再各自
    重复实现缺参判定（W2/W5，code-review 修订）——修复此前口径不一致：
    catalog 曾对 ``spec.primary`` 全部轨道判缺参、未按 direction 过滤，而引擎
    是先按 direction 过滤再判定，导致「仅提供部分方向所需参数」时两者矛盾
    （复现场景：ranges 只有 ``EyeLeftX``/``EyeRightX``，触发
    ``glance(direction="left")``——catalog 因 primary 里还含缺席的
    ``EyeLeftY``/``EyeRightY`` 而报 degraded，但引擎按 direction 过滤后
    只需要 X 轴两个已在场的参数，实际 accepted 无降级）。

    主计划任一所需参数缺席（gate 轨道除外）即整体切降级计划——部分在场也
    降级，保证行为观感完整而非缺一半轨道 [工程假设]。``direction=None``
    对无方向词等价于「不过滤」（各轨道 ``direction_filter`` 均为 None 时
    天然全部生效）。
    """

    def active_plans(plans: tuple[TrackPlan, ...]) -> tuple[TrackPlan, ...]:
        return tuple(
            p for p in plans if p.direction_filter is None or direction in p.direction_filter
        )

    def missing_params(plans: tuple[TrackPlan, ...]) -> list[str]:
        return [p.param for p in plans if p.waveform != WAVE_GATE and p.param not in ranges]

    plans = active_plans(spec.primary)
    channels = spec.channels
    degraded: tuple[str, ...] = ()
    absent = missing_params(plans)
    if absent and spec.fallback:
        plans = active_plans(spec.fallback)
        channels = spec.fallback_channels
        degraded = spec.degraded_channels
        absent = missing_params(plans)
    return Degradation(
        plans=plans,
        channels=channels,
        degraded_channels=degraded,
        resolvable=not absent,
        missing=tuple(absent),
    )


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------


class BehaviorOverlayEngine:
    """手势合成引擎：词表触发 → 活跃包络集 → 每帧叠加输出（纯同步、无 I/O）。

    状态三件：活跃包络列表（含交叉淡化中的旧包络）、per-behavior 冷却表、
    全局节流时戳。所有方法单步完成、时间由调用方注入——与渲染循环同一事件循环
    内交错调用天然原子，无需锁（蓝图 §6 并发纪律）。

    用法（时间由调用方注入，引擎不自取时钟）::

        engine = BehaviorOverlayEngine()
        receipt = engine.trigger(BehaviorRequest(name="nod"), now=t, ranges=sink.ranges)
        frame = engine.apply(now=t + 0.05)       # 渲染循环每帧调用
        snap = engine.snapshot(now=t + 0.05)     # behavior_status 装配用
    """

    def __init__(self) -> None:
        self.envelopes: list[ActiveEnvelope] = []
        self.cooldown_until: dict[str, float] = {}
        self.last_accepted_at: float | None = None

    # ── 触发 / 打断 ──────────────────────────────────────────────────────────

    def trigger(self, request: BehaviorRequest, now: float, ranges: Ranges) -> BehaviorReceipt:
        """触发一个行为词，按 AD-6 四层仲裁后返回三态回执（全部单步完成）。

        Args:
            request: 契约层已过量纲校验的触发请求；direction 合法性在本层查
                （合法值集随行为词而异，契约层有意不管）。
            now:     调用方注入的单调时钟（秒）。
            ranges:  触发时刻所连部署的参数值域表——幅度定标与降级判定（AD-5）
                均以其键集/量程为准，触发时定死。

        Returns:
            accepted / replaced（detail 带被抢占行为名）/ rejected（code 带
            机读令牌，业务性拒绝不抛异常——AD-11）。
        """
        self._prune(now)
        spec = VOCABULARY.get(request.name)
        if spec is None:
            return self._rejected(
                VTSB_UNKNOWN_BEHAVIOR,
                f"未知行为词 {request.name!r}（词表见 behavior_list）",
            )

        # direction 校验（AD-11 invalid_params：执行侧校验）
        direction: str | None = None
        if spec.directions is None:
            if request.direction is not None:
                return self._rejected(
                    VTSB_INVALID_PARAMS,
                    f"{spec.name} 不接受 direction 参数（收到 {request.direction!r}）",
                    channels=spec.channels,
                )
        else:
            direction = (
                request.direction if request.direction is not None else spec.default_direction
            )
            if direction not in spec.directions:
                return self._rejected(
                    VTSB_INVALID_PARAMS,
                    f"direction={request.direction!r} 不合法，"
                    f"{spec.name} 可选：{list(spec.directions)}",
                    channels=spec.channels,
                )

        # 仲裁第 1 层：per-behavior 冷却
        until = self.cooldown_until.get(spec.name)
        if until is not None and now < until:
            remaining = max(1, int(round((until - now) * 1000.0)))
            return self._rejected(
                VTSB_COOLDOWN,
                f"{spec.name} 冷却未过，剩余 {remaining}ms",
                channels=spec.channels,
            )

        # 仲裁第 2 层：全局节流（防 LLM 侧连珠炮）
        if self.last_accepted_at is not None and now - self.last_accepted_at < GLOBAL_THROTTLE_S:
            return self._rejected(
                VTSB_THROTTLED,
                f"全局节流：距上次接受的触发不足 {int(GLOBAL_THROTTLE_S * 1000)}ms",
                channels=spec.channels,
            )

        # 触发时定标 + 降级判定（AD-5）
        try:
            tracks, channels, degraded = self._resolve_tracks(spec, request, direction, ranges)
        except UnresolvableParamsError as exc:
            return self._rejected(VTSB_INVALID_PARAMS, str(exc), channels=spec.channels)

        # 仲裁第 3 层：同通道优先级（REPLACE-with-crossfade / channel_busy；
        # 淡出中的包络通道已视为释放，不参与仲裁也不被重复淡出）
        overlapping = [
            env
            for env in self.envelopes
            if env.fade_out_at is None and set(env.channels) & set(channels)
        ]
        blocking = [env for env in overlapping if env.priority > spec.priority]
        if blocking:
            names = "、".join(sorted({env.name for env in blocking}))
            return self._rejected(
                VTSB_CHANNEL_BUSY,
                f"通道被更高优先级行为占用：{names}",
                channels=channels,
            )
        for env in overlapping:
            env.fade_out_at = now  # 旧包络 CROSSFADE_S 淡出，与新包络共存（AD-6）

        if request.duration_ms is not None:
            duration_ms = request.duration_ms
        elif spec.repeat_scales_duration:
            duration_ms = spec.base_duration_ms * request.repeat
        else:
            duration_ms = spec.base_duration_ms
        duration_s = duration_ms / 1000.0
        if spec.kind == KIND_HOLD:
            attack_s = duration_s * ATTACK_FRACTION
            release_s = duration_s * RELEASE_FRACTION
            sustain_s = duration_s - attack_s - release_s
        else:
            attack_s = sustain_s = release_s = 0.0

        envelope = ActiveEnvelope(
            name=spec.name,
            behavior_id=uuid.uuid4().hex[:16],
            channels=channels,
            degraded_channels=degraded,
            priority=spec.priority,
            kind=spec.kind,
            started_at=now,
            duration_s=duration_s,
            repeat=request.repeat,
            attack_s=attack_s,
            sustain_s=sustain_s,
            release_s=release_s,
            tracks=tracks,
            fade_in=bool(overlapping),
        )
        self.envelopes.append(envelope)
        self.cooldown_until[spec.name] = now + spec.cooldown_s
        self.last_accepted_at = now

        if overlapping:
            replaced_names = "、".join(sorted({env.name for env in overlapping}))
            detail: str | None = f"已抢占同通道行为：{replaced_names}"
        else:
            detail = None
        return BehaviorReceipt(
            status="replaced" if overlapping else "accepted",
            behavior_id=envelope.behavior_id,
            channels=list(channels),
            estimated_duration_ms=duration_ms,
            degraded_channels=list(degraded),
            detail=detail,
        )

    def interrupt(self, channel: str | None, now: float) -> BehaviorReceipt:
        """打断活跃行为：channel=None 清全部，否则只清含该通道的包络
        （``CROSSFADE_S`` 淡出回语义静息基准，AD-6 第 4 层）。

        幂等：无匹配活跃包络时也返回 accepted（channels 空）。不触碰冷却表——
        冷却是触发频率纪律，与打断无关；也不触碰表情 target（手势层与表情通路
        的唯一耦合在 sink 的 clamp/乘法门链，蓝图 §4）。
        """
        if channel is not None and channel not in CHANNELS:
            return self._rejected(
                VTSB_INVALID_PARAMS,
                f"未知通道 {channel!r}，合法：{sorted(CHANNELS)}",
            )
        self._prune(now)
        cleared: list[ActiveEnvelope] = []
        for env in self.envelopes:
            if env.fade_out_at is not None:
                continue  # 已在淡出，不重复置戳（否则会拉长淡出）
            if channel is None or channel in env.channels:
                env.fade_out_at = now
                cleared.append(env)
        names = "、".join(env.name for env in cleared)
        return BehaviorReceipt(
            status="accepted",
            behavior_id=uuid.uuid4().hex[:16],
            channels=sorted({c for env in cleared for c in env.channels}),
            estimated_duration_ms=int(CROSSFADE_S * 1000.0) if cleared else 0,
            detail=f"已打断：{names}" if cleared else "无匹配的活跃行为（幂等）",
        )

    # ── 每帧合成 / 快照 ──────────────────────────────────────────────────────

    def apply(self, now: float) -> OverlayFrame:
        """合成当前帧叠加输出；顺带剪除已结束/淡出完成的包络（键随之消失）。

        offsets 为各活跃包络（含淡化共存期的新旧两条）逐参数线性求和（AD-3
        异通道 MERGE / 同通道交叉淡化的物理落点）；越界截断留给 sink 的逐参
        clamp（AD-4）。eye_gate 为各包络门值连乘（闭×闭=闭）。
        """
        self._prune(now)
        offsets: dict[str, float] = {}
        eye_gate = 1.0
        for env in self.envelopes:
            for param, value in env.offsets_at(now).items():
                offsets[param] = offsets.get(param, 0.0) + value
            eye_gate *= env.gate_at(now)
        return OverlayFrame(offsets=offsets, eye_gate=eye_gate)

    def snapshot(self, now: float) -> EngineSnapshot:
        """活跃行为与冷却快照（供 ``BehaviorStatus`` 装配）。

        active 含交叉淡出中的包络（phase="fading"，如实呈现）；cooldowns 只列
        冷却中的词（剩余 ms），过期项顺带剪除。
        """
        self._prune(now)
        active = [
            ActiveBehavior(
                name=env.name,
                channels=list(env.channels),
                phase=env.phase(now),
                remaining_ms=env.remaining_ms(now),
            )
            for env in self.envelopes
        ]
        cooldowns: dict[str, int] = {}
        for name, until in list(self.cooldown_until.items()):
            remaining = int(round((until - now) * 1000.0))
            if remaining > 0:
                cooldowns[name] = remaining
            else:
                del self.cooldown_until[name]
        return EngineSnapshot(active=active, cooldowns=cooldowns)

    # ── 内部 ─────────────────────────────────────────────────────────────────

    def _prune(self, now: float) -> None:
        self.envelopes = [env for env in self.envelopes if not env.expired(now)]

    @staticmethod
    def _resolve_tracks(
        spec: BehaviorSpec,
        request: BehaviorRequest,
        direction: str | None,
        ranges: Ranges,
    ) -> tuple[tuple[Track, ...], tuple[str, ...], tuple[str, ...]]:
        """触发时定标（AD-5）：按 ranges × intensity 折算轨道幅度。

        缺参/降级判定委托给单一真相 ``resolve_degradation``（W2/W5，与
        catalog `_behavior_info` 同口径）；本方法只负责在其给出的生效轨道
        计划上做幅度折算（方向符号 / 角度参数半量程定标）。

        Raises:
            UnresolvableParamsError: 主计划与降级计划所需参数均缺席。
        """
        resolution = resolve_degradation(spec, ranges, direction)
        if not resolution.resolvable:
            raise UnresolvableParamsError(
                f"{spec.name} 所需参数缺席且无可用降级：{list(resolution.missing)}"
            )

        tracks: list[Track] = []
        for plan in resolution.plans:
            amplitude = request.intensity * plan.scale * plan.sign
            if plan.signed_by_direction and direction is not None:
                amplitude *= DIRECTION_SIGNS[direction]
            if plan.angle_scaled:
                lo, hi, _ = ranges[plan.param]
                amplitude *= (hi - lo) / 2.0
            tracks.append(Track(param=plan.param, waveform=plan.waveform, amplitude=amplitude))
        return tuple(tracks), resolution.channels, resolution.degraded_channels

    @staticmethod
    def _rejected(code: str, detail: str, channels: tuple[str, ...] = ()) -> BehaviorReceipt:
        return BehaviorReceipt(
            status="rejected",
            behavior_id=uuid.uuid4().hex[:16],
            channels=list(channels),
            estimated_duration_ms=0,
            code=code,
            detail=detail,
        )
