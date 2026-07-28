"""Q3 external_priors 接线接口——MCP → Zero session.step(state_overrides=...) 载荷构造。

跨仓协议锚点：本模块数据形状与
D:\\Zero\\src\\orchestration\\external_prior.py 的 ExternalPrior / EXTERNAL_PRIOR_SCHEMA_VERSION
严格对齐（现场核验 2026-07-14）。本仓不 import Zero，对齐靠镜像类型别名 + 版本常量断言。

M3/M6 客户端 fail-fast（早于 Zero 报错、消息更清晰，阈值默认对齐 Zero 且走同名 env）：
- M6 流数上界 max_streams（默认 ZERO_MAX_EXTERNAL_STREAMS=5）。
- M3 精度上界 precision_cap（默认 ZERO_EXTERNAL_PRIOR_PRECISION_CAP=0.8）。
生理流的效价精度 Πv 在 Zero 侧 M2 被无条件覆写为 MIN_PRECISION，故客户端 M3 校验按 MIN 计
（不因 MCP 透传的高 Πv 误报），MCP 侧仍原样透传由 Zero 权威覆写。

各模态推荐精度默认（design.md §五·三席调和，env EXTERNAL_* 可调）经 recommended_precision /
build_recommended_prior 提供，供感知侧盖精度时参考。

用法示例::

    from src.mcp.zero.external_priors import build_external_priors_override
    # priors 由 PerceptionHub.collect() 产出
    override = build_external_priors_override(priors)  # 默认 M3/M6 阈值对齐 Zero
    # 未来 Zero MCP client 调用（当前不做真调用）：
    # await session.step(state_overrides=override)
"""

from __future__ import annotations

import logging
import math
import os
from enum import StrEnum

from src.agents.models.zero_affect import ModalityPrior

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 跨仓协议版本（M5）
# ---------------------------------------------------------------------------

EXTERNAL_PRIOR_SCHEMA_VERSION: int = 1
"""跨仓协议版本锚点。

须与 D:\\Zero\\src\\orchestration\\external_prior.py 的同名常量保持一致（M5，
跨仓回归断言应在 zerorepo 集成测试中 assertEqual 此值与 Zero 侧值）。
修改此值前须与 Zero 侧窗口协调，并同步更新两仓。
"""

# ---------------------------------------------------------------------------
# Zero 侧校验默认值镜像（M3 精度上界 / M6 流数上界 / MIN_PRECISION）
# ---------------------------------------------------------------------------

MIN_PRECISION: float = 1e-3
"""最小高斯精度，镜像 D:\\Zero\\src\\agents\\affect_math.py:21 的 MIN_PRECISION。

生理流（physio/eda/hrv/pupil/scr 前缀）的效价精度 Πv 会被 Zero M2 无条件覆写为此值
（EDA/HRV/瞳孔对效价盲，Kreibig 2010）；MCP 侧构造生理先验时也应直接给此值以示意图一致。
"""

ZERO_EXTERNAL_PRIOR_PRECISION_CAP_DEFAULT: float = 0.8
"""M3 单条外部先验精度上界默认值。

镜像 Zero AffectState.external_prior_precision_cap（D:\\Zero\\src\\orchestration\\state.py:228，
env ZERO_EXTERNAL_PRIOR_PRECISION_CAP 默认 0.8）。修改须与 Zero 侧协调（M3，防默认值漂移；
跨仓回归 assertEqual 此值与 Zero 侧 AffectState 字段默认）。
"""

ZERO_MAX_EXTERNAL_STREAMS_DEFAULT: int = 5
"""M6 每轮外部流数上界默认值。

镜像 Zero AffectState.max_external_streams（D:\\Zero\\src\\orchestration\\state.py:232，
env ZERO_MAX_EXTERNAL_STREAMS 默认 5）。修改须与 Zero 侧协调（M6，防默认值漂移；
跨仓回归 assertEqual 此值与 Zero 侧 AffectState 字段默认）。
"""


def _resolve_precision_cap(precision_cap: float | None) -> float:
    """解析 M3 精度上界：显式值优先，否则走 env ZERO_EXTERNAL_PRIOR_PRECISION_CAP（默认 0.8）。

    env 变量名与 Zero 侧同名（同一旋钮），保证两仓 fail-fast 阈值同步。
    env 值非法（无法解析为 float / ≤0）时 raise 带语境的 ValueError——与 M3 业务
    ValueError 区分，避免线上「配置错误」被误当成「精度超上界」（镜像 Zero AffectState
    字段 gt=0.0 约束，D:\\Zero\\src\\orchestration\\state.py:228）。
    """
    if precision_cap is not None:
        return precision_cap
    raw = os.getenv("ZERO_EXTERNAL_PRIOR_PRECISION_CAP")
    if raw is None:
        return ZERO_EXTERNAL_PRIOR_PRECISION_CAP_DEFAULT
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"env ZERO_EXTERNAL_PRIOR_PRECISION_CAP={raw!r} 无法解析为 float：{exc}"
        ) from exc
    if value <= 0.0:
        raise ValueError(
            f"env ZERO_EXTERNAL_PRIOR_PRECISION_CAP={value} 须 >0"
            "（镜像 Zero AffectState.external_prior_precision_cap 的 gt=0.0 约束）"
        )
    return value


def _resolve_max_streams(max_streams: int | None) -> int:
    """解析 M6 流数上界：显式值优先，否则走 env ZERO_MAX_EXTERNAL_STREAMS（默认 5）。

    env 变量名与 Zero 侧同名（同一旋钮），保证两仓 fail-fast 阈值同步。
    env 值非法（无法解析为 int / <0）时 raise 带语境的 ValueError——与 M6 业务
    ValueError 区分（镜像 Zero AffectState 字段 ge=0 约束，
    D:\\Zero\\src\\orchestration\\state.py:232）。
    """
    if max_streams is not None:
        return max_streams
    raw = os.getenv("ZERO_MAX_EXTERNAL_STREAMS")
    if raw is None:
        return ZERO_MAX_EXTERNAL_STREAMS_DEFAULT
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"env ZERO_MAX_EXTERNAL_STREAMS={raw!r} 无法解析为 int：{exc}") from exc
    if value < 0:
        raise ValueError(
            f"env ZERO_MAX_EXTERNAL_STREAMS={value} 须 ≥0"
            "（镜像 Zero AffectState.max_external_streams 的 ge=0 约束）"
        )
    return value


# ---------------------------------------------------------------------------
# 类型别名（镜像 Zero ExternalPrior）
# ---------------------------------------------------------------------------

ExternalPriorTuple = tuple[str, tuple[float, float], tuple[float, float]]
"""镜像 D:\\Zero\\src\\orchestration\\external_prior.py 的 ExternalPrior。

形状：(name: str, (μ_v, μ_a): tuple[float,float], (Π_v, Π_a): tuple[float,float])

- name   ：流标识，如 "vision" / "audio" / "physio/eda"；生理类流应以
           PHYSIO_STREAM_PREFIXES 中的前缀命名以触发 Zero M2 处理。
- (μ_v, μ_a)  ：效价/唤醒度均值，各维 [-1, 1]。
- (Π_v, Π_a)  ：效价/唤醒度精度（高斯），各维 > 0。

本仓不 import Zero，对齐靠此镜像别名 + EXTERNAL_PRIOR_SCHEMA_VERSION 版本常量。
"""

# ---------------------------------------------------------------------------
# 生理流前缀（M2）
# ---------------------------------------------------------------------------

PHYSIO_STREAM_PREFIXES: tuple[str, ...] = ("physio", "eda", "hrv", "pupil", "scr")
"""Zero 强制覆写效价精度 Πv 的流名前缀集合（M2，生理信号对效价盲）。

Zero 侧行为（D:\\Zero\\src\\orchestration\\external_prior.py，Kreibig 2010）：
凡流名匹配这些前缀的流，Zero 的 expand_external_priors() 将其效价精度 Πv
强制覆写为 MIN_PRECISION（极低精度=对效价几乎不贡献），以反映 EDA/HRV/瞳孔
对效价盲的生理学约束。唤醒度精度 Πa 保留原值（生理信号对唤醒可靠）。

MCP 侧行为：
- 本常量供 MCP 侧生理模态先验**命名自查**——生理模态的 ModalityPrior.modality
  应以这些前缀命名（如 "eda/sc"、"hrv/rmssd"、"pupil"），以确保 Zero 正确触发 M2。
- is_physio_stream() 提供快速检测；命名权威覆写仍由 Zero 侧做。
- 参考文献：Kreibig (2010) Autonomic nervous system activity in emotion.
  Biological Psychology, 84(3), 394-421.
"""


def is_physio_stream(name: str) -> bool:
    """判断流名是否属于生理类前缀（M2 命名自查）。

    匹配规则（任一即为 True）：
    - name 等于某前缀（完整匹配，如 "eda"）。
    - name 以「前缀 + '/'」开头（层级命名，如 "eda/sc"、"hrv/rmssd"）。
    - name 以「前缀 + '_'」开头（下划线命名，如 "pupil_diameter"）。

    本函数为 **advisory**——Zero 侧才是 M2 的权威检测与覆写方；
    MCP 侧在构造先验流时可调用此函数做命名自查与日志，不做任何精度覆写。

    Args:
        name: 流名（ModalityPrior.modality）。

    Returns:
        True 表示该流名匹配生理类前缀，Zero 将触发 M2 效价精度覆写。
    """
    for prefix in PHYSIO_STREAM_PREFIXES:
        if name == prefix or name.startswith(prefix + "/") or name.startswith(prefix + "_"):
            return True
    return False


def _triggers_zero_m2(name: str) -> bool:
    """精确镜像 Zero M2 生理流判定，用于 M3 客户端校验的生理流 Πv 豁免。

    Zero 的判定（D:\\Zero\\src\\agents\\affect_math.py:998）::

        if name.lower().startswith(_PHYSIO_PREFIXES):  # 大小写不敏感·裸前缀（无需分隔符）
            pi_v = MIN_PRECISION

    与 is_physio_stream 的**区别**（不可混用）：
    - is_physio_stream：面向 MCP 的**命名规范 advisory**——区分大小写、要求「前缀+分隔符」
      （如 "eda/sc"），故意严格以督促干净命名；用于 build_recommended_prior 的命名强制。
    - _triggers_zero_m2：对 **Zero 权威行为的忠实镜像**——大小写不敏感、裸前缀匹配
      （"EDA/SC"、"edax" 均触发）。M3 客户端校验必须用它做生理流 Πv 豁免，否则会**比 Zero
      更严**：Zero 对这些流先把 Πv 覆写为 MIN（必过 M3），客户端若判其非生理流则用原始高 Πv
      触发 M3 假阳性拒绝——误拒 Zero 会接受的载荷（违 mcp-integration「跨语言契约一致」）。
    """
    return name.lower().startswith(PHYSIO_STREAM_PREFIXES)


# ---------------------------------------------------------------------------
# 各模态推荐精度默认（design.md §五；EXTERNAL_* env 可调）
# ---------------------------------------------------------------------------


class ModalityKind(StrEnum):
    """感知模态类别，决定推荐精度默认（design.md §五·生物/心理/数学三席调和）。"""

    FACE = "face"
    AUDIO = "audio"
    PHYSIO = "physio"


# 各模态推荐 (Πv, Πa) 精度默认（design.md §五表；与 Zero .env.example EXTERNAL_* 数值对齐）。
# 依据：face valence 强 / audio arousal 强 / physio valence 盲——逐维信噪比不对称（M1）。
# ⚠ 新增 ModalityKind 成员时须同步补齐本 dict 与 _PRECISION_ENV_KEYS
#   （否则 recommended_precision 会 KeyError）。
_RECOMMENDED_PRECISION_DEFAULTS: dict[ModalityKind, tuple[float, float]] = {
    ModalityKind.FACE: (0.20, 0.12),
    ModalityKind.AUDIO: (0.10, 0.25),
    ModalityKind.PHYSIO: (MIN_PRECISION, 0.18),
}

# 各模态精度的 env 覆盖键（与 Zero .env.example 同名，供两仓一致调参）。
# ⚠ 新增 ModalityKind 成员时须同步补齐本 dict 与 _RECOMMENDED_PRECISION_DEFAULTS。
_PRECISION_ENV_KEYS: dict[ModalityKind, tuple[str, str]] = {
    ModalityKind.FACE: ("EXTERNAL_FACE_PRECISION_V", "EXTERNAL_FACE_PRECISION_A"),
    ModalityKind.AUDIO: ("EXTERNAL_AUDIO_PRECISION_V", "EXTERNAL_AUDIO_PRECISION_A"),
    ModalityKind.PHYSIO: ("EXTERNAL_PHYSIO_PRECISION_V", "EXTERNAL_PHYSIO_PRECISION_A"),
}


def recommended_precision(kind: ModalityKind) -> tuple[float, float]:
    """返回该模态推荐的 (Π_v, Π_a) 精度默认，供感知侧盖精度时参考。

    - 数值来自 design.md §五（三席调和），env EXTERNAL_*_PRECISION_{V,A} 可覆盖。
    - **physio 的 Π_v 恒返回 MIN_PRECISION**：Zero M2 对生理流无条件覆写 Π_v=MIN
      （对效价盲，Kreibig 2010），env 旋钮对 physio Π_v 仅作记录不实际放大，故此处
      直接给 MIN 以与最终注入形状一致；唤醒度 Π_a 仍取 env/默认。

    Args:
        kind: 模态类别（FACE / AUDIO / PHYSIO）。

    Returns:
        (Π_v, Π_a) 推荐精度二元组。
    """
    v_key, a_key = _PRECISION_ENV_KEYS[kind]
    default_v, default_a = _RECOMMENDED_PRECISION_DEFAULTS[kind]
    pi_a = float(os.getenv(a_key, str(default_a)))
    # physio Π_v 恒 MIN（Zero M2 会覆写），不读 env 避免无效路径；其余模态取 env/默认。
    if kind is ModalityKind.PHYSIO:
        pi_v = MIN_PRECISION
    else:
        pi_v = float(os.getenv(v_key, str(default_v)))
    return (pi_v, pi_a)


def build_recommended_prior(
    modality: str,
    mu: tuple[float, float],
    kind: ModalityKind,
    *,
    coping: float | None = None,
) -> ModalityPrior:
    """按模态推荐精度构造 ModalityPrior（感知侧盖精度的便捷入口）。

    精度取 recommended_precision(kind)（env 可调；physio Π_v=MIN）。kind=PHYSIO 时
    强制校验 name 带生理前缀，否则 Zero 无法触发 M2 覆写——早失败给出清晰命名指引。

    Args:
        modality: 流标识（physio 类须以 PHYSIO_STREAM_PREFIXES 前缀命名）。
        mu:       (μ_v, μ_a) 均值，各维 [-1, 1]。
        kind:     模态类别，决定推荐精度。
        coping:   可选 coping 分量（透传 ModalityPrior.coping）。

    Returns:
        以推荐精度构造的 ModalityPrior。

    Raises:
        ValueError: kind=PHYSIO 但 modality 未带生理前缀（Zero M2 无法触发覆写）。
    """
    if kind is ModalityKind.PHYSIO and not is_physio_stream(modality):
        raise ValueError(
            f"physio 模态先验 name={modality!r} 未带生理前缀 {PHYSIO_STREAM_PREFIXES}；"
            "Zero M2 无法触发效价精度覆写，请用 physio/eda/hrv/pupil/scr 前缀命名"
        )
    precision = recommended_precision(kind)
    return ModalityPrior(modality=modality, mu=mu, precision=precision, coping=coping)


# ---------------------------------------------------------------------------
# 载荷构造（build_external_priors_override）
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# EDA/HRV 预合并（Zero 科学家议会 2026-07-28 终裁 · MCP 侧执行项）
# ---------------------------------------------------------------------------

PHYSIO_MERGE_OMEGA_DEFAULT: float = 0.5
"""EDA/HRV 协方差交叉（CI）预合并的固定权重 ω —— **唯一不重复计可靠度的取值**。

背景：1 维 CI 严格最优会退化到 ω*∈{0,1}（完全弃用一方——二者都只测同一标量 arousal、
天然对齐，落入退化角），不实用；故采次优保守固定权重（Niehsen 2002 快速 CI）。

**为何必须是 0.5**（Zero 议会 2026-07-28 ω 档位终裁，本仓现场复核）：本仓约定 ω 乘在
``Π_eda`` 上，HRV 在合并均值里的实际权重是

    w_hrv(ω) = (1-ω)·Π_hrv / [ω·Π_eda + (1-ω)·Π_hrv]

ω=0.5 时分子分母同除 0.5，**精确退化**为 ``Π_hrv/(Π_eda+Π_hrv)``——这是对**任意**
(Π_eda, Π_hrv) 都成立的恒等式（现场随机核验 1e4 组，最大偏差 0.0），非本组数值的巧合。
即 ω=0.5 时 HRV **已自动拿到「它该得的」可靠度权重**（本组数值下 0.5714）；若再把 ω 设成
可靠度比例，等于**二次施加**可靠度。

更干净的判据（现场复核）：不预合并、双流直接进 Zero ``fuse_terms`` 得 μ=0.845714/Σπ=0.35；
ω=0.5 预合并得 **μ 完全不变（Δ=0.0）、Π 精确减半（比值 2.000000）**——只调「保守度」这一个
维度。任何 ω≠0.5 都会**同时**扰动 μ（实测 Δμ≈1e-2）与 Π，把本该正交的两个自由度耦合起来。

⚠ **勿换档**：ω=0.571 与 ω=0.4286 是「同一个错误的两个方向」（分别在 EDA/HRV 上多算一次
可靠度，实测把 μ 往相反方向各推 ~1e-2），Zero 议会均已弃用。
⚠ 跨仓约定错位提醒：Zero 材料里的 ω≈0.571 是加在 **HRV** 上的权重，与本仓「ω 乘 Π_eda」
约定相反（对应本仓 ω=0.4286）——该档本身已作废，此处记录以免日后误代入。

env ``ZERO_PHYSIO_MERGE_OMEGA`` 可覆盖，**仅供实验/对照，生产不应改**。
"""

PHYSIO_SUBSOURCE_PRECISION_A: dict[str, float] = {"eda": 0.15, "hrv": 0.20}
"""EDA / HRV 各自的唤醒度精度 Πa（可靠度分层，Zero 议会生物席，2026-07-28 采纳）。

依据：EDA 仅受**交感胆碱能**支配（Critchley 2002），与 Zero 内部 survival 流共源程度最高
→ 增量信息最少、可靠度最低（0.15）；HRV 主要由**迷走张力**驱动（Task Force 1996），走
前额叶-迷走神经内脏整合环路（Thayer & Lane 2000），是**不同的中枢-外周环路**→ 0.20。

⚠ 与 `_RECOMMENDED_PRECISION_DEFAULTS[PHYSIO]` 的 Πa=0.18 不冲突：0.18 是 §五 对
「physio 模态整体」的推荐默认（议会明确**不动**）；此处是合并前的**子源可靠度分层**，
ω=0.5 合并后 Π_merged=0.175 ≈ 0.18，量级自洽。
"""

PHYSIO_MERGED_MODALITY: str = "physio"
"""预合并后单条生理流的流名。

必须落在 `PHYSIO_STREAM_PREFIXES` 内以继续触发 Zero M2（Πv→MIN_PRECISION）——
"physio" 是 Zero `_PHYSIO_PREFIXES`（affect_math.py:971）的首项，现场核验通过。
"""

_PHYSIO_MERGE_SOURCES: tuple[str, str] = ("eda", "hrv")

_PHYSIO_MERGE_ARITY: int = 2
"""本模块合并式的**推导元数**（当前 = 二元 CI）。

治理不变量（Zero 议会 2026-07-28 转 CS 席治理项）：``build_external_priors_override`` 的 M6
流数按**合并后**计数，其成立前提是「合并流背后的相关性风险已被 CI 推导吸收」。若日后新增同源
子通道（如呼吸率 RSP）却**直接塞进现有二元公式复用 ω=0.5**，则 M6 计数为 1 的 ``physio``
背后可能藏进 3、4 条**朴素求和**的原始证据——**M6 的保险会在 Zero 与本仓双方都感知不到的
地方失效**。故新增源必须重走 N 元 CI 推导；本常量 + ``_assert_merge_arity_invariant()``
使「只加源、不改推导」在运行期硬失败，而非静默降低保守度。

----------------------------------------------------------------------------------
**N 元推导备案**（Zero 数学席 2026-07-28 裁定，*非当前行动项*；触发本守卫时直接引用）
----------------------------------------------------------------------------------

1. **ω_i=1/N 外推成立**，条件不比二元更严格（Chen, Arambel & Mehra 2002 已把 CI 推广到多源
   未知相关融合）。唯一条件——二元时也隐含存在、非 N 元新增——是各源上报的 (μ_i, Π_i) 须
   **诚实**（不低估自身真实误差）；这正是 Zero M2/M3 校验在工程化的东西。
2. **「加一条证据反而拉低 Π_merged」是 CI 的设计目的、非缺陷**：CI 不对「新源到底独不独立」
   下注，宁可在新源恰好高度冗余时也不虚增置信度。反例是朴素独立求和——重复上报同一信号 N 次
   会让 Π→∞（分布式融合领域的「信息回声/重复计数」病态）。
3. **ω_i=1/N 可证唯一**（二元「0.5 唯一」的严格推广）：要求 μ_merged(ω) 对**任意** μ_i 都等于
   自然精度加权均值，则两归一化权向量须成比例 ⟺ ω_i 对所有 i 相同 ⟺ 配合 Σω_i=1 得 ω_i=1/N。
   推论：**「保守度过高」不能靠调 ω_i 形状解决**（那会立刻重新引入可靠度重复计数），
   只能靠下述分组结构解决。
4. **通用定理（可复用，勿每加一源重推）**：标量情形 Y_CI(ω)=Σω_iΠ_i 对 ω **线性**，其最优化
   是单纯形上的线性规划；由 LP 基本定理（线性函数在凸多面体上极值必在顶点取得，与 N 无关），
   **对任意 N，标量源的 det/tr-最优 CI 恒退化为「全押 Π_i 最大者、其余弃权」**。既然该「最优
   解」已被判定不可接受（整支丢弃证据），工程上永远不该真去解它；正确起点始终是 ω_i=1/N。
5. ⚠ **分层合并优于扁平均分**（有已知耦合机制时）：RSA（呼吸性窦性心律不齐）是 HRV–RSP 间
   **有机制文献支撑**的耦合（Berntson et al. 1993），扁平三元均分会把这条**已知**结构信息扔掉。
   模板（两层嵌套 CI，组内均取均匀权重）：第 1 层合并强耦合的 (HRV, RSP) ω=0.5 → 「心肺」复合源；
   第 2 层该复合源与 EDA（路径更独立）再 ω=0.5。展开后隐含 ω：**EDA=1/2、HRV=RSP=1/4**
   （扁平三元则人人 1/3），正确体现「耦合对合计只算一票，不让它们联手稀释独立源」。
   ⚠ **巧合陷阱**（本仓曾踩）：用 (EDA .15 / HRV .20 / RSP .10) 算出的扁平 0.150 与分层 0.150
   **恰好重合**——因该组恰满足 Π_eda=(Π_hrv+Π_rsp)/2。故**该数字不能用来论证任何扁平/分层优劣**。
   现场复核：Π_eda=0.30 时 扁平 0.200 vs 分层 0.225；一般式 ``hier−flat=(2Π_eda−Π_hrv−Π_rsp)/12``
   （即重合点两侧符号相反：Π_eda=0.10 时反而扁平更高）。
6. **两条实现边界**：(a) 此处的「分层」与 Zero ``hierarchical_fuse``（跨处理流的预测编码层级）
   是**不同维度**，不应共用实现，需独立的更简单函数；(b) 「哪些通道分一组、耦合到什么程度」是
   **生物学事实判断**——届时流程是 **Zero 生物席定分组 → 数学席套分层 CI 模板 → 本仓实现**，
   而非直接沿用扁平 ω_i=1/N。本守卫的存在正是为了强制这个流程发生。
"""


def _assert_merge_arity_invariant() -> None:
    """守卫：子源集合与可靠度权重表须与推导元数一致，否则拒绝合并。

    Raises:
        NotImplementedError: 子源数 ≠ 推导元数，或子源集合与权重表键集不一致
            （即有人加了源却没重走 N 元推导）。
    """
    sources = set(_PHYSIO_MERGE_SOURCES)
    weighted = set(PHYSIO_SUBSOURCE_PRECISION_A)
    if len(sources) != _PHYSIO_MERGE_ARITY or sources != weighted:
        raise NotImplementedError(
            f"physio 预合并的推导元数为 {_PHYSIO_MERGE_ARITY}（二元 CI），但当前子源="
            f"{sorted(sources)}、可靠度权重表={sorted(weighted)}。**新增同源子通道必须重走 "
            "N 元 CI 推导**，不可复用二元式的 ω=0.5——否则 M6 按合并后计数=1，却掩盖 N 条"
            "朴素求和的原始证据，Zero 与本仓均无从察觉（Zero 议会 2026-07-28 治理项）。"
        )


def _resolve_merge_omega(omega: float | None) -> float:
    """解析 CI 合并权重 ω：显式值优先，否则走 env（默认 0.5）；须落 (0,1) 开区间。"""
    if omega is None:
        raw = os.getenv("ZERO_PHYSIO_MERGE_OMEGA")
        omega = PHYSIO_MERGE_OMEGA_DEFAULT if raw is None else float(raw)
    if not 0.0 < omega < 1.0:
        raise ValueError(
            f"physio 合并权重 ω={omega} 须落 (0,1) 开区间"
            "（端点等于完全弃用一方，即 1 维 CI 的退化角，议会已排除）"
        )
    return omega


def merge_physio_priors(
    priors: list[ModalityPrior],
    *,
    omega: float | None = None,
) -> list[ModalityPrior]:
    """把 EDA 与 HRV 两条**相关**生理流按固定权重 CI 预合并为单条 ``physio`` 流。

    **为何必须合并**（Zero 议会 2026-07-28 终裁，MCP 侧执行项）：EDA 与 HRV 高度相关
    （同测交感唤醒）。作两条独立 streams 直接进 Zero `fuse_terms` 时朴素 `Σπ = 0.15+0.20
    = 0.35`，**相当于把合并精度虚增 2 倍**——等于把「physio vs survival 共源」的风险
    在「EDA vs HRV」内部重新引入一遍（Berntson 1991：交感/副交感非严格独立轴）。

    **合并式**（协方差交叉信息形式，Julier & Uhlmann 1997 / Niehsen 2002 固定 ω 路线）::

        Π_merged = ω·Π_eda + (1-ω)·Π_hrv
        μ_merged = (ω·Π_eda·μ_eda + (1-ω)·Π_hrv·μ_hrv) / Π_merged

    ω=0.5 且采可靠度分层 (0.15, 0.20) 时 `Π_merged = 0.175`（议会给出的确切值）。

    效价维：两者均对效价盲（Kreibig 2010，Zero M2 无条件覆写 Πv=MIN_PRECISION），
    故合并结果恒 `μ_v=0.0`、`Π_v=MIN_PRECISION`——与 M2 最终形状一致。

    **精度取自可靠度分层常量而非入参先验的 Πa**：入参 Πa 通常是
    `recommended_precision(PHYSIO)` 的模态级 0.18（不区分 eda/hrv），而合并需要子源级
    可靠度；μ 仍全部取自真实读数。差异已在 `PHYSIO_SUBSOURCE_PRECISION_A` 说明。

    Args:
        priors: ModalityPrior 列表（通常来自 ``PerceptionHub.collect()``）。
        omega:  CI 固定权重，None = 走 env ``ZERO_PHYSIO_MERGE_OMEGA``（默认 0.5）。

    Returns:
        新列表：EDA/HRV 被替换为单条 ``physio``（落在首个生理流原位置），其余流保持原序。
        **不足两条**（缺 EDA 或缺 HRV）时原样返回——无相关性双计问题，无需合并。

    ⚠ **勿把 Π_merged 解释为「假设 EDA/HRV 完全相关（ρ=1）」**——数值上站不住：若真设 ρ=1，
    标准 GLS 约束权重下的最优融合是 ``max(Π_eda, Π_hrv)=0.20``（方差 5.0），而 0.175 对应
    方差 5.714，**比「假设完全相关」下能达到的最优还保守**（现场复核）。正确表述：
    ω=0.5 对 EDA–HRV 间**任意未知**相关系数 ρ∈[-1,1] 均给出一致的保守界
    （Julier & Uhlmann 1997），且是使均值估计不受可靠度重复加权污染的唯一取值；
    **不代表对 ρ 做出任何具体假设，尤其不等价于假设 ρ=1**。

    Raises:
        ValueError:          ω 不在 (0,1) 开区间。
        NotImplementedError: 子源数与推导元数不符（见 _assert_merge_arity_invariant）。
    """
    # 治理不变量：新增同源子通道必须重走 N 元 CI 推导，否则 M6 的保险会静默失效
    _assert_merge_arity_invariant()

    found: dict[str, int] = {}
    for index, prior in enumerate(priors):
        name = prior.modality.lower()
        for source in _PHYSIO_MERGE_SOURCES:
            # 首条命中为准（同源多流非预期，取首条并在下方 debug 记录）
            if source not in found and (
                name == source or name.startswith(source + "/") or name.startswith(source + "_")
            ):
                found[source] = index

    if len(found) < len(_PHYSIO_MERGE_SOURCES):
        logger.debug(
            "physio 预合并跳过：EDA/HRV 未同时在场（命中 %s），无相关性双计问题",
            sorted(found),
        )
        return list(priors)

    resolved_omega = _resolve_merge_omega(omega)
    eda_index, hrv_index = found["eda"], found["hrv"]
    weights = {
        "eda": resolved_omega * PHYSIO_SUBSOURCE_PRECISION_A["eda"],
        "hrv": (1.0 - resolved_omega) * PHYSIO_SUBSOURCE_PRECISION_A["hrv"],
    }
    pi_merged_a = weights["eda"] + weights["hrv"]
    mu_merged_a = (
        weights["eda"] * priors[eda_index].mu[1] + weights["hrv"] * priors[hrv_index].mu[1]
    ) / pi_merged_a

    merged = ModalityPrior(
        modality=PHYSIO_MERGED_MODALITY,
        mu=(0.0, mu_merged_a),
        precision=(MIN_PRECISION, pi_merged_a),
    )
    logger.debug(
        "physio 预合并：eda μa=%.4f + hrv μa=%.4f → %s μa=%.4f Πa=%.4f（ω=%.3f）",
        priors[eda_index].mu[1],
        priors[hrv_index].mu[1],
        PHYSIO_MERGED_MODALITY,
        mu_merged_a,
        pi_merged_a,
        resolved_omega,
    )

    keep_at, drop_at = min(eda_index, hrv_index), max(eda_index, hrv_index)
    result = list(priors)
    result[keep_at] = merged
    del result[drop_at]
    return result


def build_external_priors_override(
    priors: list[ModalityPrior],
    *,
    max_streams: int | None = None,
    precision_cap: float | None = None,
    merge_physio: bool = True,
) -> dict[str, list[ExternalPriorTuple]]:
    """将多模态先验列表构造为 Zero session.step(state_overrides=...) 的载荷。

    每条 ModalityPrior 经 as_stream() 转为 ExternalPriorTuple，形状天然匹配
    D:\\Zero\\src\\orchestration\\external_prior.py 的 ExternalPrior 定义（逐维 tuple
    精度，M1 逐维精度契约）。

    M6 客户端 fail-fast（流数上界）：
        流数超过 max_streams 立即 raise ValueError。默认对齐 Zero
        ZERO_MAX_EXTERNAL_STREAMS（5，走同名 env）；显式传值优先。客户端早失败比等
        Zero 内核抛错更清晰，阈值与 Zero 同步故不会误报。

    M3 客户端 fail-fast（精度上界）：
        每条先验的 Πv/Πa 须 ≤ precision_cap，否则 raise ValueError。默认对齐 Zero
        ZERO_EXTERNAL_PRIOR_PRECISION_CAP（0.8，走同名 env）；显式传值优先。
        ModalityPrior 已保证 Πv,Πa>0，此处补上界 cap。**生理流的 Πv 按 MIN_PRECISION
        计**（镜像 Zero M2-先于-M3 顺序）——MCP 透传的高 Πv 不会在此误报，Zero 侧才权威
        覆写为 MIN；生理流的 Πa 仍照常校验上界。

    M2 命名建议：
        生理模态先验（EDA/HRV/瞳孔/SCR）的 ModalityPrior.modality 应以
        PHYSIO_STREAM_PREFIXES 中的前缀命名，以触发 Zero 侧效价精度覆写（M2）。
        is_physio_stream() 可供命名自查；build_recommended_prior() 会强制该命名。

    Args:
        priors:        ModalityPrior 列表，由 PerceptionHub.collect() 产出。
        max_streams:   M6 客户端流数上限。None = 走 env ZERO_MAX_EXTERNAL_STREAMS（默认 5）。
        precision_cap: M3 客户端精度上界。None = 走 env（默认 0.8，见 _resolve_precision_cap）。
        merge_physio:  是否把 EDA/HRV 预合并为单条 physio 流（**默认 True**，Zero 议会
                       2026-07-28 终裁的 MCP 侧执行项，见 merge_physio_priors）。传 False
                       保留「两条独立生理流」的旧行为——仅供对照/回归，正常接线不应关闭
                       （会使 Zero 侧朴素 Σπ 虚增 2 倍）。

    Returns:
        ``{"external_priors": [(name,(μ_v,μ_a),(Π_v,Π_a)), ...]}``
        即传给 ``session.step(state_overrides=...)`` 的 dict 载荷（精度原样透传）。

    Raises:
        ValueError: len(priors) > max_streams（M6）；或某条先验 Πv/Πa > precision_cap（M3）。

    典型用法（未来 Zero MCP client 接入后，当前不做真调用）::

        priors = await hub.collect()
        override = build_external_priors_override(priors)  # M3/M6 阈值默认对齐 Zero
        # await session.step(state_overrides=override)
    """
    resolved_max = _resolve_max_streams(max_streams)
    resolved_cap = _resolve_precision_cap(precision_cap)

    # EDA/HRV 预合并（Zero 议会 2026-07-28 终裁：**默认开**）。二者高度相关，作两条独立流
    # 注入会把合并精度虚增 2 倍（Σπ=0.35）。默认开而非默认关的理由：这不是「新增能力」而是
    # **载荷构造的正确性修正**，且在 Zero 当前判据下 physio 流本就恒不点燃 → 今日**零可观测
    # 回归**，待 Zero 按轴加权公式落地后自动生效正确语义。需保留旧行为传 merge_physio=False。
    # M6 流数校验置于合并**之后**：合并减少流数，按最终注入形状计数才是 Zero 实际收到的。
    if merge_physio:
        priors = merge_physio_priors(priors)

    # M6：流数上界（默认对齐 Zero ZERO_MAX_EXTERNAL_STREAMS）
    if len(priors) > resolved_max:
        raise ValueError(
            f"M6 流数超过上限：收到 {len(priors)} 条先验流，max_streams={resolved_max}"
            "（默认对齐 Zero ZERO_MAX_EXTERNAL_STREAMS）。请减少注入流数或调高上限。"
        )

    streams: list[ExternalPriorTuple] = []
    for i, prior in enumerate(priors):
        stream = prior.as_stream()
        name, mu, (pi_v, pi_a) = stream
        # M7：μ 域校验，镜像 Zero affect_math.py:1039-1043（commit 0d4edb1，2026-07-28 已在其
        # main）。**必须校验出线 tuple 而非入参 priors**：① 本函数默认 merge_physio=True，
        # 合并会产出新 μ（μ_a 为 CI 加权值、μ_v 硬置 0.0），入口校验看不到它；② 入参是模型实例，
        # 而 ModalityPrior 的构造期校验可被 model_construct / model_copy / 鸭子类型伪造绕过
        # （实测四条绕过口均能把 (7.7, nan) 送出网），信任模型实例等于没有守卫。
        # 越界 μ 会直接抬高 Zero 的 stream_salience 买到本不该有的点燃资格；不拦则 Zero 侧
        # raise 后经 server ToolError → 我方 graceful_step 降级为 None = **整轮 step 静默丢失**。
        # NaN 亦由此条拦下（`-1.0 <= nan` 恒 False → 取反成立）。
        for dim, coord in (("μv", mu[0]), ("μa", mu[1])):
            if not (-1.0 <= coord <= 1.0):
                raise ValueError(
                    f"M7 μ 越界：先验流[{i}] {name!r} 的 {dim}={coord} 不在 [-1, 1] 内"
                    "（镜像 Zero affect_math.py:1039 的 M7 fail-fast）。"
                    "注意本校验作用于合并后的出线值，请检查 ModalityPrior 构造或合并输入。"
                )
        # M3：精度上界。镜像 Zero M2-先于-M3——生理流 Πv 会被 Zero 覆写为 MIN，故校验时按 MIN
        # 计（不因 MCP 透传的高 Πv 误报），MCP 侧仍原样透传由 Zero 权威覆写。生理流判定用
        # _triggers_zero_m2（忠实镜像 Zero 的 name.lower().startswith）而非 is_physio_stream，
        # 确保客户端不比 Zero 更严（大写/裸前缀流名也正确豁免）。
        triggers_m2 = _triggers_zero_m2(name)
        effective_pi_v = MIN_PRECISION if triggers_m2 else pi_v
        for dim, value in (("Πv", effective_pi_v), ("Πa", pi_a)):
            # 有限性先于上界：`value > cap` 对 NaN **恒 False**，NaN 精度会静默穿过本关。
            # Zero 侧无对应兜底——其 affect_math.py:1052 `pi_v <= 0.0` 与 :1058 `pi_v > cap`
            # 同为 NaN-恒 False，M7 又只守 μ 不守 Π → NaN 精度两侧都不 fail-fast，直接进
            # 融合数学产出 NaN 后验（比越界 μ 更隐蔽：后者至少被 Zero M7 响亮 raise）。
            if not math.isfinite(value):
                raise ValueError(
                    f"M3 精度非有限值：先验流[{i}] {name!r} 的 {dim}={value}。"
                    "NaN/inf 精度会静默污染 Zero 融合后验（Zero 侧无对应 fail-fast，"
                    "其 :1052/:1058 判据对 NaN 恒 False），故由 MCP 侧单边拦截。"
                )
            if value > resolved_cap:
                raise ValueError(
                    f"M3 精度超上界：先验流[{i}] {name!r} 的 {dim}={value} 超过 "
                    f"precision_cap={resolved_cap}（默认对齐 Zero "
                    "ZERO_EXTERNAL_PRIOR_PRECISION_CAP）。请降低精度或调高上界。"
                )
        streams.append(stream)
        if triggers_m2:
            logger.debug(
                "流 %r 匹配生理类前缀，Zero 将触发 M2 效价精度覆写（Πv→MIN_PRECISION）",
                name,
            )

    logger.debug(
        "build_external_priors_override: %d 条先验流（max_streams=%d, precision_cap=%s）",
        len(streams),
        resolved_max,
        resolved_cap,
    )
    return {"external_priors": streams}
