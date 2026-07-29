"""Q3 external_priors 接线接口——MCP → Zero session.step(state_overrides=...) 载荷构造。

跨仓协议锚点：本模块数据形状与 Zero
`src/orchestration/external_prior.py::{ExternalPrior, EXTERNAL_PRIOR_SCHEMA_VERSION}`
严格对齐（现场核验 2026-07-14 / R7 复核 2026-07-29）。本仓不 import Zero，对齐靠镜像类型
别名 + 版本常量断言。跨仓引用一律「仓内相对路径::符号名」，**禁写对方行号与绝对路径**
（R7 双方约定：行号一次编辑即失效且腐烂不驱红；绝对路径把「Zero 装在哪」这个部署事实
钉进源码，是第二个易腐维度）。

M3/M6 客户端 fail-fast（早于 Zero 报错、消息更清晰，阈值默认对齐 Zero 且走同名 env）：
- M6 流数上界 max_streams（默认 ZERO_MAX_EXTERNAL_STREAMS=5）。
- M3 精度上界 precision_cap（默认 ZERO_EXTERNAL_PRIOR_PRECISION_CAP=0.8）。
生理流的效价精度 Πv 在 Zero 侧 M2 被无条件覆写为 MIN_PRECISION，故客户端 M3 校验按 MIN 计
（不因 MCP 透传的高 Πv 误报），MCP 侧仍原样透传由 Zero 权威覆写。

出网收口点另有两条 physio 专属 fail-fast，**覆盖面不相交、互不顶替**：
- M8 **MCP 侧单边**自律守卫（Zero 无对应旋钮、也拦不住）——「**配置越顶**」：出线 physio 流的
  Πa 不得高到使该流**不经开门动作**即越过 Zero 点燃门 `SALIENCE_THRESHOLD`。判据按最坏情形
  现算而非比常量，见 `PHYSIO_PRECISION_A_SELF_IGNITE_BOUND` / `_physio_self_ignite_salience`。
- M9 **跨仓协议**守卫——「**契约违反**」：出线 physio 流的 μv 必须恒 0.0（「生理对效价盲」，
  我方回件 §6-8 已入协议，Zero 侧同步落 `mu=(0.0, mu[1])`）。非零 μv 能在不换取任何后验
  影响力的前提下单买点燃资格，而 Πa 配得低时 M8 完全看不见它。

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

须与 Zero `src/orchestration/external_prior.py::EXTERNAL_PRIOR_SCHEMA_VERSION` 保持一致（M5，
跨仓回归断言应在 zerorepo 集成测试中 assertEqual 此值与 Zero 侧值）。
修改此值前须与 Zero 侧窗口协调，并同步更新两仓。
"""

# ---------------------------------------------------------------------------
# Zero 侧校验默认值镜像（M3 精度上界 / M6 流数上界 / MIN_PRECISION / 点燃门阈值）
# ---------------------------------------------------------------------------

MIN_PRECISION: float = 1e-3
"""最小高斯精度，镜像 Zero `src/agents/affect_math.py::MIN_PRECISION`。

生理流（physio/eda/hrv/pupil/scr 前缀）的效价精度 Πv 会被 Zero M2 无条件覆写为此值
（EDA/HRV/瞳孔对效价盲，Kreibig 2010）；MCP 侧构造生理先验时也应直接给此值以示意图一致。
"""

ZERO_EXTERNAL_PRIOR_PRECISION_CAP_DEFAULT: float = 0.8
"""M3 单条外部先验精度上界默认值。

镜像 Zero `src/orchestration/state.py::AffectState.external_prior_precision_cap`
（`Field(default=0.8, gt=0.0)`，env ZERO_EXTERNAL_PRIOR_PRECISION_CAP 默认 0.8）。
修改须与 Zero 侧协调（M3，防默认值漂移；
跨仓回归 assertEqual 此值与 Zero 侧 AffectState 字段默认）。
"""

ZERO_SALIENCE_THRESHOLD: float = 0.18
"""点燃门阈值，镜像 Zero `src/agents/affect_math.py::SALIENCE_THRESHOLD`（现场核验 2026-07-29）。

Zero 的判据是 `_select_fired` 里的 ``s >= threshold``——**取等即点燃**，故本仓 M8 守卫的
比较也必须用 ``>=``（见 `_physio_self_ignite_salience`）。

⚠ **这是本模块唯一一个「Zero 侧阈值」镜像，漂移风险实在**：Zero 改 SALIENCE_THRESHOLD 时
本常量不会自动跟随。防漂移靠两道既有锚点，**不是**靠这句注释：
- `tests/mcp/test_zero_contract_crosscheck.py::_ZERO_GATE_CONSTANTS` 逐值 pin Zero 源码里的
  `SALIENCE_THRESHOLD`（Zero 一改即红）；
- 同文件 ``test_physio_default_precision_stays_below_self_ignite_bound`` 从 Zero 源码**现算**
  上界并与 `PHYSIO_PRECISION_A_SELF_IGNITE_BOUND` 对账。
本常量另有一条直接 pin（``test_mirrored_salience_threshold_matches_zero_source``）把「产品码里
的镜像值」与 Zero 源码绑死——前两道锚点只覆盖测试侧的现算，覆盖不到产品码这个新镜像。
"""

ZERO_MAX_EXTERNAL_STREAMS_DEFAULT: int = 5
"""M6 每轮外部流数上界默认值。

镜像 Zero `src/orchestration/state.py::AffectState.max_external_streams`
（`Field(default=5, ge=0)`，env ZERO_MAX_EXTERNAL_STREAMS 默认 5）。
修改须与 Zero 侧协调（M6，防默认值漂移；
跨仓回归 assertEqual 此值与 Zero 侧 AffectState 字段默认）。
"""


def _resolve_precision_cap(precision_cap: float | None) -> float:
    """解析 M3 精度上界：显式值优先，否则走 env ZERO_EXTERNAL_PRIOR_PRECISION_CAP（默认 0.8）。

    env 变量名与 Zero 侧同名（同一旋钮），保证两仓 fail-fast 阈值同步。
    env 值非法（无法解析为 float / ≤0）时 raise 带语境的 ValueError——与 M3 业务
    ValueError 区分，避免线上「配置错误」被误当成「精度超上界」（镜像 Zero
    `src/orchestration/state.py::AffectState.external_prior_precision_cap` 的 gt=0.0 约束）。
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
    ValueError 区分（镜像 Zero
    `src/orchestration/state.py::AffectState.max_external_streams` 的 ge=0 约束）。
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
"""镜像 Zero `src/orchestration/external_prior.py::ExternalPrior`。

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

Zero 侧行为（`src/agents/affect_math.py::_PHYSIO_PREFIXES` 的 M2 消费点，Kreibig 2010）：
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

    Zero 的判定（`src/agents/affect_math.py::expand_external_priors` 的 M2 生理流分支）::

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

PHYSIO_PRECISION_A_SELF_IGNITE_BOUND: float = 0.359
"""**自点燃硬上界**：physio 的 Πa 一旦 ≥ 此值，我方 physio 流可自行越过 Zero 的点燃门。

推导（Zero `src/agents/affect_math.py::stream_salience` = hypot(μ)·mean(Π)、
同模块 `SALIENCE_THRESHOLD` = 0.18）：
本上界成立**依赖一条我方单边前提**——physio 流的 μv 恒 0，故 hypot(μ)=|μa| ≤ 1；
Πv 恒 MIN_PRECISION（**这一条才是 Zero M2 给的保证**）。
最坏情形 |μa|=1 时 salience = (MIN_PRECISION + Πa)/2 ≥ 0.18 ⟺ Πa ≥ 0.36 − 1e-3 = 0.359。

⚠ **归因订正（2026-07-29 跨仓现场核验，Zero 指认成立）**：μv≡0 **不是 Zero 保证的**。
Zero `expand_external_priors` 的 M2 分支（按 `_PHYSIO_PREFIXES` 命中）**只覆写 Πv、全程不碰 μ**；
紧邻的 M7 也只做 μ∈[-1,1] 域校验、不置零。Kreibig 2010 支撑的是「EDA/HRV 对效价盲」这一
**建模选择**，把它落成 `μv = 0.0` 的是**我方通道侧硬写**，共三处锚点：
`src/mcp/zero/channels/physio_channel.py` 的 `EdaChannel` 与 `HrvChannel` 各一处 `mu_v = 0.0`，
以及本模块 `merge_physio_priors` 出线的 `mu=(0.0, mu_merged_a)`。
⇒ 一旦任一 physio 流带**非零 μv**（v2 改口径 / 未来 RSP 子源 / 合并式产出新 μv），
hypot(μ) 最大到 √2，真实自点燃上界**收紧到 `2·SALIENCE_THRESHOLD/√2 − MIN_PRECISION`**
（今日阈值下真值 **0.2535584412271571**；对外文书里的 `0.2536` 是**向上取整的口径**，
差 4.16e-5）；届时 0.359 会**松约 30%**。故三处硬写锚点**不得删注释**。
⚠ **禁止手抄 0.2536 当判据**（回件 §6-8 数值订正 1，两仓同款约定）：钉一个比真界**松**的常量
等于守卫自带一条永远测不到的缝。凡进入断言/比较的地方一律从 `ZERO_SALIENCE_THRESHOLD`
现算；`0.2536` 只可出现在叙述性文字里，且须标明是向上取整口径。

✅ **该缺口已于 2026-07-29 堵上**（本条原文案为「现有守卫按 μv≡0 的闭式复算、**不会报错**
——即文案说的缺口在、断言却测不到」，现按实际实现如实改写）：出网收口点新增的 **M8 运行期
守卫**（`build_external_priors_override` 内，助手 `_physio_self_ignite_salience`）**不复算
本常量、也不与本常量比较**，而是对每条出线 physio 流按 ``hypot(μv, 1.0)·mean(Π)`` **现算**
最坏情形 salience 再与 `ZERO_SALIENCE_THRESHOLD` 比。**μv 取出线实测值、不假定其为 0**，
故 μv 一旦变非零，守卫施加的 Πa 硬顶**自动**从 0.359 收紧到 2·0.18/hypot(μv,1) − MIN
（μv=±1 时即上文 0.2535584412271571），无需任何人记得回来改数。0.359 因此降级为
「μv=0 这一特例下的闭式解 + 对外承诺的口径」，**不再是守卫的判据来源**——判据只有
Zero 阈值这一个镜像。

⚠ **不依赖 Zero 侧 μv 归零**：现场核验（2026-07-29，Zero main `69f9e88`，工作树对
affect_math.py 无改动）发现 Zero 已在 M2 分支补上 ``mu = (0.0, mu[1])``，即 μv≡0 **今天**
也是 Zero 侧的结构保证了。M8 **仍按未归零算**，两条理由：① 该归零是当日新增，而「把判别力
挂在对方当前处于哪一态」是本仓已立案的跨仓硬教训（同 M3′ isfinite 一条，实测对方状态可在
一天内变三次）；② 按未归零算只会**更保守**（我方今天出线 μv 恒 0，两种算法逐位同值，
零回归），代价为零。前提的守卫分布在三处（对应三处硬写）：通道侧 EDA/HRV 见
tests/mcp/test_zero_physio_channel.py 的 `test_mu_v_is_zero` / `test_hrv_mu_v_is_zero`；
合并出线见 tests/mcp/test_zero_external_priors.py::TestPhysioOutboundMuVZeroPremise；
通道→wire 端到端见 tests/mcp/test_zero_physio_channel.py::
`test_channel_priors_reach_wire_with_zero_mu_v`。

✅ **「写不写进跨仓协议」已拍板，不再是待议项**（原文案写「仍待双方拍板」，与我方**已发出**的
回件矛盾，2026-07-29 如实改写）：我方在
`notes/2026-07-29-mcp-reply-to-zero-asks.md` §6-8 **选 (a) 入协议**——Zero 在 M2 里对 physio
前缀流强制 `mu = (0.0, mu[1])`，**且我方同时在出境侧加 fail-fast**，不让对方一家兜。落地状态：
- **Zero 那半：已落**（其 `src/agents/affect_math.py` M2 分支现有 `mu = (0.0, mu[1])`，
  commit `8043176`，其 message 直接引用本仓回件 §6-5A/§6-8）。
- **我方那半：本次补齐 = M9**（`build_external_priors_override` 出网收口点，出线 μv≠0 即 raise；
  回归面见 tests/mcp/test_zero_external_priors.py::TestM9PhysioValenceContract）。
理由（回件原话）：合并产出的新 μ、`model_construct`/鸭子类型绕过口**只有我方出境侧看得见**；
两侧各封一半，μv≡0 才成为**两侧各自的**结构不变量。
⚠ **主句限定**（同 §6-8）：这只对**我方载荷**把 0.359 升为不变量；任何**其它** MCP client 仍可
向 Zero 送非零 μv，故 **Zero 侧的通用真界仍是 0.2535584412271571**。
⚠ **M8 的算法不因此改动**：它继续按「未归零」现算（μv 取实测），两条理由同上 ①②。M9 在前
使这条通路上 μv 恒 0，M8 的一般性成为纵深余量而非死码——M9 一旦被放宽/摘除，M8 自动按 √2
收紧而不是静默按 0.359 放行。

⚠ **为什么这条要写成常量而不是注释**：Πa 由 env ``EXTERNAL_PHYSIO_PRECISION_A`` 控，
Zero 侧只有 cap=0.8 的宽上界。而 Zero 的 D7 承诺（`exclude_physio_fusion` 默认 True，
按前缀把 physio 剔出融合）**只写在门开分支里**——门关（默认）走硬门阈值路径，D7 管不到。
即：我方单方面把 Πa 调过本值，就能在默认配置下让被判定为反号的 physio 真正进入 Zero 的
数值后验，绕过 Zero 应我方之请所做的跨仓承诺。这不是 Zero 能拦的，只能我方自律。

⚠ **两条运行期守卫的覆盖面是两条不相交的轴，一条顶不了另一条**（读到这里最容易犯的错）：

===========  ==========================  ================================================
守卫          管什么                       它**接不住**什么（须由另一条接）
===========  ==========================  ================================================
**M8**       **配置越顶**：μv 合规（=0）   μv=0.9、Πa=0.05 的伪造流——最坏 salience≈0.034
             但 Πa 高到该流不经开门动作     ≪0.18，**M8 全程放行**，却违反「physio 对效价盲」
             即可自行越过点燃门             这条被承诺要 fail-fast 的契约
**M9**       **契约违反**：出线 μv≠0.0     Πa 被调到 0.39（μv 正常为 0）——**M9 全程放行**，
             （该流根本不该带效价）         只有 M8 拦得住
===========  ==========================  ================================================

即：M8 只看「这个 Πa 危不危险」，M9 只看「这条流该不该有 μv」；两条**各自**都存在整类
它看不见的失效。M9 的完整论述见 `build_external_priors_override` 内 M9 处的块注释。

M8 这一条自身又分**两道**，缺一不可（覆盖面不同，别当重复）：
- **常量态**（`tests/mcp/test_zero_contract_crosscheck.py::`
  ``test_physio_default_precision_stays_below_self_ignite_bound``，推荐态与合并态双查）：
  只看 `_RECOMMENDED_PRECISION_DEFAULTS` 与子源可靠度这两个**源码常量**，
  故对 ``EXTERNAL_PHYSIO_PRECISION_A=0.39`` 这类 **env 覆盖恒绿**——这正是 Zero 18:25 裁定件
  点的那条缺口（我方交付 hrv 残差 σ=1.6 ⇒ Πa=0.39 时只核了 Zero 的 cap=0.8，漏了本硬顶）。
- **运行期态**（M8，`build_external_priors_override` 出网收口点）：校验**最终出线值**，
  env / 显式入参 / 合并产出 / `model_construct` 伪造 **四条通路全覆盖**。
"""


def _physio_self_ignite_salience(
    mu: tuple[float, float],
    pi_v_effective: float,
    pi_a: float,
) -> tuple[float, float]:
    """现算一条出线 physio 流的**最坏情形 salience** 及其对应的 Πa 硬顶（M8 判据核心）。

    镜像 Zero `src/agents/affect_math.py::stream_salience`::

        salience = hypot(μv, μa) · (Πv + Πa)/2

    与之的**两点刻意偏离**（都是「按最坏情形定，不按当轮读数定」的同一个决定）：

    1. **|μa| 一律按域上界 1.0 代入，不用当轮实测 μa**。M8 要拦的是**配置**级危险
       （「这个 Πa 使该流*有能力*自点燃」），不是「本轮这条载荷恰好越了线」。按实测 μa 算
       会让 ``Πa=0.39`` 这种越顶配置在低唤醒时段一路绿灯上线，只在某个高唤醒瞬间才首次
       触发——那时它已经在生产里了。按最坏情形算则**配置一落地就红**。
       代价是拦下「配了高 Πa 但恰好这轮 μa 小」的载荷；这正是意图，不是误报。
    2. **μv 取出线实测值，不假定为 0**（见 `PHYSIO_PRECISION_A_SELF_IGNITE_BOUND` 的
       「不依赖 Zero 侧 μv 归零」段）。今天出线 μv 恒 0 ⇒ hypot=1 ⇒ 硬顶恰为 0.359；
       μv 一旦变非零，hypot>1 ⇒ 硬顶**自动收紧**（μv=±1 时收到 2·T/√2 − MIN
       = 0.2535584412271571，即对外口径向上取整写的那个 0.2536），守卫无需改一个字。
       ⚠ 自 M9 落地起，`build_external_priors_override` 这条通路上 physio 的非零 μv 会被 M9
       先行拒绝，故本函数的非零-μv 分支在**该通路**上不可达；判别力改由本函数的**纯函数级**
       用例直接覆盖（见 tests 里对本函数的逐格调用）。保留该一般性是**有意的纵深**：M9 若被
       放宽/摘除，硬顶自动收紧，而不是静默退回 0.359。

    `pi_v_effective` 由调用方按 Zero M2 镜像给出（physio 恒 MIN_PRECISION）：Zero 对 physio
    无条件覆写 Πv=MIN，故用 wire 上的原始 Πv 会**比 Zero 更严**（与 M3 的 `_triggers_zero_m2`
    豁免同理）；而 wire Πv 若小于 MIN，Zero 反而把它抬到 MIN——取 MIN 两个方向都不低估。

    Args:
        mu:             出线 (μv, μa)（**已过 M7 域校验**，故必为 [-1,1] 内的有限数）。
        pi_v_effective: 按 Zero M2 镜像后的效价精度（physio 恒 MIN_PRECISION）。
        pi_a:           出线唤醒度精度（**已过 M3 有限性校验**）。

    Returns:
        ``(最坏情形 salience, 该 μv 下的 Πa 硬顶)``。硬顶 = 使最坏 salience 恰等于阈值的
        Πa，即 ``2·threshold/hypot(μv,1) − Πv_eff``；salience ≥ 阈值 ⟺ Πa ≥ 硬顶。
    """
    # |μa| 打满到域上界 1.0（最坏情形），μv 取实测——见上文偏离说明
    deviation = math.hypot(mu[0], 1.0)
    worst_salience = deviation * 0.5 * (pi_v_effective + pi_a)
    pi_a_ceiling = 2.0 * ZERO_SALIENCE_THRESHOLD / deviation - pi_v_effective
    return worst_salience, pi_a_ceiling


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

env ``ZERO_PHYSIO_MERGE_OMEGA`` 可覆盖，**仅供实验/对照，生产不应改**。该自律文字已于
2026-07-29（Zero 回执 R6）升级为可执行守卫：`_resolve_merge_omega` 对**任何**非默认 ω
（env 或显式入参两条通路）发 warning——不 raise（实验/对照是正当用途），数值行为不变。
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
"physio" 是 Zero `src/agents/affect_math.py::_PHYSIO_PREFIXES` 的首项，现场核验通过
（2026-07-29 复核仍成立；该元组同时被 M2 覆写与 D7 融合排除两处消费）。
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


_MERGE_OMEGA_WARN_MARKER: str = "physio 合并权重 ω 非默认"
"""非默认 ω 告警的稳定前缀（供测试按原因筛选，避免只断言「有 warning」而红在别的原因上）。"""


def _resolve_merge_omega(omega: float | None) -> float:
    """解析 CI 合并权重 ω：显式值优先，否则走 env（默认 0.5）；须落 (0,1) 开区间。

    **非默认 ω 一律 warning**（Zero 2026-07-29 回执 R6：把 `PHYSIO_MERGE_OMEGA_DEFAULT`
    docstring 里「仅供实验/对照，生产不应改」的**自律文字**升级成可执行守卫）。三条设计决定：

    1. **判据写在「解析后的值」上，不是「env 是否设了」**：按 `raw is not None` 判会漏掉显式入参
       ``merge_physio_priors(..., omega=0.6)`` 这条通路；按 ``omega != DEFAULT`` 判两条通路全覆盖。
    2. **warn 而非 raise**：实验/对照是这个旋钮的正当用途（既有对照用例就设非默认值），raise 会
       破零回归。数值行为一字未动。
    3. **有意不去重**：模块级 `_warned` 标志会让测试顺序相关（先跑的用例吃掉 warning、后跑的假
       绿）。代价是非默认 ω 下每次合并各一条 warn——本就不该在生产出现。

    Args:
        omega: 显式权重；None = 走 env ``ZERO_PHYSIO_MERGE_OMEGA``，仍缺省则取终裁默认。

    Returns:
        落在 (0,1) 开区间的 ω。

    Raises:
        ValueError: ω 不在 (0,1) 开区间。
    """
    if omega is None:
        raw = os.getenv("ZERO_PHYSIO_MERGE_OMEGA")
        omega = PHYSIO_MERGE_OMEGA_DEFAULT if raw is None else float(raw)
    if not 0.0 < omega < 1.0:
        raise ValueError(
            f"physio 合并权重 ω={omega} 须落 (0,1) 开区间"
            "（端点等于完全弃用一方，即 1 维 CI 的退化角，议会已排除）"
        )
    if omega != PHYSIO_MERGE_OMEGA_DEFAULT:
        logger.warning(
            "%s：ω=%s ≠ 议会终裁默认 %s（来源：env ZERO_PHYSIO_MERGE_OMEGA 或显式入参）。"
            "ω≠0.5 会**同时**扰动 μ 与 Π，并直接改变发往 Zero 的 wire Πa；而这是**我方私有旋钮**"
            "——Zero 侧无同名旋钮、无守卫能观测到这次偏移，出境数值的变动在对面完全不可见。"
            "仅供实验/对照，生产勿用。",
            _MERGE_OMEGA_WARN_MARKER,
            omega,
            PHYSIO_MERGE_OMEGA_DEFAULT,
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

    M8 自点燃硬顶 fail-fast（**MCP 侧单边自律，Zero 拦不住**）：
        每条**出线** physio 流按 ``hypot(μv, 1.0)·mean(Π)`` 现算最坏情形 salience，
        ≥ Zero `SALIENCE_THRESHOLD`（0.18）即 raise ValueError。等价于对 Πa 施加硬顶
        （μv=0 时即 `PHYSIO_PRECISION_A_SELF_IGNITE_BOUND`=0.359，μv≠0 时自动收紧）。
        与 M3 的关系是**互补而非冗余**：Zero 的 cap=0.8 宽得多、根本接不住这条，且越过
        点火阈值**不报错、只静默放行**（Zero 2026-07-29 裁定件原话要点）。详见
        `PHYSIO_PRECISION_A_SELF_IGNITE_BOUND` 与 `_physio_self_ignite_salience`。

    M9 physio 效价契约 fail-fast（跨仓协议·**我方那半**）：
        每条**出线** physio 流的 μv 必须恒 0.0（「生理信号对效价盲」），否则 raise ValueError。
        我方回件 §6-8 已选 (a) 入协议，Zero 侧同步在 M2 落 `mu = (0.0, mu[1])`（commit
        8043176）；本条是我方承诺的出境侧那半，**不依赖对方状态**。
        与 M8 **不可互相顶替**：M8 只看 Πa 危不危险，对「μv=0.9 但 Πa=0.05」的伪造流
        （最坏 salience≈0.034 ≪ 0.18）全程放行；M9 只看该流该不该带 μv，对「μv=0 但 Πa=0.39」
        全程放行。同时违反时**先报 M9**（契约违反是根因，按不该存在的 μv 去调 Πa 只修症状）。

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
        ValueError: len(priors) > max_streams（M6）；某条出线先验 μ 越 [-1,1] 域（M7）；
            某条出线 physio 流的 μv ≠ 0.0（M9）；Πv/Πa 非有限或 > precision_cap（M3）；
            或某条出线 physio 流的 Πa 达到自点燃硬顶（M8）。**M3/M7/M8/M9 均校验合并后的
            出线值**，不是入参 priors。逐流内的执行序为 M7 → M9 → M3 → M8。

    典型用法（未来 Zero MCP client 接入后，当前不做真调用）::

        priors = await hub.collect()
        override = build_external_priors_override(priors)  # M3/M6 阈值默认对齐 Zero
        # await session.step(state_overrides=override)
    """
    resolved_max = _resolve_max_streams(max_streams)
    resolved_cap = _resolve_precision_cap(precision_cap)

    # EDA/HRV 预合并（Zero 议会 2026-07-28 终裁：**默认开**）。二者高度相关，作两条独立流
    # 注入会把合并精度虚增 2 倍（Σπ=0.35）。默认开而非默认关的理由：这不是「新增能力」而是
    # **载荷构造的正确性修正**，且在 Zero 默认配置下 physio 流恒亚阈 → 今日**零可观测回归**。
    # 需保留旧行为传 merge_physio=False。
    #
    # ⚠ 2026-07-29 跨仓核验订正：原注释写的「待 Zero 按轴加权公式落地后自动生效正确语义」
    # **不成立**。「按轴加权 D + θ'=0.28」方案已在 Zero 议会第三轮被「硬门摘出数值通路
    # （线A，ZERO_IGNITION_GATE_FUSION）」取代；且 Zero 落的是 **D7 默认排除 physio**
    # （`ignite` 门开分支按 `_PHYSIO_PREFIXES` 剔除，`exclude_physio_fusion` 默认 True，
    # 系应我方「EDA 反号，宁可继续门掉」之请）。故不存在「自动生效」——门开后 physio 反而
    # 被显式剔出融合，须 Zero 单边显式关闭 exclude_physio_fusion 才可能参与数值。
    # 另注：「恒亚阈」是**默认精度下的性质、非结构保证**——Πa 走 env
    # ZERO_EXTERNAL_PHYSIO_PRECISION_A、Zero 侧仅有 cap=0.8 的上界，抬到 ≥0.359 时
    # physio 在**门关**硬门下即可自行过阈进入融合，且该路径不受 D7 排除保护（见下方
    # PHYSIO_PRECISION_A_SELF_IGNITE_BOUND）。
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
        # M7：μ 域校验，镜像 Zero `src/agents/affect_math.py::expand_external_priors` 的 M7
        # μ 域 fail-fast（commit 0d4edb1，2026-07-28 起已在其 main）。
        # **必须校验出线 tuple 而非入参 priors**：① 本函数默认 merge_physio=True，
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
                    "（镜像 Zero expand_external_priors 的 M7 μ 域 fail-fast）。"
                    "注意本校验作用于合并后的出线值，请检查 ModalityPrior 构造或合并输入。"
                )
        # 生理流判定用 _triggers_zero_m2（忠实镜像 Zero 的 name.lower().startswith）而非
        # is_physio_stream，确保「Zero 认定是 physio 的流」与「我方施加额外约束的流」**完全同集**
        # ——下方 M3 豁免、M8、M9 三处共用这一个判定，任何一处换判定都会造成两仓认定错位。
        # effective_pi_v：Zero M2 对 physio 无条件覆写 Πv=MIN，三处论断一律按覆写后的值算。
        triggers_m2 = _triggers_zero_m2(name)
        effective_pi_v = MIN_PRECISION if triggers_m2 else pi_v

        # M9：physio「对效价盲」契约 fail-fast —— 我方 2026-07-29 回件
        # `notes/2026-07-29-mcp-reply-to-zero-asks.md` §6-8 承诺的**我方那半**，本次补齐落码。
        # 协议内容：physio 前缀流的出线 μv 恒 0.0。Zero 已落它那半（其
        # `src/agents/affect_math.py::expand_external_priors` 的 M2 分支现有 `mu = (0.0, mu[1])`，
        # commit 8043176 的 message 直接引用本仓回件 §6-5A/§6-8）；我方**不依赖对方当前处于
        # 哪一态**——合并产出的新 μ、`model_construct`/鸭子类型伪造这两类绕过口**只有我方出境侧
        # 看得见**，两侧各自封一半，μv≡0 才成为**两侧各自的**结构不变量（回件 §6-8 原话）。
        #
        # **与 M8 的分工（覆盖面不相交，别当重复）**：
        # - M9 = **契约违反**：这条流根本不该带 μv，与 Πa 配得多低无关。μv=0.9、Πa=0.05 的伪造
        #   流最坏 salience≈0.034 ≪ 0.18，**M8 全程放行**，只有 M9 接得住。
        # - M8 = **配置越顶**：μv 合规（=0）但 Πa 高到该流不经开门动作就能自行过点燃门。
        #
        # **为什么 M9 排在 M8 之前**：两条同时被违反时（μv≠0 且 Πa 越顶），M8 的消息会按那个
        # **本就不该存在**的 μv 现算出一个收紧后的硬顶，把结论导向「把 Πa 降到 0.2535… 以下」；
        # 照做则契约违反原样上 wire，只是不再点燃——修掉了症状、留下了病。M9 的消息直指根因
        # （μv 不该非零），且 μv 归 0 后 M8 的硬顶自动放回 0.359，原 Πa 配置往往当场即合规。
        # **但 M7 仍先于 M9**：μv=NaN/越域时 M7 的诊断更具体（域错误），且 M9 的 `!= 0.0` 要有
        # 意义，前提正是 M7 已确立 mu[0] 是 [-1,1] 内的有限数（`nan != 0.0` 恒 True，若无 M7 在
        # 前，NaN μv 会被误报成「契约违反」而非「域错误」）。
        # 与 M7/M3/M8 同理，校验的是 as_stream() 的**出线值**而非入参 priors：合并会产出新 μ，
        # 而 ModalityPrior 的构造期校验有四条绕过路径，出网函数是唯一必经收口点。
        if triggers_m2 and mu[0] != 0.0:
            raise ValueError(
                f"M9 physio 效价契约违反：先验流[{i}] {name!r} 的出线 μv={mu[0]} ≠ 0.0。"
                "「生理信号对效价盲」是本仓已发出的跨仓协议条款（回件 §6-8），physio 流的 μv "
                "必须恒为 0.0。⚠ 归因（勿再写反）：Kreibig 2010 只是**建模依据**；真正把它落成 "
                "μv=0.0 的是**我方通道侧硬写**（EdaChannel / HrvChannel 各一处 `mu_v = 0.0`，"
                "加 merge_physio_priors 出线的 `mu=(0.0, μa)`），**不是** Zero M2 的保证"
                "——M2 只覆写 Πv。Zero 已按 §6-8 在其 M2 分支一并归零（commit 8043176），但本"
                "守卫**不依赖对方状态**：合并产出的新 μ 与 model_construct/鸭子类型伪造只有我方"
                "出境侧看得见，两侧各封一半，μv≡0 才是**两侧各自的**结构不变量。"
                f"危害：非零 μv 能在**不换取任何后验影响力**的前提下（Πv_eff={effective_pi_v} "
                "已把效价贡献压到可忽略）**单买点燃资格**——Zero 的 "
                "stream_salience=hypot(μv,μa)·mean(Π) 里 μv 只经 hypot 进入，与 Πv 无关；本条把"
                f"偏离模长从 |μa|={abs(mu[1]):.6g} 抬到 hypot={math.hypot(mu[0], mu[1]):.6g}，"
                "与越界 μ「买到本不该有的点燃资格」是同一失效模式换了个入口。"
                "怎么办：查上游通道/合并式是否改了 μv 口径，或调用方是否绕过 ModalityPrior 的"
                "构造期校验伪造了先验；确需让 physio 携带效价，须先跨仓与 Zero 重开 §6-8——"
                "这是契约级语义变更，不是本仓可单方面决定的参数调整。"
            )

        # M3：精度上界。镜像 Zero M2-先于-M3——生理流 Πv 会被 Zero 覆写为 MIN，故校验时按 MIN
        # 计（不因 MCP 透传的高 Πv 误报），MCP 侧仍原样透传由 Zero 权威覆写；生理流的 Πa 照常
        # 校验上界。豁免用的 triggers_m2/effective_pi_v 已在上方求出（客户端不得比 Zero 更严）。
        for dim, value in (("Πv", effective_pi_v), ("Πa", pi_a)):
            # 有限性先于上界：`value > cap` 对 NaN **恒 False**，NaN 精度会静默穿过本关。
            # 本条是 MCP 侧**单边收口，与对方状态解耦**：Zero
            # `src/agents/affect_math.py::expand_external_priors` 的 M3 判据
            # （`pi_v <= 0.0` 正值关、`pi_v > cap` 上界关）同为比较式、对 NaN 恒 False，
            # 其 M7 又只守 μ 不守 Π。对方是否在 M3 前置有限性校验，是**随其提交/回退变动的
            # 运行时事实**——2026-07-29 当天就实测到三态：main `11c25b0` 无 → 其未提交工作树
            # 出现 M3′ `isfinite` → main `332cb40` 起落地。我方一律不依赖对方当前处于哪一态，
            # 「同一天内变了三次」本身就是不该把判别力挂在对方状态上的证据。
            # NaN 精度若漏过会直接进融合数学产出 NaN 后验，比越界 μ 更隐蔽
            # （后者至少被 M7 响亮 raise）。
            if not math.isfinite(value):
                raise ValueError(
                    f"M3 精度非有限值：先验流[{i}] {name!r} 的 {dim}={value}。"
                    "NaN/inf 精度会静默污染 Zero 融合后验（其 M3 两条判据均为比较式、"
                    "对 NaN 恒 False，M7 只守 μ），故不论对侧是否兜底均由 MCP 侧单边拦截。"
                )
            if value > resolved_cap:
                raise ValueError(
                    f"M3 精度超上界：先验流[{i}] {name!r} 的 {dim}={value} 超过 "
                    f"precision_cap={resolved_cap}（默认对齐 Zero "
                    "ZERO_EXTERNAL_PRIOR_PRECISION_CAP）。请降低精度或调高上界。"
                )
        # M8：physio 自点燃硬顶（Zero 2026-07-29 18:25 裁定件点名的缺口 → 本仓承诺落码）。
        # 位置**必须**在 M7/M3 之后：M7 保证 μ 有限且 ∈[-1,1]、M3 保证 Π 有限，M8 才敢做算术
        # （NaN 会让 `>=` 恒 False，守卫静默塌缩——与 M3 有限性先于上界是同一条教训）。
        # 也在 M9 之后（见 M9 处「为什么 M9 排在 M8 之前」）。**一条后果要显式承认**：M9 在前
        # ⇒ 凡走到这里的 physio 流 μv 必为 0.0 ⇒ hypot(μv,1)≡1 ⇒ 本函数这条通路上 M8 的硬顶
        # 恒为 0.359，其「按实测 μv 自动收紧」的一般性**在这条通路上不再可达**。仍保留不改，
        # 因为那正是纵深：M9 若被放宽/摘除（协议重开、或有人只删这一条），M8 会**自动**改按
        # √2 收紧到 2·T/√2 − MIN 而不是静默按 0.359 放行——判据留在 `_physio_self_ignite_salience`
        # 的纯函数层，那里对非零 μv 仍逐格可测（见其测试）。
        # 落在出网函数内、逐条校验 `as_stream()` 的**出线值**，而非入参 priors：本函数默认
        # merge_physio=True 会产出**新的** μ 与 Πa（子源可靠度合并），校验入参根本看不到它；
        # 且 ModalityPrior 的构造期校验有四条绕过路径（构造后赋值 / model_construct /
        # model_copy / 鸭子类型伪造），出网函数是唯一必经收口点。
        # 只对 physio 流施加：vision/audio 过阈点燃是它们的本职（face 推荐精度下最坏
        # salience=0.16，本就贴着 0.18），而 physio 才是那条「Zero 应我方之请门掉、
        # 却只在门开分支兑现」的流。判定用 _triggers_zero_m2（忠实镜像 Zero 的前缀判定，
        # 与 M3 豁免同一套），确保「Zero 认定是 physio 的流」与「我方施加硬顶的流」完全同集。
        if triggers_m2:
            worst_salience, pi_a_ceiling = _physio_self_ignite_salience(mu, effective_pi_v, pi_a)
            # `>=` 而非 `>`：Zero `_select_fired` 的判据是 `s >= threshold`——**取等即点燃**。
            # 实测 Πa 恰为 0.359 时 salience 逐位等于 0.18（浮点无残差），故 0.359 本身必红。
            if worst_salience >= ZERO_SALIENCE_THRESHOLD:
                raise ValueError(
                    f"M8 physio 自点燃越界：先验流[{i}] {name!r} 的 Πa={pi_a} 已达自点燃硬顶 "
                    f"{pi_a_ceiling:.6g}——该流出线 μv={mu[0]}，|μa| 打满到域上界 1.0 时 "
                    f"salience={worst_salience:.6g} ≥ Zero 点燃门阈值 "
                    f"{ZERO_SALIENCE_THRESHOLD}（硬顶 = 2·阈值/hypot(μv,1) − Πv_eff，"
                    f"Πv_eff={effective_pi_v} 为 Zero M2 覆写后的值）。"
                    "后果：该 physio 流**不经任何开门动作**就能在 Zero 默认配置（硬门、"
                    "IGNITION_BETA=None）下自行过阈进入数值后验，绕过 Zero 应我方"
                    "「EDA 反号、宁可继续门掉」之请所落的 D7 承诺——D7 按前缀剔除 physio "
                    "**只写在门开分支里**，门关这条默认路径它管不到。"
                    "⚠ 归因：先撞到的是 Zero 的**点火阈值** SALIENCE_THRESHOLD"
                    f"={ZERO_SALIENCE_THRESHOLD}，**不是**它的精度上界 "
                    f"cap={resolved_cap}——cap 宽得多、接不住这一条，而且越过点火阈值"
                    "**不会报错**，只会**静默**让 physio 过门。故只能由本仓在出网口自拦。"
                    "怎么办：调低 Πa（env EXTERNAL_PHYSIO_PRECISION_A / 子源常量 "
                    "PHYSIO_SUBSOURCE_PRECISION_A / 显式传入的 ModalityPrior.precision）；"
                    "确需抬到硬顶之上，须先跨仓与 Zero 确认——这是契约级语义变更，"
                    "不是本仓可单方面决定的参数调整。"
                )
            logger.debug(
                "流 %r 匹配生理类前缀，Zero 将触发 M2 效价精度覆写（Πv→MIN_PRECISION）；"
                "M8 自点燃余量：最坏 salience=%.6g < 阈值 %s（Πa=%.6g，硬顶 %.6g）",
                name,
                worst_salience,
                ZERO_SALIENCE_THRESHOLD,
                pi_a,
                pi_a_ceiling,
            )
        streams.append(stream)

    logger.debug(
        "build_external_priors_override: %d 条先验流（max_streams=%d, precision_cap=%s）",
        len(streams),
        resolved_max,
        resolved_cap,
    )
    return {"external_priors": streams}
