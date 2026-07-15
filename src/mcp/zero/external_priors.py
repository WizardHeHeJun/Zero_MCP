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


def build_external_priors_override(
    priors: list[ModalityPrior],
    *,
    max_streams: int | None = None,
    precision_cap: float | None = None,
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

    # M6：流数上界（默认对齐 Zero ZERO_MAX_EXTERNAL_STREAMS）
    if len(priors) > resolved_max:
        raise ValueError(
            f"M6 流数超过上限：收到 {len(priors)} 条先验流，max_streams={resolved_max}"
            "（默认对齐 Zero ZERO_MAX_EXTERNAL_STREAMS）。请减少注入流数或调高上限。"
        )

    streams: list[ExternalPriorTuple] = []
    for i, prior in enumerate(priors):
        stream = prior.as_stream()
        name, _mu, (pi_v, pi_a) = stream
        # M3：精度上界。镜像 Zero M2-先于-M3——生理流 Πv 会被 Zero 覆写为 MIN，故校验时按 MIN
        # 计（不因 MCP 透传的高 Πv 误报），MCP 侧仍原样透传由 Zero 权威覆写。生理流判定用
        # _triggers_zero_m2（忠实镜像 Zero 的 name.lower().startswith）而非 is_physio_stream，
        # 确保客户端不比 Zero 更严（大写/裸前缀流名也正确豁免）。
        triggers_m2 = _triggers_zero_m2(name)
        effective_pi_v = MIN_PRECISION if triggers_m2 else pi_v
        for dim, value in (("Πv", effective_pi_v), ("Πa", pi_a)):
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
