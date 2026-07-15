"""Q3 external_priors 接线接口——MCP → Zero session.step(state_overrides=...) 载荷构造。

跨仓协议锚点：本模块数据形状与
D:\\Zero\\src\\orchestration\\external_prior.py 的 ExternalPrior / EXTERNAL_PRIOR_SCHEMA_VERSION
严格对齐（现场核验 2026-07-14）。本仓不 import Zero，对齐靠镜像类型别名 + 版本常量断言。

用法示例::

    from src.mcp.zero.external_priors import build_external_priors_override
    # priors 由 PerceptionHub.collect() 产出
    override = build_external_priors_override(priors, max_streams=8)
    # 未来 Zero MCP client 调用（当前不做真调用）：
    # await session.step(state_overrides=override)
"""

from __future__ import annotations

import logging

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


# ---------------------------------------------------------------------------
# 载荷构造（build_external_priors_override）
# ---------------------------------------------------------------------------


def build_external_priors_override(
    priors: list[ModalityPrior],
    *,
    max_streams: int | None = None,
) -> dict[str, list[ExternalPriorTuple]]:
    """将多模态先验列表构造为 Zero session.step(state_overrides=...) 的载荷。

    每条 ModalityPrior 经 as_stream() 转为 ExternalPriorTuple，形状天然匹配
    D:\\Zero\\src\\orchestration\\external_prior.py 的 ExternalPrior 定义
    （逐维 tuple 精度，docstring 明写"与 MCP as_zero_streams() 输出对齐"）。

    M6 客户端 fail-fast：
        若指定 max_streams 且流数超过上限，立即 raise ValueError，
        镜像 Zero 侧 expand_external_priors(max_streams=...) 的上界检查。
        客户端早失败比等 Zero 内核抛错更清晰，但不重复 Zero 的权威检查：
        max_streams=None（默认）时跳过客户端检查，由 Zero 兜底。

    M2 命名建议：
        生理模态先验（EDA/HRV/瞳孔/SCR）的 ModalityPrior.modality 应以
        PHYSIO_STREAM_PREFIXES 中的前缀命名，以触发 Zero 侧效价精度覆写（M2）。
        is_physio_stream() 可供命名自查。

    Args:
        priors:      ModalityPrior 列表，由 PerceptionHub.collect() 产出。
        max_streams: 客户端流数上限（可选）。None = 不做客户端检查，交 Zero 兜底。

    Returns:
        ``{"external_priors": [(name,(μ_v,μ_a),(Π_v,Π_a)), ...]}``
        即传给 ``session.step(state_overrides=...)`` 的 dict 载荷。

    Raises:
        ValueError: 当 max_streams 非 None 且 len(priors) > max_streams 时（M6）。

    典型用法（未来 Zero MCP client 接入后，当前不做真调用）::

        priors = await hub.collect()
        override = build_external_priors_override(priors, max_streams=8)
        # await session.step(state_overrides=override)
    """
    streams: list[ExternalPriorTuple] = [p.as_stream() for p in priors]

    if max_streams is not None and len(streams) > max_streams:
        raise ValueError(
            "M6 流数超过上限：收到 %d 条先验流，max_streams=%d。"
            "镜像 Zero expand_external_priors 的上界检查（客户端 fail-fast）。"
            % (len(streams), max_streams)
        )

    logger.debug(
        "build_external_priors_override: %d 条先验流（max_streams=%r）",
        len(streams),
        max_streams,
    )
    for stream in streams:
        name = stream[0]
        if is_physio_stream(name):
            logger.debug(
                "流 %r 匹配生理类前缀，Zero 将触发 M2 效价精度覆写（Πv→MIN_PRECISION）",
                name,
            )

    return {"external_priors": streams}
