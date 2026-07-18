"""Zero↔MCP 边界契约层公开 API（蓝图 §4 T5）。

本包聚合：
- 契约数据模型（来自 src/agents/models/zero_affect）
- 感知输入通路（PerceptionHub / PerceptionChannel / CallablePerceptionChannel）
- 表达消费通路（ExpressionRouter / ExpressionSink / HeadPolicy）
- 渲染终端（RenderFrame / RenderingExpressionSink）
- 韵律映射器（ProsodyParams / LinearProsodyMapper）
- FACS 映射器（AU_TO_ARKIT / ArkitFacsMapper）
- 生理映射器（PhysiologyParams / LinearPhysiologyMapper）
- Zero 协议镜像（Zero*Protocol，runtime_checkable）
- 规范常量（FACS_KEYS / FACS_KEYS_EXT / COPING_DRIVEN_AUS / TEXT_LABELS）
- external_priors 接线（Q3，EXTERNAL_PRIOR_SCHEMA_VERSION / build_external_priors_override）
- Zero MCP Client（Task 1-4，ZeroLinkClient + 三个自定义异常）
"""

from __future__ import annotations

# -- 契约数据模型 & 常量 -----------------------------------------------------
from src.agents.models.zero_affect import (
    COPING_DRIVEN_AUS,
    FACS_KEYS,
    FACS_KEYS_EXT,
    TEXT_LABELS,
    AffectStimulus,
    ExpressionBundle,
    ExpressionHead,
    LanguageOutput,
    ModalityPrior,
    PhysiologyChannel,
    ProsodyChannel,
)

# -- 感知输入通路 -------------------------------------------------------------
from src.mcp.zero.channels import (
    AudioChannel,
    CallablePerceptionChannel,
    EdaChannel,
    HrvChannel,
    VisionChannel,
)

# -- Zero MCP Client（Task 1-4）-----------------------------------------------
from src.mcp.zero.client import (
    ZeroLinkCallError,
    ZeroLinkClient,
    ZeroLinkConnectionError,
    ZeroLinkDisabledError,
)

# -- 表达消费通路 -------------------------------------------------------------
from src.mcp.zero.expression_sink import (
    ExpressionRouter,
    ExpressionSink,
    FacsMapper,
    HeadPolicy,
    PhysiologyMapper,
    ProsodyMapper,
)

# -- external_priors 接线（Q3）------------------------------------------------
from src.mcp.zero.external_priors import (
    EXTERNAL_PRIOR_SCHEMA_VERSION,
    MIN_PRECISION,
    PHYSIO_STREAM_PREFIXES,
    ZERO_EXTERNAL_PRIOR_PRECISION_CAP_DEFAULT,
    ZERO_MAX_EXTERNAL_STREAMS_DEFAULT,
    ExternalPriorTuple,
    ModalityKind,
    build_external_priors_override,
    build_recommended_prior,
    is_physio_stream,
    recommended_precision,
)

# -- FACS / 生理 / 韵律 映射器 ------------------------------------------------
from src.mcp.zero.mappers import (
    AU_TO_ARKIT,
    ArkitFacsMapper,
    LinearPhysiologyMapper,
    LinearProsodyMapper,
    PhysiologyParams,
    ProsodyParams,
)
from src.mcp.zero.perception import (
    PerceptionChannel,
    PerceptionHub,
)

# -- Zero 协议镜像 ------------------------------------------------------------
from src.mcp.zero.protocols import (
    ZeroChannelDecoder,
    ZeroConversationModel,
    ZeroCopingChannelDecoder,
    ZeroLanguageModel,
)

# -- 渲染终端 -----------------------------------------------------------------
from src.mcp.zero.sinks import RenderFrame, RenderingExpressionSink

__all__ = [
    # 常量
    "COPING_DRIVEN_AUS",
    "FACS_KEYS",
    "FACS_KEYS_EXT",
    "TEXT_LABELS",
    # 契约数据模型
    "AffectStimulus",
    "ExpressionBundle",
    "ExpressionHead",
    "LanguageOutput",
    "ModalityPrior",
    "PhysiologyChannel",
    "ProsodyChannel",
    # 感知输入通路
    "CallablePerceptionChannel",
    "EdaChannel",
    "HrvChannel",
    "AudioChannel",
    "VisionChannel",
    "PerceptionChannel",
    "PerceptionHub",
    # 表达消费通路
    "ExpressionRouter",
    "ExpressionSink",
    "FacsMapper",
    "HeadPolicy",
    "PhysiologyMapper",
    "ProsodyMapper",
    # FACS 映射器
    "AU_TO_ARKIT",
    "ArkitFacsMapper",
    # 生理映射器
    "LinearPhysiologyMapper",
    "PhysiologyParams",
    # 韵律映射器
    "LinearProsodyMapper",
    "ProsodyParams",
    # 渲染终端
    "RenderFrame",
    "RenderingExpressionSink",
    # Zero 协议镜像
    "ZeroChannelDecoder",
    "ZeroCopingChannelDecoder",
    "ZeroConversationModel",
    "ZeroLanguageModel",
    # external_priors 接线（Q3）
    "EXTERNAL_PRIOR_SCHEMA_VERSION",
    "MIN_PRECISION",
    "PHYSIO_STREAM_PREFIXES",
    "ZERO_EXTERNAL_PRIOR_PRECISION_CAP_DEFAULT",
    "ZERO_MAX_EXTERNAL_STREAMS_DEFAULT",
    "ExternalPriorTuple",
    "ModalityKind",
    "build_external_priors_override",
    "build_recommended_prior",
    "is_physio_stream",
    "recommended_precision",
    # Zero MCP Client（Task 1-4）
    "ZeroLinkClient",
    "ZeroLinkDisabledError",
    "ZeroLinkConnectionError",
    "ZeroLinkCallError",
]
