"""Zero↔MCP 边界契约层公开 API（蓝图 §4 T5）。

本包聚合：
- 契约数据模型（来自 src/agents/models/zero_affect）
- 感知输入通路（PerceptionHub / PerceptionChannel）
- 表达消费通路（ExpressionRouter / ExpressionSink / HeadPolicy）
- 韵律映射器（ProsodyParams / LinearProsodyMapper）
- Zero 协议镜像（Zero*Protocol，runtime_checkable）
- 规范常量（FACS_KEYS / FACS_KEYS_EXT / COPING_DRIVEN_AUS / TEXT_LABELS）
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

# -- 表达消费通路 -------------------------------------------------------------
from src.mcp.zero.expression_sink import (
    ExpressionRouter,
    ExpressionSink,
    FacsMapper,
    HeadPolicy,
    PhysiologyMapper,
    ProsodyMapper,
)

# -- 韵律映射器 ---------------------------------------------------------------
from src.mcp.zero.mappers import (
    LinearProsodyMapper,
    ProsodyParams,
)

# -- 感知输入通路 -------------------------------------------------------------
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
    "LinearProsodyMapper",
    "ModalityPrior",
    "PhysiologyChannel",
    "ProsodyChannel",
    "ProsodyParams",
    # 感知输入通路
    "PerceptionChannel",
    "PerceptionHub",
    # 表达消费通路
    "ExpressionRouter",
    "ExpressionSink",
    "FacsMapper",
    "HeadPolicy",
    "PhysiologyMapper",
    "ProsodyMapper",
    # Zero 协议镜像
    "ZeroChannelDecoder",
    "ZeroCopingChannelDecoder",
    "ZeroConversationModel",
    "ZeroLanguageModel",
]
