"""Zero 协议结构化镜像（runtime_checkable Protocol）——T2。

**不 import Zero 代码库**（CLAUDE.md AD-2 / mcp-integration.md）。
本仓以 typing.Protocol 逐字镜像 Zero 的协议签名，docstring 证据一律写 Zero 侧
**仓内相对路径 + `::` + 符号名**，**禁写对方行号**（含 `xxx.py:123` 与裸 `:123` 两种形态），
也不写 `D:\\Zero\\...` 绝对路径——R7 双方约定（2026-07-29）。理由：跨仓行号一次编辑即全量
失效，且**腐烂不驱红=静默失效**；符号名才是契约，路径只是加速定位的提示（Zero 的 agents 层
仍在迁移，路径本身也会再漂）。
契约漂移由跨仓子进程核对测试兜底（marker zerorepo，不构成 import 耦合）。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# ChannelDecoder 镜像
# ---------------------------------------------------------------------------


@runtime_checkable
class ZeroChannelDecoder(Protocol):
    """镜像 Zero `src/agents/expression.py::ChannelDecoder`。

    Zero 原签名（同步）::

        class ChannelDecoder(Protocol):
            def predict_channels(
                self, valence: float, arousal: float
            ) -> dict[str, Any]: ...

    predict_channels 是**同步**方法。
    返回 dict 内含 facs_au / text_label / physiology / prosody 通道；
    具体结构见 bprint §2 / Zero
    `src/agents/models/composite.py::CompositeChannelDecoder.predict_channels`。
    """

    def predict_channels(self, valence: float, arousal: float) -> dict[str, Any]:
        """同步解码 (v, a) → 表达通道 dict。

        证据：Zero `src/agents/expression.py::ChannelDecoder`（Protocol 原定义）。
        """
        ...


@runtime_checkable
class ZeroCopingChannelDecoder(ZeroChannelDecoder, Protocol):
    """镜像 Zero `src/agents/expression.py::ExpressionAgent._decode` 的 coping 扩展接口。

    Zero 运行时通过 `getattr(self.decoder, "predict_channels_coping", None)` 探测该方法
    是否存在来判断是否走 coping 分支。本协议继承 ZeroChannelDecoder 以保持
    子协议语义——实现方须同时满足两个协议。

    predict_channels_coping 是**同步**方法，四个必填位置参数。
    """

    def predict_channels_coping(
        self,
        valence: float,
        arousal: float,
        coping_potential: float,
        facs_extended: bool,
    ) -> dict[str, Any]:
        """同步解码 (v, a, coping) → 表达通道 dict（含 coping 相关 AU）。

        证据：Zero `src/agents/expression.py::ExpressionAgent._decode` 的 getattr 探测分支
        （该分支按 `(v, a, coping_potential, facs_extended)` 四位置参数调用）。
        参数均为必填位置参数（无默认值），与 Zero 调用侧一致。
        facs_extended=True 时解码器输出 FACS_KEYS_EXT（13 维，任务 D 起含 AU17/AU26）；
        False 时输出 FACS_KEYS（5 维）——两常量见 Zero
        `src/agents/models/facs_decoder.py::{FACS_KEYS, FACS_KEYS_EXT}`。
        """
        ...


# ---------------------------------------------------------------------------
# ConversationModel 镜像
# ---------------------------------------------------------------------------


@runtime_checkable
class ZeroConversationModel(Protocol):
    """镜像 Zero `src/agents/language.py::ConversationModel`。

    ⚠ 归属订正（2026-07-29 跨仓现场核验）：该 Protocol **已从 `orchestration/chat_driver.py`
    迁出**（chat_driver.py 内 `class ConversationModel` 现零命中，只余 `ChatTurn` 与
    `ChatDriver`）；调用侧仍在 `src/orchestration/chat_driver.py::ChatDriver`。

    两个方法均为 **async**（调用侧见 Zero
    `src/orchestration/chat_driver.py::ChatDriver.step`）。

    注：`build_graph()` 无 conversation_model 参数（Zero
    `src/orchestration/graph.py::build_graph`）——ConversationModel 由图外 ChatDriver
    持有，不注入图内。
    """

    async def appraise_text(self, text: str) -> tuple[float, float]:
        """async：将输入文本评价为 (valence, arousal) 二元组，各维 [-1, 1]。

        证据：Zero `src/agents/language.py::ConversationModel.appraise_text`（返回类型）；
        Zero `src/orchestration/chat_driver.py::ChatDriver.step` 的 Stimulus 组装段
        （v→goal_congruence，a→intensity=min(1, max(floor, |a|))，a 走强度通道非直传）。

        不含 coping_potential（coping 是独立通道，state.coping_potential_state）。
        """
        ...

    async def converse(
        self,
        history: list[dict[str, str]],
        affect: tuple[float, float],
        retrieved: str = "",
        *,
        push: bool = False,
        relationship_hint: str = "",
    ) -> str:
        """async：给定对话历史与当前情感状态，生成下一轮回复文本。

        证据：Zero `src/orchestration/chat_driver.py::ChatDriver.step` 的
        `self.lm.converse(...)` 调用侧。
        history: 对话历史，每条 {"role": "user"|"assistant", "content": str}。
        affect: (valence, arousal)，当前情感状态。
        push: 是否主动推送（关系动力学门控）。
        relationship_hint: 关系语境提示词（可选）。
        """
        ...


# ---------------------------------------------------------------------------
# LanguageModel 镜像
# ---------------------------------------------------------------------------


@runtime_checkable
class ZeroLanguageModel(Protocol):
    """镜像 Zero `src/agents/language.py::LanguageModel`。

    generate 为 **async**。返回 Zero `src/agents/language.py::LanguageDraft`——
    **现形状仅 `{text: str, affect: tuple[float, float]}`**（2026-07-29 跨仓现场核验：
    该 dataclass 上已无 `iters` / `consistency`）。
    本仓不 import Zero，注解为 Any，消费方按 LanguageOutput 契约解析。
    """

    async def generate(
        self,
        *,
        affect: tuple[float, float],
        context: str,
        retrieved: str,
        feedback: str | None,
        appraisal: str = "",
    ) -> Any:
        """async：生成情感着色的语言输出（LanguageDraft）。

        证据：Zero `src/agents/language.py::LanguageModel.generate`（签名）；
        Zero `src/agents/language.py::LanguageDraft`（返回结构）。

        返回值形状（本仓不 import Zero，故注解 Any）::

            {
                "text": str,               # 生成文本
                "affect": (float, float),  # (v, a)，JSON 后为 list
            }

        ⚠ 内容订正（2026-07-29 跨仓现场核验）：此处此前写的 `iters` / `consistency`
        两键**已不在 Zero 的 LanguageDraft 上**。本仓消费模型 `LanguageOutput` 仍**宽松保留**
        这两个带默认值的可选字段（保超集不收窄，防旧载荷解析炸），但它们不再构成
        Zero 侧的契约要求。
        消费方用 LanguageOutput.model_validate(...) 解析。
        """
        ...
