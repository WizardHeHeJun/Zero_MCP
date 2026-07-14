"""Zero 协议结构化镜像（runtime_checkable Protocol）——T2。

**不 import Zero 代码库**（CLAUDE.md AD-2 / mcp-integration.md）。
本仓以 typing.Protocol 逐字镜像 Zero 的协议签名，docstring 挂 D:\\Zero path:line 证据。
契约漂移由跨仓子进程核对测试兜底（marker zerorepo，不构成 import 耦合）。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# ChannelDecoder 镜像
# ---------------------------------------------------------------------------


@runtime_checkable
class ZeroChannelDecoder(Protocol):
    """镜像 D:\\Zero\\src\\expression.py:21-24 ChannelDecoder Protocol。

    Zero 原签名（同步）::

        class ChannelDecoder(Protocol):
            def predict_channels(
                self, valence: float, arousal: float
            ) -> dict[str, Any]: ...

    predict_channels 是**同步**方法（expression.py:21-24）。
    返回 dict 内含 facs_au / text_label / physiology / prosody 通道；
    具体结构见 bprint §2 / composite.py:57-99。
    """

    def predict_channels(self, valence: float, arousal: float) -> dict[str, Any]:
        """同步解码 (v, a) → 表达通道 dict。

        证据：D:\\Zero\\src\\expression.py:21-24（ChannelDecoder Protocol 原定义）。
        """
        ...


@runtime_checkable
class ZeroCopingChannelDecoder(ZeroChannelDecoder, Protocol):
    """镜像 D:\\Zero\\src\\expression.py:48-54 的 coping 扩展接口。

    Zero 运行时通过 getattr 探测 predict_channels_coping 是否存在来判断是否
    走 coping 分支（expression.py:48-54）。本协议继承 ZeroChannelDecoder 以保持
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

        证据：D:\\Zero\\src\\expression.py:48-54。
        参数均为必填位置参数（无默认值），与 Zero 调用侧一致。
        facs_extended=True 时解码器输出 FACS_KEYS_EXT（11 维）；
        False 时输出 FACS_KEYS（5 维）（facs_decoder.py:16-42）。
        """
        ...


# ---------------------------------------------------------------------------
# ConversationModel 镜像
# ---------------------------------------------------------------------------


@runtime_checkable
class ZeroConversationModel(Protocol):
    """镜像 D:\\Zero\\src\\orchestration\\chat_driver.py:184 ConversationModel Protocol。

    两个方法均为 **async**（chat_driver.py:286-322）。

    注：build_graph() 无 conversation_model 参数（graph.py:54-61）——
    ConversationModel 由图外 ChatDriver 持有，不注入图内。
    """

    async def appraise_text(self, text: str) -> tuple[float, float]:
        """async：将输入文本评价为 (valence, arousal) 二元组，各维 [-1, 1]。

        证据：D:\\Zero\\src\\agents\\language.py:85（appraise_text 返回类型）；
        D:\\Zero\\src\\orchestration\\chat_driver.py:297-306（v→goal_congruence，
        a→intensity=min(1, max(floor, |a|))，a 走强度通道非直传）。

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

        证据：D:\\Zero\\src\\orchestration\\chat_driver.py:313-320（converse 调用侧）。
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
    """镜像 D:\\Zero\\src\\agents\\language.py:15-17 LanguageModel Protocol。

    generate 为 **async**（language.py:35-54）。
    返回 LanguageDraft（language.py:15-17），形状为 {text: str, affect: tuple[float,float], ...}；
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

        证据：D:\\Zero\\src\\agents\\language.py:35-54（generate 签名）；
        language.py:58-85（LanguageDraft 结构：{text:str, affect:tuple, iters:int,
        consistency:float}）。

        返回值形状（本仓不 import Zero，故注解 Any）::

            {
                "text": str,               # 生成文本
                "affect": (float, float),  # (v, a)，JSON 后为 list
                "iters": int,              # 迭代轮次
                "consistency": float,      # 一致性分数
            }

        消费方用 LanguageOutput.model_validate(...) 解析。
        """
        ...
