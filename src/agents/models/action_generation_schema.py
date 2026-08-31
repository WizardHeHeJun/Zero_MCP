"""instruction→ActionSpec 生成层的 LLM 工具调用 schema（蓝图决策 B）。

与 `src.agents.models.screen_snapshot.ActionSpec`（内部执行契约，唯一真相）
**故意分离**：本文件是**面向 LLM strict tool use 的生成期 schema**，二者经
`to_action_spec()` 适配器单向转换（生成 schema → ActionSpec），不反向依赖。

分离原因（硬坑，见现场核验）：Anthropic strict tool use 编译出的 JSON Schema
对数组只支持 `minItems ∈ {0, 1}`；`ActionSpec.coordinates: tuple[int, int]`
（等价 `prefixItems` + `maxItems=2`）在 strict 模式下编译必 400。生成层因此
用 `GroundingCoordinate{x, y}` 对象承载坐标，只在 `to_action_spec()` 内部转回
`ActionSpec.coordinates` 期望的 tuple——LLM 侧从不产出 tuple 形状。

设计依据：
- notes/2026-08-05-llm-integration-survey-k3k4-actionspec.md §4.1（Skyvern 按
  action_type 拆模型的正例 / browser-use #3293 大 union 单模型反例）。
- notes/2026-08-31-actionspec-litgate-refresh.md 复核②（prefixItems 硬坑维持
  有效，`{x,y}` 对象方案不变）。
- notes/2026-08-31-actionspec-generation-blueprint.md 决策 B/E（v1 生成集
  click/type/key/window_close/wait；risk_level 必填但只能被 classify_risk
  升级、不能降级——见 `src.orchestration.safety.action_guard.classify_risk`）。

v1 生成集刻意不含 `done`/`fail`：二者已被 Supervisor 的 task_status 终态覆盖
（`generate_action` 只在 Supervisor 判定 next_agent=="control" 即"未完成"时
才被调用），归入 Supervisor 层级而非本执行层 schema（复核⑤订正）。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.agents.models.screen_snapshot import ActionRisk, ActionSpec


class GroundingCoordinate(BaseModel):
    """坐标对象（strict tool use 兼容——见模块 docstring 的 prefixItems 硬坑）。

    仅在 `uia_hollow` 场景（感知层 UIA 内容树为空）由 LLM 自报，作 OCR 兜底
    定位通道；主通道是 `target_element_id` 引用感知快照 id，由生成层服务端
    解析 bbox 中心坐标（蓝图决策 C，不信 LLM 自报坐标防像素幻觉）。
    """

    model_config = ConfigDict(extra="forbid")

    x: int
    y: int


class ActionGenerationBase(BaseModel):
    """五个动作生成子模型的共享基类。

    - reasoning: LLM 对本次动作决策的说明（人读，供审计/调试，不参与执行逻辑）。
    - risk_level: LLM 自陈风险级别，必填。生成层原样透传进 `ActionSpec`，
      `ActionGuard.classify_risk` 取声明值与白名单基线的**较高者**——本字段
      只能被升级、不能降级绕过安全门（蓝图决策 E；UFO 论文实证 LLM 自陈
      safe_guard 是弱设计，故不能单独作为判定依据）。
    """

    model_config = ConfigDict(extra="forbid")

    reasoning: str
    risk_level: ActionRisk


class ClickActionInput(ActionGenerationBase):
    """点击动作。定位二选一：`target_element_id`（主通道）或 `coordinate`（OCR 兜底）。"""

    model_config = ConfigDict(extra="forbid")

    action_type: Literal["click"] = "click"
    target_element_id: str | None = None
    coordinate: GroundingCoordinate | None = None

    @model_validator(mode="after")
    def _require_target_or_coordinate(self) -> ClickActionInput:
        if self.target_element_id is None and self.coordinate is None:
            raise ValueError("click 动作必须提供 target_element_id 或 coordinate 之一")
        return self

    def to_action_spec(self, action_id: str) -> ActionSpec:
        """适配为内部执行契约 `ActionSpec`（坐标对象→tuple 在此发生）。"""
        return ActionSpec(
            action_id=action_id,
            action_type="click",
            target_element_id=self.target_element_id,
            coordinates=(self.coordinate.x, self.coordinate.y)
            if self.coordinate is not None
            else None,
            text_payload=None,
            risk_level=self.risk_level,
        )


class TypeActionInput(ActionGenerationBase):
    """输入文本动作。定位同 `ClickActionInput`（先定位焦点，再输入 `text`）。"""

    model_config = ConfigDict(extra="forbid")

    action_type: Literal["type"] = "type"
    text: str
    target_element_id: str | None = None
    coordinate: GroundingCoordinate | None = None

    @model_validator(mode="after")
    def _require_target_or_coordinate(self) -> TypeActionInput:
        if self.target_element_id is None and self.coordinate is None:
            raise ValueError("type 动作必须提供 target_element_id 或 coordinate 之一")
        return self

    def to_action_spec(self, action_id: str) -> ActionSpec:
        """适配为内部执行契约 `ActionSpec`（`text` → `text_payload`）。"""
        return ActionSpec(
            action_id=action_id,
            action_type="type",
            target_element_id=self.target_element_id,
            coordinates=(self.coordinate.x, self.coordinate.y)
            if self.coordinate is not None
            else None,
            text_payload=self.text,
            risk_level=self.risk_level,
        )


class KeyActionInput(ActionGenerationBase):
    """按键动作（如 "enter"、"ctrl+c"）。无坐标/元素定位——直接发往焦点控件。"""

    model_config = ConfigDict(extra="forbid")

    action_type: Literal["key"] = "key"
    key: str

    def to_action_spec(self, action_id: str) -> ActionSpec:
        """适配为内部执行契约 `ActionSpec`（`key` → `text_payload`，同现有 dispatch 约定）。"""
        return ActionSpec(
            action_id=action_id,
            action_type="key",
            target_element_id=None,
            coordinates=None,
            text_payload=self.key,
            risk_level=self.risk_level,
        )


class WindowCloseActionInput(ActionGenerationBase):
    """关闭窗口动作。`window_handle` 是独立 int 字段，不与坐标/元素定位混用。"""

    model_config = ConfigDict(extra="forbid")

    action_type: Literal["window_close"] = "window_close"
    window_handle: int

    def to_action_spec(self, action_id: str) -> ActionSpec:
        """适配为内部执行契约 `ActionSpec`。

        `window_handle` 经 `str()` 填入 `ActionSpec.target_element_id`——与
        `DesktopControlAgent._dispatch_write` 既有约定一致（该字段按整数字符串
        解析出 hwnd）。**不**借道 `coordinates`/`text_payload` 表达句柄，防止
        与坐标点击/文本输入语义双重污染（同蓝图 §拍板① 对 `text_payload` 的
        同款顾虑）。
        """
        return ActionSpec(
            action_id=action_id,
            action_type="window_close",
            target_element_id=str(self.window_handle),
            coordinates=None,
            text_payload=None,
            risk_level=self.risk_level,
        )


class WaitActionInput(ActionGenerationBase):
    """等待动作（元动作，K4 白名单 LOW_RISK）。

    `wait_ms` 边界 [100, 10000] 为**工程假设**（蓝图「工程假设清单」标注，
    实机标定项，留待后续）。
    """

    model_config = ConfigDict(extra="forbid")

    action_type: Literal["wait"] = "wait"
    wait_ms: int = Field(ge=100, le=10000)

    def to_action_spec(self, action_id: str) -> ActionSpec:
        """适配为内部执行契约 `ActionSpec`（拍板①：`wait_ms` 走独立可选字段，
        不复用 `text_payload` 防双重语义污染）。
        """
        return ActionSpec(
            action_id=action_id,
            action_type="wait",
            target_element_id=None,
            coordinates=None,
            text_payload=None,
            risk_level=self.risk_level,
            wait_ms=self.wait_ms,
        )
