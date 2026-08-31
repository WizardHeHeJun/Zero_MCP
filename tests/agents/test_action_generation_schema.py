"""action_generation_schema 单测（ActionSpec 生成层蓝图 PR-α，tasks 1+2）。

覆盖：
  1. strict tool use 兼容自查（判别性核心，口径=2026-08-31 现场核验的官方
     Not Supported 清单）：5 个子模型 `model_json_schema()` 递归无
     `prefixItems`、无超出 minItems∈{0,1} 的数组约束、**无数值约束
     minimum/maximum/multipleOf、无字符串约束 minLength/maxLength**（PR #28
     审查后补齐——此前只扫数组类，wait_ms 的 Field(ge/le) 漏网即实证）。
     `default`/optional 字段在官方 Supported 列表，action_type 带默认值合法。
  2. 5 类 `to_action_spec()` 字段映射（含 coordinate→tuple、window_handle→str、
     wait_ms 透传；risk_level 透传——wait 例外，强制 READ_ONLY 见 7）。
  3. Click/Type「target 与 coordinate 至少其一」校验。
  4. WaitActionInput.wait_ms 边界 [100, 10000]（model_validator 实现，不进 schema）。
  5. `_ACTION_RISK_WHITELIST["wait"]` == READ_ONLY（task 2 + PR #28 WARN②）。
  6. `ActionSpec.wait_ms` 加字段零回归：默认构造不传仍合法。
  7. wait 的 risk_level 强制 READ_ONLY（忽略 LLM 声明，防 TOCTOU 语义倒置）。
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from src.agents.models.action_generation_schema import (
    ClickActionInput,
    GroundingCoordinate,
    KeyActionInput,
    TypeActionInput,
    WaitActionInput,
    WindowCloseActionInput,
)
from src.agents.models.screen_snapshot import ActionRisk, ActionSpec
from src.orchestration.safety.action_guard import _ACTION_RISK_WHITELIST

# ---------------------------------------------------------------------------
# 1. strict tool use 兼容自查：无 prefixItems / 无受限数组约束
# ---------------------------------------------------------------------------

_GENERATION_MODELS: list[type[BaseModel]] = [
    ClickActionInput,
    TypeActionInput,
    KeyActionInput,
    WindowCloseActionInput,
    WaitActionInput,
]


def _walk_schema_nodes(node: Any) -> list[dict[str, Any]]:
    """递归展开 JSON Schema 树（含 $defs），收集所有 dict 节点。"""
    nodes: list[dict[str, Any]] = []
    if isinstance(node, dict):
        nodes.append(node)
        for value in node.values():
            nodes.extend(_walk_schema_nodes(value))
    elif isinstance(node, list):
        for item in node:
            nodes.extend(_walk_schema_nodes(item))
    return nodes


class TestStrictToolUseSchemaCompat:
    """5 个子模型的 `model_json_schema()` 不得触发 Anthropic strict 硬坑。"""

    @pytest.mark.parametrize("model_cls", _GENERATION_MODELS, ids=lambda c: c.__name__)
    def test_no_prefix_items(self, model_cls: type[BaseModel]) -> None:
        """无 `prefixItems`（tuple 编译产物，strict 下必 400 的根因）。"""
        schema = model_cls.model_json_schema()
        for node in _walk_schema_nodes(schema):
            assert "prefixItems" not in node, (
                f"{model_cls.__name__} schema 含 prefixItems: {node}"
            )

    @pytest.mark.parametrize("model_cls", _GENERATION_MODELS, ids=lambda c: c.__name__)
    def test_array_constraints_within_strict_bounds(self, model_cls: type[BaseModel]) -> None:
        """任何数组类型节点的 `minItems`/`maxItems` 若声明，须落在 {0, 1} 内。"""
        schema = model_cls.model_json_schema()
        for node in _walk_schema_nodes(schema):
            if node.get("type") != "array":
                continue
            for key in ("minItems", "maxItems"):
                if key in node:
                    assert node[key] in (0, 1), (
                        f"{model_cls.__name__} schema 数组节点 {key}={node[key]} "
                        "超出 strict 兼容范围 {0, 1}"
                    )

    @pytest.mark.parametrize("model_cls", _GENERATION_MODELS, ids=lambda c: c.__name__)
    def test_no_numeric_or_string_constraints(self, model_cls: type[BaseModel]) -> None:
        """无数值/字符串约束关键字（官方 Not Supported 清单，2026-08-31 现场核验：
        "Numerical constraints (such as minimum, maximum, multipleOf)" 与
        "String constraints (minLength, maxLength)" 均不支持）。

        判别性实证：PR #28 首版 `wait_ms: int = Field(ge=100, le=10000)` 编译出
        minimum/maximum——本用例在该版本上必红，改为 model_validator 后转绿。
        """
        schema = model_cls.model_json_schema()
        forbidden = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                     "multipleOf", "minLength", "maxLength")
        for node in _walk_schema_nodes(schema):
            for key in forbidden:
                assert key not in node, (
                    f"{model_cls.__name__} schema 含 strict 不支持的约束 {key}: {node}"
                )


# ---------------------------------------------------------------------------
# 2. to_action_spec() 字段映射
# ---------------------------------------------------------------------------


class TestClickToActionSpec:
    def test_coordinate_maps_to_tuple(self) -> None:
        model = ClickActionInput(
            reasoning="点击确认按钮",
            risk_level=ActionRisk.LOW_RISK,
            coordinate=GroundingCoordinate(x=120, y=340),
        )
        spec = model.to_action_spec("act-1")

        assert spec.action_id == "act-1"
        assert spec.action_type == "click"
        assert spec.coordinates == (120, 340)
        assert spec.target_element_id is None
        assert spec.text_payload is None
        assert spec.risk_level == ActionRisk.LOW_RISK

    def test_target_element_id_passthrough(self) -> None:
        model = ClickActionInput(
            reasoning="点击已知元素",
            risk_level=ActionRisk.LOW_RISK,
            target_element_id="uia:btn-ok",
        )
        spec = model.to_action_spec("act-2")

        assert spec.target_element_id == "uia:btn-ok"
        assert spec.coordinates is None


class TestTypeToActionSpec:
    def test_text_maps_to_text_payload(self) -> None:
        model = TypeActionInput(
            reasoning="输入搜索词",
            risk_level=ActionRisk.LOW_RISK,
            text="hello world",
            target_element_id="uia:search-box",
        )
        spec = model.to_action_spec("act-3")

        assert spec.action_type == "type"
        assert spec.text_payload == "hello world"
        assert spec.target_element_id == "uia:search-box"
        assert spec.risk_level == ActionRisk.LOW_RISK

    def test_coordinate_maps_to_tuple(self) -> None:
        model = TypeActionInput(
            reasoning="输入到坐标定位的输入框",
            risk_level=ActionRisk.LOW_RISK,
            text="abc",
            coordinate=GroundingCoordinate(x=10, y=20),
        )
        spec = model.to_action_spec("act-4")

        assert spec.coordinates == (10, 20)


class TestKeyToActionSpec:
    def test_key_maps_to_text_payload(self) -> None:
        model = KeyActionInput(
            reasoning="回车提交",
            risk_level=ActionRisk.LOW_RISK,
            key="enter",
        )
        spec = model.to_action_spec("act-5")

        assert spec.action_type == "key"
        assert spec.text_payload == "enter"
        assert spec.target_element_id is None
        assert spec.coordinates is None
        assert spec.risk_level == ActionRisk.LOW_RISK


class TestWindowCloseToActionSpec:
    def test_window_handle_maps_to_target_element_id_str(self) -> None:
        model = WindowCloseActionInput(
            reasoning="关闭已完成任务的窗口",
            risk_level=ActionRisk.DESTRUCTIVE,
            window_handle=0x0005_1C42,
        )
        spec = model.to_action_spec("act-6")

        assert spec.action_type == "window_close"
        assert spec.target_element_id == str(0x0005_1C42)
        assert spec.coordinates is None
        assert spec.text_payload is None
        assert spec.risk_level == ActionRisk.DESTRUCTIVE


class TestWaitToActionSpec:
    def test_wait_ms_passthrough(self) -> None:
        model = WaitActionInput(
            reasoning="等待页面加载",
            risk_level=ActionRisk.LOW_RISK,
            wait_ms=500,
        )
        spec = model.to_action_spec("act-7")

        assert spec.action_type == "wait"
        assert spec.wait_ms == 500
        assert spec.target_element_id is None
        assert spec.coordinates is None
        assert spec.text_payload is None

    def test_risk_level_forced_read_only_ignoring_declared(self) -> None:
        """PR #28 WARN② 收口：wait 无落点无副作用，声明值被忽略、强制 READ_ONLY
        ——否则 LOW_RISK 恒触发 TOCTOU，在等待过渡态场景被「界面已变」误杀。"""
        model = WaitActionInput(
            reasoning="LLM 高报风险",
            risk_level=ActionRisk.DESTRUCTIVE,
            wait_ms=500,
        )
        spec = model.to_action_spec("act-7b")
        assert spec.risk_level == ActionRisk.READ_ONLY


# ---------------------------------------------------------------------------
# 3. Click/Type「target 与 coordinate 至少其一」校验
# ---------------------------------------------------------------------------


class TestClickTypeRequireLocator:
    def test_click_rejects_neither_target_nor_coordinate(self) -> None:
        with pytest.raises(ValidationError):
            ClickActionInput(reasoning="无定位", risk_level=ActionRisk.LOW_RISK)

    def test_type_rejects_neither_target_nor_coordinate(self) -> None:
        with pytest.raises(ValidationError):
            TypeActionInput(reasoning="无定位", risk_level=ActionRisk.LOW_RISK, text="x")


# ---------------------------------------------------------------------------
# 4. WaitActionInput.wait_ms 边界 [100, 10000]
# ---------------------------------------------------------------------------


class TestWaitMsBounds:
    def test_below_minimum_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WaitActionInput(reasoning="太短", risk_level=ActionRisk.LOW_RISK, wait_ms=99)

    def test_minimum_accepted(self) -> None:
        model = WaitActionInput(reasoning="下限", risk_level=ActionRisk.LOW_RISK, wait_ms=100)
        assert model.wait_ms == 100

    def test_maximum_accepted(self) -> None:
        model = WaitActionInput(reasoning="上限", risk_level=ActionRisk.LOW_RISK, wait_ms=10000)
        assert model.wait_ms == 10000

    def test_above_maximum_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WaitActionInput(reasoning="太长", risk_level=ActionRisk.LOW_RISK, wait_ms=10001)


# ---------------------------------------------------------------------------
# 5. wait 白名单（task 2）
# ---------------------------------------------------------------------------


class TestWaitWhitelist:
    def test_wait_is_read_only_in_whitelist(self) -> None:
        """READ_ONLY（PR #28 WARN②）：配合适配器强制 READ_ONLY，使
        toctou_verify 的 needs_toctou 对 wait 自然为 False。"""
        assert _ACTION_RISK_WHITELIST["wait"] == ActionRisk.READ_ONLY


# ---------------------------------------------------------------------------
# 6. ActionSpec.wait_ms 加字段零回归
# ---------------------------------------------------------------------------


class TestActionSpecWaitMsFieldAddition:
    def test_legacy_construction_without_wait_ms_still_valid(self) -> None:
        spec = ActionSpec(
            action_id="act-legacy",
            action_type="click",
            target_element_id=None,
            coordinates=(1, 2),
            text_payload=None,
            risk_level=ActionRisk.LOW_RISK,
        )
        assert spec.wait_ms is None
