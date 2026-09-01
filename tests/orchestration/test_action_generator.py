"""ActionGeneratorAgent 单测（ActionSpec 生成层蓝图 PR-β 任务 8/10）。

覆盖：
  1. 缺 key / llm_client None / 无 snapshot_ref / snapshot_store None / 快照加载
     失败 → 不调 LLM（除快照加载失败外），失败增量（control_error 含机读令牌 +
     pending_action=None + step_history 追加一步）。
  2. 5 类成功路径（mock tool_use 响应）→ pending_action 为对应 ActionSpec。
  3. target_element_id 不在本次可用元素表 → 失败令牌。
  4. 服务端坐标覆写：LLM 给了 id + 错误 coordinate 时以 bbox 中心为准。
  5. 未知工具名 / 无 tool_use 内容块 / 子模型校验失败 → 失败令牌。
  6. 主备单次切换透传（LLMFallbackError → 失败增量，不崩溃）。
  7. 5 个工具定义形状：strict:true 且 input_schema 无 minimum/maximum
     （复用 PR-α 的扫描器思路）。
  8. make_generate_action_node 节点签名 (state) -> dict。
  9. target_bbox 透传（TOCTOU 方案③）：target_element_id 命中时 ActionSpec.
     target_bbox 等于查表 bbox；LLM 只给 coordinate 兜底通道时 target_bbox
     保持 None（不信自报边界）。
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.models.screen_snapshot import (
    ActionRisk,
    BBox,
    ScreenSnapshot,
    UIAElement,
)
from src.orchestration.action_generator import (
    ACTION_GENERATION_FAILED_TOKEN,
    ActionGeneratorAgent,
    _build_tools,
    make_generate_action_node,
)
from src.orchestration.prompt_loader import PromptLoader
from src.orchestration.state import DesktopTaskState

# ── 测试辅助 ──────────────────────────────────────────────────────────────────


def _make_snapshot(uia_elements: list[UIAElement] | None = None) -> ScreenSnapshot:
    return ScreenSnapshot(
        snapshot_id="snap-gen-001",
        timestamp_ms=1_700_000_000_000,
        screen_width=1920,
        screen_height=1080,
        active_window_title="生成层测试窗口",
        uia_elements=uia_elements or [],
        text_blocks=[],
        visual_objects=[],
        screenshot_path=None,
        perception_mode="uia_ocr",
        capability_flags={},
        uia_hollow=False,
    )


def _make_uia_element(bbox: BBox | None = None) -> UIAElement:
    return UIAElement(
        element_id="uia_1234_0",
        control_type="Button",
        name="确定",
        automation_id="btnOK",
        bbox=bbox or BBox(x=100, y=200, width=80, height=40),
        is_enabled=True,
        is_visible=True,
        value=None,
        source="uia",
    )


class _FakeSnapshotStore:
    """最小 SnapshotStore 打桩：固定返回一份快照，或按需抛异常。"""

    def __init__(self, snapshot: ScreenSnapshot | None = None, raise_on_load: bool = False) -> None:
        self.snapshot = snapshot or _make_snapshot()
        self.raise_on_load = raise_on_load
        self.load_calls: list[str] = []

    async def save(self, snapshot: ScreenSnapshot) -> str:
        return snapshot.snapshot_id

    async def load(self, snapshot_id: str) -> ScreenSnapshot:
        self.load_calls.append(snapshot_id)
        if self.raise_on_load:
            raise RuntimeError("快照加载失败（模拟）")
        return self.snapshot


def _make_state(
    snapshot_ref: str | None = "snap-gen-001",
    current_instruction: str = "点击确定按钮",
) -> DesktopTaskState:
    return DesktopTaskState(
        task_id="task-gen-001",
        task_description="生成层测试任务",
        current_instruction=current_instruction,
        snapshot_ref=snapshot_ref,
    )


def _make_tool_use_block(name: str, input_: dict[str, Any]) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = input_
    return block


def _make_tool_use_response(name: str, input_: dict[str, Any]) -> MagicMock:
    response = MagicMock()
    response.content = [_make_tool_use_block(name, input_)]
    return response


def _make_mock_llm(response: MagicMock) -> MagicMock:
    llm = MagicMock()
    llm.messages = MagicMock()
    llm.messages.create = AsyncMock(return_value=response)
    return llm


def _make_key_tool_llm() -> MagicMock:
    """构造一个总是返回合法 key 工具调用的 mock LLM（前置失败路径用，工具内容
    本身不重要——测的是"根本不该调到这里"）。"""
    return _make_mock_llm(
        _make_tool_use_response("key", {"reasoning": "x", "risk_level": "low_risk", "key": "enter"})
    )


def _make_agent(
    llm_client: Any,
    snapshot_store: Any = None,
) -> ActionGeneratorAgent:
    return ActionGeneratorAgent(
        llm_client=llm_client,
        prompt_loader=PromptLoader(),
        snapshot_store=snapshot_store if snapshot_store is not None else _FakeSnapshotStore(),
    )


# ── 1. 前置失败路径（不调 LLM 或调用前失败）──────────────────────────────────


async def test_missing_api_key_fails_without_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    llm = _make_key_tool_llm()
    agent = _make_agent(llm)
    state = _make_state()

    result = await agent.generate(state)

    assert result["pending_action"] is None
    assert ACTION_GENERATION_FAILED_TOKEN in result["control_error"]
    llm.messages.create.assert_not_called()
    assert len(result["step_history"]) == 1
    assert result["step_history"][0].agent == "generate_action"


async def test_none_llm_client_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    agent = _make_agent(llm_client=None)
    state = _make_state()

    result = await agent.generate(state)

    assert result["pending_action"] is None
    assert ACTION_GENERATION_FAILED_TOKEN in result["control_error"]


async def test_missing_snapshot_ref_fails_without_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    llm = _make_key_tool_llm()
    agent = _make_agent(llm)
    state = _make_state(snapshot_ref=None)

    result = await agent.generate(state)

    assert result["pending_action"] is None
    assert ACTION_GENERATION_FAILED_TOKEN in result["control_error"]
    llm.messages.create.assert_not_called()


async def test_missing_snapshot_store_fails_without_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    llm = _make_key_tool_llm()
    agent = ActionGeneratorAgent(llm_client=llm, prompt_loader=PromptLoader(), snapshot_store=None)
    state = _make_state()

    result = await agent.generate(state)

    assert result["pending_action"] is None
    assert ACTION_GENERATION_FAILED_TOKEN in result["control_error"]
    llm.messages.create.assert_not_called()


async def test_snapshot_load_failure_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    llm = _make_key_tool_llm()
    store = _FakeSnapshotStore(raise_on_load=True)
    agent = _make_agent(llm, snapshot_store=store)
    state = _make_state()

    result = await agent.generate(state)

    assert result["pending_action"] is None
    assert ACTION_GENERATION_FAILED_TOKEN in result["control_error"]
    llm.messages.create.assert_not_called()


# ── 2. 5 类成功路径 ───────────────────────────────────────────────────────────


async def test_generate_key_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    llm = _make_mock_llm(
        _make_tool_use_response(
            "key", {"reasoning": "回车提交", "risk_level": "low_risk", "key": "enter"}
        )
    )
    agent = _make_agent(llm)
    state = _make_state(current_instruction="按回车")

    result = await agent.generate(state)

    assert result.get("control_error") is None
    spec = result["pending_action"]
    assert spec is not None
    assert spec.action_type == "key"
    assert spec.text_payload == "enter"
    assert len(result["step_history"]) == 1
    assert result["step_history"][0].agent == "generate_action"


async def test_generate_window_close_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    llm = _make_mock_llm(
        _make_tool_use_response(
            "window_close",
            {"reasoning": "关闭窗口", "risk_level": "destructive", "window_handle": 12345},
        )
    )
    agent = _make_agent(llm)
    state = _make_state(current_instruction="关闭窗口")

    result = await agent.generate(state)

    spec = result["pending_action"]
    assert spec.action_type == "window_close"
    assert spec.target_element_id == "12345"


async def test_generate_wait_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    llm = _make_mock_llm(
        _make_tool_use_response(
            "wait",
            {"reasoning": "等待加载", "risk_level": "low_risk", "wait_ms": 500},
        )
    )
    agent = _make_agent(llm)
    state = _make_state(current_instruction="等待页面加载")

    result = await agent.generate(state)

    spec = result["pending_action"]
    assert spec.action_type == "wait"
    assert spec.wait_ms == 500
    assert spec.risk_level == ActionRisk.READ_ONLY, "适配器强制 READ_ONLY，忽略声明值"


async def test_generate_click_with_target_element_id_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    llm = _make_mock_llm(
        _make_tool_use_response(
            "click",
            {
                "reasoning": "点击确定",
                "risk_level": "low_risk",
                "target_element_id": "uia:0",
            },
        )
    )
    store = _FakeSnapshotStore(_make_snapshot(uia_elements=[_make_uia_element()]))
    agent = _make_agent(llm, snapshot_store=store)
    state = _make_state(current_instruction="点击确定按钮")

    result = await agent.generate(state)

    spec = result["pending_action"]
    assert spec.action_type == "click"
    # bbox=(100,200,80,40) → 中心 (140, 220)
    assert spec.coordinates == (140, 220)
    # TOCTOU 方案③：target_element_id 命中时同步注入 target_bbox
    assert spec.target_bbox == BBox(x=100, y=200, width=80, height=40)


async def test_generate_type_with_coordinate_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    llm = _make_mock_llm(
        _make_tool_use_response(
            "type",
            {
                "reasoning": "在坐标处输入",
                "risk_level": "low_risk",
                "text": "hello",
                "coordinate": {"x": 5, "y": 6},
            },
        )
    )
    agent = _make_agent(llm)
    state = _make_state(current_instruction="输入 hello")

    result = await agent.generate(state)

    spec = result["pending_action"]
    assert spec.action_type == "type"
    assert spec.text_payload == "hello"
    assert spec.coordinates == (5, 6)
    # TOCTOU 方案③：LLM 自报 coordinate 的兜底通道不信自报边界，target_bbox 保持 None
    assert spec.target_bbox is None


# ── 3. id 不存在 → 失败令牌 ────────────────────────────────────────────────────


async def test_unknown_target_element_id_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    llm = _make_mock_llm(
        _make_tool_use_response(
            "click",
            {
                "reasoning": "点击一个不存在的元素",
                "risk_level": "low_risk",
                "target_element_id": "uia:999",
            },
        )
    )
    store = _FakeSnapshotStore(_make_snapshot(uia_elements=[_make_uia_element()]))
    agent = _make_agent(llm, snapshot_store=store)
    state = _make_state()

    result = await agent.generate(state)

    assert result["pending_action"] is None
    assert ACTION_GENERATION_FAILED_TOKEN in result["control_error"]
    assert "uia:999" in result["control_error"]


# ── 4. 服务端坐标覆写：LLM 给了 id + 错误 coordinate 时以 bbox 中心为准 ─────────


async def test_server_side_coordinate_override_wins_over_llm_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    llm = _make_mock_llm(
        _make_tool_use_response(
            "click",
            {
                "reasoning": "点击确定",
                "risk_level": "low_risk",
                "target_element_id": "uia:0",
                # LLM 自报的坐标（像素幻觉）——应被服务端 bbox 中心覆写
                "coordinate": {"x": 9999, "y": 9999},
            },
        )
    )
    store = _FakeSnapshotStore(_make_snapshot(uia_elements=[_make_uia_element()]))
    agent = _make_agent(llm, snapshot_store=store)
    state = _make_state()

    result = await agent.generate(state)

    spec = result["pending_action"]
    assert spec.coordinates == (140, 220), "服务端应以 bbox 中心覆写 LLM 自报坐标"


# ── 5. 未知工具名 / 无 tool_use / 校验失败 ──────────────────────────────────────


async def test_unknown_tool_name_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    llm = _make_mock_llm(_make_tool_use_response("scroll", {"reasoning": "x"}))
    agent = _make_agent(llm)
    state = _make_state()

    result = await agent.generate(state)

    assert result["pending_action"] is None
    assert ACTION_GENERATION_FAILED_TOKEN in result["control_error"]


async def test_no_tool_use_block_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    response = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    response.content = [text_block]
    llm = _make_mock_llm(response)
    agent = _make_agent(llm)
    state = _make_state()

    result = await agent.generate(state)

    assert result["pending_action"] is None
    assert ACTION_GENERATION_FAILED_TOKEN in result["control_error"]


async def test_tool_input_validation_failure_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    # click 缺 target_element_id 与 coordinate 两者（model_validator 应拒绝）
    llm = _make_mock_llm(
        _make_tool_use_response("click", {"reasoning": "无定位", "risk_level": "low_risk"})
    )
    agent = _make_agent(llm)
    state = _make_state()

    result = await agent.generate(state)

    assert result["pending_action"] is None
    assert ACTION_GENERATION_FAILED_TOKEN in result["control_error"]


# ── 6. 主备单次切换透传 ─────────────────────────────────────────────────────────


async def test_llm_fallback_error_produces_failure_increment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    llm = MagicMock()
    llm.messages = MagicMock()
    llm.messages.create = AsyncMock(side_effect=RuntimeError("network boom"))
    agent = _make_agent(llm)
    agent.fallback_model = None
    state = _make_state()

    result = await agent.generate(state)

    assert result["pending_action"] is None
    assert ACTION_GENERATION_FAILED_TOKEN in result["control_error"]
    llm.messages.create.assert_called_once()


async def test_fallback_model_used_on_primary_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    ok_response = _make_tool_use_response(
        "key", {"reasoning": "回车", "risk_level": "low_risk", "key": "enter"}
    )
    llm = MagicMock()
    llm.messages = MagicMock()
    llm.messages.create = AsyncMock(side_effect=[RuntimeError("primary boom"), ok_response])
    agent = _make_agent(llm)
    agent.fallback_model = "backup-model"
    state = _make_state()

    result = await agent.generate(state)

    assert result["pending_action"] is not None
    assert llm.messages.create.call_count == 2
    models = [c.kwargs["model"] for c in llm.messages.create.call_args_list]
    assert models == [agent.model, "backup-model"]


# ── 7. 工具定义形状：strict:true 且 input_schema 无 minimum/maximum ─────────────


def _walk_schema_nodes(node: Any) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    if isinstance(node, dict):
        nodes.append(node)
        for value in node.values():
            nodes.extend(_walk_schema_nodes(value))
    elif isinstance(node, list):
        for item in node:
            nodes.extend(_walk_schema_nodes(item))
    return nodes


class TestToolDefinitionShape:
    """5 个工具定义的 strict 兼容自查（复用 PR-α 扫描器思路）。"""

    def test_all_tools_have_strict_true(self) -> None:
        tools = _build_tools()
        assert len(tools) == 5
        for tool in tools:
            assert tool["strict"] is True
            assert "name" in tool
            assert "input_schema" in tool

    def test_no_forbidden_constraints_in_any_tool_schema(self) -> None:
        forbidden = (
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "minLength",
            "maxLength",
            "prefixItems",
        )
        for tool in _build_tools():
            for node in _walk_schema_nodes(tool["input_schema"]):
                for key in forbidden:
                    assert key not in node, f"{tool['name']} schema 含 {key}: {node}"


# ── 8. make_generate_action_node 节点签名 ───────────────────────────────────────


async def test_make_generate_action_node_returns_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    llm = _make_mock_llm(
        _make_tool_use_response(
            "key", {"reasoning": "回车", "risk_level": "low_risk", "key": "enter"}
        )
    )
    agent = _make_agent(llm)
    node_fn = make_generate_action_node(agent)
    state = _make_state()

    result = await node_fn(state)

    assert isinstance(result, dict)
    assert "pending_action" in result


# ── PR #29 审查 BLOCK 回归：grounding 表与渲染同一次构建 ─────────────────────


class TestGroundingTableBudgetCoupling:
    """审查实证的坐标级安全场景：注入小预算 loader（prompt 只展示部分元素），
    LLM 引用未展示的 id——修复前（消费侧按模块常量大预算重建表）会被放行并
    产出真实坐标；修复后（表随渲染同一次构建返回）必须拒绝并带失败令牌。"""

    async def test_llm_reference_to_unrendered_id_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        # 30 个元素 + 200 字符预算：渲染表只装得下前两三个，uia:10 不在其中
        elements = [_make_uia_element() for _ in range(30)]
        snapshot = _make_snapshot(uia_elements=elements)
        llm = _make_mock_llm(
            _make_tool_use_response(
                "click",
                {
                    "reasoning": "点一个没见过的元素",
                    "risk_level": "low_risk",
                    "target_element_id": "uia:10",
                },
            )
        )
        agent = ActionGeneratorAgent(
            llm_client=llm,
            prompt_loader=PromptLoader(summary_max_chars=200),
            snapshot_store=_FakeSnapshotStore(snapshot=snapshot),
        )

        result = await agent.generate(_make_state())

        assert result["pending_action"] is None, (
            "LLM 引用 prompt 未展示的 id 必须被拒绝——放行即坐标级安全缺口"
        )
        assert ACTION_GENERATION_FAILED_TOKEN in (result["control_error"] or "")

    async def test_rendered_id_still_resolves_under_small_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """正对照：同一小预算下，引用表内 id（uia:0）仍正常解析出 bbox 中心。"""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        elements = [_make_uia_element() for _ in range(30)]
        snapshot = _make_snapshot(uia_elements=elements)
        llm = _make_mock_llm(
            _make_tool_use_response(
                "click",
                {
                    "reasoning": "点第一个元素",
                    "risk_level": "low_risk",
                    "target_element_id": "uia:0",
                },
            )
        )
        agent = ActionGeneratorAgent(
            llm_client=llm,
            prompt_loader=PromptLoader(summary_max_chars=200),
            snapshot_store=_FakeSnapshotStore(snapshot=snapshot),
        )

        result = await agent.generate(_make_state())

        spec = result["pending_action"]
        assert spec is not None
        assert spec.coordinates == (140, 220), "bbox(100,200,80,40) 中心应为 (140,220)"
