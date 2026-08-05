"""screen_snapshot 契约模型单测（feat/desktop-hardening 契约增量）。

覆盖：
  1. 零回归：旧 payload（不含新字段）反序列化成功，新字段全取默认值
     （desktop_locked=False / window_captured=False / degradations=[] /
     expected_root_hwnd=None）。
  2. round-trip：新字段显式赋值后 model_dump → model_validate 逐值保真
     （字段被删/被静默丢弃时本组红）。
  3. degradations 实例间不共享列表（行为钉死，防脱离 pydantic 的重构引入
     可变默认值别名）。
  4. 降级枚举 docstring 钉子：类 docstring（契约唯一真相）必须列全 8 个
     已定机读码——改码不改文档、或改文档漏码，本组红。

判别力实证（绿灯先证能红，见本次交付报告）：
  - desktop_locked 默认改 True → 零回归组红；
  - degradations 去默认（改必填）→ 零回归组红；
  - 删除 expected_root_hwnd 字段 → 默认值与 round-trip 组红；
  - docstring 改码名（ocr_crop_invalid → ocr_bbox_invalid）→ 枚举钉子红。
"""

from __future__ import annotations

from typing import Any

from src.agents.models.screen_snapshot import ActionRisk, ActionSpec, ScreenSnapshot

# ---------------------------------------------------------------------------
# 辅助构造函数（**旧 payload 形状**：不含 feat/desktop-hardening 新字段）
# ---------------------------------------------------------------------------


def _make_legacy_snapshot_payload(**overrides: Any) -> dict[str, Any]:
    """构造加固前形状的 ScreenSnapshot dict（无新字段，验零回归用）。"""
    payload: dict[str, Any] = {
        "snapshot_id": "snap-001",
        "timestamp_ms": 1_722_400_000_000,
        "screen_width": 2560,
        "screen_height": 1440,
        "active_window_title": "记事本",
        "uia_elements": [],
        "text_blocks": [],
        "visual_objects": [],
        "screenshot_path": None,
        "perception_mode": "uia_ocr",
        "capability_flags": {"ocr": True},
    }
    payload.update(overrides)
    return payload


def _make_legacy_action_payload(**overrides: Any) -> dict[str, Any]:
    """构造加固前形状的 ActionSpec dict（无 expected_root_hwnd，验零回归用）。"""
    payload: dict[str, Any] = {
        "action_id": "act-001",
        "action_type": "click",
        "target_element_id": None,
        "coordinates": (120, 340),
        "text_payload": None,
        "risk_level": ActionRisk.LOW_RISK,
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# 1. ScreenSnapshot 加固字段：零回归 + round-trip
# ---------------------------------------------------------------------------


class TestScreenSnapshotHardeningFields:
    """desktop_locked / window_captured / degradations 三字段的默认值与保真。"""

    def test_legacy_payload_parses_with_all_defaults(self) -> None:
        """零回归：旧 payload 解析成功，三个新字段全取默认值。

        判别力：任一新字段默认值被改（如 desktop_locked 默认 True）或被改成
        必填（degradations 去 default_factory），本例即红。
        """
        snap = ScreenSnapshot(**_make_legacy_snapshot_payload())

        assert snap.desktop_locked is False
        assert snap.window_captured is False
        assert snap.degradations == []

    def test_new_fields_roundtrip(self) -> None:
        """显式赋值 → model_dump → model_validate 逐值保真。

        判别力：字段被删后，构造期该键被默认 extra="ignore" 静默吞掉、
        属性读取 AttributeError，本例红——这是「字段还在契约里」的存在性证明。
        """
        snap = ScreenSnapshot(
            **_make_legacy_snapshot_payload(),
            desktop_locked=True,
            window_captured=True,
            degradations=["desktop_locked", "ocr_error"],
        )
        assert snap.desktop_locked is True
        assert snap.window_captured is True

        restored = ScreenSnapshot.model_validate(snap.model_dump())

        assert restored.desktop_locked is True
        assert restored.window_captured is True
        assert restored.degradations == ["desktop_locked", "ocr_error"]

    def test_degradations_not_shared_between_instances(self) -> None:
        """两实例的 degradations 列表互不别名（就地 append 不串味）。

        诚实标注：pydantic v2 对字面量可变默认也会逐实例深拷贝，故本例
        **不能**证明 default_factory 相对 default=[] 的必要性；它钉住的是
        消费方可依赖的行为本身，防日后脱离 pydantic 的重构（如改 dataclass
        并写共享默认列表）引入别名。
        """
        first = ScreenSnapshot(**_make_legacy_snapshot_payload())
        second = ScreenSnapshot(**_make_legacy_snapshot_payload(snapshot_id="snap-002"))

        first.degradations.append("ocr_error")

        assert second.degradations == []


# ---------------------------------------------------------------------------
# 2. 降级枚举清单：类 docstring 是契约唯一真相，码集钉死
# ---------------------------------------------------------------------------

_DEGRADATION_CODES: frozenset[str] = frozenset(
    {
        "desktop_locked",
        "lock_probe_failed",
        "window_capture_failed",
        "screenshot_failed",
        "mss_unavailable",
        "ocr_error",
        "ocr_unavailable",
        "ocr_crop_invalid",
    }
)
"""degradations 的已定机读码全集（与 ScreenSnapshot 类 docstring 清单一一对应）。

消费方按集合成员判断降级，码名即契约——感知侧改码必须同步 docstring 与此处，
否则下方钉子红。
"""


class TestDegradationEnumDocstringPin:
    """降级码集与类 docstring（契约唯一真相）同步的钉子。"""

    def test_docstring_lists_every_code(self) -> None:
        """8 个已定码逐一出现在 ScreenSnapshot 类 docstring 中。

        判别力：docstring 里改/删任一码名（如 ocr_crop_invalid → ocr_bbox_invalid）
        即红；本测试常量与 docstring 双侧都要动才能绿，防单侧漂移。
        """
        doc = ScreenSnapshot.__doc__
        assert doc is not None
        missing = {code for code in _DEGRADATION_CODES if code not in doc}
        assert not missing, (
            f"ScreenSnapshot 类 docstring 缺降级码 {sorted(missing)}——"
            "契约唯一真相在 docstring，改码需两侧同步"
        )

    def test_code_set_has_expected_cardinality(self) -> None:
        """码集恰 8 个——增删降级码时强制回到这里登记（防只改 docstring 不改钉子）。"""
        assert len(_DEGRADATION_CODES) == 8


# ---------------------------------------------------------------------------
# 3. ActionSpec.expected_root_hwnd：零回归 + 保真
# ---------------------------------------------------------------------------


class TestActionSpecExpectedRootHwnd:
    """坐标点击落点核验期望句柄：None=不核验（零回归），int 保真。"""

    def test_legacy_payload_defaults_none(self) -> None:
        """零回归：旧 payload（无该键）解析成功且默认 None（不核验）。

        判别力：默认值被改（如 0）或字段改必填，本例即红。
        """
        spec = ActionSpec(**_make_legacy_action_payload())
        assert spec.expected_root_hwnd is None

    def test_int_value_roundtrip(self) -> None:
        """显式 hwnd → dump → validate 保真（字段被删则静默吞键、读取红）。"""
        spec = ActionSpec(**_make_legacy_action_payload(), expected_root_hwnd=0x0005_1C42)
        assert spec.expected_root_hwnd == 0x0005_1C42

        restored = ActionSpec.model_validate(spec.model_dump())

        assert restored.expected_root_hwnd == 0x0005_1C42

    def test_explicit_none_accepted(self) -> None:
        """显式传 None 与缺省同义（菜单点击方按 docstring 约定不设期望值）。"""
        spec = ActionSpec(**_make_legacy_action_payload(), expected_root_hwnd=None)
        assert spec.expected_root_hwnd is None
