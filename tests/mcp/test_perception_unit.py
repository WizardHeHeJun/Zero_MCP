"""test_perception_unit.py — perception.py 单元测试（mock 不走真实 UIA/mss/OCR）。

覆盖：
- hollow 探测：0 子元素 → hollow=True
- hollow 探测：3 子元素（等于阈值）→ hollow=True
- hollow 探测：10 子元素 + 含 ButtonControl → hollow=False
- hollow 且 mode=uia_only → 自动升 uia_ocr（do_screen_snapshot 逻辑）
- OCR 结果 → TextBlock 映射（block_id / text / confidence / bbox / source）
- do_ocr_region：caps.ocr=False → raise RuntimeError
- do_ocr_region：文件不存在 → raise FileNotFoundError

实机项（realenv marker）：不在本文件内，仅注释标注。
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import src.mcp.desktop.tools.perception as perception_mod
from src.agents.models.screen_snapshot import ScreenSnapshot, TextBlock
from src.mcp.desktop.capability_probe import CapabilityFlags

# ── fixture helpers ───────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_ocr_engine_singleton() -> Any:
    """RapidOCR 进程级单例在测试间重置，避免 fake engine 跨用例泄漏。"""
    perception_mod._OCR_ENGINE = None
    yield
    perception_mod._OCR_ENGINE = None


def _make_caps(
    ocr: bool = True,
    mss_available: bool = True,
    omniparser: bool = False,
) -> CapabilityFlags:
    return CapabilityFlags(
        ocr=ocr,
        omniparser=omniparser,
        cuda_accel=False,
        dml_accel=False,
        mss_available=mss_available,
        effective_device="cpu",
    )


def _make_uia_ctrl(control_type: str = "PaneControl", name: str = "") -> MagicMock:
    """构造一个 fake uiautomation 控件 mock。"""
    ctrl = MagicMock()
    ctrl.ControlTypeName = control_type
    ctrl.Name = name
    ctrl.AutomationId = ""
    r = MagicMock()
    r.left, r.top, r.right, r.bottom = 0, 0, 100, 50
    ctrl.BoundingRectangle = r
    ctrl.IsEnabled = True
    ctrl.IsOffscreen = False
    ctrl.GetChildren.return_value = []
    return ctrl


# ── UIA hollow 探测（直接测同步函数） ────────────────────────────────────────


def _run_hollow_with_children(children: list[Any], threshold: int = 3) -> bool:
    """patch uiautomation + env 后跑 _probe_window_uia_hollow_sync。"""
    fake_root = MagicMock()
    fake_root.GetChildren.return_value = children

    fake_auto = MagicMock()
    fake_auto.ControlFromHandle.return_value = fake_root

    with patch.dict("sys.modules", {"uiautomation": fake_auto}):
        with patch.dict("os.environ", {"UIA_HOLLOW_THRESHOLD": str(threshold)}):
            return perception_mod._probe_window_uia_hollow_sync(hwnd=0x1234)


def test_hollow_zero_children() -> None:
    """0 子元素 → hollow=True（子元素数 0 ≤ 阈值 3）。"""
    assert _run_hollow_with_children([]) is True


def test_hollow_three_children_equal_threshold() -> None:
    """3 子元素（等于默认阈值 3）→ hollow=True（≤ 阈值判定）。"""
    children = [_make_uia_ctrl("PaneControl") for _ in range(3)]
    assert _run_hollow_with_children(children, threshold=3) is True


def test_not_hollow_ten_children_with_button() -> None:
    """10 子元素且含 ButtonControl → hollow=False。"""
    children = [_make_uia_ctrl("PaneControl") for _ in range(9)]
    children.append(_make_uia_ctrl("ButtonControl", name="OK"))
    # 让 ButtonControl 的 GetChildren 也返回空（孙子层）
    for c in children:
        c.GetChildren.return_value = []
    result = _run_hollow_with_children(children, threshold=3)
    assert result is False


def test_hollow_children_all_non_interactive() -> None:
    """10 子元素但全为非可交互 PaneControl → hollow=True（可交互数=0）。"""
    children = [_make_uia_ctrl("PaneControl") for _ in range(10)]
    for c in children:
        c.GetChildren.return_value = []
    result = _run_hollow_with_children(children, threshold=3)
    assert result is True


def test_hollow_get_children_raises() -> None:
    """GetChildren 抛异常时，安全判 hollow=True。"""
    fake_root = MagicMock()
    fake_root.GetChildren.side_effect = RuntimeError("COM error")
    fake_auto = MagicMock()
    fake_auto.ControlFromHandle.return_value = fake_root
    with patch.dict("sys.modules", {"uiautomation": fake_auto}):
        result = perception_mod._probe_window_uia_hollow_sync(hwnd=0x5678)
    assert result is True


# ── hollow + uia_only → 自动升 uia_ocr ───────────────────────────────────────


async def test_hollow_uia_only_auto_upgrades_to_uia_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """hollow=True 且 mode=uia_only 时 do_screen_snapshot 内部升级为 uia_ocr。

    断言：返回 snapshot.perception_mode == "uia_ocr"（而非 "uia_only"）。
    """
    caps = _make_caps(ocr=False, mss_available=False)  # 关掉截图/OCR，只测模式升级

    # mock _get_active_window_info → 返回有效 hwnd
    monkeypatch.setattr(perception_mod, "_get_active_window_info", lambda: (0x9999, "TestWin"))
    monkeypatch.setattr(perception_mod, "_get_screen_size", lambda: (1920, 1080))

    # mock _probe_window_uia_hollow_sync → True（hollow）
    monkeypatch.setattr(
        perception_mod,
        "_probe_window_uia_hollow_sync",
        lambda hwnd: True,
    )
    # mock _collect_uia_tree_sync → 空列表
    monkeypatch.setattr(
        perception_mod,
        "_collect_uia_tree_sync",
        lambda hwnd, max_depth: [],
    )

    snapshot = await perception_mod.do_screen_snapshot(
        mode="uia_only",
        capture_screenshot=False,
        caps=caps,
        screenshot_tmp_dir="",
    )

    assert isinstance(snapshot, ScreenSnapshot)
    assert snapshot.uia_hollow is True
    assert snapshot.perception_mode == "uia_ocr", (
        f"期望 uia_ocr（hollow 自动升级），实际 {snapshot.perception_mode}"
    )


async def test_not_hollow_uia_only_stays(monkeypatch: pytest.MonkeyPatch) -> None:
    """hollow=False 且 mode=uia_only 时 perception_mode 保持 uia_only。"""
    caps = _make_caps(ocr=False, mss_available=False)

    monkeypatch.setattr(perception_mod, "_get_active_window_info", lambda: (0x1111, "TestWin"))
    monkeypatch.setattr(perception_mod, "_get_screen_size", lambda: (1920, 1080))
    monkeypatch.setattr(perception_mod, "_probe_window_uia_hollow_sync", lambda hwnd: False)
    monkeypatch.setattr(perception_mod, "_collect_uia_tree_sync", lambda hwnd, max_depth: [])

    snapshot = await perception_mod.do_screen_snapshot(
        mode="uia_only",
        capture_screenshot=False,
        caps=caps,
        screenshot_tmp_dir="",
    )

    assert snapshot.uia_hollow is False
    assert snapshot.perception_mode == "uia_only"


# ── OCR 裁剪口径：前台窗口 rect（与 L1 UIA 对齐，Task 12 实测修正） ────────────


async def test_screen_snapshot_crops_ocr_to_active_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """do_screen_snapshot 把前台窗口 rect（clamp 到主屏）作为 OCR 裁剪 bbox 传入。

    Task 12 e2e 实测：全图 OCR 会把其他应用文本混入 perception_summary
    （跨窗口注入面），故 L2 OCR 必须与 L1 UIA 同口径裁剪到前台窗口。
    """
    caps = _make_caps(ocr=True, mss_available=True)
    monkeypatch.setattr(perception_mod, "_get_active_window_info", lambda: (0x2222, "钉钉"))
    monkeypatch.setattr(perception_mod, "_get_screen_size", lambda: (1920, 1080))
    monkeypatch.setattr(perception_mod, "_probe_window_uia_hollow_sync", lambda hwnd: True)
    monkeypatch.setattr(perception_mod, "_collect_uia_tree_sync", lambda hwnd, max_depth: [])
    monkeypatch.setattr(
        perception_mod, "_take_screenshot_sync", lambda snap_id, tmp_dir: "C:/fake/shot.png"
    )
    # 窗口右/下缘越出主屏 → 期望 clamp 到 (100,50)-(1920,1080)
    monkeypatch.setattr(perception_mod, "_get_window_rect", lambda hwnd: (100, 50, 2000, 1200))

    seen_bbox: list[Any] = []

    def fake_ocr(
        screenshot_path: str, bbox: Any, snapshot_id: str, origin: tuple[int, int] = (0, 0)
    ) -> list[Any]:
        seen_bbox.append((bbox, origin))
        return []

    monkeypatch.setattr(perception_mod, "_run_ocr_on_file_sync", fake_ocr)

    snapshot = await perception_mod.do_screen_snapshot(
        mode="uia_ocr", capture_screenshot=False, caps=caps, screenshot_tmp_dir=""
    )

    assert snapshot.perception_mode == "uia_ocr"
    assert seen_bbox == [({"x": 100, "y": 50, "width": 1820, "height": 1030}, (0, 0))]


async def test_screen_snapshot_ocr_falls_back_fullscreen_when_window_offscreen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """前台窗口完全在主屏外（如副屏）→ 裁剪无效，OCR 回退全图（bbox=None）。"""
    caps = _make_caps(ocr=True, mss_available=True)
    monkeypatch.setattr(perception_mod, "_get_active_window_info", lambda: (0x3333, "副屏窗"))
    monkeypatch.setattr(perception_mod, "_get_screen_size", lambda: (1920, 1080))
    monkeypatch.setattr(perception_mod, "_probe_window_uia_hollow_sync", lambda hwnd: False)
    monkeypatch.setattr(perception_mod, "_collect_uia_tree_sync", lambda hwnd, max_depth: [])
    monkeypatch.setattr(
        perception_mod, "_take_screenshot_sync", lambda snap_id, tmp_dir: "C:/fake/shot.png"
    )
    monkeypatch.setattr(perception_mod, "_get_window_rect", lambda hwnd: (2560, 0, 4480, 1080))

    seen_bbox: list[Any] = []

    def fake_ocr(
        screenshot_path: str, bbox: Any, snapshot_id: str, origin: tuple[int, int] = (0, 0)
    ) -> list[Any]:
        seen_bbox.append((bbox, origin))
        return []

    monkeypatch.setattr(perception_mod, "_run_ocr_on_file_sync", fake_ocr)

    await perception_mod.do_screen_snapshot(
        mode="uia_ocr", capture_screenshot=False, caps=caps, screenshot_tmp_dir=""
    )

    assert seen_bbox == [(None, (0, 0))]


# ── 指定 window_handle 感知（解除前台耦合，Task 12 实测需求） ─────────────────


async def test_screen_snapshot_targets_specified_window_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """window_handle 指定时，UIA 树 / hollow 探测 / OCR 裁剪均以该窗口为准，
    完全不读前台窗口（前台可被第三方窗口/自家子进程控制台抢占）。"""
    caps = _make_caps(ocr=True, mss_available=True)

    def fail_active() -> tuple[int | None, str | None]:
        raise AssertionError("指定 window_handle 时不得读取前台窗口")

    monkeypatch.setattr(perception_mod, "_get_active_window_info", fail_active)
    monkeypatch.setattr(perception_mod, "_get_window_title", lambda hwnd: "钉钉")
    monkeypatch.setattr(perception_mod, "_get_screen_size", lambda: (1920, 1080))

    seen_hollow_hwnd: list[int] = []
    seen_tree_hwnd: list[int] = []

    def fake_hollow(hwnd: int) -> bool:
        seen_hollow_hwnd.append(hwnd)
        return False

    def fake_tree(hwnd: int, max_depth: int) -> list[Any]:
        seen_tree_hwnd.append(hwnd)
        return []

    monkeypatch.setattr(perception_mod, "_probe_window_uia_hollow_sync", fake_hollow)
    monkeypatch.setattr(perception_mod, "_collect_uia_tree_sync", fake_tree)
    # PrintWindow 捕获失败（如窗口最小化）→ 回退 mss 全屏 + rect 裁剪
    monkeypatch.setattr(perception_mod, "_capture_window_sync", lambda hwnd, snap_id, tmp_dir: None)
    monkeypatch.setattr(
        perception_mod, "_take_screenshot_sync", lambda snap_id, tmp_dir: "C:/fake/shot.png"
    )
    monkeypatch.setattr(perception_mod, "_get_window_rect", lambda hwnd: (100, 100, 900, 700))

    seen_bbox: list[Any] = []

    def fake_ocr(
        screenshot_path: str, bbox: Any, snapshot_id: str, origin: tuple[int, int] = (0, 0)
    ) -> list[Any]:
        seen_bbox.append((bbox, origin))
        return []

    monkeypatch.setattr(perception_mod, "_run_ocr_on_file_sync", fake_ocr)

    snapshot = await perception_mod.do_screen_snapshot(
        mode="uia_ocr",
        capture_screenshot=False,
        caps=caps,
        screenshot_tmp_dir="",
        window_handle=0x4444,
    )

    assert seen_hollow_hwnd == [0x4444]
    assert seen_tree_hwnd == [0x4444]
    assert seen_bbox == [({"x": 100, "y": 100, "width": 800, "height": 600}, (0, 0))]
    assert snapshot.active_window_title == "钉钉"


async def test_screen_snapshot_window_handle_uses_print_window_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """window_handle 指定且 PrintWindow 捕获成功 → OCR 直接全图 + origin 补偿回
    屏幕绝对坐标（被遮挡/无焦点也能感知，Task 12 实测核心需求）。"""
    caps = _make_caps(ocr=True, mss_available=True)
    monkeypatch.setattr(perception_mod, "_get_window_title", lambda hwnd: "钉钉")
    monkeypatch.setattr(perception_mod, "_get_screen_size", lambda: (1920, 1080))
    monkeypatch.setattr(perception_mod, "_probe_window_uia_hollow_sync", lambda hwnd: True)
    monkeypatch.setattr(perception_mod, "_collect_uia_tree_sync", lambda hwnd, max_depth: [])
    monkeypatch.setattr(
        perception_mod, "_capture_window_sync", lambda hwnd, snap_id, tmp_dir: "C:/fake/win.png"
    )

    def fail_mss(snap_id: str, tmp_dir: str) -> str:
        raise AssertionError("PrintWindow 成功时不得走 mss 全屏截图")

    monkeypatch.setattr(perception_mod, "_take_screenshot_sync", fail_mss)
    monkeypatch.setattr(perception_mod, "_get_window_rect", lambda hwnd: (730, 304, 1754, 944))

    seen: list[Any] = []

    def fake_ocr(
        screenshot_path: str, bbox: Any, snapshot_id: str, origin: tuple[int, int] = (0, 0)
    ) -> list[Any]:
        seen.append((screenshot_path, bbox, origin))
        return []

    monkeypatch.setattr(perception_mod, "_run_ocr_on_file_sync", fake_ocr)

    snapshot = await perception_mod.do_screen_snapshot(
        mode="uia_ocr",
        capture_screenshot=False,
        caps=caps,
        screenshot_tmp_dir="",
        window_handle=0x5555,
    )

    # 窗口图全图 OCR（bbox=None），origin=窗口左上角 → 坐标恒为屏幕绝对
    assert seen == [("C:/fake/win.png", None, (730, 304))]
    assert snapshot.screenshot_path == "C:/fake/win.png"


# ── OCR TextBlock 映射 ────────────────────────────────────────────────────────


def test_run_ocr_on_file_sync_maps_to_text_blocks(tmp_path: Any) -> None:
    """_run_ocr_on_file_sync mock RapidOCR 返回结果 → TextBlock 字段正确映射。

    mock 策略：patch rapidocr_onnxruntime.RapidOCR 构造器（函数调用层 mock），
    保留 numpy 已加载状态（避免 native extension "cannot load more than once" 错误）。
    """
    import rapidocr_onnxruntime as _rapidocr_real
    from PIL import Image

    img_path = tmp_path / "test.png"
    Image.new("RGB", (200, 100), color=(0, 0, 0)).save(str(img_path))

    # RapidOCR 返回格式：list of [box_points, text, confidence]
    fake_result = [
        [
            [[10, 5], [90, 5], [90, 25], [10, 25]],  # box_points（顺时针四点）
            "Hello World",
            0.95,
        ],
        [
            [[10, 30], [150, 30], [150, 50], [10, 50]],
            "你好",
            0.87,
        ],
    ]
    fake_engine = MagicMock()
    fake_engine.return_value = (fake_result, None)

    # patch 函数调用层：RapidOCR() 构造器返回 fake_engine
    with patch.object(_rapidocr_real, "RapidOCR", return_value=fake_engine):
        blocks = perception_mod._run_ocr_on_file_sync(
            screenshot_path=str(img_path),
            bbox=None,
            snapshot_id="test-snap",
        )

    assert len(blocks) == 2

    b0 = blocks[0]
    assert isinstance(b0, TextBlock)
    assert b0.text == "Hello World"
    assert abs(b0.confidence - 0.95) < 1e-6
    assert b0.source == "ocr_rapidocr"
    assert b0.block_id.startswith("ocr_test-snap_")
    # bbox 来自四点包围盒
    assert b0.bbox.x == 10
    assert b0.bbox.y == 5
    assert b0.bbox.width == 80  # 90 - 10
    assert b0.bbox.height == 20  # 25 - 5

    b1 = blocks[1]
    assert b1.text == "你好"
    assert abs(b1.confidence - 0.87) < 1e-6


def test_run_ocr_with_bbox_offset(tmp_path: Any) -> None:
    """bbox 裁剪时坐标要加 offset。"""
    import rapidocr_onnxruntime as _rapidocr_real
    from PIL import Image

    img_path = tmp_path / "test.png"
    Image.new("RGB", (400, 300), color=(255, 255, 255)).save(str(img_path))

    fake_result = [
        [[[5, 5], [45, 5], [45, 25], [5, 25]], "Text", 0.99],
    ]
    fake_engine = MagicMock()
    fake_engine.return_value = (fake_result, None)

    with patch.object(_rapidocr_real, "RapidOCR", return_value=fake_engine):
        blocks = perception_mod._run_ocr_on_file_sync(
            screenshot_path=str(img_path),
            bbox={"x": 100, "y": 50, "width": 200, "height": 150},
            snapshot_id="offset-test",
        )

    assert len(blocks) == 1
    # x_min(5) + offset_x(100) = 105
    assert blocks[0].bbox.x == 105
    # y_min(5) + offset_y(50) = 55
    assert blocks[0].bbox.y == 55


def test_run_ocr_empty_result(tmp_path: Any) -> None:
    """RapidOCR 返回空列表时 TextBlock 列表为空。"""
    import rapidocr_onnxruntime as _rapidocr_real
    from PIL import Image

    img_path = tmp_path / "blank.png"
    Image.new("RGB", (100, 100)).save(str(img_path))

    fake_engine = MagicMock()
    fake_engine.return_value = (None, None)  # 无识别结果

    with patch.object(_rapidocr_real, "RapidOCR", return_value=fake_engine):
        blocks = perception_mod._run_ocr_on_file_sync(
            screenshot_path=str(img_path),
            bbox=None,
            snapshot_id="empty",
        )
    assert blocks == []


# ── do_ocr_region 错误路径 ────────────────────────────────────────────────────


async def test_do_ocr_region_raises_when_ocr_false() -> None:
    """caps.ocr=False 时 do_ocr_region raise RuntimeError。"""
    caps = _make_caps(ocr=False)
    with pytest.raises(RuntimeError, match="caps.ocr=False"):
        await perception_mod.do_ocr_region(
            bbox={"x": 0, "y": 0, "width": 100, "height": 100},
            screenshot_path="/nonexistent/file.png",
            caps=caps,
        )


async def test_do_ocr_region_raises_when_file_missing() -> None:
    """截图文件不存在时 do_ocr_region raise FileNotFoundError。"""
    caps = _make_caps(ocr=True)
    with pytest.raises(FileNotFoundError):
        await perception_mod.do_ocr_region(
            bbox={"x": 0, "y": 0, "width": 100, "height": 100},
            screenshot_path="/definitely/does/not/exist.png",
            caps=caps,
        )


# ── _uia_elements_to_models ───────────────────────────────────────────────────


def test_uia_elements_to_models_filters_zero_area() -> None:
    """零面积元素（width=0 或 height=0）被过滤掉。"""
    raw = [
        {
            "control_type": "ButtonControl",
            "name": "OK",
            "automation_id": "btn_ok",
            "rect": (10, 20, 110, 70),  # 100x50 → 有效
            "is_enabled": True,
            "is_offscreen": False,
            "depth": 0,
            "children_count": 0,
        },
        {
            "control_type": "PaneControl",
            "name": "",
            "automation_id": "",
            "rect": (0, 0, 0, 0),  # 零面积 → 应被过滤
            "is_enabled": True,
            "is_offscreen": False,
            "depth": 0,
            "children_count": 0,
        },
    ]
    result = perception_mod._uia_elements_to_models(raw, hwnd=0x1234)
    assert len(result) == 1
    assert result[0].control_type == "ButtonControl"
    assert result[0].bbox.width == 100
    assert result[0].bbox.height == 50
    assert result[0].source == "uia"


def test_uia_elements_to_models_element_id_format() -> None:
    """element_id 格式为 'uia_{hwnd}_{index}'。"""
    raw = [
        {
            "control_type": "EditControl",
            "name": "input",
            "automation_id": "",
            "rect": (0, 0, 200, 40),
            "is_enabled": True,
            "is_offscreen": False,
            "depth": 1,
            "children_count": 0,
        },
    ]
    result = perception_mod._uia_elements_to_models(raw, hwnd=0xABCD)
    assert result[0].element_id == "uia_43981_0"  # 0xABCD = 43981
