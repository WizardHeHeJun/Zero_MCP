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

import src.mcp.desktop.session_state as session_state_mod
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


@pytest.fixture(autouse=True)
def desktop_unlocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认注桩「桌面未锁」：避免单测真调 Win32（CI 无桌面会话会被判锁定而

    do_screen_snapshot 全线短路成骨架快照）。锁屏用例内再显式覆写。"""
    monkeypatch.setattr(session_state_mod, "_is_desktop_locked_sync", lambda: (False, False))


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
    """do_screen_snapshot 把前台窗口 rect（clamp 到虚拟屏）作为 OCR 裁剪 bbox 传入。

    Task 12 e2e 实测：全图 OCR 会把其他应用文本混入 perception_summary
    （跨窗口注入面），故 L2 OCR 必须与 L1 UIA 同口径裁剪到前台窗口。
    2026-07-11 多显示器修正：mss 抓全虚拟屏，clamp 口径从主屏改为虚拟屏 rect。
    """
    caps = _make_caps(ocr=True, mss_available=True)
    monkeypatch.setattr(perception_mod, "_get_active_window_info", lambda: (0x2222, "钉钉"))
    monkeypatch.setattr(perception_mod, "_get_screen_size", lambda: (1920, 1080))
    monkeypatch.setattr(perception_mod, "_probe_window_uia_hollow_sync", lambda hwnd: True)
    monkeypatch.setattr(perception_mod, "_collect_uia_tree_sync", lambda hwnd, max_depth: [])
    monkeypatch.setattr(
        perception_mod,
        "_take_screenshot_sync",
        lambda snap_id, tmp_dir: ("C:/fake/shot.png", (0, 0, 3840, 1080)),
    )
    # 窗口下缘越出虚拟屏 → 期望 clamp 到 (100,50)-(2000,1080)（右缘 2000 在虚拟屏内）
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
    assert seen_bbox == [({"x": 100, "y": 50, "width": 1900, "height": 1030}, (0, 0))]
    assert snapshot.capture_origin == (0, 0)


async def test_screen_snapshot_ocr_skipped_when_window_outside_virtual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """窗口 rect 与虚拟屏截图完全无交集（如已断开的显示器残留 rect）→
    **不再回退全图 OCR**（旧行为是跨窗口注入面，K2 修正）：OCR 完全不执行，
    text_blocks 置空 + degradations 落机读枚举 ocr_crop_invalid。"""
    caps = _make_caps(ocr=True, mss_available=True)
    monkeypatch.setattr(perception_mod, "_get_active_window_info", lambda: (0x3333, "幽灵窗"))
    monkeypatch.setattr(perception_mod, "_get_screen_size", lambda: (1920, 1080))
    monkeypatch.setattr(perception_mod, "_probe_window_uia_hollow_sync", lambda hwnd: False)
    monkeypatch.setattr(perception_mod, "_collect_uia_tree_sync", lambda hwnd, max_depth: [])
    monkeypatch.setattr(
        perception_mod,
        "_take_screenshot_sync",
        lambda snap_id, tmp_dir: ("C:/fake/shot.png", (0, 0, 3840, 1080)),
    )
    # rect 完全在虚拟屏 (0,0)-(3840,1080) 之外
    monkeypatch.setattr(perception_mod, "_get_window_rect", lambda hwnd: (4000, 200, 4800, 600))

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

    assert seen_bbox == [], "裁剪无效时不得执行任何 OCR（全图 OCR 是跨窗口注入面）"
    assert snapshot.text_blocks == []
    assert "ocr_crop_invalid" in snapshot.degradations


async def test_screen_snapshot_secondary_screen_window_crops_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """副屏窗口（rect 在 (1920,0)-(3840,1080) 内）经 mss 回退路径 OCR 裁剪正确。

    2026-07-11 实测修 bug：旧实现抓 mss monitors[1]（枚举顺序不保证是主屏）
    且 clamp 到主屏 → 副屏窗口被判「不在截图范围内」回退全图。新实现抓
    monitors[0] 全虚拟屏 → 副屏窗口正常裁剪（图像坐标 = 屏幕绝对坐标 − origin）。
    """
    caps = _make_caps(ocr=True, mss_available=True)
    monkeypatch.setattr(perception_mod, "_get_active_window_info", lambda: (0x6666, "副屏窗"))
    monkeypatch.setattr(perception_mod, "_get_screen_size", lambda: (1920, 1080))
    monkeypatch.setattr(perception_mod, "_probe_window_uia_hollow_sync", lambda hwnd: False)
    monkeypatch.setattr(perception_mod, "_collect_uia_tree_sync", lambda hwnd, max_depth: [])
    monkeypatch.setattr(
        perception_mod,
        "_take_screenshot_sync",
        lambda snap_id, tmp_dir: ("C:/fake/shot.png", (0, 0, 3840, 1080)),
    )
    # 窗口完全在副屏 (1920,0)-(3840,1080) 内
    monkeypatch.setattr(perception_mod, "_get_window_rect", lambda hwnd: (2000, 100, 3000, 900))

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

    # 不再回退全图：bbox 为图像坐标（origin=(0,0) 时与屏幕绝对坐标一致）
    assert seen_bbox == [({"x": 2000, "y": 100, "width": 1000, "height": 800}, (0, 0))]
    assert snapshot.capture_origin == (0, 0)
    # 主屏尺寸字段语义不变（不受虚拟屏影响）
    assert snapshot.screen_width == 1920
    assert snapshot.screen_height == 1080


async def test_screen_snapshot_negative_virtual_origin_crop_correct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """负虚拟屏 origin（副屏排在主屏左侧，SM_XVIRTUALSCREEN<0）下裁剪正确：

    - capture_origin = 虚拟屏 origin（负值）；
    - OCR crop bbox 为图像坐标 = 屏幕绝对坐标 − 虚拟屏 origin（恒非负）。
    """
    caps = _make_caps(ocr=True, mss_available=True)
    monkeypatch.setattr(perception_mod, "_get_active_window_info", lambda: (0x7777, "左副屏窗"))
    monkeypatch.setattr(perception_mod, "_get_screen_size", lambda: (1920, 1080))
    monkeypatch.setattr(perception_mod, "_probe_window_uia_hollow_sync", lambda hwnd: False)
    monkeypatch.setattr(perception_mod, "_collect_uia_tree_sync", lambda hwnd, max_depth: [])
    monkeypatch.setattr(
        perception_mod,
        "_take_screenshot_sync",
        lambda snap_id, tmp_dir: ("C:/fake/shot.png", (-1920, 0, 3840, 1080)),
    )
    # 窗口在左副屏（屏幕绝对坐标为负）
    monkeypatch.setattr(perception_mod, "_get_window_rect", lambda hwnd: (-1800, 100, -800, 600))

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

    # 图像坐标：x = -1800 - (-1920) = 120；origin 传 capture_origin（负值）
    assert seen_bbox == [({"x": 120, "y": 100, "width": 1000, "height": 500}, (-1920, 0))]
    assert snapshot.capture_origin == (-1920, 0)


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
    # PrintWindow 捕获失败（如窗口最小化）→ 回退 mss 全虚拟屏 + rect 裁剪
    monkeypatch.setattr(perception_mod, "_capture_window_sync", lambda hwnd, snap_id, tmp_dir: None)
    monkeypatch.setattr(
        perception_mod,
        "_take_screenshot_sync",
        lambda snap_id, tmp_dir: ("C:/fake/shot.png", (0, 0, 3840, 1080)),
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

    def fail_mss(snap_id: str, tmp_dir: str) -> tuple[str, tuple[int, int, int, int]]:
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


def test_run_ocr_with_bbox_and_negative_origin(tmp_path: Any) -> None:
    """负虚拟屏 origin 下坐标补偿正确：TextBlock.bbox = 图像坐标 + bbox + origin。

    模拟左副屏场景：全虚拟屏截图 origin=(-1920,0)，窗口图像坐标 bbox.x=120
    （屏幕绝对 -1800），OCR 识别点 x_min=5 → 屏幕绝对 x = 5 + 120 + (-1920) = -1795。
    """
    import rapidocr_onnxruntime as _rapidocr_real
    from PIL import Image

    img_path = tmp_path / "virtual.png"
    Image.new("RGB", (400, 300), color=(255, 255, 255)).save(str(img_path))

    fake_result = [
        [[[5, 5], [45, 5], [45, 25], [5, 25]], "Left", 0.9],
    ]
    fake_engine = MagicMock()
    fake_engine.return_value = (fake_result, None)

    with patch.object(_rapidocr_real, "RapidOCR", return_value=fake_engine):
        blocks = perception_mod._run_ocr_on_file_sync(
            screenshot_path=str(img_path),
            bbox={"x": 120, "y": 100, "width": 200, "height": 150},
            snapshot_id="neg-origin",
            origin=(-1920, 0),
        )

    assert len(blocks) == 1
    # x_min(5) + bbox.x(120) + origin.x(-1920) = -1795（屏幕绝对坐标，允许为负）
    assert blocks[0].bbox.x == -1795
    # y_min(5) + bbox.y(100) + origin.y(0) = 105
    assert blocks[0].bbox.y == 105


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


# ── K2：锁屏骨架快照 + 各回退分支机读降级枚举 ─────────────────────────────────


def _patch_healthy_pipeline(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """注桩一条「全通」感知流水线（前台窗口 + mss + OCR），返回 OCR 调用记录。"""
    monkeypatch.setattr(perception_mod, "_get_active_window_info", lambda: (0x2222, "健康窗"))
    monkeypatch.setattr(perception_mod, "_get_screen_size", lambda: (1920, 1080))
    monkeypatch.setattr(perception_mod, "_probe_window_uia_hollow_sync", lambda hwnd: False)
    monkeypatch.setattr(perception_mod, "_collect_uia_tree_sync", lambda hwnd, max_depth: [])
    monkeypatch.setattr(
        perception_mod,
        "_take_screenshot_sync",
        lambda snap_id, tmp_dir: ("C:/fake/shot.png", (0, 0, 1920, 1080)),
    )
    monkeypatch.setattr(perception_mod, "_get_window_rect", lambda hwnd: (100, 100, 900, 700))

    seen_ocr: list[Any] = []

    def fake_ocr(
        screenshot_path: str, bbox: Any, snapshot_id: str, origin: tuple[int, int] = (0, 0)
    ) -> list[Any]:
        seen_ocr.append((bbox, origin))
        return []

    monkeypatch.setattr(perception_mod, "_run_ocr_on_file_sync", fake_ocr)
    return seen_ocr


async def test_screen_snapshot_locked_returns_skeleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """锁定态 → 跳过截图/UIA/OCR，返回 desktop_locked=True + ["desktop_locked"] 骨架。"""
    caps = _make_caps(ocr=True, mss_available=True)
    monkeypatch.setattr(session_state_mod, "_is_desktop_locked_sync", lambda: (True, False))
    monkeypatch.setattr(perception_mod, "_get_screen_size", lambda: (1920, 1080))

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("锁定态不得触碰 UIA/截图/OCR")

    monkeypatch.setattr(perception_mod, "_get_active_window_info", fail)
    monkeypatch.setattr(perception_mod, "_probe_window_uia_hollow_sync", fail)
    monkeypatch.setattr(perception_mod, "_collect_uia_tree_sync", fail)
    monkeypatch.setattr(perception_mod, "_capture_window_sync", fail)
    monkeypatch.setattr(perception_mod, "_take_screenshot_sync", fail)
    monkeypatch.setattr(perception_mod, "_run_ocr_on_file_sync", fail)

    snapshot = await perception_mod.do_screen_snapshot(
        mode="uia_ocr", capture_screenshot=True, caps=caps, screenshot_tmp_dir=""
    )

    assert snapshot.desktop_locked is True
    assert snapshot.degradations == ["desktop_locked"]
    assert snapshot.uia_elements == []
    assert snapshot.text_blocks == []
    assert snapshot.screenshot_path is None
    assert snapshot.window_captured is False
    assert snapshot.active_window_title is None
    assert snapshot.is_untrusted is True


async def test_screen_snapshot_lock_probe_failed_marks_degradation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """探测自身失败 → 按未锁继续感知，degradations 落 lock_probe_failed。"""
    caps = _make_caps(ocr=True, mss_available=True)
    monkeypatch.setattr(session_state_mod, "_is_desktop_locked_sync", lambda: (False, True))
    _patch_healthy_pipeline(monkeypatch)

    snapshot = await perception_mod.do_screen_snapshot(
        mode="uia_ocr", capture_screenshot=False, caps=caps, screenshot_tmp_dir=""
    )

    assert snapshot.desktop_locked is False
    assert "lock_probe_failed" in snapshot.degradations
    assert snapshot.screenshot_path == "C:/fake/shot.png"  # 感知未被短路


async def test_screen_snapshot_healthy_path_no_degradations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全通路径 → degradations 为空列表、desktop_locked=False（正对照）。"""
    caps = _make_caps(ocr=True, mss_available=True)
    _patch_healthy_pipeline(monkeypatch)

    snapshot = await perception_mod.do_screen_snapshot(
        mode="uia_ocr", capture_screenshot=False, caps=caps, screenshot_tmp_dir=""
    )

    assert snapshot.degradations == []
    assert snapshot.desktop_locked is False


async def test_screen_snapshot_window_capture_failed_degradation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """window_handle 指定且 PrintWindow 失败 → window_capture_failed + 回退 mss，
    window_captured=False。"""
    caps = _make_caps(ocr=True, mss_available=True)
    _patch_healthy_pipeline(monkeypatch)
    monkeypatch.setattr(perception_mod, "_get_window_title", lambda hwnd: "目标窗")
    monkeypatch.setattr(perception_mod, "_capture_window_sync", lambda hwnd, snap_id, tmp_dir: None)

    snapshot = await perception_mod.do_screen_snapshot(
        mode="uia_ocr",
        capture_screenshot=False,
        caps=caps,
        screenshot_tmp_dir="",
        window_handle=0x4444,
    )

    assert "window_capture_failed" in snapshot.degradations
    assert snapshot.window_captured is False
    assert snapshot.screenshot_path == "C:/fake/shot.png"  # mss 回退成功


async def test_screen_snapshot_window_captured_field_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PrintWindow 成功 → window_captured=True 写进快照，无 window_capture_failed。"""
    caps = _make_caps(ocr=True, mss_available=True)
    _patch_healthy_pipeline(monkeypatch)
    monkeypatch.setattr(perception_mod, "_get_window_title", lambda hwnd: "目标窗")
    monkeypatch.setattr(
        perception_mod, "_capture_window_sync", lambda hwnd, snap_id, tmp_dir: "C:/fake/win.png"
    )

    snapshot = await perception_mod.do_screen_snapshot(
        mode="uia_ocr",
        capture_screenshot=False,
        caps=caps,
        screenshot_tmp_dir="",
        window_handle=0x5555,
    )

    assert snapshot.window_captured is True
    assert snapshot.degradations == []


async def test_screen_snapshot_mss_exception_degradation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mss 截图抛异常 → screenshot_failed，无像素产物。"""
    caps = _make_caps(ocr=True, mss_available=True)
    _patch_healthy_pipeline(monkeypatch)

    def raise_mss(snap_id: str, tmp_dir: str) -> tuple[str, tuple[int, int, int, int]]:
        raise RuntimeError("mss boom")

    monkeypatch.setattr(perception_mod, "_take_screenshot_sync", raise_mss)

    snapshot = await perception_mod.do_screen_snapshot(
        mode="uia_ocr", capture_screenshot=False, caps=caps, screenshot_tmp_dir=""
    )

    assert "screenshot_failed" in snapshot.degradations
    assert snapshot.screenshot_path is None


async def test_screen_snapshot_mss_unavailable_degradation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """caps.mss_available=False 且需截图 → mss_unavailable。"""
    caps = _make_caps(ocr=True, mss_available=False)
    _patch_healthy_pipeline(monkeypatch)

    snapshot = await perception_mod.do_screen_snapshot(
        mode="uia_ocr", capture_screenshot=False, caps=caps, screenshot_tmp_dir=""
    )

    assert "mss_unavailable" in snapshot.degradations
    assert snapshot.screenshot_path is None


async def test_screen_snapshot_ocr_exception_degradation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OCR 执行抛异常 → ocr_error，text_blocks 为空但快照仍返回。"""
    caps = _make_caps(ocr=True, mss_available=True)
    _patch_healthy_pipeline(monkeypatch)

    def raise_ocr(
        screenshot_path: str, bbox: Any, snapshot_id: str, origin: tuple[int, int] = (0, 0)
    ) -> list[Any]:
        raise RuntimeError("ocr boom")

    monkeypatch.setattr(perception_mod, "_run_ocr_on_file_sync", raise_ocr)

    snapshot = await perception_mod.do_screen_snapshot(
        mode="uia_ocr", capture_screenshot=False, caps=caps, screenshot_tmp_dir=""
    )

    assert "ocr_error" in snapshot.degradations
    assert snapshot.text_blocks == []


async def test_screen_snapshot_ocr_unavailable_degradation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """caps.ocr=False 且模式需 OCR → ocr_unavailable（截图正常落盘）。"""
    caps = _make_caps(ocr=False, mss_available=True)
    seen_ocr = _patch_healthy_pipeline(monkeypatch)

    snapshot = await perception_mod.do_screen_snapshot(
        mode="uia_ocr", capture_screenshot=False, caps=caps, screenshot_tmp_dir=""
    )

    assert "ocr_unavailable" in snapshot.degradations
    assert seen_ocr == []
    assert snapshot.screenshot_path == "C:/fake/shot.png"


# ── UIA 采集失败防御（2026-09-01 实机标定遗留②，PR #34 审查后调用方级落地）────


def _raise_com(*args: Any, **kwargs: Any) -> Any:
    raise OSError("COMError: (-2147220991, '事件无法调用任何订户')")


class TestUiaCollectFailedDegradation:
    """_collect_uia_tree_sync 抛异常（实测 COMError）时：do_screen_snapshot 记
    机读标记 uia_collect_failed 不炸调用；uia_only 模式强制升级 OCR 兜底
    （否则产出「UIA 空+无截图+无 OCR+无标记」的静默全空快照）；
    do_get_uia_tree 返回空列表（已知局限：与真空树不可区分）。"""

    async def test_snapshot_marks_degradation_and_survives(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        caps = _make_caps(ocr=True, mss_available=True)
        _patch_healthy_pipeline(monkeypatch)
        monkeypatch.setattr(perception_mod, "_collect_uia_tree_sync", _raise_com)

        snapshot = await perception_mod.do_screen_snapshot(
            mode="uia_ocr", capture_screenshot=False, caps=caps, screenshot_tmp_dir=""
        )

        assert "uia_collect_failed" in snapshot.degradations
        assert snapshot.uia_elements == []

    async def test_uia_only_mode_upgrades_to_ocr_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """uia_only + 采集失败 → 强制升级 uia_ocr（OCR 真被调、截图真落盘），
        修复前该场景是静默全空快照（审查 WARN② 场景固化）。"""
        caps = _make_caps(ocr=True, mss_available=True)
        seen_ocr = _patch_healthy_pipeline(monkeypatch)
        monkeypatch.setattr(perception_mod, "_collect_uia_tree_sync", _raise_com)

        snapshot = await perception_mod.do_screen_snapshot(
            mode="uia_only", capture_screenshot=False, caps=caps, screenshot_tmp_dir=""
        )

        assert "uia_collect_failed" in snapshot.degradations
        assert snapshot.screenshot_path is not None, "升级后应有截图产物"
        assert len(seen_ocr) == 1, "升级后 OCR 应被调用恰一次"

    async def test_uia_only_healthy_does_not_upgrade(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """正对照：uia_only 健康路径不升级、不触 OCR、无标记（防误吞）。"""
        caps = _make_caps(ocr=True, mss_available=True)
        seen_ocr = _patch_healthy_pipeline(monkeypatch)

        snapshot = await perception_mod.do_screen_snapshot(
            mode="uia_only", capture_screenshot=False, caps=caps, screenshot_tmp_dir=""
        )

        assert "uia_collect_failed" not in snapshot.degradations
        assert seen_ocr == []

    async def test_get_uia_tree_returns_empty_on_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        caps = _make_caps(ocr=True, mss_available=True)
        monkeypatch.setattr(perception_mod, "_collect_uia_tree_sync", _raise_com)

        result = await perception_mod.do_get_uia_tree(
            window_handle=0x12345, max_depth=5, caps=caps
        )
        assert result == []
