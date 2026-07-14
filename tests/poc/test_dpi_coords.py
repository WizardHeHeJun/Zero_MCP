"""蓝图 Task 13：DPI/多显示器坐标一致性 PoC 驱动（gap#3）。

验收口径：感知坐标 → 点击落点误差 **< 2px**；达不到时把逐点偏差数字文档化
（notes/ 报告），不静默放行。

背景（2026-07-11 本机被动探测）：
- 双屏各 1920x1080 / 96dpi；mss.monitors 枚举顺序**不保证** [1]=主显示器
  （本机实测 [1]=副屏），perception._take_screenshot_sync 的注释假设已被证伪；
- 显示器可排在主屏左/上方 → 虚拟屏原点可为负（SM_XVIRTUALSCREEN<0），
  坐标链任何一环若按「(0,0)=屏幕左上」假设换算即错位。

方法：自建 tkinter 校准窗口（白底 + 大号黑字 CLICKME + <Button-1> 落点记录），
对每块屏幕的 3 个摆位（左上/中央/右下）跑全链路回环：

    起窗口 → screen_snapshot(mode="uia_ocr", capture_screenshot=True,
    window_handle=hwnd) 定向感知 → text_blocks 找 CLICKME bbox 中心
    → click_element(coordinates=中心) → 窗口记录实际落点（屏幕绝对坐标）
    → 误差 = |感知中心 - 实际落点|。

误差 < 2px 表示「感知 → 操控」坐标系全链一致。注意本指标与 OCR 识别精度无关：
点击目标 = 感知中心本身，量的是 TextBlock.bbox（capture_origin 补偿后的屏幕
绝对坐标）、pyautogui 注入、窗口自报坐标三方的**坐标口径**是否同一。

实现要点：
- 本进程既是 Tk 窗口宿主又是 MCP client——await MCP 调用期间必须持续
  root.update() 泵消息（call_with_pump），否则 server 侧 PrintWindow 的
  WM_PRINT SendMessage 会因目标窗口线程不泵而卡死。
- 不开 mainloop；点击注入后继续泵消息等 <Button-1> 事件入队处理。
- hwnd 用唯一随机标题 FindWindowW 取（比 winfo_id/frame 换算稳）。

用法（先 conda activate affective-expression，见 CLAUDE.md）：
    set SCREEN_CAPABILITY_ENABLED=true
    python tests/poc/test_dpi_coords.py                     # 主屏 + 副屏
    python tests/poc/test_dpi_coords.py --screen primary    # 只跑主屏
    python tests/poc/test_dpi_coords.py --screen secondary  # 只跑副屏

pytest 面：纯函数（bbox 中心/误差/摆位规划/选屏/汇总/渲染）无 IO，CI 可跑；
全链路回环标 @pytest.mark.realenv，实机手动：
    set SCREEN_CAPABILITY_ENABLED=true
    pytest tests/poc/test_dpi_coords.py -m realenv -s

产出：stdout markdown 摘要；原始 JSON 落 tests/poc/output/dpi_coords_<时间戳>.json。
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import ctypes.wintypes as wintypes
import json
import logging
import math
import os
import secrets
import statistics
import sys
import time
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

import pytest

logger = logging.getLogger("dpi_coords_poc")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "tests" / "poc" / "output"

TARGET_TEXT = "CLICKME"
ERROR_TOLERANCE_PX = 2.0  # gap#3 验收阈值（欧氏距离，严格小于）
WINDOW_W, WINDOW_H = 480, 240  # 校准窗口尺寸（物理像素，96dpi 下）
POSITION_MARGIN = 60  # 角落摆位距屏幕边缘的边距
MONITORINFOF_PRIMARY = 0x1


# ══════════════════════════════════════════════════════════════════════════════
# 纯函数（无 IO，可单测）
# ══════════════════════════════════════════════════════════════════════════════


def bbox_center(bbox: dict[str, int]) -> tuple[int, int]:
    """{x,y,width,height}（屏幕绝对物理像素）的中心点（整除取整）。"""
    return bbox["x"] + bbox["width"] // 2, bbox["y"] + bbox["height"] // 2


def point_error(expected: tuple[int, int], actual: tuple[int, int]) -> dict[str, float]:
    """感知中心 vs 实际落点的误差分量。

    Returns:
        {dx, dy, chebyshev, euclidean}：dx/dy 为 actual-expected 的有符号分量，
        chebyshev = max(|dx|,|dy|)，euclidean 为欧氏距离（验收口径）。
    """
    dx = float(actual[0] - expected[0])
    dy = float(actual[1] - expected[1])
    return {
        "dx": dx,
        "dy": dy,
        "chebyshev": max(abs(dx), abs(dy)),
        "euclidean": round(math.hypot(dx, dy), 3),
    }


def find_target_block(
    text_blocks: list[dict[str, Any]], target: str = TARGET_TEXT
) -> dict[str, Any] | None:
    """text_blocks 里找目标文本块（去空白 + 大小写不敏感，含即匹配）。

    多个匹配取置信度最高的一个；无匹配返回 None。
    """
    norm_target = target.replace(" ", "").upper()
    matched = [
        b for b in text_blocks if norm_target in str(b.get("text", "")).replace(" ", "").upper()
    ]
    return max(matched, key=lambda b: float(b.get("confidence", 0.0)), default=None)


def plan_window_positions(
    screen_rect: tuple[int, int, int, int],
    win_w: int,
    win_h: int,
    margin: int = POSITION_MARGIN,
) -> list[tuple[str, int, int]]:
    """一块屏幕上的三个摆位（左上/中央/右下），返回 (label, x, y) 列表。

    直接基于该屏 rect 计算，天然支持负原点（显示器排在主屏左/上方时
    left/top 为负——gap#3 修复必须覆盖的场景）。
    """
    left, top, right, bottom = screen_rect
    center_x = left + (right - left - win_w) // 2
    center_y = top + (bottom - top - win_h) // 2
    return [
        ("左上", left + margin, top + margin),
        ("中央", center_x, center_y),
        ("右下", right - win_w - margin, bottom - win_h - margin),
    ]


def pick_screen_rect(
    monitors: list[dict[str, Any]], which: str
) -> tuple[int, int, int, int] | None:
    """按 primary/secondary 从显示器清单选屏 rect；该屏不存在返回 None。

    枚举顺序不保证主屏在前（本机 mss 实测 monitors[1]=副屏），一律按
    is_primary 位判定，不按下标。
    """
    if which == "primary":
        pool = [m for m in monitors if m.get("is_primary")] or list(monitors)
    elif which == "secondary":
        pool = [m for m in monitors if not m.get("is_primary")]
    else:
        raise ValueError(f"未知屏幕选择：{which!r}（可选 primary/secondary）")
    if not pool:
        return None
    left, top, right, bottom = pool[0]["rect"]
    return int(left), int(top), int(right), int(bottom)


def summarize_errors(
    records: list[dict[str, Any]], tolerance: float = ERROR_TOLERANCE_PX
) -> dict[str, Any]:
    """汇总逐点误差 → max/mean + 通过计数 + 总判定。

    只有全部点闭环（error 非 None）且每点欧氏误差 < tolerance 才 overall_pass。
    未闭环点（OCR miss / 点击丢失）计入 failed_count 并使总判定不通过——
    「无法闭环」本身就是 gap#3 要文档化的偏差。
    """
    closed = [r for r in records if r.get("error") is not None]
    eucl = [float(r["error"]["euclidean"]) for r in closed]
    cheb = [float(r["error"]["chebyshev"]) for r in closed]
    return {
        "point_count": len(records),
        "closed_loop_count": len(closed),
        "failed_count": len(records) - len(closed),
        "max_euclidean": max(eucl) if eucl else None,
        "mean_euclidean": round(statistics.mean(eucl), 3) if eucl else None,
        "max_chebyshev": max(cheb) if cheb else None,
        "pass_count": sum(1 for e in eucl if e < tolerance),
        "overall_pass": bool(closed)
        and len(closed) == len(records)
        and all(e < tolerance for e in eucl),
        "tolerance_px": tolerance,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """输出可直接贴进 notes 报告的 markdown 摘要（纯函数）。"""
    meta = report["meta"]
    tolerance = meta["tolerance_px"]
    lines: list[str] = [
        "## Task 13 DPI/多显示器坐标一致性 PoC（gap#3）",
        "",
        f"- 采集时间 {meta['collected_at']}；DPI 感知={meta['dpi_awareness']}；"
        f"验收阈值 <{tolerance}px（欧氏）",
        "- 显示器："
        + "；".join(
            f"{'主' if m['is_primary'] else '副'}屏 rect={tuple(m['rect'])}"
            for m in meta["monitors"]
        ),
    ]
    for screen_label, data in report["screens"].items():
        lines.append("")
        if data.get("skipped"):
            lines.append(f"### {screen_label}：跳过（{data['skipped']}）")
            continue
        lines.append(f"### {screen_label} rect={tuple(data['screen_rect'])}")
        lines.append("")
        lines.append("| 摆位 | 感知中心 | 实际落点 | dx | dy | 欧氏误差 | 判定 |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in data["records"]:
            err = r.get("error")
            if err is None:
                center = r.get("perceived_center")
                center_str = str(tuple(center)) if center else "—"
                note = r.get("note") or "未闭环"
                lines.append(f"| {r['position']} | {center_str} | — | — | — | — | ❌ {note} |")
            else:
                ok = "✅" if err["euclidean"] < tolerance else "❌"
                lines.append(
                    f"| {r['position']} | {tuple(r['perceived_center'])} "
                    f"| {tuple(r['actual_click'])} | {err['dx']:+.0f} | {err['dy']:+.0f} "
                    f"| {err['euclidean']} | {ok} |"
                )
        s = data["summary"]
        verdict = "✅ 通过" if s["overall_pass"] else "❌ 未过（逐点偏差需文档化）"
        lines.append("")
        lines.append(
            f"- 汇总：闭环 {s['closed_loop_count']}/{s['point_count']}；"
            f"欧氏 max={s['max_euclidean']} mean={s['mean_euclidean']}；"
            f"chebyshev max={s['max_chebyshev']} → {verdict}"
        )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 纯函数单元测试（CI 无桌面可跑）
# ══════════════════════════════════════════════════════════════════════════════


def test_bbox_center_basic() -> None:
    """中心 = 左上角 + 宽高整除 2。"""
    assert bbox_center({"x": 10, "y": 20, "width": 100, "height": 50}) == (60, 45)


def test_bbox_center_odd_size_floor() -> None:
    """奇数宽高向下取整（与消费侧整型坐标一致）。"""
    assert bbox_center({"x": 0, "y": 0, "width": 5, "height": 3}) == (2, 1)


def test_point_error_zero() -> None:
    """同点误差为 0。"""
    err = point_error((720, 420), (720, 420))
    assert err["euclidean"] == 0.0
    assert err["chebyshev"] == 0.0


def test_point_error_components() -> None:
    """dx/dy 有符号；chebyshev/euclidean 按定义。"""
    err = point_error((100, 100), (103, 96))
    assert (err["dx"], err["dy"]) == (3.0, -4.0)
    assert err["chebyshev"] == 4.0
    assert err["euclidean"] == 5.0


def test_find_target_block_case_space_insensitive() -> None:
    """OCR 可能给 'click me' 之类变体——去空白 + 大小写不敏感匹配。"""
    blocks = [{"text": "click me", "confidence": 0.8}]
    assert find_target_block(blocks) is blocks[0]


def test_find_target_block_prefers_confidence() -> None:
    """多个匹配取置信度最高；不含目标文本的块不参与。"""
    blocks = [
        {"text": "CLICKME", "confidence": 0.5},
        {"text": "xCLICKMEx", "confidence": 0.9},
        {"text": "发送", "confidence": 0.99},
    ]
    best = find_target_block(blocks)
    assert best is not None
    assert best["confidence"] == 0.9


def test_find_target_block_missing_returns_none() -> None:
    """无匹配（或空列表）返回 None，不抛。"""
    assert find_target_block([{"text": "发送", "confidence": 0.9}]) is None
    assert find_target_block([]) is None


def test_plan_window_positions_three_spots_in_bounds() -> None:
    """三摆位（左上/中央/右下）均完整落在屏内。"""
    rect = (0, 0, 1920, 1080)
    plans = plan_window_positions(rect, 480, 240, margin=60)
    assert [p[0] for p in plans] == ["左上", "中央", "右下"]
    for _, x, y in plans:
        assert 0 <= x and x + 480 <= 1920
        assert 0 <= y and y + 240 <= 1080
    assert plans[1][1:] == (720, 420)  # 中央
    assert plans[2][1:] == (1380, 780)  # 右下


def test_plan_window_positions_negative_origin() -> None:
    """副屏排在主屏左侧 → 屏 rect 原点为负（gap#3 修复必须覆盖）。"""
    rect = (-1920, 0, 0, 1080)
    plans = plan_window_positions(rect, 480, 240, margin=60)
    assert plans[0][1:] == (-1860, 60)
    for _, x, y in plans:
        assert -1920 <= x and x + 480 <= 0
        assert 0 <= y and y + 240 <= 1080


def test_pick_screen_rect_by_primary_flag_not_order() -> None:
    """枚举顺序不保证主屏在前（本机实测副屏在 [0]）——按 is_primary 位选。"""
    monitors = [
        {"rect": (1920, 0, 3840, 1080), "is_primary": False},
        {"rect": (0, 0, 1920, 1080), "is_primary": True},
    ]
    assert pick_screen_rect(monitors, "primary") == (0, 0, 1920, 1080)
    assert pick_screen_rect(monitors, "secondary") == (1920, 0, 3840, 1080)


def test_pick_screen_rect_no_secondary_returns_none() -> None:
    """单显示器环境选 secondary 返回 None（驱动侧据此 skip）。"""
    monitors = [{"rect": (0, 0, 1920, 1080), "is_primary": True}]
    assert pick_screen_rect(monitors, "secondary") is None
    assert pick_screen_rect(monitors, "primary") == (0, 0, 1920, 1080)


def test_summarize_errors_all_within_tolerance() -> None:
    """全部闭环且 <2px → overall_pass=True，max/mean 正确。"""
    records = [
        {"error": {"euclidean": 0.0, "chebyshev": 0.0}},
        {"error": {"euclidean": 1.0, "chebyshev": 1.0}},
        {"error": {"euclidean": 1.4, "chebyshev": 1.0}},
    ]
    s = summarize_errors(records)
    assert s["overall_pass"] is True
    assert s["max_euclidean"] == 1.4
    assert s["mean_euclidean"] == 0.8
    assert s["pass_count"] == 3


def test_summarize_errors_at_tolerance_fails() -> None:
    """恰等于阈值不算过（验收口径为严格 <2px）。"""
    s = summarize_errors([{"error": {"euclidean": 2.0, "chebyshev": 2.0}}])
    assert s["overall_pass"] is False
    assert s["pass_count"] == 0


def test_summarize_errors_open_loop_counts_failed() -> None:
    """有未闭环点（OCR miss / 点击丢失）→ failed_count 记数且总判定不过。"""
    records = [
        {"error": {"euclidean": 0.0, "chebyshev": 0.0}},
        {"error": None, "note": "OCR 未识别到 CLICKME"},
    ]
    s = summarize_errors(records)
    assert s["closed_loop_count"] == 1
    assert s["failed_count"] == 1
    assert s["overall_pass"] is False


def test_render_markdown_smoke() -> None:
    """渲染含通过行、未闭环行、跳过屏三种形态，关键信息在位。"""
    records = [
        {
            "position": "中央",
            "perceived_center": (960, 540),
            "actual_click": (960, 540),
            "error": {"dx": 0.0, "dy": 0.0, "chebyshev": 0.0, "euclidean": 0.0},
        },
        {"position": "右下", "perceived_center": None, "error": None, "note": "OCR miss"},
    ]
    report = {
        "meta": {
            "collected_at": "2026-07-14 00:00:00",
            "dpi_awareness": "per_monitor_v2",
            "monitors": [
                {"rect": (0, 0, 1920, 1080), "is_primary": True},
                {"rect": (1920, 0, 3840, 1080), "is_primary": False},
            ],
            "tolerance_px": ERROR_TOLERANCE_PX,
        },
        "screens": {
            "primary": {
                "screen_rect": (0, 0, 1920, 1080),
                "records": records,
                "summary": summarize_errors(records),
            },
            "secondary": {"skipped": "该屏不存在（单显示器环境）"},
        },
    }
    md = render_markdown(report)
    assert "Task 13" in md
    assert "中央" in md and "(960, 540)" in md
    assert "OCR miss" in md
    assert "跳过" in md
    assert "未过" in md  # 有未闭环点 → 总判定不过


# ══════════════════════════════════════════════════════════════════════════════
# Win32 辅助（驱动侧）
# ══════════════════════════════════════════════════════════════════════════════


def set_dpi_awareness() -> str:
    """在创建 tkinter 窗口前设进程 DPI 感知，保证 winfo/事件坐标是物理像素。

    与 perception.py._setup_dpi 同口径：PER_MONITOR_AWARE_V2(-4) → shcore
    per-monitor → system 三级兜底。
    """
    user32 = ctypes.windll.user32
    try:
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return "per_monitor_v2"
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return "per_monitor"
    except (AttributeError, OSError):
        pass
    user32.SetProcessDPIAware()
    return "system"


def list_monitor_rects() -> list[dict[str, Any]]:
    """EnumDisplayMonitors 枚举各物理显示器 rect（虚拟屏坐标，支持负原点）。

    Returns:
        [{"rect": (left, top, right, bottom), "is_primary": bool}, ...]
    """
    user32 = ctypes.windll.user32

    class MonitorInfo(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
        ]

    monitors: list[dict[str, Any]] = []

    @ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.RECT),
        wintypes.LPARAM,
    )
    def enum_cb(hmonitor: int, _hdc: int, _lprect: Any, _lparam: int) -> bool:
        info = MonitorInfo()
        info.cbSize = ctypes.sizeof(MonitorInfo)
        if user32.GetMonitorInfoW(hmonitor, ctypes.byref(info)):
            r = info.rcMonitor
            monitors.append(
                {
                    "rect": (int(r.left), int(r.top), int(r.right), int(r.bottom)),
                    "is_primary": bool(info.dwFlags & MONITORINFOF_PRIMARY),
                }
            )
        return True

    user32.EnumDisplayMonitors(None, None, enum_cb, 0)
    return monitors


def get_window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """窗口物理像素 rect（含 Win10 不可见边框），失败返回 None（诊断用）。"""
    rect = wintypes.RECT()
    if ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# tkinter 校准窗口（驱动侧，需真实桌面）
# ══════════════════════════════════════════════════════════════════════════════


class ClickProbeWindow:
    """自建校准窗口：白底 + 大号黑字 CLICKME + <Button-1> 落点记录。

    不开 mainloop——用 root.update() 手动泵消息，与 asyncio 同线程共存。
    topmost 保证注入点击落在本窗口而非遮挡者（Win32 状态不可信，落点须自证）。
    """

    def __init__(self, x: int, y: int, width: int = WINDOW_W, height: int = WINDOW_H) -> None:
        import tkinter as tk

        self.title = f"DPIPOC_{secrets.token_hex(4)}"  # 唯一随机标题供 FindWindowW
        self.clicks: list[tuple[int, int]] = []
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.configure(bg="white")
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)
        label = tk.Label(
            self.root,
            text=TARGET_TEXT,
            font=("Arial", 44, "bold"),
            fg="black",
            bg="white",
        )
        label.place(relx=0.5, rely=0.5, anchor="center")
        # 点击可能落在 root 或 label 上，两处都绑；
        # event.widget.winfo_rootx()+event.x 即屏幕绝对坐标（对任意绑定 widget 成立）
        self.root.bind("<Button-1>", self._on_click)
        label.bind("<Button-1>", self._on_click)
        self.pump(rounds=5)

    def _on_click(self, event: Any) -> None:
        screen_x = int(event.widget.winfo_rootx() + event.x)
        screen_y = int(event.widget.winfo_rooty() + event.y)
        self.clicks.append((screen_x, screen_y))

    def pump(self, rounds: int = 3, interval_s: float = 0.05) -> None:
        """手动泵 Tk 消息循环若干轮（代替 mainloop；同步小步，仅驱动内部用）。"""
        for _ in range(rounds):
            self.root.update()
            time.sleep(interval_s)

    def move_to(self, x: int, y: int) -> None:
        """移动窗口到屏幕绝对坐标（Tk geometry 接受负坐标 '+-100+300'）。"""
        self.root.geometry(f"+{x}+{y}")
        self.pump(rounds=5)

    def find_hwnd(self) -> int:
        """经唯一随机标题 FindWindowW 取顶层 HWND。"""
        hwnd = ctypes.windll.user32.FindWindowW(None, self.title)
        if not hwnd:
            raise RuntimeError(f"FindWindowW 未找到标题 {self.title!r} 的窗口")
        return int(hwnd)

    async def wait_click(self, timeout_s: float = 3.0) -> tuple[int, int] | None:
        """泵消息直到收到点击或超时，返回最后一次点击的屏幕绝对坐标。"""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.root.update()
            if self.clicks:
                return self.clicks[-1]
            await asyncio.sleep(0.05)
        return None

    def close(self) -> None:
        try:
            self.root.destroy()
        except Exception as exc:  # noqa: BLE001 — 清理失败不影响结果
            logger.debug("窗口销毁失败（忽略）：%s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# 全链路回环驱动（需真实桌面 + 真实 MCP 栈）
# ══════════════════════════════════════════════════════════════════════════════


def ensure_capability_enabled() -> None:
    """SCREEN_CAPABILITY_ENABLED 未启用时 SystemExit 给提示（由运行者设置）。"""
    if os.environ.get("SCREEN_CAPABILITY_ENABLED", "").lower() not in {"1", "true", "yes"}:
        raise SystemExit(
            "SCREEN_CAPABILITY_ENABLED 未启用，全链路回环需要真实 MCP 栈。请先设置：\n"
            "  PowerShell: $env:SCREEN_CAPABILITY_ENABLED = 'true'\n"
            "  cmd:        set SCREEN_CAPABILITY_ENABLED=true"
        )


async def call_with_pump(win: ClickProbeWindow, awaitable: Awaitable[Any]) -> Any:
    """await 一个 MCP 调用的同时持续泵 Tk 消息。

    本进程既是 Tk 窗口宿主又是 MCP client：await 期间不泵消息，server 侧
    PrintWindow 的 WM_PRINT SendMessage 会因本窗口线程无响应而卡死/超时。
    """
    task: asyncio.Task[Any] = asyncio.ensure_future(awaitable)
    try:
        while not task.done():
            win.root.update()
            await asyncio.sleep(0.05)
    except Exception:
        task.cancel()
        raise
    return task.result()


async def run_roundtrip_on_screen(
    client: Any,
    screen_label: str,
    screen_rect: tuple[int, int, int, int],
    perceive_via: str = "window_handle",
) -> list[dict[str, Any]]:
    """在指定屏幕跑 N=3 摆位的「感知 → 点击 → 落点」闭环，返回逐点记录。

    Args:
        client: 已连接的 DesktopMCPClient。
        screen_label: "primary" | "secondary"（仅记录用）。
        screen_rect: 该屏 rect (left, top, right, bottom)，虚拟屏坐标。
        perceive_via: "window_handle"（PrintWindow 定向感知优先路径）|
            "foreground"（不传 window_handle，走前台窗口 + mss 全虚拟屏 +
            rect 裁剪路径——Task 13 修复的新数学只有此路径实机可验，
            code-review WARN-3）。
    """
    positions = plan_window_positions(screen_rect, WINDOW_W, WINDOW_H)
    records: list[dict[str, Any]] = []
    win = ClickProbeWindow(positions[0][1], positions[0][2])
    try:
        for pos_label, x, y in positions:
            rec: dict[str, Any] = {
                "screen": screen_label,
                "perceive_via": perceive_via,
                "position": pos_label,
                "window_xy": (x, y),
                "hwnd": None,
                "window_rect": None,
                "capture_origin": None,
                "perception_mode": None,
                "screenshot_path": None,
                "target_block": None,
                "perceived_center": None,
                "actual_click": None,
                "error": None,
                "note": None,
            }
            try:
                win.move_to(x, y)
                await asyncio.sleep(0.3)  # 等合成器落位 + 首帧渲染
                win.pump(rounds=3)
                hwnd = win.find_hwnd()
                rec["hwnd"] = hwnd
                rec["window_rect"] = get_window_rect(hwnd)

                if perceive_via == "foreground":
                    # 前台路径：窗口置前台后不传 window_handle——感知层走
                    # GetForegroundWindow + mss 全虚拟屏 + 窗口 rect 裁剪（新数学）
                    ctypes.windll.user32.SetForegroundWindow(hwnd)
                    win.pump(rounds=3)
                    await asyncio.sleep(0.2)
                    snap = await call_with_pump(
                        win,
                        client.screen_snapshot(mode="uia_ocr", capture_screenshot=True),
                    )
                else:
                    snap = await call_with_pump(
                        win,
                        client.screen_snapshot(
                            mode="uia_ocr", capture_screenshot=True, window_handle=hwnd
                        ),
                    )
                dump = snap.model_dump()
                rec["capture_origin"] = dump.get("capture_origin")
                rec["perception_mode"] = dump.get("perception_mode")
                rec["screenshot_path"] = dump.get("screenshot_path")

                block = find_target_block(dump.get("text_blocks", []))
                if block is None:
                    rec["note"] = (
                        f"OCR 未识别到 {TARGET_TEXT}"
                        f"（text_blocks={len(dump.get('text_blocks', []))} 块）"
                    )
                else:
                    rec["target_block"] = {
                        "text": block.get("text"),
                        "confidence": block.get("confidence"),
                        "bbox": block.get("bbox"),
                    }
                    center = bbox_center(block["bbox"])
                    rec["perceived_center"] = center
                    win.clicks.clear()
                    action = await call_with_pump(win, client.click_element(coordinates=center))
                    if not action.success:
                        rec["note"] = f"click_element 失败：{action.error_message}"
                    else:
                        landed = await win.wait_click(timeout_s=3.0)
                        if landed is None:
                            rec["note"] = "点击注入后窗口未收到 <Button-1>（落点在窗口外或被吞）"
                        else:
                            rec["actual_click"] = landed
                            rec["error"] = point_error(center, landed)
            except Exception as exc:  # noqa: BLE001 — PoC 单点失败记录后继续
                logger.error("[%s/%s] 闭环异常：%s", screen_label, pos_label, exc, exc_info=True)
                rec["note"] = f"闭环异常：{exc}"
            logger.info(
                "[%s/%s] 感知中心=%s 落点=%s 误差=%s %s",
                screen_label,
                pos_label,
                rec["perceived_center"],
                rec["actual_click"],
                rec["error"],
                rec["note"] or "",
            )
            records.append(rec)
            await asyncio.sleep(0.2)
    finally:
        win.close()
    return records


async def collect_report(screens: list[str]) -> dict[str, Any]:
    """跑指定屏幕清单的全链路回环，返回完整 report（需真实桌面 + flag）。"""
    ensure_capability_enabled()
    dpi_mode = set_dpi_awareness()  # 必须先于 tkinter 窗口创建
    monitors = list_monitor_rects()
    logger.info("DPI 感知=%s；显示器：%s", dpi_mode, monitors)

    from src.mcp.desktop_mcp_client import DesktopMCPClient

    report: dict[str, Any] = {
        "meta": {
            "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dpi_awareness": dpi_mode,
            "monitors": monitors,
            "tolerance_px": ERROR_TOLERANCE_PX,
            "target_text": TARGET_TEXT,
            "window_size": (WINDOW_W, WINDOW_H),
        },
        "screens": {},
    }
    async with DesktopMCPClient() as client:
        for which in screens:
            rect = pick_screen_rect(monitors, which)
            if rect is None:
                report["screens"][which] = {"skipped": "该屏不存在（单显示器环境）"}
                continue
            # 双感知路径各跑一遍：PrintWindow 定向（生产优先路径）+
            # 前台 mss 全虚拟屏（Task 13 修复的新数学只有此路径实机可验）
            for via in ("window_handle", "foreground"):
                records = await run_roundtrip_on_screen(client, which, rect, perceive_via=via)
                report["screens"][f"{which}/{via}"] = {
                    "screen_rect": rect,
                    "perceive_via": via,
                    "records": records,
                    "summary": summarize_errors(records),
                }
    return report


# ══════════════════════════════════════════════════════════════════════════════
# realenv 用例（实机手动：设 flag 后 pytest -m realenv 本文件）
# ══════════════════════════════════════════════════════════════════════════════

_REALENV_READY = os.environ.get("SCREEN_CAPABILITY_ENABLED", "").lower() in {"1", "true", "yes"}
_REALENV_SKIP_REASON = "SCREEN_CAPABILITY_ENABLED 未设置——实机手动跑（见模块 docstring）"


@pytest.mark.realenv
@pytest.mark.skipif(not _REALENV_READY, reason=_REALENV_SKIP_REASON)
async def test_dpi_click_roundtrip_primary_realenv() -> None:
    """gap#3 验收（主屏）：双感知路径 × 3 摆位闭环，max 欧氏误差 < 2px。"""
    report = await collect_report(["primary"])
    print(render_markdown(report))
    for key, data in report["screens"].items():
        summary = data["summary"]
        assert summary["overall_pass"], (
            f"{key} 坐标一致性未达验收（<{ERROR_TOLERANCE_PX}px）：{summary}——"
            "请把逐点偏差记入 notes/ 报告（gap#3 允许「文档化偏差」路径）"
        )


@pytest.mark.realenv
@pytest.mark.skipif(not _REALENV_READY, reason=_REALENV_SKIP_REASON)
async def test_dpi_click_roundtrip_secondary_realenv() -> None:
    """gap#3 验收（副屏）：同主屏口径；单显示器环境 skip。"""
    set_dpi_awareness()
    if pick_screen_rect(list_monitor_rects(), "secondary") is None:
        pytest.skip("单显示器环境，无副屏")
    report = await collect_report(["secondary"])
    print(render_markdown(report))
    for key, data in report["screens"].items():
        summary = data["summary"]
        assert summary["overall_pass"], (
            f"{key} 坐标一致性未达验收（<{ERROR_TOLERANCE_PX}px）：{summary}——"
            "请把逐点偏差记入 notes/ 报告（gap#3 允许「文档化偏差」路径）"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    # Windows 控制台默认 GBK，✅/❌ 等字符会崩；统一 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screen",
        choices=["primary", "secondary", "both"],
        default="both",
        help="窗口摆到哪块屏（副屏摆位自动按其 rect 计算，如 +2200+300 一类坐标）",
    )
    args = parser.parse_args()
    screens = ["primary", "secondary"] if args.screen == "both" else [args.screen]

    report = asyncio.run(collect_report(screens))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"dpi_coords_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(render_markdown(report))
    print(f"\n[raw json] {out_path}")


if __name__ == "__main__":
    main()
