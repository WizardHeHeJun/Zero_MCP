"""蓝图 §10 Task 1：国产中文软件 UIA 覆盖率 PoC。

遍历微信/钉钉主窗口 UIA 树，量化三项指标（口径见 notes/poc-uia-coverage-result.md）：
1. 有效 name/automation_id 控件比例（全部 / 叶子 / 可交互三种分母）
2. 重点功能区覆盖（导航/搜索/会话列表/输入框等能否经 UIA 定位）
3. DirectUI 自绘盲区占比（窗口面积中无任何 UIA 叶子元素覆盖的栅格比例）

用法（先 conda activate affective-expression，见 CLAUDE.md）：
    python tests/poc/test_uia_coverage.py --app weixin
    python tests/poc/test_uia_coverage.py --app dingtalk
    python tests/poc/test_uia_coverage.py --hwnd 0x00012345 --label custom

产出：stdout 打 markdown 摘要；全量原始数据落 tests/poc/output/uia_coverage_<label>_<时间戳>.json。

本文件是可独立运行的 PoC 采集脚本，无 pytest 用例；纯分析函数（analyze_*）
不碰 COM，后续可直接被单测复用。
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wintypes
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger("uia_coverage_poc")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "tests" / "poc" / "output"

# 可交互控件类型（uiautomation 的 ControlTypeName 口径）
INTERACTIVE_TYPES = {
    "ButtonControl",
    "EditControl",
    "ComboBoxControl",
    "CheckBoxControl",
    "RadioButtonControl",
    "ListItemControl",
    "TreeItemControl",
    "TabItemControl",
    "MenuItemControl",
    "HyperlinkControl",
    "SplitButtonControl",
    "SliderControl",
}

# 目标应用：进程名 → 重点功能区匹配规则
APP_PROFILES: dict[str, dict] = {
    "weixin": {
        "process_names": {"weixin.exe", "wechat.exe"},
        "display": "微信",
        "main_title_re": r"^(微信|WeChat)$",
        "key_regions": [
            # (区域名, 允许的控件类型集合或 None=不限, name 正则)
            (
                "导航-聊天/微信",
                {
                    "ButtonControl",
                    "TabItemControl",
                    "ListItemControl",
                    "RadioButtonControl",
                    "TextControl",
                },
                r"^(微信|聊天)$",
            ),
            ("导航-通讯录", None, r"^通讯录$"),
            ("搜索框", {"EditControl"}, r"搜索"),
            ("发送按钮", {"ButtonControl"}, r"^发送"),
            ("消息区标识", None, r"(消息|聊天记录|会话)"),
        ],
    },
    "dingtalk": {
        "process_names": {"dingtalk.exe"},
        "display": "钉钉",
        "main_title_re": r"^(钉钉|DingTalk)$",
        "key_regions": [
            ("导航-消息", None, r"^(消息|钉钉)$"),
            ("导航-通讯录", None, r"^通讯录$"),
            ("搜索框", {"EditControl"}, r"搜索"),
            ("发送按钮", {"ButtonControl"}, r"^发送"),
            ("消息区标识", None, r"(消息|聊天记录|会话)"),
        ],
    },
}


@dataclass
class ElemRecord:
    """一条 UIA 元素记录（COM 属性只在采集时读一次，之后纯数据）。"""

    idx: int
    parent_idx: int
    depth: int
    control_type: str
    name: str
    automation_id: str
    class_name: str
    rect: tuple[int, int, int, int]  # left, top, right, bottom（物理像素）
    is_offscreen: bool
    is_enabled: bool
    children_count: int = 0

    @property
    def area(self) -> int:
        w = max(0, self.rect[2] - self.rect[0])
        h = max(0, self.rect[3] - self.rect[1])
        return w * h

    def has_name(self) -> bool:
        return bool(self.name.strip())

    def has_automation_id(self) -> bool:
        return bool(self.automation_id.strip())


@dataclass
class TraversalResult:
    hwnd: int
    window_title: str
    window_class: str
    window_rect: tuple[int, int, int, int]
    elements: list[ElemRecord] = field(default_factory=list)
    truncated_reason: str | None = None
    elapsed_s: float = 0.0


# ---------------------------------------------------------------- Win32 辅助


def set_dpi_awareness() -> str:
    """在导入 uiautomation 前设 DPI 感知，保证 UIA rect 是物理像素（蓝图工程假设）。"""
    user32 = ctypes.windll.user32
    try:
        # PER_MONITOR_AWARE_V2 = -4
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


def list_top_windows_of_processes(process_names: set[str]) -> list[dict]:
    """EnumWindows 找目标进程的顶层窗口（含不可见的托盘态窗口）。"""
    psapi = ctypes.windll.kernel32
    user32 = ctypes.windll.user32
    results: list[dict] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def enum_cb(hwnd: int, _lparam: int) -> bool:
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        h_process = psapi.OpenProcess(0x1000, False, pid.value)  # QUERY_LIMITED_INFORMATION
        if not h_process:
            return True
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(1024)
            if not psapi.QueryFullProcessImageNameW(h_process, 0, buf, ctypes.byref(size)):
                return True
            exe_name = Path(buf.value).name.lower()
        finally:
            psapi.CloseHandle(h_process)
        if exe_name not in process_names:
            return True

        title_buf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, title_buf, 512)
        class_buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_buf, 256)
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        results.append(
            {
                "hwnd": hwnd,
                "pid": pid.value,
                "title": title_buf.value,
                "class_name": class_buf.value,
                "visible": bool(user32.IsWindowVisible(hwnd)),
                "rect": (rect.left, rect.top, rect.right, rect.bottom),
                "area": max(0, rect.right - rect.left) * max(0, rect.bottom - rect.top),
            }
        )
        return True

    user32.EnumWindows(enum_cb, 0)
    return results


# 非主窗口类名特征（托盘/IME/系统消息/工具提示窗），实测见 notes 报告「窗口发现」节
NON_MAIN_CLASS_RE = re.compile(r"TrayIcon|MessageWindow|IME|Tooltip|ToolSaveBits", re.IGNORECASE)


def pick_main_window(candidates: list[dict], main_title_re: str | None = None) -> dict | None:
    """主窗口启发式：排类名黑名单 → 标题匹配应用名 > 可见 > 面积大。

    注意：微信 4.x 托盘态下主窗口 IsWindowVisible=False，而托盘消息窗反而
    visible=True 且面积更大，不能只按可见性/面积挑。
    """
    filtered = [w for w in candidates if not NON_MAIN_CLASS_RE.search(w["class_name"])]
    pool = filtered or candidates
    title_re = re.compile(main_title_re) if main_title_re else None
    return max(
        pool,
        key=lambda w: (
            bool(title_re and title_re.search(w["title"])),
            w["visible"],
            w["area"],
        ),
        default=None,
    )


def bring_to_front(hwnd: int) -> None:
    """托盘/最小化窗口先恢复再前置，保证 UIA rect 与栅格统计有效。"""
    user32 = ctypes.windll.user32
    SW_RESTORE = 9
    if user32.IsIconic(hwnd) or not user32.IsWindowVisible(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    user32.SetForegroundWindow(hwnd)
    time.sleep(1.0)  # 等 DirectUI 首帧渲染完成


# ---------------------------------------------------------------- UIA 采集


def collect_tree(
    hwnd: int, max_elements: int, max_depth: int, time_budget_s: float
) -> TraversalResult:
    """DFS 遍历 UIA 树；每个元素的 COM 属性各读一次，失败单点跳过不中断。"""
    import uiautomation as auto  # 延迟导入：DPI 感知须在其加载前设好

    user32 = ctypes.windll.user32
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))

    root = auto.ControlFromHandle(hwnd)
    result = TraversalResult(
        hwnd=hwnd,
        window_title=root.Name or "",
        window_class=root.ClassName or "",
        window_rect=(rect.left, rect.top, rect.right, rect.bottom),
    )

    started = time.monotonic()
    stack: list[tuple[object, int, int]] = [(root, -1, 0)]  # (control, parent_idx, depth)
    while stack:
        if len(result.elements) >= max_elements:
            result.truncated_reason = f"element cap {max_elements}"
            break
        if time.monotonic() - started > time_budget_s:
            result.truncated_reason = f"time budget {time_budget_s}s"
            break
        control, parent_idx, depth = stack.pop()
        try:
            r = control.BoundingRectangle
            record = ElemRecord(
                idx=len(result.elements),
                parent_idx=parent_idx,
                depth=depth,
                control_type=control.ControlTypeName,
                name=(control.Name or "")[:120].replace("\n", "\\n"),
                automation_id=(control.AutomationId or "")[:120],
                class_name=(control.ClassName or "")[:120],
                rect=(r.left, r.top, r.right, r.bottom),
                is_offscreen=bool(control.IsOffscreen),
                is_enabled=bool(control.IsEnabled),
            )
        except Exception as exc:  # noqa: BLE001 — COM 单点失败只跳过该元素
            logger.debug("元素属性读取失败（跳过）：%s", exc)
            continue
        result.elements.append(record)
        if depth >= max_depth:
            continue
        try:
            children = control.GetChildren()
        except Exception as exc:  # noqa: BLE001
            logger.debug("GetChildren 失败（idx=%d）：%s", record.idx, exc)
            continue
        record.children_count = len(children)
        for child in reversed(children):
            stack.append((child, record.idx, depth + 1))

    result.elapsed_s = round(time.monotonic() - started, 2)
    return result


# ---------------------------------------------------------------- 纯分析（可单测）


def analyze_identifier_rates(elements: list[ElemRecord]) -> dict:
    """指标 1：有效 name/automation_id 比例，三种分母。"""

    def rates(subset: list[ElemRecord]) -> dict:
        n = len(subset)
        if n == 0:
            return {"count": 0, "name_pct": 0.0, "automation_id_pct": 0.0, "either_pct": 0.0}
        return {
            "count": n,
            "name_pct": round(100 * sum(e.has_name() for e in subset) / n, 1),
            "automation_id_pct": round(100 * sum(e.has_automation_id() for e in subset) / n, 1),
            "either_pct": round(
                100 * sum(e.has_name() or e.has_automation_id() for e in subset) / n, 1
            ),
        }

    leaves = [e for e in elements if e.children_count == 0]
    interactive = [e for e in elements if e.control_type in INTERACTIVE_TYPES]
    return {
        "all": rates(elements),
        "leaves": rates(leaves),
        "interactive": rates(interactive),
    }


def analyze_key_regions(elements: list[ElemRecord], profile: dict) -> list[dict]:
    """指标 2：重点功能区能否经 UIA 定位（规则匹配 + 结构启发式）。"""
    findings: list[dict] = []
    for region_name, allowed_types, name_pattern in profile["key_regions"]:
        matched = [
            e
            for e in elements
            if (allowed_types is None or e.control_type in allowed_types)
            and re.search(name_pattern, e.name)
            and e.area > 0
        ]
        best = max(matched, key=lambda e: e.area, default=None)
        findings.append(
            {
                "region": region_name,
                "found": best is not None,
                "match_count": len(matched),
                "evidence": (
                    {
                        "control_type": best.control_type,
                        "name": best.name[:60],
                        "automation_id": best.automation_id[:60],
                        "rect": best.rect,
                    }
                    if best
                    else None
                ),
            }
        )

    # 结构启发式：会话列表 = ListItem 子元素最多的 List
    lists = [e for e in elements if e.control_type == "ListControl"]
    by_parent: dict[int, int] = {}
    for e in elements:
        if e.control_type == "ListItemControl":
            by_parent[e.parent_idx] = by_parent.get(e.parent_idx, 0) + 1
    best_list = max(lists, key=lambda e: by_parent.get(e.idx, 0), default=None)
    item_count = by_parent.get(best_list.idx, 0) if best_list else 0
    findings.append(
        {
            "region": "会话/消息列表（结构启发式）",
            "found": item_count >= 3,
            "match_count": item_count,
            "evidence": (
                {
                    "control_type": best_list.control_type,
                    "name": best_list.name[:60],
                    "automation_id": best_list.automation_id[:60],
                    "rect": best_list.rect,
                    "list_item_children": item_count,
                }
                if best_list
                else None
            ),
        }
    )

    # 结构启发式：输入框 = 窗口下半部、宽度≥25% 窗宽的 Edit
    return findings


def find_input_box(elements: list[ElemRecord], window_rect: tuple[int, int, int, int]) -> dict:
    win_w = window_rect[2] - window_rect[0]
    win_mid_y = (window_rect[1] + window_rect[3]) / 2
    edits = [
        e
        for e in elements
        if e.control_type == "EditControl"
        and e.area > 0
        and not e.is_offscreen
        and e.rect[1] > win_mid_y
        and (e.rect[2] - e.rect[0]) >= 0.25 * win_w
    ]
    best = max(edits, key=lambda e: e.area, default=None)
    return {
        "region": "消息输入框（结构启发式）",
        "found": best is not None,
        "match_count": len(edits),
        "evidence": (
            {
                "control_type": best.control_type,
                "name": best.name[:60],
                "automation_id": best.automation_id[:60],
                "rect": best.rect,
            }
            if best
            else None
        ),
    }


def analyze_blind_area(
    elements: list[ElemRecord],
    window_rect: tuple[int, int, int, int],
    cell: int = 8,
    sectors: tuple[int, int] = (3, 2),
) -> dict:
    """指标 3：栅格法盲区占比。

    盲区 = 窗口内没有任何「可见非零叶子元素」覆盖的栅格；另算一版只认
    「有 name/automation_id 的叶子」的口径（对 LLM 定位更真实）。

    渲染表面排除：占窗口面积 ≥50% 的 Pane/Window 叶子视为自绘渲染表面
    （如微信 4.x 的 MMUIRenderSubWindowHW）——它对元素定位零贡献，
    计入覆盖会把 100% 盲区伪装成 ~0%。被排除的表面单独列出。
    """
    wl, wt, wr, wb = window_rect
    win_w, win_h = wr - wl, wb - wt
    if win_w <= 0 or win_h <= 0:
        return {"error": "窗口 rect 无效"}
    win_area = win_w * win_h
    gw, gh = (win_w + cell - 1) // cell, (win_h + cell - 1) // cell
    grid_any = bytearray(gw * gh)
    grid_identified = bytearray(gw * gh)

    def mark(grid: bytearray, e: ElemRecord) -> None:
        left = max(e.rect[0], wl) - wl
        top = max(e.rect[1], wt) - wt
        right = min(e.rect[2], wr) - wl
        bottom = min(e.rect[3], wb) - wt
        if right <= left or bottom <= top:
            return
        for gy in range(top // cell, min((bottom - 1) // cell + 1, gh)):
            row = gy * gw
            for gx in range(left // cell, min((right - 1) // cell + 1, gw)):
                grid[row + gx] = 1

    all_leaves = [
        e for e in elements if e.children_count == 0 and e.area > 0 and not e.is_offscreen
    ]
    render_surfaces = [
        e
        for e in all_leaves
        if e.control_type in {"PaneControl", "WindowControl"} and e.area >= 0.5 * win_area
    ]
    leaves = [e for e in all_leaves if e not in render_surfaces]
    for e in leaves:
        mark(grid_any, e)
        if e.has_name() or e.has_automation_id():
            mark(grid_identified, e)

    total = gw * gh
    covered_any = sum(grid_any)
    covered_id = sum(grid_identified)

    # 分扇区盲区（3 列 x 2 行），看盲区集中在哪个功能带
    sx, sy = sectors
    sector_stats: list[dict] = []
    for row_i in range(sy):
        for col_i in range(sx):
            x0, x1 = col_i * gw // sx, (col_i + 1) * gw // sx
            y0, y1 = row_i * gh // sy, (row_i + 1) * gh // sy
            cells = [(gy * gw + gx) for gy in range(y0, y1) for gx in range(x0, x1)]
            n = len(cells)
            blind = sum(1 for c in cells if not grid_any[c])
            sector_stats.append(
                {
                    "sector": f"行{row_i + 1}列{col_i + 1}",
                    "blind_pct": round(100 * blind / n, 1) if n else 0.0,
                }
            )

    return {
        "cell_px": cell,
        "grid": f"{gw}x{gh}",
        "blind_pct_any_leaf": round(100 * (total - covered_any) / total, 1),
        "blind_pct_identified_leaf": round(100 * (total - covered_id) / total, 1),
        "leaf_count_used": len(leaves),
        "render_surfaces_excluded": [
            {
                "control_type": e.control_type,
                "name": e.name[:60],
                "class_name": e.class_name[:60],
                "rect": e.rect,
                "area_pct_of_window": round(100 * e.area / win_area, 1),
            }
            for e in render_surfaces
        ],
        "sectors": sector_stats,
    }


def analyze_control_type_distribution(elements: list[ElemRecord], top_n: int = 15) -> list[dict]:
    counts: dict[str, int] = {}
    for e in elements:
        counts[e.control_type] = counts.get(e.control_type, 0) + 1
    return [
        {"control_type": k, "count": v}
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]
    ]


# ---------------------------------------------------------------- 汇总输出


def render_markdown(report: dict) -> str:
    """输出可直接贴进 notes 报告的 markdown 摘要。"""
    lines: list[str] = []
    meta = report["meta"]
    lines.append(f"### {meta['app_display']}（{meta['window_title'] or '无标题'}）")
    lines.append("")
    lines.append(
        f"- 窗口：hwnd={meta['hwnd']:#x} class=`{meta['window_class']}` "
        f"rect={meta['window_rect']} DPI 感知={meta['dpi_awareness']}"
    )
    lines.append(
        f"- 遍历：{meta['element_count']} 元素 / 最大深度 {meta['max_depth_seen']} / "
        f"{meta['elapsed_s']}s"
        + (f"（**截断：{meta['truncated_reason']}**）" if meta["truncated_reason"] else "")
    )
    lines.append("")
    lines.append("| 分母 | 数量 | name 有效 | automation_id 有效 | 任一有效 |")
    lines.append("|---|---|---|---|---|")
    for label, key in (("全部元素", "all"), ("叶子元素", "leaves"), ("可交互元素", "interactive")):
        r = report["identifier_rates"][key]
        lines.append(
            f"| {label} | {r['count']} | {r['name_pct']}% | "
            f"{r['automation_id_pct']}% | {r['either_pct']}% |"
        )
    lines.append("")
    blind = report["blind_area"]
    lines.append(
        f"- 盲区（任意叶子口径）：**{blind['blind_pct_any_leaf']}%**；"
        f"盲区（有标识叶子口径）：**{blind['blind_pct_identified_leaf']}%**"
        f"（栅格 {blind['grid']}，cell={blind['cell_px']}px）"
    )
    lines.append(
        "- 分扇区盲区：" + "，".join(f"{s['sector']} {s['blind_pct']}%" for s in blind["sectors"])
    )
    for surf in blind.get("render_surfaces_excluded", []):
        lines.append(
            f"- ⚠ 排除自绘渲染表面：`{surf['class_name']}`"
            f"（{surf['area_pct_of_window']}% 窗口面积，不计入覆盖）"
        )
    lines.append("")
    lines.append("| 重点功能区 | 找到 | 匹配数 | 证据 |")
    lines.append("|---|---|---|---|")
    for f in report["key_regions"]:
        ev = f["evidence"]
        ev_str = (
            f"`{ev['control_type']}` name=`{ev['name']}` aid=`{ev['automation_id']}`" if ev else "—"
        )
        lines.append(
            f"| {f['region']} | {'✅' if f['found'] else '❌'} | {f['match_count']} | {ev_str} |"
        )
    lines.append("")
    lines.append(
        "Top 控件类型："
        + "，".join(f"{d['control_type']}×{d['count']}" for d in report["control_types"][:8])
    )
    return "\n".join(lines)


def run(
    app: str | None,
    hwnd_arg: int | None,
    label: str,
    max_elements: int,
    max_depth: int,
    time_budget_s: float,
    cell: int,
) -> dict:
    dpi_mode = set_dpi_awareness()

    if hwnd_arg is not None:
        hwnd = hwnd_arg
        candidates: list[dict] = []
        profile = APP_PROFILES.get(app or "", {"display": label, "key_regions": []})
    else:
        assert app is not None
        profile = APP_PROFILES[app]
        candidates = list_top_windows_of_processes(profile["process_names"])
        main = pick_main_window(candidates, profile.get("main_title_re"))
        if main is None:
            raise SystemExit(f"未找到 {profile['display']} 的顶层窗口——请确认应用已启动并已登录。")
        hwnd = main["hwnd"]
        logger.info(
            "候选窗口 %d 个，选中 hwnd=%#x title=%r class=%r visible=%s",
            len(candidates),
            hwnd,
            main["title"],
            main["class_name"],
            main["visible"],
        )

    bring_to_front(hwnd)
    result = collect_tree(hwnd, max_elements, max_depth, time_budget_s)

    key_regions = analyze_key_regions(result.elements, profile)
    key_regions.append(find_input_box(result.elements, result.window_rect))

    report = {
        "meta": {
            "app_display": profile["display"],
            "label": label,
            "hwnd": result.hwnd,
            "window_title": result.window_title,
            "window_class": result.window_class,
            "window_rect": result.window_rect,
            "dpi_awareness": dpi_mode,
            "element_count": len(result.elements),
            "max_depth_seen": max((e.depth for e in result.elements), default=0),
            "elapsed_s": result.elapsed_s,
            "truncated_reason": result.truncated_reason,
            "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "candidate_windows": candidates,
        },
        "identifier_rates": analyze_identifier_rates(result.elements),
        "blind_area": analyze_blind_area(result.elements, result.window_rect, cell=cell),
        "key_regions": key_regions,
        "control_types": analyze_control_type_distribution(result.elements),
        "elements": [asdict(e) for e in result.elements],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"uia_coverage_{label}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("原始数据已写入 %s", out_path)

    print(render_markdown(report))
    print(f"\n[raw json] {out_path}")
    return report


def main() -> None:
    # Windows 控制台默认 GBK，✅/❌ 等字符会崩；统一 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", choices=sorted(APP_PROFILES), help="目标应用")
    parser.add_argument(
        "--hwnd", type=lambda s: int(s, 0), help="直接指定窗口句柄（可 0x 十六进制）"
    )
    parser.add_argument("--label", default=None, help="输出文件标签，默认取 --app")
    parser.add_argument("--max-elements", type=int, default=30000)
    parser.add_argument("--max-depth", type=int, default=60)
    parser.add_argument("--time-budget", type=float, default=180.0)
    parser.add_argument("--cell", type=int, default=8, help="盲区栅格边长（物理像素）")
    args = parser.parse_args()

    if args.app is None and args.hwnd is None:
        parser.error("--app 与 --hwnd 至少给一个")
    label = args.label or args.app or f"hwnd_{args.hwnd:#x}"
    run(args.app, args.hwnd, label, args.max_elements, args.max_depth, args.time_budget, args.cell)


if __name__ == "__main__":
    main()
