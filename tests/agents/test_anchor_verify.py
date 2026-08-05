"""像素锚点验证单测（K7批1，合成 ScreenSnapshot，纯离线）。

覆盖：
  1. 命中：整块精确 / 整行 OCR 块内子串 / 多锚点任一命中。
  2. 未命中：候选块内无锚点，reason 带 [desk:anchor_miss] 与最接近对诊断。
  3. region 过滤：区内命中（含部分重叠）/ 区外匹配块被排除 / 区内无任何块 /
     边线相触不算相交（严格正面积重叠钉死）。
  4. NFKC 归一：全半角折叠命中；空白差异命中（与 NFKC 独立）。
  5. 编辑距离容错：阈内 OCR 近似命中 / 超阈拒绝 / 短锚点零容错 /
     比例常量归零后容错翻红（阈值确由 ANCHOR_EDIT_DISTANCE_RATIO 驱动）。
  6. 短路：desktop_locked（有匹配块也不命中）/ 空 text_blocks / 空(全空白)锚点。
  7. 机读令牌：[desk:<code>] 用 re.search 位置无关提取（含外层加壳前缀形态）。

判别力实证（绿灯先证能红，逐变异跑过，见交付报告）：
  - _normalize_for_match 去掉 NFKC 归一 → 全半角用例红；
  - _bboxes_intersect 改恒 True → region 排除用例红（区外匹配块混入候选）；
  - verify_pixel_anchor 去掉 desktop_locked 短路 → 锁屏用例红。
"""

from __future__ import annotations

import re

import pytest

from src.agents import anchor_verify
from src.agents.anchor_verify import verify_pixel_anchor
from src.agents.models.screen_snapshot import BBox, ScreenSnapshot, TextBlock

# ── 测试辅助 ──────────────────────────────────────────────────────────────────


def _bbox(x: int = 0, y: int = 0, width: int = 100, height: int = 30) -> BBox:
    """构造测试用 BBox（物理像素口径）。"""
    return BBox(x=x, y=y, width=width, height=height)


def _block(block_id: str, text: str, bbox: BBox | None = None) -> TextBlock:
    """构造测试用 OCR 文本块。"""
    return TextBlock(
        block_id=block_id,
        text=text,
        bbox=bbox if bbox is not None else _bbox(),
        confidence=0.92,
        source="ocr_rapidocr",
    )


def _snapshot(blocks: list[TextBlock], desktop_locked: bool = False) -> ScreenSnapshot:
    """构造测试用 ScreenSnapshot（只填锚点验证读取的字段，其余取最小合法值）。"""
    return ScreenSnapshot(
        snapshot_id="snap-anchor-test",
        timestamp_ms=1_722_400_000_000,
        screen_width=2560,
        screen_height=1440,
        active_window_title="钉钉",
        uia_elements=[],
        text_blocks=blocks,
        visual_objects=[],
        screenshot_path=None,
        perception_mode="uia_ocr",
        capability_flags={"ocr": True},
        desktop_locked=desktop_locked,
    )


def _extract_code(reason: str) -> str | None:
    """按消费侧口径提取机读令牌（位置无关 re.search，mcp-integration 纪律）。"""
    match = re.search(r"\[desk:([a-z_]+)\]", reason)
    return match.group(1) if match else None


# ── 1. 命中 ──────────────────────────────────────────────────────────────────


class TestAnchorHit:
    """锚点命中：精确块 / 整行子串 / 多锚点任一命中。"""

    def test_exact_block_hit(self) -> None:
        """锚点与文本块逐字相同 → 命中，score=1.0，matched_* 指向命中块。"""
        snap = _snapshot([_block("blk-1", "消息"), _block("blk-2", "通讯录")])

        verdict = verify_pixel_anchor(snap, ["消息"])

        assert verdict.anchor_hit is True
        assert verdict.matched_block_id == "blk-1"
        assert verdict.matched_text == "消息"
        assert verdict.match_score == pytest.approx(1.0)
        assert _extract_code(verdict.reason) == "anchor_hit"

    def test_substring_in_ocr_line_hit(self) -> None:
        """整行 OCR 块内含锚点（子串）→ 命中且 score=1.0（Sellers 子串距离 0）。"""
        snap = _snapshot([_block("blk-line", "消息 通讯录 工作台")])

        verdict = verify_pixel_anchor(snap, ["通讯录"])

        assert verdict.anchor_hit is True
        assert verdict.matched_block_id == "blk-line"
        assert verdict.match_score == pytest.approx(1.0)

    def test_any_anchor_hits(self) -> None:
        """多锚点任一命中即 anchor_hit；reason 标注命中的那个锚点。"""
        snap = _snapshot([_block("blk-1", "消息")])

        verdict = verify_pixel_anchor(snap, ["不存在的字样", "消息"])

        assert verdict.anchor_hit is True
        assert "'消息'" in verdict.reason


# ── 2. 未命中 ────────────────────────────────────────────────────────────────


class TestAnchorMiss:
    """候选块内无锚点 → anchor_miss，matched_* 为 None，附最接近对诊断。"""

    def test_miss_reports_closest_pair(self) -> None:
        snap = _snapshot([_block("blk-1", "消息")])

        verdict = verify_pixel_anchor(snap, ["通讯录"])

        assert verdict.anchor_hit is False
        assert verdict.matched_text is None
        assert verdict.matched_block_id is None
        assert 0.0 <= verdict.match_score < 1.0
        assert _extract_code(verdict.reason) == "anchor_miss"
        # 诊断信息：最接近对（锚点与块）在散文里可见
        assert "blk-1" in verdict.reason


# ── 3. region 过滤 ───────────────────────────────────────────────────────────


class TestRegionFilter:
    """region 非 None 时仅统计与之相交（严格正面积重叠）的 text_blocks。"""

    def test_region_hit_with_partial_overlap(self) -> None:
        """region 与块部分重叠即算相交；命中的必须是区内块。"""
        inside = _block("blk-in", "消息", _bbox(x=10, y=10, width=80, height=24))
        outside = _block("blk-out", "工作台", _bbox(x=500, y=10, width=80, height=24))
        snap = _snapshot([outside, inside])

        # region 只盖住 inside 块的左上角一角（部分重叠）
        verdict = verify_pixel_anchor(snap, ["消息"], region=_bbox(x=0, y=0, width=50, height=20))

        assert verdict.anchor_hit is True
        assert verdict.matched_block_id == "blk-in"

    def test_region_excludes_outside_match(self) -> None:
        """锚点文本只出现在区外块 → 区内候选无命中，anchor_miss。

        判别力：_bboxes_intersect 改恒 True 时区外匹配块混入候选 → 假命中，
        本例即红（region 相交判定被改坏的哨兵）。
        """
        outside_match = _block("blk-out", "消息", _bbox(x=500, y=10, width=80, height=24))
        inside_other = _block("blk-in", "工作台", _bbox(x=10, y=10, width=80, height=24))
        snap = _snapshot([outside_match, inside_other])

        verdict = verify_pixel_anchor(snap, ["消息"], region=_bbox(x=0, y=0, width=200, height=600))

        assert verdict.anchor_hit is False
        assert _extract_code(verdict.reason) == "anchor_miss"

    def test_region_with_no_blocks_at_all(self) -> None:
        """region 落在无任何文本块的区域 → region_no_text_blocks 短路。"""
        snap = _snapshot([_block("blk-1", "消息", _bbox(x=10, y=10, width=80, height=24))])

        verdict = verify_pixel_anchor(
            snap, ["消息"], region=_bbox(x=1000, y=1000, width=50, height=50)
        )

        assert verdict.anchor_hit is False
        assert verdict.match_score == 0.0
        assert _extract_code(verdict.reason) == "region_no_text_blocks"

    def test_edge_touching_is_not_intersecting(self) -> None:
        """仅边线相触（region.x == block 右边界）不算相交——严格正面积重叠钉死。"""
        block = _block("blk-1", "消息", _bbox(x=10, y=10, width=90, height=24))  # x ∈ [10,100)
        snap = _snapshot([block])

        verdict = verify_pixel_anchor(snap, ["消息"], region=_bbox(x=100, y=0, width=50, height=50))

        assert verdict.anchor_hit is False
        assert _extract_code(verdict.reason) == "region_no_text_blocks"


# ── 4. NFKC 归一与空白差异 ───────────────────────────────────────────────────


class TestNormalization:
    """OCR 常见近似：全半角（NFKC）与空白差异分别独立验证。"""

    def test_fullwidth_halfwidth_hit(self) -> None:
        """块文本为全角拉丁 → NFKC 折叠后与半角锚点距离 0 命中。

        判别力：_normalize_for_match 去掉 NFKC 归一后全角字符不折叠、
        8 字符全不同（casefold 不做全半角转换），超阈必红。
        """
        snap = _snapshot([_block("blk-fw", "ＤｉｎｇＴａｌｋ")])

        verdict = verify_pixel_anchor(snap, ["DingTalk"])

        assert verdict.anchor_hit is True
        assert verdict.match_score == pytest.approx(1.0)
        # matched_text 是未归一的 OCR 原文
        assert verdict.matched_text == "ＤｉｎｇＴａｌｋ"

    def test_whitespace_difference_hit(self) -> None:
        """OCR 把字间距识别成空格（「文 件 传 输」）→ 去空白归一后命中。"""
        snap = _snapshot([_block("blk-ws", "文 件 传 输")])

        verdict = verify_pixel_anchor(snap, ["文件传输"])

        assert verdict.anchor_hit is True
        assert verdict.match_score == pytest.approx(1.0)


# ── 5. 编辑距离容错 ──────────────────────────────────────────────────────────


class TestEditDistanceTolerance:
    """阈值 = int(len(norm_anchor) * ANCHOR_EDIT_DISTANCE_RATIO)，默认 ratio=0.34。"""

    def test_ocr_near_miss_within_threshold(self) -> None:
        """6 字锚点 1 处误识（输→榆）→ 距离 1 ≤ 阈值 2，命中，score=1-1/6。"""
        snap = _snapshot([_block("blk-1", "文件传榆助手")])

        verdict = verify_pixel_anchor(snap, ["文件传输助手"])

        assert verdict.anchor_hit is True
        assert verdict.match_score == pytest.approx(1.0 - 1.0 / 6.0)
        assert _extract_code(verdict.reason) == "anchor_hit"

    def test_over_threshold_rejected(self) -> None:
        """4 字锚点 2 处误识 → 距离 2 > 阈值 1，拒绝，score=0.5。"""
        snap = _snapshot([_block("blk-1", "消患申心")])

        verdict = verify_pixel_anchor(snap, ["消息中心"])

        assert verdict.anchor_hit is False
        assert verdict.match_score == pytest.approx(0.5)
        assert _extract_code(verdict.reason) == "anchor_miss"

    def test_short_anchor_zero_tolerance(self) -> None:
        """2 字锚点阈值为 0（防「消息/消费」单字替换假命中）→ 1 处不同即拒绝。"""
        snap = _snapshot([_block("blk-1", "消费")])

        verdict = verify_pixel_anchor(snap, ["消息"])

        assert verdict.anchor_hit is False

    def test_ratio_zero_disables_tolerance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """比例常量归零 → 阈内近似用例翻红（阈值确由 ANCHOR_EDIT_DISTANCE_RATIO 驱动，
        env 同名覆盖走的正是这个模块常量）。"""
        monkeypatch.setattr(anchor_verify, "ANCHOR_EDIT_DISTANCE_RATIO", 0.0)
        snap = _snapshot([_block("blk-1", "文件传榆助手")])

        verdict = verify_pixel_anchor(snap, ["文件传输助手"])

        assert verdict.anchor_hit is False
        assert _extract_code(verdict.reason) == "anchor_miss"


# ── 6. 短路路径 ──────────────────────────────────────────────────────────────


class TestShortCircuits:
    """锁屏 / 无文本块 / 无有效锚点：anchor_hit=False 且 reason 写明原因。"""

    def test_desktop_locked_short_circuits_even_with_match(self) -> None:
        """锁屏下即使存在逐字匹配块也不命中——锁屏像素是旧帧/黑屏，不可信。

        判别力：verify_pixel_anchor 去掉 desktop_locked 短路后本例假命中即红。
        """
        snap = _snapshot([_block("blk-1", "消息")], desktop_locked=True)

        verdict = verify_pixel_anchor(snap, ["消息"])

        assert verdict.anchor_hit is False
        assert verdict.matched_text is None
        assert verdict.matched_block_id is None
        assert verdict.match_score == 0.0
        assert _extract_code(verdict.reason) == "desktop_locked"
        assert "锁" in verdict.reason  # 人读散文写明锁屏原因

    def test_empty_text_blocks(self) -> None:
        snap = _snapshot([])

        verdict = verify_pixel_anchor(snap, ["消息"])

        assert verdict.anchor_hit is False
        assert verdict.match_score == 0.0
        assert _extract_code(verdict.reason) == "no_text_blocks"
        assert "无" in verdict.reason  # 人读散文写明无文本块

    def test_empty_anchor_list(self) -> None:
        snap = _snapshot([_block("blk-1", "消息")])

        verdict = verify_pixel_anchor(snap, [])

        assert verdict.anchor_hit is False
        assert _extract_code(verdict.reason) == "no_anchor_texts"

    def test_whitespace_only_anchors(self) -> None:
        """全空白锚点归一后为空——空模式对一切文本距离 0，必须按无锚点短路。"""
        snap = _snapshot([_block("blk-1", "消息")])

        verdict = verify_pixel_anchor(snap, ["   ", ""])

        assert verdict.anchor_hit is False
        assert _extract_code(verdict.reason) == "no_anchor_texts"


# ── 7. 机读令牌形态 ──────────────────────────────────────────────────────────


class TestMachineToken:
    """[desk:<code>] 位置无关：外层加壳（如 FastMCP 前缀）后仍可 re.search 提取。"""

    def test_token_survives_wrapping_prefix(self) -> None:
        """模拟消费侧拿到加壳文案（前缀散文）——令牌提取不依赖位置 0。"""
        snap = _snapshot([], desktop_locked=True)
        verdict = verify_pixel_anchor(snap, ["消息"])

        wrapped = f"Error executing tool verify_anchor: {verdict.reason}"

        assert _extract_code(wrapped) == "desktop_locked"

    def test_every_terminal_path_carries_exactly_one_token(self) -> None:
        """六条出口路径的 reason 各携带恰一枚 [desk:<code>] 令牌，码值在已定集合内。"""
        known_codes = {
            "anchor_hit",
            "anchor_miss",
            "desktop_locked",
            "no_text_blocks",
            "region_no_text_blocks",
            "no_anchor_texts",
        }
        cases = [
            verify_pixel_anchor(_snapshot([_block("b", "消息")]), ["消息"]),  # hit
            verify_pixel_anchor(_snapshot([_block("b", "消息")]), ["通讯录"]),  # miss
            verify_pixel_anchor(_snapshot([], desktop_locked=True), ["消息"]),  # locked
            verify_pixel_anchor(_snapshot([]), ["消息"]),  # no blocks
            verify_pixel_anchor(  # region 内无块
                _snapshot([_block("b", "消息", _bbox(x=0, y=0, width=10, height=10))]),
                ["消息"],
                region=_bbox(x=500, y=500, width=10, height=10),
            ),
            verify_pixel_anchor(_snapshot([_block("b", "消息")]), []),  # no anchors
        ]
        for verdict in cases:
            tokens = re.findall(r"\[desk:([a-z_]+)\]", verdict.reason)
            assert len(tokens) == 1, f"reason 应恰含一枚令牌: {verdict.reason!r}"
            assert tokens[0] in known_codes
