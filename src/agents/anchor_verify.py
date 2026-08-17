"""像素锚点验证（agents 层共享工具，完全离线纯函数）。

「所见即目标」的**唯一可靠证明**：Win32 窗口状态全线撒谎——前台归属、标题、
可见位、DWM CLOAKED 可以同时"正确"而屏幕像素是另一个应用（假前台态，见
ai-docs/pitfalls.md「Win32 窗口状态全线撒谎，只有像素锚点可信」）。只有快照
OCR 认出目标应用的**已知锚点文本**（如钉钉左导航「消息」）才证明所见即目标。

用途：

- 操控前验证目标应用锚点文本（坐标点击/输入前的「所见即目标」证明）；
- e2e 采集器 ``foreground_ok``（tests/e2e/test_desktop_e2e.py 现基于
  active_window_title 标题子串判据——标题恰是会撒谎的 Win32 状态之一）
  的后续换装目标。

state 信号接线（``target_visible``）归 algo-team 拍板，本批不接。

匹配容错（OCR 常见近似）：

1. NFKC 归一折叠全半角/兼容字符 + 去空白 + casefold——NFKC 先例复用
   text_filter.py 混淆检测层的同一 stdlib 调用（``unicodedata.normalize``，
   该模块未导出可复用函数，此处引同一原语并注明出处）；
2. 编辑距离容错（Sellers 子串近似匹配 DP：锚点对文本块**任意子串**的最小
   编辑距离，整行 OCR 块内含锚点即距离 0），阈值按锚点归一化长度比例
   （``ANCHOR_EDIT_DISTANCE_RATIO``，env 同名可覆盖）。

``reason`` 中机读令牌统一 ``[desk:<code>]`` 形态、位置无关（消费侧用
``re.search`` 提取），与人读散文并存。本模块已定码：

- ``anchor_hit``：锚点命中；
- ``anchor_miss``：候选文本块内无锚点命中（含超编辑距离阈值拒绝）；
- ``desktop_locked``：桌面会话锁定（与 ScreenSnapshot.degradations 同码），
  锁屏下像素/OCR 不可信，验证短路为未命中；
- ``no_text_blocks``：快照无 OCR 文本块；
- ``region_no_text_blocks``：region 过滤后无候选文本块；
- ``no_anchor_texts``：锚点文本列表为空或全为空白。

层约束：只 import 标准库 + 契约模型（src/agents/models，同层共享契约），
不做任何 I/O、不 import 上层模块。
"""

from __future__ import annotations

import logging
import os
import unicodedata

from pydantic import BaseModel, Field

from src.agents.models.screen_snapshot import BBox, ScreenSnapshot, TextBlock

logger = logging.getLogger(__name__)

ANCHOR_EDIT_DISTANCE_RATIO: float = float(os.environ.get("ANCHOR_EDIT_DISTANCE_RATIO", "0.34"))
"""编辑距离容错阈值占锚点归一化长度的比例（工程假设，待真实 OCR 语料标定）。

允许编辑数 = ``int(len(norm_anchor) * ratio)``：默认 0.34 下长度 ≤2 → 0
（短锚点不容错，防「消息/消费」一类单字替换假命中）、3–5 → 1、6–8 → 2。
"""


class AnchorVerdict(BaseModel):
    """像素锚点验证结论（verify_pixel_anchor 的返回契约）。"""

    anchor_hit: bool
    matched_text: str | None  # 命中文本块的 OCR 原文（未归一）；未命中为 None
    matched_block_id: str | None
    # 命中=该对的相似度（1 - 编辑距离/锚点长度）；未命中=最接近对的得分（诊断用）；
    # 锁屏/无文本块/无锚点等短路路径=0.0。
    match_score: float = Field(ge=0.0, le=1.0)
    reason: str  # 机读令牌 [desk:<code>]（位置无关）+ 人读散文


def _normalize_for_match(text: str) -> str:
    """匹配用归一：NFKC（全半角/兼容字符折叠）+ 去全部空白 + casefold。

    NFKC 先例见 text_filter.py 混淆检测层（同一 stdlib 原语）。
    """
    return "".join(unicodedata.normalize("NFKC", text).split()).casefold()


def _bboxes_intersect(a: BBox, b: BBox) -> bool:
    """两 BBox 是否相交（严格正面积重叠；仅边线相触/零面积框不算相交）。"""
    return (
        a.x < b.x + b.width
        and b.x < a.x + a.width
        and a.y < b.y + b.height
        and b.y < a.y + a.height
    )


def _substring_edit_distance(pattern: str, text: str) -> int:
    """pattern 与 text **任意子串**的最小编辑距离（Sellers 近似匹配 DP）。

    首行全 0 = 匹配可从 text 任意位置开始；取末行最小值 = 可在任意位置结束。
    子串包含即距离 0（整行 OCR 块内含锚点的常见形态）。pattern 为空返回 0
    （调用方须先滤掉空锚点，防空模式全命中）；text 为空退化为 len(pattern)。
    """
    if not pattern:
        return 0
    if not text:
        return len(pattern)
    prev = [0] * (len(text) + 1)
    for i, p_ch in enumerate(pattern, start=1):
        curr = [i] + [0] * len(text)
        for j, t_ch in enumerate(text, start=1):
            cost = 0 if p_ch == t_ch else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return min(prev)


def verify_pixel_anchor(
    snapshot: ScreenSnapshot,
    anchor_texts: list[str],
    region: BBox | None = None,
) -> AnchorVerdict:
    """在快照 OCR 文本块上验证锚点文本（「所见即目标」的像素证明，纯函数）。

    Args:
        snapshot: 一次屏幕感知的完整快照（只读 text_blocks / desktop_locked）。
        anchor_texts: 目标应用已知锚点文本列表（任一命中即 anchor_hit），如
            钉钉左导航的「消息」；匹配前做 NFKC 归一 + 去空白 + casefold。
        region: 非 None 时仅统计与之相交（严格正面积重叠）的 text_blocks，
            用于把锚点限定在预期屏幕区域（防其它窗口同名文本假命中）。

    Returns:
        AnchorVerdict。锁屏（desktop_locked）或无（候选）文本块或无有效锚点时
        anchor_hit=False 且 reason 带对应 ``[desk:<code>]`` 令牌写明原因；
        未命中时 reason 附最接近对的编辑距离与阈值（诊断用）。
    """
    verdict = _verify_pixel_anchor(snapshot, anchor_texts, region)
    # 留痕（桌面排障证据先行）：reason 自带 [desk:<code>] 机读令牌，整条直出。
    logger.debug("像素锚点验证：%s", verdict.reason)
    return verdict


def _verify_pixel_anchor(
    snapshot: ScreenSnapshot,
    anchor_texts: list[str],
    region: BBox | None = None,
) -> AnchorVerdict:
    """`verify_pixel_anchor` 的判定主体（结论统一在公开入口留痕）。"""
    if snapshot.desktop_locked:
        return AnchorVerdict(
            anchor_hit=False,
            matched_text=None,
            matched_block_id=None,
            match_score=0.0,
            reason=(
                "[desk:desktop_locked] 桌面会话锁定（OpenInputDesktop 探测），"
                "锁屏下像素/OCR 不可信，锚点验证短路为未命中"
            ),
        )

    # 空锚点滤除：归一后为空的锚点会让空模式对一切文本距离 0，必须剔除
    valid_anchors: list[tuple[str, str]] = []
    for raw in anchor_texts:
        normalized = _normalize_for_match(raw)
        if normalized:
            valid_anchors.append((raw, normalized))
    if not valid_anchors:
        return AnchorVerdict(
            anchor_hit=False,
            matched_text=None,
            matched_block_id=None,
            match_score=0.0,
            reason="[desk:no_anchor_texts] 锚点文本列表为空（或归一后全为空白），无从验证",
        )

    if not snapshot.text_blocks:
        return AnchorVerdict(
            anchor_hit=False,
            matched_text=None,
            matched_block_id=None,
            match_score=0.0,
            reason="[desk:no_text_blocks] 快照无 OCR 文本块，无法做像素锚点验证",
        )

    candidate_blocks: list[TextBlock] = snapshot.text_blocks
    if region is not None:
        candidate_blocks = [
            block for block in snapshot.text_blocks if _bboxes_intersect(block.bbox, region)
        ]
        if not candidate_blocks:
            return AnchorVerdict(
                anchor_hit=False,
                matched_text=None,
                matched_block_id=None,
                match_score=0.0,
                reason=(
                    f"[desk:region_no_text_blocks] region(x={region.x},y={region.y},"
                    f"w={region.width},h={region.height}) 内无任何 OCR 文本块"
                    "（相交判定=严格正面积重叠）"
                ),
            )

    normalized_blocks = [(block, _normalize_for_match(block.text)) for block in candidate_blocks]

    # 全对扫描取最优（分数=1-距离/锚点长度）。数学性质：命中对分数 ≥ 1-ratio、
    # 未命中对分数 < 1-ratio，故只要存在命中，最优对必是命中对，单一 best 即可。
    best_score = -1.0
    best: tuple[str, TextBlock, int, int] | None = None  # (锚点原文, 块, 距离, 允许编辑数)
    for anchor_raw, anchor_norm in valid_anchors:
        allowed_edits = int(len(anchor_norm) * ANCHOR_EDIT_DISTANCE_RATIO)
        for block, block_norm in normalized_blocks:
            distance = _substring_edit_distance(anchor_norm, block_norm)
            score = max(0.0, 1.0 - distance / len(anchor_norm))
            if score > best_score:
                best_score = score
                best = (anchor_raw, block, distance, allowed_edits)
            if distance == 0:
                break  # 精确（子串）命中，无需继续扫本锚点
        if best is not None and best[2] == 0:
            break

    if best is None:
        # F5（code-review INFO）：数学不可达分支——显式 raise 而非裸 assert，
        # 因 `python -O` 会剥掉 assert（生产环境静默跳过此不变式检查）。
        # candidate_blocks 与 valid_anchors 均已判非空（见上方短路返回），
        # 双重非空循环必产出至少一次 best 赋值；此处触发即代表调用前提被破坏
        # （上游校验逻辑被改动导致空列表漏检），非正常运行时可到达的分支。
        raise RuntimeError(
            "verify_pixel_anchor: best 扫描结果为 None——candidate_blocks/"
            "valid_anchors 非空前提被破坏（数学不可达分支，检查上游空值校验）"
        )
    anchor_raw, block, distance, allowed_edits = best
    if distance <= allowed_edits:
        return AnchorVerdict(
            anchor_hit=True,
            matched_text=block.text,
            matched_block_id=block.block_id,
            match_score=best_score,
            reason=(
                f"[desk:anchor_hit] 锚点 {anchor_raw!r} 命中文本块 {block.block_id}"
                f"（NFKC 归一后子串编辑距离 {distance}，阈值 {allowed_edits}）"
            ),
        )
    return AnchorVerdict(
        anchor_hit=False,
        matched_text=None,
        matched_block_id=None,
        match_score=best_score,
        reason=(
            f"[desk:anchor_miss] {len(valid_anchors)} 个锚点均未命中 "
            f"{len(candidate_blocks)} 个候选文本块；最接近：锚点 {anchor_raw!r} vs 块 "
            f"{block.block_id}（编辑距离 {distance} > 阈值 {allowed_edits}）"
        ),
    )
