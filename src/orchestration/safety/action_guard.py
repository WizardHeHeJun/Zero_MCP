"""安全门：动作风险判定 + TOCTOU 验证 + 屏幕文本过滤。

位于编排层（src/orchestration/safety/），不依赖记忆层或存储层。
ActionGuard 注入 DesktopMCPClient，不持底层连接句柄。

设计依据：
- TOCTOU 防御 (arXiv:2604.18860)：Pre-execution UI State Verification，
  坐标点击是 notification hijacking 的主命中点，强制走 TOCTOU 验证。
- 注入过滤三层 (arXiv:2506.02456, OWASP LLM01:2025, Unit42)：
  结构标记正则 → 关键词词表 → 混淆检测（NFKC + Base64）。
- 护栏在编排层 (arXiv:2506.02456, arXiv:2505.10924)：
  系统提示防御效果有限，须在编排层做结构化过滤。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Literal

from src.agents.models.screen_snapshot import ActionRisk, ActionSpec, ScreenSnapshot
from src.mcp.desktop_mcp_client import DesktopMCPClient
from src.orchestration.phash import average_hash_from_path as _compute_phash_bits
from src.orchestration.phash import hamming_ratio as _hamming_distance_ratio

logger = logging.getLogger(__name__)

# ── 环境配置 ──────────────────────────────────────────────────────────────────
# Task 12 已标定（notes/e2e-desktop-task-results.md §四.2/§五）：
#   TOCTOU_WAIT_MS=200 相对单快照 ~1.2s 延迟为安全下限，保留。
#   TOCTOU_HASH_THRESHOLD=0.1 —— 全窗口 phash 对有动画的现代应用无静止基线
#   （钉钉无人操作时 delta 在 0/0.47 间跳），此阈值只在 hash 裁剪到「操作目标
#   局部邻域」口径下成立 → toctou_verify 对坐标动作按 TOCTOU_CROP_HALF_PX
#   邻域裁剪比对；无坐标的动作（如 close_window）退回整图口径。
#   TOCTOU_CROP_HALF_PX=150（300×300 邻域）为工程假设：覆盖常见按钮/菜单目标，
#   小于动画区到目标的典型距离；动效恰在目标上时 abort 是正确行为（目标不稳定）。

TOCTOU_WAIT_MS: int = int(os.environ.get("TOCTOU_WAIT_MS", "200"))
TOCTOU_HASH_THRESHOLD: float = float(os.environ.get("TOCTOU_HASH_THRESHOLD", "0.1"))
TOCTOU_CROP_HALF_PX: int = int(os.environ.get("TOCTOU_CROP_HALF_PX", "150"))

# ── 三级白名单（action_type → 最大允许 ActionRisk） ───────────────────────────
# 不在白名单中的 action_type 一律升级为 DESTRUCTIVE。
# 工程假设：此处为初始保守白名单，扩展在 Task 12。

_ACTION_RISK_WHITELIST: dict[str, ActionRisk] = {
    "screenshot": ActionRisk.READ_ONLY,
    "get_uia_tree": ActionRisk.READ_ONLY,
    "window_list": ActionRisk.READ_ONLY,
    # code-review F3：当前 DesktopControlAgent._dispatch_write 不识别
    # "focus_window"/"pin_topmost" action_type（Agent 侧未接线，ActionSpec 无
    # 窗口句柄字段），此白名单值为将来接线预留，不代表本条目已在受控路径生效。
    "focus_window": ActionRisk.LOW_RISK,
    "move_mouse": ActionRisk.LOW_RISK,
    "scroll": ActionRisk.LOW_RISK,
    "click": ActionRisk.LOW_RISK,
    "type": ActionRisk.LOW_RISK,
    "key": ActionRisk.LOW_RISK,
    "window_close": ActionRisk.DESTRUCTIVE,
    "delete": ActionRisk.DESTRUCTIVE,
    "submit": ActionRisk.DESTRUCTIVE,
}

# ActionRisk 大小顺序（升级判断用）
_RISK_ORDER: dict[ActionRisk, int] = {
    ActionRisk.READ_ONLY: 0,
    ActionRisk.LOW_RISK: 1,
    ActionRisk.DESTRUCTIVE: 2,
}

# sanitize_screen_text 已移至 src.agents.text_filter（agents 层纯函数）。
# 本模块通过 import 引用，供 ActionGuard.toctou_verify 在文本预处理时调用（可选），
# 以及供编排层其他节点直接 from src.orchestration.safety.action_guard import sanitize_screen_text
# 做向后兼容（re-export 路径：action_guard → text_filter）。


# ── 感知哈希 ──────────────────────────────────────────────────────────────────
# phash 计算（average hash）已统一至 src.orchestration.phash（消除双实现技术债）。
# 本模块通过别名 import 保留 _compute_phash_bits / _hamming_distance_ratio 名字，
# 供 toctou_verify 调用与既有测试引用，底层实现单一。


# ── ActionGuard ────────────────────────────────────────────────────────────────


class ActionGuard:
    """动作安全门（编排层注入，不持底层连接句柄）。

    注入 DesktopMCPClient 实例（已初始化的 context manager 内部），
    自身不负责连接生命周期管理。

    使用方式（在 control_node 内）：
        guard = ActionGuard(client)
        risk = await guard.classify_risk(action)
        verdict = await guard.toctou_verify(action, snapshot_before)
    """

    def __init__(self, client: DesktopMCPClient) -> None:
        """初始化 ActionGuard。

        Args:
            client: 已建立连接的 DesktopMCPClient 实例。
        """
        self.client = client

    async def classify_risk(self, action: ActionSpec) -> ActionRisk:
        """三级风险判定：白名单二次确认 + 声明与白名单基线取较高者。

        判定逻辑：
          1. 取 action.risk_level（上游声明级别）。
          2. 查白名单 _ACTION_RISK_WHITELIST，若 action_type 不在白名单，
             直接升级为 DESTRUCTIVE。
          3. 若 action_type 在白名单，返回声明级别与白名单基线级别中的**较高者**
             （K1：白名单是风险下限——上游把 window_close 等高危动作低报为
             LOW_RISK/READ_ONLY 时按白名单升级，防止绕过 DESTRUCTIVE interrupt
             确认；声明高于白名单时保留声明）。

        Args:
            action: 待判定的动作规格。

        Returns:
            ActionRisk（可能比 action.risk_level 更高）。
        """
        declared_risk = action.risk_level
        whitelist_max = _ACTION_RISK_WHITELIST.get(action.action_type)

        if whitelist_max is None:
            # action_type 不在白名单，升级为 DESTRUCTIVE
            logger.warning(
                "classify_risk: action_type=%r 不在白名单，升级为 DESTRUCTIVE",
                action.action_type,
            )
            return ActionRisk.DESTRUCTIVE

        # 声明高于白名单基线 → 保留声明（取较高者）
        if _RISK_ORDER[declared_risk] > _RISK_ORDER[whitelist_max]:
            logger.warning(
                "classify_risk: action_type=%r 声明级别 %r 超出白名单基线 %r，保留声明",
                action.action_type,
                declared_risk,
                whitelist_max,
            )
            return declared_risk

        # 声明低于白名单基线 → 按白名单升级（K1 修复：此前白名单值从未被返回，
        # window_close 低报 LOW_RISK 即绕过 DESTRUCTIVE interrupt 确认）
        if _RISK_ORDER[declared_risk] < _RISK_ORDER[whitelist_max]:
            logger.warning(
                "classify_risk: action_type=%r 声明级别 %r 低于白名单基线 %r，升级为白名单值",
                action.action_type,
                declared_risk,
                whitelist_max,
            )
            return whitelist_max

        return declared_risk

    async def toctou_verify(
        self,
        action: ActionSpec,
        snapshot_before: ScreenSnapshot | None = None,
        effective_risk: ActionRisk | None = None,
    ) -> Literal["pass", "abort", "abort_degraded"]:
        """TOCTOU 验证（Pre-execution UI State Verification）。

        触发条件（满足任一即执行验证）：
          - 有效风险为 DESTRUCTIVE 或 LOW_RISK（K1 ②：按 effective_risk 判定，
            不信 action.risk_level 声明值——上游低报时仍强制走验证；
            effective_risk=None 回退声明值，保旧调用方兼容）
          - action.coordinates 非 None（坐标点击强制走 TOCTOU，防 notification hijacking）

        验证流程：
          1. 若 snapshot_before 提供截图路径，直接用其 phash；否则调一次 screen_snapshot。
          2. 等待 TOCTOU_WAIT_MS 毫秒。
          3. 再调一次 screen_snapshot 取第二张截图。
          4. 比对两次 phash，delta > TOCTOU_HASH_THRESHOLD 则 abort（界面已变）。
             坐标动作按 TOCTOU_CROP_HALF_PX 邻域**局部裁剪**比对（Task 12 实测：
             整图 hash 被应用自身动效持续误报，无静止基线）；坐标经各图
             capture_origin 换算，兼容全屏截图与 PrintWindow 窗口图混用。

        降级语义（K1 ③ + 二期三态化）：验证链路降级（截图无路径 / phash 失败）
        时按有效风险分级——DESTRUCTIVE 返回 "abort_degraded"（验证不可得时
        fail-closed 拒绝执行，与「界面真变了」的 "abort" 区分，消费侧据此在
        control_error 带机读令牌 [desk:toctou_degraded]），非 DESTRUCTIVE
        保留放行（logger.warning）。

        注意：本方法只做只读操作（两次截图 + hash 比对），适合放在 interrupt 前只读区。

        Args:
            action: 待验证的动作规格。
            snapshot_before: 可选的执行前快照（已有截图则复用，减少一次 RPC）。
            effective_risk: classify_risk 判定后的有效风险级别；None 时回退
                action.risk_level 声明值（向后兼容旧调用方）。

        Returns:
            "pass"（界面稳定，可执行）、"abort"（界面已变，拒绝执行）或
            "abort_degraded"（验证降级且 DESTRUCTIVE，fail-closed 拒绝执行）。
        """
        # K1 ②：触发判定用有效风险（classify_risk 结果），None 回退声明值保兼容
        risk_for_gate = effective_risk if effective_risk is not None else action.risk_level

        # 判断是否需要 TOCTOU 验证
        needs_toctou = (
            risk_for_gate in (ActionRisk.DESTRUCTIVE, ActionRisk.LOW_RISK)
            or action.coordinates is not None
        )

        if not needs_toctou:
            logger.debug(
                "toctou_verify: action=%r 有效风险 READ_ONLY 且无坐标，跳过", action.action_id
            )
            return "pass"

        # --- interrupt 前只读区 ---
        # 局部裁剪口径（Task 12 §四.2/§五 实测）：坐标动作只比对目标邻域，
        # 避免窗口自身动效（动画/红点/时钟）造成整图 hash 无静止基线。
        # crop 按各自图像的 capture_origin 把屏幕绝对坐标换算为图像坐标——
        # 两张图可能一张全屏(origin=(0,0))一张 PrintWindow 窗口图，但比对的
        # 屏幕区域一致。
        def _crop_for(origin: tuple[int, int]) -> tuple[int, int, int, int] | None:
            if action.coordinates is None:
                return None  # 无坐标动作（如 close_window）退回整图口径
            cx, cy = action.coordinates[0] - origin[0], action.coordinates[1] - origin[1]
            return (
                cx - TOCTOU_CROP_HALF_PX,
                cy - TOCTOU_CROP_HALF_PX,
                cx + TOCTOU_CROP_HALF_PX,
                cy + TOCTOU_CROP_HALF_PX,
            )

        # 取第一张截图（phash 计算）
        path_before: str | None = None
        origin_before: tuple[int, int] = (0, 0)
        if snapshot_before is not None and snapshot_before.screenshot_path is not None:
            path_before = snapshot_before.screenshot_path
            origin_before = snapshot_before.capture_origin
        else:
            snap_a = await self.client.screen_snapshot(capture_screenshot=True, mode="uia_only")
            path_before = snap_a.screenshot_path
            origin_before = snap_a.capture_origin

        if path_before is None:
            return self._degraded_verdict(action, risk_for_gate, "第一次截图无路径，无法比对")

        try:
            bits_before = _compute_phash_bits(path_before, _crop_for(origin_before))
        except ValueError as exc:
            return self._degraded_verdict(action, risk_for_gate, f"第一次 phash 失败（{exc}）")

        # 等待 TOCTOU 窗口（Task 12 标定：200ms 相对单快照 ~1.2s 延迟为安全下限）
        await asyncio.sleep(TOCTOU_WAIT_MS / 1000.0)

        # 取第二张截图
        snap_b = await self.client.screen_snapshot(capture_screenshot=True, mode="uia_only")
        path_after = snap_b.screenshot_path

        if path_after is None:
            return self._degraded_verdict(action, risk_for_gate, "第二次截图无路径，无法比对")

        try:
            bits_after = _compute_phash_bits(path_after, _crop_for(snap_b.capture_origin))
        except ValueError as exc:
            return self._degraded_verdict(action, risk_for_gate, f"第二次 phash 失败（{exc}）")
        # --- interrupt 前只读区结束 ---

        delta = _hamming_distance_ratio(bits_before, bits_after)
        logger.info(
            "toctou_verify: action=%r phash delta=%.4f threshold=%.4f",
            action.action_id,
            delta,
            TOCTOU_HASH_THRESHOLD,
        )

        if delta > TOCTOU_HASH_THRESHOLD:
            logger.warning(
                "toctou_verify: ABORT action=%r，界面已变（delta=%.4f > %.4f）",
                action.action_id,
                delta,
                TOCTOU_HASH_THRESHOLD,
            )
            return "abort"

        return "pass"

    def _degraded_verdict(
        self,
        action: ActionSpec,
        risk_for_gate: ActionRisk,
        detail: str,
    ) -> Literal["pass", "abort_degraded"]:
        """TOCTOU 验证链路降级（截图无路径 / phash 失败）时的分级裁决（K1 ③）。

        二期三态化（收口 K1 ③ 遗留）：降级拒绝返回 "abort_degraded" 而非 "abort"，
        让消费侧（DesktopControlAgent）能区分「界面真变了」与「验证不可得」，
        在 control_error 带机读令牌：
          - DESTRUCTIVE：验证不可得即拒绝执行（abort_degraded），logger.error
            文案含机读令牌 [desk:toctou_degraded]（位置无关，消费侧用 re.search 提取）。
          - 非 DESTRUCTIVE：保留旧行为放行（pass），logger.warning。

        Args:
            action: 待验证的动作规格。
            risk_for_gate: 本次验证使用的有效风险级别。
            detail: 降级原因（人读散文，与机读令牌并存于同一 error 文案）。

        Returns:
            "abort_degraded"（DESTRUCTIVE fail-closed）或 "pass"（非 DESTRUCTIVE 降级放行）。
        """
        if risk_for_gate == ActionRisk.DESTRUCTIVE:
            logger.error(
                "toctou_verify: TOCTOU 验证降级（%s），DESTRUCTIVE 动作 fail-closed "
                "拒绝执行 [desk:toctou_degraded] action=%r",
                detail,
                action.action_id,
            )
            return "abort_degraded"
        logger.warning(
            "toctou_verify: %s，非 DESTRUCTIVE 放行（降级）action=%r",
            detail,
            action.action_id,
        )
        return "pass"
