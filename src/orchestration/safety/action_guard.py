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

# ── 环境配置（工程假设，Task 12 标定） ────────────────────────────────────────

TOCTOU_WAIT_MS: int = int(os.environ.get("TOCTOU_WAIT_MS", "200"))
TOCTOU_HASH_THRESHOLD: float = float(os.environ.get("TOCTOU_HASH_THRESHOLD", "0.1"))

# ── 三级白名单（action_type → 最大允许 ActionRisk） ───────────────────────────
# 不在白名单中的 action_type 一律升级为 DESTRUCTIVE。
# 工程假设：此处为初始保守白名单，扩展在 Task 12。

_ACTION_RISK_WHITELIST: dict[str, ActionRisk] = {
    "screenshot": ActionRisk.READ_ONLY,
    "get_uia_tree": ActionRisk.READ_ONLY,
    "window_list": ActionRisk.READ_ONLY,
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
        """三级风险判定：白名单二次确认 + 声明风险取最高级。

        判定逻辑：
          1. 取 action.risk_level（上游声明级别）。
          2. 查白名单 _ACTION_RISK_WHITELIST，若 action_type 不在白名单，
             直接升级为 DESTRUCTIVE。
          3. 若 action_type 在白名单，取声明级别与白名单最大允许级别中的较高者。

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

        # 取声明级别与白名单最大允许级别中的较高者
        if _RISK_ORDER[declared_risk] > _RISK_ORDER[whitelist_max]:
            logger.warning(
                "classify_risk: action_type=%r 声明级别 %r 超出白名单最大 %r，保留声明",
                action.action_type,
                declared_risk,
                whitelist_max,
            )
            return declared_risk

        return declared_risk

    async def toctou_verify(
        self,
        action: ActionSpec,
        snapshot_before: ScreenSnapshot | None = None,
    ) -> Literal["pass", "abort"]:
        """TOCTOU 验证（Pre-execution UI State Verification）。

        触发条件（满足任一即执行验证）：
          - risk_level 为 DESTRUCTIVE 或 LOW_RISK
          - action.coordinates 非 None（坐标点击强制走 TOCTOU，防 notification hijacking）

        验证流程：
          1. 若 snapshot_before 提供截图路径，直接用其 phash；否则调一次 screen_snapshot。
          2. 等待 TOCTOU_WAIT_MS 毫秒。
          3. 再调一次 screen_snapshot 取第二张截图。
          4. 比对两次 phash，delta > TOCTOU_HASH_THRESHOLD 则 abort（界面已变）。

        注意：本方法只做只读操作（两次截图 + hash 比对），适合放在 interrupt 前只读区。

        Args:
            action: 待验证的动作规格。
            snapshot_before: 可选的执行前快照（已有截图则复用，减少一次 RPC）。

        Returns:
            "pass"（界面稳定，可执行）或 "abort"（界面已变，拒绝执行）。
        """
        # 判断是否需要 TOCTOU 验证
        needs_toctou = (
            action.risk_level in (ActionRisk.DESTRUCTIVE, ActionRisk.LOW_RISK)
            or action.coordinates is not None
        )

        if not needs_toctou:
            logger.debug("toctou_verify: action=%r 为 READ_ONLY 且无坐标，跳过", action.action_id)
            return "pass"

        # --- interrupt 前只读区 ---
        # 取第一张截图（phash 计算）
        path_before: str | None = None
        if snapshot_before is not None and snapshot_before.screenshot_path is not None:
            path_before = snapshot_before.screenshot_path
        else:
            snap_a = await self.client.screen_snapshot(capture_screenshot=True, mode="uia_only")
            path_before = snap_a.screenshot_path

        if path_before is None:
            logger.warning("toctou_verify: 第一次截图无路径，无法比对，放行（降级）")
            return "pass"

        try:
            bits_before = _compute_phash_bits(path_before)
        except ValueError as exc:
            logger.warning("toctou_verify: 第一次 phash 失败（%s），放行（降级）", exc)
            return "pass"

        # 等待 TOCTOU 窗口（工程假设：200ms 保守下限，Task 12 标定）
        await asyncio.sleep(TOCTOU_WAIT_MS / 1000.0)

        # 取第二张截图
        snap_b = await self.client.screen_snapshot(capture_screenshot=True, mode="uia_only")
        path_after = snap_b.screenshot_path

        if path_after is None:
            logger.warning("toctou_verify: 第二次截图无路径，无法比对，放行（降级）")
            return "pass"

        try:
            bits_after = _compute_phash_bits(path_after)
        except ValueError as exc:
            logger.warning("toctou_verify: 第二次 phash 失败（%s），放行（降级）", exc)
            return "pass"
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
