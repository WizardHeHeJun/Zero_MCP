"""异常现场上报（Task 14）：log + 截屏 + 前 N 步流程落盘为「现场包」。

FileIncidentReporter 实现 IncidentReporter Protocol（src/orchestration/protocols.py，
签名不改），由 error_report_node 在停滞超阈值时调用（graph 条件边保证）。

每次 report 生成一个现场包目录：
    <incident_dir>/incident_<task_id>_<UTC时间戳>_<短uuid>/
        incident.json    — 全部入参 + iso 时间戳 + schema_version=1
                           （metadata.recent_steps 即前 N 步流程，由节点侧注入）
        screenshot.png   — 快照截图副本（snapshot_store 可加载且截图文件存在时）

设计约束：
  - 位于编排层（src/orchestration/safety/），不依赖记忆层或存储层；
    仅经 SnapshotStore Protocol 取快照（orchestration → agents 下调允许），
    不 import desktop_graph（避免循环）。
  - report 对外**永不抛**：截图 load/拷贝任一失败降级为 json 里记 screenshot_error；
    整体失败仅 log warning（error_report_node 虽有 try 兜底，reporter 自身也要稳）。
  - 全部文件 I/O 走 asyncio.to_thread（python-code.md：I/O async 不阻塞事件循环）。
  - incident_dir 未显式给出时读 env INCIDENT_DIR，仍无则系统 TEMP 下
    zero_mcp_incidents/（feature-flag 接线在 get_graph：env 未设默认 Noop 零回归）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.agents.protocols import SnapshotStore

logger = logging.getLogger(__name__)

INCIDENT_SCHEMA_VERSION: int = 1
"""incident.json 的版本字段值（字段增删时递增，供事后解析方判别）。"""


def _sanitize_for_path(raw: str) -> str:
    """把 task_id 清洗为安全的目录名片段（仅保留字母数字、`.`、`-`、`_`）。

    Args:
        raw: 原始字符串（可能含路径分隔符等非法字符）。

    Returns:
        可安全用于目录名的字符串（空串时返回 "unknown"）。
    """
    # 限长：超长 task_id 会使目录路径超 Windows MAX_PATH（mkdir WinError 3 → 整包丢失）
    cleaned = re.sub(r"[^\w.\-]", "_", raw)[:64]
    return cleaned or "unknown"


class FileIncidentReporter:
    """IncidentReporter 文件落盘实现：每次上报生成一个异常现场包目录。

    用法（get_graph 工厂内 feature-flag 接线，env INCIDENT_DIR 已设时启用）：
        reporter = FileIncidentReporter(snapshot_store=store)
        graph = get_graph(incident_reporter=reporter, ...)
    """

    def __init__(
        self,
        incident_dir: str | None = None,
        snapshot_store: SnapshotStore | None = None,
    ) -> None:
        """初始化 FileIncidentReporter。

        Args:
            incident_dir: 现场包根目录；None 时读 env INCIDENT_DIR，
                仍无则系统 TEMP 下 zero_mcp_incidents/。
            snapshot_store: 快照存取接口（用于取截图）；None 时现场包不含截图。
        """
        if incident_dir is None:
            incident_dir = os.environ.get("INCIDENT_DIR") or None
        if incident_dir is None:
            incident_dir = str(Path(tempfile.gettempdir()) / "zero_mcp_incidents")
        self.incident_dir = Path(incident_dir)
        self.snapshot_store = snapshot_store

    async def report(
        self,
        task_id: str,
        stall_count: int,
        errors: dict[str, str | None],
        snapshot_ref: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """上报错误/停滞事件：落盘 incident.json + 截图副本，对外永不抛。

        Args:
            task_id: 任务唯一 ID。
            stall_count: 当前停滞计数。
            errors: 错误信息字典（perception_error / control_error 等）。
            snapshot_ref: 最新感知快照 ID（用于取截图现场）。
            metadata: 可选附加信息（error_report_node 注入 recent_steps 等）。
        """
        try:
            package_dir = await self._write_incident_package(
                task_id=task_id,
                stall_count=stall_count,
                errors=errors,
                snapshot_ref=snapshot_ref,
                metadata=metadata,
            )
            logger.info(
                "FileIncidentReporter: 现场包已落盘 task_id=%r stall_count=%d dir=%s",
                task_id,
                stall_count,
                package_dir,
            )
        except Exception as exc:
            # reporter 自身永不抛（error_report_node 的 try 兜底之外的第二道保险）
            logger.warning(
                "FileIncidentReporter: 现场包落盘失败（降级不抛）task_id=%r: %s",
                task_id,
                exc,
            )

    async def _write_incident_package(
        self,
        task_id: str,
        stall_count: int,
        errors: dict[str, str | None],
        snapshot_ref: str | None,
        metadata: dict[str, Any] | None,
    ) -> Path:
        """创建现场包目录并写 incident.json（+ 截图副本，可降级）。

        Returns:
            现场包目录路径。
        """
        reported_at = datetime.now(UTC)
        dir_name = "incident_{}_{}_{}".format(
            _sanitize_for_path(task_id),
            reported_at.strftime("%Y%m%dT%H%M%SZ"),
            uuid.uuid4().hex[:8],
        )
        package_dir = self.incident_dir / dir_name
        await asyncio.to_thread(package_dir.mkdir, parents=True, exist_ok=True)

        record: dict[str, Any] = {
            "schema_version": INCIDENT_SCHEMA_VERSION,
            "reported_at": reported_at.isoformat(),
            "task_id": task_id,
            "stall_count": stall_count,
            "errors": errors,
            "snapshot_ref": snapshot_ref,
            "metadata": metadata,
        }
        # reporter 是最后取证通道：截图采集的任何未预期异常（如违 Protocol 的
        # load 返回 None → AttributeError）都不得连带吞掉 incident.json
        try:
            record.update(await self._collect_screenshot(package_dir, snapshot_ref))
        except Exception as exc:
            logger.warning("FileIncidentReporter: 截图采集异常（json 照常落盘）: %s", exc)
            record["screenshot_error"] = f"截图采集异常: {exc}"

        json_path = package_dir / "incident.json"
        payload = json.dumps(record, ensure_ascii=False, indent=2, default=str)
        await asyncio.to_thread(json_path.write_text, payload, "utf-8")
        return package_dir

    async def _collect_screenshot(
        self,
        package_dir: Path,
        snapshot_ref: str | None,
    ) -> dict[str, Any]:
        """尝试把快照截图拷入现场包，失败降级为 screenshot_error 字段（不抛）。

        Args:
            package_dir: 现场包目录（已创建）。
            snapshot_ref: 快照 ID；None 时直接跳过（无截图字段）。

        Returns:
            合并进 incident.json 的截图相关字段：
              成功 → {"screenshot": "screenshot.png", "screenshot_source_path": 原路径}
              降级 → {"screenshot_error": 原因}（load 失败/无截图路径/拷贝失败）
              跳过 → {}（snapshot_store 或 snapshot_ref 为 None）
        """
        if self.snapshot_store is None or snapshot_ref is None:
            return {}

        try:
            snapshot = await self.snapshot_store.load(snapshot_ref)
        except Exception as exc:
            logger.warning(
                "FileIncidentReporter: 快照加载失败 snapshot_ref=%r: %s", snapshot_ref, exc
            )
            return {"screenshot_error": f"快照加载失败: {exc}"}

        source_path = snapshot.screenshot_path
        if source_path is None:
            logger.warning(
                "FileIncidentReporter: 快照无截图路径 snapshot_ref=%r（screenshot_path=None）",
                snapshot_ref,
            )
            return {"screenshot_error": "快照无截图路径（screenshot_path=None）"}

        dest_path = package_dir / "screenshot.png"
        try:
            await asyncio.to_thread(shutil.copyfile, source_path, dest_path)
        except OSError as exc:
            logger.warning(
                "FileIncidentReporter: 截图拷贝失败 %s → %s: %s", source_path, dest_path, exc
            )
            return {
                "screenshot_error": f"截图拷贝失败: {exc}",
                "screenshot_source_path": source_path,
            }

        return {"screenshot": "screenshot.png", "screenshot_source_path": source_path}
