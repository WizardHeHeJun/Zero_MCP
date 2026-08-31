"""FileIncidentReporter 异常现场上报单测（Task 14）。

覆盖：
  1. report 落盘：现场包目录结构 / incident.json 字段齐全 / 截图副本字节一致。
  2. 降级路径（全部不抛）：snapshot_store=None / snapshot_ref=None / load 抛异常 /
     screenshot_path=None / 截图文件不存在 / incident_dir 不可写。
  3. 构造参数解析：显式 incident_dir > env INCIDENT_DIR > 系统 TEMP 兜底。
  4. error_report_node metadata 含 recent_steps 且按 INCIDENT_STEP_WINDOW 截断（12→10）。
  5. get_graph feature-flag 接线：无 env → None（Noop 兜底）；env INCIDENT_DIR 已设
     → FileIncidentReporter；显式注入优先于 env。
  6. 红线自查：实现文件不 import desktop_graph（避免循环）、不 import 记忆/存储层。

设计约束（测试侧同样遵守红线）：
  - 不依赖 Postgres/Neo4j（tmp_path + InMemorySnapshotStore）。
  - 不 import Zero，不直连存储层。
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import src.orchestration.desktop_graph as desktop_graph
from src.agents.models.screen_snapshot import ScreenSnapshot
from src.agents.screen_perception_agent import InMemorySnapshotStore
from src.orchestration.desktop_graph import (
    _resolve_incident_reporter,
    get_graph,
    make_error_report_node,
)
from src.orchestration.desktop_supervisor import MAX_ITERATIONS_EXCEEDED
from src.orchestration.protocols import IncidentReporter, NoopIncidentReporter
from src.orchestration.safety.incident_reporter import FileIncidentReporter
from src.orchestration.state import DesktopTaskState, StepRecord, TaskStatus

# ── 测试辅助 ──────────────────────────────────────────────────────────────────

_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"zero-mcp-task14-test-png-payload"


def _make_snapshot(
    snapshot_id: str = "snap-14-001",
    screenshot_path: str | None = None,
) -> ScreenSnapshot:
    """构造测试用 ScreenSnapshot（最小字段）。"""
    return ScreenSnapshot(
        snapshot_id=snapshot_id,
        timestamp_ms=1_700_000_000_000,
        screen_width=1920,
        screen_height=1080,
        active_window_title="测试窗口",
        uia_elements=[],
        text_blocks=[],
        visual_objects=[],
        screenshot_path=screenshot_path,
        perception_mode="uia_ocr",
        capability_flags={},
    )


def _make_step(index: int) -> StepRecord:
    """构造测试用 StepRecord。"""
    return StepRecord(
        step_index=index,
        agent="perceive",
        instruction=f"步骤 {index}",
        snapshot_ref=None,
        perception_summary=None,
        control_error=None,
        perception_error="感知失败" if index % 2 else None,
        task_status="RUNNING",
    )


def _single_incident_dir(root: Path) -> Path:
    """断言 root 下有且仅有一个现场包目录并返回。"""
    dirs = [p for p in root.iterdir() if p.is_dir()]
    assert len(dirs) == 1, f"应恰有 1 个现场包目录，实际 {[p.name for p in dirs]}"
    return dirs[0]


def _load_record(package_dir: Path) -> dict[str, Any]:
    """读取现场包内的 incident.json。"""
    json_path = package_dir / "incident.json"
    assert json_path.is_file(), "现场包应含 incident.json"
    record: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    return record


# ── 1. report 落盘（happy path）──────────────────────────────────────────────


class TestFileIncidentReporterHappyPath:
    """report 落盘：目录结构 / json 字段齐全 / 截图副本字节一致。"""

    async def test_report_writes_package_with_screenshot(self, tmp_path: Path) -> None:
        """完整链路：造 PNG + InMemorySnapshotStore → 现场包含 json + 截图副本。"""
        src_png = tmp_path / "source.png"
        src_png.write_bytes(_PNG_BYTES)
        store = InMemorySnapshotStore()
        await store.save(_make_snapshot("snap-ok", str(src_png)))

        incidents = tmp_path / "incidents"
        reporter = FileIncidentReporter(
            incident_dir=str(incidents),
            snapshot_store=store,
        )
        await reporter.report(
            task_id="task-14-001",
            stall_count=3,
            errors={"perception_error": "持续感知失败", "control_error": None},
            snapshot_ref="snap-ok",
            metadata={"task_description": "落盘测试", "recent_steps": []},
        )

        package_dir = _single_incident_dir(incidents)
        assert package_dir.name.startswith("incident_task-14-001_"), (
            f"现场包目录名应为 incident_<task_id>_<时间戳>_<短uuid>，实际 {package_dir.name}"
        )

        record = _load_record(package_dir)
        # 全部入参 + iso 时间戳 + 版本字段
        assert record["schema_version"] == 1
        assert record["task_id"] == "task-14-001"
        assert record["stall_count"] == 3
        assert record["errors"] == {"perception_error": "持续感知失败", "control_error": None}
        assert record["snapshot_ref"] == "snap-ok"
        assert record["metadata"]["task_description"] == "落盘测试"
        assert "T" in record["reported_at"], "reported_at 应为 iso 时间戳"

        # 截图副本存在且字节一致 + json 记录原路径
        assert record["screenshot"] == "screenshot.png"
        assert record["screenshot_source_path"] == str(src_png)
        copied = package_dir / "screenshot.png"
        assert copied.is_file()
        assert copied.read_bytes() == _PNG_BYTES == src_png.read_bytes()
        assert "screenshot_error" not in record

    async def test_report_sanitizes_task_id_in_dir_name(self, tmp_path: Path) -> None:
        """task_id 含路径分隔符时目录名被清洗（不逃逸 incident_dir）。"""
        incidents = tmp_path / "inc"
        reporter = FileIncidentReporter(incident_dir=str(incidents), snapshot_store=None)
        await reporter.report(
            task_id="../evil/task\\id",
            stall_count=1,
            errors={},
            snapshot_ref=None,
        )
        package_dir = _single_incident_dir(incidents)
        assert "/" not in package_dir.name
        assert "\\" not in package_dir.name

    def test_file_reporter_satisfies_incident_reporter_protocol(self) -> None:
        """FileIncidentReporter 满足 IncidentReporter Protocol（结构子类型，签名未改）。"""
        reporter = FileIncidentReporter(incident_dir=None, snapshot_store=None)
        assert isinstance(reporter, IncidentReporter)


# ── 2. 降级路径（全部不抛）───────────────────────────────────────────────────


class TestFileIncidentReporterDegradation:
    """截图链路任一环失败 → json 仍落盘（含 screenshot_error 或无截图字段），不抛。"""

    async def test_snapshot_load_returns_none_still_writes_json(self, tmp_path: Path) -> None:
        """load 违 Protocol 返回 None（AttributeError 级异常）→ json 必落 + screenshot_error。

        review medium finding：reporter 是最后取证通道，_collect_screenshot 的
        任何未预期异常都不得连带吞掉 incident.json。
        """
        store = MagicMock()
        store.load = AsyncMock(return_value=None)
        reporter = FileIncidentReporter(incident_dir=str(tmp_path), snapshot_store=store)
        await reporter.report(
            task_id="t-none-load", stall_count=3, errors={}, snapshot_ref="snap-x"
        )
        pkg = _single_incident_dir(tmp_path)
        record = json.loads((pkg / "incident.json").read_text(encoding="utf-8"))
        assert "截图采集异常" in record["screenshot_error"]

    async def test_overlong_task_id_dir_name_bounded(self, tmp_path: Path) -> None:
        """300 字符 task_id → 目录名片段截断到 64，包正常落盘（防 MAX_PATH）。"""
        reporter = FileIncidentReporter(incident_dir=str(tmp_path), snapshot_store=None)
        await reporter.report(task_id="x" * 300, stall_count=1, errors={}, snapshot_ref=None)
        pkg = _single_incident_dir(tmp_path)
        assert (pkg / "incident.json").is_file()
        # incident_<sanitized>_<ts>_<uuid8>：sanitized 段 ≤64
        sanitized = pkg.name.split("_")[1]
        assert len(sanitized) <= 64

    async def test_snapshot_store_none_skips_screenshot(self, tmp_path: Path) -> None:
        """snapshot_store=None → json 落盘成功，无截图字段。"""
        reporter = FileIncidentReporter(incident_dir=str(tmp_path), snapshot_store=None)
        await reporter.report(
            task_id="task-no-store",
            stall_count=3,
            errors={"perception_error": "x", "control_error": None},
            snapshot_ref="snap-any",
        )
        record = _load_record(_single_incident_dir(tmp_path))
        assert "screenshot" not in record
        assert "screenshot_error" not in record

    async def test_snapshot_ref_none_skips_screenshot(self, tmp_path: Path) -> None:
        """snapshot_ref=None → json 落盘成功，无截图字段。"""
        reporter = FileIncidentReporter(
            incident_dir=str(tmp_path),
            snapshot_store=InMemorySnapshotStore(),
        )
        await reporter.report(
            task_id="task-no-ref",
            stall_count=3,
            errors={},
            snapshot_ref=None,
        )
        record = _load_record(_single_incident_dir(tmp_path))
        assert record["snapshot_ref"] is None
        assert "screenshot" not in record
        assert "screenshot_error" not in record

    async def test_snapshot_load_failure_degrades(self, tmp_path: Path) -> None:
        """load 抛异常（空 store 的 KeyError）→ json 含 screenshot_error，不抛。"""
        reporter = FileIncidentReporter(
            incident_dir=str(tmp_path),
            snapshot_store=InMemorySnapshotStore(),  # 空 store，load 抛 KeyError
        )
        await reporter.report(
            task_id="task-load-fail",
            stall_count=3,
            errors={},
            snapshot_ref="snap-missing",
        )
        package_dir = _single_incident_dir(tmp_path)
        record = _load_record(package_dir)
        assert "快照加载失败" in record["screenshot_error"]
        assert not (package_dir / "screenshot.png").exists()

    async def test_snapshot_without_screenshot_path_degrades(self, tmp_path: Path) -> None:
        """screenshot_path=None → json 含 screenshot_error，不抛。"""
        store = InMemorySnapshotStore()
        await store.save(_make_snapshot("snap-nopath", screenshot_path=None))
        reporter = FileIncidentReporter(incident_dir=str(tmp_path), snapshot_store=store)
        await reporter.report(
            task_id="task-nopath",
            stall_count=3,
            errors={},
            snapshot_ref="snap-nopath",
        )
        record = _load_record(_single_incident_dir(tmp_path))
        assert "无截图路径" in record["screenshot_error"]

    async def test_screenshot_file_missing_degrades(self, tmp_path: Path) -> None:
        """截图文件不存在（拷贝失败）→ json 含 screenshot_error + 原路径，不抛。"""
        gone = tmp_path / "nonexistent.png"
        store = InMemorySnapshotStore()
        await store.save(_make_snapshot("snap-gone", str(gone)))
        reporter = FileIncidentReporter(incident_dir=str(tmp_path / "inc"), snapshot_store=store)
        await reporter.report(
            task_id="task-copy-fail",
            stall_count=3,
            errors={},
            snapshot_ref="snap-gone",
        )
        package_dir = _single_incident_dir(tmp_path / "inc")
        record = _load_record(package_dir)
        assert "截图拷贝失败" in record["screenshot_error"]
        assert record["screenshot_source_path"] == str(gone)
        assert not (package_dir / "screenshot.png").exists()

    async def test_report_never_raises_when_incident_dir_unwritable(
        self, tmp_path: Path, caplog: Any
    ) -> None:
        """incident_dir 的父路径是文件（mkdir 必失败）→ report 不抛，仅 log warning。"""
        blocker = tmp_path / "blocker"
        blocker.write_text("占位文件", encoding="utf-8")
        reporter = FileIncidentReporter(
            incident_dir=str(blocker / "sub"),
            snapshot_store=None,
        )
        with caplog.at_level(logging.WARNING):
            await reporter.report(
                task_id="task-unwritable",
                stall_count=1,
                errors={},
                snapshot_ref=None,
            )
        assert "现场包落盘失败" in caplog.text


# ── 3. 构造参数解析（incident_dir > env > TEMP 兜底）─────────────────────────


class TestFileIncidentReporterConstructor:
    """incident_dir 解析优先级：显式参数 > env INCIDENT_DIR > 系统 TEMP。"""

    def test_explicit_incident_dir_wins_over_env(self, monkeypatch: Any, tmp_path: Path) -> None:
        monkeypatch.setenv("INCIDENT_DIR", str(tmp_path / "from-env"))
        reporter = FileIncidentReporter(incident_dir=str(tmp_path / "explicit"))
        assert reporter.incident_dir == tmp_path / "explicit"

    def test_env_incident_dir_used_when_not_given(self, monkeypatch: Any, tmp_path: Path) -> None:
        monkeypatch.setenv("INCIDENT_DIR", str(tmp_path / "from-env"))
        reporter = FileIncidentReporter()
        assert reporter.incident_dir == tmp_path / "from-env"

    def test_fallback_to_tempdir_without_env(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("INCIDENT_DIR", raising=False)
        reporter = FileIncidentReporter()
        assert reporter.incident_dir == Path(tempfile.gettempdir()) / "zero_mcp_incidents"

    def test_empty_env_falls_back_to_tempdir(self, monkeypatch: Any) -> None:
        """env 为空串（.env.example 默认）视为未设置。"""
        monkeypatch.setenv("INCIDENT_DIR", "")
        reporter = FileIncidentReporter()
        assert reporter.incident_dir == Path(tempfile.gettempdir()) / "zero_mcp_incidents"


# ── 4. error_report_node metadata.recent_steps ───────────────────────────────


class TestErrorReportNodeRecentSteps:
    """error_report_node metadata 含 recent_steps 且按 INCIDENT_STEP_WINDOW 截断。"""

    async def test_metadata_contains_recent_steps_truncated(self) -> None:
        """12 步 → recent_steps 只取最近 10 步（默认窗口），为 json 可序列化 dict。"""
        reporter = MagicMock()
        reporter.report = AsyncMock()
        node = make_error_report_node(incident_reporter=reporter)

        state = DesktopTaskState(
            task_id="task-recent-steps",
            task_description="窗口截断测试",
            task_status=TaskStatus.RUNNING,
            step_history=[_make_step(i) for i in range(12)],
            stall_count=3,
        )
        result = await node(state)

        assert result == {"task_status": TaskStatus.FAILED}
        assert reporter.report.await_count == 1
        metadata = reporter.report.await_args.kwargs["metadata"]
        # 既有字段不动
        assert metadata["task_description"] == "窗口截断测试"
        assert metadata["step_count"] == 12
        # recent_steps：最近 10 步（step_index 2..11），json 模式 dict
        recent = metadata["recent_steps"]
        assert len(recent) == 10, f"12 步应截断为 10 步，实际 {len(recent)}"
        assert [s["step_index"] for s in recent] == list(range(2, 12))
        assert all(isinstance(s, dict) for s in recent)
        # dict 可直接 json 序列化（model_dump(mode="json") 保证）
        json.dumps(recent, ensure_ascii=False)

    async def test_metadata_carries_failure_reason_and_iteration_count(self) -> None:
        """K4 紧后 §3.3：现场包 metadata 带 failure_reason 与 iteration_count——
        回路硬上限失败在事后排障时与停滞/LLM 失败可区分。"""
        reporter = MagicMock()
        reporter.report = AsyncMock()
        node = make_error_report_node(incident_reporter=reporter)

        state = DesktopTaskState(
            task_id="task-max-iter",
            task_description="硬上限现场包测试",
            task_status=TaskStatus.RUNNING,
            failure_reason=MAX_ITERATIONS_EXCEEDED,
            iteration_count=31,
        )
        await node(state)

        metadata = reporter.report.await_args.kwargs["metadata"]
        assert metadata["failure_reason"] == MAX_ITERATIONS_EXCEEDED
        assert metadata["iteration_count"] == 31

    async def test_recent_steps_perception_summary_truncated(self) -> None:
        """超长 perception_summary 逐步截断到 INCIDENT_SUMMARY_MAX_CHARS + 截断标记。

        review low finding：现场包单包体积有界；全文可经 snapshot_ref 回查。
        """
        reporter = MagicMock()
        reporter.report = AsyncMock()
        node = make_error_report_node(incident_reporter=reporter)

        long_step = _make_step(0).model_copy(update={"perception_summary": "长" * 3000})
        state = DesktopTaskState(
            task_id="task-long-summary",
            task_description="截断测试",
            task_status=TaskStatus.RUNNING,
            step_history=[long_step],
            stall_count=3,
        )
        await node(state)

        recent = reporter.report.await_args.kwargs["metadata"]["recent_steps"]
        summary = recent[0]["perception_summary"]
        assert summary.startswith("长" * 100)
        assert "…[截断 1000 字符]" in summary
        assert len(summary) < 3000
        json.dumps(recent, ensure_ascii=False)

    async def test_metadata_recent_steps_short_history_not_padded(self) -> None:
        """步数少于窗口时全量携带，不补齐。"""
        reporter = MagicMock()
        reporter.report = AsyncMock()
        node = make_error_report_node(incident_reporter=reporter)

        state = DesktopTaskState(
            task_id="task-short-history",
            task_description="短历史",
            task_status=TaskStatus.RUNNING,
            step_history=[_make_step(i) for i in range(3)],
            stall_count=3,
        )
        await node(state)
        recent = reporter.report.await_args.kwargs["metadata"]["recent_steps"]
        assert [s["step_index"] for s in recent] == [0, 1, 2]

    async def test_node_with_file_reporter_writes_full_incident_package(
        self, tmp_path: Path
    ) -> None:
        """节点 + FileIncidentReporter 端到端：现场包含截屏 + 步骤历史（验收口径）。"""
        src_png = tmp_path / "screen.png"
        src_png.write_bytes(_PNG_BYTES)
        store = InMemorySnapshotStore()
        await store.save(_make_snapshot("snap-e2e", str(src_png)))

        incidents = tmp_path / "incidents"
        reporter = FileIncidentReporter(incident_dir=str(incidents), snapshot_store=store)
        node = make_error_report_node(incident_reporter=reporter)

        state = DesktopTaskState(
            task_id="task-e2e-14",
            task_description="端到端现场包",
            task_status=TaskStatus.RUNNING,
            step_history=[_make_step(i) for i in range(4)],
            stall_count=3,
            snapshot_ref="snap-e2e",
        )
        result = await node(state)

        assert result == {"task_status": TaskStatus.FAILED}
        package_dir = _single_incident_dir(incidents)
        record = _load_record(package_dir)
        assert record["snapshot_ref"] == "snap-e2e"
        assert len(record["metadata"]["recent_steps"]) == 4
        assert (package_dir / "screenshot.png").read_bytes() == _PNG_BYTES


# ── 5. get_graph feature-flag 接线（默认关零回归）─────────────────────────────


class TestGetGraphIncidentReporterWiring:
    """get_graph：无 env → Noop 兜底；env INCIDENT_DIR 已设 → FileIncidentReporter。"""

    def test_resolve_returns_none_without_env(self, monkeypatch: Any) -> None:
        """无 env 且未显式注入 → None（make_error_report_node 内落 Noop，零回归）。"""
        monkeypatch.delenv("INCIDENT_DIR", raising=False)
        assert _resolve_incident_reporter(None, None) is None

    def test_resolve_env_enables_file_reporter(self, monkeypatch: Any, tmp_path: Path) -> None:
        """env INCIDENT_DIR 已设 → FileIncidentReporter，snapshot_store 透传。"""
        monkeypatch.setenv("INCIDENT_DIR", str(tmp_path))
        store = InMemorySnapshotStore()
        resolved = _resolve_incident_reporter(None, store)
        assert isinstance(resolved, FileIncidentReporter)
        assert resolved.incident_dir == tmp_path
        assert resolved.snapshot_store is store

    def test_resolve_explicit_injection_wins_over_env(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        """显式注入优先于 env（调用方选择权最高）。"""
        monkeypatch.setenv("INCIDENT_DIR", str(tmp_path))
        injected = NoopIncidentReporter()
        assert _resolve_incident_reporter(injected, None) is injected

    def test_get_graph_default_wires_noop(self, monkeypatch: Any) -> None:
        """get_graph 默认（无 env、不注入）→ resolver 返回 None → Noop 兜底。"""
        monkeypatch.delenv("INCIDENT_DIR", raising=False)
        captured: list[Any] = []
        original = _resolve_incident_reporter

        def spy(reporter: Any, store: Any) -> Any:
            resolved = original(reporter, store)
            captured.append(resolved)
            return resolved

        monkeypatch.setattr(desktop_graph, "_resolve_incident_reporter", spy)
        graph = get_graph(checkpointer=None)
        assert graph is not None
        assert captured == [None]

    def test_get_graph_env_wires_file_reporter(self, monkeypatch: Any, tmp_path: Path) -> None:
        """monkeypatch env INCIDENT_DIR 后 get_graph 接线 FileIncidentReporter。"""
        monkeypatch.setenv("INCIDENT_DIR", str(tmp_path))
        captured: list[Any] = []
        original = _resolve_incident_reporter

        def spy(reporter: Any, store: Any) -> Any:
            resolved = original(reporter, store)
            captured.append(resolved)
            return resolved

        monkeypatch.setattr(desktop_graph, "_resolve_incident_reporter", spy)
        graph = get_graph(checkpointer=None)
        assert graph is not None
        assert len(captured) == 1
        assert isinstance(captured[0], FileIncidentReporter)
        assert captured[0].incident_dir == tmp_path


# ── 6. 红线自查 ───────────────────────────────────────────────────────────────


class TestIncidentReporterRedlines:
    """实现文件不 import desktop_graph（避免循环）、不 import 记忆/存储层。"""

    def test_module_has_no_forbidden_imports(self) -> None:
        import ast

        import src.orchestration.safety.incident_reporter as mod

        source = Path(str(mod.__file__)).read_text(encoding="utf-8")
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

        forbidden = {
            "desktop_graph": "incident_reporter 不得 import desktop_graph（循环）",
            "src.memory": "编排层安全门不得 import 记忆层（层红线）",
            "src.storage": "编排层不得 import 存储层（层红线）",
            "src.mcp": "incident_reporter 无需触达 MCP 层",
        }
        for name in imported:
            for pattern, reason in forbidden.items():
                assert pattern not in name, f"{reason}：实际 import {name!r}"
