"""src/logging_config.py 的行为守卫（跨层基础设施，仓根平铺先例同 test_no_cross_repo_line_refs）。

覆盖面（每条断言都在开发期做过变异实证——见 PR 描述）：
- 默认形态：INFO + "%(asctime)s %(name)s %(levelname)s %(message)s" + stderr 单 handler。
- 接管语义：configure_logging 摘光 root 既有 handler（FastMCP 构造时 SDK 抢注的
  RichHandler 不摘会 stderr 双份日志）；重复调用幂等；被摘 handler 必须 close
  （Windows 下句柄不放会让轮转 rename 报 WinError 32）。
- env 接线：ZERO_MCP_LOG_LEVEL / ZERO_MCP_LOG_FILE 显式 setenv 生效、显式入参优先；
  非法级别 ValueError、落盘路径不可用 OSError——两者都不得破坏已生效的旧配置。
- stdio 安全：任何 handler 不得写 stdout（stdout 是 JSON-RPC 线路）。
- DEBUG 防泄密：第三方 httpx/httpcore/mcp 钉 INFO，重配回非 DEBUG 后恢复。

⚠ 本仓不调用 load_dotenv：env 一律 monkeypatch.setenv 设进程环境，
非法值/缺省分支必须显式设置（夹具"干净环境"会让这些分支结构上不可达）。
"""

from __future__ import annotations

import importlib
import logging
import logging.handlers
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import src.logging_config as logging_config
from src.logging_config import (
    _HANDLER_MARKER,
    _THIRD_PARTY_LOGGERS,
    LOG_FILE_BACKUP_COUNT,
    LOG_FILE_ENV,
    LOG_FILE_MAX_BYTES,
    LOG_FORMAT,
    LOG_LEVEL_ENV,
    configure_logging,
    resolve_log_level,
)

# 旧 desktop_mcp_server basicConfig 的字面量（零回归基准钉死在测试里，
# 不与 LOG_FORMAT 互证——常量改了这里必须红）。
LEGACY_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


def _marked_handlers() -> list[logging.Handler]:
    """取 root 上由 configure_logging 装的 handler（按幂等标记识别）。"""
    root = logging.getLogger()
    return [h for h in root.handlers if getattr(h, _HANDLER_MARKER, False)]


@pytest.fixture(autouse=True)
def clean_logging_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """隔离全局 logging 状态与日志 env：测试前清 env，测试后还原 root/第三方。"""
    monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    monkeypatch.delenv(LOG_FILE_ENV, raising=False)

    root = logging.getLogger()
    handlers_before = list(root.handlers)
    level_before = root.level
    third_party_before = {name: logging.getLogger(name).level for name in _THIRD_PARTY_LOGGERS}

    yield

    for handler in list(root.handlers):
        if handler not in handlers_before:
            root.removeHandler(handler)
            handler.close()
    for handler in handlers_before:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level_before)
    for name, level in third_party_before.items():
        logging.getLogger(name).setLevel(level)


# ── 零回归基准 ────────────────────────────────────────────────────────────────


def test_default_shape_info_legacy_format_stderr() -> None:
    """无 env、无入参：INFO + 旧 format 字面量 + stderr，root 上只剩本模块 handler。"""
    configure_logging()

    root = logging.getLogger()
    assert root.level == logging.INFO
    marked = _marked_handlers()
    assert len(marked) == 1, "默认只应装 1 个 console handler"
    assert root.handlers == marked, "接管语义：root 不得残留任何外来 handler"
    (console,) = marked
    assert isinstance(console, logging.StreamHandler)
    assert console.stream is sys.stderr
    assert console.formatter is not None
    assert console.formatter._fmt == LEGACY_FORMAT
    assert LOG_FORMAT == LEGACY_FORMAT


def test_import_has_no_side_effect() -> None:
    """import（含 reload）路径上不得配置任何全局日志状态。"""
    root = logging.getLogger()
    before_handlers = list(root.handlers)
    before_level = root.level
    importlib.reload(logging_config)
    assert list(root.handlers) == before_handlers
    assert root.level == before_level


# ── env 接线与优先级 ──────────────────────────────────────────────────────────


def test_env_level_wired_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOG_LEVEL_ENV, "debug")
    configure_logging()
    assert logging.getLogger().level == logging.DEBUG


def test_explicit_param_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOG_LEVEL_ENV, "ERROR")
    configure_logging("warning")
    assert logging.getLogger().level == logging.WARNING


def test_blank_env_level_falls_back_to_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOG_LEVEL_ENV, "   ")
    configure_logging()
    assert logging.getLogger().level == logging.INFO


def test_invalid_level_fail_fast_no_half_configured_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非法级别：ValueError（含 env 名/原值/合法域），已生效的旧配置原样保留。

    判别力说明：从干净 root 只调一次再断言"没 handler"是恒真式（「先摘后校验」的
    错误实现也绿，变异实证存活过）——须先成功配置一次，再让第二次失败，
    断言旧配置未被破坏。
    """
    configure_logging()
    before = _marked_handlers()
    assert before, "前置：第一次配置必须已装上 handler"
    before_level = logging.getLogger().level

    monkeypatch.setenv(LOG_LEVEL_ENV, "VERBOSE")
    with pytest.raises(ValueError, match=LOG_LEVEL_ENV) as excinfo:
        configure_logging()
    message = str(excinfo.value)
    assert "'VERBOSE'" in message
    assert "DEBUG|INFO|WARNING|ERROR|CRITICAL" in message
    assert _marked_handlers() == before, "fail-fast 必须发生在触碰 root 之前"
    assert logging.getLogger().level == before_level


def test_resolve_log_level_pure_function(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    assert resolve_log_level() == logging.INFO
    assert resolve_log_level("Critical") == logging.CRITICAL
    with pytest.raises(ValueError, match=LOG_LEVEL_ENV):
        resolve_log_level("TRACE")


# ── 落盘 ─────────────────────────────────────────────────────────────────────


def _file_handlers() -> list[logging.handlers.RotatingFileHandler]:
    return [h for h in _marked_handlers() if isinstance(h, logging.handlers.RotatingFileHandler)]


def test_log_file_env_wired(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """ZERO_MCP_LOG_FILE：轮转参数钉死、UTF-8、父目录自动创建、非 GBK 字符可落盘。"""
    log_path = tmp_path / "nested" / "zero_mcp.log"
    monkeypatch.setenv(LOG_FILE_ENV, str(log_path))
    configure_logging()

    (file_handler,) = _file_handlers()
    assert Path(file_handler.baseFilename) == log_path
    assert file_handler.encoding == "utf-8"
    assert file_handler.maxBytes == LOG_FILE_MAX_BYTES == 10 * 1024 * 1024
    assert file_handler.backupCount == LOG_FILE_BACKUP_COUNT == 5

    # client.py 契约漂移文案含 ⚠ 等非 GBK 字符——encoding 配错时这条会丢
    logging.getLogger("tests.logging_probe").warning("落盘探针 ⚠⇒")
    file_handler.flush()
    assert "落盘探针 ⚠⇒" in log_path.read_text(encoding="utf-8")


def test_log_file_param_beats_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(LOG_FILE_ENV, str(tmp_path / "from_env.log"))
    explicit = tmp_path / "from_param.log"
    configure_logging(log_file=explicit)
    (file_handler,) = _file_handlers()
    assert Path(file_handler.baseFilename) == explicit


def test_blank_log_file_env_means_stderr_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """空串 = 不落盘（.env.example 里 `ZERO_MCP_LOG_FILE=` 置空的语义）。"""
    monkeypatch.setenv(LOG_FILE_ENV, "")
    configure_logging()
    assert _file_handlers() == []
    assert len(_marked_handlers()) == 1


# ── stdio 安全 ───────────────────────────────────────────────────────────────


def test_no_handler_writes_stdout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """stdout 是 stdio MCP 的 JSON-RPC 线路：任何配置组合都不得出现 stdout handler。"""
    monkeypatch.setenv(LOG_FILE_ENV, str(tmp_path / "app.log"))
    configure_logging("DEBUG")
    for handler in _marked_handlers():
        stream = getattr(handler, "stream", None)
        assert stream is not sys.stdout


# ── 幂等 ─────────────────────────────────────────────────────────────────────


def test_takeover_removes_preexisting_handlers_and_stays_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """接管语义：既有 handler（模拟 FastMCP 抢注的 SDK handler）被摘除；重复调用不翻倍。"""
    alien = logging.NullHandler()
    logging.getLogger().addHandler(alien)
    try:
        monkeypatch.setenv(LOG_FILE_ENV, str(tmp_path / "app.log"))
        configure_logging()
        root = logging.getLogger()
        assert alien not in root.handlers, "不摘抢注 handler ⇒ 生产 server stderr 双份日志"
        assert len(_marked_handlers()) == 2  # stderr + file
        assert root.handlers == _marked_handlers()

        configure_logging()
        assert len(_marked_handlers()) == 2, "重配后 handler 数不变（幂等）"
    finally:
        logging.getLogger().removeHandler(alien)


def test_reconfigure_closes_replaced_file_handler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """被替换的落盘 handler 必须 close——Windows 句柄不放会让轮转 rename 报 WinError 32。"""
    monkeypatch.setenv(LOG_FILE_ENV, str(tmp_path / "app.log"))
    configure_logging()
    (old_file_handler,) = _file_handlers()

    configure_logging()
    assert old_file_handler not in logging.getLogger().handlers
    assert old_file_handler.stream is None, "close() 未被调用（FileHandler.close 会置 stream=None）"


def test_bad_log_file_keeps_previous_configuration(tmp_path: Path) -> None:
    """落盘路径不可用（指向已存在目录）：OSError，已生效的旧配置原样保留。"""
    configure_logging(log_file=tmp_path / "good.log")
    before = _marked_handlers()
    before_level = logging.getLogger().level

    blocker = tmp_path / "blocker_dir"
    blocker.mkdir()
    with pytest.raises(OSError):
        configure_logging(log_file=blocker)
    assert _marked_handlers() == before, "OSError 不得留下半配置态（旧 handler 被摘/丢）"
    assert logging.getLogger().level == before_level


# ── DEBUG 防第三方泄密 ────────────────────────────────────────────────────────


def test_third_party_clamped_at_debug_and_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    """DEBUG 时第三方钉 INFO（防请求头含 Bearer token 倾倒）；重配非 DEBUG 恢复 NOTSET。"""
    configure_logging("DEBUG")
    for name in _THIRD_PARTY_LOGGERS:
        assert logging.getLogger(name).level == logging.INFO, name

    configure_logging("INFO")
    for name in _THIRD_PARTY_LOGGERS:
        assert logging.getLogger(name).level == logging.NOTSET, name


def test_third_party_tuple_covers_bearer_bearing_stacks() -> None:
    """httpx/httpcore（Bearer 注入处）与 mcp（SDK）必须在钉住名单里。"""
    assert {"httpx", "httpcore", "mcp"} <= set(_THIRD_PARTY_LOGGERS)


# ── encoding 判别力（子进程）──────────────────────────────────────────────────


def test_file_encoding_explicit_survives_non_utf8_mode(tmp_path: Path) -> None:
    """encoding="utf-8" 必须显式写：用 -X utf8=0 子进程实证。

    判别力说明（变异 M4 的教训）：本机 PYTHONUTF8=1 时 UTF-8 模式会把
    FileHandler(encoding=None) 经 io.text_encoding() 归一成 "utf-8"，进程内的
    encoding 断言对「漏写 encoding」变异无判别力。故起 -X utf8=0 子进程（关掉
    UTF-8 模式兜底、覆盖 env）：届时 encoding=None 在 Windows locale(cp936/cp1252)
    下写 ⚠ 会 UnicodeEncodeError 丢日志，本断言变红；显式 utf-8 则任何 locale 都绿。
    （UTF-8 locale 平台上该变异行为等价，本测试恒绿，不构成 flake。）
    """
    import subprocess  # noqa: PLC0415

    repo_root = Path(__file__).resolve().parents[1]
    log_path = tmp_path / "probe.log"
    child = tmp_path / "probe_child.py"
    child.write_text(
        "import logging\n"
        "import sys\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        "from src.logging_config import configure_logging\n"
        f"configure_logging(log_file={str(log_path)!r})\n"
        'logging.getLogger("tests.logging_probe").warning("落盘探针 ⚠⇒")\n'
        "logging.shutdown()\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-X", "utf8=0", str(child)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "落盘探针 ⚠⇒" in log_path.read_text(encoding="utf-8")
