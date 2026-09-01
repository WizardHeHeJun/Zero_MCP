"""`ZERO_LINK_E2E_STRICT` 覆盖归零守卫（`tests/mcp/conftest.py`）的判别性回归。

守卫本身是**测试基础设施**：它若失效不会有任何用例变红——正是它要防的那种静默。故此处既测纯
函数决策矩阵，也**真起 pytest 子进程**端到端验「设 env 才转 fail、不设逐字零回归、非 zerorepo
用例不受影响」。子进程用 tmp_path 造微型工程，其 conftest 直接 import 本仓的 hook。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.mcp.conftest import (
    STRICT_ENV,
    should_convert_skip,
    strict_failure_message,
    zerorepo_strict_enabled,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


class TestStrictEnvParsing:
    """`zerorepo_strict_enabled()` 的 env 解析——默认关是零回归的前提。"""

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "True", "yes", "on", " 1 "])
    def test_truthy_values_enable(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv(STRICT_ENV, raw)
        assert zerorepo_strict_enabled() is True

    @pytest.mark.parametrize("raw", ["", "   ", "0", "false", "no", "off", "maybe"])
    def test_falsy_values_disable(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv(STRICT_ENV, raw)
        assert zerorepo_strict_enabled() is False

    def test_unset_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(STRICT_ENV, raising=False)
        assert zerorepo_strict_enabled() is False


class TestShouldConvertSkipMatrix:
    """决策矩阵：三条件全真才转 fail（判别性——逐条翻假都不该转）。"""

    def test_converts_when_all_conditions_hold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(STRICT_ENV, "1")
        assert should_convert_skip(report_skipped=True, has_zerorepo_marker=True) is True

    def test_not_converted_when_strict_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """零回归：不设 env 时 zerorepo 的 skip 仍是 skip。"""
        monkeypatch.delenv(STRICT_ENV, raising=False)
        assert should_convert_skip(report_skipped=True, has_zerorepo_marker=True) is False

    def test_not_converted_without_marker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """作用域：非 zerorepo 用例（如缺可选依赖的普通 skip）不受 STRICT 影响。"""
        monkeypatch.setenv(STRICT_ENV, "1")
        assert should_convert_skip(report_skipped=True, has_zerorepo_marker=False) is False

    def test_not_converted_when_not_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(STRICT_ENV, "1")
        assert should_convert_skip(report_skipped=False, has_zerorepo_marker=True) is False


class TestStrictFailureMessage:
    """报告文案须带 nodeid 与原始 skip 原因，否则转红后反而更难排障。"""

    def test_message_carries_nodeid_and_reason(self) -> None:
        msg = strict_failure_message("tests/mcp/test_x.py::TestY::test_z", "torch 未安装")
        assert "tests/mcp/test_x.py::TestY::test_z" in msg
        assert "torch 未安装" in msg
        assert STRICT_ENV in msg


def _write_probe_project(root: Path) -> None:
    """造微型 pytest 工程：一个 zerorepo skip + 一个普通 skip；conftest 复用本仓 hook。"""
    (root / "conftest.py").write_text(
        "from tests.mcp.conftest import pytest_runtest_makereport  # noqa: F401\n",
        encoding="utf-8",
    )
    # 独立 pytest.ini → rootdir 落在 tmp_path，不继承本仓 addopts/asyncio 配置
    (root / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    zerorepo: 跨仓契约回归（探针用）\n",
        encoding="utf-8",
    )
    (root / "test_probe.py").write_text(
        "import pytest\n"
        "\n"
        "\n"
        "@pytest.mark.zerorepo\n"
        "def test_marked_skips():\n"
        '    pytest.skip("探针：模拟跨仓前置缺失")\n'
        "\n"
        "\n"
        "def test_plain_skips():\n"
        '    pytest.skip("探针：非 zerorepo，应始终保持 skip")\n',
        encoding="utf-8",
    )


def _run_probe(root: Path, *, strict: bool) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT)  # 供探针 conftest import tests.mcp.conftest
    # 双侧钉 UTF-8（2026-09-01 实测缺陷）：Windows 下 text=True 不带 encoding 会按
    # 系统 GBK 解码子进程输出——子 pytest 输出一旦出现 GBK 解不了的 UTF-8 字节
    # （随 Zero HEAD 前进、skip 理由含中文/特殊字符时触发）即 UnicodeDecodeError
    # 假红。PYTHONIOENCODING 钉子进程自身 stdout，encoding 钉父进程解码。
    env["PYTHONIOENCODING"] = "utf-8"
    if strict:
        env[STRICT_ENV] = "1"
    else:
        env.pop(STRICT_ENV, None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(root), "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=180,
    )


class TestStrictGuardEndToEnd:
    """真起 pytest 子进程验 hook 生效——纯函数测试证明不了 hook 真的被 pytest 调用。"""

    def test_strict_on_converts_only_marked_skip(self, tmp_path: Path) -> None:
        _write_probe_project(tmp_path)
        result = _run_probe(tmp_path, strict=True)

        assert "1 failed" in result.stdout, (
            f"STRICT=1 时 zerorepo 的 skip 应转 fail，实际输出：\n{result.stdout}\n{result.stderr}"
        )
        assert "1 skipped" in result.stdout, (
            f"非 zerorepo 的 skip 应仍为 skip（作用域不外溢），实际输出：\n{result.stdout}"
        )
        assert STRICT_ENV in result.stdout, (
            f"失败报告应点名 {STRICT_ENV} 以指明病因，实际输出：\n{result.stdout}"
        )

    def test_strict_off_keeps_both_skipped(self, tmp_path: Path) -> None:
        """零回归负对照：不设 env → 两条都仍是 skip、无 failed。"""
        _write_probe_project(tmp_path)
        result = _run_probe(tmp_path, strict=False)

        assert "2 skipped" in result.stdout, (
            f"未设 {STRICT_ENV} 时行为应逐字不变（2 skipped），实际输出：\n{result.stdout}"
        )
        assert "failed" not in result.stdout, (
            f"未设 {STRICT_ENV} 时不应有 failed，实际输出：\n{result.stdout}"
        )
