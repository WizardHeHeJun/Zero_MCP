"""test_session_state.py — 桌面锁定探测三态单测（mock 私有 WinDLL，不真调 Win32）。

覆盖（K2）：
- OpenInputDesktop 返回 NULL → (locked=True, probe_failed=False)，不调 CloseDesktop
- OpenInputDesktop 返回句柄 → (False, False)，句柄必被 CloseDesktop 关闭
- 探测自身抛异常 → (False, True)（容错判「未锁」+ lock_probe_failed 语义）

判别力实证（绿灯先证能红）：临时把 `if not hdesk` 改为 `if hdesk`（反转判定）
→ NULL/句柄两用例均红；临时把 except 分支改为 `return True, False` → 异常用例红。
详见分支落地报告的 mutation_evidence。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import src.mcp.desktop.session_state as ss_mod


def test_locked_when_open_input_desktop_null(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenInputDesktop 返回 NULL → 判锁定；无句柄可关，CloseDesktop 不被调。"""
    fake_user32 = MagicMock()
    fake_user32.OpenInputDesktop.return_value = None  # HDESK restype：NULL → None
    monkeypatch.setattr(ss_mod, "_ss_user32", fake_user32)

    assert ss_mod._is_desktop_locked_sync() == (True, False)
    fake_user32.OpenInputDesktop.assert_called_once_with(0, False, ss_mod.DESKTOP_SWITCHDESKTOP)
    fake_user32.CloseDesktop.assert_not_called()


def test_unlocked_closes_desktop_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenInputDesktop 返回句柄 → 判未锁；句柄必须 CloseDesktop（不泄漏）。"""
    fake_user32 = MagicMock()
    fake_user32.OpenInputDesktop.return_value = 0x1234
    monkeypatch.setattr(ss_mod, "_ss_user32", fake_user32)

    assert ss_mod._is_desktop_locked_sync() == (False, False)
    fake_user32.CloseDesktop.assert_called_once_with(0x1234)


def test_probe_exception_reports_probe_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """探测自身抛异常 → 容错判「未锁」+ probe_failed=True（不新增硬失败面）。"""
    fake_user32 = MagicMock()
    fake_user32.OpenInputDesktop.side_effect = OSError("access denied")
    monkeypatch.setattr(ss_mod, "_ss_user32", fake_user32)

    assert ss_mod._is_desktop_locked_sync() == (False, True)


def test_close_desktop_failure_does_not_flip_probe_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """F6（code-review INFO）：CloseDesktop 清理阶段自身抛异常，不回滚已确定的
    探测结果——OpenInputDesktop 已成功证明未锁，清理失败只记日志、不误报
    probe_failed=True（区别于 OpenInputDesktop 自身失败的语义，见上一用例）。
    """
    fake_user32 = MagicMock()
    fake_user32.OpenInputDesktop.return_value = 0x1234
    fake_user32.CloseDesktop.side_effect = OSError("handle already closed")
    monkeypatch.setattr(ss_mod, "_ss_user32", fake_user32)

    assert ss_mod._is_desktop_locked_sync() == (False, False)
    fake_user32.CloseDesktop.assert_called_once_with(0x1234)
