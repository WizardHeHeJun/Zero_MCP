"""桌面会话状态探测（锁屏 / 安全桌面）。

feat/desktop-hardening K2：锁屏下 mss 采到的是锁屏前的**旧帧**、PrintWindow 常整幅
黑屏、注入的输入事件会落到凭据界面或被系统吞掉——感知与操控在锁定桌面下都
不可信。本模块提供唯一的同步探测原语 `_is_desktop_locked_sync`，感知侧
（do_screen_snapshot 开头）与操控侧（全部写动作入口）经 asyncio.to_thread 调用。

探测原理：OpenInputDesktop(0, False, DESKTOP_SWITCHDESKTOP) 尝试以「可切换」
访问权打开**当前输入桌面**——工作站锁定（输入桌面切到 Winlogon）时返回 NULL。
**UAC 安全桌面（提权确认弹窗）同样会被判为锁定，这属语义正确**：安全桌面期间
用户桌面同样收不到注入输入、像素同样不可信，拒绝操控/跳过感知是正确行为。

容错口径：探测**自身**抛异常时按「未锁」处理并回报 probe_failed=True（调用方落
机读降级标记 ``lock_probe_failed``）——探测是防护增强，服务态下不为它新增硬失败面。

Win32 纪律（perception.py _pw_* 先例 / ai-docs/pitfalls.md「ctypes.windll 是进程级
共享对象」）：进程级共享的 ctypes.windll 会被第三方库（pyautogui/pywinauto 等）
篡改签名，故本模块持有**私有 ctypes.WinDLL 实例**并声明全量 argtypes/restype。
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import logging

logger = logging.getLogger(__name__)

# OpenInputDesktop 所需最小访问权：能「切换到该桌面」即证明它是可交互的用户桌面
DESKTOP_SWITCHDESKTOP = 0x0100

# 私有 DLL 实例 + 全量显式签名（勿改用进程级共享的 ctypes.windll，见模块 docstring）
_ss_user32 = ctypes.WinDLL("user32", use_last_error=True)
_ss_user32.OpenInputDesktop.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_ss_user32.OpenInputDesktop.restype = wintypes.HDESK
_ss_user32.CloseDesktop.argtypes = [wintypes.HDESK]
_ss_user32.CloseDesktop.restype = wintypes.BOOL


def _is_desktop_locked_sync() -> tuple[bool, bool]:
    """探测当前输入桌面是否锁定（阻塞，应在 asyncio.to_thread 中调用）。

    Returns:
        (desktop_locked, probe_failed)：

        - desktop_locked=True：OpenInputDesktop 返回 NULL——桌面锁定或处于
          UAC 安全桌面（后者同判锁定属语义正确，见模块 docstring）；
        - probe_failed=True：探测自身抛异常，**容错判「未锁」**
          （desktop_locked=False），调用方应落机读降级标记 ``lock_probe_failed``。

    F6（code-review INFO）：OpenInputDesktop 成功取得句柄后，CloseDesktop 的
    清理尝试放在 `finally`——保证「已确定 desktop_locked=False」这一判定结果
    不因清理阶段的异常而丢失（原结构里 CloseDesktop 若自身抛异常，会被外层
    `except` 一并吞掉、错误上报为 probe_failed=True，虽然探测本身已经成功）。
    CloseDesktop 失败仅记日志（句柄可能未释放，不影响本次探测结果，也不新增
    硬失败面——与「探测异常容错为未锁」同一工程假设）。
    """
    try:
        hdesk = _ss_user32.OpenInputDesktop(0, False, DESKTOP_SWITCHDESKTOP)
    except Exception as exc:  # noqa: BLE001  — 探测容错为「未锁」+ lock_probe_failed
        logger.warning("桌面锁定探测失败（按未锁处理，降级标记 lock_probe_failed）：%s", exc)
        return False, True

    if not hdesk:
        return True, False

    try:
        return False, False
    finally:
        try:
            _ss_user32.CloseDesktop(hdesk)
        except Exception as exc:  # noqa: BLE001  — 清理失败不回滚已确定的探测结果
            logger.warning("CloseDesktop 清理失败（句柄可能未释放，不影响本次探测结果）：%s", exc)
