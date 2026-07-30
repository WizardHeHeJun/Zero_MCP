"""Zero_MCP 跨层日志统一配置（仅进程入口点调用）。

分工（全仓既有约定）：库模块只做 ``logger = logging.getLogger(__name__)``，
不在 import 路径上做任何全局配置；本模块的 :func:`configure_logging` 只在
进程入口（``if __name__ == "__main__"`` 块 / 未来的编排 CLI）显式调用。
本模块位于依赖图最底端（无业务逻辑、无上层依赖），任何层 import 它都是
向下依赖——同 ``src/agents/models/`` 共享契约层的豁免判据（project-root.md）。

环境变量（⚠ 本仓不调用 load_dotenv：写进 .env 不会自动生效，须设进程环境；
本仓 client 显式整份透传 os.environ 给 spawn 的子进程故子进程同样生效，
外部 host 按 MCP SDK 默认最小 env spawn 时**不**继承、回默认值）：

- ``ZERO_MCP_LOG_LEVEL``：DEBUG|INFO|WARNING|ERROR|CRITICAL（大小写不敏感）。
  缺省/空 = INFO；非法值 ValueError fail-fast（与数值类 env 的既有分级一致，
  不静默回落）。
- ``ZERO_MCP_LOG_FILE``：日志落盘路径。缺省/空 = 不落盘（仅 stderr）；
  设置后追加 UTF-8 RotatingFileHandler，父目录自动创建。

安全约束：stdio MCP server 的 stdout 是 JSON-RPC 线路——本模块不提供任何
写 stdout 的路径，console handler 恒为 stderr。
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

LOG_LEVEL_ENV = "ZERO_MCP_LOG_LEVEL"
LOG_FILE_ENV = "ZERO_MCP_LOG_FILE"

LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
"""全仓统一日志格式。字面量沿用 desktop_mcp_server 旧内联 basicConfig（该处
在 FastMCP 抢注 root 之后实为 no-op 死码，见 configure_logging 接管语义）。"""

_LEVEL_BY_NAME: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

VALID_LEVELS: tuple[str, ...] = tuple(_LEVEL_BY_NAME)

# 落盘轮转参数（工程假设：数字人宿主是长跑进程，10MB×5 ≈ 上限 60MB，足够回溯
# 数小时会话且不吃满磁盘；无文献依据，按实际量级再调）。
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024
LOG_FILE_BACKUP_COUNT = 5

# 本模块装的 handler 带此标记，供测试/诊断识别归属（接管语义下摘除是无差别的，
# 标记不参与摘除判定）。
_HANDLER_MARKER = "zero_mcp_configured"

# root 降到 DEBUG 时钉住的第三方命名空间。覆盖面（实测 mcp 1.28.1，SDK 升级后复核）：
# - httpx/httpcore 全层级——Bearer token（ZERO_HTTP_TOKEN）经 httpx 请求头注入，
#   wire 级 DEBUG 倾倒的主威胁路径在 httpcore，被完整钉住；
# - mcp 只覆盖其 mcp.* 命名模块；SDK 内个别模块用裸名 logger（"client"/"server"/
#   "client.stdio.win32"），不在此层级内——现核其输出无敏感信息，不钉（钉裸名
#   "client"/"server" 会误伤其他库）。
_THIRD_PARTY_LOGGERS = ("httpx", "httpcore", "mcp")


def resolve_log_level(level: str | None = None) -> int:
    """解析日志级别：显式入参 > ZERO_MCP_LOG_LEVEL > INFO（大小写不敏感）。

    Args:
        level: 显式级别名；None 时读环境变量。

    Returns:
        logging 数值级别。

    Raises:
        ValueError: 级别名不在合法域（消息含 env 名、原值与合法域，fail-fast
            不静默回落——静默回落会让配错的 env 看起来"生效了"）。
    """
    raw = level if level is not None else os.environ.get(LOG_LEVEL_ENV)
    if raw is None or not raw.strip():
        return logging.INFO
    name = raw.strip().upper()
    if name not in _LEVEL_BY_NAME:
        raise ValueError(
            f"{LOG_LEVEL_ENV} 非法值 {raw!r}：合法域 {'|'.join(VALID_LEVELS)}（大小写不敏感）"
        )
    return _LEVEL_BY_NAME[name]


def resolve_log_file(log_file: str | Path | None = None) -> Path | None:
    """解析落盘路径：显式入参 > ZERO_MCP_LOG_FILE > None（不落盘）。

    Args:
        log_file: 显式路径；None 时读环境变量。

    Returns:
        落盘路径；缺省/空白 = None（仅 stderr）。
    """
    raw: str | Path | None = log_file if log_file is not None else os.environ.get(LOG_FILE_ENV)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return Path(text)


def configure_logging(
    level: str | None = None,
    *,
    log_file: str | Path | None = None,
) -> None:
    """接管 root logger：stderr console handler + 可选 UTF-8 轮转落盘。

    接管语义（工程假设）：root 上的既有 handler **全部摘除并 close**——进程
    入口是 root 日志形态的唯一所有者。动因（实测 mcp 1.28.1）：FastMCP 构造时
    SDK 即抢注 root（RichHandler(stderr) + "%(message)s"），旧内联 basicConfig
    因 root 已有 handler 而恒 no-op；若只追加不接管，每条日志 stderr 双份。
    因此**只在进程入口调用**；库模块与测试夹具不得调用（会摘掉 caplog handler）。

    第三方 logger 的 level 归本函数所有：DEBUG 时钉 _THIRD_PARTY_LOGGERS 为
    INFO（防请求头含 Bearer token 倾倒），非 DEBUG 恢复 NOTSET——每次调用
    重设，入口调用前对这些 logger 的手工 setLevel 会被覆盖。

    失败语义：级别非法 ValueError、落盘路径不可用 OSError，都在触碰 root
    之前抛出（新 handler 先构建成功再做替换），失败不留半配置态。

    Args:
        level: 显式级别名（优先于 ZERO_MCP_LOG_LEVEL）。
        log_file: 显式落盘路径（优先于 ZERO_MCP_LOG_FILE）。

    Raises:
        ValueError: 级别非法（见 :func:`resolve_log_level`）。
        OSError: 落盘路径不可用（父目录创建失败 / 路径是目录 / 无权限）。
    """
    level_no = resolve_log_level(level)
    file_path = resolve_log_file(log_file)

    # 先把新 handler 全部构建成功——mkdir/开文件的 OSError 在此抛出，root 未动。
    formatter = logging.Formatter(LOG_FORMAT)
    new_handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if file_path is not None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # encoding 必须显式 utf-8：日志文案含 ⚠/⇒ 等非 GBK 字符（如 client.py
        # 的契约漂移警告），Windows 默认 locale(cp936) 会 UnicodeEncodeError 丢日志。
        new_handlers.append(
            logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=LOG_FILE_MAX_BYTES,
                backupCount=LOG_FILE_BACKUP_COUNT,
                encoding="utf-8",
            )
        )
    for handler in new_handlers:
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_MARKER, True)

    # 再替换：摘光 root 既有 handler（接管语义）→ 装新 → 设级。close 旧 handler
    # 释放文件句柄——Windows 下不 close 会让轮转 os.rename 报 WinError 32。
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    for handler in new_handlers:
        root.addHandler(handler)
    root.setLevel(level_no)

    third_party_level = logging.INFO if level_no <= logging.DEBUG else logging.NOTSET
    for name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(third_party_level)
