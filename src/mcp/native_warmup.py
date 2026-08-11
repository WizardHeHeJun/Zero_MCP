"""MCP stdio server 的原生扩展预热（numpy 系 import 死锁规避）。

## 背景（2026-08-11 现场复现，Zero 跨仓通报「vts_connect 经 MCP 调用时挂起」）

在 FastMCP stdio server **已进入事件循环之后**（``mcp.run()`` 之内）首次
``import numpy``——或任何传递性 import numpy 的包（scipy / onnxruntime /
RapidOCR …）——进程会**无限期**卡死在 ``numpy._core._multiarray_umath``
的扩展模块加载（``importlib._bootstrap_external.create_module`` →
Windows ``LoadLibrary``）：既不返回也不抛错，调用方只看到工具调用挂起。
本仓 ``vts_behavior_mcp_server.vts_connect`` 的 ``_get_service()`` 延迟
import 正踩此坑——Zero 侧 25s 超时放弃，而进程内直连同一份代码正常。

## 现场核验过的边界（最小复现 = 一个只在 tool 体里写 ``import numpy`` 的
## stock FastMCP server，不涉及本仓任何代码）

- 同一 import 放在 ``mcp.run()`` **之前**：正常，0.0x s 完成 → 即本模块的修法。
- 塞进 ``asyncio.to_thread`` 里 import：**无效**，照样死锁。
- 非 numpy 系原生扩展（``sqlite3`` / ``decimal`` / ``zoneinfo`` / ``mss`` /
  ``PIL.Image``）不复现；``onnxruntime`` / ``scipy.signal`` 复现（均拉 numpy）。
- 与 BLAS 线程数无关：``OPENBLAS_NUM_THREADS`` / ``MKL_NUM_THREADS`` /
  ``OMP_NUM_THREADS`` 全设 1 仍死锁。
- 卡死可持续 ≥120s（每 15s faulthandler dump 栈帧完全一致），客户端取消该
  请求后才解除——故这是**阻塞**，不是「冷 import 慢」。

Windows loader 层面的确切成因未查到底（不臆断）；上述边界足以支撑修法。

## 用法（进程入口，``mcp.run()`` 之前）

    if enabled:
        warm_native_extensions(("src.mcp.behavior.service",))
    mcp.run(transport="stdio")

约束同 ``src/logging_config.py``：本模块位于依赖图最底端（无业务逻辑、无上层
依赖），任何层 import 它都是向下依赖；只在进程入口调用，不在 import 路径上
产生副作用。预热失败**不拦启动**（缺包场景仍应能起 server，由工具体自行
优雅回退），只记 warning。
"""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Sequence

logger = logging.getLogger(__name__)


def warm_native_extensions(modules: Sequence[str]) -> list[str]:
    """在事件循环启动前逐个 import ``modules``，把原生扩展加载掉。

    必须在 ``mcp.run()`` **之前**调用——这正是规避死锁的全部要点（模块
    docstring）。之后 tool 体里的同名延迟 import 命中 ``sys.modules``，
    退化为常数开销。

    Args:
        modules: 待预热的模块名（按依赖顺序给出即可，重复无害）。

    Returns:
        实际预热成功的模块名列表（失败项不在内，只记 warning 不抛）。
    """
    warmed: list[str] = []
    for name in modules:
        start = time.monotonic()
        try:
            importlib.import_module(name)
        except Exception as exc:
            # 缺包/环境不全不应拦住 server 启动：工具体各自有优雅回退路径。
            logger.warning("原生扩展预热失败（%s）：%s——相关工具将按缺依赖回退。", name, exc)
            continue
        warmed.append(name)
        logger.debug("原生扩展预热完成：%s（%.2fs）", name, time.monotonic() - start)
    return warmed
