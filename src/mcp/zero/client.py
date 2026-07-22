"""Zero MCP Client（Python 侧）。

生命周期：async context manager（AsyncExitStack 嵌套 transport + ClientSession）。
工具面：open_session / step / close_session / graceful_step 四个 async 方法。

设计约束（蓝图 Task 1-3）：
- 不 import Zero 代码库；经 call_tool 字符串工具名调用（AD-2）。
- 传输层零业务逻辑：异常封装 + 工具转发，情感/agent 逻辑在 Python src/* 层。
- 跨语言契约：从 src/agents/models/zero_affect（共享契约层）import 数据形状。
- ZERO_LINK_ENABLED=false 时拒绝连接（双侧 flag 检查）；新能力默认关。
- env 传递：StdioServerParameters.env 注入完整 os.environ 副本（同 desktop 侧），
  确保子进程能解析项目 src 包（PYTHONPATH / conda PATH 等由 conda 配置）。
- session_id 不由 client 持有（无状态句柄，可服务多会话）。
- Windows ProactorEventLoop（Python 3.12 默认）：anyio 在 Windows 上用 asyncio backend，
  stdio_client 内部的 create_windows_process 已处理兼容性。
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import types
from typing import Any

import httpx
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError
from mcp.types import TextContent
from pydantic import ValidationError

from src.agents.models.zero_affect import AffectStimulus, ExpressionBundle, ModalityPrior
from src.mcp.zero.external_priors import build_external_priors_override

logger = logging.getLogger(__name__)

# Zero 侧 unknown-session **机读标记**（zero-link T6·②）：与 Zero server
# `src/mcp_server/server.py::_UNKNOWN_SESSION_MARKER` 逐字对齐。step 命中未知/过期 session_id
# 时，Zero 抛的 ToolError 文本以此前缀打头（`f"{_UNKNOWN_SESSION_MARKER}: 未知 session_id=…"`）。
# 用机读前缀而非中文文本判定 → 抗 Zero 侧文案漂移（回执明言「靠字符串匹配脆弱」）；
# 两仓须同步改本常量，漂移由 `tests/mcp/test_zero_contract_crosscheck.py` 跨仓回归拦截。
_UNKNOWN_SESSION_MARKER = "unknown-session"

# ── 自定义异常 ─────────────────────────────────────────────────────────────────


class ZeroLinkDisabledError(RuntimeError):
    """ZERO_LINK_ENABLED=false 时尝试连接/调用抛出。

    客户端侧 flag 检查（双侧检查策略：client 禁用即拒绝连接，不等到 server 报错）。
    """


class ZeroLinkConnectionError(OSError):
    """transport 连接失败或 ClientSession.initialize() 失败。

    附带 stderr 输出供诊断（stdio 模式下尤其有用）。
    """

    def __init__(self, message: str, stderr: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr


class ZeroLinkCallError(RuntimeError):
    """工具调用失败（isError=True 或 McpError）。"""

    def __init__(self, tool: str, message: str) -> None:
        super().__init__(f"[{tool}] {message}")
        self.tool = tool


class ZeroLinkUnknownSessionError(ZeroLinkCallError):
    """step 命中 Zero 侧**未知/过期 session_id**（server 重启或会话已 close）。

    是 `ZeroLinkCallError` 子类（zero-link T6·②）：
    - `graceful_step` 仍按既有分支降级返回 `None`（零回归），但日志显式区分为「可 resume 续会话」；
    - 直接调 `step()` 的编排层可 **catch 本子类** → 用同 id `open_session(session_id=…)` 重开
      续会话再重试（配合 Zero resume-by-id，T6·④），区别于连接失败/畸形响应（不可 resume）。

    判定走机读前缀 `_UNKNOWN_SESSION_MARKER`（非中文文本），抗 Zero 侧文案漂移。
    """


# ── 内部辅助 ───────────────────────────────────────────────────────────────────


def _is_enabled() -> bool:
    """检查客户端侧 ZERO_LINK_ENABLED feature flag。

    宽松真值判定（与 SCREEN_CAPABILITY_ENABLED 解析风格一致）：
    "1" / "true" / "yes"（大小写不敏感）均视为 True，其余视为 False。
    """
    return os.environ.get("ZERO_LINK_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
    }


def _build_subprocess_env() -> dict[str, str]:
    """构造子进程环境变量。

    StdioServerParameters.env 若非 None，SDK 内部会做 {**get_default_environment(), **env}，
    即只继承有限白名单 env（Windows: APPDATA/PATH/TEMP 等）再叠加我们传入的 key。
    为确保子进程能正确解析项目 src 包（依赖 PYTHONPATH / conda PATH 等），
    直接传递完整 os.environ 的副本（字符串化），再显式透传 ZERO_LINK_ENABLED。
    """
    env: dict[str, str] = {k: v for k, v in os.environ.items() if isinstance(v, str)}
    # 显式透传 ZERO_LINK_ENABLED，保证子进程能看到（parent 进程中可能已设）
    env["ZERO_LINK_ENABLED"] = os.environ.get("ZERO_LINK_ENABLED", "false")
    return env


def _build_transport_params() -> tuple[str, Any]:
    """根据 ZERO_LINK_TRANSPORT 构造传输参数。

    Returns:
        ("stdio", StdioServerParameters) 或 ("http", (endpoint_url, token))。

    默认值均为 Zero server 未建前的**临时占位**，可通过 .env 覆盖：
    - ZERO_SERVER_COMMAND：stdio 模式 server 命令（默认 sys.executable）。
    - ZERO_SERVER_ARGS：stdio 模式 server 参数 JSON 列表（默认 ["-m","src.mcp_server"]）。
    - ZERO_SERVER_CWD：stdio 模式 server 工作目录（默认 D:\\Zero）。
    - ZERO_HTTP_ENDPOINT：http 模式 endpoint URL。
    - ZERO_HTTP_TOKEN：http 模式 Bearer token。
    """
    transport = os.getenv("ZERO_LINK_TRANSPORT", "stdio").lower()

    if transport == "stdio":
        command = os.getenv("ZERO_SERVER_COMMAND", sys.executable)
        args_raw = os.getenv("ZERO_SERVER_ARGS", '["-m","src.mcp_server"]')
        args: list[str] = json.loads(args_raw)
        cwd = os.getenv("ZERO_SERVER_CWD", r"D:\Zero")
        params = StdioServerParameters(
            command=command,
            args=args,
            cwd=cwd,
            env=_build_subprocess_env(),
        )
        return ("stdio", params)

    # http 传输
    endpoint = os.getenv("ZERO_HTTP_ENDPOINT", "")
    token = os.getenv("ZERO_HTTP_TOKEN", "")
    return ("http", (endpoint, token))


def _build_http_client(token: str) -> httpx.AsyncClient | None:
    """http 传输的鉴权客户端：有 token 则预置 ``Authorization: Bearer <token>`` 头的
    ``httpx.AsyncClient``（新 SDK ``streamable_http_client`` 不直收 headers，须经 ``http_client``
    注入）；无 token 返回 ``None``（不鉴权——默认 127.0.0.1 本地场景零回归）。

    ⚠ Bearer 是标准方案（RFC 6750 ``Authorization: Bearer <token>``），与 Zero server 侧
    对齐的只是**共享 token 值**（两侧 .env），格式无歧义。抽成独立函数以便单测 header 构造
    （连接路径难在单测里跑，构造逻辑可）。
    """
    if not token:
        return None
    return httpx.AsyncClient(headers={"Authorization": f"Bearer {token}"})


def _is_unknown_session_text(text: str) -> bool:
    """按 Zero 侧机读前缀判定 unknown-session（zero-link T6·②）。

    Zero 的 step ToolError 文本形如 ``"unknown-session: 未知 session_id=…"``——用**前缀**判定
    而非「未知 session」等中文子串，抗 Zero 侧文案改动。容忍前导空白（防未来包裹换行/缩进）。
    仅前缀命中才算，避免把恰好含 "unknown-session" 子串的其它消息误判（判别性）。
    """
    return text.lstrip().startswith(_UNKNOWN_SESSION_MARKER)


def _extract_text(result: Any, tool_name: str) -> str:
    """从 CallToolResult 中提取文本内容。

    result.isError=True 时抛 ZeroLinkCallError；错误文本带 unknown-session 机读前缀时
    抛更精确的 ZeroLinkUnknownSessionError 子类（zero-link T6·②，供上层 resume 判定）。
    content 为空或首元素无 text 属性时也抛错。
    """
    if result.isError:
        err_text = ""
        if result.content and isinstance(result.content[0], TextContent):
            err_text = result.content[0].text
        message = err_text or "server 返回 isError=True"
        # unknown-session 机读标记语义**只对 zero.step 成立**（会话不存在 → 可用同 id resume）；
        # open/close_session 即便文本恰好带前缀也不升级为该子类，保子类语义与触发路径严格对齐
        # （code-review W1）。工具名沿用调用处字面量口径（无常量层）。
        if tool_name == "zero.step" and _is_unknown_session_text(err_text):
            raise ZeroLinkUnknownSessionError(tool_name, message)
        raise ZeroLinkCallError(tool_name, message)

    if not result.content:
        raise ZeroLinkCallError(tool_name, "server 返回空 content")

    first = result.content[0]
    if not isinstance(first, TextContent):
        raise ZeroLinkCallError(
            tool_name,
            f"期望 TextContent，得到 {type(first).__name__}",
        )
    return first.text


# ── 主类 ───────────────────────────────────────────────────────────────────────


class ZeroLinkClient:
    """Zero MCP Client，async context manager。

    把 Zero 当外部服务经 MCP call_tool 调用（不 import Zero 代码库，AD-2）。
    session_id 不由 client 持有，单实例可服务多 Zero 会话（无状态句柄）。

    用法::

        async with ZeroLinkClient() as client:
            sid = await client.open_session(persona="default")
            bundle = await client.step(sid, AffectStimulus(valence=0.3, arousal=0.5))
            await client.close_session(sid)

    生命周期：
        - __aenter__：flag 检查 → transport 连接 → ClientSession.initialize()。
        - __aexit__：session=None，关 AsyncExitStack（清理 transport + session）。
        - 连接失败统一包装为 ZeroLinkConnectionError（stdio 尽量带 stderr 诊断）。
    """

    def __init__(self) -> None:
        """初始化 ZeroLinkClient。

        无参构造：传输参数（stdio 命令/cwd 或 http endpoint/token）全部由 .env 提供
        （见 _build_transport_params）。session_id 不由 client 持有，单实例可服务多
        Zero 会话（无状态句柄）。
        """
        self.exit_stack: contextlib.AsyncExitStack | None = None
        self.session: ClientSession | None = None

    # ── context manager ───────────────────────────────────────────────────────

    async def __aenter__(self) -> ZeroLinkClient:
        # 1. 客户端侧 feature flag 检查
        if not _is_enabled():
            raise ZeroLinkDisabledError(
                "Zero Link 未启用（ZERO_LINK_ENABLED=false）。"
                "请设置 ZERO_LINK_ENABLED=true 后重试。"
            )

        # 2. 选择传输
        transport_kind, transport_params = _build_transport_params()

        # 3. AsyncExitStack 嵌套管理 transport + ClientSession
        stack = contextlib.AsyncExitStack()
        try:
            await stack.__aenter__()

            if transport_kind == "stdio":
                # stdio_client yield (read, write)
                read_stream, write_stream = await stack.enter_async_context(
                    stdio_client(transport_params)
                )
            else:
                # http 传输：streamable_http_client(url, *, http_client) yield 三元组。
                # 新 API 不直接收 headers——Bearer token 经预置 httpx.AsyncClient 注入。
                endpoint, token = transport_params
                http_client = _build_http_client(token)
                if http_client is not None:
                    await stack.enter_async_context(http_client)
                read_stream, write_stream, _get_session_id = await stack.enter_async_context(
                    streamable_http_client(endpoint, http_client=http_client)
                )

            # 建立 ClientSession
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            try:
                await session.initialize()
            except Exception as exc:
                raise ZeroLinkConnectionError(
                    f"ClientSession 初始化失败：{exc}",
                    stderr="",
                ) from exc

            self.exit_stack = stack
            self.session = session
            logger.info(
                "ZeroLinkClient 连接成功（transport=%s）",
                transport_kind,
            )

        except ZeroLinkConnectionError:
            if self.exit_stack is None:
                try:
                    await stack.aclose()
                except Exception:
                    pass
            raise
        except Exception as exc:
            if self.exit_stack is None:
                try:
                    await stack.aclose()
                except Exception:
                    pass
            raise ZeroLinkConnectionError(
                f"Zero Link 连接失败（transport={transport_kind}）：{exc}",
                stderr="",
            ) from exc

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.session = None
        if self.exit_stack is not None:
            await self.exit_stack.__aexit__(exc_type, exc_val, exc_tb)
            self.exit_stack = None

    # ── 内部工具调用 ──────────────────────────────────────────────────────────

    def _require_session(self) -> ClientSession:
        """断言 session 存在（在 context 内调用时始终满足）。

        Raises:
            ZeroLinkConnectionError: 在 async with 块外调用时。
        """
        if self.session is None:
            raise ZeroLinkConnectionError("ZeroLinkClient 尚未初始化，请在 async with 块内使用。")
        return self.session

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """调用 MCP 工具并返回 TextContent.text。

        Args:
            tool_name: 工具名（与 Zero server 注册名一致，如 "zero.open_session"）。
            arguments: 工具调用参数字典。

        Returns:
            server 返回的 JSON 字符串（调用方负责解析）。

        Raises:
            ZeroLinkCallError: server 返回 isError=True 或 McpError。
        """
        session = self._require_session()
        try:
            result = await session.call_tool(tool_name, arguments)
        except McpError as exc:
            raise ZeroLinkCallError(tool_name, str(exc)) from exc
        return _extract_text(result, tool_name)

    # ── 公开工具方法 ──────────────────────────────────────────────────────────

    async def open_session(
        self,
        *,
        persona: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:
        """在 Zero 侧创建一个新会话，返回 session_id。

        生命周期失败须明确报错（不 graceful），上层须显式处理异常。

        Args:
            persona: 可选人格标识（传给 Zero zero.open_session 工具）。
            config:  可选配置字典（传给 Zero zero.open_session 工具）。

        Returns:
            Zero 分配的 session_id 字符串。

        Raises:
            ZeroLinkCallError:      工具调用失败（server 返回 isError=True 或协议错误）。
            ZeroLinkConnectionError: 未在 async with 内调用。
        """
        args: dict[str, Any] = {}
        if persona is not None:
            args["persona"] = persona
        if config is not None:
            args["config"] = config
        text = await self._call_tool("zero.open_session", args)
        # 响应解析防御：畸形 JSON / 缺 session_id 键统一封装为 ZeroLinkCallError，
        # 不让原始 JSONDecodeError/KeyError 穿透异常封装边界（调用方只预期 ZeroLink* 异常）。
        try:
            data: dict[str, Any] = json.loads(text)
            session_id: str = data["session_id"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ZeroLinkCallError("zero.open_session", f"响应格式非预期：{exc}") from exc
        return session_id

    async def step(
        self,
        session_id: str,
        stimulus: AffectStimulus,
        priors: list[ModalityPrior] | None = None,
    ) -> ExpressionBundle:
        """向 Zero 发送单步情感刺激，返回表达包。

        Args:
            session_id: 由 open_session() 获得的 Zero 会话 ID。
            stimulus:   情感刺激（valence/arousal/coping_potential）。
            priors:     可选多模态先验列表（非空时构造 external_priors 载荷注入）。

        Returns:
            ExpressionBundle 解析结果。

        Raises:
            ValueError:              priors 不满足 M3/M6 约束（由 build_external_priors_override
                                     抛出，不 catch——调用方参数错误须透传，fail-fast）。
            ZeroLinkCallError:       工具调用失败。
            ZeroLinkConnectionError: 未在 async with 内调用。
        """
        # exclude_none=True：coping_potential 为 None 时略去该键（最小合法载荷），
        # 避免线上 stim 带 "coping_potential": null 依赖 Zero 侧对 null 可选字段的宽容性。
        stim_dict: dict[str, Any] = stimulus.model_dump(exclude_none=True)
        arguments: dict[str, Any] = {"session_id": session_id, "stim": stim_dict}

        if priors:
            override = build_external_priors_override(priors)
            # tuple→list 显式转换（可见可测，不靠 json 隐式处理）
            arguments["external_priors"] = [
                [name, list(mu), list(precision)]
                for name, mu, precision in override["external_priors"]
            ]

        text = await self._call_tool("zero.step", arguments)
        # 响应解析防御：畸形 JSON / expression 结构不合契约统一封装为 ZeroLinkCallError，
        # 使 graceful_step 能兜住畸形响应降级为 None（ValidationError 是 ValueError 子类，
        # 但此处是 server 响应问题而非调用方 M3/M6 参数错误——后者在 _call_tool 之前已抛）。
        try:
            return ExpressionBundle.from_step_output(json.loads(text))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ZeroLinkCallError("zero.step", f"响应解析失败：{exc}") from exc

    async def close_session(self, session_id: str) -> None:
        """关闭 Zero 侧的会话。

        生命周期失败须明确报错（不 graceful），上层须显式处理异常。

        Args:
            session_id: 要关闭的 Zero 会话 ID。

        Raises:
            ZeroLinkCallError:       工具调用失败。
            ZeroLinkConnectionError: 未在 async with 内调用。
        """
        await self._call_tool("zero.close_session", {"session_id": session_id})

    async def graceful_step(
        self,
        session_id: str | None,
        stimulus: AffectStimulus,
        priors: list[ModalityPrior] | None = None,
    ) -> ExpressionBundle | None:
        """容错版单步调用，供编排层在「非关键路径」降级使用。

        不 catch ValueError（M3/M6 fail-fast 是调用方参数错误，须透传）。
        以下情况静默返回 None：
        - ZERO_LINK_ENABLED=false（未启用）。
        - session_id 为 None（会话未建立）。
        - ZeroLinkCallError / ZeroLinkConnectionError / McpError（连接/调用失败）。
        - ZeroLinkUnknownSessionError（unknown-session 子类，session 未知/过期）：同降级返回 None，
          但日志显式标明「可用同 id resume」，便于上层据此触发 resume（zero-link T6·②·④）。

        Args:
            session_id: Zero 会话 ID（None 时立即返回 None）。
            stimulus:   情感刺激。
            priors:     可选多模态先验列表。

        Returns:
            ExpressionBundle 或 None（降级时）。

        Raises:
            ValueError: priors 不满足 M3/M6 约束（透传，不 graceful）。
        """
        if not _is_enabled():
            logger.debug("graceful_step: ZERO_LINK_ENABLED=false，跳过")
            return None
        if session_id is None:
            logger.debug("graceful_step: session_id=None，跳过")
            return None
        try:
            return await self.step(session_id, stimulus, priors)
        except ZeroLinkUnknownSessionError:
            # 机读标记命中 Zero 侧未知/过期 session（server 重启 / 会话已 close）：这是「可用同 id
            # resume 重开续会话」的可恢复态，区别于连接/畸形失败。graceful 契约仍返回 None，
            # 但日志显式区分（零回归），便于上层据此触发 resume（T6·②·④）而非当作不可恢复失败。
            logger.warning(
                "graceful_step 降级（session=%s）：Zero unknown-session（未知/过期）；"
                "上层可用同 id resume 重开续会话。",
                session_id,
            )
            return None
        except (ZeroLinkCallError, ZeroLinkConnectionError, McpError) as exc:
            logger.warning(
                "graceful_step 降级（session=%s, exc=%s）：%s",
                session_id,
                type(exc).__name__,
                exc,
            )
            return None
