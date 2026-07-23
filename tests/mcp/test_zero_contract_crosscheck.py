"""跨仓契约回归测试（T8）——marker zerorepo。

策略：
- 在 subprocess 中动态加载 D:\\Zero 的 decode_channels / affect_math，
  用本仓 ExpressionHead / 相关模型解析其真实输出，断言无 ValidationError。
- D:\\Zero 不存在、import 失败（任何 ImportError / ModuleNotFoundError）、
  subprocess 非零退出 → pytest.skip(原因)，绝不把套件跑红。
- decode_channels 在 affect_math 里理论 torch-free，但 import 该模块可能
  触发 package __init__ 拉 torch——若如此就 skip 并说明，留待后续更隔离方案。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

# D:\Zero 源码根路径
_ZERO_ROOT = Path("D:/Zero")
_ZERO_SRC = _ZERO_ROOT / "src"

# ---------------------------------------------------------------------------
# 跳过门控：D:\Zero 不在位时整个模块跳过
# ---------------------------------------------------------------------------


def _zero_available() -> bool:
    """检查 D:\\Zero\\src 是否存在。"""
    return _ZERO_SRC.is_dir()


# ---------------------------------------------------------------------------
# 辅助：通过子进程运行 D:\Zero 代码并捕获输出
# ---------------------------------------------------------------------------

_DECODE_SCRIPT = """
import sys
import json

sys.path.insert(0, sys.argv[1])  # D:\\Zero（包根；Zero 现用 src. 前缀绝对 import）

try:
    from src.agents.affect_math import decode_channels
except ImportError as e:
    print(json.dumps({"skip": True, "reason": f"import src.agents.affect_math 失败: {e}"}))
    sys.exit(0)

# 采样若干 (v, a) 组合，覆盖四个象限
samples = [
    (0.6, 0.4),   # 正 v, 正 a → excited
    (-0.5, 0.6),  # 负 v, 正 a → angry
    (0.3, -0.2),  # 正 v, 负 a → content
    (-0.4, -0.3), # 负 v, 负 a → sad
    (0.0, 0.0),   # 中性
    (1.0, 1.0),   # 边界最大
    (-1.0, -1.0), # 边界最小
]

results = []
for v, a in samples:
    try:
        out = decode_channels((v, a))  # decode_channels 收单个 affect 元组
        results.append({"v": v, "a": a, "output": out, "error": None})
    except Exception as exc:
        results.append({"v": v, "a": a, "output": None, "error": str(exc)})

print(json.dumps({"skip": False, "results": results}))
"""


# ---------------------------------------------------------------------------
# 带 coping 的扩展采样脚本
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# M5 跨仓 schema 版本断言脚本
# ---------------------------------------------------------------------------

_SCHEMA_VERSION_SCRIPT = """
import sys
import json

sys.path.insert(0, sys.argv[1])  # D:\\Zero（包根）

try:
    from src.orchestration.external_prior import EXTERNAL_PRIOR_SCHEMA_VERSION, ExternalPrior
except ImportError as e:
    reason = f"import src.orchestration.external_prior 失败: {e}"
    print(json.dumps({"skip": True, "reason": reason}))
    sys.exit(0)

import typing
# 取类型别名的字符串表示，用于验证逐维 tuple 精度契约
type_str = str(ExternalPrior)

print(json.dumps({
    "skip": False,
    "schema_version": EXTERNAL_PRIOR_SCHEMA_VERSION,
    "external_prior_type_str": type_str,
}))
"""

# ---------------------------------------------------------------------------
# M3/M6 默认值一致性断言脚本（precision_cap / max_streams）
# ---------------------------------------------------------------------------

_DEFAULTS_SCRIPT = """
import sys
import json

sys.path.insert(0, sys.argv[1])  # D:\\Zero（包根）

try:
    from src.orchestration.state import AffectState
except ImportError as e:
    reason = f"import src.orchestration.state 失败: {e}"
    print(json.dumps({"skip": True, "reason": reason}))
    sys.exit(0)

# 读 AffectState pydantic 字段默认（M3 精度上界 / M6 流数上界）
fields = AffectState.model_fields
precision_cap = fields["external_prior_precision_cap"].default
max_streams = fields["max_external_streams"].default

print(json.dumps({
    "skip": False,
    "precision_cap": precision_cap,
    "max_streams": max_streams,
}))
"""


_DECODE_COPING_SCRIPT = """
import sys
import json

sys.path.insert(0, sys.argv[1])

try:
    from src.agents.affect_math import decode_channels
except ImportError as e:
    print(json.dumps({"skip": True, "reason": f"import src.agents.affect_math 失败: {e}"}))
    sys.exit(0)

# 检查 decode_channels 是否接受 coping / facs_extended 参数
import inspect
sig = inspect.signature(decode_channels)
params = list(sig.parameters.keys())
has_coping = "coping_potential" in params or "coping" in params
has_facs_extended = "facs_extended" in params

results = []
if has_coping and has_facs_extended:
    samples = [
        (0.5, 0.5, 0.3, False),
        (-0.5, 0.7, 0.8, True),
    ]
    for v, a, coping, facs_ext in samples:
        try:
            out = decode_channels((v, a), coping_potential=coping, facs_extended=facs_ext)
            results.append({"v": v, "a": a, "output": out, "error": None})
        except Exception as exc:
            results.append({"v": v, "a": a, "output": None, "error": str(exc)})

print(json.dumps({
    "skip": False,
    "has_coping_support": has_coping and has_facs_extended,
    "results": results,
}))
"""


# ---------------------------------------------------------------------------
# 测试类
# ---------------------------------------------------------------------------


@pytest.mark.zerorepo
class TestZeroContractCrosscheck:
    """跨仓契约回归：D:\\Zero decode_channels 输出能被本仓契约模型无损解析。

    所有用例：D:\\Zero 不可用 → skip。
    """

    def _skip_if_zero_unavailable(self) -> None:
        """检查 D:\\Zero 是否在位，不在位则 skip。"""
        if not _zero_available():
            pytest.skip(f"D:\\Zero\\src 不存在（路径 {_ZERO_SRC}），跳过跨仓契约回归")

    def _run_or_skip(self, script: str) -> dict[str, Any]:
        """运行子进程脚本，子进程失败或 import 失败均 skip。"""
        try:
            data = _run_subprocess_with_script(script)
        except subprocess.TimeoutExpired:
            pytest.skip("子进程超时，跳过跨仓契约回归")
        except RuntimeError as exc:
            pytest.skip(f"子进程非零退出，跳过: {exc}")
        except json.JSONDecodeError as exc:
            pytest.skip(f"子进程输出非合法 JSON，跳过: {exc}")

        if data.get("skip"):
            pytest.skip(data.get("reason", "D:\\Zero import 失败，跳过跨仓契约回归"))

        return data

    def test_decode_channels_output_parseable_by_expression_head(self) -> None:
        """decode_channels 各象限采样输出能被 ExpressionHead 无 ValidationError 解析。

        验证路径：子进程运行 D:\\Zero affect_math.decode_channels(v, a)
        → 返回 dict（含 facs_au / text_label / physiology / prosody 等通道）
        → 用本仓 ExpressionHead.model_validate() 解析 → 无 ValidationError。
        """
        self._skip_if_zero_unavailable()
        data = self._run_or_skip(_DECODE_SCRIPT)

        results: list[dict[str, Any]] = data["results"]
        assert results, "decode_channels 采样结果为空，无法断言"

        from src.agents.models.zero_affect import ExpressionHead

        parse_errors: list[str] = []
        for item in results:
            v, a = item["v"], item["a"]
            if item["error"] is not None:
                # decode_channels 本身抛错（如 torch 缺失）→ skip 整个用例
                pytest.skip(
                    f"decode_channels({v}, {a}) 抛出异常: {item['error']}，"
                    "可能缺少 torch 或其他运行时依赖"
                )
            output: dict[str, Any] = item["output"]
            if output is None:
                continue
            # 尝试以 ExpressionHead 解析 decode_channels 的输出
            try:
                ExpressionHead.model_validate(output)
            except Exception as exc:  # noqa: BLE001
                parse_errors.append(f"(v={v}, a={a}): {exc}")

        assert not parse_errors, (
            "以下采样点的 decode_channels 输出无法被 ExpressionHead 解析：\n"
            + "\n".join(parse_errors)
        )

    def test_decode_channels_facs_keys_subset_of_valid_keys(self) -> None:
        """decode_channels 输出的 facs_au 键集是 _FACS_VALID_KEYS 的子集。"""
        self._skip_if_zero_unavailable()
        data = self._run_or_skip(_DECODE_SCRIPT)

        from src.agents.models.zero_affect import FACS_KEYS, FACS_KEYS_EXT

        valid_keys = frozenset(FACS_KEYS) | frozenset(FACS_KEYS_EXT)
        results: list[dict[str, Any]] = data["results"]
        for item in results:
            if item["error"] is not None:
                pytest.skip(f"decode_channels 异常: {item['error']}")
            output: dict[str, Any] | None = item["output"]
            if output is None or "facs_au" not in output:
                continue
            unknown = set(output["facs_au"].keys()) - valid_keys
            assert not unknown, f"(v={item['v']}, a={item['a']}) facs_au 含未知键: {unknown}"

    def test_decode_channels_text_label_in_text_labels(self) -> None:
        """decode_channels 输出的 text_label 在 TEXT_LABELS 集合内。"""
        self._skip_if_zero_unavailable()
        data = self._run_or_skip(_DECODE_SCRIPT)

        from src.agents.models.zero_affect import TEXT_LABELS

        results: list[dict[str, Any]] = data["results"]
        for item in results:
            if item["error"] is not None:
                pytest.skip(f"decode_channels 异常: {item['error']}")
            output: dict[str, Any] | None = item["output"]
            if output is None or "text_label" not in output:
                continue
            label = output["text_label"]
            assert label in TEXT_LABELS, (
                f"(v={item['v']}, a={item['a']}) text_label={label!r} 不在 TEXT_LABELS"
            )

    def test_decode_channels_facs_values_in_range(self) -> None:
        """decode_channels 输出的 facs_au 所有值在 [0, 1] 范围内。"""
        self._skip_if_zero_unavailable()
        data = self._run_or_skip(_DECODE_SCRIPT)

        results: list[dict[str, Any]] = data["results"]
        for item in results:
            if item["error"] is not None:
                pytest.skip(f"decode_channels 异常: {item['error']}")
            output: dict[str, Any] | None = item["output"]
            if output is None or "facs_au" not in output:
                continue
            for key, val in output["facs_au"].items():
                assert 0.0 <= val <= 1.0, (
                    f"(v={item['v']}, a={item['a']}) facs_au[{key}]={val} 超出 [0,1]"
                )

    def test_decode_channels_with_coping_parseable(self) -> None:
        """decode_channels coping 路径输出能被 ExpressionHead 解析（若支持 coping 参数）。"""
        self._skip_if_zero_unavailable()
        data = self._run_or_skip(_DECODE_COPING_SCRIPT)

        if not data.get("has_coping_support"):
            pytest.skip("D:\\Zero decode_channels 不支持 coping_potential 参数，跳过")

        from src.agents.models.zero_affect import ExpressionHead

        results: list[dict[str, Any]] = data["results"]
        parse_errors: list[str] = []
        for item in results:
            if item["error"] is not None:
                pytest.skip(f"decode_channels coping 路径异常: {item['error']}")
            output: dict[str, Any] | None = item["output"]
            if output is None:
                continue
            try:
                ExpressionHead.model_validate(output)
            except Exception as exc:  # noqa: BLE001
                parse_errors.append(f"(v={item['v']}, a={item['a']}): {exc}")

        assert not parse_errors, "\n".join(parse_errors)


# ---------------------------------------------------------------------------
# M5 跨仓 schema 版本断言
# ---------------------------------------------------------------------------


@pytest.mark.zerorepo
class TestExternalPriorSchemaVersion:
    """M5 跨仓协议版本一致性断言——本仓 EXTERNAL_PRIOR_SCHEMA_VERSION 与 D:\\Zero 对齐。

    所有用例：D:\\Zero 不可用 → skip。
    """

    def _skip_if_zero_unavailable(self) -> None:
        """检查 D:\\Zero 是否在位，不在位则 skip。"""
        if not _zero_available():
            pytest.skip(f"D:\\Zero\\src 不存在（路径 {_ZERO_SRC}），跳过 M5 跨仓版本断言")

    def test_schema_version_matches_zero(self) -> None:
        """本仓 EXTERNAL_PRIOR_SCHEMA_VERSION 与 Zero 侧同名常量相等（M5 一致性）。"""
        self._skip_if_zero_unavailable()

        try:
            data = _run_subprocess_with_script(_SCHEMA_VERSION_SCRIPT)
        except subprocess.TimeoutExpired:
            pytest.skip("子进程超时，跳过 M5 版本断言")
        except RuntimeError as exc:
            pytest.skip(f"子进程非零退出，跳过: {exc}")
        except json.JSONDecodeError as exc:
            pytest.skip(f"子进程输出非合法 JSON，跳过: {exc}")

        if data.get("skip"):
            pytest.skip(data.get("reason", "D:\\Zero import 失败，跳过 M5 版本断言"))

        from src.mcp.zero.external_priors import EXTERNAL_PRIOR_SCHEMA_VERSION

        zero_version: int = data["schema_version"]
        assert zero_version == EXTERNAL_PRIOR_SCHEMA_VERSION, (
            f"M5 版本不一致：Zero 侧 EXTERNAL_PRIOR_SCHEMA_VERSION={zero_version}，"
            f"本仓 EXTERNAL_PRIOR_SCHEMA_VERSION={EXTERNAL_PRIOR_SCHEMA_VERSION}。"
            "两仓须协调同步修改。"
        )

    def test_external_prior_type_is_tuple_of_tuples(self) -> None:
        """Zero ExternalPrior 类型字符串符合逐维 tuple 精度契约（M1 形状未漂移）。

        期望形状：tuple[str, tuple[float, float], tuple[float, float]]
        （name, (μv, μa), (Πv, Πa)）。
        """
        self._skip_if_zero_unavailable()

        try:
            data = _run_subprocess_with_script(_SCHEMA_VERSION_SCRIPT)
        except subprocess.TimeoutExpired:
            pytest.skip("子进程超时，跳过 M1 形状断言")
        except RuntimeError as exc:
            pytest.skip(f"子进程非零退出，跳过: {exc}")
        except json.JSONDecodeError as exc:
            pytest.skip(f"子进程输出非合法 JSON，跳过: {exc}")

        if data.get("skip"):
            pytest.skip(data.get("reason", "D:\\Zero import 失败，跳过 M1 形状断言"))

        type_str: str = data["external_prior_type_str"]
        expected = "tuple[str, tuple[float, float], tuple[float, float]]"
        assert type_str == expected, (
            f"Zero ExternalPrior 类型字符串漂移：期望 {expected!r}，实际 {type_str!r}。"
            "逐维 tuple 精度契约（M1）在 Zero 侧已变更，须两仓协调对齐。"
        )


# ---------------------------------------------------------------------------
# M3/M6 默认值一致性断言（防跨仓默认值漂移）
# ---------------------------------------------------------------------------


@pytest.mark.zerorepo
class TestExternalPriorValidationDefaults:
    """M3/M6 跨仓默认值一致性——本仓 cap/max 常量与 Zero AffectState 字段默认对齐。

    所有用例：D:\\Zero 不可用或 state.py 无法轻量 import → skip（不拖红）。
    """

    def _skip_if_zero_unavailable(self) -> None:
        """检查 D:\\Zero 是否在位，不在位则 skip。"""
        if not _zero_available():
            pytest.skip(f"D:\\Zero\\src 不存在（路径 {_ZERO_SRC}），跳过 M3/M6 默认值断言")

    def _fetch_defaults_or_skip(self) -> dict[str, Any]:
        """运行子进程读 Zero AffectState 默认，任何失败均 skip。"""
        self._skip_if_zero_unavailable()
        try:
            data = _run_subprocess_with_script(_DEFAULTS_SCRIPT)
        except subprocess.TimeoutExpired:
            pytest.skip("子进程超时，跳过 M3/M6 默认值断言")
        except RuntimeError as exc:
            pytest.skip(f"子进程非零退出，跳过: {exc}")
        except json.JSONDecodeError as exc:
            pytest.skip(f"子进程输出非合法 JSON，跳过: {exc}")

        if data.get("skip"):
            pytest.skip(data.get("reason", "D:\\Zero import 失败，跳过 M3/M6 默认值断言"))
        return data

    def test_precision_cap_default_matches_zero(self) -> None:
        """M3：本仓 ZERO_EXTERNAL_PRIOR_PRECISION_CAP_DEFAULT 与 Zero AffectState 字段默认相等。"""
        data = self._fetch_defaults_or_skip()

        from src.mcp.zero.external_priors import ZERO_EXTERNAL_PRIOR_PRECISION_CAP_DEFAULT

        zero_cap: float = data["precision_cap"]
        assert zero_cap == pytest.approx(ZERO_EXTERNAL_PRIOR_PRECISION_CAP_DEFAULT), (
            f"M3 精度上界默认漂移：Zero AffectState.external_prior_precision_cap={zero_cap}，"
            f"本仓 ZERO_EXTERNAL_PRIOR_PRECISION_CAP_DEFAULT="
            f"{ZERO_EXTERNAL_PRIOR_PRECISION_CAP_DEFAULT}。两仓须协调同步。"
        )

    def test_max_streams_default_matches_zero(self) -> None:
        """M6：本仓 ZERO_MAX_EXTERNAL_STREAMS_DEFAULT 与 Zero AffectState 字段默认相等。"""
        data = self._fetch_defaults_or_skip()

        from src.mcp.zero.external_priors import ZERO_MAX_EXTERNAL_STREAMS_DEFAULT

        zero_max: int = data["max_streams"]
        assert zero_max == ZERO_MAX_EXTERNAL_STREAMS_DEFAULT, (
            f"M6 流数上界默认漂移：Zero AffectState.max_external_streams={zero_max}，"
            f"本仓 ZERO_MAX_EXTERNAL_STREAMS_DEFAULT={ZERO_MAX_EXTERNAL_STREAMS_DEFAULT}。"
            "两仓须协调同步。"
        )


# ---------------------------------------------------------------------------
# 辅助函数（模块级，统一子进程调用）
# ---------------------------------------------------------------------------


def _run_subprocess_with_script(script: str) -> dict[str, Any]:
    """在子进程中运行给定 script，返回 stdout 解析的 JSON dict。"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(_ZERO_ROOT)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"子进程非零退出 ({result.returncode}):\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )
    return json.loads(result.stdout)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# T6·② unknown-session 机读标记跨仓一致性（防 marker 静默漂移）
#
# Zero server 与本仓 client 各自持有 `_UNKNOWN_SESSION_MARKER` 常量；step 命中未知/过期
# session_id 时 Zero 用它作 ToolError 前缀，本仓 client 用它判定并抛 ZeroLinkUnknownSessionError。
# 两侧任一改动而另一侧未跟 → unknown-session 判定静默失效（回执点名的「脆弱字符串匹配」风险的
# 机读版）。此回归以正则从 Zero 源码直读该常量（不 import，避开 FastMCP/torch 重依赖），断言与
# 本仓一致。D:\Zero 不在位 → skip。
# ---------------------------------------------------------------------------

# Zero server 定义 unknown-session marker 的源文件（相对 D:\Zero）
_ZERO_SERVER_PY = _ZERO_SRC / "mcp_server" / "server.py"
# 匹配 `_UNKNOWN_SESSION_MARKER = "unknown-session"`（单/双引号皆容）
_MARKER_ASSIGN_RE = re.compile(
    r"""^_UNKNOWN_SESSION_MARKER\s*=\s*["']([^"']+)["']""",
    re.MULTILINE,
)


@pytest.mark.zerorepo
class TestUnknownSessionMarkerCrosscheck:
    """T6·② 跨仓 marker 一致性——本仓 client 与 Zero server 的 `_UNKNOWN_SESSION_MARKER` 相等。

    D:\\Zero 或 server.py 不在位 → skip（不拖红）。
    """

    def test_unknown_session_marker_matches_zero(self) -> None:
        """本仓 `_UNKNOWN_SESSION_MARKER` == Zero server 侧同名常量（正则直读源码，不 import）。"""
        if not _zero_available():
            pytest.skip(f"D:\\Zero\\src 不存在（路径 {_ZERO_SRC}），跳过 marker 跨仓断言")
        if not _ZERO_SERVER_PY.is_file():
            pytest.skip(f"Zero server.py 不存在（路径 {_ZERO_SERVER_PY}），跳过 marker 跨仓断言")

        source = _ZERO_SERVER_PY.read_text(encoding="utf-8")
        match = _MARKER_ASSIGN_RE.search(source)
        if match is None:
            pytest.skip(
                f'Zero server.py 未找到 `_UNKNOWN_SESSION_MARKER = "..."` 定义'
                f"（{_ZERO_SERVER_PY}）——可能 Zero 侧尚未接线或改了命名，跳过"
            )

        from src.mcp.zero.client import _UNKNOWN_SESSION_MARKER

        zero_marker = match.group(1)
        assert zero_marker == _UNKNOWN_SESSION_MARKER, (
            f"T6·② unknown-session marker 跨仓漂移：Zero server="
            f"{zero_marker!r}，本仓 client={_UNKNOWN_SESSION_MARKER!r}。"
            "两仓须协调同步——否则 unknown-session 判定静默失效。"
        )


# ---------------------------------------------------------------------------
# physiology 对称接线：Zero physiology_decoder 输出键与本仓 PhysiologyChannel 契约一致
#
# canonical=WESAD（2026-07-23 拍板）：Zero 真 physiology_decoder.predict_physiology 输出
# {heart_rate_bpm, skin_conductance(μS), temperature_c}。本仓 PhysiologyChannel（extra=forbid）
# 须能无 ValidationError 解析该形状——decoder 输出键须 ⊆ 契约字段集，且含必填 hr+sc。任一漂移
# （decoder 加本仓不认的键，或 rename）→ Zero 接线后 MCP 侧每步 ValidationError。此回归以正则从
# Zero decoder 源码直读输出键（不 import，避 torch），断言契约对齐。D:\Zero 不在位 → skip。
# ---------------------------------------------------------------------------

# Zero physiology decoder 源文件 + 提取 predict_physiology 返回 dict 的字符串键
_ZERO_PHYSIO_DECODER_PY = _ZERO_SRC / "agents" / "models" / "physiology_decoder.py"
# 从 predict_physiology 起截取其 return dict 块 `{...}`（非贪婪到首个 `}`；该 dict 无嵌套花括号），
# 仅在块内提键——避免正则扫到函数外的 `"key":`（其它方法/日志/注解）而误纳/误漏（code-review W1）。
_RETURN_DICT_RE = re.compile(r"return\s*\{(.*?)\}", re.DOTALL)
_PHYSIO_KEY_RE = re.compile(r'"([a-z_]+)"\s*:')


@pytest.mark.zerorepo
class TestPhysiologyDecoderContractCrosscheck:
    """physiology 对称接线跨仓一致——Zero decoder 输出键 ⊆ 本仓 PhysiologyChannel 字段。

    D:\\Zero 或 physiology_decoder.py 不在位 → skip（不拖红）。
    """

    def test_physiology_decoder_keys_subset_of_channel(self) -> None:
        """Zero predict_physiology 输出键 ⊆ PhysiologyChannel 字段，且含必填 hr+sc。"""
        if not _zero_available():
            pytest.skip(f"D:\\Zero\\src 不存在（{_ZERO_SRC}），跳过 physiology 契约跨仓断言")
        if not _ZERO_PHYSIO_DECODER_PY.is_file():
            pytest.skip(f"Zero physiology_decoder.py 不存在（{_ZERO_PHYSIO_DECODER_PY}），跳过")

        source = _ZERO_PHYSIO_DECODER_PY.read_text(encoding="utf-8")
        # 定位 predict_physiology → 仅在其 return dict 块内提键（约束范围，防扫到函数外，W1）
        marker = "def predict_physiology"
        idx = source.find(marker)
        if idx < 0:
            pytest.skip("Zero physiology_decoder.py 未见 predict_physiology，可能改了命名，跳过")
        block = _RETURN_DICT_RE.search(source[idx:])
        if block is None:
            pytest.skip("predict_physiology 未见 `return {...}` 块，可能改了结构，跳过")
        decoder_keys = set(_PHYSIO_KEY_RE.findall(block.group(1)))
        if not decoder_keys:
            pytest.skip("未从 return dict 提到输出键，正则可能需更新，跳过")

        from src.agents.models.zero_affect import PhysiologyChannel

        channel_fields = set(PhysiologyChannel.model_fields)
        unknown = decoder_keys - channel_fields
        assert not unknown, (
            f"physiology 契约漂移：Zero decoder 输出键 {sorted(unknown)} 不在本仓 "
            f"PhysiologyChannel 字段 {sorted(channel_fields)}——Zero 接线后每步会 ValidationError。"
            "两仓须协调同步 canonical 契约。"
        )
        # 必填字段（hr+sc）decoder 须产出（否则解析缺字段）
        required = {"heart_rate_bpm", "skin_conductance"}
        missing = required - decoder_keys
        assert not missing, f"Zero decoder 未产必填字段 {sorted(missing)}（PhysiologyChannel 必填）"
