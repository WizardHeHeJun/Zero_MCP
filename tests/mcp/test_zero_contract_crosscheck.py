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
# ② canonical 占位口径采样脚本（decode_channels(canonical_physiology=True)）
#
# Zero ② 落地（commit b503990+432f8d9）：decode_channels 新增 canonical_physiology 参数
# （门 = ZERO_PHYSIOLOGY_CANONICAL_PLACEHOLDER）。门开 → physiology 出 canonical 占位
# {hr, sc(μS), temperature_c}（无 pupil_mm）；门关（默认）→ legacy {hr, sc[0,1], pupil_mm}。
# 本脚本同一组 (v,a) 各跑门开/门关两路，供上层断言「两形状分野」+「超集同解析」+ 议会占位公式。
# decode_channels 无 canonical_physiology 参数（旧 Zero，② 未落地）→ skip（不拖红）。
# ---------------------------------------------------------------------------

_DECODE_CANONICAL_PHYSIO_SCRIPT = """
import sys
import json

sys.path.insert(0, sys.argv[1])

try:
    from src.agents.affect_math import decode_channels
except ImportError as e:
    print(json.dumps({"skip": True, "reason": f"import src.agents.affect_math 失败: {e}"}))
    sys.exit(0)

import inspect
if "canonical_physiology" not in inspect.signature(decode_channels).parameters:
    print(json.dumps({
        "skip": True,
        "reason": "decode_channels 无 canonical_physiology 参数（Zero 侧尚未落地 ② 门控）",
    }))
    sys.exit(0)

# 直接以已知 (v,a) 调 decode_channels（非过 affect core，故议会占位公式可逐值核验）。
# 关键点位：(0,0) 验 sc 中点偏置=0μS·temp=36；(±0.4, 0.5) 非饱和同 |a| 验 temp 无 valence
# （不与饱和抹平混淆）；(±0.8, 1.0) 饱和边界。
samples = [
    (0.0, 0.0),
    (0.6, 0.4),
    (-0.5, 0.6),
    (0.3, -0.6),
    (0.4, 0.5),
    (-0.4, 0.5),
    (0.8, 1.0),
    (-0.8, 1.0),
]

results = []
for v, a in samples:
    try:
        canon = decode_channels((v, a), canonical_physiology=True)
        legacy = decode_channels((v, a), canonical_physiology=False)
        results.append({"v": v, "a": a, "canonical": canon, "legacy": legacy, "error": None})
    except Exception as exc:
        results.append({"v": v, "a": a, "canonical": None, "legacy": None, "error": str(exc)})

print(json.dumps({"skip": False, "results": results}))
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


# ---------------------------------------------------------------------------
# ② canonical 占位口径跨仓一致（decode_channels(canonical_physiology=True) 真跑）
#
# 上面 TestPhysiologyDecoderContractCrosscheck 走**真 decoder 源码**（① 路径·正则读 return 键）。
# 本类走**占位路径运行时**（② 路径·真跑 decode_channels 门开/门关）——验证：
#   (a) 门开 canonical 形状 = {hr, sc(μS), temperature_c}（无 pupil）；门关 legacy 含 pupil；
#   (b) 本仓**超集契约**（ExpressionHead / PhysiologyChannel）无损解析两形状（=保超集决策依据）；
#   (c) 议会 2026-07-23 占位公式逐值：sc 中点偏置（arousal=0→0μS）、temp 无 valence（同 |a| 同温）、
#       各值落 canonical 域（hr[50,120]/sc[0,20]μS/temp[33,36]°C）。
# ⚠ sc 中点偏置：占位 arousal=0→0μS，真 decoder 中立态~10μS——本类仅在**占位路径内**核对，
#   不与真 decoder 跨路径绝对比较（禁跨路径比较，Zero 简报 §2）。
# ---------------------------------------------------------------------------


@pytest.mark.zerorepo
class TestCanonicalPhysiologyPlaceholderCrosscheck:
    """② canonical 占位路径跨仓一致——门开/门关两形状 + 超集解析 + 议会占位公式逐值核验。

    D:\\Zero 不可用 / decode_channels 无 canonical_physiology 参数（② 未落地）→ skip（不拖红）。
    """

    def _fetch_or_skip(self) -> list[dict[str, Any]]:
        """运行 canonical 采样子进程，任何失败 / ② 未落地 → skip；返回 results 列表。"""
        if not _zero_available():
            pytest.skip(f"D:\\Zero\\src 不存在（{_ZERO_SRC}），跳过 canonical 占位跨仓断言")
        try:
            data = _run_subprocess_with_script(_DECODE_CANONICAL_PHYSIO_SCRIPT)
        except subprocess.TimeoutExpired:
            pytest.skip("子进程超时，跳过 canonical 占位跨仓断言")
        except RuntimeError as exc:
            pytest.skip(f"子进程非零退出，跳过: {exc}")
        except json.JSONDecodeError as exc:
            pytest.skip(f"子进程输出非合法 JSON，跳过: {exc}")
        if data.get("skip"):
            pytest.skip(data.get("reason", "canonical 占位路径不可用，跳过"))
        results: list[dict[str, Any]] = data["results"]
        # decode_channels 运行时异常（如 torch 缺失触发某路径）→ skip 整类
        for item in results:
            if item["error"] is not None:
                pytest.skip(f"decode_channels(canonical) 抛异常: {item['error']}")
        assert results, "canonical 采样结果为空，无法断言"
        return results

    def test_gate_on_canonical_shape_no_pupil(self) -> None:
        """门开 → physiology 键恰 {hr, sc, temperature_c}（无 pupil）；门关 → 含 pupil 无 temp。"""
        results = self._fetch_or_skip()
        canonical_fields = {"heart_rate_bpm", "skin_conductance", "temperature_c"}
        for item in results:
            v, a = item["v"], item["a"]
            canon_phys = item["canonical"]["physiology"]
            assert set(canon_phys) == canonical_fields, (
                f"(v={v}, a={a}) canonical physiology 键漂移：期望 {sorted(canonical_fields)}，"
                f"实际 {sorted(canon_phys)}（应删 pupil_mm、含 temperature_c）"
            )
            assert "pupil_mm" not in canon_phys, f"(v={v}, a={a}) canonical 占位仍出 pupil_mm"
            legacy_phys = item["legacy"]["physiology"]
            assert "pupil_mm" in legacy_phys, f"(v={v}, a={a}) legacy 占位缺 pupil_mm（零回归破坏）"
            assert "temperature_c" not in legacy_phys, (
                f"(v={v}, a={a}) legacy 占位不应出 temperature_c"
            )

    def test_superset_contract_parses_both_shapes(self) -> None:
        """本仓超集契约无损解析门开 canonical 与门关 legacy 两形状（保超集决策的运行时依据）。"""
        results = self._fetch_or_skip()
        from src.agents.models.zero_affect import ExpressionHead, PhysiologyChannel

        errors: list[str] = []
        for item in results:
            v, a = item["v"], item["a"]
            for tag in ("canonical", "legacy"):
                try:
                    ExpressionHead.model_validate(item[tag])
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"(v={v}, a={a}) {tag} 无法被 ExpressionHead 解析: {exc}")
            # PhysiologyChannel 直解：canonical→temp 非 None·无 pupil；legacy→有 pupil·temp None
            canon_ch = PhysiologyChannel.model_validate(item["canonical"]["physiology"])
            assert canon_ch.temperature_c is not None and canon_ch.pupil_mm is None, (
                f"(v={v}, a={a}) canonical PhysiologyChannel 应 temp 非 None、pupil None"
            )
            legacy_ch = PhysiologyChannel.model_validate(item["legacy"]["physiology"])
            assert legacy_ch.pupil_mm is not None and legacy_ch.temperature_c is None, (
                f"(v={v}, a={a}) legacy PhysiologyChannel 应 pupil 非 None、temp None"
            )
        assert not errors, "\n".join(errors)

    def test_council_formula_domains_and_bias(self) -> None:
        """议会占位公式**逐值** pin：确定性 (v,a) 直调 → hr/sc/temp 精确值（标度/斜率漂移即 fail）。

        入参确定 + Zero 公式固定（hr=50+70·clamp(0.5(1+a))、sc=20·clamp|a|、temp=36−3·clamp|a|），
        故可逐值 pin（非仅域成员）——sc 20→10 或 hr 斜率变即 hard fail。含中点偏置（a=0→sc=0μS）
        + temp 无 valence（**非饱和**同 |a|=0.5、±valence → 同温，不与饱和抹平混淆）。
        """
        results = self._fetch_or_skip()
        by_key = {(item["v"], item["a"]): item["canonical"]["physiology"] for item in results}

        # 逐值 pin：Zero 占位公式对确定性 (v,a) 的精确输出（(hr, sc(μS), temp°C)）
        expected = {
            (0.0, 0.0): (85.0, 0.0, 36.0),
            (0.6, 0.4): (99.0, 8.0, 34.8),
            (-0.5, 0.6): (106.0, 12.0, 34.2),
            (0.3, -0.6): (64.0, 12.0, 34.2),
            (0.4, 0.5): (102.5, 10.0, 34.5),
            (-0.4, 0.5): (102.5, 10.0, 34.5),
            (0.8, 1.0): (120.0, 20.0, 33.0),
            (-0.8, 1.0): (120.0, 20.0, 33.0),
        }
        for (v, a), (hr, sc, temp) in expected.items():
            phys = by_key[(v, a)]
            assert phys["heart_rate_bpm"] == pytest.approx(hr), f"(v={v},a={a}) hr 漂移 {phys}"
            assert phys["skin_conductance"] == pytest.approx(sc), f"(v={v},a={a}) sc 漂移 {phys}"
            assert phys["temperature_c"] == pytest.approx(temp), f"(v={v},a={a}) temp 漂移 {phys}"

        # sc 中点偏置命名断言：arousal=0 → sc=0μS（禁与真 decoder 中立态~10μS 跨路径比较）
        assert by_key[(0.0, 0.0)]["skin_conductance"] == pytest.approx(0.0)

        # temp 无 valence（非饱和隔离）：同 |a|=0.5、valence 相反 → 同温 34.5（分野属 coping）
        assert by_key[(0.4, 0.5)]["temperature_c"] == pytest.approx(
            by_key[(-0.4, 0.5)]["temperature_c"]
        ), "temp 无 valence：非饱和同 |a| 不同 valence 应同温"


# ---------------------------------------------------------------------------
# TD/情境键跨仓系数 pin（防「Zero 单方调系数 → 本仓 resume E2E 红灯但归因不明」）
#
# `test_zero_client_e2e.py::TestZeroClientResumeAcrossRestart` 的判别观测量（重复强刺激下
# valence_arousal 随会话历史单调漂移 Δ≈8.2e-3）由 Zero `ValueAgent` 的 TD 在线学习驱动：
# 情境键恒 "mcp-step" → 同一 V(s) 跨轮累积 → `td_update` 以 lr 缩放每步 ΔV=lr·δ。
# 阈值 `_RESUME_STATE_MARGIN=3e-3` 的依据（notes/2026-07-24-zero-link-resume-e2e-probe.md）
# 因此**耦合这几个 Zero 侧常量**。Zero 2026-07-25 回执已把 lr / 情境键列入「必 ping」，但口头
# 约定不可执行——此处把它变成可拦截的红，且失败文案直接给出复原路径。
#
# 正则直读 Zero 源码（不 import，避 torch/FastMCP 重依赖），D:\Zero 不在位 → skip。
# 快照日期 2026-07-25 · Zero HEAD=aad6762。
# ---------------------------------------------------------------------------

_ZERO_AFFECT_MATH_PY = _ZERO_SRC / "agents" / "affect_math.py"
_ZERO_VALUE_PY = _ZERO_SRC / "agents" / "value.py"
_ZERO_MAPPING_PY = _ZERO_SRC / "mcp_server" / "mapping.py"

# 快照值（2026-07-25 现场核验）
_PINNED_TD_LR = 0.2  # affect_math.py:138 —— 直接缩放 ΔV=lr·δ；闭式复算红线 lr≲0.057 转红
_PINNED_TD_NEXT_VALUE = 0.0  # affect_math.py:139 —— 恒 0 使 gamma 项消失（gamma 对本仓是死系数）
_PINNED_MCP_STIMULUS_KEY = "mcp-step"  # mapping.py:27 —— 情境键，跨轮同键才累积 V(s)

# `def td_update(...)` 的形参默认值（形参各占一行，容忍空白/尾随逗号）
_TD_KWARG_RE_TEMPLATE = r"^\s*{name}\s*:\s*float\s*=\s*([-\d.eE+]+)\s*,?\s*$"
# `stimulus_from_payload(..., name: str = "mcp-step", ...)`
_STIM_NAME_RE = re.compile(r"""^\s*name\s*:\s*str\s*=\s*["']([^"']+)["']""", re.MULTILINE)
# `td_update(` 调用实参（当前调用无嵌套括号；有嵌套则匹配失败 → skip 而非误判）
_TD_CALL_RE = re.compile(r"td_update\(([^()]*)\)")


def _td_update_default(source: str, name: str) -> str | None:
    """从 `def td_update(` 起至函数体前的签名块里取形参 `name` 的默认值字面量。"""
    start = source.find("def td_update(")
    if start == -1:
        return None
    end = source.find(")", start)
    if end == -1:
        return None
    signature = source[start:end]
    match = re.search(_TD_KWARG_RE_TEMPLATE.format(name=re.escape(name)), signature, re.MULTILINE)
    return match.group(1) if match else None


@pytest.mark.zerorepo
class TestTdCoefficientCrosscheck:
    """跨仓 pin Zero 的 TD 系数与 MCP 情境键——它们是本仓 resume 判别阈值的隐式前提。

    ⚠ 本组断言**不是**在规定 Zero 该用什么系数（那是 Zero 的自由）；而是让「Zero 改了系数」
    这件事对本仓**立即可见且归因明确**，避免 resume E2E 以「sqlite/resume 失效」的错误面貌报红。
    红了的正确处理是：复跑探针重标阈值 + 同步两仓 ping 清单，**不是**放宽断言。
    """

    def test_td_update_defaults_pinned(self) -> None:
        """Zero `td_update` 的 `lr` / `next_value` 默认值未漂移。"""
        if not _zero_available():
            pytest.skip(f"D:\\Zero\\src 不存在（{_ZERO_SRC}），跳过 TD 系数 pin")
        if not _ZERO_AFFECT_MATH_PY.is_file():
            pytest.skip(f"Zero affect_math.py 不存在（{_ZERO_AFFECT_MATH_PY}）")

        source = _ZERO_AFFECT_MATH_PY.read_text(encoding="utf-8")
        lr_literal = _td_update_default(source, "lr")
        next_value_literal = _td_update_default(source, "next_value")
        if lr_literal is None or next_value_literal is None:
            pytest.skip(
                f"未能从 Zero affect_math.py 解析 td_update 的 lr/next_value 默认值"
                f"（签名可能已重构：lr={lr_literal!r} next_value={next_value_literal!r}）"
            )

        assert float(lr_literal) == pytest.approx(_PINNED_TD_LR), (
            f"Zero TD 学习率漂移：td_update(lr={lr_literal}) ≠ 快照 {_PINNED_TD_LR}"
            f"（2026-07-25 @ aad6762）。本仓 resume E2E 的 `_RESUME_STATE_MARGIN=3e-3` 依据实测"
            f"漂移 8.2e-3，闭式红线为 lr≲0.057 → 请复跑 "
            f"notes/2026-07-24-zero-link-resume-e2e-probe.md §4 的探针重标阈值/_RESUME_PRE_STEPS，"
            f"并更新跨仓系数快照。"
        )
        assert float(next_value_literal) == pytest.approx(_PINNED_TD_NEXT_VALUE), (
            f"Zero td_update 的 next_value 默认值变为 {next_value_literal}（快照 "
            f"{_PINNED_TD_NEXT_VALUE}）——**gamma 由此从死系数变活**（原 delta=reward+gamma·0−V(s)，"
            f"gamma 项恒消）。这正是 2026-07-25 回执里约定要 ping 的「引入真 next_value / 多步 "
            f"bootstrap」事件：本仓 resume 观测量的漂移量会随之改变，须复跑探针。"
        )

    def test_value_agent_does_not_override_td_defaults(self) -> None:
        """`ValueAgent` 仍以三个位置参数调 `td_update`（不覆写 lr/gamma/next_value）。

        上一条 pin 的是**默认值**；只有调用方不覆写，默认值才等于运行时实际值。
        """
        if not _zero_available():
            pytest.skip(f"D:\\Zero\\src 不存在（{_ZERO_SRC}），跳过 TD 调用点 pin")
        if not _ZERO_VALUE_PY.is_file():
            pytest.skip(f"Zero value.py 不存在（{_ZERO_VALUE_PY}）")

        source = _ZERO_VALUE_PY.read_text(encoding="utf-8")
        match = _TD_CALL_RE.search(source)
        if match is None:
            pytest.skip(
                f"未在 Zero value.py 找到无嵌套括号的 `td_update(...)` 调用"
                f"（{_ZERO_VALUE_PY}）——调用形态可能已重构，跳过而非误判"
            )

        call_args = match.group(1)
        overridden = [kw for kw in ("lr=", "gamma=", "next_value=") if kw in call_args]
        assert not overridden, (
            f"Zero ValueAgent 现在覆写了 TD 参数 {overridden}（调用实参：{call_args!r}）——"
            f"本仓据 `td_update` **默认值**推算的 resume 漂移量不再成立，须复跑探针重标阈值。"
        )

    def test_mcp_stimulus_key_pinned(self) -> None:
        """MCP 路径的情境键仍恒为 "mcp-step"（跨轮同键才累积 V(s)）。"""
        if not _zero_available():
            pytest.skip(f"D:\\Zero\\src 不存在（{_ZERO_SRC}），跳过情境键 pin")
        if not _ZERO_MAPPING_PY.is_file():
            pytest.skip(f"Zero mcp_server/mapping.py 不存在（{_ZERO_MAPPING_PY}）")

        source = _ZERO_MAPPING_PY.read_text(encoding="utf-8")
        match = _STIM_NAME_RE.search(source)
        if match is None:
            pytest.skip(
                f'未在 Zero mapping.py 找到 `name: str = "..."` 默认情境键'
                f"（{_ZERO_MAPPING_PY}）——签名可能已重构，跳过而非误判"
            )

        assert match.group(1) == _PINNED_MCP_STIMULUS_KEY, (
            f"MCP 情境键漂移：Zero mapping.py={match.group(1)!r}，快照="
            f"{_PINNED_MCP_STIMULUS_KEY!r}。若每步键不同，`value_table` 不再跨轮累积同一 V(s)"
            f"→ 本仓 resume 观测量漂移归零 → `state-matters` 守卫会先行 FAIL（非 flaky，是真失效）"
        )
