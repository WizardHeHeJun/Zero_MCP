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

from src.mcp.zero.external_priors import (
    ModalityKind,
    ModalityPrior,
    merge_physio_priors,
    recommended_precision,
)

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
# 在线真模型路径的**反归一化常量** pin（Zero 07-28 回执 ③(b) 补正）
#
# 此前我方只 pin 了**占位**路径（affect_math.py:479-481，见 TestCanonicalPlaceholder…）与真模型
# 路径的**键名**（上一个类）——漏了真模型路径的**值级常量**。Zero 指出：设 ZERO_PHYSIOLOGY_MODEL_PATH
# 后 composite.py:134-135 **整块覆盖** channels["physiology"]，占位式根本不执行，走的是
# physiology_decoder.py:35-37 的**另一套常量**（temp 对 30/10 ≠ 占位的 36/3）。
#
# 为何这对**消费正确性**要命（W6 同类：静默标度差）：本仓 LinearPhysiologyMapper 的默认量纲
# （skin_conductance_max_us=20.0、temperature_range=(30,40)）正是按**在线 decoder** 标定的。
# Zero 若把 `vec[1]*20.0` 改成 `*10.0`，我方解析照样成功、mapper 照样不报错，但 level 静默错 2×
# （与 W6 legacy sc 欠标度 20× 同族）。故此处不止 pin 常量，还断言**常量 ⇔ mapper 默认**的耦合。
#
# 另 wesad.py:59-62 的训练侧归一化与 decoder 反归一化是**逆变换对，必须成对同改**（Zero ③(b)）：
# 单改一侧 → 权重与解码口径错配，输出物理量整体偏移。故一并 pin，任一侧漂移即红 → 触发 ping。
# ---------------------------------------------------------------------------

_ZERO_WESAD_PY = _ZERO_SRC / "agents" / "datasets" / "wesad.py"
# 逐键取 return dict 的 RHS 表达式（到行尾/逗号），再从中提浮点字面量
_PHYSIO_RHS_RE = re.compile(r'"([a-z_]+)"\s*:\s*([^,\n]+)')
_FLOAT_LITERAL_RE = re.compile(r"\d+\.\d+")
# 训练侧归一化（逆变换对）——锚在变量名上，结构改了则 skip、常量漂移则 fail
_WESAD_HR_RE = re.compile(r"clamp\(\(hr\s*-\s*([\d.]+)\)\s*/\s*([\d.]+)")
_WESAD_EDA_RE = re.compile(r"clamp\(eda_mean\s*/\s*([\d.]+)")
_WESAD_TEMP_RE = re.compile(r"clamp\(\(temp_mean\s*-\s*([\d.]+)\)\s*/\s*([\d.]+)")

# 在线 decoder 反归一化常量（Zero 07-28 现场核验：physiology_decoder.py:35-37）
# key → (offset, span)；offset=None 表示纯标度（无偏置）
_ONLINE_PHYSIO_CONSTANTS: dict[str, tuple[float | None, float]] = {
    "heart_rate_bpm": (50.0, 70.0),  # 50 + vec*70 → [50,120] bpm
    "skin_conductance": (None, 20.0),  # vec*20 → [0,20] μS
    "temperature_c": (30.0, 10.0),  # 30 + vec*10 → [30,40] °C（≠占位 36/3）
}


@pytest.mark.zerorepo
class TestPhysiologyDecoderOnlineConstantsPin:
    """在线真模型路径的反归一化常量 pin + 与本仓 mapper 消费标度的耦合断言。

    D:\\Zero / physiology_decoder.py / wesad.py 不在位或结构变更 → skip（不拖红）；
    **常量漂移 → hard fail**（这正是要触发跨仓 ping 的信号）。
    """

    def _decoder_constants(self) -> dict[str, list[float]]:
        """从 predict_physiology 的 return dict 逐键提浮点字面量；结构变更 → skip。"""
        if not _zero_available():
            pytest.skip(f"D:\\Zero\\src 不存在（{_ZERO_SRC}），跳过在线常量 pin")
        if not _ZERO_PHYSIO_DECODER_PY.is_file():
            pytest.skip(f"Zero physiology_decoder.py 不存在（{_ZERO_PHYSIO_DECODER_PY}），跳过")
        source = _ZERO_PHYSIO_DECODER_PY.read_text(encoding="utf-8")
        idx = source.find("def predict_physiology")
        if idx < 0:
            pytest.skip("Zero physiology_decoder.py 未见 predict_physiology，跳过")
        block = _RETURN_DICT_RE.search(source[idx:])
        if block is None:
            pytest.skip("predict_physiology 未见 `return {...}` 块，跳过")
        found = {
            key: [float(x) for x in _FLOAT_LITERAL_RE.findall(rhs)]
            for key, rhs in _PHYSIO_RHS_RE.findall(block.group(1))
        }
        if not found:
            pytest.skip("未从 return dict 提到常量，正则可能需更新，跳过")
        return found

    def test_online_decoder_constants_pinned(self) -> None:
        """真模型反归一化常量逐值 pin（漂移即红 → 触发 Zero ping 约定）。"""
        found = self._decoder_constants()
        for key, (offset, span) in _ONLINE_PHYSIO_CONSTANTS.items():
            assert key in found, (
                f"在线 decoder 缺键 {key}（实际 {sorted(found)}）——canonical 契约漂移"
            )
            expected = [span] if offset is None else [offset, span]
            assert found[key] == expected, (
                f"在线 decoder 反归一化常量漂移：{key} 期望 {expected}、实际 {found[key]}。"
                "这会让本仓 mapper 的物理量标度静默错配（W6 同类静默标度差）——"
                "须与 Zero 协调（并同步 datasets/wesad.py 的逆变换对）。"
            )

    def test_online_constants_match_mapper_defaults(self) -> None:
        """在线 decoder 量纲 ⇔ 本仓 LinearPhysiologyMapper 默认值（消费标度正确性的真断言）。

        decoder sc 标度 == mapper skin_conductance_max_us；decoder temp [off, off+span]
        == mapper temperature_range。任一侧改动而另一侧未跟 → level 静默错标度。
        """
        found = self._decoder_constants()
        from src.mcp.zero.mappers.physiology import LinearPhysiologyMapper

        mapper = LinearPhysiologyMapper()  # 默认值即消费标定
        sc_span = found["skin_conductance"][0]
        assert sc_span == mapper.skin_conductance_max_us, (
            f"皮电标度失配：Zero decoder ×{sc_span}μS vs 本仓 mapper 默认上界 "
            f"{mapper.skin_conductance_max_us}μS —— skin_conductance_level 会静默错标度。"
        )
        temp_off, temp_span = found["temperature_c"][0], found["temperature_c"][1]
        assert (temp_off, temp_off + temp_span) == mapper.temperature_range, (
            f"体温量程失配：Zero decoder 域 [{temp_off}, {temp_off + temp_span}] vs 本仓 mapper "
            f"默认 temperature_range={mapper.temperature_range} —— skin_temperature_level 会错。"
        )

    def test_wesad_normalization_is_inverse_pair_of_decoder(self) -> None:
        """训练侧 wesad 归一化 ⇔ decoder 反归一化为逆变换对（Zero ③(b)：必须成对同改）。"""
        found = self._decoder_constants()
        if not _ZERO_WESAD_PY.is_file():
            pytest.skip(f"Zero wesad.py 不存在（{_ZERO_WESAD_PY}），跳过逆变换对断言")
        src_text = _ZERO_WESAD_PY.read_text(encoding="utf-8")
        hr_m = _WESAD_HR_RE.search(src_text)
        eda_m = _WESAD_EDA_RE.search(src_text)
        temp_m = _WESAD_TEMP_RE.search(src_text)
        if not (hr_m and eda_m and temp_m):
            pytest.skip("wesad.py 归一化结构变更（变量名/写法），正则未命中，跳过")

        pairs = {
            "heart_rate_bpm": [float(hr_m.group(1)), float(hr_m.group(2))],
            "skin_conductance": [float(eda_m.group(1))],
            "temperature_c": [float(temp_m.group(1)), float(temp_m.group(2))],
        }
        for key, train_consts in pairs.items():
            assert train_consts == found[key], (
                f"逆变换对断裂：{key} 训练侧 wesad 常量 {train_consts} ≠ decoder 反归一化 "
                f"{found[key]}——单改一侧会让权重与解码口径错配，输出物理量整体偏移。"
            )


# ---------------------------------------------------------------------------
# ignition 点燃门跨仓耦合：Zero 门控常量 ⊗ 本仓推荐精度的**可达性**
#
# Zero 07-28 回执 A 条提醒 SALIENCE_THRESHOLD 会门掉 value 流。本仓据此追查发现该门**同样**
# 作用于 external_priors：`affect_core.py:100-108` 在 expand 后**无条件**进 ignite()，而
# `expand_external_priors`（affect_math.py:974）只校验不改精度。默认 IGNITION_BETA=None 即硬门。
#
# salience(μ,Π) = hypot(μ)·mean(Π)（affect_math.py:673）——|μ| 的线性函数、门是**锐阶跃**。
# 跨阈需 |μ| ≥ threshold/mean(Π)；而 ModalityPrior 各维 ∈[-1,1] → |μ| ≤ √2≈1.4142。
# 本仓推荐精度的可达性（2026-07-28 真 server A/B 实测，与解析值吻合到小数点后四位）：
#   face   Π̄=0.1600 → 需 |μ|≥1.1250 ≤ √2  **可达**（实测 μ=(-0.95,0.95) 点燃 |Δ|=1.55e-02）
#   audio  Π̄=0.1750 → 需 |μ|≥1.0286 ≤ √2  **可达**（实测 μ=(0.9,0.9)   点燃 |Δ|=1.24e-02）
#   physio Π̄=0.0905 → 需 |μ|≥1.9890 > √2  **不可达**（实测极限 μ=(-1,1) 仍 |Δ|=0.0）
# → physio 先验对 Zero 内核**恒无影响**（M2 强制 Πv=MIN_PRECISION 是主因）。
#
# ⚠ physio 有**两档口径**，本类两档都 pin（此前只 pin 了模态级，而线上发的是合并后那档）：
#   模态级 `recommended_precision(PHYSIO)` = (MIN, 0.18) → Π̄=0.0905、门槛 1.9890；
#   **线上 wire** = EDA/HRV 经 ω=0.5 CI 预合并（`merge_physio` **默认开**）后的单条 physio 流，
#   Πa=0.5·0.15+0.5·0.20=0.175 → Π̄=0.0880、门槛 2.0455（更不可达，方向不变）。
#   即改 `PHYSIO_SUBSOURCE_PRECISION_A` 才是改线上载荷，改 `EXTERNAL_PHYSIO_PRECISION_A` 在
#   EDA+HRV 同在时是**死旋钮**。
#
# 本类不是「缺陷断言」而是**契约级事实的特征化**。同时把 Zero 必 ping 清单里尚无自动守卫的
# 三项（σ 标度 / precision α·β / 点燃门常量）一并 pin。
#
# ⚠ **本类全绿 ≠ Zero 认可现判据**。Zero 科学家议会 2026-07-28 终裁（其
# `notes/2026-07-28-ignition-gate-external-priors-council.md` §六 Q1 + `PRP/外部多模态先验流注入口/
# design.md` §五附）已判定 `hypot(μ)·mean(Π)` 是**范畴错误**（同一数字既当齐次相对权重又当非齐次
# 绝对判据），修法=按轴加权马氏距离 `D=sqrt(Πv·μv²+Πa·μa²)` + `θ'=0.28`，须走完整 PRP、近期不落地。
# 绿只表示**旧判据尚未被替换**。
#
# ⚠ **本类探测得到什么、探测不到什么**（前一版注释在此处写了兑现不了的承诺，已订正）：
#   探测得到：常量**值**漂移（`_ZERO_GATE_CONSTANTS` 逐值 pin）、本仓推荐精度/子源可靠度改动。
#   探测**不到**：判据**公式**被替换。因为 `_mean_precision` 是旧式的**手抄镜像**，从不读 Zero
#   `stream_salience` 的函数体；而终裁明令新公式以 **default-off 新增** selector + **新** θ' 常量
#   落地、`SALIENCE_THRESHOLD=0.18` 与旧式**原样保留**（对标 fuse_independence_correct 先例）
#   → Zero PRP 落地当天本类**全绿通过**。届时须按新公式重标，而非把绿当作「无事发生」。
#
# ⚠ **待重标（Zero 07-28 二轮回执，等最终形态 ping 后动手）**：去地板+注入本仓 wire 载荷实测
#   D=sqrt(0.175·0.845714²)=0.3538 ≥ θ'=0.28（本仓真 merge_physio_priors 复算一致）→
#   「physio 恒不可点燃」「兜底分支结构性不可达」两条特征化届时**双双失效**；且归一化从
#   「推后」改为「与去地板打包必做」（否则 physio 低精度单流独占后验，arousal 被拉到 0.846）。
#   Zero 将 ping 最终公式+θ'+是否含归一化+建议 pin 项；落地前本节保持现状=特征化旧世界。
# ---------------------------------------------------------------------------

_ZERO_AFFECT_MATH_PY = _ZERO_SRC / "agents" / "affect_math.py"
_NUM = r"[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"

# Zero 模块级门控/边界常量（必 ping：点燃门 + σ/精度边界）
_ZERO_GATE_CONSTANTS: dict[str, float] = {
    "MIN_SIGMA": 0.05,
    "MIN_PRECISION": 1e-3,
    "MAX_SAMPLE_SIGMA": 0.5,
    "SURVIVAL_PRECISION": 0.4,
    "SALIENCE_THRESHOLD": 0.18,
    "AROUSAL_GAIN": 1.0,
}

# occ_prior σ 标度系数（affect_math.py:126-127）——Zero 07-28 ④ 订正：**这才是真旋钮**
# （MIN_SIGMA 在 MCP 路径恒不咬合：mapping.py:53 钳 |I|≤1 → σ∈[0.10,0.35] 恒 >0.05）。
_ZERO_SIGMA_CONF_RE = re.compile(rf"conf\s*=\s*clamp\(({_NUM})\s*\+\s*({_NUM})\s*\*\s*abs\(")
_ZERO_SIGMA_RE = re.compile(rf"sigma\s*=\s*max\(MIN_SIGMA,\s*({_NUM})\s*\*\s*\(1\.0\s*-\s*conf\)")
# precision() 的 α/β 默认（affect_math.py:155）——在 sigmoid **分子**，直接线性缩放判别信号
_ZERO_ALPHA_BETA_RE = re.compile(
    rf"def precision\([^)]*alpha:\s*float\s*=\s*({_NUM})[^)]*beta:\s*float\s*=\s*({_NUM})",
    re.DOTALL,
)
# 默认硬门哨兵：IGNITION_BETA = None → ignite 走 step 分支（软门是显式非默认路径）
_ZERO_IGNITION_BETA_RE = re.compile(r"^IGNITION_BETA\s*:[^=]*=\s*(\w+)", re.MULTILINE)

_MU_MAX_NORM = 2.0**0.5  # ModalityPrior 各维 ∈[-1,1] → hypot 上界 √2


def _mean_precision(precision: tuple[float, float]) -> float:
    """镜像 Zero stream_salience 的精度项：mean(Π)。"""
    return 0.5 * (precision[0] + precision[1])


# fast_survival_prior 的 arousal 式，**基线项可选**。Zero 07-28 二轮回执确认去地板
# （clamp(0.5+0.5|I|) → clamp(0.5|I|)）已列复议终裁必改项，并要求本守卫在其动手当天
# 变红、勿改宽松。上一版正则硬性要求 `+` 基线，恰好对这条改动 no-match→skip（默认
# 模式黄灯，仅 STRICT 转红）——守卫承诺的红是它给不出的。现基线缺省按 0.0 进入主
# 断言 → 去地板落地当天全模式红。
# ⚠ 必须先切出 fast_survival_prior 函数体再匹配：基线一变可选，全文件 search 会被
# occ_prior 的多行 `clamp(\n 0.4·|I| + …)`（affect_math.py:118）抢先匹配成「无基线」，
# 在 Zero 未动手时就假阳性红（本仓 07-28 实踩，见判别性自证测试）。
_ZERO_SURVIVAL_AROUSAL_RE = re.compile(
    rf"arousal\s*=\s*clamp\(\s*(?:({_NUM})\s*\+\s*)?({_NUM})\s*\*\s*abs\(intensity\)"
)
_ZERO_SURVIVAL_FUNC_RE = re.compile(
    r"def fast_survival_prior\(.*?(?=\ndef |\nclass |\Z)", re.DOTALL
)


def _survival_arousal_floors(source: str) -> list[float] | None:
    """列出 fast_survival_prior 函数体内**所有** arousal 式的基线（地板）。

    返回 None = 函数不在位（Zero 结构大改/文件不对）→ 调用方 skip。
    返回 []   = 函数在位但无可解析 arousal 式（如基线被写成条件表达式）→ 调用方判红。
    返回多元素 = 存在多个分支形态（门控落地的典型形状）→ 调用方判红。
    无基线项（`clamp(0.5*|I|)`）记 0.0，使其进入主断言而非逃进 skip。
    """
    func_match = _ZERO_SURVIVAL_FUNC_RE.search(source)
    if func_match is None:
        return None
    return [
        float(m.group(1)) if m.group(1) is not None else 0.0
        for m in _ZERO_SURVIVAL_AROUSAL_RE.finditer(func_match.group(0))
    ]


@pytest.mark.zerorepo
class TestIgnitionGateReachabilityCrosscheck:
    """Zero 点燃门常量 pin + 本仓推荐精度在该门下的可达性特征化。

    D:\\Zero / affect_math.py 不在位或结构变更 → skip；**常量或可达性漂移 → hard fail**。
    """

    def _source(self) -> str:
        if not _zero_available():
            pytest.skip(f"D:\\Zero\\src 不存在（{_ZERO_SRC}），跳过点燃门跨仓断言")
        if not _ZERO_AFFECT_MATH_PY.is_file():
            pytest.skip(f"Zero affect_math.py 不存在（{_ZERO_AFFECT_MATH_PY}），跳过")
        return _ZERO_AFFECT_MATH_PY.read_text(encoding="utf-8")

    def test_gate_and_boundary_constants_pinned(self) -> None:
        """Zero 门控/边界常量逐值 pin（必 ping：点燃门 + σ/精度边界）。"""
        source = self._source()
        for name, expected in _ZERO_GATE_CONSTANTS.items():
            match = re.search(rf"^{name}\s*=\s*({_NUM})", source, re.MULTILINE)
            if match is None:
                pytest.skip(f"未在 Zero affect_math.py 找到常量 {name}，结构可能变更，跳过")
            actual = float(match.group(1))
            assert actual == expected, (
                f"Zero 门控常量漂移：{name} 期望 {expected}、实际 {actual}。"
                "本仓判别裕度/先验可达性依赖此值——须跨仓协调（Zero 07-28 必 ping 约定）。"
            )

    def test_occ_prior_sigma_coefficients_pinned(self) -> None:
        """occ_prior σ 标度系数 pin（Zero ④ 订正：真旋钮，非 MIN_SIGMA）。"""
        source = self._source()
        conf_m = _ZERO_SIGMA_CONF_RE.search(source)
        sigma_m = _ZERO_SIGMA_RE.search(source)
        if conf_m is None or sigma_m is None:
            pytest.skip("Zero occ_prior σ 公式结构变更，正则未命中，跳过")
        conf_base, conf_gain = float(conf_m.group(1)), float(conf_m.group(2))
        sigma_scale = float(sigma_m.group(1))
        assert (conf_base, conf_gain, sigma_scale) == (0.3, 0.5, 0.5), (
            f"occ_prior σ 标度漂移：conf={conf_base}+{conf_gain}·|I|、σ={sigma_scale}·(1−conf)，"
            "期望 (0.3, 0.5, 0.5)。σ 减半 ⇒ π_total×4 ⇒ 判别裕度≈1/4（Zero 07-28 ④）。"
        )

    def test_precision_alpha_beta_defaults_pinned(self) -> None:
        """precision() 的 α=1.0 / β=0.5 默认 pin（在 sigmoid 分子，直接缩放判别信号）。"""
        source = self._source()
        match = _ZERO_ALPHA_BETA_RE.search(source)
        if match is None:
            pytest.skip("Zero precision() 签名结构变更，正则未命中，跳过")
        alpha, beta = float(match.group(1)), float(match.group(2))
        assert (alpha, beta) == (1.0, 0.5), (
            f"precision() α/β 漂移：期望 (1.0, 0.5)、实际 ({alpha}, {beta})——"
            "π_da=sigmoid(α|δ|+βV) 的分子项，改动直接线性缩放我方判别信号。"
        )

    def test_ignition_default_is_hard_gate(self) -> None:
        """IGNITION_BETA 默认 None（硬 step 门）——软门是显式非默认路径。"""
        source = self._source()
        match = _ZERO_IGNITION_BETA_RE.search(source)
        if match is None:
            pytest.skip("未找到 IGNITION_BETA 声明，结构可能变更，跳过")
        assert match.group(1) == "None", (
            f"IGNITION_BETA 默认漂移为 {match.group(1)!r}（期望 None=硬 step 门）。"
            "转软门后所有流均参与融合、亚阈先验不再被丢——本仓可达性结论须整体重评。"
        )

    def test_physio_prior_is_unreachable_under_gate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """**特征化**：physio 推荐精度下最大可达 salience < 阈值 → 对内核恒无影响。

        这是当前跨仓事实（非期望）。Zero 降阈值 / 本仓抬 EXTERNAL_PHYSIO_PRECISION_A
        使其可达时本例即红——正是需要跨仓知会的时刻。
        """
        source = self._source()
        match = re.search(rf"^SALIENCE_THRESHOLD\s*=\s*({_NUM})", source, re.MULTILINE)
        if match is None:
            pytest.skip("未找到 SALIENCE_THRESHOLD，跳过")
        threshold = float(match.group(1))

        for key in ("EXTERNAL_PHYSIO_PRECISION_V", "EXTERNAL_PHYSIO_PRECISION_A"):
            monkeypatch.delenv(key, raising=False)
        physio_prec = recommended_precision(ModalityKind.PHYSIO)
        max_salience = _MU_MAX_NORM * _mean_precision(physio_prec)

        assert max_salience < threshold, (
            f"physio 可达性变了：推荐精度 {physio_prec} 下最大 salience={max_salience:.4f} "
            f"已 ≥ 阈值 {threshold}——physio 先验从「恒被丢弃」变为可点燃，"
            "跨仓消费语义改变，须与 Zero 确认（本仓 07-28 实测原为不可达）。"
        )

    def test_merged_physio_wire_is_unreachable_under_gate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**特征化·线上口径**：EDA/HRV 经默认 ω=0.5 预合并后真正发出的 physio 流仍不可达。

        与上一例的区别：上例盯模态级 `recommended_precision(PHYSIO)`（Πa=0.18），
        但 `merge_physio` **默认开**，线上 wire 是合并后的单条流（Πa=0.175）。
        改 `PHYSIO_SUBSOURCE_PRECISION_A` 才会改线上载荷，而上一例对它完全不敏感——
        少了本例，抬子源可靠度可让生产 physio 真变可点燃而两组守卫全绿。

        这里跑**真的** `merge_physio_priors`（非手抄推导），故合并公式本身改动亦会被本例接住。
        """
        source = self._source()
        match = re.search(rf"^SALIENCE_THRESHOLD\s*=\s*({_NUM})", source, re.MULTILINE)
        if match is None:
            pytest.skip("未找到 SALIENCE_THRESHOLD，跳过")
        threshold = float(match.group(1))

        monkeypatch.delenv("ZERO_PHYSIO_MERGE_OMEGA", raising=False)
        for key in ("EXTERNAL_PHYSIO_PRECISION_V", "EXTERNAL_PHYSIO_PRECISION_A"):
            monkeypatch.delenv(key, raising=False)

        physio_prec = recommended_precision(ModalityKind.PHYSIO)
        # μ 取合法域顶格（|μa|=1），本例只关心精度项决定的可达上界
        merged = merge_physio_priors(
            [
                ModalityPrior(modality="eda/sc", mu=(0.0, 1.0), precision=physio_prec),
                ModalityPrior(modality="hrv/rmssd", mu=(0.0, 1.0), precision=physio_prec),
            ]
        )
        assert len(merged) == 1, f"预合并未产出单条流：{[p.modality for p in merged]}"

        wire_prec = merged[0].precision
        max_salience = _MU_MAX_NORM * _mean_precision(wire_prec)
        onset = threshold / _mean_precision(wire_prec)

        assert wire_prec[1] == pytest.approx(0.175, abs=1e-9), (
            f"线上 physio 流的 Πa 漂移为 {wire_prec[1]}（期望 0.175=0.5·0.15+0.5·0.20）——"
            "子源可靠度分层或 ω 档位被改动，须与 Zero 议会裁定核对（ω=0.5 为终裁值，勿换档）。"
        )
        assert max_salience < threshold, (
            f"**线上** physio 可达性变了：合并后 Π={wire_prec} 下最大 salience="
            f"{max_salience:.4f} 已 ≥ 阈值 {threshold}（门槛 |μ|≥{onset:.4f}）——"
            "实际发给 Zero 的生理先验从「恒被丢弃」变为可点燃，须跨仓知会。"
        )

    def test_face_audio_ignition_thresholds_pinned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """face/audio 跨阈 |μ| 门槛在合法域内且逐值 pin（锐阶跃边界，实测吻合）。"""
        source = self._source()
        match = re.search(rf"^SALIENCE_THRESHOLD\s*=\s*({_NUM})", source, re.MULTILINE)
        if match is None:
            pytest.skip("未找到 SALIENCE_THRESHOLD，跳过")
        threshold = float(match.group(1))

        for key in (
            "EXTERNAL_FACE_PRECISION_V",
            "EXTERNAL_FACE_PRECISION_A",
            "EXTERNAL_AUDIO_PRECISION_V",
            "EXTERNAL_AUDIO_PRECISION_A",
        ):
            monkeypatch.delenv(key, raising=False)

        expected_onsets = {ModalityKind.FACE: 1.1250, ModalityKind.AUDIO: 1.0286}
        for kind, expected in expected_onsets.items():
            onset = threshold / _mean_precision(recommended_precision(kind))
            assert onset == pytest.approx(expected, abs=1e-4), (
                f"{kind} 跨阈门槛漂移：期望 |μ|≥{expected}、实际 {onset:.4f}"
            )
            assert onset <= _MU_MAX_NORM, (
                f"{kind} 跨阈门槛 {onset:.4f} 已超出 μ 合法域 √2={_MU_MAX_NORM:.4f}——"
                "该模态先验变为恒不可点燃（与 physio 同命），须跨仓知会。"
            )

    def test_survival_stream_always_ignites(self) -> None:
        """SURVIVAL_PRECISION ⊗ SALIENCE_THRESHOLD **绑定**：默认配置下 survival 恒过阈。

        推论：`ignite` 的 `if not fired: fired=[max by salience]` 兜底分支在**默认配置下**
        不可达（survival arousal 恒 ≥0.5、Π 恒 (0.4,0.4) → salience ≥0.200 > 0.18）。故外部
        亚阈先验等不到「全场亚阈时当选 max」的逃逸机会。两常量任一动即须同评。

        ⚠ **2026-07-29 重标**：Zero 已落 `arousal_floor_fix` 门控（commit 4760dfb，
        「线A」），本例按上一版结构断言如期变红——那是设计意图，不是缺陷。重标后本例改为
        **按分支各自判定**：默认分支（带地板）仍须过阈；门开分支（去地板）则**已知不过阈**，
        这条不再是「结构性不可达」而是「默认配置下不可达」。见下方 `test_survival_max_fallback_
        becomes_reachable_when_floor_fix_gate_opens` —— 门开后的可达性单列一条正面刻画。
        """
        source = self._source()
        consts = {}
        for name in ("SURVIVAL_PRECISION", "SALIENCE_THRESHOLD"):
            match = re.search(rf"^{name}\s*=\s*({_NUM})", source, re.MULTILINE)
            if match is None:
                pytest.skip(f"未找到 {name}，跳过")
            consts[name] = float(match.group(1))

        # ⚠ 此处**从 Zero 源码读** fast_survival_prior 的 arousal 基线系数，不再手抄。
        # 现行形态（Zero 复议 §六(e):186 裁定的 default-off 门控）：两分支，基线 {0.0, 0.5}。
        # 仍要求「形态一变就红」：任何第三种分支、或基线集合变化，都落到下面的逐值断言上。
        floors = _survival_arousal_floors(source)
        if floors is None:
            pytest.skip("未找到 fast_survival_prior 函数体，Zero 结构可能大改，跳过")
        assert sorted(floors) == [0.0, 0.5], (
            f"fast_survival_prior 的 arousal 分支基线集合变为 {sorted(floors)}（期望 "
            "[0.0, 0.5] = 门开去地板 / 门关带地板两分支）——Zero 又动了 survival 证据式，"
            "须重新核对默认路径走哪条，勿放宽本断言。"
        )

        # 默认路径 pin：`arousal_floor_fix` 形参默认 False = 走**带地板**分支。
        # 这一条是「哪条是默认」的唯一真相；只 pin 基线集合而不 pin 默认值，Zero 把默认翻成
        # True 的那天本例照样绿——正是本仓 pitfalls ② 的失败模式。
        default_match = re.search(r"arousal_floor_fix\s*:\s*bool\s*=\s*(True|False)", source)
        assert default_match is not None, (
            "未找到 arousal_floor_fix 形参默认值——门控形态又变了，须按新形态重标。"
        )
        assert default_match.group(1) == "False", (
            "arousal_floor_fix 默认值翻为 True（去地板成为默认路径）——survival 在 I=0 时"
            "salience=0，ignite 的 max-fallback 变为默认可达，本仓可达性结论须整体重评。"
        )

        # intensity=0 时取基线项；斜率项 ≥0 故基线即下确界（clamp 下界 0.0 不咬合）
        min_survival_arousal = max(floors)  # 默认分支 = 带地板那条

        min_survival_salience = min_survival_arousal * consts["SURVIVAL_PRECISION"]
        assert min_survival_salience > consts["SALIENCE_THRESHOLD"], (
            f"survival 流最小 salience={min_survival_salience:.4f} 不再 > 阈值 "
            f"{consts['SALIENCE_THRESHOLD']}——ignite 的 max-fallback 分支变为可达，"
            "外部亚阈先验可能在全场亚阈时被保留，本仓可达性结论须重评。"
        )

    def test_survival_max_fallback_becomes_reachable_when_floor_fix_gate_opens(self) -> None:
        """门开（`ZERO_IGNITION_GATE_FUSION=false`）后 survival 在 I=0 时**不再过阈**。

        这是 Zero 07-29 落地带来的**新事实**，本仓可达性结论的适用范围因此收窄：
        「外部亚阈先验永远等不到 max-fallback」只在**门关（默认）**下成立。门一开，
        足够平淡的输入下全场可能亚阈，此时 `ignite` 的 max-fallback 会挑出 salience
        最大者——**本仓的 physio 先验（恒亚阈）由此获得被保留的路径**。

        ⚠ 「平淡」不等于「intensity=0」：去地板只清掉 arousal 项，**valence 项
        `clamp(0.6·goal)` 原样保留**。故门开后 survival 亚阈的真实条件是
        `intensity≈0` **且** `|goal| < 阈值/(Π·0.6)`。本例按该条件判定，不用
        「基线=0 → 恒亚阈」那种看着成立、实则与 Zero 改什么都无关的空断言。

        本例把这条正面刻画钉住：若 Zero 日后给去地板分支补回下限、或调高 valence 系数
        使可行域塌空，本例变红 → 提醒我们把「门开=physio 有逃逸机会」这个结论撤回。
        """
        source = self._source()
        consts = {}
        for name in ("SURVIVAL_PRECISION", "SALIENCE_THRESHOLD"):
            match = re.search(rf"^{name}\s*=\s*({_NUM})", source, re.MULTILINE)
            if match is None:
                pytest.skip(f"未找到 {name}，跳过")
            consts[name] = float(match.group(1))
        threshold = consts["SALIENCE_THRESHOLD"]
        precision = consts["SURVIVAL_PRECISION"]

        floors = _survival_arousal_floors(source)
        if floors is None:
            pytest.skip("未找到 fast_survival_prior 函数体，Zero 结构可能大改，跳过")
        assert sorted(floors) == [0.0, 0.5], f"分支基线集合变为 {sorted(floors)}，须重标"

        # valence 系数从源码读（不手抄）：`valence = clamp(0.6 * goal, -1.0, 1.0)`
        func_match = _ZERO_SURVIVAL_FUNC_RE.search(source)
        assert func_match is not None
        v_match = re.search(rf"valence\s*=\s*clamp\(\s*({_NUM})\s*\*\s*goal", func_match.group(0))
        if v_match is None:
            pytest.skip("survival valence 式形态变更，跳过（须按新形态重标）")
        valence_coef = float(v_match.group(1))

        # 门开分支 = 无基线项那条 → intensity=0 时 arousal=0，salience = |coef·goal|·Π。
        # 亚阈可行域：|goal| < threshold/(Π·coef)。goal∈[-1,1]，故该域非空 ⇔ 上界 > 0。
        gate_open_floor = min(floors)
        assert gate_open_floor == 0.0, "门开分支不再是「无基线」形态，须重标"
        goal_bound = threshold / (precision * valence_coef)
        assert 0.0 < goal_bound, "亚阈可行域为空——门开后 survival 仍恒过阈，结论须撤回"
        assert goal_bound == pytest.approx(0.75, abs=1e-9), (
            f"门开后 survival 的亚阈条件漂移：|goal| < {goal_bound:.4f}（期望 0.75="
            f"{threshold}/({precision}·{valence_coef})）——本仓「门开=physio 有逃逸机会」"
            "的适用范围随之改变，须跨仓同评。"
        )

        # 反面对照（非冗余）：门关分支下**任何** goal 都过阈——0.5·Π=0.200 > 0.18 与 goal 无关。
        # 这条是上一例结论的另一面：可达性的翻转完全由门决定，不由输入决定。
        assert max(floors) * precision > threshold, (
            "门关分支下 survival 已非恒过阈——与 test_survival_stream_always_ignites 冲突，"
            "两例须同评。"
        )

        # 门与门的绑定：floor_fix 不是独立旋钮，而是 gate_fusion 的取反（Zero 议会 D5 强制
        # 共用同一开关）。这条绑定一旦解开，两个门可各自独立开合 → 组合态爆炸，本仓的
        # 二态刻画（门关/门开）不再充分。
        core_py = _ZERO_SRC / "agents" / "affect_core.py"
        if not core_py.is_file():
            pytest.skip(f"Zero affect_core.py 不存在（{core_py}），跳过绑定核对")
        core_source = core_py.read_text(encoding="utf-8")
        assert "arousal_floor_fix=not state.gate_fusion" in core_source, (
            "affect_core 不再把 arousal_floor_fix 绑定为 `not gate_fusion`——两门可独立开合，"
            "本仓「门关/门开」二态刻画不再充分，须按新组合态重标可达性结论。"
        )

    def test_survival_floor_guard_goes_red_not_skip_on_confirmed_change(self) -> None:
        """判别性自证：对 Zero 已确认必改的去地板形态，上例走**红**（断言失败）而非 skip。

        「绿灯必须先证明它能红」同族第五例：上一版正则硬性要求 `+` 基线项，恰好对
        「Zero 动手当天」那条改动（clamp(0.5+0.5|I|)→clamp(0.5|I|)）no-match→skip
        （默认模式黄灯，仅 STRICT 兜底转红）——守卫承诺的红是它给不出的。本例实证五态：
        (1) 现行地板形态解析 [0.5]——且 decoy 在场不被抢注（反例非假想：首版可选基线正则
        就被 occ_prior 的多行 clamp(0.4·|I|+…) 全文件抢先匹配、把真地板误读成 0.0，
        Zero 未动手先假阳性红，07-28 实踩）；(2) 裸去地板形态解析 **[0.0] 而非 None**
        （不再逃进 skip，主断言 0.0·0.4 ≤ 0.18 必失败=红）；(3) **门控多分支形态**必须解析出
        **两个**元素而非只取第一个匹配——2026-07-29 Zero 已按此形态落地（commit 4760dfb），
        上例遂改为「基线集合逐值 pin + 默认值 pin」；只取首个匹配会让「走 legacy 分支照样绿」，
        即 default-off 落地当天守卫失明（本仓 pitfalls ② 失败模式）；(4) 条件表达式等不可解析
        形态 → **空列表**（非 None）→ 断言红；(5) 函数整体缺位 → None → skip。
        (6) **默认值翻转**（`arousal_floor_fix: bool = True`）→ 默认值 pin 断言红——这是重标后
        新增的一格：门控形态已成既定事实，此后唯一能悄悄改变默认路径的就是翻这个默认值。
        如实标注守不住什么：(5) 这一格（Zero 整个重命名/删除 fast_survival_prior）本守卫
        只能 skip、靠 STRICT 兜底转红。不依赖 D:\\Zero 在位，判别力常年在测。
        """
        decoy_occ_prior = (
            "def occ_prior(appraisal):\n"
            "    arousal = clamp(\n"
            "        0.4 * abs(intensity)\n"
            "        + va_coupling_neg * max(-valence, 0.0),\n"
            "        -1.0,\n"
            "        1.0,\n"
            "    )\n"
        )
        floored = decoy_occ_prior + (
            "def fast_survival_prior(features):\n"
            "    arousal = clamp(0.5 + 0.5 * abs(intensity), 0.0, 1.0)\n"
        )
        floorless = decoy_occ_prior + (
            "def fast_survival_prior(features):\n"
            "    arousal = clamp(0.5 * abs(intensity), 0.0, 1.0)\n"
        )
        # Zero 复议 §六(e) 的 default-off 门控落地形状（floor 与 D formula 同一总开关）
        gated_branches = decoy_occ_prior + (
            "def fast_survival_prior(features, use_axis_weighted_gate=False):\n"
            "    if use_axis_weighted_gate:\n"
            "        arousal = clamp(0.5 * abs(intensity), 0.0, 1.0)\n"
            "    else:\n"
            "        arousal = clamp(0.5 + 0.5 * abs(intensity), 0.0, 1.0)\n"
        )
        unknown_shape = decoy_occ_prior + (
            "def fast_survival_prior(features):\n"
            "    arousal = clamp(sigmoid(intensity), 0.0, 1.0)\n"
        )

        assert _survival_arousal_floors(floored) == [0.5], (
            "地板在场时必须解析出 [0.5]——若为 [0.0] 说明又被函数体外的 decoy 抢注（假阳性）"
        )
        floors = _survival_arousal_floors(floorless)
        assert floors == [0.0], "去地板形态必须解析为 [0.0] 而非 None——否则上例 skip 不红"
        assert not (
            floors[0] * _ZERO_GATE_CONSTANTS["SURVIVAL_PRECISION"]
            > _ZERO_GATE_CONSTANTS["SALIENCE_THRESHOLD"]
        ), "去地板后上例主断言应不成立（红）——若此处成立说明判别力自证失效"
        gated = _survival_arousal_floors(gated_branches)
        assert gated is not None and len(gated) == 2, (
            f"门控多分支形态须解析出两个形态（实际 {gated}）——只取第一个匹配会让"
            "「走 legacy 分支照样绿」，即 default-off 落地当天守卫失明"
        )
        assert _survival_arousal_floors(unknown_shape) == [], (
            "不可解析形态须返回空列表（→上例断言红），返回 None 会逃进 skip"
        )
        assert _survival_arousal_floors(decoy_occ_prior) is None

        # (6) 默认值 pin 的判别力：门控形态既已落地，唯一能悄悄改默认路径的就是翻这个默认值。
        # 用与上例**同一条正则**跑两份合成源码，证明它分得开 False / True，而非恒绿。
        default_re = r"arousal_floor_fix\s*:\s*bool\s*=\s*(True|False)"
        legacy_sig = "def fast_survival_prior(features, *, arousal_floor_fix: bool = False):\n"
        flipped_sig = "def fast_survival_prior(features, *, arousal_floor_fix: bool = True):\n"
        legacy_match = re.search(default_re, legacy_sig)
        flipped_match = re.search(default_re, flipped_sig)
        assert legacy_match is not None and legacy_match.group(1) == "False"
        assert flipped_match is not None and flipped_match.group(1) == "True", (
            "默认值翻为 True 时正则须解析出 True（→上例断言红）；解析不出会逃进"
            "「未找到默认值」那条 assert，同样是红，但归因会指错方向"
        )


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
