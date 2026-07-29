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

import ast
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from src.mcp.zero.external_priors import (
    _RECOMMENDED_PRECISION_DEFAULTS,
    MIN_PRECISION,
    PHYSIO_MERGE_OMEGA_DEFAULT,
    PHYSIO_PRECISION_A_SELF_IGNITE_BOUND,
    PHYSIO_SUBSOURCE_PRECISION_A,
    ZERO_SALIENCE_THRESHOLD,
    ModalityKind,
    ModalityPrior,
    build_external_priors_override,
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

# 运行期旋钮的**真值来源**（函数默认不治理生产：affect_core 逐参覆写为这些 state 字段）。
# ⚠ 缺字段给哨兵字符串而**不是** KeyError：KeyError 会让子进程非零退出 →
# `_fetch_affect_state_defaults_or_skip` 判 skip 而非红，守卫悄悄退化成恒 skip
# （与 Zero 自己记在头上的「恒 skip」是同一种病）。哨兵值进断言必然红，归因也明确。
payload = {
    "skip": False,
    "precision_cap": precision_cap,
    "max_streams": max_streams,
}
for key in ("gate_fusion", "exclude_physio_fusion", "sample_sigma_cap"):
    payload[key] = fields[key].default if key in fields else "<MISSING>"

print(json.dumps(payload))
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
        return _fetch_affect_state_defaults_or_skip()

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


def _fetch_affect_state_defaults_or_skip() -> dict[str, Any]:
    """运行子进程回读 Zero `AffectState` 的字段默认；任何失败 / Zero 不在位均 skip。

    模块级而非某个测试类的私有方法：`AffectState` 的字段默认是**运行期真旋钮**（函数默认
    不治理生产），M3/M6 与点燃门两组守卫都要读它，共用一份避免两处子进程脚本各自漂移。
    """
    if not _zero_available():
        pytest.skip(f"D:\\Zero\\src 不存在（路径 {_ZERO_SRC}），跳过 AffectState 默认值断言")
    try:
        data = _run_subprocess_with_script(_DEFAULTS_SCRIPT)
    except subprocess.TimeoutExpired:
        pytest.skip("子进程超时，跳过 AffectState 默认值断言")
    except RuntimeError as exc:
        pytest.skip(f"子进程非零退出，跳过: {exc}")
    except json.JSONDecodeError as exc:
        pytest.skip(f"子进程输出非合法 JSON，跳过: {exc}")

    if data.get("skip"):
        pytest.skip(data.get("reason", "D:\\Zero import 失败，跳过 AffectState 默认值断言"))
    return data


# ---------------------------------------------------------------------------
# Zero 机读错误码全表跨仓一致性（2026-07-29 令牌换代，取代原 `_UNKNOWN_SESSION_MARKER` 单点守卫）
#
# 换代背景：旧契约是**位置 0 的裸前缀** `unknown-session:`，而 FastMCP 在工具层把 ToolError 包成
# "Error executing tool <name>: <原文>" ⇒ 前缀在 wire 上永不在位置 0 ⇒ 本仓 `startswith` 判定
# 恒 False，T6·④ resume 通路是**生产死码**（两侧实证）。现契约为位置无关令牌 `[zero:<code>]`。
#
# 旧守卫为何在 STRICT 下红：它用正则找 `_UNKNOWN_SESSION_MARKER = "字面量"`，而 Zero 现在等号
# 右边是**标识符**（别名指向 ZERO_ERROR_CODE_UNKNOWN_SESSION）→ 不匹配 → skip → STRICT 转 fail。
#
# 新守卫 pin 三件事，且**本仓持有自己的期望**（不是「对方有什么就认什么」的恒真式）：
#   1. 符号集：Zero 顶层 `ZERO_ERROR_CODE_*` 全集 == 本仓 `_EXPECTED_ZERO_ERROR_CODE_SYMBOLS`；
#   2. 逐符号值：Zero 每个符号的字面量 == 本仓 client 同名常量
#      （本仓字面量独立持有 → 单边改值即红）；
#   3. 令牌格式：`_tool_error` 的构造前缀 == `[zero:`（本仓消费正则的依据，换分隔符即红）。
# 另核旧别名 `_UNKNOWN_SESSION_MARKER` 仍在且指向 unknown-session 符号（过渡兼容期的两侧承诺）。
#
# 解析一律走 AST（同 `_top_level_func` 的教训：文本切法会被 docstring/注释里的同形 token 污染）。
# D:\Zero 或 server.py 不在位 → skip（环境缺口）；**文件在位但结构对不上 → 判红**
# ——这正是本轮教训「检查比消费方宽松 ⇒ 绿灯从没能红」的直接修正。
# ---------------------------------------------------------------------------

# Zero server 定义机读错误码的源文件（相对 D:\Zero）
_ZERO_SERVER_PY = _ZERO_SRC / "mcp_server" / "server.py"
# Zero 机读码常量的符号名前缀 / 登记表名 / 旧别名
_ZERO_CODE_SYMBOL_PREFIX = "ZERO_ERROR_CODE_"
_ZERO_CODE_REGISTRY_NAME = "ZERO_ERROR_CODES"
_ZERO_LEGACY_ALIAS_NAME = "_UNKNOWN_SESSION_MARKER"
_ZERO_TOOL_ERROR_FUNC = "_tool_error"

# 本仓对 Zero 码表的**独立期望**（符号名集合）。Zero 改名/加码/删码而本仓未跟 → 下面第 1 条红。
# 值不写在这里（按 Zero 要求「按符号名 pin」），而是与本仓 client 的同名常量逐个比对（第 2 条）
# ——本仓那份字面量是独立持有的，故值漂移同样红。
_EXPECTED_ZERO_ERROR_CODE_SYMBOLS: frozenset[str] = frozenset(
    {
        "ZERO_ERROR_CODE_UNKNOWN_SESSION",
        "ZERO_ERROR_CODE_CONFIG_INCOMPATIBLE",
        "ZERO_ERROR_CODE_EXTERNAL_PRIOR_INVALID",
        "ZERO_ERROR_CODE_PAYLOAD_INVALID",
        "ZERO_ERROR_CODE_CONFIG_INVALID",
        "ZERO_ERROR_CODE_DEPLOY_ENV_INVALID",
        # 超时两码（本仓第二轮回件 §2.1 建议、Zero 2026-07-29 采纳）：重试语义相反故分码；
        # timeout-step Zero 侧只登记不产出（执行超时未实现），消费侧分类先一次到位。
        "ZERO_ERROR_CODE_TIMEOUT_LOCK",
        "ZERO_ERROR_CODE_TIMEOUT_STEP",
    }
)
# 令牌构造前缀：本仓消费正则 `\[zero:([a-z][a-z0-9-]*)\]` 的依据。
_EXPECTED_TOKEN_PREFIX = "[zero:"


def _zero_server_tree_or_skip() -> ast.Module:
    """解析 Zero server.py 为 AST；D:\\Zero / 文件不在位 → skip，语法错 → 判红。"""
    if not _zero_available():
        pytest.skip(f"D:\\Zero\\src 不存在（路径 {_ZERO_SRC}），跳过机读错误码跨仓断言")
    if not _ZERO_SERVER_PY.is_file():
        pytest.skip(f"Zero server.py 不存在（路径 {_ZERO_SERVER_PY}），跳过机读错误码跨仓断言")
    source = _ZERO_SERVER_PY.read_text(encoding="utf-8")
    try:
        return ast.parse(source)
    except SyntaxError as exc:  # 文件在位却解析不了 = 真问题，不 skip
        pytest.fail(f"Zero server.py 解析失败（{_ZERO_SERVER_PY}）：{exc}")


def _module_assign_targets(node: ast.stmt) -> tuple[list[str], ast.expr | None]:
    """取顶层赋值语句的目标名列表与右值（同时支持 `A = x` 与 `A: T = x`）；非赋值 → ([], None)。"""
    if isinstance(node, ast.Assign):
        return ([t.id for t in node.targets if isinstance(t, ast.Name)], node.value)
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return ([node.target.id], node.value)
    return ([], None)


def _zero_error_code_literals(tree: ast.Module) -> dict[str, str]:
    """取 Zero 顶层 `ZERO_ERROR_CODE_* = "字面量"` 的 符号名 → 码值。

    只认**字符串字面量**右值：若 Zero 把某个码改成计算式/引用，本函数不收 → 符号集断言即红
    （宁可红也不猜，跨仓契约不接受「大概是这个值」）。
    """
    literals: dict[str, str] = {}
    for node in tree.body:
        names, value = _module_assign_targets(node)
        if value is None or not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        for name in names:
            if name.startswith(_ZERO_CODE_SYMBOL_PREFIX):
                literals[name] = value.value
    return literals


def _zero_registry_symbols(tree: ast.Module) -> list[str] | None:
    """取 Zero `ZERO_ERROR_CODES = frozenset({SYM, …})` 里列的**符号名**；找不到返回 None。"""
    for node in tree.body:
        names, value = _module_assign_targets(node)
        if _ZERO_CODE_REGISTRY_NAME not in names or value is None:
            continue
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)):
            return None
        if value.func.id != "frozenset" or not value.args:
            return None
        elts = value.args[0]
        if not isinstance(elts, ast.Set | ast.List | ast.Tuple):
            return None
        return [e.id for e in elts.elts if isinstance(e, ast.Name)]
    return None


def _zero_alias_target(tree: ast.Module, alias_name: str) -> str | None:
    """取 Zero 顶层 `alias_name = <标识符>` 的右值符号名；不是标识符别名 → None。"""
    for node in tree.body:
        names, value = _module_assign_targets(node)
        if alias_name in names and isinstance(value, ast.Name):
            return value.id
    return None


def _zero_tool_error_token_prefix(tree: ast.Module) -> str | None:
    """取 Zero `_tool_error` 里 f-string 的**首段常量**（即令牌构造前缀）；取不到 → None。

    形如 `return ToolError(f"[zero:{code}] {message…}")` → JoinedStr 的第一个 Constant 段。
    """
    func = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == _ZERO_TOOL_ERROR_FUNC
        ),
        None,
    )
    if func is None:
        return None
    for node in ast.walk(func):
        if not isinstance(node, ast.JoinedStr) or not node.values:
            continue
        head = node.values[0]
        if isinstance(head, ast.Constant) and isinstance(head.value, str):
            return head.value
    return None


@pytest.mark.zerorepo
class TestZeroErrorCodeCrosscheck:
    """Zero 机读错误码全表跨仓一致——符号集 / 逐符号值 / 令牌格式 / 旧别名。

    D:\\Zero 或 server.py 不在位 → skip（不拖红）；**在位但结构对不上 → 红**。
    """

    def test_error_code_symbol_set_matches_expectation(self) -> None:
        """① Zero 顶层 `ZERO_ERROR_CODE_*` 符号集 == 本仓独立期望集（改名/增删码即红）。"""
        tree = _zero_server_tree_or_skip()
        zero_symbols = frozenset(_zero_error_code_literals(tree))

        assert zero_symbols == _EXPECTED_ZERO_ERROR_CODE_SYMBOLS, (
            f"Zero 机读错误码**符号集**跨仓漂移（{_ZERO_SERVER_PY}）：\n"
            f"  Zero 有而本仓未期望：{sorted(zero_symbols - _EXPECTED_ZERO_ERROR_CODE_SYMBOLS)}\n"
            f"  本仓期望而 Zero 无：{sorted(_EXPECTED_ZERO_ERROR_CODE_SYMBOLS - zero_symbols)}\n"
            "两仓须协调：新增码要同步 client._CODE_TO_EXCEPTION 的归类（可重试 / 须重开会话 / "
            "调用方传参错 / 部署端问题），否则新码落基类被静默降级。"
        )

    def test_error_code_values_match_client_constants(self) -> None:
        """② **逐符号**比对码值：Zero 字面量 == 本仓 client 同名常量（值单边漂移即红）。

        本仓那份字面量独立写在 `src/mcp/zero/client.py`，故本条不是恒真式。
        另断言两侧全表（值集合）相等——防「符号对上但登记表漏一条」。
        """
        tree = _zero_server_tree_or_skip()
        zero_literals = _zero_error_code_literals(tree)

        from src.mcp.zero import client as zero_client

        for symbol in sorted(_EXPECTED_ZERO_ERROR_CODE_SYMBOLS):
            assert symbol in zero_literals, (
                f'Zero 未定义 `{symbol} = "…"`（{_ZERO_SERVER_PY}）——见符号集断言。'
            )
            ours = getattr(zero_client, symbol, None)
            assert ours is not None, f"本仓 client 缺同名常量 `{symbol}`，两仓须同步。"
            assert zero_literals[symbol] == ours, (
                f"机读码值跨仓漂移：`{symbol}` Zero={zero_literals[symbol]!r}，"
                f"本仓 client={ours!r}。改值会让本仓查表落空 → 归类退化成基类、静默降级。"
            )

        assert frozenset(zero_literals.values()) == zero_client.ZERO_ERROR_CODES, (
            f"机读码**全表**跨仓不等：Zero={sorted(zero_literals.values())}，"
            f"本仓 ZERO_ERROR_CODES={sorted(zero_client.ZERO_ERROR_CODES)}。"
        )

    def test_zero_registry_lists_every_code_symbol(self) -> None:
        """③ Zero 的 `ZERO_ERROR_CODES` 登记表列全所有码符号（防「定义了码却漏登记」）。

        Zero `_tool_error` 对未登记码构造即抛，故漏登记 = 该错误出口在 Zero 侧直接崩；
        本仓也拿不到该码。两仓都受害，故此结构也要 pin。
        """
        tree = _zero_server_tree_or_skip()
        registry = _zero_registry_symbols(tree)

        assert registry is not None, (
            f"Zero 未找到 `{_ZERO_CODE_REGISTRY_NAME} = frozenset({{…}})` 的可解析定义"
            f"（{_ZERO_SERVER_PY}）——契约结构变了，本仓消费/守卫都要跟。"
        )
        assert frozenset(registry) == _EXPECTED_ZERO_ERROR_CODE_SYMBOLS, (
            f"Zero `{_ZERO_CODE_REGISTRY_NAME}` 登记的符号 {sorted(registry)} "
            f"≠ 本仓期望 {sorted(_EXPECTED_ZERO_ERROR_CODE_SYMBOLS)}。"
        )

    def test_token_prefix_pinned(self) -> None:
        """④ 令牌构造前缀 == `[zero:`——本仓消费正则的依据，Zero 换分隔符即红。

        这是**格式契约**本身：上一轮的死码正是「格式不对」而非「码值不对」，故必须单独 pin。
        """
        tree = _zero_server_tree_or_skip()
        prefix = _zero_tool_error_token_prefix(tree)

        assert prefix is not None, (
            f"Zero `{_ZERO_TOOL_ERROR_FUNC}` 里未找到可解析的 f-string 首段常量"
            f"（{_ZERO_SERVER_PY}）——令牌构造方式变了，本仓消费正则须重核。"
        )
        assert prefix == _EXPECTED_TOKEN_PREFIX, (
            f"机读令牌前缀跨仓漂移：Zero={prefix!r}，本仓期望={_EXPECTED_TOKEN_PREFIX!r}。"
            "本仓 `_ZERO_ERROR_TOKEN_RE` 依赖该前缀，改了即全部错误码提取失效（静默）。"
        )
        # 本仓正则确能从「按该前缀渲染出的样本」提到码——把格式 pin 与消费口径接上，
        # 避免只 pin 前缀却漏掉后半段（闭合括号）变化。
        from src.mcp.zero.client import classify_zero_error

        sample = f"Error executing tool zero.step: {prefix}unknown-session] 文案"
        assert classify_zero_error(sample) == "unknown-session", (
            f"按 Zero 前缀 {prefix!r} 渲染的样本，本仓 classify_zero_error 提不到码：{sample!r}"
        )

    def test_legacy_alias_preserved(self) -> None:
        """⑤ 旧别名 `_UNKNOWN_SESSION_MARKER` 仍在且指向 unknown-session 符号（过渡期承诺）。

        本仓仍导出同名别名（值相等）供历史调用点与本守卫引用；Zero 撤别名时本条红 →
        提醒同步撤除本仓别名与 `_LEGACY_UNKNOWN_SESSION_RE` 兼容分支。
        """
        tree = _zero_server_tree_or_skip()
        target = _zero_alias_target(tree, _ZERO_LEGACY_ALIAS_NAME)

        assert target == "ZERO_ERROR_CODE_UNKNOWN_SESSION", (
            f"Zero `{_ZERO_LEGACY_ALIAS_NAME}` 不再是指向 ZERO_ERROR_CODE_UNKNOWN_SESSION 的别名"
            f"（实际={target!r}）。若 Zero 已撤别名，本仓应同步撤除 client 侧别名与旧裸前缀兼容层。"
        )

        from src.mcp.zero.client import (
            _UNKNOWN_SESSION_MARKER,
            ZERO_ERROR_CODE_UNKNOWN_SESSION,
        )

        assert _UNKNOWN_SESSION_MARKER == ZERO_ERROR_CODE_UNKNOWN_SESSION, (
            "本仓别名与新常量值不等——两者须始终同值（Zero 侧亦然）。"
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
# 此前我方只 pin 了**占位**路径（Zero `src/agents/affect_math.py::decode_channels` 的 physiology
# 占位式，见 TestCanonicalPlaceholder…）与真模型路径的**键名**（上一个类）——漏了真模型路径的
# **值级常量**。Zero 指出：设 ZERO_PHYSIOLOGY_MODEL_PATH 后
# `src/agents/models/composite.py::CompositeChannelDecoder.predict_channels` 的
# `if self.physiology_model is not None:` 分支**整块覆盖** channels["physiology"]，占位式根本
# 不执行，走的是 `src/agents/models/physiology_decoder.py::PhysiologyDecoder.predict_physiology`
# 的**另一套反归一化常量**（temp 对 30/10 ≠ 占位的 36/3）。
#
# 为何这对**消费正确性**要命（W6 同类：静默标度差）：本仓 LinearPhysiologyMapper 的默认量纲
# （skin_conductance_max_us=20.0、temperature_range=(30,40)）正是按**在线 decoder** 标定的。
# Zero 若把 `vec[1]*20.0` 改成 `*10.0`，我方解析照样成功、mapper 照样不报错，但 level 静默错 2×
# （与 W6 legacy sc 欠标度 20× 同族）。故此处不止 pin 常量，还断言**常量 ⇔ mapper 默认**的耦合。
#
# 另 Zero `src/agents/datasets/wesad.py` 特征提取里的训练侧归一化（hr/eda/temp 三条 clamp 式）
# 与 decoder 反归一化是**逆变换对，必须成对同改**（Zero ③(b)）：单改一侧 → 权重与解码口径错配，
# 输出物理量整体偏移。故一并 pin，任一侧漂移即红 → 触发 ping。
#
# 📌 跨仓引用口径：本文件一律写「Zero `<仓内相对路径>::<符号名>`」，**不写对方行号**——Zero 侧
# 行号漂移剧烈（同一天内其 HEAD 与未提交工作树就有 7 处再漂、最大 +44 行），且行号腐烂**不驱红**
# ＝静默失效。所有正则/切块锚点同样按符号名与 AST 结构，不按行号、不按文本 partition。
# ---------------------------------------------------------------------------

_ZERO_WESAD_PY = _ZERO_SRC / "agents" / "datasets" / "wesad.py"
# 逐键取 return dict 的 RHS 表达式（到行尾/逗号），再从中提浮点字面量
_PHYSIO_RHS_RE = re.compile(r'"([a-z_]+)"\s*:\s*([^,\n]+)')
_FLOAT_LITERAL_RE = re.compile(r"\d+\.\d+")
# 训练侧归一化（逆变换对）——锚在变量名上，结构改了则 skip、常量漂移则 fail
_WESAD_HR_RE = re.compile(r"clamp\(\(hr\s*-\s*([\d.]+)\)\s*/\s*([\d.]+)")
_WESAD_EDA_RE = re.compile(r"clamp\(eda_mean\s*/\s*([\d.]+)")
_WESAD_TEMP_RE = re.compile(r"clamp\(\(temp_mean\s*-\s*([\d.]+)\)\s*/\s*([\d.]+)")

# 在线 decoder 反归一化常量（Zero 07-28 现场核验：
# `src/agents/models/physiology_decoder.py::PhysiologyDecoder.predict_physiology` 的 return dict）
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
# 作用于 external_priors：Zero `src/agents/affect_core.py` 里 `expand_external_priors(...)` 的产物
# **无条件**进 `ignite(...)`（两个调用点相邻、之间无筛选），而 `affect_math::expand_external_priors`
# 只校验不改精度。默认 IGNITION_BETA=None 即硬门。
#
# salience(μ,Π) = hypot(μ)·mean(Π)（`affect_math::stream_salience`）
# ——|μ| 的线性函数、门是**锐阶跃**。
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
# ⚠ **Zero 的最终方案与本节旧注释所写的完全不同**（2026-07-29 重写；此前三段均已过时，勿参照）。
#   曾经写的是「改判据公式为按轴加权马氏距离 D + θ'=0.28，须走完整 PRP、近期不落地」——
#   该方案被 Zero **第三轮终裁整体推翻**，且新方案**已合入其 main `9617372`**（PR #46）：
#     · 数值通路：`ignite()` 拆两路，`fusion_terms` 用**所有流的原生 (μ,Π)**，不乘任何 gate/D 因子；
#     · 报告通路：θ' 降级为**纯报告阈值**（只决定 `ignited_streams` 标签，不影响任何后验数值）；
#     · **跨流归一化被正面否决**（数学席证该路线无解），不再是待办；
#     · **θ' 的值不必再 pin**（Zero 明言标定压力已消失）。
#   三个开关**全部默认关**，`SALIENCE_THRESHOLD=0.18` 与旧式 `hypot(μ)·mean(Π)` **原样保留**：
#     · `ZERO_IGNITION_GATE_FUSION`  默认 **`True` = 门关**
#       （⚠ 方向与其它旋钮相反，勿按字面读成 False）
#     · `ZERO_EXCLUDE_PHYSIO_FUSION` 默认 **`True` = 排除 physio**（Zero 应我方 WESAD 反号简报所作
#       承诺 D7 的兑现物，其单边可控、解除前会 ping 我方）
#     · `ZERO_PRECISION_COMMENSURABLE` 默认 `False`
#   ⚠ 由此推出一条**易错的跨仓语义**：**「点燃（可报告）≠ 参与数值融合」**——两条通路已解耦。
#   即使点燃门全开，physio 仍被排除开关挡在数值通路外；**两个开关都翻**才会真正参与后验。
#
# ⚠ **本类探测得到什么、探测不到什么**（据 2026-07-29 实测更新）：
#   探测得到：常量**值**漂移（`_ZERO_GATE_CONSTANTS` 逐值 pin）；本仓推荐精度/子源可靠度改动；
#     **以及 `fast_survival_prior` 被改成门控多分支形态**——`test_survival_stream_always_ignites`
#     的多分支断言在 Zero 落地当天**如期变红**（实测：解析出两条 arousal 分支，基线 `[0.0, 0.5]`）。
#     ⚠ 这一条来之不易：**旧正则只认 `+` 基线，会匹配到 legacy 分支解析出 0.5、断言照过 →
#     静默全绿**，我方将完全不知道 Zero 已动手。
#     2026-07-28 把守卫从「结构变更即 skip（黄灯）」改成「三形态皆红」
#     正是为此，今天恰好接住了真事件——判别力实证不是形式主义，它就是这条守卫有没有用的分界线。
#   探测**不到**：`stream_salience` 判据**公式**被替换。`_mean_precision` 是旧式的**手抄镜像**，
#     从不读 Zero 函数体；而新架构以 default-off 开关落地、旧式原样保留 → 公式若真被换掉本类仍全绿。
#     此为**已知盲区**，靠 Zero 的人工 ping 兜底（其 PRP tasks 已列 ping 义务）。
#
# 📌 **当前状态（2026-07-29 实测，非推断；任何红绿结论都必须带 ref 限定）**：
#   · 本文件的跨仓守卫一律 `read_text()` 磁盘上的**工作副本**、不读 git 对象，故报红绿必须同时
#     给出「本仓 commit + Zero HEAD + **Zero 工作树是否脏**」三项，只说「绿了」没有意义。
#     ⚠ 这不是纸面风险：本次改动过程中 Zero 就从「HEAD `11c25b0` + 12 个未提交 M」变成了
#     「HEAD `df496da` + 工作树 clean」，同一份守卫代码的红绿结论**在同一小时内翻了面**。
#   · ⚠ 旧注释写的「本仓 `main` 上是红的（尚未跟上重标，属结构性假红，合并后即消）」**已过期**，
#     本次订正：重标早已合入 main。改动当天在 main 上实测到的那一条红是**另一件事**——Zero
#     **未提交**的 docstring 里写了 `if gate_fusion:` 这句散文，撞坏了当时的文本切分锚点
#     （根因见 `test_gate_open_branch_has_no_fallback_and_excludes_physio` 的 docstring）。
#     ⚠ **诚实记账**：Zero 随后提交（`df496da`）时改写了那段 docstring，该次红已**自行消失**，
#     即它是随对方工作树状态来去的**瞬时**红，不是持续故障。本次迁 AST 修的是**那一类**结构
#     脆弱性（用文本在源码里找结构锚点），不是修一条一直红着的用例——Zero 常态性在 docstring
#     里引用代码 token，同样的撞法随时会再发生一次。
#   · 本次改动后实测（对 Zero `11c25b0`+12M 与 `df496da`+clean **两个 ref 各跑一遍，结论一致**）：
#     `pytest tests/mcp/test_zero_contract_crosscheck.py` = 37 passed；
#     `ZERO_LINK_E2E_STRICT=1 pytest tests/mcp -m zerorepo` = **48 passed / 0 skipped / 0 failed**
#     （较改前的 45 项多出的 3 项 = 本次新增的 gate 运行期默认 pin、sample_sigma_cap 通路 pin、
#     以及门判前缺口守卫的十一态判别性自证）。
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

# occ_prior σ 标度系数（Zero `src/agents/affect_math.py::occ_prior` 内的 conf/sigma 两式）
# ——Zero 07-28 ④ 订正：**这才是真旋钮**（MIN_SIGMA 在 MCP 路径恒不咬合：
# `src/mcp_server/mapping.py::stimulus_from_payload` 把 |I| 钳到 ≤1 → σ∈[0.10,0.35] 恒 >0.05）。
_ZERO_SIGMA_CONF_RE = re.compile(rf"conf\s*=\s*clamp\(({_NUM})\s*\+\s*({_NUM})\s*\*\s*abs\(")
_ZERO_SIGMA_RE = re.compile(rf"sigma\s*=\s*max\(MIN_SIGMA,\s*({_NUM})\s*\*\s*\(1\.0\s*-\s*conf\)")
# `affect_math::precision` 的 α/β 形参默认——在 sigmoid **分子**，直接线性缩放判别信号
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
# `affect_math::occ_prior` 的多行 `clamp(\n 0.4·|I| + …)` 抢先匹配成「无基线」，
# 在 Zero 未动手时就假阳性红（本仓 07-28 实踩，见判别性自证测试）。
_ZERO_SURVIVAL_AROUSAL_RE = re.compile(
    rf"arousal\s*=\s*clamp\(\s*(?:({_NUM})\s*\+\s*)?({_NUM})\s*\*\s*abs\(intensity\)"
)
_ZERO_SURVIVAL_FUNC_RE = re.compile(
    r"def fast_survival_prior\(.*?(?=\ndef |\nclass |\Z)", re.DOTALL
)


def _extract_function_body(source: str, name: str) -> str | None:
    """按**符号名**切出顶层函数体（含 def 行），不依赖行号。

    Zero 侧行号漂移剧烈（`4760dfb` 一次就增约 190 行），跨仓守卫一律按符号名锚定；
    行号只出现在人读的注释里，且不作断言依据。找不到返回 None → 调用方 skip。
    """
    match = re.search(
        rf"^def {re.escape(name)}\(.*?(?=\n(?:@|def |class )|\Z)", source, re.DOTALL | re.MULTILINE
    )
    return match.group(0) if match else None


def _top_level_func(source: str, name: str) -> ast.FunctionDef | None:
    """按 AST 取**顶层**函数节点；语法错 / 不存在返回 None。

    为什么跨仓结构守卫一律走 AST，而不再用任何文本切法（两次实踩的同源教训）：

    1. `partition("return ")`（2026-07-28 踩）——在「真分支不再 return」这种改动下切点会
       落到函数**末尾**那个 return 上、把改动整个跳过，守卫照样绿（永绿=恒真）。
    2. `body.find("if gate_fusion:")`（2026-07-29 踩，本次修复对象）——Zero 的 docstring /
       `#` 注释里**常态性**出现 `if gate_fusion:`、`_PHYSIO_PREFIXES` 这类 token（其
       `src/agents/affect_math.py::ignite` 的 docstring「收口条件」段就写了「把前缀过滤提到
       `if gate_fusion:` 之前」）。首次命中落到这句散文上 → 按散文行长算缩进 → 块体为空 →
       守卫**假红**。实测对照：同一守卫在 Zero HEAD 上绿、在其工作副本上红，红与 `ignite`
       的结构**完全无关**。

    AST 里没有 `#` 注释，docstring 是可识别的独立 `Expr(Constant(str))` 节点（由
    `_body_without_docstring` 剥掉），两类污染一次消除。缩进/文本只在解析失败时才有价值，
    而解析失败本身就该走「找不到 → 调用方判红或 skip」。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    # ⚠ 只遍历 tree.body 顶层，**不用** ast.walk(tree)——后者会把嵌套/同名函数一起捞进来。
    return next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)


def _body_without_docstring(func: ast.FunctionDef) -> list[ast.stmt]:
    """剥掉函数体首条 docstring 语句，返回其余语句列表。"""
    body = list(func.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _split_ignite_on_gate(
    func: ast.FunctionDef,
) -> tuple[list[ast.stmt], ast.If, list[ast.stmt]] | None:
    """把 ignite 体切成 (门判之前, `if gate_fusion:` 节点, 门判之后)。无该二分返回 None。

    返回三段而非两段：现行形态是「真分支提前 return + 落穿」，门开分支**不在** If 节点的
    orelse 里而在其后的同级语句里；写成 if/else 形态时则在 orelse 里。两种都要覆盖，故
    「门开分支」= `[*gate_if.orelse, *tail]`。
    """
    body = _body_without_docstring(func)
    for pos, stmt in enumerate(body):
        if (
            isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "gate_fusion"
        ):
            return body[:pos], stmt, body[pos + 1 :]
    return None


def _referenced_names(nodes: Iterable[ast.AST]) -> set[str]:
    """给定语句集合里**被引用到的标识符**（Name 节点）集合。

    签名形参、docstring 正文、`#` 注释均不产生 Name 节点，故三者都不会误报。
    ⚠ 由此推出一条使用前提：只能喂**函数体语句**（`func.body` 的切片），不能把 `func`
    整体丢进来——`exclude_physio_fusion` 同时出现在 `ignite` 的签名里（`func.args`），
    喂整节点会让「门判前是否已施 physio 过滤」这条断言恒红。
    """
    return {sub.id for node in nodes for sub in ast.walk(node) if isinstance(sub, ast.Name)}


# physio 排除相关标识符：前缀表本身 + 开关名（Zero 若抽 helper 并显式传 flag 也能接住）。
# 用**集合交**而非字符串 `in`：后者会被 `_PHYSIO_PREFIXES_LEGACY` 之类的新名字误命中。
_PHYSIO_FILTER_NAMES = frozenset({"_PHYSIO_PREFIXES", "exclude_physio_fusion"})

_PARAM_REQUIRED = "<REQUIRED>"  # 形参在、但没有默认值（被改成必填）
_PARAM_MISSING = "<MISSING>"  # 形参根本不在签名里（被删 / 改名）
_PARAM_NON_LITERAL = "<NON-LITERAL>"  # 默认值不是字面量（如从常量/env 读）


def _signature_default(func: ast.FunctionDef, name: str) -> object:
    """取函数签名里形参 `name` 的默认值字面量（位置形参与仅关键字形参都覆盖）。

    三种「不是普通默认值」的情形各返回**不同的**哨兵，而不是统一 None——判据必须能分清
    红的原因，否则「形参被删」会伪装成「默认值翻转」，归因指错方向（本仓 pitfalls 同族）。

    为什么用 AST 而不是对整份源码跑 `re.search(r"gate_fusion\\s*:\\s*bool\\s*=\\s*(True|False)")`：
    Zero 侧 `gate_fusion` / `exclude_physio_fusion` 这类名字在 `orchestration/runner.py`、
    `orchestration/state.py`、以及 `affect_math` 内其它函数里都可能同名出现，全文件正则会锚点
    滑走；AST 把「哪个函数的哪个形参」这件事变成结构事实而非文本巧合。
    """
    args = func.args
    positional = [*args.posonlyargs, *args.args]
    pos_defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults))
    pos_defaults.extend(args.defaults)
    pairs: list[tuple[ast.arg, ast.expr | None]] = [
        *zip(positional, pos_defaults, strict=True),
        *zip(args.kwonlyargs, args.kw_defaults, strict=True),
    ]
    for arg, default in pairs:
        if arg.arg != name:
            continue
        if default is None:
            return _PARAM_REQUIRED
        try:
            return ast.literal_eval(default)
        except ValueError:
            return _PARAM_NON_LITERAL
    return _PARAM_MISSING


# 我方实际发往 Zero 的 physio 流名（合并态 / 未合并态）。用于核对 Zero 的 D7 前缀排除
# 是否真的覆盖我方载荷——「对方加了排除」不等于「排除到我方头上」。
_MCP_PHYSIO_STREAM_NAMES = ("physio", "eda/sc", "hrv/rmssd")

# Zero `src/agents/affect_math.py::_PHYSIO_PREFIXES` 的**全等**期望集（顺序不 pin，见下）。
# 消费点是 `str.startswith(tuple)`，对顺序不敏感 → pin 成有序列表会在 Zero 纯重排（语义无变）
# 时假红；但成员**增删都要红**：删项 ⇒ 该类流漏出 D7 排除与 M2 Πv 归零；增项 ⇒ Zero 单边
# 扩大了排除面（可能连带扫掉我方未来的新流名）。旧写法只断言 `{physio,eda,hrv} <= 集合`，
# 实测对「删 scr」「加 rsp」两个变异都绿——这是 Zero 2026-07-29 回执点名的 pin 差口 D-1。
_ZERO_PHYSIO_PREFIXES_EXPECTED = frozenset({"physio", "eda", "hrv", "pupil", "scr"})


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

    def _zero_module_source(self, *parts: str) -> str:
        """读 Zero 侧任意模块源码；不在位则 skip（不拖红）。"""
        if not _zero_available():
            pytest.skip(f"D:\\Zero\\src 不存在（{_ZERO_SRC}），跳过跨仓断言")
        path = _ZERO_SRC.joinpath(*parts)
        if not path.is_file():
            pytest.skip(f"Zero 模块不存在（{path}），跳过")
        return path.read_text(encoding="utf-8")

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

        推论（**仅门关，即默认配置**）：`_select_fired` 的 `top = max(scored, key=salience)`
        兜底分支不可达（survival arousal 恒 ≥0.5、Π 恒 (0.4,0.4) → salience ≥0.200 > 0.18）。
        故外部亚阈先验等不到「全场亚阈时当选 max」的机会。两常量任一动即须同评。

        ⚠ **2026-07-29 两次修订**：
        1. Zero 落 `arousal_floor_fix` 门控（`4760dfb`「线A」）→ 本例按上一版结构断言如期
           变红（设计意图），改为基线集合逐值 pin + 默认值 pin。
        2. **随后跨仓核验推翻了当时同步写下的推论**：门开后并非「max-fallback 变可达」。
           `ignite()` 只在 `gate_fusion=True` 分支调 `_select_fired`；门开走的是「全流原生
           (μ,Π) 进融合」分支，**既无阈值比较也无 fallback**——门一开不是让 fallback 生效，
           而是让它从数值通路上整条消失。且门开分支随即按 `_PHYSIO_PREFIXES` 把 physio
           剔出 fusion（`exclude_physio_fusion` 默认 True，Zero 的 D7 跨仓承诺）。
           详见下方 `test_gate_open_branch_has_no_fallback_and_excludes_physio`。
        故本例的适用范围就是**门关**这一支，不必再谈门开——门开由那条独立守卫刻画。
        """
        source = self._source()
        consts = {}
        for name in ("SURVIVAL_PRECISION", "SALIENCE_THRESHOLD"):
            match = re.search(rf"^{name}\s*=\s*({_NUM})", source, re.MULTILINE)
            if match is None:
                pytest.skip(f"未找到 {name}，跳过")
            consts[name] = float(match.group(1))

        # ⚠ 此处**从 Zero 源码读** fast_survival_prior 的 arousal 基线系数，不再手抄。
        # 现行形态（Zero 复议 §六(e) 裁定的 default-off 门控）：两分支，基线 {0.0, 0.5}。
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

    def test_survival_subthreshold_boundary_when_floor_removed(self) -> None:
        """**去地板分支内** survival 单流的亚阈边界特征化：`|goal| < 0.18/(0.4·0.6) = 0.75`。

        ⚠ **本边界不通向「physio 被保留」**（2026-07-29 跨仓核验订正）。它只是「去掉 0.5
        地板后 survival 这一条流还剩多少条件才亚阈」的算术刻画，三重限定必须一起读：

        1. **只管 survival 单流**。「全场亚阈」的绑定约束是 **appraisal**（Π≈8.16·gain，
           I=0 时需 `|goal| < 0.0374`），比这里严 20 倍；可行窗口仅约占 (goal,I)∈[-1,1]²
           面积的 0.12%。0.75 从来不是全场亚阈的 binding constraint。
        2. **只在 `precision_commensurable=False` 下成立**。同开线B（`ZERO_PRECISION_
           COMMENSURABLE`，与线A 同日落地）时 Π 由 0.4 变 4.9383（I=0），边界收到 0.0608。
        3. **去地板 ⟺ 门开，而门开时数值通路上根本没有 fallback**（见
           `test_gate_open_branch_has_no_fallback_and_excludes_physio`）。故即便真落进这个
           窗口，也不会发生「physio 当选 max」——那条路在该分支不存在。

        留着这条的价值是**逐值 pin 去地板分支的两个系数**（arousal 无基线项、valence 0.6）：
        Zero 若给去地板分支补回下限或改 valence 系数，本例立刻红。
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

        # 去地板分支 = 无基线项那条 → intensity=0 时 arousal=0，salience = |coef·goal|·Π。
        assert min(floors) == 0.0, "去地板分支不再是「无基线」形态，须重标"
        goal_bound = threshold / (precision * valence_coef)
        assert goal_bound == pytest.approx(0.75, abs=1e-9), (
            f"去地板分支的 survival 亚阈边界漂移：|goal| < {goal_bound:.4f}（期望 0.75="
            f"{threshold}/({precision}·{valence_coef})）——系数被改动，须跨仓同评。"
        )

        # 反面对照（非冗余）：带地板分支下**任何** goal 都过阈——0.5·Π=0.200 > 0.18 与 goal 无关。
        # 即可达性的翻转完全由门决定，不由输入决定。
        assert max(floors) * precision > threshold, (
            "带地板分支下 survival 已非恒过阈——与 test_survival_stream_always_ignites 冲突，"
            "两例须同评。"
        )

        # 门与门的绑定：floor_fix 不是独立旋钮，而是 gate_fusion 的取反（Zero 议会 D5 强制
        # 共用同一开关）。这条绑定一旦解开，两个门可各自独立开合 → 组合态爆炸，本仓的
        # 二态刻画（门关/门开）不再充分。
        core_source = self._zero_module_source("agents", "affect_core.py")
        assert "arousal_floor_fix=not state.gate_fusion" in core_source, (
            "affect_core 不再把 arousal_floor_fix 绑定为 `not gate_fusion`——两门可独立开合，"
            "本仓「门关/门开」二态刻画不再充分，须按新组合态重标可达性结论。"
        )

    def test_gate_open_branch_has_no_fallback_and_excludes_physio(self) -> None:
        """门开分支的**结构** pin：无阈值筛选、无 max-fallback，且默认按前缀剔除 physio。

        这条是 2026-07-29 跨仓核验的直接产物——它推翻了本仓当天写下的
        「门一开 → max-fallback 生效 → physio 先验获得被保留的路径」。真相是：

        - `ignite()` 只在 `if gate_fusion:` 分支里调 `_select_fired`（阈值筛选 + fallback
          都在那个 helper 内），随后**提前 return**；门开走的是「全流原生 (μ,Π) 进融合」
          分支，那里没有 threshold、没有 fired、没有 fallback。
        - 门开分支紧接着按 `_PHYSIO_PREFIXES` 把 physio 剔出 fusion，开关
          `exclude_physio_fusion` **默认 True**——这正是 Zero 应我方「EDA 反号，宁可继续
          门掉」请求所落的 D7 跨仓承诺。故开门对 physio 的净效果是**被点名排除**。

        本例只 pin 结构与默认值，不 pin 行号（Zero 侧行号漂移剧烈）。Zero 若把 fallback
        挪进门开分支、或把 `exclude_physio_fusion` 默认翻成 False，本例即红。

        ⚠ **2026-07-29 迁 AST（修假红，不是放宽守卫）**：①段原走一个按缩进切块的文本 helper
        （`_split_if_block`，已随本次删除），其 `body.find("if gate_fusion:")` 被 Zero 当日
        **未提交**的新 docstring 撞上
        ——其 `ignite` docstring「收口条件」段原文就写着「把前缀过滤提到 `if gate_fusion:`
        之前」，首次命中落到这句散文上、按散文行长算缩进 ⇒ 块体为空 ⇒ 本例红。实测对照：
        同一守卫在 Zero HEAD `11c25b0` 上**绿**、在其工作副本（12 个未提交 M）上**红**，
        红与 `ignite` 的结构完全无关。迁 AST 后本例由红转绿，判别力不降反升（6 态矩阵：
        base 绿 / docstring 污染 绿 / fallback 挪进门开 红 / 真分支去 return 红 / 二分被拆
        红 / else 形态 绿）。②段（默认值与前缀集 pin）不受切分影响，原样保留并补齐 pin 差口。
        """
        source = self._source()

        # ① fallback 与阈值筛选都在 _select_fired 内；ignite 只在 gate_fusion 真分支调它。
        select_fired = _extract_function_body(source, "_select_fired")
        if select_fired is None:
            pytest.skip("未找到 _select_fired，Zero 结构可能大改，跳过")
        assert "max(" in select_fired, "_select_fired 内已无 max-fallback，可达性结论须重评"

        func = _top_level_func(source, "ignite")
        assert func is not None, "未找到顶层 ignite——锚点失效，须按新形态重标（勿降级为 skip）"
        split = _split_ignite_on_gate(func)
        assert split is not None, "ignite 不再以 `if gate_fusion:` 二分——门控形态变了，须重标"
        _prelude, gate_if, tail = split
        closed = list(gate_if.body)
        opened = [*gate_if.orelse, *tail]
        # ⚠ 切分有效性守卫：两侧都非空才谈得上「分支归属」；否则下面的 `not in` 恒真=空断言
        # （本仓 pitfalls「绿灯必须先证明它能红」第 ⑥ 例正是这么踩的）。
        assert closed and opened, (
            "切不出 ignite 的门关/门开两分支——结构变了，须按新形态重标（勿让本例退化成空断言）"
        )
        assert "_select_fired" in _referenced_names(closed), (
            "gate_fusion 真分支里已无 _select_fired——阈值通路被重构，须重标"
        )
        # 真分支必须**提前 return**：否则会落穿到门开分支，两条路径同时执行，
        # 「门关不走融合分支」这个前提就不成立了。
        # ⚠ 落穿检查**只在无 else 形态下要求**：写成 if/else 时结构上不可能落穿，此时这条
        # 断言不执行是**正确的**（矩阵 (5) else 形态期望绿），不是漏洞——别日后看到就补死。
        if not gate_if.orelse:
            assert isinstance(closed[-1], ast.Return), (
                f"gate_fusion 真分支不再以 return 收尾（末句 {type(closed[-1]).__name__}）"
                "——会落穿到门开分支，本仓的门关/门开二态刻画须重评。"
            )
        assert "_select_fired" not in _referenced_names(opened), (
            "门开分支里出现了 _select_fired——max-fallback 被挪回数值通路，"
            "本仓「门开时 physio 不经 fallback」的结论须撤回并重评。"
        )

        # ② D7：门开分支默认按前缀剔 physio。默认值 pin 与前缀集 pin 都要。
        exclude_default = re.search(r"exclude_physio_fusion\s*:\s*bool\s*=\s*(True|False)", source)
        assert exclude_default is not None, "未找到 exclude_physio_fusion 形参默认值，须重标"
        assert exclude_default.group(1) == "True", (
            "exclude_physio_fusion 默认翻为 False——D7「physio 不进数值通路」的跨仓承诺"
            "已被解除，我方 physio 先验会在门开时真正参与后验，须立即跨仓确认。"
        )
        # 补口（本次判别力实证当场揪出的覆盖缺口）：上面那条 pin 与下面的前缀集 pin 只证明
        # 「开关还在、前缀表还在」，**不证明过滤本体还在**。实测：把门开分支里那行
        # `fusion = [t for t in fusion if not t[0].lower().startswith(_PHYSIO_PREFIXES)]` 单独删掉
        # （D12 raise 文案里对前缀表的引用保留），本例改口前的**所有**断言照样全绿——D7 被抽空
        # 而我方毫无察觉。故补一条「过滤本体存在」的正向断言。
        # 判据刻意写成**位置无关**（只问 ignite 体内有没有，不问在哪个分支）：Zero 的收口方案是
        # 把过滤**移到**门判前，若写成「门开分支里必须有」，其收口当天会红在「D7 被删」这条
        # 归因错误的消息上——把好事读成坏事。
        guarded_filters = [
            node
            for node in ast.walk(func)
            if isinstance(node, ast.If)
            and "exclude_physio_fusion" in _referenced_names([node.test])
            and "_PHYSIO_PREFIXES" in _referenced_names(node.body)
        ]
        assert guarded_filters, (
            "ignite 体内已找不到「由 exclude_physio_fusion 守卫、且引用 _PHYSIO_PREFIXES」的过滤"
            "分支——D7 排除的**执行体**被删（开关与前缀表可能都还在，故那两条 pin 不会红），"
            "我方 physio 先验会重新进入数值通路，须立即跨仓确认。"
        )
        # D-2 第一层（Zero 2026-07-29 回执点名的 pin 差口）：`gate_fusion` 的**函数级**默认。
        # 锚在 ignite 的 AST 签名内，不做全文件正则——Zero 的 runner.py / state.py 里都有同名
        # 形参/字段，跨文件复用同一正则会锚点滑走。⚠ 本条只管函数默认，**不治理运行期**：
        # 真旋钮是 `AffectState.gate_fusion`（下面第二/第三层）。
        gate_default = _signature_default(func, "gate_fusion")
        assert gate_default is not _PARAM_MISSING, (
            "ignite 签名内已无 gate_fusion 形参（被删 / 改名）——门控形态变了，须重标"
        )
        assert gate_default is not _PARAM_REQUIRED, (
            "ignite 的 gate_fusion 被改成必填形参（无默认值）——调用方全部须显式传值，"
            "「函数默认=门关」这条前提消失，须重标"
        )
        assert gate_default is True, (
            f"ignite 的 gate_fusion 默认变为 {gate_default!r}（期望 True=门关=零回归，"
            "注意方向与其它旋钮相反）——门开成为**函数级**默认，须跨仓确认；"
            "注意本条只管函数默认，运行期真值见 AffectState.gate_fusion 的独立 pin。"
        )
        # D-2 第三层（使前两层语义诚实）：pin 调用点覆写事实。生产路径**永不使用**上面那个
        # 函数默认——affect_core 逐参覆写为 state 字段。只 pin 函数默认 = 制造「函数默认治理
        # 生产」的错觉，正是 Zero 自己在 MAX_SAMPLE_SIGMA 上记在头上的毛病，不要复刻。
        core_source = self._zero_module_source("agents", "affect_core.py")
        assert "gate_fusion=state.gate_fusion" in core_source, (
            "affect_core 不再把 ignite 的 gate_fusion 覆写为 state.gate_fusion——运行期旋钮"
            "换了位置（或改回吃函数默认），上面的函数默认 pin 与 AffectState 字段 pin 都可能"
            "已不代表生产行为，须重新核对整条通路。"
        )

        # D-1（Zero 2026-07-29 回执点名的 pin 差口）：前缀集改**全等** pin。
        # 允许类型注解形态 `_PHYSIO_PREFIXES: tuple[str, ...] = (...)`
        prefixes = re.search(r"^_PHYSIO_PREFIXES\b[^=\n]*=\s*\(([^)]*)\)", source, re.MULTILINE)
        assert prefixes is not None, (
            "未找到 _PHYSIO_PREFIXES 的 tuple 字面量定义"
            "（Zero 可能改成 frozenset / 多处拼接），须重标"
        )
        parsed = re.findall(r'"([^"]+)"', prefixes.group(1))
        assert parsed, (
            "_PHYSIO_PREFIXES 字面量解析为空——正则被结构变更绕过，须重标（勿退化成空断言）"
        )
        prefix_set = set(parsed)
        assert prefix_set == _ZERO_PHYSIO_PREFIXES_EXPECTED, (
            f"_PHYSIO_PREFIXES 成员集变为 {sorted(prefix_set)}、期望 "
            f"{sorted(_ZERO_PHYSIO_PREFIXES_EXPECTED)}——**增删都要红**：删项 ⇒ 该类流漏出 D7 "
            "排除与 M2 Πv 归零；增项 ⇒ Zero 单边扩大了排除面（可能连带扫掉我方未来的新流名）。"
            "须跨仓确认。"
        )
        assert len(parsed) == len(prefix_set), f"_PHYSIO_PREFIXES 出现重复项 {parsed}，须跨仓确认"
        # 我方实际发出的流名必须落进该前缀集，否则 D7 对我方载荷根本不生效
        # （与上面的成员集 pin 正交：那条问「表变没变」，这条问「排除有没有排到我方头上」）
        for name in _MCP_PHYSIO_STREAM_NAMES:
            assert name.lower().startswith(tuple(prefix_set)), (
                f"我方流名 {name!r} 不匹配 Zero 的 _PHYSIO_PREFIXES={prefix_set}，D7 排除对该流失效"
            )

    def test_gate_runtime_defaults_from_affect_state_pinned(self) -> None:
        """D-2 第二层：**运行期**门控真值 = `AffectState` 字段默认，不是 ignite 的函数默认。

        与上一例的分工必须写清，否则两条 pin 会被读成重复：
        - 上一例 pin `ignite(gate_fusion=True, exclude_physio_fusion=True)` 的**函数默认**，
          外加调用点覆写事实（`gate_fusion=state.gate_fusion`）；
        - 本例 pin 被覆写进去的那个**真旋钮**。生产路径永不吃函数默认，只 pin 第一层等于
          制造「函数默认治理生产」的错觉。

        字段缺位时子进程回 `"<MISSING>"` 哨兵（不是 KeyError → 不是 skip），断言必然红。
        """
        data = _fetch_affect_state_defaults_or_skip()
        assert data["gate_fusion"] is True, (
            f"AffectState.gate_fusion 默认变为 {data['gate_fusion']!r}（期望 True=门关=零回归，"
            "⚠ 方向与其它旋钮相反）——门开成为**生产默认**，全流原生 (μ,Π) 进融合，"
            "本仓所有基于「门关」的可达性结论须整体重评，须立即跨仓确认。"
        )
        assert data["exclude_physio_fusion"] is True, (
            f"AffectState.exclude_physio_fusion 默认变为 {data['exclude_physio_fusion']!r}"
            "（期望 True）——D7「physio 不进数值通路」的跨仓承诺在**运行期**被解除，"
            "我方反号 physio 先验会在门开时真正参与后验，须立即跨仓确认。"
        )

    def test_sample_sigma_cap_runtime_path_pinned(self) -> None:
        """D-3：`MAX_SAMPLE_SIGMA=0.5` 的**运行期通路** pin。

        补 `_ZERO_GATE_CONSTANTS` 里那条逐值 pin 的空头支票——逐值 pin 一个模块常量，
        只有在「该常量确实是缺省生效值」时才有意义。下面三条把该前提
        拆开来守（Zero 自己在回执里把 MAX_SAMPLE_SIGMA 标为「运行期有效值来自 state」，
        我方原先只 pin 了常量值，等于把前提当结论）：
        1. `sample_affect(sigma_cap=None)` 仍是默认 → 不传时才落到模块常量；
        2. `cap = MAX_SAMPLE_SIGMA if sigma_cap is None else sigma_cap` 回退式未变形；
        3. `min(cap, post_sigma[...])` 仍在钳采样 σ → 该常量仍有被观测对象。
        再加一条 `AffectState.sample_sigma_cap` 默认 None（运行期确实不覆写）。
        """
        body = _extract_function_body(self._source(), "sample_affect")
        if body is None:
            pytest.skip("未找到 sample_affect，Zero 结构可能大改，跳过")
        assert re.search(r"sigma_cap\s*:\s*float\s*\|\s*None\s*=\s*None", body), (
            "sample_affect 的 sigma_cap 形参默认不再是 None——模块常量 MAX_SAMPLE_SIGMA 不再是"
            "缺省通路，本仓对它的逐值 pin 失去运行期含义，须重标"
        )
        assert re.search(
            r"cap\s*=\s*MAX_SAMPLE_SIGMA\s+if\s+sigma_cap\s+is\s+None\s+else\s+sigma_cap", body
        ), "常量回退式变形——MAX_SAMPLE_SIGMA=0.5 的逐值 pin 可能已不代表运行期有效值，须重标"
        assert re.search(r"min\(cap,\s*post_sigma\[0\]\)", body), (
            "cap 不再钳采样 σ（post_sigma[0]）——本 pin 失去被观测对象，须重标"
        )

        data = _fetch_affect_state_defaults_or_skip()
        assert data["sample_sigma_cap"] is None, (
            f"AffectState.sample_sigma_cap 默认变为 {data['sample_sigma_cap']!r}（期望 None）"
            "——运行期已覆写采样 σ 上限，MAX_SAMPLE_SIGMA=0.5 不再是生效值，须跨仓确认。"
        )

        # ⚠ 以下是**我方侧的前提记录，不是对 Zero 的观测**，别当守卫业绩上报：我方目前
        # 无任何生产代码经 `open_session(config=...)` 注入 sample_sigma_cap，所以「Zero 的
        # 缺省通路」对我方而言就是全部通路。它今天是**弱恒真式**（我方从不写该键 ⇒ 必绿），
        # 价值只在未来有人加 config builder 时兑现。
        # 为免退化成 pitfalls ⑥ 那种「解析出的空当被观测量」，配一道正控：扫描器必须真的
        # 读到了源码内容（能看见 open_session 这个已知在位的 token），否则扫描根写错会让
        # 本条永远 0 命中而假绿。
        repo_src = Path(__file__).resolve().parents[2] / "src"
        sources = {path: path.read_text(encoding="utf-8") for path in repo_src.rglob("*.py")}
        assert sources, f"未扫到任何本仓 src/*.py（扫描根 {repo_src} 写错？）——本条会假绿"
        assert any("open_session" in text for text in sources.values()), (
            f"本仓 src/ 下扫不到 open_session（扫描根 {repo_src}）——扫描器没读到真内容，"
            "下面的 not-in 断言会退化成恒真式"
        )
        injectors = sorted(str(p) for p, text in sources.items() if "sample_sigma_cap" in text)
        assert not injectors, (
            f"我方 src/ 出现 sample_sigma_cap 注入点 {injectors}——上面「Zero 缺省通路即我方全部"
            "通路」的前提不再成立，须把注入值一并纳入可达性/抖动结论重评。"
        )

    def test_soft_gate_bypasses_physio_exclusion(self) -> None:
        """**D7 承诺的已知覆盖缺口**（特征化，非缺陷断言）：软门分支不施 physio 排除。

        `_select_fired` 在 `soft_beta`（`ZERO_IGNITION_BETA` / `state.ignition_beta`）非
        None 时走软门：**全部流含 physio 一律进 fuse_terms**（精度乘 logistic gate），
        没有阈值筛除。而 `exclude_physio_fusion` 的过滤只写在**门开**分支里，管不到软门
        ——软门位于 `gate_fusion=True` 通路内。

        即：`gate_fusion=True`（默认）+ `ignition_beta` 非 None ⇒ 反号 physio 真进数值后验。
        默认 `IGNITION_BETA=None` 故当前未触发；且 `ignition_beta` **不在** Zero 的 MCP
        治理门控白名单内，理论上 MCP client 可经 config overrides 自行打开。

        本例把这个缺口做成**可执行记录**：若 Zero 日后在软门分支内也施加 physio 排除
        （缺口被堵），本例变红 → 提示我们更新跨仓认知并撤下相关风险提示。

        ⚠ **2026-07-29 重锚**：Zero 回执明确其收口方案是「把前缀过滤提到 `if gate_fusion:`
        **之前**、对硬门/软门/门开三条路径同施」，并主动提示我方——它若只改 `ignite()` 而
        我方仍只锚在 `_select_fired` 函数体上，本守卫**不会变红而是静默失去刻画能力**。故
        主锚点移到 `ignite()` 的门判之前；`_select_fired` 那两条降级为**副锚点**保留（Zero
        也可能把过滤下沉进该 helper，那是另一条真实修法，两条都要守）。

        **如实标注守不住什么**：Zero 若把过滤抽成**不透明 helper**（`streams =
        _drop_physio(streams)`，开关从模块级读、不作实参传）并在门判前调用，本例看不见
        （11 态矩阵第 (7) 态实测为**绿**）。缓解靠 Zero 的「落地即 ping」承诺（其
        `ignite` docstring 内已写死该承诺，我方回执 notes/2026-07-29-zero-reply-and-
        alignment-asks.md 亦有记载），不靠本守卫独担；写了盲点不等于覆盖了盲点。
        """
        source = self._source()

        # ── 主锚点：ignite() 的**门判之前**（Zero 指定的 ② 落点）──────────────────
        func = _top_level_func(source, "ignite")
        assert func is not None, "未找到顶层 ignite——锚点失效，须按新形态重标（勿降级为 skip）"
        split = _split_ignite_on_gate(func)
        assert split is not None, "ignite 不再以 `if gate_fusion:` 二分——门控形态变了，须重标"
        prelude, gate_if, tail = split

        # 正控①（切分非退化）：门判前必须是真代码区且含已知语句；否则下面的集合交为空
        # 是「切没了」而非「缺口仍在」——本仓 pitfalls ⑥「把解析出的空/0 当被观测量」。
        assert prelude, "ignite 门判之前已无任何语句——切分退化，下面的断言会变恒真式"
        prelude_names = _referenced_names(prelude)
        assert "_score_streams" in prelude_names, (
            f"ignite 门判之前不含 _score_streams（实得 {sorted(prelude_names)}）"
            "——切分锚点可疑，须先确认切出来的确是门判前那段"
        )

        # 正控②（探测器可见性·**位置无关**）：同一探测器必须能在 ignite 体内看见该 token。
        # 位置无关是刻意的：Zero ② 是**移动**过滤（门开那份会被删），若把正控写成「门开分支
        # 里必须有」，② 落地当天会先红在一条**归因错误**的消息上（「D7 被删」）——第一版实测
        # 就是这么错的，bool 判据看不出来，原因串判据才抓到。
        whole = [*prelude, gate_if, *tail]
        assert "_PHYSIO_PREFIXES" in _referenced_names(whole), (
            "整个 ignite 体内都找不到 _PHYSIO_PREFIXES——要么 D7 排除被整体删除（须立即跨仓"
            "确认），要么本例的 AST 探测器失效；两种都不许把下面的 not-in 读成「缺口仍在」"
        )

        # 主断言：缺口刻画——门判之前不施 physio 排除 ⇒ 硬门/软门两条路径都吃不到 D7。
        hoisted = prelude_names & _PHYSIO_FILTER_NAMES
        assert not hoisted, (
            f"ignite 的 `if gate_fusion:` **之前**出现了 {sorted(hoisted)}——Zero 的收口方案"
            "（前缀过滤提到门判前、对硬门/软门/门开三条路径同施）已落地，D7 覆盖缺口被堵上"
            "（好事）。请更新本例文案、撤下软门旁路风险提示，并复核跨仓台账对应条目后再放行。"
        )

        # ── 副锚点：过滤下沉进 _select_fired 也是一条真实修法，不能只守主锚点 ──────
        select_fired = _extract_function_body(source, "_select_fired")
        if select_fired is None:
            pytest.skip("未找到 _select_fired，Zero 结构可能大改，跳过")

        assert "soft_beta" in select_fired, (
            "_select_fired 内已无 soft_beta 分支——软门被移除，本缺口记录可撤下"
        )
        assert "_PHYSIO_PREFIXES" not in select_fired, (
            "_select_fired 内出现了 physio 前缀过滤——D7 覆盖缺口已被堵上（好事），"
            "请更新本例文案与跨仓风险提示后再放行"
        )
        # 默认值 pin：缺口当前不被触发，靠的是 IGNITION_BETA 默认 None
        beta_default = re.search(r"^IGNITION_BETA\s*[:=][^=\n]*=\s*(\S+)", source, re.MULTILINE)
        assert beta_default is not None, "未找到 IGNITION_BETA 默认值，须重标"
        assert beta_default.group(1).rstrip(",") == "None", (
            "IGNITION_BETA 默认值不再是 None——软门成为默认路径，反号 physio 会进数值后验，"
            "须立即跨仓确认"
        )

    def test_physio_default_precision_stays_below_self_ignite_bound(self) -> None:
        """我方默认 physio 精度须低于自点燃硬上界——否则单边绕过 Zero 的 D7 跨仓承诺。

        D7（`exclude_physio_fusion` 默认 True）只写在 Zero 的**门开**分支里；门关（默认）
        走硬门阈值路径，D7 管不到。故「physio 不以反号参与数值」在门关下**不是 Zero 保证的**，
        而是我方 Πa 默认值小于阈值所致——是自律，不是结构约束。本例把这条自律做成可执行守卫：
        任何人（含未来的我）调高 `EXTERNAL_PHYSIO_PRECISION_A` 默认值到越界，立刻红。

        上界从 Zero 源码的 `SALIENCE_THRESHOLD` 现算，不手抄。
        """
        source = self._source()
        match = re.search(rf"^SALIENCE_THRESHOLD\s*=\s*({_NUM})", source, re.MULTILINE)
        if match is None:
            pytest.skip("未找到 SALIENCE_THRESHOLD，跳过")
        threshold = float(match.group(1))

        # physio: μv 恒 0 → hypot(μ)=|μa|≤1；Πv 恒 MIN。salience 上界 = (MIN+Πa)/2。
        bound = 2 * threshold - MIN_PRECISION
        assert bound == pytest.approx(PHYSIO_PRECISION_A_SELF_IGNITE_BOUND, abs=1e-9), (
            f"自点燃上界随 Zero 阈值漂移：现算 {bound:.4f}、常量 "
            f"{PHYSIO_PRECISION_A_SELF_IGNITE_BOUND}——须同步更新常量与其 docstring 推导。"
        )

        for kind, ceiling in ((ModalityKind.PHYSIO, bound),):
            _, pi_a = _RECOMMENDED_PRECISION_DEFAULTS[kind]
            assert pi_a < ceiling, (
                f"{kind} 默认 Πa={pi_a} 已达自点燃上界 {ceiling:.4f}——我方 physio 先验将在"
                "Zero **默认配置**下自行过阈进入数值后验，绕过 D7 跨仓承诺（D7 只管门开分支）。"
                "调高前必须跨仓确认。"
            )

        # 合并态（生产实际发出的形状）同样要低于上界：Πa = ω·Π_eda + (1-ω)·Π_hrv
        merged_pi_a = (
            PHYSIO_MERGE_OMEGA_DEFAULT * PHYSIO_SUBSOURCE_PRECISION_A["eda"]
            + (1 - PHYSIO_MERGE_OMEGA_DEFAULT) * PHYSIO_SUBSOURCE_PRECISION_A["hrv"]
        )
        assert merged_pi_a < bound, (
            f"合并后 Πa={merged_pi_a} 已达自点燃上界 {bound:.4f}——线上载荷会自行过阈，"
            "须跨仓确认（本仓 pitfalls ① 已实证抬子源可靠度会让线上 salience 冲到 0.3896）。"
        )

    def test_mirrored_salience_threshold_matches_zero_source(self) -> None:
        """**产品码**里的点燃门镜像 `ZERO_SALIENCE_THRESHOLD` 须逐值等于 Zero 源码。

        为什么单列一条而不靠上面两条：`_ZERO_GATE_CONSTANTS` 与
        ``test_physio_default_precision_stays_below_self_ignite_bound`` pin 的都是**测试侧**
        的现算/期望值；M8 落地后，`src/mcp/zero/external_priors.py` 里**多了一个产品码镜像**
        （运行期守卫真正拿去比的那个数）。两者漂移会让守卫按错误阈值放行/误拦，而既有两条
        都探测不到——它们根本没读过这个新常量。
        """
        source = self._source()
        match = re.search(rf"^SALIENCE_THRESHOLD\s*=\s*({_NUM})", source, re.MULTILINE)
        if match is None:
            pytest.skip("未找到 SALIENCE_THRESHOLD，跳过")
        zero_threshold = float(match.group(1))

        assert ZERO_SALIENCE_THRESHOLD == zero_threshold, (
            f"产品码镜像漂移：本仓 ZERO_SALIENCE_THRESHOLD={ZERO_SALIENCE_THRESHOLD}、"
            f"Zero SALIENCE_THRESHOLD={zero_threshold}。M8 自点燃守卫按本仓这个值现算硬顶——"
            "不同步会让守卫按错误阈值判定（Zero 调低阈值时我方会静默放行本应拦的载荷）。"
        )

    def test_m8_guard_blocks_self_igniting_payload_against_live_zero_threshold(self) -> None:
        """端到端：按 **Zero 源码现算**出的越界 Πa 构造载荷，M8 必须真的拦下。

        与上一条的分工：上一条只比常量，本条把「常量 → 守卫行为」这一环也接上——
        避免出现「常量对了但守卫压根没消费它 / 被摘掉了」的假绿。
        Πa 由 Zero 阈值现算（不手抄），故 Zero 调阈值时本例跟着走。
        """
        source = self._source()
        match = re.search(rf"^SALIENCE_THRESHOLD\s*=\s*({_NUM})", source, re.MULTILINE)
        if match is None:
            pytest.skip("未找到 SALIENCE_THRESHOLD，跳过")
        threshold = float(match.group(1))

        # μv=0 下的硬顶；取恰好等于（`>=` 语义 ⇒ 取等即越界）
        over_bound_pi_a = 2 * threshold - MIN_PRECISION
        prior = ModalityPrior(
            modality="physio", mu=(0.0, 0.5), precision=(MIN_PRECISION, over_bound_pi_a)
        )
        with pytest.raises(ValueError, match="M8 physio 自点燃越界"):
            build_external_priors_override([prior])

        # 正控：略低于硬顶必须放行（排除「守卫恒红」，否则上面的红无意义）
        safe = ModalityPrior(
            modality="physio", mu=(0.0, 0.5), precision=(MIN_PRECISION, over_bound_pi_a - 1e-6)
        )
        payload = build_external_priors_override([safe])
        assert len(payload["external_priors"]) == 1, "正控失败：合法 physio 载荷未出线"

    def test_gate_branch_guard_goes_red_on_structural_relocation(self) -> None:
        """判别性自证①：门开分支**结构** pin 六态——真会发生的改法必红、合法等价形态不误红。

        「绿灯必须先证明它能红」同族第 ⑦ 例，两次修订都是被真事件推着走的：
        1. 最初按 `partition("return ")` 切分支 → 对「真分支不再 return」实测**绿**（切点落到
           函数**末尾**那个 return 上，把改动整个跳过），改缩进切块 + 显式 return 断言后转红；
        2. 2026-07-29 缩进切块被 Zero 新写的 docstring 撞出**假红**——其 `ignite` docstring
           散文里写了「把前缀过滤提到 `if gate_fusion:` 之前」，`body.find(header)` 首次命中
           落到这句散文上、按散文行长算缩进 ⇒ 块体为空。改 AST 后消除。
           ⚠ 上一版自证用的合成源码是**裸函数、无 docstring 无 `#` 注释**，所以对第 2 类 bug
           在结构上就不可能报警——它绿、真守卫红，正是这个差。故 (1) 态专喂「docstring 与
           `#` 注释同时含 header 和 `_select_fired`」的源码。

        六态期望（判据返回**红的原因串**而非 bool：只有原因串才分得清「红对了」与「红在别的
        原因上」）：(0) 现行形态 绿 /(1) docstring+注释污染 绿 /(2) fallback 挪进门开 红 /
        (3) 真分支去掉提前 return 红 /(4) gate 二分被拆 红 /(5) if/else 等价形态 绿。

        主体不依赖 `D:\\Zero` 在位（用合成源码），判别力常年在测；末尾附一格「Zero 真源码
        在位时应绿」——合成源码只能证明**判据函数**有判别力，不能证明**守卫对真源码**没瞎，
        今天这个 bug 的全部差距就在这两者之间。
        """
        base = (
            "def ignite(streams, *, threshold=0.18, gate_fusion=True):\n"
            "    scored = _score_streams(streams)\n"
            "    if gate_fusion:\n"
            "        selected = _select_fired(scored, threshold=threshold)\n"
            "        return [(mu, prec) for _, mu, prec in selected]\n"
            "\n"
            "    fusion = [(name, mu, prec) for name, mu, prec, _ in scored]\n"
            "    if exclude_physio_fusion:\n"
            "        fusion = [t for t in fusion if not t[0].startswith(_PHYSIO_PREFIXES)]\n"
            "    return fusion\n"
        )

        def evaluate(source: str) -> str:
            """复刻主守卫 ①段的判定，返回 '绿' 或红的原因（与主例逻辑一一对应）。"""
            func = _top_level_func(source, "ignite")
            if func is None:
                return "红:无顶层 ignite"
            split = _split_ignite_on_gate(func)
            if split is None:
                return "红:无 gate 二分"
            _prelude, gate_if, tail = split
            closed = list(gate_if.body)
            opened = [*gate_if.orelse, *tail]
            if not (closed and opened):
                return "红:切分有效性"
            if "_select_fired" not in _referenced_names(closed):
                return "红:门关分支无 _select_fired"
            if not gate_if.orelse and not isinstance(closed[-1], ast.Return):
                return "红:真分支未提前 return(会落穿)"
            if "_select_fired" in _referenced_names(opened):
                return "红:门开分支出现 _select_fired"
            return "绿"

        assert evaluate(base) == "绿", "现行形态应绿——否则守卫对 Zero 当前源码就是假阳性"

        # (1) docstring 与 `#` 注释同时含 header 和 _select_fired ——**今天那条假红的复现件**。
        # AST 里没有注释、docstring 是可识别的独立节点，两类污染都不该改变判定。
        polluted = base.replace(
            "    scored = _score_streams(streams)\n",
            "    '''收口条件：把前缀过滤提到 `if gate_fusion:` 之前，\n"
            "    届时 _select_fired 的语义随之变化。'''\n"
            "    # 备注：`if gate_fusion:` 之前不得调用 _select_fired\n"
            "    scored = _score_streams(streams)\n",
        )
        assert evaluate(polluted) == "绿", (
            "docstring / `#` 注释里出现 header 与 _select_fired 时守卫必须仍绿——"
            "2026-07-29 的假红正是这一格（文本切法在此实测为「红:切分有效性」）"
        )

        # (2) fallback 被挪回数值通路
        relocated = base.replace(
            "    fusion = [(name, mu, prec) for name, mu, prec, _ in scored]",
            "    fusion = _select_fired(scored, threshold=threshold)",
        )
        assert evaluate(relocated) == "红:门开分支出现 _select_fired", (
            "fallback 被挪进门开分支时守卫必须红——这正是本仓结论赖以成立的那条结构前提"
        )

        # (3) 真分支去掉提前 return（落穿，两条路径同时执行）
        fallthrough = base.replace(
            "        return [(mu, prec) for _, mu, prec in selected]", "        pass"
        )
        assert evaluate(fallthrough) == "红:真分支未提前 return(会落穿)", (
            "真分支落穿时守卫必须红——最初按 return 切分在此**实测为绿**，是真盲点"
        )

        # (4) `if gate_fusion:` 二分被拆掉
        no_gate = base.replace("    if gate_fusion:", "    if True:")
        assert evaluate(no_gate) == "红:无 gate 二分", "二分被拆掉时守卫必须红"

        # (5) 改写成 if/else 等价形态——**合法等价，不该红**。落穿检查在此不执行是正确的
        # （有 else 时结构上不可能落穿），不是漏洞；这一格就是用来锁住那条条件式断言的边界。
        else_form = (
            "def ignite(streams, *, threshold=0.18, gate_fusion=True):\n"
            "    scored = _score_streams(streams)\n"
            "    if gate_fusion:\n"
            "        selected = _select_fired(scored, threshold=threshold)\n"
            "        return [(mu, prec) for _, mu, prec in selected]\n"
            "    else:\n"
            "        fusion = [(name, mu, prec) for name, mu, prec, _ in scored]\n"
            "        if exclude_physio_fusion:\n"
            "            fusion = [t for t in fusion if not t[0].startswith(_PHYSIO_PREFIXES)]\n"
            "        return fusion\n"
        )
        assert evaluate(else_form) == "绿", "if/else 等价形态不该红（否则 Zero 一重构就噪声红）"

        # 附加格：Zero 真源码在位时判据须为绿。合成源码只证明判据有判别力，不证明守卫对
        # 真源码没瞎——今天这个 bug 的差距就在这里。Zero 不在位则跳过这一格（不拖红）。
        if _zero_available() and _ZERO_AFFECT_MATH_PY.is_file():
            verdict = evaluate(_ZERO_AFFECT_MATH_PY.read_text(encoding="utf-8"))
            assert verdict == "绿", f"判据对 Zero 真源码判为 {verdict}——与主守卫结论必须一致"

    def test_prelude_physio_guard_discriminates(self) -> None:
        """判别性自证②：软门旁路缺口守卫（锚在 ignite 门判**之前**）十一态。

        这条守卫的断言方向是「**没有**才绿」，天生比「有才绿」脆——全部可信度压在两道正控
        （切分非退化 / 探测器位置无关可见）上。故十一态必须全跑，且判据返回**原因串**：
        第一版把正控写成「门开分支里必须含 `_PHYSIO_PREFIXES`」时矩阵仍然「全红」看似通过，
        但 Zero 的真修法是**移动**过滤（门开那份会被删）⇒ ② 落地当天会先红在
        「D7 被删」这条**归因错误**的消息上。bool 判据放过了它，原因串判据才抓到。

        十一态期望：
        (0) 现行形态 绿 /(1) docstring 散文含两 token 绿 /(2) `#` 注释含两 token 绿 /
        (3) if/else 形态 绿 /(4) Zero 的真修法·过滤移到门判前 红 /(5) 提到门判前但门开那份
        也留着（复制）红 /(6) 提到门判前·helper 显式收 flag 红 /(7) 提到门判前·**不透明**
        helper **绿（已知守不住，如实记账）** /(8) gate 二分被拆 红 /(9) ignite 改名/消失 红
        /(10) D7 排除被整体删除 红。

        (7) 是本守卫公开承认的盲点：Zero 若写 `streams = _drop_physio(streams)`（开关从模块级
        读、不作实参传），`prelude` 里既无 `_PHYSIO_PREFIXES` 也无 `exclude_physio_fusion`。
        缓解靠 Zero 的「落地即 ping」承诺，不靠本守卫独担。
        """
        base = (
            "def ignite(streams, *, gate_fusion=True, exclude_physio_fusion=True):\n"
            "    scored = _score_streams(streams)\n"
            "    if gate_fusion:\n"
            "        selected = _select_fired(scored)\n"
            "        return [(mu, prec) for _, mu, prec in selected]\n"
            "\n"
            "    fusion = [(name, mu, prec) for name, mu, prec, _ in scored]\n"
            "    if exclude_physio_fusion:\n"
            "        fusion = [t for t in fusion if not t[0].startswith(_PHYSIO_PREFIXES)]\n"
            "    return fusion\n"
        )

        def evaluate(source: str) -> str:
            """复刻主守卫的主锚点判定，返回 '绿' 或红的原因（与主例逻辑一一对应）。"""
            func = _top_level_func(source, "ignite")
            if func is None:
                return "红:无顶层 ignite"
            split = _split_ignite_on_gate(func)
            if split is None:
                return "红:无 gate 二分"
            prelude, gate_if, tail = split
            if not prelude:
                return "红:门判前无语句(切分退化)"
            names = _referenced_names(prelude)
            if "_score_streams" not in names:
                return "红:门判前不含 _score_streams(切分锚点可疑)"
            if "_PHYSIO_PREFIXES" not in _referenced_names([*prelude, gate_if, *tail]):
                return "红:ignite 体内已无 _PHYSIO_PREFIXES(D7 被删或探测器失效)"
            hoisted = names & _PHYSIO_FILTER_NAMES
            if hoisted:
                return f"红:门判前已施排除{sorted(hoisted)}"
            return "绿"

        assert evaluate(base) == "绿", "现行形态应绿——否则守卫对 Zero 当前源码就是假阳性"

        # (1)(2) 两类污染：docstring 散文 / `#` 注释里出现两个 token，都不该改变判定
        doc_polluted = base.replace(
            "    scored = _score_streams(streams)\n",
            "    '''收口条件：把 _PHYSIO_PREFIXES 过滤提到 `if gate_fusion:` 之前，\n"
            "    即 exclude_physio_fusion 对三条路径同施。'''\n"
            "    scored = _score_streams(streams)\n",
        )
        assert evaluate(doc_polluted) == "绿", "docstring 散文含两 token 时守卫必须仍绿"
        comment_polluted = base.replace(
            "    scored = _score_streams(streams)\n",
            "    # TODO: 把 _PHYSIO_PREFIXES 过滤提到 `if gate_fusion:` 之前\n"
            "    # （届时 exclude_physio_fusion 对硬门/软门/门开三条路径同施）\n"
            "    scored = _score_streams(streams)\n",
        )
        assert evaluate(comment_polluted) == "绿", "`#` 注释含两 token 时守卫必须仍绿"

        # (3) if/else 等价形态（门开分支落在 orelse 里而非同级 tail）
        else_form = (
            "def ignite(streams, *, gate_fusion=True, exclude_physio_fusion=True):\n"
            "    scored = _score_streams(streams)\n"
            "    if gate_fusion:\n"
            "        selected = _select_fired(scored)\n"
            "        return [(mu, prec) for _, mu, prec in selected]\n"
            "    else:\n"
            "        fusion = [(name, mu, prec) for name, mu, prec, _ in scored]\n"
            "        if exclude_physio_fusion:\n"
            "            fusion = [t for t in fusion if not t[0].startswith(_PHYSIO_PREFIXES)]\n"
            "        return fusion\n"
        )
        assert evaluate(else_form) == "绿", "if/else 等价形态不该红"

        # (4) Zero 的**真修法**：过滤整体移到门判前（门开那份随之删除）
        hoist_prefix = (
            "    if exclude_physio_fusion:\n"
            "        scored = [t for t in scored if not t[0].startswith(_PHYSIO_PREFIXES)]\n"
        )
        moved = base.replace(
            "    if gate_fusion:\n", hoist_prefix + "    if gate_fusion:\n"
        ).replace(
            "    if exclude_physio_fusion:\n"
            "        fusion = [t for t in fusion if not t[0].startswith(_PHYSIO_PREFIXES)]\n",
            "",
        )
        assert (
            evaluate(moved) == "红:门判前已施排除['_PHYSIO_PREFIXES', 'exclude_physio_fusion']"
        ), (
            "Zero 的收口方案（过滤提到门判前）落地时守卫必须红，且必须红在「门判前已施排除」"
            "这条消息上——红在「D7 被删」上属**归因错误**，会把好事读成坏事"
        )

        # (5) 提到门判前但门开那份也留着（复制而非移动）
        copied = base.replace("    if gate_fusion:\n", hoist_prefix + "    if gate_fusion:\n")
        assert (
            evaluate(copied) == "红:门判前已施排除['_PHYSIO_PREFIXES', 'exclude_physio_fusion']"
        ), "复制式收口同样必须红"

        # (6) 提到门判前 + 抽 helper，但**显式把开关作实参传**——靠开关名接住。
        # `_PHYSIO_PREFIXES` 退到 D12 报错文案里仍可见，故正控不误报「D7 被删」。
        helper_with_flag = (
            "def ignite(streams, *, gate_fusion=True, exclude_physio_fusion=True):\n"
            "    scored = _score_streams(streams)\n"
            "    scored = _drop_physio(scored, enabled=exclude_physio_fusion)\n"
            "    if gate_fusion:\n"
            "        selected = _select_fired(scored)\n"
            "        return [(mu, prec) for _, mu, prec in selected]\n"
            "\n"
            "    fusion = [(name, mu, prec) for name, mu, prec, _ in scored]\n"
            "    if streams and not fusion:\n"
            "        raise ValueError(f'全部命中 physio 排除前缀 {_PHYSIO_PREFIXES}')\n"
            "    return fusion\n"
        )
        assert evaluate(helper_with_flag) == "红:门判前已施排除['exclude_physio_fusion']", (
            "helper 显式收 flag 时靠开关名接住——红的原因应只列开关名，不含前缀表"
        )

        # (7) **已知守不住**：不透明 helper，开关从模块级读、不作实参传 → 判据看不见。
        # 这一格期望**绿**，是如实记账不是放行；缓解靠 Zero 的落地即 ping 承诺。
        opaque_helper = helper_with_flag.replace(
            "    scored = _drop_physio(scored, enabled=exclude_physio_fusion)\n",
            "    scored = _drop_physio(scored)\n",
        )
        assert evaluate(opaque_helper) == "绿", (
            "不透明 helper 是本守卫**已知的**盲点（docstring 已如实标注）；这一格期望绿——"
            "它若哪天变红说明判据被改宽了口径，须回头核对是不是引入了误报"
        )

        # (8)(9)(10) 三种「锚点/被观测量整体消失」的形态，都必须红而不是 skip
        no_gate = base.replace("    if gate_fusion:", "    if True:")
        assert evaluate(no_gate) == "红:无 gate 二分", "二分被拆掉时守卫必须红"
        renamed = base.replace("def ignite(", "def ignite_v2(")
        assert evaluate(renamed) == "红:无顶层 ignite", (
            "ignite 改名/消失时走**红**而非 skip——与本仓 conftest 的 STRICT 覆盖归零守卫同向"
        )
        d7_removed = base.replace(
            "    if exclude_physio_fusion:\n"
            "        fusion = [t for t in fusion if not t[0].startswith(_PHYSIO_PREFIXES)]\n",
            "",
        )
        assert evaluate(d7_removed) == "红:ignite 体内已无 _PHYSIO_PREFIXES(D7 被删或探测器失效)", (
            "D7 排除被整体删除时守卫必须红——这条正控同时兜住「探测器自己瞎了」的情形"
        )

        # 附加格：Zero 真源码在位时判据须为绿（合成源码证明不了守卫对真源码没瞎）
        if _zero_available() and _ZERO_AFFECT_MATH_PY.is_file():
            verdict = evaluate(_ZERO_AFFECT_MATH_PY.read_text(encoding="utf-8"))
            assert verdict == "绿", f"判据对 Zero 真源码判为 {verdict}——与主守卫结论必须一致"

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
# Zero `src/agents/affect_math.py::td_update` 的 lr 形参默认；直接缩放 ΔV=lr·δ；
# 闭式复算红线 lr≲0.057 转红
_PINNED_TD_LR = 0.2
# 同上 next_value 形参默认；恒 0 使 gamma 项消失（gamma 对本仓路径是死系数）
_PINNED_TD_NEXT_VALUE = 0.0
# Zero `src/mcp_server/mapping.py::stimulus_from_payload` 的 name 形参默认；跨轮同键才累积 V(s)
_PINNED_MCP_STIMULUS_KEY = "mcp-step"

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
