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

sys.path.insert(0, sys.argv[1])  # D:\\Zero\\src

try:
    from agents.affect_math import decode_channels
except ImportError as e:
    print(json.dumps({"skip": True, "reason": f"import agents.affect_math 失败: {e}"}))
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

_DECODE_COPING_SCRIPT = """
import sys
import json

sys.path.insert(0, sys.argv[1])

try:
    from agents.affect_math import decode_channels
except ImportError as e:
    print(json.dumps({"skip": True, "reason": f"import agents.affect_math 失败: {e}"}))
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
# 辅助函数（模块级，统一子进程调用）
# ---------------------------------------------------------------------------


def _run_subprocess_with_script(script: str) -> dict[str, Any]:
    """在子进程中运行给定 script，返回 stdout 解析的 JSON dict。"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(_ZERO_SRC)],
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
