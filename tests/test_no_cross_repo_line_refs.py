"""防回归守卫：禁止本仓新增「指向 Zero 源码的行号引用」（R7，2026-07-29）。

背景（为什么值一条守卫）：跨仓行号**腐烂时不驱红**——Zero 一次编辑就让本仓几十处注释
静默指错，读者按图索骥落到不相干的行，而全套测试照绿。2026-07-29 实测：Zero 的
`_PHYSIO_PREFIXES` 当天出现 `:971 → :1166 →(未提交工作树) :1193` 三个值，`stream_salience`
`:673 → :797 → :798`——行号连「同一天」都不稳。R7 因此一次性清空全仓 Zero 侧行号，
统一改写为 ``Zero `<仓内相对路径>::<符号名>` ``；本文件负责让它不再长回来。

**范围只限「指向 Zero 的」**：本仓**自指**行号（如 `src/orchestration/persistence.py:85`、
`.env.example` 里指向本仓测试用例的裸 `:276`）不在禁令内——它们与被引文件同仓同 commit，
腐烂时至少可被本仓改动 review 到，且约 33 处存量，一并禁会把守卫变成噪声源。

判别力设计（对照本仓 pitfalls ⑥「断言退化成恒真式」）：
1. **被观测量是本仓自己的文件文本**，非「从 Zero 解析出来的值」，不存在「解析出的 0 当被
   观测量」那条塌缩路径。
2. 另一条恒真路径是**扫描根写错/目录不存在 → 永远 0 命中**。故：
   - 不用 `git ls-files` 取文件表——`ai-docs/` 与 `notes/` 都在 `.git/info/exclude` 里，
     用 git 取表会把 ai-docs 整个跳过，守卫在 ai-docs 腐烂时永绿；这里走文件系统遍历。
   - `test_scan_is_live` 用**正控**证明遍历真的读到了内容（文件数下限 + 已知哨兵串可见）。
   - `test_judge_discriminates` 用**同一判据函数**跑 4 类阳性 + 5 类阴性样本，且判据返回
     **原因串**而非 bool——这样「红了」还能核「红在正确的原因上」。
3. 判据分层是因为**两仓有同名模块**（`state.py` / `client.py` / `perception.py` 本仓也有）：
   Zero 独有基名无条件红；同名基名需同行有 Zero 标记才红。宁可漏报也不误伤本仓自指。
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 扫描根。`notes/` 刻意排除：那是历史纪要与 Zero 原文回执，引用对方行号是**当时的现场快照**，
# 不是导航坐标，删掉等于抽掉证据（同理由见 ai-docs/docs/engineering-practices.md 的行号教训条）。
_SCAN_DIRS: tuple[str, ...] = ("src", "tests", "mcp-server", "docs", "ai-docs")
_SCAN_FILES: tuple[str, ...] = (".env.example", "README.md", "pyproject.toml")
# 必须存在的根（git 跟踪）；缺失即说明扫描根写错，直接失败而不是静默 0 命中。
_REQUIRED_DIRS: tuple[str, ...] = ("src", "tests")
_SCAN_SUFFIXES: frozenset[str] = frozenset({".py", ".md", ".toml", ".ts", ".mmd", ".example"})
_SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {"__pycache__", ".pytest_cache", ".git", ".ruff_cache", ".mypy_cache", "node_modules", "notes"}
)

# Tier A：**Zero 独有**的模块基名（已逐一核对本仓 src/、tests/ 无同名 .py）。带行号即红。
_ZERO_ONLY_MODULES: tuple[str, ...] = (
    "affect_math",
    "affect_core",
    "appraisal",
    "auth",
    "chat_driver",
    "composite",
    "emotion_lexicon",
    "expression",
    "external_prior",
    "facs_decoder",
    "graph",
    "language",
    "mapping",
    "physiology_decoder",
    "prosody_decoder",
    "runner",
    "server",
    "supervisor",
    "wesad",
)
# Tier B：**两仓同名**基名（本仓 src/orchestration/state.py、src/mcp/zero/{client,perception}.py）。
# 只在同行另有 Zero 标记时才判为跨仓引用，否则视为本仓自指放行。
_AMBIGUOUS_MODULES: tuple[str, ...] = ("state", "client", "perception", "physiology", "prosody")

_ZERO_ONLY_RE = re.compile(r"\b(?:" + "|".join(_ZERO_ONLY_MODULES) + r")\.py:\d+")
# Tier B 的 Zero 标记必须**紧贴引用**（`Zero` 后只隔非空白的路径片段），不能是「本行某处出现
# Zero 三个字母」。实测教训：`.env.example` 的「透传给 Zero server 的能力门控……只经
# client.py:115」是本仓自指，宽松的「同行有 Zero」判据把它误报了（1 轮扫描 27 命中里 2 处误报）。
_AMBIGUOUS_RE = re.compile(
    r"Zero\s*\S*\b(?:" + "|".join(_AMBIGUOUS_MODULES) + r")\.py:\d+",
)
# Zero 绝对路径带行号（兜住不在上面两张表里的对方模块）。
_ZERO_ABS_PATH_RE = re.compile(r"[Dd]:\\{1,2}Zero\\{1,2}[^\s\"'`]*\.py:\d+")
# 裸行号：前面不能紧跟词字符/路径分隔符，排除 `http://host:8000`、`12:25`、`§六(e):186` 之类。
_BARE_LINENO_RE = re.compile(r"(?<![\w./\\)])\s?:\d{2,4}\b")
# 裸行号只在**上下文窗口**里出现 Zero 模块名时才算——它总是接在 `affect_math.py:1052` 这类
# 完整引用后面做枚举（如「其 :1052 与 :1058 判据」），孤立的裸数字不判。
_BARE_CONTEXT_LINES = 3

_FIX_HINT = "改写为 ``Zero `<仓内相对路径>::<符号名>` ``（跨仓引用只写符号名，禁写对方行号）"


def judge_line(line: str, context: str = "") -> str | None:
    """判据：该行是否含「指向 Zero 源码的行号引用」。

    返回**原因串**（而非 bool）——守卫红时要能核「红在正确的原因上」，bool 会把归因
    错误的红也放过（本仓已有前车之鉴）。无违规返回 None。

    Args:
        line: 待判定的单行文本。
        context: 该行及其前若干行拼成的上下文窗口，仅供裸行号规则判定「是否处在 Zero 语境」。
    """
    hit = _ZERO_ONLY_RE.search(line)
    if hit is not None:
        return f"Zero 独有模块基名带行号：{hit.group(0)!r}"
    hit = _ZERO_ABS_PATH_RE.search(line)
    if hit is not None:
        return f"Zero 绝对路径带行号：{hit.group(0)!r}"
    hit = _AMBIGUOUS_RE.search(line)
    if hit is not None:
        return f"两仓同名模块带行号、且紧贴 Zero 标记：{hit.group(0)!r}"
    hit = _BARE_LINENO_RE.search(line)
    if hit is not None and (_ZERO_ONLY_RE.search(context) or _ZERO_ABS_PATH_RE.search(context)):
        return f"Zero 语境内的裸行号：{hit.group(0).strip()!r}"
    return None


def iter_scan_files() -> list[Path]:
    """列出受本守卫管辖的文件（文件系统遍历，**不用 git ls-files**，见模块头注释 2）。"""
    found: list[Path] = []
    for name in _SCAN_DIRS:
        root = REPO_ROOT / name
        if not root.is_dir():
            continue  # ai-docs/ 等本地目录在 git worktree 中可能不存在
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in _SCAN_SUFFIXES:
                continue
            if _SKIP_DIR_NAMES & set(path.relative_to(REPO_ROOT).parts):
                continue
            found.append(path)
    for name in _SCAN_FILES:
        path = REPO_ROOT / name
        if path.is_file():
            found.append(path)
    return sorted(set(found))


def scan_repo() -> list[str]:
    """扫描全仓，返回违规描述列表（`相对路径:行号: 原因 | 原文`）。"""
    violations: list[str] = []
    for path in iter_scan_files():
        if path.name == Path(__file__).name:
            continue  # 本文件自身含示例串，跳过
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        for idx, line in enumerate(lines):
            window = "\n".join(lines[max(0, idx - _BARE_CONTEXT_LINES) : idx + 1])
            reason = judge_line(line, window)
            if reason is not None:
                violations.append(f"{rel}:{idx + 1}: {reason} | {line.strip()[:100]}")
    return violations


class TestJudgeDiscriminates:
    """判据函数的判别力自证：阳性必红且**红在正确的原因上**，阴性必绿。"""

    def test_zero_only_module_ref_is_flagged(self) -> None:
        reason = judge_line("# 镜像 Zero affect_math.py:1039-1043 的 M7 fail-fast")
        assert reason is not None and "Zero 独有模块基名" in reason, reason

    def test_zero_abs_path_ref_is_flagged(self) -> None:
        reason = judge_line(r'"""镜像 D:\Zero\src\agents\foo_module.py:21 的常量。"""')
        assert reason is not None and "绝对路径" in reason, reason

    def test_ambiguous_module_adjacent_to_zero_mark_is_flagged(self) -> None:
        """两仓同名的 `state.py`：写成 Zero 仓相对路径带行号时判红（Tier B）。"""
        reason = judge_line("镜像 Zero `src/orchestration/state.py:228` 的 cap 字段")
        assert reason is not None and "两仓同名模块" in reason, reason

    def test_bare_lineno_in_zero_context_is_flagged(self) -> None:
        context = "# Zero 侧无兜底——其 affect_math.py:1052 `pi_v <= 0.0`\n# 与 :1058 `pi_v > cap`"
        reason = judge_line("# 与 :1058 `pi_v > cap` 对 NaN 同样恒 False", context)
        assert reason is not None and "裸行号" in reason, reason

    def test_self_ref_module_is_not_flagged(self) -> None:
        assert judge_line("见 src/orchestration/persistence.py:85 的两表分离") is None

    def test_ambiguous_module_without_zero_mark_is_not_flagged(self) -> None:
        """本仓自指的 state.py 行号（.env.example 实例）不得误伤。"""
        assert (
            judge_line("# src/orchestration/state.py:120、desktop_graph.py:71 读 os.environ")
            is None
        )

    def test_self_ref_on_line_that_merely_mentions_zero_is_not_flagged(self) -> None:
        """回归钉：这两行是**本仓自指**，只因同句提到「Zero server」被早期宽判据误报过。

        它们是守卫开发期实测出的**真实误报**（首轮全仓 27 命中里的 2 条，均在 .env.example）。
        Tier B 由「同行有 Zero」收紧为「Zero 标记须紧贴引用」正是为消掉它们——放宽回去会
        让守卫开始报本仓自指，几次之后就没人再看它的红了。
        """
        assert (
            judge_line("# 透传给 Zero server 的能力门控（本仓不读，只经 client.py:115 传） ")
            is None
        )
        assert (
            judge_line("# src/mcp/zero/client.py:107-118；http 模式下须在 Zero server 侧设") is None
        )

    def test_bare_lineno_without_zero_context_is_not_flagged(self) -> None:
        """`.env.example` 指向本仓测试用例的裸行号（上文声明的是本仓测试文件）不得误伤。"""
        context = "# 名称与取值口径见 tests/mcp/test_zero_client_e2e.py。\n# ZERO_FACS_MODEL_PATH="
        assert (
            judge_line("# ZERO_FACS_MODEL_PATH=  # 不设 → 非真模型输出（用例 :276）", context)
            is None
        )

    def test_url_port_is_not_flagged(self) -> None:
        assert judge_line("# ZERO_HTTP_ENDPOINT=http://localhost:8000/mcp  # Zero server") is None

    def test_symbol_form_reference_is_not_flagged(self) -> None:
        """R7 规定的正确写法本身必须绿——否则守卫会逼人写回行号。"""
        assert (
            judge_line("镜像 Zero `src/agents/affect_math.py::expand_external_priors` 的 M7")
            is None
        )


class TestScanIsLive:
    """正控：证明扫描真的读到了本仓内容（防「扫描根写错 → 永远 0 命中」的恒真塌缩）。"""

    def test_required_roots_exist(self) -> None:
        for name in _REQUIRED_DIRS:
            assert (REPO_ROOT / name).is_dir(), f"扫描根 {name}/ 不存在 → 守卫会静默空跑"

    def test_scan_reaches_known_files_and_content(self) -> None:
        files = iter_scan_files()
        rels = {p.relative_to(REPO_ROOT).as_posix() for p in files}
        assert len(files) >= 60, f"仅遍历到 {len(files)} 个文件，扫描根/后缀白名单可能失效"
        for expected in (
            "src/mcp/zero/external_priors.py",
            "tests/mcp/test_zero_client_e2e.py",
            ".env.example",
        ):
            assert expected in rels, f"{expected} 未被遍历到 → 扫描覆盖有洞"
        sentinel = (REPO_ROOT / "src/mcp/zero/external_priors.py").read_text(encoding="utf-8")
        assert "build_external_priors_override" in sentinel, "读到的文件内容不是预期源码"

    def test_ai_docs_is_scanned_when_present(self) -> None:
        """`ai-docs/` 是 git 排除的本地目录：在位时**必须**被扫到（它曾是最易假绿的一块）。"""
        if not (REPO_ROOT / "ai-docs").is_dir():
            return  # worktree 里不存在（本地目录不随 git 走），无可观测对象
        rels = {p.relative_to(REPO_ROOT).as_posix() for p in iter_scan_files()}
        assert any(r.startswith("ai-docs/") for r in rels), "ai-docs/ 在位却未被遍历"


def test_no_cross_repo_line_refs() -> None:
    """全仓不得存在指向 Zero 源码的行号引用。"""
    violations = scan_repo()
    assert not violations, (
        f"发现 {len(violations)} 处指向 Zero 源码的行号引用（跨仓行号腐烂时不驱红，"
        f"故按 R7 一律禁写）。{_FIX_HINT}：\n  " + "\n  ".join(violations)
    )
