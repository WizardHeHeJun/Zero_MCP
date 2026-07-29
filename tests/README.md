# tests

Zero_MCP 的测试：Python 侧 `pytest`（节点契约、边界、默认关零回归、跨仓契约回归）。TS 侧 MCP 服务层测试在 `mcp-server/`（vitest / `node:test`，随该层开工再补）。

约 1.2k 条用例（精确数以 `pytest --collect-only -q` 为准，不要在文档里钉死数字——每次合并都会漂）。

## 目录

| 目录 | 覆盖 |
| --- | --- |
| `agents/` | Worker Agent 层：屏幕感知 / 桌面操控 / 注入过滤 / `models/zero_affect` 契约模型 |
| `mcp/` | MCP 边界：桌面 server 注册面与 client、能力探测、感知/操控原语；`test_zero_*` 为 zero-link 契约面（24 个文件） |
| `orchestration/` | 编排层：图装配与路由函数、Supervisor、prompt 模板、phash/停滞、现场包、持久化接线 |
| `memory/` · `storage/` | 记忆层 `ScopedMemoryAPI` 四条硬约束；存储层 SQLite 两表物理分离与时序失效 |
| `safety/` | 安全门：三级白名单、TOCTOU 比对、人工确认门 |
| `poc/` | 早期 PoC harness（UIA 覆盖率、DPI 坐标），非常规回归 |
| `e2e/` | 桌面端到端（多数需真实桌面，标 `realenv`） |

## marker（定义在 `pyproject.toml`）

| marker | 含义 | 默认行为 |
| --- | --- | --- |
| `realenv` | 需要真实 Windows 桌面/窗口 | 本地手动跑，CI 用 `-m "not realenv"` 排除 |
| `zerorepo` | 跨仓活体回归，需 `D:\Zero` 在位 | 仓不在位自动 `skip`，不拖红套件 |

`zerorepo` 现标在 13 个测试类上、展开 **45 条**用例（`test_zero_contract_crosscheck.py` 34 · `test_zero_client_e2e.py` 9 · `test_zero_perception_e2e.py` 2）。另有两个**刻意不标 `zerorepo`** 的集成文件（避免被 `-m "not zerorepo"` 误剔除，见各文件头注释）：`test_zero_client_smoke.py` 不带任何 marker、恒跑；`test_zero_client_integration.py` 里真 spawn 子进程的 4 条标了 `realenv`，默认命令下会被剔除。

## 跑

```bash
conda activate affective-expression          # 与 D:\Zero 共用的环境，勿 --prune
pytest tests/ -m "not realenv"               # 常规
ZERO_LINK_E2E_STRICT=1 pytest -m zerorepo    # 跨仓对齐 / 发版前
```

**`ZERO_LINK_E2E_STRICT` 是覆盖归零守卫**（实现见 `tests/mcp/conftest.py`）：开启后 `zerorepo` 用例的**任何 skip 一律转 fail**，覆盖 setup/call/teardown 三阶段。它防的是这样一类失效——跨仓依赖悄悄失位（Zero 仓路径变了、导入方式变了、marker 判据不再命中），套件表面全绿，其实**一条跨仓断言都没真跑**。用 `pytest_runtest_makereport` 钩子集中改判，因此那三个文件里的 110 余处 `pytest.skip()` 调用点零改动。

## 约定

- 断言**行为**而非「测试绿」：新写守卫先回答「什么改动应该让它红」，再实证由绿转红（做法见 `ai-docs/docs/engineering-practices.md`，反面案例见 `ai-docs/pitfalls.md`「绿灯必须先证明它能红」）。
- 重依赖用 `pytest.importorskip`，但**不要放在模块级**——那会连带跳过整个文件里的无关用例，形成假通过（同见 pitfalls）。
- 跨仓 pin 优先**读对方源码**而非手抄镜像；同一条守卫内不要读源与硬编码混用。
- 默认关的能力必须有「flag 关时不生效」的零回归用例。
