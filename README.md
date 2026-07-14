# Zero_MCP

经 **MCP（[Model Context Protocol](https://modelcontextprotocol.io)）**，给情感引擎驱动的 AI 数字人项目 **Zero** 扩展新的 **Agent 能力模块**。Zero_MCP 与 Zero 相对独立：把 Zero 当作**外部服务**经 MCP 调用，不直接耦合其代码库。

首个能力模块「**桌面屏幕感知 + 电脑操控**」已落地并通过真实桌面端到端标定（Task 1-12，见下）；后续能力沿同一架构扩展。

## 已实现能力：桌面屏幕感知 + 操控

让 Agent 感知 Windows 桌面并执行操作，输出**模型无关的结构化文本**（不假设消费方是多模态模型）。

- **感知栈三层降级**：L1 UIA 系统接口（窗口级定位）· L2 RapidOCR（文本主通道）· L3 OpenCV 模板匹配 + 可选 OmniParser（视觉）。**零训练**（不自训任何模型），**GPU 可选**（CPU 全功能兜底，ONNX EP 自适应 CUDA>DirectML>CPU）。
- **定向后台感知**：`screen_snapshot(window_handle=)` 指定目标窗口，经 `PrintWindow(PW_RENDERFULLCONTENT)` 直取窗口自身 DWM 渲染面——**被遮挡/无焦点/压 z 序底均可感知，不打扰正在用机的用户**；坐标恒为屏幕绝对（可直接驱动点击）。
- **关键实证**：微信 4.x（mmui 自绘）与钉钉 7.x 的 UIA 内容树实测覆盖率均 **≈0%**（钉钉 7/7 界面 hollow）——故感知默认 OCR 主通道、操控主路径为坐标点击，逐窗口探测 `uia_hollow` 自动切 OCR（[微信 PoC](notes/poc-uia-coverage-result.md) · [钉钉 e2e 标定](notes/e2e-desktop-task-results.md)）。
- **LangGraph 编排**：Supervisor / StateGraph，高危动作经 `interrupt` 人工确认门，停滞检测 + 断点续跑（Checkpointer）。
- **安全护栏在编排层**：三级白名单 + TOCTOU 二次截图比对（目标局部裁剪口径）+ 屏幕文字注入过滤（含 23 条中文越权词表，实测 FP=0/159；不依赖模型层防御）。

交付门禁：**422 单测 + 53 行为 evals 全绿**，ruff + mypy strict 通过，独立 code-reviewer 审查 PASS；真实钉钉桌面端到端标定（感知成功率 7/7，OCR conf 均值 0.954，快照延迟 ~1.2s）。实施全程见 [Task 1-11 实施记录](notes/2026-07-10-task1-11-implementation-log.md) · [Task 12 e2e 标定](notes/e2e-desktop-task-results.md)。

## 架构

三层单向依赖，叠加 MCP 互操作边界（依赖只能自上而下）：

```text
编排层  src/orchestration, src/agents   （LangGraph StateGraph / Supervisor / 安全门）
   ↓
记忆层  src/memory                       （长期记忆 / 图谱读写封装；当前 Protocol 打桩）
   ↓
存储层  src/storage                      （Postgres / Neo4j / Redis；待建）

MCP 互操作边界（横切）
  · src/mcp        Python MCP server + client：内部能力封装（感知/操控原语，官方 mcp python-sdk，stdio）
      └ zero/      Zero↔MCP 情感对接边界契约层：(v,a) 刺激注入 + expression 消费骨架（不 import Zero，协议结构化镜像）
  · mcp-server/    TypeScript MCP 服务层：对外聚合层，向 Zero host 等外部方暴露能力（本能力模块内部不使用）
```

- **运行态**（LangGraph Checkpointer）与**长期记忆**分离存储。
- **MCP 传输层不塞业务逻辑**：server 只注册工具 + 转发原语，感知/agent 业务逻辑留在 Python `src/*`。
- **MCP 层语言判据**：TS = 对外互操作边界；Python = 内部能力封装。感知库（pywinauto/mss/RapidOCR）是 Python 原语，故 desktop 感知 server 用 Python 直连自建，不加 TS 桥（决策依据见 [CLAUDE.md](CLAUDE.md) MCP 层语言澄清）。

## 技术栈

| 侧 | 语言 | 用途 |
| --- | --- | --- |
| 主 | Python 3.12 | LLM agent 框架（LangGraph）+ 编排/记忆 + MCP server/client + 感知/操控原语 |
| MCP 对外层 | TypeScript | `mcp-server/`，`@modelcontextprotocol/sdk`（对外聚合，本模块未用） |

核心依赖：`langgraph` · `mcp`（python-sdk）· `pydantic` · `mss` · `rapidocr-onnxruntime` · `pywinauto`/`uiautomation` · `pyautogui` · `jinja2`。GPU 加速走 extras（`gpu-cuda`/`gpu-dml`/`omniparser`），CPU 为默认兜底。

## 目录结构

```text
src/
  orchestration/   Supervisor / StateGraph / 安全门（safety/）/ prompt 模板    ✅ 已实现
    safety/          三级白名单 + TOCTOU + 注入过滤
    prompts/         Jinja2 模板（与代码分离）
  agents/          ScreenPerceptionAgent / DesktopControlAgent / 契约模型      ✅ 已实现
    models/          screen_snapshot（桌面契约）/ zero_affect（Zero↔MCP 情感契约）
  mcp/             Python MCP server + client + 感知/操控原语                   ✅ 已实现
    desktop/         capability_probe / tools（perception, control）
    zero/            Zero↔MCP 边界契约层：感知(v,a)注入 + expression 消费骨架   ✅ 第一阶段
  memory/          记忆读写 API（当前经 Protocol 打桩，待实现）                 ⬜ 骨架
  storage/         Postgres / Neo4j / Redis 连接与 schema                       ⬜ 骨架
mcp-server/        TS MCP 对外聚合层（独立 package.json，本模块未用）            ⬜ 骨架
tests/             单测（poc/mcp/agents/orchestration/safety）+ 行为回归        ✅ 563 绿
evals/             agent 行为级 evals                                          ✅ 53 绿
ai-docs/           知识库：模块三件套 / catalog / pitfalls
notes/             设计纪要 / 实测报告 / 工程实施记录
```

## 开发

```bash
# Python 侧（复用 conda 环境 affective-expression）
conda activate affective-expression
uv pip install -e ".[dev]"          # CPU 全功能；GPU 加速用 .[gpu-cuda,dev] 或 .[gpu-dml,dev]
pytest tests/ -m "not realenv"      # 单测（realenv 用例需真实桌面，本地手动跑）
ruff check . && mypy

# MCP 对外服务层（TS，本能力模块未用）
cd mcp-server && npm install && npm run typecheck
```

配置与密钥走 `.env`（复制 `.env.example` 填值，已 gitignore 不入库）。屏幕能力默认关：`SCREEN_CAPABILITY_ENABLED=false`。

## 状态

- ✅ **Task 1-11 已实现收口**：屏幕感知/操控 MCP 层 + agents + LangGraph 编排 + 安全门（code-reviewer PASS）。
- ✅ **Task 12 端到端标定完成**：真实钉钉桌面标定全部 gap（成功率/OCR 精度/UIA 覆盖/TOCTOU/注入 FP/延迟），修复 3 个 e2e 独有 bug（OCR 全屏污染、RapidOCR 引擎重建 9 倍延迟、rect 契约），新增 PrintWindow 定向后台感知。
- ✅ **Task 13 DPI/多显示器坐标一致性完成**：双屏 × 双感知路径 × 12 点 PoC 全过（11 点 0.0px），mss 全虚拟屏修复消除副屏盲区（`monitors[1]≠主屏` 实测证伪）、支持负虚拟屏原点。
- ✅ **Task 14 异常现场上报完成**：`FileIncidentReporter` 落盘现场包（incident.json + 截图副本 + 最近 N 步流程），四级降级永不吞包；feature-flag `INCIDENT_DIR` 默认关零回归。
- ⬜ **Task 15 待推进**：录制→回放（成本优化后期）。
- ⬜ **memory / storage 层**：当前 Protocol 打桩，待随记忆/存储模块实现落地。

## 深入阅读

- 能力设计蓝图：[notes/2026-07-10-screen-capability-blueprint.md](notes/2026-07-10-screen-capability-blueprint.md)
- 工程实施记录：[notes/2026-07-10-task1-11-implementation-log.md](notes/2026-07-10-task1-11-implementation-log.md)
- e2e 实测标定（Task 12，含「Win32 状态不可信链」与敌对桌面守卫）：[notes/e2e-desktop-task-results.md](notes/e2e-desktop-task-results.md)
- 模块文档（编辑前必读）：[ai-docs/docs/modules/](ai-docs/docs/modules/)（mcp / orchestration / agents 三件套）
- 知识总目录：[ai-docs/docs/catalog.md](ai-docs/docs/catalog.md) · 踩坑记录：[ai-docs/pitfalls.md](ai-docs/pitfalls.md)

> 说明：本仓库仅跟踪 Zero_MCP 工程代码（`src/` · `tests/` · 配置）。项目自用的 Claude Code harness（`.claude/`）、知识库（`ai-docs/`）与设计纪要（`notes/`）经 `.git/info/exclude` 本地排除，不随本仓库提交，仅在本地开发环境维护。
