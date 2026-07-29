# Zero_MCP

经 **MCP（[Model Context Protocol](https://modelcontextprotocol.io)）**，给情感引擎驱动的 AI 数字人项目 **Zero** 扩展新的 **Agent 能力模块**。Zero_MCP 与 Zero 相对独立：把 Zero 当作**外部服务**经 MCP 调用，不直接耦合其代码库。

已落地两条能力线：**「桌面屏幕感知 + 电脑操控」**（Task 1-14 收口，真实桌面端到端标定）与 **「Zero↔MCP 情感对接（zero-link）」**（第二阶段：契约层 + client 真连 + 三模态感知真接入 + 真采集 I/O 适配层 + 表达映射器 + physiology 消费门控 + EDA 唤醒度量 v2）。另已落地**记忆层 + 存储层与编排层的持久化接线**（ScopedMemoryAPI + SQLite 后端 + 组装根，默认关零回归）。后续能力沿同一架构扩展。

## 能力一：桌面屏幕感知 + 操控

让 Agent 感知 Windows 桌面并执行操作，输出**模型无关的结构化文本**（不假设消费方是多模态模型）。

- **感知栈：两层可用 + 一层预留**。L1 UIA 系统接口（控件树，窗口级定位）· L2 RapidOCR（文本主通道，**实际主力**）· L3 视觉**尚未开工**（`visual_objects` 恒为空列表，OmniParser 只做能力探测不跑推理，OpenCV 模板匹配无任何实现）。**零训练**（不自训任何模型）；**GPU 可选**——启动期按 CUDA>DirectML>CPU 探测 ONNX EP 并写进 `capability_flags`，但该选择当前只用于门控 OmniParser 与日志，**OCR 推理路径尚未消费它**（CPU 全功能兜底）。
- **定向后台感知**：`screen_snapshot(window_handle=)` 指定目标窗口，经 `PrintWindow(PW_RENDERFULLCONTENT)` 直取窗口自身 DWM 渲染面——**被遮挡/无焦点/压 z 序底均可感知，不打扰正在用机的用户**；坐标恒为屏幕绝对（可直接驱动点击）。
- **关键实证**：微信 4.1.11.24（mmui 自绘）与钉钉 7.x 的 UIA 内容树实测覆盖率均 **≈0%**（钉钉 7/7 界面 hollow）——故感知默认 OCR 主通道、操控主路径为坐标点击，逐窗口探测 `uia_hollow` 自动切 OCR。
- **LangGraph 编排**：7 节点 StateGraph + 3 条件边，高危动作经 `interrupt` 人工确认门，三信号停滞检测 + 断点续跑（Checkpointer）。
- **安全护栏在编排层**：三级白名单（不在表内一律升 DESTRUCTIVE）+ TOCTOU 二次截图比对（动作坐标 ±`TOCTOU_CROP_HALF_PX` 局部裁剪口径）+ 屏幕文字注入过滤（12 条英文 + 23 条中文越权词表，实测 FP=0/159）；不依赖模型层防御。

感知 → 安全门 → 操控 → 停滞/续跑的一轮闭环：

<img src="docs/v2/desktop-pipeline.png" alt="桌面感知+操控管线：UIA/OCR 两层可用+视觉层预留，经注入过滤产出 ScreenSnapshot；操控侧三级白名单→TOCTOU 二次截图→interrupt 人工确认门才放行；停滞检测与 Checkpointer 断点续跑" width="900">

> 图源 [`docs/v2/desktop-pipeline.mmd`](docs/v2/desktop-pipeline.mmd)（mermaid，飞书画板渲染）。

交付质量：单测 + 53 条行为 evals 全绿，ruff + mypy strict 通过，独立 code-reviewer 审查 PASS；真实钉钉 7 个功能界面端到端标定 **7/7** 全通过，OCR conf（各界面均值再平均）0.954，快照延迟 0.72–2.03s（均值 ≈1.2s）。⚠ 后三项是一次性标定记录（依据在本地 `notes/e2e-desktop-task-results.md`，不入库），**不是 CI 持续回归项**。

## 能力二：Zero↔MCP 情感对接（zero-link）

把 Zero 的情感/认知内核当**外部服务**经 MCP 对接：输入端把多模态感知归一成 `(valence, arousal)` 先验注入内核，输出端消费内核每轮的 `expression` 字典驱动渲染。**不 import Zero 代码库**——协议结构化镜像 + 跨仓活体回归（pytest marker `zerorepo`，`D:\Zero` 不在位自动 skip）。

- **边界契约层**：跨层数据形状唯一真相 `src/agents/models/zero_affect.py`（pydantic，字段/量纲逐条对 Zero 源码现场核验）——`(v,a)[+coping]` 刺激、多模态**独立先验流**（禁均值，竞争融合归内核）、`spontaneous`/`voluntary` 双通路 × 4 通道 expression（13 维 FACS〔12 AU + intensity 强度标量〕/ 韵律 / 生理 / 标签）。
- **Zero MCP client**（`ZeroLinkClient`）：会话句柄三段式 `zero.open_session / step / close_session` + `graceful_step` 优雅回退；stdio（本地子进程）与 Streamable HTTP（远程）双传输经 `.env` 切换，HTTP 侧 Bearer 鉴权、跨重启 resume、unknown-session 机读区分均已对齐 Zero 上线形态。**已与 `D:\Zero` 真 MCP server 端到端联调通过**（stdio + HTTP + 真 13 维 FACS 权重全契约验证）。
- **三模态感知真接入**（算法团队文献门选型，全程可点击引文）：生理 **EDA + HRV**（EDA = SCL 窗间基线 Δ〔纯 numpy，不依赖 NeuroKit2〕· HRV = NeuroKit2 RMSSD，二者经 ω=0.5 协方差交叉预合并为单条 `physio` 流）· 语音 audeering w2v2 维度 SER · 视觉 EmotiEffLib **ONNX 后端**（零 timm，不动共用 conda 环境）。全部 pip 直装、CPU 可跑、零自训；缺库/缺模型/无输入一律 warning + 单通道降级，不拖垮整体。
- **EDA 唤醒度量 v2**（2026-07-29 起**唯一路径**，无 env 开关）：`scl_baseline_delta` = 窗内原始 SCL 均值 − 近 30 分钟各窗 SCL 中位数基线 → 对称归一化。**不做 phasic 分解**（实测窗内 `mean(eda)` 与 `EDA_Tonic.mean()` 相对差 <0.1%），连带消除 cvxEDA/highpass 双分支这个跨采样率失败成因。v1 的 `scr_amplitude` 经 WESAD 真被试实测与唤醒**系统性反相关**（「stress>baseline」正确率 1/5，经典 SCL 是 4/5），已连同 `ZERO_EDA_AROUSAL_METRIC` 开关一并删除。⚠ **冷启动约 4.5 分钟返回 `None`**（无基线证据不臆造读数），期间生理流有意降级为裸 `hrv/rmssd`。⚠ 两通道成熟度不对称：EDA 经 WESAD 全会话回放验收，**HrvChannel 仍是单一固定 ref、零跨被试自适应的初版**（判别力正常但抗漂移不及格，待独立立项改造）。
- **真采集 I/O 适配层**（`io_adapters/`）：把本地文件 / 合成信号包装成 async `signal_source` 注入四通道——`make_audio_file_source`（librosa 16kHz mono）· `make_vision_file_source`（FaceDetectorYN 人脸裁剪，BGR→RGB）· `make_synthetic_eda/hrv_source`（NeuroKit2）；**真硬件已实现**（`hardware_adapters`）：`make_mic_source`（sounddevice）· `make_camera_source`（cv2 VideoCapture + 可选 YuNet 人脸裁剪，即开即关不占设备）· `make_wearable_source`（pyserial 串口）——依赖为可选 extra（`hardware-audio` / `hardware-wearable`，默认不装），**工厂构造永不抛**，缺库/无设备/读取失败一律回退 `None`。适配层经构造器注入、**不进 Channel 核心**、脱硬件可测。
- **表达映射器**（引擎无关）：`ArkitFacsMapper`（12 AU → ARKit 标准 52 blendshape 命名空间中的 21 个，intensity 作全局增益，稀疏输出）· `LinearProsodyMapper`（韵律→倍率/半音/dB，可出 SSML `<prosody>`）· `LinearPhysiologyMapper`（**WESAD canonical**：心率/呼吸 + 皮肤电导 μS 归一 + 体温 °C）；经 `RenderingExpressionSink` 产出 `RenderFrame`（`DUAL` 策略含 spontaneous 微表情泄漏帧——"真笑/假笑"表现力来源）。
- **physiology 契约 = WESAD 真信号**：canonical 三键 `{heart_rate_bpm, skin_conductance(μS[0,20]), temperature_c(°C)}`。跨仓迁移不原子，**保超集不收窄**（2026-07-23 定论、非过渡态：`hr`+`sc` 必填，`temperature_c`/`pupil_mm` **恒可选**）——收到 Zero 的 legacy 或 canonical 两形状都不 `ValidationError`。⚠ **保超集 = 解析层零回归 ≠ 消费标度正确**：mapper 按 μS 标定，连 legacy 门关 Zero（sc∈[0,1]）须显式配 `skin_conductance_max_us=1.0`，否则默认再除 20 → 静默欠标度 ~20×（对抗审查揪出的 W6）。
- **出境守卫**：`external_priors` 多流逐维 `(Πv,Πa)` 精度载荷 + 客户端 fail-fast——M3 精度上界 / M6 流数上界（按**合并后**计数）/ M7 μ 域校验 / Π 有限性校验（`isfinite` 先于上界判定，堵 NaN 静默穿透），M5 schema 版本跨仓断言（`zerorepo` 测试期）。校验装在**出网函数**、作用于最终出线的值——构造期校验挡不住赋值/`model_construct`/鸭子类型四条绕过路径。
- **默认关零回归**：`ZERO_LINK_ENABLED` 与三个感知通道 flag 全部默认 `false`，不影响既有能力。

zero-link 一轮数据流（感知先验注入 → Zero 确定性计算 → 表达消费驱动渲染）：

<img src="docs/v2/zero-link-dataflow.png" alt="zero-link 一轮数据流：EDA/HRV 经 ω=0.5 协方差交叉预合并 + audio/vision 独立先验 → PerceptionHub 多流独立 → external_priors 出境守卫 → ZeroLinkClient → Zero 确定性情感计算 → ExpressionRouter 双通路 → 三映射器 → RenderFrame" width="900">

> 图源 [`docs/v2/zero-link-dataflow.mmd`](docs/v2/zero-link-dataflow.mmd)（mermaid，飞书画板渲染）。

## 记忆层与存储层（持久化接线）

- **记忆层** `src/memory/api.py::ScopedMemoryAPI`：四条记忆纪律**代码强制**——显式 scope 两级 fail-fast（禁默认 user）· 每任务一条摘要（写入按任务完成节流）· 与运行态物理分表 · 新事实使旧事实失效（打戳不删行）。未做：自动抽取、跨事实去重、向量检索。
- **存储层** `src/storage/`：默认后端 **SQLite（aiosqlite）**，零 infra、可离线、支持 `:memory:`；`snapshots`（运行态快照）与 `memory_facts`（长期记忆）**物理分表**。Postgres Checkpointer / Neo4j / Graphiti 为平行后端扩展点，未建。
- **组装根** `src/orchestration/persistence.py::persistent_stores()`：依赖注入里**唯一**允许出现具体后端类型的地方，节点与 Agent 仍只见 Protocol。按 `ZERO_MCP_PERSISTENCE_DB` 决定接真后端还是退 `NoopMemoryAPI` 打桩（**不设 = 关，零回归**）；一旦开启，`ZERO_MCP_MEMORY_SCOPE_KEY` 必填、缺失即启动 fail-fast（刻意不静默退打桩）。

## 架构

三层单向依赖，叠加 MCP 互操作边界（依赖只能自上而下）：

<img src="docs/v2/mcp-architecture.png" alt="三层单向依赖 + MCP 互操作边界：编排层→记忆层→存储层只能自上而下调用，组装根 persistence.py 注入真实后端；MCP 边界（src/mcp Python 内部封装 + zero-link，mcp-server/ TS 对外聚合）把 Zero 当外部服务经 MCP 调用，不 import 其代码库" width="760">

> 图源 [`docs/v2/mcp-architecture.mmd`](docs/v2/mcp-architecture.mmd)（mermaid，飞书画板渲染）。

- **运行态**（LangGraph Checkpointer）与**长期记忆**分离存储——已在 SQLite 后端落为 `snapshots` / `memory_facts` 两张物理分表。
- **MCP 传输层不塞业务逻辑**：server 只注册工具 + 转发原语，感知/agent 业务逻辑留在 Python `src/*`。
- **MCP 层语言判据**：TS = 对外互操作边界；Python = 内部能力封装。感知库（pywinauto/mss/RapidOCR/NeuroKit2 等）是 Python 原语，故内部能力 server/client 用 Python 直连自建，不加 TS 桥。

## 技术栈

| 侧 | 语言 | 用途 |
| --- | --- | --- |
| 主 | Python 3.12 | LLM agent 框架（LangGraph）+ 编排/记忆/存储 + MCP server/client + 感知/操控原语 |
| MCP 对外层 | TypeScript | `mcp-server/`，`@modelcontextprotocol/sdk`（对外聚合，暂未用） |

核心依赖：`langgraph` · `mcp`（python-sdk）· `pydantic` · `httpx` · `mss` · `rapidocr-onnxruntime` · `pywinauto`/`uiautomation` · `pyautogui` · `jinja2`；存储层另需 `aiosqlite`（延迟 import，缺库时报带指引的 ImportError）。可选 extras：`dev`/`poc` · GPU 加速 `gpu-cuda`/`gpu-dml`/`omniparser`（CPU 为默认兜底）· zero-link 感知 `physio`（NeuroKit2）/`perception-audio`（torch/transformers）/`perception-vision`（emotiefflib ONNX）· 真硬件采集 `hardware-audio`（sounddevice）/`hardware-wearable`（pyserial）。⚠ 文件适配层用到的 `librosa` 当前未在任何 extra 声明，需手动装。

## 目录结构

```text
Zero_MCP/
├── src/
│   ├── orchestration/              # 编排层：LangGraph 装配 + Supervisor + 安全门 + 组装根
│   │   ├── desktop_graph.py            #   build 桌面任务 StateGraph：7 节点 + 3 条件边 + interrupt 断点续跑
│   │   ├── desktop_supervisor.py       #   Supervisor：LLM 决策下一步（只分发协调不含业务，模型 ID 走 .env）
│   │   ├── state.py                    #   结构化 state（pydantic）：节点只返回增量，大对象放引用
│   │   ├── protocols.py                #   注入契约：MemoryAPI / SnapshotStore / IncidentReporter + Noop 打桩
│   │   ├── persistence.py              #   组装根：按 env 把 memory/storage 真实现接进图；未配则退打桩（零回归）
│   │   ├── phash.py                    #   感知哈希：TOCTOU 比对 + 停滞检测信号（目标局部裁剪口径）
│   │   ├── prompt_loader.py · prompts/ #   Jinja2 模板加载 + supervisor 模板（prompt 与代码分离）
│   │   └── safety/                     #   安全门
│   │       ├── action_guard.py             #     三级白名单 + TOCTOU 二次截图比对 + interrupt 人工确认
│   │       └── incident_reporter.py        #     异常现场包落盘 incident.json+截图（INCIDENT_DIR 默认关）
│   ├── agents/                     # Worker Agent 层（职责单一，经 state + Supervisor 协作）
│   │   ├── screen_perception_agent.py  #   屏幕感知：快照 → 模型无关结构化文本
│   │   ├── desktop_control_agent.py    #   桌面操控：动作规划与执行封装
│   │   ├── text_filter.py              #   屏幕文字注入过滤（12 英文 + 23 中文越权词表，实测 FP=0/159）
│   │   ├── protocols.py                #   agent 层协议（SnapshotStore 权威定义在此，编排层只 re-export）
│   │   └── models/                     #   共享契约层（跨层数据形状唯一真相，pydantic）
│   │       ├── screen_snapshot.py          #     桌面感知契约（ScreenSnapshot / TextBlock / capture_origin）
│   │       └── zero_affect.py              #     Zero↔MCP 情感契约（(v,a) 刺激 / 先验流 / 13 维 FACS 双通路 expression）
│   ├── mcp/                        # MCP 互操作边界（Python = 内部能力封装）
│   │   ├── desktop_mcp_server.py       #   桌面能力 MCP server（FastMCP + stdio，10 工具，flag 默认关）
│   │   ├── desktop_mcp_client.py       #   编排层侧 client（spawn 子进程，异常三件套）
│   │   ├── desktop/
│   │   │   ├── capability_probe.py         #     启动能力探测：GPU EP 自适应 / OCR / OmniParser（幂等缓存）
│   │   │   └── tools/                      #     perception.py 感知原语（UIA/mss/RapidOCR/PrintWindow）· control.py 操控原语
│   │   └── zero/                       #   zero-link：Zero↔MCP 情感对接层（不 import Zero，协议镜像）
│   │       ├── client.py                   #     ZeroLinkClient：三段式会话 + stdio/HTTP + Bearer + resume + graceful_step
│   │       ├── perception.py               #     PerceptionHub 多流汇聚（禁均值）+ prepare_all 串行预热重依赖
│   │       ├── expression_sink.py          #     ExpressionRouter：step_out 解析 + HeadPolicy 双通路分发
│   │       ├── external_priors.py          #     出境载荷构造 + M3/M6/M7/有限性 fail-fast + EDA/HRV ω=0.5 预合并
│   │       ├── protocols.py                #     Zero 协议结构化镜像（runtime_checkable，挂 path:line 证据）
│   │       ├── channels/                   #     感知通道：physio(EDA 纯 numpy / HRV NeuroKit2) / audio / vision / callable
│   │       ├── io_adapters/                #     真采集 I/O 适配层：文件/合成 signal_source + hardware_adapters 真硬件
│   │       ├── mappers/                    #     表达映射：facs(12 AU→ARKit) / prosody(→SSML) / physiology(WESAD μS/°C)
│   │       └── sinks/                      #     rendering.py：RenderingExpressionSink → RenderFrame（DUAL 含微表情泄漏帧）
│   ├── memory/                     # 记忆层
│   │   └── api.py                      #   ScopedMemoryAPI：显式 scope · 任务完成节流 · 新事实使旧失效（只调存储层）
│   └── storage/                    # 存储层（SQLite 已落地；PG/Neo4j 为平行扩展点）
│       ├── sqlite_backend.py           #   aiosqlite 连接 + schema（snapshots / memory_facts 物理分表）
│       ├── memory_store.py             #   长期记忆事实读写（invalidated_at 时序失效，不物理删）
│       └── snapshot_store.py           #   运行态快照存取（同 ID 幂等覆盖，load 缺失抛 KeyError）
├── mcp-server/                     # TS MCP 对外聚合层（@modelcontextprotocol/sdk，独立 package.json，暂未用）
├── docs/v2/                        # 架构图：*.mmd（mermaid 源）+ *.png（飞书画板渲染，随仓提交，README 内嵌）
├── tests/                          # 约 1.2k 用例（以 pytest --collect-only 为准）：agents/mcp/memory/orchestration/
│                                   #   safety/storage/poc/e2e（marker：realenv 实机 · zerorepo 跨仓回归 45 条）
├── evals/                          # 53 条 agent 行为级 evals（感知/操控/停滞）+ 8 个 WESAD 真被试生理度量脚本
├── ai-docs/                        # 知识库：模块三件套 + catalog + pitfalls + engineering-practices（本地维护，不入库）
├── notes/                          # 设计纪要 / 实测报告 / 跨仓对齐总账（本地，不入库）
├── .env.example                    # 配置模板（cp 为 .env 启用；所有能力 flag 默认关）
├── pyproject.toml                  # 依赖与工具链（uv 装包；10 个 extras，见「技术栈」）
└── environment.yml                 # conda 环境声明（复用 D:\Zero 的 affective-expression，勿 prune）
```

## 开发

```bash
# Python 侧（复用 conda 环境 affective-expression）
conda activate affective-expression
uv pip install -e ".[dev]"          # CPU 全功能；按需加 extras：.[gpu-cuda,dev]、.[physio,perception-audio,perception-vision,dev]
pytest tests/ -m "not realenv"      # 单测（realenv 需真实桌面本地手动跑；zerorepo 缺 D:\Zero 自动 skip）
ZERO_LINK_E2E_STRICT=1 pytest -m zerorepo   # 跨仓对齐/发版前：zerorepo 的任何 skip 转 fail，防覆盖静默归零
ruff check . && mypy

# MCP 对外服务层（TS，暂未用）
cd mcp-server && npm install && npm run typecheck
```

配置与密钥走 `.env`（复制 `.env.example` 填值，已 gitignore 不入库）。所有能力默认关零回归：屏幕能力 `SCREEN_CAPABILITY_ENABLED=false`、Zero 对接 `ZERO_LINK_ENABLED=false`、三感知通道 `ZERO_{PHYSIO,AUDIO,VISION}_CHANNEL_ENABLED=false`、本仓持久化 `ZERO_MCP_PERSISTENCE_DB` 不设即关（设了则 `ZERO_MCP_MEMORY_SCOPE_KEY` 必填、缺失启动 fail-fast）；zero-link 传输/模型/精度旋钮全走 `.env`（见 `.env.example` 注释）。

> 说明：本仓库仅跟踪 Zero_MCP 工程代码（`src/` · `tests/` · `docs/` · 配置）。项目自用的 Claude Code harness（`.claude/` · `CLAUDE.md`）、知识库（`ai-docs/`）、设计纪要（`notes/`）、PRP 工作区（`PRP/` · `ai-shared/`）、行为 evals（`evals/`）与交接文档（`HANDOFF.md`）经 `.git/info/exclude` 本地排除，不随本仓库提交，仅在本地开发环境维护——README 中引用它们的实测数字（FP=0/159、7/7、conf 0.954 等）因此无法从克隆件回溯。
