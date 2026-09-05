# experiments/

极简实验管理框架：一个 queue + 一组 record，配合两阶段工作流。

## 目录结构

```
experiments/
├── PHASE1_RULES.md               # Phase 1 长期规则 / policy（AI 创建实验必须遵守）
├── PHASE1_PROMPT.md              # Phase 1 启动提示词（直接复制给 Coding Agent）
├── PHASE2_RULES.md               # Phase 2 长期规则 / policy（固定执行规范）
├── PHASE2_SUPERVISOR_PROMPT.md   # Phase 2 Supervisor 启动提示词（复制即用）
├── README.md                     # 本文件
├── queue.yaml                    # 真实实验队列（执行器消费它）
├── runner.py                     # Phase 2 唯一正式执行器
├── records/                      # 所有实验记录，例如 004-market-gated-z1.md
└── templates/                    # 仅保存用于复制生成具体文件的模板
    ├── experiment.template.md    # 单个实验记录模板
    └── queue.template.yaml       # queue.yaml 的格式模板
```

## 各文件职责

规则（RULES）与启动提示词（PROMPT）命名对称、职责分离：

- `PHASE1_RULES.md` = Phase 1 的**长期规则 / policy**：仓库所有权、
  immutable experiment 原则、Final Experiment Commit 流程、入队一致性
  校验。任何 AI 创建新实验前必须完整阅读。
- `PHASE1_PROMPT.md` = Phase 1 的**启动入口**：一个可直接复制给独立
  Coding Agent 的提示词模板，替换 `【】` 中的 Idea 与约束即可发起一个
  新实验的开发。它引用 RULES，不替代 RULES。
- `PHASE2_RULES.md` = Phase 2 的**长期规则 / policy**：固定执行流程、
  Phase 1/2 边界、错误分类（A/B/C）、batch 行为、marker 与 resume 原则。
- `PHASE2_SUPERVISOR_PROMPT.md` = Phase 2 Supervisor 的**启动入口**：
  可直接复制的提示词模板，让 AI 作为 Supervisor 逐实验编排调用
  `runner.py --only <ID>` 并按 A/B/C 分类处置异常。
- `templates/` 只保存真正用于**复制生成具体文件**的模板（record 模板、
  queue 格式模板），不放规则与提示词。

## 核心理念

一个实验只对应：**一个 branch、一份 record、一条 queue 记录**。
进入 queue 的实验是 immutable experiment：一个实验 ID 唯一对应
`Final Experiment Commit + Stage 1 provenance + 固定执行协议 + 正式结果`。

### 第一阶段（AI 开发，需要推理能力）

```
Idea → 独立实验分支 exp/<ID>-<name> → AI 开发 → tests PASS → smoke PASS
     → Final Experiment Commit → 冻结分支
     → 回 main 写实验记录 records/<ID>-<name>.md → queue 追加 pending
     → 校验 queue.commit == record Commit == 分支冻结 HEAD
```

第一阶段需要 AI 的推理能力：理解 Idea、最小化修改、验证可行性。
规则见 `PHASE1_RULES.md`，启动模板见 `PHASE1_PROMPT.md`。

### 第二阶段（固定执行器 + Supervisor）

```
按 ID 顺序消费 pending/running：
逐实验调用 runner --only <ID> → train → predict → backtest
→ collect metrics → 写 Result → done/failed（异常按 A/B/C 分类处置）
```

第二阶段标准化、确定性执行：Runner（`runner.py`）是唯一正式执行器，
不创建实验、不改代码、只做训练与评估，把结果写回 record 和 queue；
AI 只作为 Supervisor 在外层编排与处置异常。
规则见 `PHASE2_RULES.md`，启动模板见 `PHASE2_SUPERVISOR_PROMPT.md`。

## 快速上手

- 想创建新实验：复制 `PHASE1_PROMPT.md` 的模板发起（规则细节见
  `PHASE1_RULES.md`）。
- 想执行正式实验（Phase 2）：复制 `PHASE2_SUPERVISOR_PROMPT.md` 的模板
  发起（规则细节见 `PHASE2_RULES.md`）。
- 想看 queue 格式：参考 `templates/queue.template.yaml`。
- 想写实验记录：复制 `templates/experiment.template.md` 到
  `records/<ID>-<name>.md` 后填写。

## 使用示例

### Phase 1 示例：比较残差量化深度

以下是一个新构想的示例，不表示该实验已创建或入队：

```text
按 PHASE1_PROMPT.md 的模板创建新实验。

构想：在两级 Residual HVQ 的基础上，验证三级残差量化是否改善收益预测。
基于：exp/001-hvq-residual-2level。
唯一实验变量：量化层数从 2 改为 3，
各级 codebook 大小由 [256, 256] 改为 [128, 128, 256]，
总 codebook 条目数保持 512。
Stage 2 使用全部三级量化输出之和。

Stage 1 来源：self，量化结构改变后必须重新训练。
其余配置和协议保持与 001 一致。

自动分配下一个 ID。完成测试和 smoke 后冻结、入队，停止在 pending。
```

这个示例验证的是“固定总码本条目数，比较残差分解深度”。

### Phase 2 示例

处理全部待执行及中断的实验：

```text
按 PHASE2_SUPERVISOR_PROMPT.md 的模板，执行全部 pending/running 实验。
已有 done 保持不动，failed 本次不重试。完成后汇总结果与剩余队列。
```

只执行某个实验（将 `004` 替换为实际已入队的 ID）：

```text
按 PHASE2_SUPERVISOR_PROMPT.md 的模板，仅执行实验 004
（runner --only 004）。若它已 done，则正常跳过。
```

明确重试失败实验：

```text
按 PHASE2_SUPERVISOR_PROMPT.md 的模板，重试 failed 的实验 004
（runner --only 004）。优先利用有效阶段产物，不清空 artifacts，
不修改实验定义。
```

### 推荐使用顺序

提出构想 → Phase 1 冻结并入队 → 累积几个实验 → Phase 2 批量执行
→ 阅读结果 → 用新 ID 验证下一轮构想。

当前执行器会切换共享仓库的检出版本，两个阶段应错开执行。
