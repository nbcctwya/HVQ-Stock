# experiments/

极简实验管理框架：一个 queue + 一组 record，配合两阶段工作流。

## 目录结构

```
experiments/
├── queue.yaml                      # 真实实验队列（执行器消费它）
├── templates/
│   ├── queue.template.yaml         # queue.yaml 的格式模板
│   ├── experiment.template.md      # 单个实验记录模板
│   └── RULES.md                    # AI 创建/管理实验必须遵守的规则
└── records/                        # 所有实验记录，例如 004-market-gated-z1.md
```

## 核心理念

一个实验只对应：**一个 branch、一份 record、一条 queue 记录**。

### 第一阶段（AI 开发，需要推理能力）

```
Idea → 独立实验分支 exp/<ID>-<name> → AI 开发 → smoke PASS
     → 写实验记录 records/<ID>-<name>.md → queue 追加 pending
```

第一阶段需要 AI 的推理能力：理解 Idea、最小化修改、验证可行性。
详细规则见 `templates/RULES.md`。

### 第二阶段（固定执行器，标准化执行）

```
按 ID 顺序消费 pending：
train → predict → backtest → collect metrics → 写 Result → done/failed
```

第二阶段以后应尽量标准化、确定性执行：不创建实验、不改代码、
只做训练与评估，把结果写回对应的 record 和 queue。
执行器为 `experiments/runner.py`，完整执行规范见
`experiments/PHASE2_RULES.md`。

## 快速上手

- 想创建新实验：先读 `templates/RULES.md`。
- 想执行正式实验（Phase 2）：先读 `PHASE2_RULES.md`。
- 想看 queue 格式：参考 `templates/queue.template.yaml`。
- 想写实验记录：复制 `templates/experiment.template.md` 到
  `records/<ID>-<name>.md` 后填写。
