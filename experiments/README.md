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

## 使用模板与示例

以下模板可以直接复制给 AI，替换 `【】` 中的内容后使用。
Phase 1 的交付物是冻结并入队的实验；Phase 2 的交付物是正式结果和更新后的队列。
模板中的项目路径均相对于 HVQ-Stock 仓库根目录。

### Phase 1：开发实验模板

```text
在 HVQ-Stock 中执行 Phase 1，开发一个新实验。

实验构想：
【描述模型改动及要验证的假设】

基于：
【main，或明确指定 exp/<ID>-<name>】

实验约束：
- 唯一实验变量：【本次允许改变的内容】
- 数据划分、训练预算、seed、回测协议及其他超参数保持与 base 一致。
- Stage 1 来源：【self / 复用实验 ID / external + checkpoint 路径】
  若复用，先验证结构、配置、数据划分与 checkpoint 兼容性。

执行要求：
1. 完整阅读 experiments/templates/RULES.md。
2. 检查当前 queue、已有 records、分支和工作区，分配下一个实验 ID。
3. 先创建实验分支，再修改代码。
4. 默认 config 必须完整代表本实验，不能依靠实验特有 CLI override 开启。
5. 添加必要测试，运行测试和最小 smoke；失败则修复后重试。
6. smoke 产物统一放在 artifacts/<ID>/smoke，避免覆盖其他实验。
7. 更新实验分支 README，说明构想、改动、与 base 的区别和 smoke 结果。
8. tests 和 smoke 通过后，提交 Final Experiment Commit 并冻结分支。
9. 回到 main，创建 record，追加 status: pending 的 queue 条目。
10. 验证 queue.commit、record Commit、实验分支 HEAD 为同一个完整 SHA。

本次只执行 Phase 1，不启动正式长时间训练。

完成后汇报：
- 实验 ID、base、branch、Final Experiment Commit
- 核心改动与保持不变的条件
- 测试和 smoke 结果、日志及产物位置
- Stage 1 来源
- 入队一致性检查结果
```

### Phase 1 示例：比较残差量化深度

以下是一个新构想的示例，不表示该实验已创建或入队。

```text
按上述 Phase 1 模板创建新实验。

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

### Phase 2：执行队列模板

```text
在 HVQ-Stock 中执行 Phase 2，处理正式实验队列。

执行范围：
【全部 pending/running，或指定实验 ID】

执行要求：
1. 完整阅读 experiments/PHASE2_RULES.md。
2. 启动前确认没有其他 runner 占用该仓库，并检查工作区、暂存区；
   不得提交用户已有改动。
3. 使用 experiments/runner.py 作为唯一正式执行器。
   按其说明，将 runner 复制到仓库外运行。
4. 从 main 的 canonical queue 读取实验，按 ID 顺序执行。
5. 严格执行 queue 中 pinned commit，不改实验代码、配置或实验定义。
6. 按 stage1_source 自训练或复用 exact checkpoint；
   来源缺失、不匹配时明确失败，不擅自改为重新训练。
7. 正式产物放在 artifacts/<ID>/run。
8. 利用有效 marker 恢复已完成阶段，保留日志和已有产物。
9. done 永远跳过；failed 只有本次明确指定时才重试。
10. 按 PHASE2_RULES 分类处理异常：
    实验自身问题保留现场并记为 failed；
    公共执行器问题先暂停批次，最小修复并通过测试后继续。
11. 完成后确认 record、queue、summary 与实际产物一致。

不要创建新实验，不要根据测试集表现修改模型或超参数。

完成后汇报：
- 本次执行及跳过的实验
- DONE / FAILED 列表与失败原因
- 各实验预测、回测指标及产物位置
- 是否复用了 Stage 1、是否跳过已完成阶段
- 是否修改执行基础设施及对应测试结果
- 剩余未完成的队列条目
```

### Phase 2 示例

处理全部待执行及中断的实验：

```text
按 Phase 2 模板，使用固定 runner 顺序执行全部 pending/running 实验。
已有 done 保持不动，failed 本次不重试。
完成后汇总结果与剩余队列。
```

只执行某个实验（将 `004` 替换为实际已入队的 ID）：

```text
按 Phase 2 模板，仅执行实验 004，使用 runner 的 --only 004。
若它已 done，则正常跳过。
```

明确重试失败实验：

```text
按 Phase 2 模板，重试 failed 的实验 004，使用 --only 004。
优先利用有效阶段产物，不清空 artifacts，不修改实验定义。
```

### 推荐使用顺序

提出构想 → Phase 1 冻结并入队 → 累积几个实验 → Phase 2 批量执行
→ 阅读结果 → 用新 ID 验证下一轮构想。

当前执行器会切换共享仓库的检出版本，两个阶段应错开执行。
