# RULES — AI 创建和管理实验时必须遵循的规则

本文件定义第一阶段的实验创建流程。任何 AI 在本仓库中创建新实验时，
必须先完整阅读本文件，并严格遵守。

## 创建一个新实验的流程

### 1. 读取现状

- 首先读取 `experiments/queue.yaml`。
- 浏览 `experiments/records/` 下已有的实验记录。
- 确认当前分支、工作区状态，以及与用户 Idea 相关的代码位置。

### 2. 分配实验 ID 和名称

- 找到历史最大实验 ID（综合 `queue.yaml` 和 `records/` 中的所有 ID）。
- 新实验 ID = 最大 ID + 1，固定为三位数字字符串（例如 `"004"`）。
- 已使用过的 ID 永不复用，即使对应实验 failed 或被废弃。
- 根据用户 Idea 生成简短、可读的 kebab-case 名称（例如 `market-gated-z1`）。
- 分支名必须为：`exp/<ID>-<name>`，例如 `exp/004-market-gated-z1`。
- 实验记录文件名必须为：`records/<ID>-<name>.md`。
- ID、名称、分支名、记录文件名、queue 条目必须一一对应。

### 3. 创建实验分支

- 必须先创建实验分支，再修改任何代码。
- 每个实验分支默认从 `main` 创建，除非用户明确指定其他 base。
- 一个实验只修改该 Idea 必需的内容，不顺手重构无关代码。
- 默认保持数据划分、seed、训练协议、回测协议和其他 baseline
  参数不变，除非 Idea 明确要求修改。

### 4. 开发与验证

- 开发完成后必须执行单元测试和最小 smoke test。
- smoke test 失败时继续修复；修复前不允许进入下一步，
  更不允许加入 queue。

### 5. 收尾（仅在 smoke PASS 之后，按顺序执行）

1. commit 实验代码（在实验分支上）。
2. 按 `templates/experiment.template.md` 创建
   `experiments/records/<ID>-<name>.md`，填写 Idea、Motivation、
   Modification、Constraints、Git（Branch + Commit）、Smoke Test。
   Result 保持 `Status: PENDING`，Conclusion 留空。
3. 向 `experiments/queue.yaml` 追加该实验（按 id 顺序，追加到末尾），
   初始状态必须为 `pending`。

### 6. 边界

- queue 中的 `pending` 实验代表：代码已经开发完成、smoke 已通过、
  可以直接进入正式训练。
- 第一阶段（AI 开发阶段）不允许启动正式的长时间训练；
  正式训练由第二阶段的固定执行器完成。
- failed 实验仍保留原 ID、分支和历史记录，不得删除或改号。

## 第二阶段（执行器）的职责边界

- 执行器只消费 queue 中 `pending` 的实验，按 ID 顺序执行。
- 执行器负责补充记录中的 Result 部分，并将 queue 状态更新为
  `done` 或 `failed`；必要时填写 Conclusion。
- 执行器不创建新实验、不修改实验代码、不改动 ID 与命名。
