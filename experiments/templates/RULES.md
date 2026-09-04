# RULES — AI 创建和管理实验时必须遵循的规则

本文件定义第一阶段的实验创建流程。任何 AI 在本仓库中创建新实验时，
必须先完整阅读本文件，并严格遵守。

## 仓库所有权规则（最重要）

- `main` 只负责保存 `experiments/queue.yaml` 和 `experiments/records/`
  （以及本框架自身的模板文件）。`main` 上不出现任何实验性代码改动。
- `main` 根目录的 `README.md` 永远保留原始 PRISM-VQ 的项目说明，
  不得在 `main` 上修改。
- `exp/<ID>-<name>` 实验分支只保存该实验的代码改动，
  不在实验分支上创建 record 或修改 queue。
- 每个 `exp/<ID>-<name>` 实验分支都必须将根目录 `README.md` 改写为
  该实验自己的说明，内容至少包括：Base、Idea / Motivation、核心修改、
  与 base 的区别、smoke 状态。分支 README 属于实验分支自身，
  随实验代码一起提交到 exp 分支；但 `queue.yaml` 和 `records/`
  仍然只允许在 `main` 上修改。
- 因此实验代码与实验元数据严格分离：代码（含分支 README）在 exp 分支，
  队列与记录永远只在 `main`。

## 创建一个新实验的流程

### 1. 读取现状

- 首先读取 `experiments/queue.yaml`。
- 浏览 `experiments/records/` 下已有的实验记录。
- 确认当前分支、工作区状态，以及与用户 Idea 相关的代码位置。

### 2. 分配实验 ID 和名称

- 找到历史最大实验 ID（综合 `queue.yaml` 和 `records/` 中的所有 ID）。
- 新实验 ID = 最大 ID + 1，固定为三位数字字符串（例如 `"004"`）。
- 如果当前没有任何历史实验，第一个实验编号从 `001` 开始；
  不存在 `000` 实验。
- 已使用过的 ID 永不复用，即使对应实验 failed 或被废弃。
- 根据用户 Idea 生成简短、可读的 kebab-case 名称（例如 `market-gated-z1`）。
- 分支名必须为：`exp/<ID>-<name>`，例如 `exp/004-market-gated-z1`。
- 实验记录文件名必须为：`records/<ID>-<name>.md`。
- ID、名称、分支名、记录文件名、queue 条目必须一一对应。

### 3. 创建实验分支（base 规则）

- 必须先创建实验分支，再修改任何代码。
- 新实验默认从 `main` 创建，此时 `base: main`。
- 只有当用户明确指定"基于某个已有实验继续开发"时，才从该实验的
  `branch` 创建，此时 `base` 填该分支名（例如
  `base: exp/001-hvq-residual-2level`）。
- 不要自动推断继承关系；用户未明确指定时，始终使用 `main`。
- 创建分支前必须先确认 `base` 分支真实存在（`git rev-parse --verify`）。
- 创建新实验分支后，必须确认 `configs/config.yaml` 中默认
  `train.seed` 为 0；当前筛选阶段所有实验统一使用 `train.seed: 0`
  （只指 Stage 2 使用的 `train.seed`；Stage 1 固定的 seed 42 逻辑
  不得改动）。
- 每个实验分支的默认 `configs/config.yaml` 必须完整代表该实验：
  直接运行默认配置（不带任何实验特有 override）即为该实验本身。
  核心实验改动不得依赖实验特有的 CLI override 才能启用。
- CLI override 只能用于统一执行层参数，例如 `train.seed`、
  `train.num_epochs`、`artifact_root`（artifact 输出目录）等；
  不得用于开启实验的核心改动。
- 从已有实验 branch 派生新实验时，也必须检查并更新默认 config，
  使其准确代表新实验（而不是沿用 base 实验的默认值）。
- queue 条目和 record（`## Git` 的 `Base:`）中的 `base` 必须一致。
- 一个实验只修改该 Idea 必需的内容，不顺手重构无关代码。
- 默认保持数据划分、seed、训练协议、回测协议和其他 baseline
  参数不变，除非 Idea 明确要求修改。

### 3.1 Artifact 隔离（artifact_root）

- 运行期产物（checkpoints、预测、回测输出、wandb 文件）必须可以按实验
  隔离，避免不同实验互相覆盖。
- 通过统一执行层参数 `artifact_root=<dir>` 指定：设置后
  `train.save_dir` 重定向为 `<dir>/checkpoints`，`train.save_res`
  重定向为 `<dir>/res`；Stage 2 解析相对路径的
  `predictor.saved_model` 时也以该 checkpoint 目录为准。
  未设置时保持项目默认的 `checkpoints/` 与 `res/`。
- 建议约定：正式运行用 `artifacts/<ID>/run`，smoke 用
  `artifacts/<ID>/smoke`。具体实验 ID 不得硬编码进模型代码，
  只能由调用方（CLI override / 未来执行器）传入。
- 控制台日志由调用方 shell 重定向到对应 artifact 目录
  （例如 `> artifacts/<ID>/smoke/stage2.log`），代码内不写死日志路径。
- `backtest_qlib.py` 已支持 `--pred_path` / `--output_dir`，
  回测产物跟随预测文件所在目录，无需额外改动。

### 4. 开发与验证（在实验分支上）

- 开发完成后必须执行单元测试和最小 smoke test。
- smoke test 失败时继续修复；修复前不允许进入下一步，
  更不允许加入 queue。

### 5. 收尾（严格按顺序执行）

1. 在实验分支上 commit 实验代码（仅代码改动）。
2. 切回 `main`。
3. 在 `main` 上按 `templates/experiment.template.md` 创建
   `experiments/records/<ID>-<name>.md`，填写 Idea、Motivation、
   Modification、Constraints、Git（Base 填创建分支所用的 base，
   Branch 填 exp 分支名，Commit 填实验分支上的代码 commit hash）、
   Smoke Test。Result 保持 `Status: PENDING`，Conclusion 留空。
4. 在 `main` 上向 `experiments/queue.yaml` 追加该实验
   （按 id 顺序，追加到末尾），初始状态必须为 `pending`。
5. 在 `main` 上 commit record + queue 的变更。

### 6. 边界

- queue 中的 `pending` 实验代表：代码已经开发完成、smoke 已通过、
  可以直接进入正式训练。
- 第一阶段（AI 开发阶段）不允许启动正式的长时间训练；
  正式训练由第二阶段的固定执行器完成。
- failed 实验仍保留原 ID、分支和历史记录，不得删除或改号。

## 第二阶段（执行器）的职责边界

- 执行器只消费 queue 中 `pending` 的实验，按 ID 顺序执行；
  执行时切到对应的 exp 分支，完成后回到 `main` 更新元数据。
- 执行器负责在 `main` 上补充记录中的 Result 部分，并将 queue
  状态更新为 `done` 或 `failed`；必要时填写 Conclusion。
- 执行器不创建新实验、不修改实验代码、不改动 ID 与命名。
