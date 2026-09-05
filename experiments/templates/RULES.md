# RULES — AI 创建和管理实验时必须遵循的规则

本文件定义第一阶段的实验创建流程。任何 AI 在本仓库中创建新实验时，
必须先完整阅读本文件，并严格遵守。

## 仓库所有权规则（最重要）

- `main` 负责且只负责三类内容：
  1. 实验元数据：`experiments/queue.yaml`、`experiments/records/`
     （以及本框架自身的模板文件）；
  2. 公共执行与测试基础设施：`experiments/runner.py`、公共回归测试
     （`tests/test_runner.py`、`tests/test_backtest_mdd.py`、
     `tests/test_protocol_metrics.py` 等）以及 corrected 执行协议代码
     （如 `backtest_qlib.py` 的回测协议与 MDD 实现）；
  3. 从上游 PRISM-VQ 继承的原始代码基线。
- `main` 上不出现任何单个实验的实验性代码改动（模型结构、量化器、
  融合方式等实验变量只能存在于 exp 分支）。
- `main` 根目录的 `README.md` 永远保留原始 PRISM-VQ 的项目说明，
  不得在 `main` 上修改。
- `exp/<ID>-<name>` 实验分支只保存该实验的代码改动，
  不在实验分支上创建 record 或修改 queue。
- 每个 `exp/<ID>-<name>` 实验分支都必须将根目录 `README.md` 改写为
  该实验自己的说明，内容至少包括：Base、Idea / Motivation、核心修改、
  与 base 的区别、smoke 状态。分支 README 属于实验分支自身，
  随实验代码一起提交到 exp 分支；但 `queue.yaml` 和 `records/`
  仍然只允许在 `main` 上修改。
- 分支 README **不得记录自己的 Final Experiment Commit SHA**：
  一个 commit 无法包含它自己的 hash（自引用问题）。Final SHA 只记录在
  `main` 的 queue 与 record 中。
- 因此实验代码与实验元数据严格分离：代码（含分支 README）在 exp 分支，
  队列与记录永远只在 `main`。

## Immutable experiment 原则

**进入 queue 的实验 = immutable experiment。** 一个实验 ID 唯一对应：

`代码版本（Final Experiment Commit）+ Stage 1 provenance + 固定执行协议 + 正式结果`

- 每个实验只有**一个**正式代码版本：**Final Experiment Commit**
  （开发完成、tests PASS、smoke PASS 后在实验分支上提交的最终 commit，
  见第 5 节的收尾流程）。`queue.commit`、record `## Git` 的 `Commit`、
  实验分支冻结时的 HEAD 三者必须是同一个完整 sha。
- 不存在"开发 commit / queue pinned commit"两个正式版本的概念。
  （001/002/003 的 record 保留了历史上多 commit 的说明，属历史记录；
  新实验只认单一 Final Experiment Commit。）
- Phase 2 执行器只 checkout queue 中 pin 的 Final Experiment Commit
  （detached HEAD），而不是实验分支的最新 HEAD。
- Final Experiment Commit 产生后，实验分支冻结：代码与 queue 条目
  （除 `status` 外）一律不得再修改。如果实验逻辑需要改变，必须创建
  新的实验 ID（新分支、新 record、新 queue 条目），不得修改已有实验。

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

### 3.2 Stage 1 provenance（stage1_source）

- queue 条目通过 `stage1_source` 明确表达 Stage 1 的精确来源，三种形式：
  - `self`（缺省）：本实验自己训练 Stage 1，保持向后兼容；
  - `"<已有实验ID>"`：复用该实验已完成的正式 Stage 1 best checkpoint
    （stage2-only 实验）；
  - `external` + `stage1_ckpt: <path>`：复用 baseline / 外部的 exact
    checkpoint（例如 PRISM-VQ 原始 Stage 1 checkpoint）。相对路径相对于
    repo 根目录解析；文件必须存在且非空，否则执行器报错并将实验标记为
    failed，绝不偷偷改为重新训练 Stage 1。
    注意：`artifacts/` 下的 checkpoint 只是**当前 local workspace** 内的
    副本——`artifacts/` 被 gitignore，普通 `git clone` 不会获得这些文件；
    缺失时执行器同样 loud fail，由使用者按 record 中的 provenance
    说明重新放置文件。
- 只修改 Stage 2 的实验（例如只改预测头或融合方式、Stage 1 完全不变），
  应显式复用 exact Stage 1 checkpoint，保证 controlled experiment 并
  节省 GPU 时间。
- 是否允许复用、复用哪一种来源，由创建实验的 AI 在第一阶段明确判断并
  写进 queue；执行器不得自动推断或猜测 Stage 1 source。
- 指定复用前必须确认 source 的 Stage 1 模型结构、量化器配置与数据
  划分和本实验完全一致（checkpoint 可以 strict 加载到本实验的 Stage 1
  模型）。
- 执行器复用 `"<ID>"` 来源时从 `artifacts/<source_id>/run/.stage1.done`
  读取精确的 Stage 1 checkpoint，并验证两件事：
  1. checkpoint 存在且非空；
  2. source marker 记录的 `commit` 等于 source 实验在 canonical queue
     （main 上的 queue）中的 pinned commit——即 source 的 Stage 1 产物
     确实产生于它自己的 Final Experiment Commit。
  任一验证失败（marker 缺 commit、commit 不匹配、checkpoint 缺失）都必须
  报错并把实验标记为 failed；不允许继续复用，也不允许偷偷改为重新训练
  Stage 1。

### 4. 开发与验证（在实验分支上）

- 开发完成后必须执行单元测试和最小 smoke test。
- smoke test 失败时继续修复；修复前不允许进入下一步，
  更不允许加入 queue。

### 5. 收尾（Final Experiment Commit 流程，严格按顺序执行）

固定顺序：`开发完成 → tests PASS → smoke PASS → 确认 diff 仅含预期实验
修改 → 提交 Final Experiment Commit → 确认 tracked 工作区 clean →
获取 HEAD SHA → 冻结实验分支 → 切回 main → 用该 SHA 写入
queue 和 record`。

1. 确认开发完成：单元测试 PASS、最小 smoke test PASS。
   失败则回到第 4 步（开发与验证）继续修复；修复前不允许进入下一步，
   更不允许加入 queue。
2. 检查未提交改动的范围（**提交前不要求工作区 clean**——此时尚未提交
   Final Experiment Commit，工作区本来就该有改动）：
   `git status --porcelain` 与 `git diff` 中必须只包含本实验预期的修改
   （实验代码、默认 config、分支 README、本实验新增的测试），
   不得混入任何无关改动；artifact、数据 symlink 等未跟踪路径除外。
   发现无关改动时先移除或拆分，不得带进 Final Experiment Commit。
3. 在实验分支上提交 **Final Experiment Commit**（仅代码改动）。
   从这一刻起实验分支**冻结**，不得再有任何提交；实验逻辑如需变化，
   必须创建新的实验 ID。
4. 确认提交后 tracked 工作区 clean：
   `git status --porcelain --untracked-files=no` 无输出——即所有预期
   修改都已进入 Final Experiment Commit，没有遗漏未提交的内容。
5. 获取该 commit 的完整 SHA：`git rev-parse HEAD`。这就是该实验唯一的
   正式代码版本。
6. 切回 `main`。
7. 在 `main` 上按 `templates/experiment.template.md` 创建
   `experiments/records/<ID>-<name>.md`，填写 Idea、Motivation、
   Modification、Constraints、Git（Base 填创建分支所用的 base，
   Branch 填 exp 分支名，Commit 填第 5 步的完整 SHA）、Smoke Test。
   Result 保持 `Status: PENDING`，Conclusion 留空。
8. 在 `main` 上向 `experiments/queue.yaml` 追加该实验（按 id 顺序追加到
   末尾），初始状态必须为 `pending`，`commit` 填第 5 步的同一个完整
   SHA。需要复用 Stage 1 时同时填写 `stage1_source`（以及 external
   形式下的 `stage1_ckpt`）。
9. **入队一致性校验（必须全部通过才算入队完成）**：
   - tests PASS、smoke PASS（第 1 步已确认）；
   - exp 分支 tracked 工作区 clean（第 4 步已确认：Final Experiment
     Commit 之后无遗留未提交改动）；
   - `git rev-parse exp/<ID>-<name>` == 第 5 步的 Final Experiment
     Commit（分支在冻结后没有被意外移动）；
   - `queue.commit` == record `## Git` 的 `Commit` == 分支冻结 HEAD，
     三者是完全相同的完整 sha。
10. 在 `main` 上 commit record + queue 的变更。

### 6. 边界

- queue 中的 `pending` 实验代表：代码已经开发完成、smoke 已通过、
  可以直接进入正式训练。
- 第一阶段（AI 开发阶段）不允许启动正式的长时间训练；
  正式训练由第二阶段的固定执行器完成。
- failed 实验仍保留原 ID、分支和历史记录，不得删除或改号。

## 第二阶段（执行器）的职责边界

- 执行器默认消费 queue 中 `pending` 和 `running` 的实验，按 ID 顺序执行；
  `running` 表示上一次执行可能中断，重新进入时依靠 artifact 中的
  `.stage1.done` / `.stage2.done` / `.backtest.done` marker 自动续跑，
  不得因此清空 artifact 或从头执行。`done` 永远跳过；`failed` 只有在
  显式指定（`--only <ID>`）时才允许重试。
- 执行时 checkout queue 条目 pin 的 Final Experiment Commit
  （detached HEAD），而不是实验分支的最新 HEAD；完成后回到 `main`
  更新元数据。
- 执行器负责在 `main` 上补充记录中的 Result 部分，并将 queue
  状态更新为 `done` 或 `failed`；必要时填写 Conclusion。
- 执行器不创建新实验、不修改实验代码、不改动 ID 与命名。
- Phase 2 的完整执行规范（职责、错误分类、batch 行为、resume 原则、
  汇报要求）见 `experiments/PHASE2_RULES.md`。
