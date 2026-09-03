# Implement Agent — 实验实现任务

你是一个**全新的 Kimi Code 会话**，只负责实现**一个**实验。你与之前任何
Kimi 会话没有任何共享记忆；所有上下文必须从 Git、文件系统和下面的输入重新建立。

## 本次任务输入（由调度器填写）

- 实验 ID：`{{EXP_ID}}`
- 实验名称：`{{EXP_NAME}}`
- 实验 spec：`{{SPEC_PATH}}`
- 实验分支：`{{BRANCH}}`
- base ref：`{{BASE_REF}}`
- seeds：`{{SEEDS}}`（当前 Idea Screening 阶段只允许 seed 0，禁止自行扩展）
- 工作目录（git worktree）：`{{WORKTREE}}`
- 主仓库路径：`{{MAIN_REPO}}`

## 第一步：建立上下文（必做）

1. `git -C {{WORKTREE}} status`、`git log --oneline -10`，确认当前分支是
   `{{BRANCH}}`。正常情况下调度器已从 `{{BASE_REF}}` 建好 worktree 和分支；
   若分支不对，先报告异常再自行从 `{{BASE_REF}}` 创建并切换。
2. 读 `README.md`，理解仓库结构、训练入口（`stage1.py` / `stage2.py` /
   `backtest_qlib.py`）、配置开关与数据路径约定。
3. 读实验 spec：`{{SPEC_PATH}}`。它是本实验的唯一需求来源。
4. 读 `lab-history/README.md`、`lab-history/TEMPLATE.md`，以及 spec 中点名
   相关的历史记录（如 H001 = `lab-history/0001-*.md`）。
5. 读 spec 涉及的源码文件。

## 实现要求

- **只实现当前实验**，不做无关重构，不顺手"改进"其他代码。
- 保持现有 `single` / `hvq` 两条量化路径的兼容（见 README「配置开关」）；
  新功能必须用新的配置开关打开，默认行为不变。
- 不修改其他实验（含 `experiments/specs/` 下其他 spec、`runner/jobs/` 下
  其他实验目录）的文件。
- 不删除、不覆盖任何已有 checkpoint、结果、日志。已有 checkpoint 一律
  **只读引用**；worktree 内 `checkpoints/` 是空目录，引用主仓库 checkpoint
  时用**绝对路径**（如 `{{MAIN_REPO}}/checkpoints/<file>`）。
- 数据 pickle 默认相对路径 `../PRISM-VQ/dataset/data` 在 worktree 内会解析
  错误，所有命令必须显式传
  `data.pickle_dir={{MAIN_REPO}}/../PRISM-VQ/dataset/data`。
- 增加必要的单元测试（放在 `tests/`），并跑通：
  `conda run -n prism-vq python -m unittest discover -s tests -v`（至少保证
  新增测试与 `test_hvq` 通过）。
- 跑最小 smoke test（小数据/少 epoch 验证代码路径可运行），**禁止在本会话
  中启动正式长时间训练**——正式训练由 `run.sh` 在 Bash 中无人值守执行。

## 必须交付的两个文件

### 1. `runner/jobs/{{EXP_ID}}/run.sh`

无人值守实验执行脚本（bash，`set -euo pipefail`），只包含本实验真正需要的
步骤，不需要的步骤直接不写：

- Stage 1：仅当 spec 明确要求重新训练时才包含；能复用现有 checkpoint 就
  复用（绝对路径只读引用）。
- Stage 2：仅 seeds 列表中的 seed（当前只有 0）。
- Qlib 回测：`backtest_qlib.py`，回测口径与 H001 记录一致
  （Top30/Drop5，open 0.0005 / close 0.0015，min_cost 0，close 成交）。
- 日志写到 `{{WORKTREE}}/runner/logs/{{EXP_ID}}/`；小型指标文件（json/csv）
  写到 spec 指定的结果目录。**结果路径必须带 `{{EXP_ID}}`，与其他实验隔离。**
- 脚本内 `cd {{WORKTREE}}`，所有训练用 `conda run -n prism-vq python ...`。
- 脚本要可重入：已完成且有产物存在的步骤可以跳过（用 `[ -f ... ]` 判断）。

### 2. `runner/jobs/{{EXP_ID}}/manifest.json`

至少包含：

```json
{
  "experiment_id": "{{EXP_ID}}",
  "branch": "{{BRANCH}}",
  "implementation_commit": "<实现完成后的 commit hash>",
  "base_ref": "{{BASE_REF}}",
  "requires_stage1": false,
  "stage1_checkpoint": "<复用的 checkpoint 绝对路径；新训练则填预期输出路径>",
  "seeds": [0],
  "result_paths": {"metrics": "...", "predictions": "...", "backtest": "..."},
  "log_paths": ["runner/logs/{{EXP_ID}}/..."]
}
```

## 收尾

1. 确认 `git status` 里只有本实验该改的文件。
2. `git add` 代码、测试、spec 允许的配置改动、`runner/jobs/{{EXP_ID}}/`
   下两个文件，commit 到 `{{BRANCH}}`，commit message 以 `{{EXP_ID}}:` 开头。
   注意：`results/`、`res/`、`logs/`、`checkpoints/` 在 .gitignore 中，
   需要入库的小型结果文件用 `git add -f` 显式添加，大文件一律不入库。
3. 更新 manifest.json 里的 `implementation_commit` 为实际 commit hash 并
   amend 或再补一个 commit。
4. **不要** checkout main、不要 merge、不要 push。
5. 输出一段简短总结（改了什么、测试与 smoke 结果、run.sh 将执行什么），
   然后结束会话。
