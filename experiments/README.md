# HVQ-Stock 自动实验调度系统

轻量级科研实验调度框架：AI（Kimi Code）负责**实现**与**验收**，Bash 负责
**调度与无人值守训练**，Git + 文件系统承担跨 AI 会话的长期状态。

```text
新 Kimi 上下文实现实验 -> 退出 Kimi -> Bash 跑训练/回测
-> 新 Kimi 上下文独立验收 -> 写 lab-history -> commit -> 下一个实验
```

## 目录

```text
experiments/
  queue.json            实验队列（id/name/spec/base_ref/enabled/seeds）
  specs/                每个实验一份 spec（唯一需求来源）
  prompts/
    implement.md        Implement Agent prompt 模板（{{占位符}} 由调度器填充）
    review.md           Review Agent prompt 模板
runner/
  run_queue.sh          总调度器
  preflight.sh          环境自检（只检查不修改）
  state.json            运行状态机（本地状态，不入库）
  jobs/<ID>/            实验交付物（run.sh/manifest.json 在 exp 分支入库；
                        manifest/review.json 验收后同步回 main）
  logs/                 调度与各阶段日志（不入库）
  worktrees/            每个实验的独立 git worktree（不入库）
lab-history/            实验档案（复用现有约定，TEMPLATE.md）
```

## 状态机

`PENDING -> IMPLEMENTING -> READY -> RUNNING -> REVIEWING -> DONE`，失败进入
`IMPLEMENT_FAILED / RUN_FAILED / REVIEW_FAILED`。状态存于
`runner/state.json`（本地文件，不 commit）。DONE 的实验重跑时自动跳过；
中断后重启会从断点阶段继续（run.sh 要求可重入）。

## 工作流细节

- 每个实验由调度器从 `base_ref` 创建独立分支 `exp/Hxxx-short-name` 和独立
  worktree `runner/worktrees/<ID>`，实验之间互不叠加，main 上不做实验代码
  修改。
- Implement / Review 各是一个**全新** `kimi -p` 非交互会话，一个会话只处理
  一个实验的一个阶段；正式训练期间没有 Kimi 会话驻留。
- Review 完成后，调度器把 lab-history 记录和 `manifest.json` /
  `review.json`（都是小文件）拷回 main 并 commit；实验代码留在 exp 分支。
- 单实验失败只记录原因并继续队列，不会自动修改科研方案或调参。

## 常用命令

```bash
# 0. 环境自检（启动队列前必跑）
bash runner/preflight.sh

# 1. 启动整个队列（唯一主命令）
bash runner/run_queue.sh

# 只看计划不执行
bash runner/run_queue.sh --dry-run

# 只跑某个实验 / 重试失败实验
bash runner/run_queue.sh --only H003
bash runner/run_queue.sh --retry-failed

# 2. 查看状态
jq . runner/state.json

# 3. 查看实验结果
ls lab-history/                        # 实验档案（人读）
cat runner/jobs/H003/review.json       # 机器可读验收结论
ls runner/worktrees/H003/results/H003/ # 实验产物（worktree 内）
ls runner/logs/H003/                   # 各阶段日志
```

## 新增一个实验

1. 写 spec：`experiments/specs/H004-<name>.md`（参考 H002/H003，写清假设、
   约束、交付物、验收标准；必须注明与 H001 一致的实验变量）。
2. 加入 `experiments/queue.json`：
   ```json
   {"id": "H004", "name": "<short-name>", "spec": "experiments/specs/H004-<name>.md",
    "base_ref": "main", "enabled": true, "seeds": [0]}
   ```
3. `bash runner/preflight.sh && bash runner/run_queue.sh --only H004`。

## 恢复中断任务

直接重新执行 `bash runner/run_queue.sh`：

- `DONE` 自动跳过；
- `IMPLEMENTING`/`PENDING` 重新做实现；
- `READY`/`RUNNING` 重跑 `run.sh`（各实验 run.sh 自行跳过已有产物）；
- `REVIEWING` 重新验收；
- `*_FAILED` 默认跳过，加 `--retry-failed` 重试。

## 从 seed0 screening 切换到 5-seed confirmation

当前 Idea Screening 阶段 `preflight.sh` 会强制所有 enabled 实验
`seeds == [0]`。未来进入 confirmation 阶段时：

1. 把 `queue.json` 中目标实验的 `"seeds"` 改为 `[0,1,2,3,4]`（或另建
   confirmation 队列文件，把 screening 实验 `enabled: false`）。
2. 移除/放宽 `preflight.sh` 中的 seed0 保护检查。
3. 对应实验的 `run.sh` 由新一轮的 Implement Kimi 按多 seed 重新生成
   （调度器只把 seeds 列表原样传给 prompt，不会自动扩展）。
4. lab-history 每个 seed 一条记录，命名沿用 `NNNN-<desc>-csi300-seedN.md`。

## 原则

- 不删除已有实验产物，不覆盖已有 checkpoint，不提交大文件。
- 调度器不理解模型细节；具体命令只在各实验自己的 `run.sh` 里。
- AI 长期记忆只依赖 Git / spec / manifest / state / lab-history。
