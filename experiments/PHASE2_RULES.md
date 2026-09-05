# PHASE2_RULES — Phase 2 固定执行规范

Phase 2 是**固定、确定性、可恢复的正式实验执行阶段**。

- `experiments/runner.py` 是唯一的正式 executor；正常执行一律通过它完成，
  AI 不得手工重新实现执行流程。
- AI 在 Phase 2 中只作为 supervisor：观察日志、分类异常、做有限处置。
- Phase 2 不提出新实验、不修改研究 Idea、不是第二个 Phase 1。

实验创建流程（Idea、分支、smoke、入队）由 Phase 1 定义，见
`PHASE1_RULES.md`；本文件只规范实验进入 queue 之后的执行行为。

## 1. 职责

Phase 2 对 queue 中每个符合条件的实验执行固定流程：

1. 按 ID 顺序消费 `pending` 与 `running` 实验；
2. checkout queue 条目 pin 的 Final Experiment Commit（detached HEAD）——
   不是实验分支的最新 HEAD；分支在入队后即使发生变化也不影响已入队实验；
3. Stage 1（或按 `stage1_source` 复用实验 / external exact checkpoint）→
   Stage 2（seed 0）→ prediction / metrics → Qlib backtest；
4. 收集 metrics（IC/ICIR/RankIC/RankICIR 与回测指标）；
5. 回到 `main` 更新 `experiments/records/<ID>-<name>.md` 的
   Result / Conclusion；
6. 更新 queue 状态为 `done` 或 `failed` 并 commit；
7. 依靠 artifact marker（`.stage1.done` / `.stage2.done` /
   `.backtest.done`）实现断点续跑；
8. 批处理中发生异常时按第 3 节分类并做有限处置。

## 2. Phase 1 / Phase 2 严格边界

实验进入 queue 意味着 Phase 1 已经完成：Idea、实验分支、代码实现、
实验核心 config、单元测试、smoke test。Phase 2 原则上**只执行，不修改**。

进入 queue 的实验是 immutable experiment：queue 条目（除 `status` 外）
与其 pin 的代码版本一律不得修改；实验逻辑的任何变更都必须以新实验 ID
重新走 Phase 1。

Phase 2 禁止为了"把实验跑通"而擅自修改：

- 模型结构、loss、VQ / HVQ、attention、MoE、fusion；
- factor construction、feature；
- 数据划分、seed protocol、回测协议；
- 实验核心超参数或实验 config。

如果实验自身代码在正式训练中暴露问题：保留现场，标记 failed，
让该实验之后重新进入 Phase 1 修复。

## 3. 错误分类与处置

supervisor 必须先把异常分类，再处置。

### A. 单个实验自身问题

特征：shape mismatch、checkpoint 与该实验模型结构不兼容、实验独有逻辑
错误、实验独有 OOM、正式数据下才暴露的实验代码错误。

处置：

- 不修改实验研究代码；
- 保留全部 artifact 与日志；
- 该实验标记 `failed`，record 中记录失败原因；
- 继续处理后续实验；
- 该实验之后重新进入 Phase 1。

### B. 环境或临时运行问题

特征：CUDA 临时异常、残留进程、wandb / 环境变量问题、数据路径或
symlink 问题、与实验逻辑无关的临时文件/运行环境异常。

处置：

- 可以修复运行环境，但不得改变实验定义；
- 修复后优先利用现有 marker / artifact resume；
- 不得无故清空并重跑已经正确完成的阶段。

### C. Runner / Phase 2 公共基础设施问题

特征：runner 通用 bug、marker / resume bug、artifact 隔离问题、
Hydra 命令构造错误、多个实验都会遇到的公共执行错误。

处置：

- 暂停继续机械消耗后续实验；
- 先确认确为公共基础设施问题（而非单个实验问题）；
- 允许对 Phase 2 基础设施做最小修复，不得改变任何实验研究逻辑；
- 修复后运行对应测试（至少 `tests/test_runner.py`）；
- 确认通过后利用已有 artifact / marker resume 批处理。

## 4. Batch 行为

- 单个实验失败不终止整个 batch：记录失败，继续后续实验。
- 多个实验出现相同错误且判断为公共基础设施问题时，停止继续消耗计算
  资源，先修复公共问题。
- `done` 永远跳过。
- `running` 按现有 resume 规则继续（上一次可能中断）。
- `failed` 默认跳过，仅显式 `--only <ID>` 允许重试。
- 状态语义以 `runner.py` 现有实现为准，不得另建不一致的状态机。

## 5. Artifact 与 Resume 原则

- 正式产物统一在 `artifacts/<ID>/run/`，默认必须保留。
- 阶段完成标记：`.stage1.done` / `.stage2.done` / `.backtest.done`。
  marker 记录产生该阶段结果的精确 provenance：
  - `.stage1.done`：exact checkpoint（`best`）、来源（self / 实验 ID /
    external）、pinned `commit`；
  - `.stage2.done`：使用的 Stage 1 checkpoint（`ckpt`）、`seed`、
    pinned `commit`；
  - `.backtest.done`：回测产物路径、`protocol` 签名（universe + 固定
    回测参数 + seed）、pinned `commit`。
- 一个已完成阶段只有在以下情况才允许重跑：

  - 输入发生变化（如 `stage1_source` 变更、Stage 1 checkpoint 变更）；
  - marker 的 provenance 与当前 queue 声明不一致（pinned commit、
    Stage 1 来源、回测 protocol 任一不匹配即失效）；
  - artifact 缺失或损坏。

  上一级失效必须级联失效其下游 marker（stage1 → stage2 → backtest）。
- experiment→experiment 复用 Stage 1 时，执行器还必须验证 source
  实验的 `.stage1.done` 记录的 commit 等于 source 实验在 canonical
  queue（main 上的 queue，而非 detached experiment commit 上的 stale
  副本）中的 pinned commit；source marker 缺 commit、commit 不匹配或
  checkpoint 缺失/损坏时 loud fail——不允许继续复用，也不允许偷偷
  重训 Stage 1。
- retry / resume 时不得随意删除已有有效结果。
- 出错时日志与现场是诊断依据，优先保留，不做清理。

## 6. 公共基础设施的可修改边界

Phase 2 必要时只允许修改：

- `experiments/runner.py`；
- Phase 2 直接相关的通用执行基础设施；
- 与执行环境相关但不改变科研实验定义的内容。

要求：

- 先定位 root cause，再动手；
- 最小修改，不做无关重构；
- 补充或更新对应测试，至少运行 `tests/test_runner.py` 并通过；
- 修复后继续利用 resume，而不是清空全部实验重跑。

## 7. 保守原则

当无法高置信度判断错误属于"环境/公共基础设施"还是"实验自身问题"时：

**默认不得修改实验代码。**

优先保留日志和 artifact、标记或保留失败状态、避免引入不可追踪的
临时 patch。

## 8. 汇报要求

一轮 Phase 2 执行结束后，supervisor 必须汇总：

- DONE 实验列表；
- FAILED 实验列表及各自主要原因；
- 每个失败的错误类型：实验自身 / 环境 / 公共基础设施；
- 是否修改过 Phase 2 公共基础设施；修改了什么；
- 是否运行并通过相关测试；
- 当前 queue 是否处理完成；
- 哪些实验需要重新进入 Phase 1。
