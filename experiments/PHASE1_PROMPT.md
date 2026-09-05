# PHASE1_PROMPT — Phase 1 启动提示词（直接复制给 Coding Agent）

下面代码块中的内容是一个完整、可直接复制的 Phase 1 启动提示词。
将 `【】` 中的内容替换为实际值后，发给一个独立的 Coding Agent 即可。
Phase 1 的交付物是**冻结并入队（pending）的实验**；长期规则见
`PHASE1_RULES.md`，本文件只是启动入口。

```text
在 HVQ-Stock 仓库中执行 Phase 1：根据下面的实验 Idea 开发并入队下一个实验。

实验 Idea：
【描述模型/方法改动，以及要验证的假设】

基于（base）：
【main；或用户明确指定的 exp/<ID>-<name>】

实验约束：
- 唯一实验变量：【本次允许改变的内容；除此之外一律与 base 保持一致】
- 数据划分、训练预算、seed 协议、回测协议及其他超参数不变。
- Stage 1 来源：【self / 复用实验 "<ID>" / external + checkpoint 路径】
  若复用，先验证 Stage 1 结构、量化器配置、数据划分与 checkpoint
  兼容性（strict 加载）。

执行要求：
1. 先完整阅读 experiments/PHASE1_RULES.md 并严格遵守。
2. 阅读 experiments/queue.yaml 与已有 records，分配下一个实验 ID
   （最大 ID + 1，三位数字字符串），创建 exp/<ID>-<name> 分支——
   先建分支，再改代码。
3. 保持单一实验变量；默认 configs/config.yaml 必须完整代表本实验，
   不得依赖实验特有 CLI override 开启核心改动。
4. 添加必要测试；运行单元测试与最小 smoke test，全部 PASS 才允许继续
   （smoke 产物放 artifacts/<ID>/smoke/）。
5. 改写分支根目录 README 为本实验说明（Base、Idea/Motivation、核心
   修改、与 base 的区别、smoke 状态；不要写入自己的 commit SHA）。
6. 提交 Final Experiment Commit（提交前确认 git status/diff 只包含
   本实验预期修改），然后确认 tracked 工作区 clean，记录 HEAD 完整
   SHA，冻结分支（此后不得再改；逻辑变化必须换新实验 ID）。
7. 切回 main：创建 experiments/records/<ID>-<name>.md
   （Result 保持 Status: PENDING），向 queue.yaml 追加
   status: pending 的条目，commit 字段填同一个完整 SHA。
8. 入队一致性校验：queue.commit == record 的 Git Commit == 实验分支
   冻结 HEAD（git rev-parse exp/<ID>-<name>），三者完全相同；
   在 main 上提交 record + queue。

本阶段只完成开发与入队，不启动正式长时间训练（正式训练由 Phase 2 的
固定执行器完成）。

完成后汇报：
- 实验 ID、name、base、branch、Final Experiment Commit（完整 SHA）
- 核心改动与保持不变的条件
- tests / smoke 结果与产物位置
- Stage 1 来源及兼容性验证结论
- 入队一致性校验结果（三者 SHA 一致）
```
