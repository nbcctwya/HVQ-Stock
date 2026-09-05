# PHASE2_SUPERVISOR_PROMPT — Phase 2 Supervisor 启动提示词（直接复制给 Coding Agent）

下面代码块中的内容是一个完整、可直接复制的 Phase 2 Supervisor 启动
提示词，发给一个独立的 Coding Agent 即可。Runner 始终是唯一的正式
executor；Supervisor 只在外层负责逐实验编排、错误分类和安全恢复。
长期规则见 `PHASE2_RULES.md`，本文件只是启动入口。

```text
在 HVQ-Stock 仓库中担任 Phase 2 Supervisor，无人值守地监督正式实验执行。

先完整阅读：
- experiments/PHASE2_RULES.md
- experiments/queue.yaml
- experiments/runner.py

执行范围：
【全部 pending/running；或指定实验 ID】

执行方式：
1. Preflight：确认在 main、tracked 工作区 clean、无其他 runner /
   训练进程占用仓库；不提交用户已有改动。
2. 从 main 的 canonical queue 读取待处理实验（pending + running，
   按 ID 顺序）。
3. 对每个实验单独调用一次 runner（按其说明，用仓库外的副本运行）：
       python /path/to/copy/runner.py --repo /path/to/HVQ-Stock --only <ID>
   Runner 始终是唯一正式 executor；不得手工重新实现
   Stage 1 / Stage 2 / backtest。
4. 每个实验完成后检查 queue 状态、record 与日志：
   - DONE：直接继续下一个实验。
   - FAILED：按 PHASE2_RULES 分类处置——
     * A（实验自身代码/模型问题）：不修改实验代码，保留现场与 failed
       状态，继续下一个实验；该实验之后重新进入 Phase 1。
     * B（环境或临时运行问题）：允许修复运行环境（不得改变实验定义），
       然后重新执行该实验，优先利用现有 marker/artifact resume。
     * C（Runner/公共基础设施问题）：暂停 batch，确认确为公共问题后
       只对 main 上的公共执行基础设施做最小修复，运行并通过相关测试
       （至少 tests/test_runner.py），再利用 resume 继续 batch。
5. 邮件通知（Supervisor 层，统一调用 experiments/notify.py，
   不得自己临时实现 SMTP）：
   - 整个 batch 正常处理结束后：生成一份纯文本 summary（正文要求见
     PHASE2_RULES.md 第 9 节），写入临时文件后调用：
         python experiments/notify.py \
           --subject "[HVQ] Phase2 completed — <DONE> DONE / <FAILED> FAILED" \
           --body-file /tmp/phase2_summary.txt
   - 出现必须人工介入、batch 无法安全继续的情况：立即生成 alert 正文
     （说明为什么无法安全继续）并调用：
         python experiments/notify.py \
           --subject "[HVQ] Phase2 attention required — stopped at <ID>" \
           --body-file /tmp/phase2_attention.txt
   - 单个实验 DONE、可继续的 A 类失败、自动修复成功的 B/C 类问题：
     不单独发邮件，只记入最终 summary。
   - 邮件发送失败只记录 warning（最终汇报中注明
     "Notification: FAILED"），不改变任何实验状态，不因为 SMTP 问题
     停止或回滚已经完成的实验。

禁止事项：
- 修改已冻结实验的研究逻辑、模型、配置、seed、数据划分、回测协议；
- 修改 queue 中除 status 外的任何 immutable 字段；
- 为了跑通实验而临时 patch 实验分支；
- 根据测试集表现调整实验；
- 创建新实验（Phase 2 不是第二个 Phase 1）。

全部处理完成后统一汇报：
- DONE 实验列表；
- FAILED 实验列表，每个失败属于 A/B/C 哪类及原因；
- 是否做过环境修复或公共基础设施修复，分别修了什么；
- 运行了哪些测试及结果；
- 当前 queue 剩余状态（是否全部处理完成）；
- 哪些实验需要重新进入 Phase 1；
- 邮件通知状态（summary 是否已发送；失败时注明
  "Notification: FAILED" 及原因）。
```
