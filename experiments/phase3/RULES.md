# Phase 3 RULES

本规则对应本目录当前可执行实现。Phase 3 始终称为 Phase 3，不使用 Phase3.1 等名称。
用户当次明确约束优先。在线真实机器见 configs/machines.yaml（2026-09-11 起为
autodl-2080ti-1 至 -5，各一块 2080 Ti）；不要把离线机器当作可用。

## 1. 科研边界

- Phase 1 定义并冻结实验，Phase 2 执行 seed0，Phase 3 补跑人工指定 training seeds。
- 唯一正式入口是 `python -m experiments.phase3.coordinator`。
- 只接受 batch 明确列出的目标；不扫描 done 后自动决定入选、不创建新实验、不改 queue。
- 使用 exact frozen commit。不得 patch 实验、调参、改 loss/数据划分/训练预算/Stage 1。
- 独立训练 seed 必须来自实际 fit；修改文件名、重复 inference 不算 multi-seed。
- 本版支持 canonical VQ Stage 2 独立训练协议；不支持的入口返回 unsupported。
- baseline 不属于 canonical queue，使用特殊名称及独立兼容性证据，不创建 000。

## 2. Batch → Device → Ordered Experiments → Ordered Seeds

- Batch spec 是执行请求，runtime state 是执行状态，二者不是第二个 experiment queue。
- 不同 device 并行；同一 device 的 experiment 和同一 experiment 的 seed 均严格按列表串行。
- Batch 默认 seeds 可被 per-experiment seeds 覆盖，必须非空、唯一、正整数，禁止 0。
- 一个实验不能重复分配；同一机器身份不能以多个 alias 重复入选。
- Worker 只执行 job 的任务顺序，不发邮件、不关机、不修改全局计划。
- 一个 experiment 的全部 requested seeds ready 后，worker 等待本机验收 acknowledgement。
  Coordinator 验收并尝试通知后发 acknowledgement，worker 才进入下一个 experiment。

## 3. 预检、来源与数据

- 对普通实验核对 queue done、branch HEAD/queue commit、record smoke PASS/Result DONE、
  seed0 marker、预测/指标/回测、Stage 1 provenance 来源链。
- 对 baseline 必须固定代码并单独审计旧 seed0 与代码/输入的可比性，不推断 main 永远没变。
- 输入文件用 SHA-256 标识；保留 Stage 1 原文件名；数据/权重不能仅按路径视为相同。
- 远程数据从冻结代码、远程已有 Qlib、显式 staging 的 JKP CSV 生成，不传本机 pickle。
- 正式 fit 前完整比较真实 sampler 的 index、columns、split、全部采样窗口值 fingerprint。
  不同 fingerprint 阻止 fit；不能为了通过检查改数据、降低容差或只比较日期/schema。
- 校验 machine identity、Python/关键包版本、GPU 可用性、空间、独立输出目录。
  CUDA build 可不同但须记录；关键包版本需匹配。GPU 数值可复现性不因版本检查自动成立。
- 不可确定即停止相应任务并通知，不能把既有文件存在当作 preflight 成功。

## 4. Seed0 和所有输出的保护

- `artifacts/<experiment>/run/` 永远只读，禁止覆盖、移动、改名、合并、创建 marker。
- 原 record Result 与 Phase 2 queue 不写入；Phase 3 用自己的 reports。
- 新输出仅在独立 attempts、incoming、Phase 3 archive、evaluation 路径内产生。
- 原 seed0 操作性 symlink 仅记录链接文本，不跟随写入；新 export/目标路径禁止 symlink。
- 上传/回传禁止 `rsync --delete`、`--inplace`、整仓库混合同步、远程代码覆盖本机仓库。
- 同路径相同内容可幂等复用；同路径不同内容是 conflict，保留现场，不选择“较新”的文件。
- 完整验收之后才无覆盖发布，持久化 receipt；rsync exit 0 不等于 accepted。

## 5. 独立训练与 acceptance

- Runtime 在真实 `Trainer.fit` 处观察 requested seed、module.config seed、torch initial seed、
  Lightning seed、真实 fit 完成/global step、best checkpoint hash、Stage 1 hash、data fingerprint。
- Fit 前后 Stage 1 三模块必须 frozen 且与原 checkpoint 张量完全一致。
- 禁止 warm-start；重试是相同 protocol 新 attempt，从头训练尚未完成的 seed，不覆盖旧 attempt。
- Acceptance 对 manifest required files/hash、checkpoint 可加载/训练状态、预测可读取/schema、
  实际 seed/commit/Stage 1/data 证据及 seed0 标签/index 一致性逐项验证。
- 合成 fixture 的结果永远不能进入正式 acceptance/aggregate。
- 不把局部读取验证说成完整模型推理复现或多 GPU 一致性证明。

## 6. State 和 resume

- Coordinator 单 batch 锁，worker 单设备 lease，原子 JSON + fsync，attempt 保留。
- Ready/accepted seed 不重训；本机 transfer/evaluation/notification 失败不改变训练成功事实。
- Worker 脱离 SSH 生命周期运行；断连先查询状态，不立即启动新训练。
- 训练子进程继承 device lease，发现 busy/orphan 不启动替身。不凭过期时间偷取锁。
- 失败/中断 attempt 需要显式 `--retry-failed`；不得清理有效 marker 以强制重跑。
- 全部验收、设备已完成之后，本机 evaluation 恢复不要求重新联系已关闭服务器。
- 批次/机器/工具/回测代码固定后不得原地变更；恢复时若来源不一致就报 conflict。

## 7. 关机硬约束

- Local 类在能力层面没有 shutdown；local 的非 disabled 关机配置非法。
- Remote worker 无任何关机代码。只有本机 coordinator 可请求关机。
- 必须：该 remote 的所有 assigned tasks 均 accepted 且 receipt 持久化；worker 已退出；
  无 pending/recovery 任务、无其他设备 lease；再次核对 identity，禁止本机身份。
- 不必等待本机 backtest/aggregate。一个实验结束不能关闭仍有其他实验的设备。
- Remote shutdown 要有显式配置能力和 `--allow-shutdown`；缺少时只 dry-run/disabled。
- 关机意图先落盘。SSH 失联不证明关机，当前 backend 仅 reported requested_unconfirmed。
  不重发不确定的关机请求、不谎称成功；记录 attention。不得启动其他机器来“验证”。
- 真实关机仅限当次用户明确授权的正式批次（配置能力 + `--allow-shutdown`）；
  开发/测试轮只走 fake/dry-run 分支。

## 8. 评估与通知

- 所有新增 seed 在本机使用固定的原 backtest_qlib.py；统一 project_portfolio 口径。
- 逐 seed + mean + 样本标准差 ddof=1 + n；seed0 只读纳入，缺 seed 不冒充完整结果。
- 若以后加入 ensemble，必须平均预测后独立回测，不平均 Sharpe 充当 ensemble。
- Coordinator 通过现有 experiments/notify.py 发邮件，远程不能得到 SMTP 凭据。
- Experiment completion：全部 requested seed accepted 后立即单独通知，说明本机评估尚待完成。
- Remote shutdown：独立事件，状态须准确；requested_unconfirmed 不能写“已关机”。
- Batch completion：全部本机评估/汇总完成后发；存在失败发 incomplete 和 attention。
- Attention：provenance/data/conflict/identity/transfer/恢复/关机异常需要人工介入时通知。
- 邮件状态独立持久化并限速重试；失败不触发重训、不改实验状态、不阻断已验收结果。
  SMTP 网络结果不确定时不保证 exactly-once delivery；按事件记录尽量避免重复。

## 9. 开发、验收和变更

- 修复公共基础设施可以改本目录和相关测试，不修改冻结研究源码。
- 必测：两层串行、设备并发、重复分配、unsupported、seed0 不变、真实 fit seed、数据内容
  fingerprint、transfer/acceptance failure、conflict、resume、不联系离线设备恢复本机评估、
  local 无 shutdown、remote identity mismatch、通知时机/独立性/失败隔离、dry-run。
- 真实远程诊断只使用 coordinator diagnostic（CPU fixture），不手工启动股票模型训练。
- 保留测试/诊断报告，准确列出尚未验证的正式 GPU、完整数据重建、云关机确认等限制。
