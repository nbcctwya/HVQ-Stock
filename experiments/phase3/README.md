# Phase 3

Phase 3 对用户明确指定的实验执行独立 Stage 2 training seeds。使用 batch
specification + runtime state，不增加实验 queue，不修改 Phase 2 queue/record。
唯一正式入口是 `python -m experiments.phase3.coordinator`；详细边界见
[RULES.md](RULES.md)，AI 执行入口见 [SUPERVISOR_PROMPT.md](SUPERVISOR_PROMPT.md)。

## 已实现的执行流程

```text
Batch
  ├─ Device A: experiment X (ordered seeds) → experiment Y (ordered seeds)
  └─ Device B: experiment Z (ordered seeds)

worker fit → prediction → sealed export
→ local staging → hash/readability/provenance acceptance → immutable archive + receipt
→ experiment notification attempt → acknowledgement → worker's next experiment
→ all device tasks accepted + worker exited → remote shutdown request/dry-run

local backtest → per-seed metrics → mean / sample std / n → batch summary
```

设备间使用线程并行协调；每个设备一个脱离 SSH 会话的 worker 进程，内部串行。
Worker 不发邮件、不关机、不选实验。完成一个实验后等待本机 acknowledgement，
协调器须先验收它的全部 requested seeds 并尝试通知，才发送 acknowledgement。
通知投递失败也发送 acknowledgement，避免 SMTP 故障阻断结果。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `coordinator.py` | 唯一入口、设备并发、验收/回执、评估、汇总、通知事件 |
| `worker.py` | detached 生命周期、设备锁、串行实验/seed、产物封存 |
| `runtime.py` | 隔离 subprocess：数据 fingerprint、实际 fit 审计、产物读取验证 |
| `preflight.py` | batch 校验、canonical queue/record、冻结代码快照、Stage 1 来源链 |
| `transport.py` | local / SSH staging、pull；仅 Remote 有 shutdown 能力 |
| `common.py` | SHA-256、路径边界、锁、原子 JSON、无覆盖发布 |
| `configs/` | machine 配置、batch 示例、当前科研目标的待完善 batch |
| [tests/test_phase3.py](../../tests/test_phase3.py) | 不变量、恢复和 CPU 生命周期测试 |

## 使用前提

从 HVQ-Stock 仓库根目录运行，使用能导入本项目依赖的 Python。本机环境通常是
`/home/nbcctwya/anaconda3/envs/prism-vq/bin/python`，已检查的远程环境是
`/root/miniconda3/envs/quantbase/bin/python`。远程需要已有 SSH 信任配置和 rsync。
协调器不关闭 host-key 检查，不启动云实例，不安装 GPU，不自动升级环境。

支持的正式协议：canonical CSI300/SP500 数据、`stage2.py` 中真正调用
Lightning `Trainer.fit` 的 VQ 系列，Stage 1 固定，Stage 2 独立训练正整数 seed。
历史非 canonical 数据入口及 inference-only AlphaMaster 明确 unsupported。
不是通过分支名判断支持性：先审计入口，运行时还验证真实 fit 和实际 module seed。

Baseline 是特殊名称，不创建 `000`。必须提供完整 commit 和 seed0 兼容性审计报告；
当前 `requested-batch.yaml` 故意留空这些字段，因旧 baseline seed0 早于 main
数据改造，不能自动宣称兼容。报告的最低 JSON 字段为 `status: "pass"`、
`commit`、`seed0_tree_sha256`（用 `identity(inventory(run, allow_symlinks=True))` 计算）。
这些字段仅绑定人工审计结论，不是自动完成科学等价性证明；报告正文还须说明
历史实现、实际模型输入/标签及数据差异的审计证据。禁止为了通过校验填写空洞的 PASS。
本实现不自动生成或批准这个科学等价性结论。

## Batch 和 machine 配置

[batch.example.yaml](configs/batch.example.yaml) 展示同一 device 串行执行多个实验，
并展示默认 seeds 与 per-experiment seeds。列表顺序就是执行顺序，不排序、不负载均衡。
同一实验不能分配给多个设备；同一物理身份不能以两个 machine 名称重复分配。

[machines.yaml](configs/machines.yaml) 目前只列 `autodl-2080ti-3`，它目前为无卡模式。
其他机器不应被当作在线。`work_root` 必须是专用 scratch；local scratch 不得位于
本仓库的 canonical artifacts 下。路径可包含空格，命令使用参数数组/SSH shell quoting。

Machine 字段：`kind: local|remote`、绝对 `python`、绝对 `work_root`、单 GPU ordinal
`gpu`、`min_free_gb`（默认 10）、`identity: {hostname, machine_id}`。
Remote 另有 `ssh_alias` 和 `shutdown: disabled|ssh_poweroff`。
Local 配置 shutdown 为任何非 disabled 值都会报错，而且 Local 类没有 shutdown 方法。

## 命令

只做本机静态预检，不联系任何服务器，不训练、不发邮件、不关机：

```bash
python -m experiments.phase3.coordinator dry-run \
  --batch experiments/phase3/configs/batch.example.yaml \
  --machines experiments/phase3/configs/machines.yaml
```

`dry-run` 验证 metadata/input/artifact 边界并保存报告，但不会计算完整数据 fingerprint、
验证远程环境或声称 ready for training。Baseline 等无法确认的目标返回非零，不跳过。

以下是**日后正式运行命令**，本轮开发不能执行：

```bash
python -m experiments.phase3.coordinator run --batch <batch.yaml> --machines <machines.yaml>
python -m experiments.phase3.coordinator resume --batch <batch.yaml> --machines <machines.yaml>
python -m experiments.phase3.coordinator status --batch <batch.yaml> --machines <machines.yaml>
```

`run` 和 `resume` 都幂等恢复同一批次。`--retry-failed` 仅在失败/中断任务已确认
没有存活执行进程时开启新 attempt，不重训 accepted/ready seeds。没有 epoch 级续训。
开始后 batch、machine、执行器 Python 文件、回测代码被固定；变更会报 conflict。
版本变更需要保留原工具以完成旧批次，不原地升级正在执行的 batch。

通知默认开启，仍通过 `experiments/notify.py`；测试可显式 `--no-notify`。
SMTP 凭据按 Phase 2 约定配置在 `~/.bashrc`；coordinator 通过 login shell
（`bash -lc`）调用 notify.py，非 login shell 启动 coordinator 不会稳定丢失通知。
邮件环境变量不写入仓库文件，不要把 `.bashrc`、SMTP 凭据上传远程。

Remote 默认不关机。只有 machine 选择 `ssh_poweroff` 且正式命令显式带
`--allow-shutdown` 才请求真实关机，否则仅 dry-run。SSH backend **只能报告
requested_unconfirmed**，不会把断连当作关机成功。尚未接入 AutoDL 实例状态 API，
因此不支持自动确认 power-off，也不宣称停止 GPU 计费。该状态独立发 shutdown
通知和 attention 通知；实验与后续本机评估保持有效。本机评估不等待关机确认。

## CPU/no-card 真实远程诊断

```bash
python -m experiments.phase3.coordinator diagnostic \
  --machines experiments/phase3/configs/machines.yaml \
  --machine autodl-2080ti-3 --diagnostic-id cpu-check-unique \
  --python /home/nbcctwya/anaconda3/envs/prism-vq/bin/python
```

使用相同 worker、队列、传输、验收和回执机制，运行两个合成实验，每个两个 CPU seeds，
每个 seed 仅优化一个 tiny Linear 两步。设置 `CUDA_VISIBLE_DEVICES=''`；没有正式
股票模型训练、不构造正式数据、不发真实邮件、不执行真实关机。产物带 `fixture=true`，
正式 acceptance 拒绝它们。相同 diagnostic id 重跑验证 resume；修改工具后用新 id。
诊断会上传工具并写专用远程 scratch，不是零副作用 dry-run。

## 数据与训练证据

正式预检读取本机已有 train/valid/test sampler，在固定实验代码下，按 128 样本块
遍历**全部实际采样窗口**（包括 CanonicalSampler 的 market override）。SHA-256 覆盖：
有序 index、index names、columns、split、sample shape、原 dtype、pandas 版本，
以及规范化为 little-endian float64 的全部数值；NaN/signed zero 规范化。
不是只 hash pickle、不抽样、不只 hash `data_arr`。内存有界，但完整检查需要 CPU 时间。

每台机器在冻结代码下从已有 Qlib 行情 + 显式上传的 JKP CSV 构造数据，再算同样
fingerprint；与本机不一致就不允许进入 fit。只上传代码、必要 Stage 1、JKP、manifest；
不上传预处理 pickle。输入使用 SHA-256 目录，同内容可复用，不同内容不覆盖。
已有数据每次 worker 恢复都会重新核验。部分生成失败目录保留，不偷偷删掉重建。

运行时 wrapper 只观测 Lightning fit，不改冻结研究源码。记录实际 module.config seed、
`torch.initial_seed`、`PL_GLOBAL_SEED`、真实 fit 调用/完成、global steps、选中 checkpoint
hash、Stage 1 hash、data fingerprint、软件/GPU 环境。fit 前后核对 Stage 1 的三个
模块权重与原 checkpoint 完全一致且 frozen。禁止 warm-start fit 和 inference-only。

Acceptance 验证 export 完整清单、hash、checkpoint 可读取且有训练 state/steps、
权重有限值、预测 schema/有限 score、实际 seed 证据以及 seed0/new-seed index/label
一致性。本版不在验收时重新执行完整模型预测；不同 GPU 数值稳定性仍须正式上线前验证。
对于历史数据没有原始 fingerprint 的 seed0，本机当前数据 fingerprint 不能倒推出
它的历史输入完全相同；应保留这个证据限制。

## 产物、恢复与结果

```text
artifacts/<experiment>/run/                         # Phase 2，永远只读
artifacts/<experiment>/phase3/<batch_id>/seedN/     # accepted，不原地增加 evaluation 文件
artifacts/_phase3/batches/<batch_id>/
  manifest.json / deploy/ / checks/ / probes/
  incoming/ / receipts/ / devices/ / events/ / power/
  evaluation/ / reports/
artifacts/_phase3/diagnostics/<diagnostic_id>/      # fixture，永不混入正式汇总
<remote_work_root>/<batch_id>/
  job.json / tool/ / code/ / inputs/ / data/
  worker_state.json / worker.log / attempts/ / acks/
```

原 seed0 tree（含 W&B symlink 的链接文本）在 preflight 与 batch 退出时核对不变。
新 export 只收普通文件，不传 W&B 操作性 symlink。路径越界/符号链接拒绝发布。
接收先写 incoming，完整验收后无覆盖发布 archive，再 fsync 持久化 receipt。
崩溃在 archive/receipt 之间时可通过相同内容重新验收完成；不同内容报 conflict。

Worker 状态是 `running → ready|failed`（每 task）及设备 active lock。Ready seed 不重训；
传输失败恢复传输；acceptance 失败保留现场；evaluation 失败只恢复本机回测。
协调器与设备各有进程锁；训练 subprocess 继承设备 lease，防止 worker 异常退出后
孤儿训练仍存活却重复启动。SSH 不通只记 attention，不判定训练已停止。
无人工取消/云端自动重启/自动迁移功能。

对全部 requested seeds 已验收的实验，用原 `backtest_qlib.py` 固定 Top30/Drop5 协议
在本机回测，读 `project_portfolio` 列，不混用 Qlib 原生 risk_analysis 指标。
原 seed0 只读加入统计，新 seed 的 IC/ICIR/RankIC/RankICIR 来自冻结代码正式输出。
生成逐 seed、mean、样本标准差 `ddof=1`、n；只有一个观测时 std=null。
有失败则先汇总已完整验收的实验，输出 batch incomplete，不能发送全部成功通知。
暂未实现 ensemble。若以后加入，必须平均预测后独立回测。

## 测试

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_phase3 -v
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

真实验证记录及未完成项见 [VALIDATION.md](VALIDATION.md)。所有后续演进仍叫 Phase 3。
