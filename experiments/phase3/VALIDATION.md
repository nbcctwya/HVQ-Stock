# Phase 3 VALIDATION

记录 Phase 3 工程验收的真实执行结果。最后更新：2026-09-11（收尾轮 + 五机上线配置轮）。
本文件只陈述实际运行过的验证；未真实执行的能力明确标注为未验证。

## 0. 五机上线配置轮（2026-09-11）

五台 autodl-2080ti-1 至 -5 全部 SSH 实测在线：各一块 RTX 2080 Ti（GPU UUID
互不相同）、远程 Python 3.11.15、关键包版本与本机一致（torch 2.8.0 /
pytorch-lightning 2.6.4 / pyqlib 0.9.7 / numpy 2.4.4 / pandas 2.3.3 /
hydra-core 1.3.3 / omegaconf 2.3.1，本机 torch 的 +cu129 后缀按实现规则剥离后
相等）、均有 `/root/.qlib/qlib_data/cn_data`、磁盘余量 39–50G。

**重要实测发现：五台机器 `/etc/machine-id` 完全相同**（AutoDL 克隆镜像）。
pinned identity 的区分半是 hostname（五台互不相同）；`load_specs` 的重复物理
设备检查与 device_loop 的身份校验仍然有效，但 machine_id 单独不再能区分机器。

关机后端实测：五台均以 root 运行、**无 sudo、无 `/sbin/shutdown`**（实际为
`/usr/bin/shutdown`）。Astra 的 `sudo -n /sbin/shutdown` 在这些机器上必然失败。
修复为运行时解析：root 直接 `shutdown -h now`，否则 `sudo -n shutdown -h now`
（新增单测锁定，25 项 phase3 测试 / 107 项全量回归均 OK）。真实关机命令仍未执行。

配置更新：`machines.yaml` 列入五台（均 `ssh_poweroff` 能力）；
`requested-batch.yaml` 改为一机一实验（baseline→-1、010→-2、019→-3、025→-4、
034→-5，seeds [1,2,3,4]）。dry-run 实测：requested batch 因 baseline 审计缺失
正确阻塞（exit 1，错误信息干净）；不含 baseline 的临时 4 实验 batch
（010/019/025/034 各一台）dry-run exit 0，034 的 queue/record/marker/Stage 1
来源链全部通过校验。正式 batch 尚未启动。

## 0a. 正式 batch r1/r2 事故与跟进（2026-09-11）

**事故 1：数据 fingerprint 闸门拦下 r1 全部 20 个任务。** Root cause：远程
Xeon 8255C 有 AVX-512，numpy 2.4.4 的 `log1p`/`expm1` 分派到 X86_V4 SIMD 路径，
与本机（i5-14400F，无 AVX-512）的 baseline 路径在最后 1 ULP 上不同，JKP prior
链（`log1p→rolling→expm1`）把差异带进了数据；两端 qlib 原始数据 md5 相同、
特征/标签列逐 bit 相同，差异只在 prior 13 列。修复：worker 数据生成固定
`NPY_DISABLE_CPU_FEATURES=X86_V4`（`worker.py`），fingerprint 检查保持严格逐
字节、不放容差。预验证：完整重建后三个 split fingerprint 与本机参考逐 bit
一致，随后 r2 五台机器全部通过闸门。

**事故 2：autodl-2080ti-2 GPU 硬件故障。** 010 的 4 个 seed 在 epoch 0 报
`CUDA illegal memory access` / `cuDNN EXECUTION FAILED`（出现在 layer_norm/GRU/
dropout 等不同算子）；独立最小健全性测试（纯 matmul+softmax 循环）立即复现，
而同软件栈的另外 4 台正常训练。判定为该实例 GPU 硬件故障，与代码无关。
用户已关闭该实例并新开 autodl-4090d-1（RTX 4090 D，GPU 健全性测试通过）；
010 由 batch `followup-010-4090d` 补跑。machines.yaml 保留 2080ti-2 条目
（含故障注释）以维持 r2 manifest 可恢复。

## 0b. 正式 batch 完成记录（2026-09-11）

Batch `baseline-010-019-025-034-r2` 与 `followup-010-4090d` 共同完成 5 个实验
× seeds 1–4 的正式 multi-seed 执行，全部真实走通：远程数据重建 + 严格
fingerprint 闸门、真实 GPU `Trainer.fit` 审计（actual seed/global_step/fit
完成逐任务核对）、产物回传、本机 acceptance（含 seed0/new-seed index/label
一致性 compare_prediction）、experiment completion 通知、ack、本机统一
backtest 评估与 mean/std(ddof=1)/n 汇总、逐台关机请求。

- r2：baseline/019/025/034 各 4 seed accepted（16 receipts），010 因事故 2
  记 failed-of-record，batch-incomplete 如实上报；4 台机器关机请求后 SSH
  不可达（状态按协议记 `requested_unconfirmed`，无云 API 确认）。
- followup-010-4090d：010 4 seed accepted（4 receipts），batch-completed；
  autodl-4090d-1 关机请求后 SSH 不可达。
- 汇总（n=5，含只读 seed0）：见两 batch 的 `reports/summary.json`；034
  IC 0.0399±0.0026 / Sharpe 0.9126±0.2437 为五实验最高。
- 跨 GPU 数值说明：r2 在 2080 Ti、followup 在 4090 D 训练；不同 GPU 的数值
  差异未消除（README 既有声明），各实验内部 4 个新 seed 均在同一台机器完成。

## 1. 实际运行的测试与结果

| 测试 | 命令 | 结果 |
| --- | --- | --- |
| Phase 3 单元/生命周期测试 | `CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_phase3 -v` | 24 tests OK（修复后复跑亦 OK，约 34s） |
| 仓库全量回归 | `CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v` | 106 tests OK（含 test_runner / test_notify / test_stage2_freeze / test_backtest_mdd / test_dataset_schema / test_protocol_metrics） |
| 静态 dry-run（示例 batch） | `coordinator dry-run --batch configs/batch.example.yaml --machines configs/machines.yaml` | exit 0，返回 010/019，不联系远程 |
| 静态 dry-run（requested batch） | `coordinator dry-run --batch configs/requested-batch.yaml ...` | exit 1，干净报错 `baseline requires exact full commit`（baseline 兼容性证据故意留空，符合预期；修复前是 TypeError 崩溃） |
| status 子命令 | `coordinator status --batch batch.example.yaml ...` | exit 0，只读本机状态 |

单元测试真实覆盖的行为（tests/test_phase3.py，含真实本地 worker 端到端）：

- Batch → Device → Ordered Experiments → Ordered Seeds 的 spec 校验与顺序保持；
  重复分配、seed 0、非正整数/重复 seed、空 seeds 全部拒绝。
- 单 device 内多 experiment 多 seed 严格串行（真实 worker，evidence mtime 递增验证）。
- 两个 device 真实并发（线程屏障证明同时进入，不只是线程池语义）。
- 实验级 completion event 在 acceptance 之后、ack 之前发出（动作序列断言
  `notify → ack`，且 transfer 失败时不发 notify）。
- Transfer 失败后 resume 不重训 ready seed（train 仅一次）。
- Accepted receipt 跨 coordinator 重启加载，重复 accept 幂等不改 archive mtime。
- 通知失败不改变 receipt、不阻断 acceptance，事件记 `failed` 并限速重试。
- fake/inference-only 与 fixture 证据不能通过正式 acceptance（provenance 拒绝）。
- 数据 fingerprint 对特征值/标签/index/columns/split 任一变化敏感。
- 真实 tiny CPU Lightning `Trainer.fit` 的 seed 审计与 acceptance；篡改 config seed 被拒。
- Stage 1 三模块张量与冻结状态校验；篡改张量被拒。
- seed0 tree（含 W&B symlink 链接文本）指纹化且不跟随写入；冲突不覆盖已有发布。
- local 无 shutdown 能力（类层面无方法 + 配置拒绝）；remote identity 不符禁止关机；
  shutdown 需要全部 receipt；shutdown 与 experiment completion 是独立事件；
  不传 `--allow-shutdown` 时只 dry-run 且不下发任何命令。
- 设备全部 accepted 后，evaluation 恢复完全不联系已离线机器。
- 远程路径不在本机 stat；`..` 逃逸拒绝；orphan device lease 拒绝重复启动 worker。

## 2. 真实远程验证（autodl-2080ti-3，无卡模式）

Diagnostic id：`cpu-check-20260911-a`。机器身份与 `configs/machines.yaml` 一致
（hostname `autodl-container-36da11a152-33f42d7d`），远程 Python 3.11.15 +
torch 2.8.0 + pandas 2.3.3，`nvidia-smi` 确认无卡，`/root/autodl-tmp` 余量 49G。

第一次运行真实走通完整闭环：本机 coordinator → SSH → rsync staging →
远程 detached worker → 单 device 两个 experiment（fixtureA、fixtureB）→
每 experiment 两个 CPU seed 严格串行（远程 evidence mtime 递增：
A1 < A2 < B1 < B2）→ artifact pull → 本机 acceptance（4 个 receipt，
真实 inspect 校验）→ experiment completion event ×2（fixture 下 suppressed，
不发真实邮件）→ 验收 acknowledgement 上传远程（worker 据此继续下一实验）→
worker 正常退出（status finished，无 active lock）→ shutdown 仅 dry_run，
远程机器确认未关机。

相同 diagnostic id 第二次运行：4 个 task 全部通过 receipt 加载，
远程 4 个 seed 的 evidence mtime 与 attempt 号完全不变——**没有重训**，
resume/幂等验证通过。

## 3. 通知链路真实验证

当前机器 SMTP 凭据只存在于 `~/.bashrc`，本机日常 shell（非 login）环境中
三个变量均缺失，`bash -lc` 下三者齐全。Astra 的实现直接继承环境调用
notify.py，在非 login shell 下会**稳定丢失全部通知**（规则要求的静默失败）。

修复：coordinator 通过 login shell（`bash -lc` + `shlex.join` 引用）调用
notify.py，与 Phase 2 文档约定一致。修复后在剥离三个环境变量的进程内，
经 `Coordinator.event` 真实发送自检邮件成功（事件状态 `sent`），
收件箱实际收到测试邮件。远程不会得到 SMTP 凭据（通知只在本机发出）。

## 4. 本轮修复的问题

1. `preflight.py`：baseline `commit: null`（YAML 空值）触发
   `TypeError: expected string or bytes-like object` 崩溃，而非干净的
   Phase3Error。修为 `b.get('commit') or ''`；requested-batch dry-run 现返回
   明确错误信息 + exit 1。
2. `coordinator.py`：通知改用 login shell 调用 notify.py（见第 3 节）。
3. `README.md`：通知凭据加载说明与实现同步；补建本文件（此前 README
   引用 VALIDATION.md 但文件不存在）。

未改动任何冻结实验研究逻辑、Phase 1/2 语义或 Phase 3 架构。

## 5. 只有 unit/mock 验证的能力

- 正式（非 fixture）acceptance 的 `compare_prediction`（seed0 index/label
  一致性）：fixture 路径跳过，正式路径只有代码审查，无真实执行。
- 本机 evaluation/backtest/aggregate（`evaluate()`）：fixture 诊断不执行；
  单元测试只覆盖 aggregate 统计（ddof=1、n、单观测 std=null）与恢复签名冲突。
- `--retry-failed` 真实远程重训失败 attempt：只有本地模拟。
- SSH 中断后的恢复行为：只有单元级模拟，未真实拔线验证。
- 通知限速重试、事件 exactly-once 边界：单点验证过 sent/failed 状态持久化，
  未做 SMTP 网络不确定性测试。

## 6. 必须等有 GPU 后才能验证的能力

- 真实 `stage2.py` 正式训练审计（`runtime.train` 的完整路径，含 Stage 1
  张量冻结前后核对、best checkpoint 约束）——单元测试只跑 tiny Lightning CPU
  模型并 mock 了 environment/verify_stage1_state。
- 远程数据重建（`dataset.get_dataset`）与完整 sampler 数据 fingerprint
  比对（本机全量 fingerprint 也尚未对真实 canonical 数据执行过）。
- 软件/GPU 环境一致性检查（fixture 诊断跳过）。
- 不同 GPU 数值稳定性（README 已声明本版不覆盖）。
- Baseline seed0 兼容性人工审计（`requested-batch.yaml` 的 commit 与
  compatibility_report 仍为空，baseline 正确地被阻止）。

## 7. Remote shutdown 当前能确认的程度

- 代码路径：能力只在 Remote 类、需要 machine 配置 `ssh_poweroff` + 显式
  `--allow-shutdown`；全部 receipt 持久化、worker 退出、无其他设备 lease、
  identity 复核通过后才请求。
- 真实验证只到 dry_run（诊断中确认 `power/autodl-2080ti-3.json` 为
  `dry_run`，远程机器未关机、未执行任何关机命令）。
- `requested_unconfirmed` 语义只有 mock 验证；未接入 AutoDL 实例状态 API，
  永远不能据此声称已关机或停止计费。真实关机命令本轮未执行（规则禁止）。

## 8. 正式使用前剩余 blocker

1. **GPU 训练链路**：五台 2080 Ti 已上线，但真实 `stage2.py` 正式训练审计、
   远程数据重建 + 全量数据 fingerprint 比对仍未执行过；首个正式 batch 即首验。
2. **Baseline 兼容性审计**：需人工完成并填写 requested-batch 的 commit /
   compatibility_report（含 `identity(inventory(run, allow_symlinks=True))`
   的 seed0_tree_sha256），否则 baseline 任务正确阻塞。
3. **完整数据 fingerprint 首跑**：本机 reference fingerprint 尚未对真实
   canonical 数据计算，首跑 preflight 需要可观 CPU 时间。
4. **多 GPU 数值一致性**：正式上线前需人工验证（README 既有声明）。
5. 可选：AutoDL 实例状态 API 接入前，关机确认只能靠人工。
