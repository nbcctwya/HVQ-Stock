# PRISM-VQ Baseline — Corrected Protocol seed0

本文件记录 PRISM-VQ 原始 baseline（单层 VQ512 + fixed fusion）在
Corrected Protocol 下的正式 seed0 结果。baseline 不是 queue 实验
（实验 ID 从 001 开始），其结果记录在本文件中。

## Corrected Protocol

与 001/002/003 完全统一：corrected Stage 2 freeze（encoder/quantizer/RevIN
全程 eval）+ corrected MDD（初始 NAV=1 计入 peak）；Stage 2 seed 0；
CSI300；test 2023-01-01 – 2025-12-31；回测 Top30/Drop5，
open 0.0005 / close 0.0015，min_cost 0，close 成交，CN limit=None。

## Stage 1 provenance

复用 PRISM-VQ 原始 Stage 1 checkpoint（不受本轮修复影响，未重训）：

`PRISM-VQ/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`

注：config 与历史记录中引用的 `epoch=9-val_loss=0.5682.ckpt` 已不存在；
epoch=7 是同一原始 PRISM-VQ Stage 1 训练当前唯一可用的正式 checkpoint，
strict 加载验证通过（missing=0 / unexpected=0）。这一实例差异会影响与
历史 baseline 数字的精确可比性。

## 执行方式

在 `main`（原始 PRISM-VQ 代码 + corrected freeze + artifact_root）上直接运行：

```
stage2.py train.seed=0 artifact_root=artifacts/baseline/run \
  predictor.saved_model="<PRISM-VQ/checkpoints/infucsi300...epoch=7...ckpt>"
backtest_qlib.py --pred_path artifacts/baseline/run/res/<run>/0_best.pkl \
  --universe csi300 --start_time 2023-01-01 --end_time 2025-12-31 \
  --topk 30 --drop 5 --open_cost 0.0005 --close_cost 0.0015 --min_cost 0
```

## Result（seed 0，test 2023-01-01 – 2025-12-31）

IC: 0.0373
ICIR: 0.2214
RankIC: 0.0552
RankICIR: 0.3313

Annual Return: 9.24%（基准 6.40%，超额 2.84%，AR 差值口径）
Sharpe: 0.5223
Sortino: 0.7690
MDD: -26.90%
Calmar: 0.3436
Turnover: 0.3295

产物：`artifacts/baseline/run/`（checkpoints/、res/、stage2.log、
backtest.log；prediction 与 metric 在 `res/VQK512_csi300_.../`）。

## Historical（不再作为正式比较依据）

历史 baseline 数字（PRISM-VQ 旧代码 + 旧 stage1 epoch=9 ckpt + 旧回测
实现，`PRISM-VQ/res/` 与 `PRISM-VQ/res/backtest_summary_top30_drop5.csv`）
产生于不同的 stage1 checkpoint 实例与旧代码路径，不作为 corrected
protocol 的比较依据。
