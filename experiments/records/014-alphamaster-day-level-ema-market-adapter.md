# 014 — alphamaster-day-level-ema-market-adapter

## Idea

基于 013 的 prediction-side historical market EMA VQ，将 codebook statistics
由 cross-section sample level 更新改为 trading-day level 更新。同一交易日的
完整股票横截面继续共享相同 historical market state、code assignment 与
quantized market state，但 EMA count 与 embedding sum 对该日只累计一次：

```text
one trading day
    -> one shared market state
    -> one shared code assignment
    -> EMA count += 1
    -> EMA embedding sum += market_state
```

## Motivation

013 的 canonical daily batch 中，每只股票都带有同一份 previous-19-day market
window，GRU 因而为完整横截面生成相同 market state；原 EMA 实现仍按股票行数
重复累计该状态，使横截面较大的交易日在 prototype 更新中权重更高。本实验验证
historical market regime 是否更适合作为 trading-day level 状态建模，以及按日
等权的 EMA statistics 能否形成语义更清晰、更平滑稳定的 regime prototypes。

## Modification

- `EMAVectorQuantizer` 的训练态 EMA 更新只消费 daily batch 第一行的共享
  `market_state` 和 `code index`，因此每个交易日对命中 code 的 count/sum
  贡献恰好一个 observation；完整横截面的 per-stock quantized output、indices、
  loss 与 STE 路径保持不变。
- 默认 `configs/config.yaml` 新增并启用
  `alphamaster.market_quantizer.statistics_level: trading_day`；trainer 验证该值
  并通过正常构造链传入模型，不依赖实验特有 CLI override。
- 更新 AlphaMaster 测试与 smoke：验证精确 day-level EMA 公式、1-row 与
  257-row 共享横截面的 EMA count/sum/prototype 逐位一致、同日共享 regime、
  checkpoint strict load 和完整 Stage 1 -> Stage 2 最小流程。
- smoke 子进程显式隐藏 CUDA，确保声明的 CPU smoke 不会因自动选择可见 GPU
  而触发 deterministic CuBLAS 环境要求；正式训练的默认 GPU 配置未改变。

## Constraints

- 唯一核心实验变量是 013 EMA codebook statistics 的更新粒度：由每个股票
  sample 改为每个 trading day 一个共享 observation。
- EMA decay 保持 `0.99`；EMA buffers、每个 code 的初始 pseudo-observation、
  prototype 归一化、training/eval 更新隔离和 checkpoint state 均不变。
- VQ 保持 `codebook_size=8`、`embedding_dim=63`、平方 L2 nearest-neighbor、
  straight-through estimator、commitment weight `0.25`；VQ loss 仍为
  `detached codebook MSE + 0.25 * commitment MSE`。
- 同一 trading day 的完整股票横截面仍共享相同 code assignment、quantized
  market state 与 Market Adapter dynamic weight。
- previous-19-day historical market slice、单层单向 GRU（63 -> 63、
  batch-first、dropout 0）、zero-initialized `Linear(63,256,bias=False)` Market
  Adapter、current-market Feature Gate、完整 MASTER backbone、decoder 及
  `y_base + sum(delta_w_t * h)` residual 形式均与 013 一致。
- canonical `158 stock + 13 prior + 63 market + 10 returns = 244` schema、
  `T=20`、prior13 unused、模型维度、attention/dropout、CSI300/SP500 beta、
  `target_day=5`、Adam `lr=8e-6` 均不变。
- 数据划分、70 epoch 预算、early stopping patience 15、Stage 1 seed 42、
  Stage 2 seed 0、指标及 Top30/Drop5 回测协议均与 013 一致；Phase 1 未启动
  正式长时间训练。
- Stage 1 provenance 为 `self`：本实验改变 EMA VQ 的正式训练机制，必须重新
  训练完整模型。013 checkpoint 因参数/buffer shapes 不变而能 strict load 到
  014（实测 PASS），但其 EMA state 来自 sample-level statistics，语义上不兼容
  014 的正式训练机制，因此不得复用为 014 Stage 1。

## Git

Base: exp/013-alphamaster-ema-market-adapter
Branch: exp/014-alphamaster-day-level-ema-market-adapter
Commit: 2fe5c384734ae553d2b6070fe9fbe5002ebfd387
Stage 1 provenance: self

## Smoke Test

Status: PASS

Notes: conda `prism-vq` 下完整单元测试 94/94 PASS；其中 day-level EMA 机制
测试验证 decay `0.99` 的精确 count/sum/prototype 更新，且同一个共享 state
以 1-row 与 257-row 横截面输入时，更新后的 EMA buffers 和 codebook weights
逐位一致。L2 assignment、8×63 维度、STE identity gradient、commitment weight
0.25、冻结 codebook gradient、optimizer/EMA 隔离、eval 不更新、daily sampler
完整横截面、同日 regime 共享及原路径回归均通过。

隔离 CPU smoke 限制为 1 epoch、2 train batches、2 validation batches，生成
`artifacts/014/smoke/checkpoints/alphamaster_smoke-epoch=0-val_loss=0.5833.ckpt`；
checkpoint strict load PASS，Stage 2 生成标准 40 行 prediction
`artifacts/014/smoke/res/alphamaster_csi300/0_best.pkl` 与 `0_metric.csv`，并被
现有 `backtest_qlib.py` normalizer 接受。训练后 10 个测试交易日 code counts
为 `[0,0,4,1,1,0,2,2]`，5/8 active；所有日横截面均共享一个 regime。
完整报告和日志位于 `artifacts/014/smoke/smoke_report.json`、`stage1.log`、
`stage2.log`。

013 smoke checkpoint 对 014 state_dict strict load 实测 missing=0 / unexpected=0；
该结果只确认结构兼容，不改变 sample-level EMA checkpoint 不得用于 014 正式
Stage 1 的结论。

## Result

Status: PENDING

IC:
ICIR:
RankIC:
RankICIR:

Annual Return:
Sharpe:
Sortino:
MDD:
Calmar:
Turnover:

## Conclusion
