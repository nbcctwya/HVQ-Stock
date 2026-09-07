# 013 — alphamaster-ema-market-adapter

## Idea

基于 012 的 prediction-side discrete historical market conditioning，将标准
VQ codebook 的普通梯度更新改为 Exponential Moving Average（EMA）更新，
decay 固定为 `0.99`：

`market history -> GRU -> continuous market state -> EMA VQ -> quantized market state -> Market Adapter -> prediction residual`。

## Motivation

012 通过 codebook loss 对 regime prototypes 做普通梯度更新。本实验只验证：
用 assignment cluster count 与 embedding sum 的 EMA 更新代替普通梯度后，
historical market regime prototypes 是否更稳定，并改善 prediction-side
market adaptation。

## Modification

- 将 012 的 `StandardVectorQuantizer` 替换为 `EMAVectorQuantizer`；默认配置
  `market_quantizer.type: ema_vq`、`decay: 0.99`。
- codebook embedding 不再接收梯度；训练态 forward 根据 L2 assignment 的
  per-code count 与 embedding sum 更新 EMA buffers，再由两者的比值更新
  prototype。每个 code 使用一个初始 pseudo-observation，使未命中 prototype
  保持不变并避免首次归一化不稳定。
- validation/eval forward 不更新 codebook；EMA count/sum 均进入 checkpoint，
  Stage 2 以 strict 方式加载。
- 012 的 VQ loss 标量仍为
  `detached codebook MSE + 0.25 * commitment MSE`，从而保持训练日志、
  validation 与 checkpoint selection 尺度不变；codebook MSE 不产生梯度，
  codebook 正式更新只来自 EMA。
- 更新 AlphaMaster 单元测试与 smoke，覆盖精确 EMA 公式、train/eval 隔离、
  optimizer/EMA 更新隔离、STE/commitment gradient、strict checkpoint 与完整
  Stage 1 -> Stage 2 最小流程。

## Constraints

- 唯一实验变量：012 的 codebook 更新机制由普通梯度更新改为 EMA 更新，
  decay 为 `0.99`。
- codebook size 保持 8、embedding dimension 保持 63、assignment 保持平方
  L2 nearest-neighbor、straight-through estimator 保持不变、commitment
  weight 保持 0.25。
- 012 的前 19 日 historical market slice、单层单向 GRU
  （63 -> 63，batch-first，dropout 0）、quantized state 接入位置、Market
  Adapter `Linear(63,256,bias=False)` 及 zero initialization 均不变。
- current-market Feature Gate 仍且仅使用最后一日 market63；MASTER backbone、
  decoder 及 `y_base + sum(delta_w_t * h)` residual 形式均与 012 一致。
- canonical `158 stock + 13 prior + 63 market + 10 returns = 244` schema、
  `T=20`、prior13 unused、模型维度、attention/dropout、CSI300/SP500 beta、
  `target_day=5`、Adam `lr=8e-6` 均不变。
- 数据划分、70 epoch 预算、early stopping patience 15、Stage 1 seed 42、
  Stage 2 seed 0、指标与 Top30/Drop5 回测协议均不变；Phase 1 未启动正式训练。
- Stage 1 provenance 为 `self`：本实验改变 VQ 正式训练机制与 checkpoint
  state，必须重新训练完整模型。

## Git

Base: exp/012-alphamaster-discrete-market-adapter
Branch: exp/013-alphamaster-ema-market-adapter
Commit: 2a019b02b1dd624fdc44ce218568f7a8ad6df89d
Stage 1 provenance: self

## Smoke Test

Status: PASS

Notes: conda `prism-vq` 下完整单元测试 93/93 PASS。EMA 机制测试验证
`decay=0.99` 的 count/sum 与 prototype 更新值、未命中 code 稳定、eval 不变、
embedding `requires_grad=False` 且 optimizer step 不改写 codebook；L2 assignment、
8×63 维度、STE identity gradient、commitment weight 0.25 及 012 loss 标量公式
均通过。

CPU smoke 限制为 1 epoch、2 train batches、2 validation batches，生成
`artifacts/013/smoke/checkpoints/alphamaster_smoke-epoch=0-val_loss=0.5791.ckpt`；
checkpoint 包含 `ema_cluster_size [8]` 与 `ema_embedding_sum [8,63]`，Stage 2
strict load PASS，生成标准 40 行 prediction
`artifacts/013/smoke/res/alphamaster_csi300/0_best.pkl` 与 `0_metric.csv`，并被
现有 `backtest_qlib.py` normalizer 接受。训练后 10 个测试交易日 code counts
为 `[0,0,4,1,0,0,1,4]`，4/8 active；所有日横截面均共享一个 regime。

真实 012 smoke checkpoint 对 013 strict load 按预期失败，缺少两个 EMA
buffers；013 自身 checkpoint strict load missing=0 / unexpected=0。因此 012
checkpoint 不可复用，`stage1_source: self` 与完整重训要求一致。完整报告与
日志位于 `artifacts/013/smoke/smoke_report.json`、`stage1.log`、`stage2.log`。

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
