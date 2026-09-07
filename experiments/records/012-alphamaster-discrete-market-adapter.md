# 012 — alphamaster-discrete-market-adapter

## Idea

基于 011 的 prediction-side historical market conditioning，将前 19 个交易日
market63 经原 GRU 得到的连续状态 `m_t [N,63]` 先通过标准 Vector
Quantization 压缩为离散 regime `z_q,t [N,63]`，再输入原 zero-initialized
Market Adapter：

`prediction = decoder(h) + sum(MarketAdapter(VQ(GRU(market[:, :-1, :]))) * h)`。

## Motivation

验证相较 011 的连续 historical market state，将市场状态压缩为少量可复用
的离散 regime，能否过滤市场噪声并改善 prediction-side adaptation。同一
trading day 的完整股票横截面共享同一 historical market window，因此共享
同一个 code assignment 和 quantized market representation。

## Modification

- 在 011 的 `TemporalMarketEncoder` 与 `Market Adapter` 之间新增标准 VQ：
  codebook size 8、embedding dimension 63、平方 L2 最近邻。
- 前向使用标准 straight-through estimator；selected embedding 是精确前向
  值，prediction gradient 以 identity 形式传至 GRU，不直接经 STE 更新
  codebook。
- 新增标准 VQ loss：`codebook MSE + 0.25 * commitment MSE`，与原 prediction
  MSE 相加用于 Stage 1 train/validation objective；codebook 由 codebook loss
  更新，GRU 同时接收 commitment 与 prediction-side straight-through 梯度。
- 默认 `configs/config.yaml` 完整声明并启用 VQ，不依赖实验特有 CLI override。
- 扩展单元测试与 smoke，覆盖 VQ shape/config/loss/STE、同日共享、路径隔离、
  VQ/GRU/Adapter 梯度与更新、strict checkpoint、Stage 1 → Stage 2、标准预测
  与 backtest 接口，以及 trading-day level codebook usage diagnostics。

## Constraints

- 唯一实验变量：在 011 的 GRU continuous market state 与原 Market Adapter
  之间加入上述标准 VQ 及其标准 loss。
- 011 的单层 GRU 保持 `input_size=63`、`hidden_size=63`、`num_layers=1`、
  `batch_first=True`、unidirectional、dropout 0。
- Market Adapter 仍为 `Linear(63,256,bias=False)` 并显式 zero-init；decoder
  residual 形式仍为 `y_base + sum(delta_w_t * h)`。
- current-market Feature Gate 仍且仅使用 `market[:, -1, :]`；historical 分支
  仍且仅使用 `market[:, :-1, :]`。
- 原 Feature Gate、MASTER backbone、原 decoder、canonical
  `158 stock + 13 prior + 63 market + 10 returns = 244` schema、`T=20` 和
  prior13 不进入模型的行为均不变。
- `d_model=256`、attention heads/dropout、CSI300/SP500 beta、`target_day=5`、
  Adam `lr=8e-6`、数据划分、70 epoch 预算、patience 15、Stage 1 seed 42、
  Stage 2 seed 0、指标及 Top30/Drop5 回测协议均与 011 一致。
- Stage 1 provenance 为 `self`：012 新增可训练 VQ 并改变正式 forward graph，
  必须重新训练完整模型；Phase 1 未启动正式长时间训练。

## Git

Base: exp/011-alphamaster-continuous-market-adapter
Branch: exp/012-alphamaster-discrete-market-adapter
Commit: 01e8eecd5e36de50a7dbf4604bdb1ad3d2d894c5
Stage 1 provenance: self

## Smoke Test

Status: PASS

Notes: `prism-vq` 环境下完整单元测试 92/92 PASS。机制测试验证 canonical
`[N,20,244]`、stock `[N,20,158]`、market `[N,20,63]`、history
`[N,19,63]`、GRU state / VQ input / quantized state `[N,63]`、adapter input
`[N,63]`、`delta_w_t [N,256]` 和 prediction `[N]`；VQ 配置为 `K=8,D=63`、
L2、STE、commitment weight 0.25。VQ total/codebook/commitment loss 均有限，
STE 梯度语义验证通过，VQ codebook、GRU 和 Adapter 均获得非零梯度并在一步
optimizer step 后更新。

current/history 双路径 hook 验证 Feature Gate 只读最后一天、GRU 只读前
19 天；受控 history 改变会切换 regime 并改变 adapter residual，而 current
day 改变不影响 GRU/VQ/Adapter 路径。同日完整横截面的 quantized state 和
code index 逐位共享。zero-init 时 residual 为零，prediction 与原 decoder
输出最大绝对误差为 0.0。

CPU smoke 限制为 1 epoch、2 train batches、2 validation batches，生成
`artifacts/012/smoke/checkpoints/alphamaster_smoke-epoch=0-val_loss=0.5845.ckpt`；
Stage 2 strict load PASS，生成标准 40 行 prediction
`artifacts/012/smoke/res/alphamaster_csi300/0_best.pkl` 与 `0_metric.csv`，并被
现有 `backtest_qlib.py` normalizer 接受。训练后按 trading-day 统计 10 个
assignment，code counts 为 `[0,0,4,1,1,0,2,2]`，active codes 为
`[2,3,4,6,7]`，即 5/8 active；所有日横截面均只使用一个共享 regime。
完整报告和日志位于 `artifacts/012/smoke/smoke_report.json`、`stage1.log`、
`stage2.log`。

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

