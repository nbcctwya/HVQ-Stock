# 015 — alphamaster-vq-continuous-warmup

## Idea

基于 012 `alphamaster-discrete-market-adapter`，保持完整模型从初始化起即包含
相同 Standard VQ，但在前 10 个 epoch 暂时 bypass 量化状态。epoch 0–9 将
GRU continuous historical market state `m_t` 直接送入原 Market Adapter，
只优化 prediction loss；epoch 10 起恢复 012 的
`GRU -> Standard VQ -> Market Adapter` 路径和
`prediction loss + VQ loss` 目标。

## Motivation

验证相比从随机 latent space 立即量化，先让 GRU 和 Market Adapter 学习具有
预测意义的 continuous historical market representation，再进行离散 regime
quantization，是否能提高 Standard VQ 的训练稳定性和最终 prediction-side
market adaptation 效果。

由于 warm-up 与 VQ 阶段的 validation objective 不同，epoch 0–9 的 validation
只用于观察，不参与正式 early stopping 或 best-checkpoint selection；正式状态
从 epoch 10 的首个 VQ validation 开始。

## Modification

- 默认配置新增 `train.warmup_epochs: 10`；总预算仍为 70 epoch，核心改动不
  依赖实验特有 CLI override。
- 使用唯一边界 `use_vq = current_epoch >= warmup_epochs`：
  - epoch 0–9：VQ 模块存在但不被调用，`adapter_input=m_t`，
    `loss=prediction_loss`；codebook 不接收梯度。
  - epoch 10–69：`adapter_input=z_q`，恢复 012 的 Standard VQ forward、STE
    和 `loss=prediction_loss+vq_loss`。
- 新增 warm-up-aware ModelCheckpoint 与 EarlyStopping：epoch 0–9 不更新
  best/wait state、不保存候选；epoch 10 以初始状态开始，之后继续监控原
  `val_loss`、mode=min、`min_delta=1e-5`、patience=15、save_top_k=1。
- epoch 10 validation 聚合并打印机器可读 diagnostics：mean quantization
  distortion、trading-day code usage、active codes、perplexity，以及同一 hidden
  state / GRU state / Adapter 参数下 continuous 与 quantized prediction 的
  MAE、RMSE 和最大绝对差。
- 扩展单元测试与 smoke，覆盖切换边界、两阶段输入和 loss、warm-up 梯度
  隔离、正式回调延迟、switch diagnostics、strict checkpoint、Stage 2 预测和
  backtest normalizer 兼容性。

## Constraints

- 唯一实验变量为 012 Standard VQ 正式介入前的 10 epoch continuous warm-up，
  以及与 objective 边界一致的 early-stopping / best-checkpoint 延迟；其余均与
  012 一致。
- 模型初始化结构不变。受控同 seed 比较确认当前 `MASTER` 与 012 的
  state_dict 名称、shape 和全部初始值逐项相同（36 tensors、841465 values）。
- Standard VQ 保持 codebook size 8、embedding dimension 63、平方 L2 最近邻、
  straight-through estimator、commitment weight 0.25。
- GRU 保持 `input_size=63`、`hidden_size=63`、单层、batch-first、单向、
  dropout 0；Market Adapter 保持 `Linear(63,256,bias=False)` 与 zero-init。
- current-market Feature Gate 仍且仅使用 `market[:, -1, :]`；historical branch
  仍且仅使用 `market[:, :-1, :]`。MASTER backbone、decoder residual、canonical
  244 维 schema、`T=20` 和 prior13 不进入模型的行为不变。
- `d_model=256`、attention heads/dropout、CSI300/SP500 beta、`target_day=5`、
  Adam `lr=8e-6`、数据划分、70 epoch 预算、Stage 1 seed 42、Stage 2 seed 0、
  指标及 Top30/Drop5 回测协议均不变。
- Stage 1 provenance 为 `self`：本实验改变正式训练流程，必须重新训练完整
  模型；Phase 1 未启动正式长时间训练。

## Git

Base: exp/012-alphamaster-discrete-market-adapter
Branch: exp/015-alphamaster-vq-continuous-warmup
Commit: bf4736892a94287de656b655016400bd67885618
Stage 1 provenance: self

## Smoke Test

Status: PASS

Notes: `prism-vq` 环境下完整 `unittest` 96/96 PASS。新增机制测试验证
epoch 0–9 完全 bypass VQ、Adapter 输入为 continuous `m_t`、目标仅为 prediction
loss，GRU/Adapter 获得梯度而 codebook 无梯度且不更新；epoch 10 恢复 012 的
quantized Adapter 输入与完整 VQ loss。回调边界测试确认 checkpoint 与 early
stopping 的父逻辑仅从 epoch 10 开始。

CPU smoke 使用 2 train batches、2 validation batches，实际运行 epoch 0–10。
checkpoint 目录中只有正式合格的
`artifacts/015/smoke/checkpoints/alphamaster_smoke-epoch=10-val_loss=0.5284.ckpt`，
证明 warm-up checkpoint 未参与 selection。epoch 10 switch diagnostics 为：
distortion `0.0889552018`，day-level code usage `[0,0,0,0,1,0,0,1]`，active
codes `2/8`，perplexity `2.0`；continuous/quantized prediction MAE
`0.0409767386`、RMSE `0.0453494099`、max abs `0.0654366612`，均为有限值，
切换未出现数值异常。

该 checkpoint 经 Stage 2 strict load 后生成标准 40 行 prediction
`artifacts/015/smoke/res/alphamaster_csi300/0_best.pkl` 与 metric CSV，并被
现有 `backtest_qlib.py` normalizer 接受。训练后 10 个 test trading days 的
code counts 为 `[0,0,4,1,1,0,2,2]`，5/8 active，所有日横截面均共享单一
regime。完整报告与日志位于
`artifacts/015/smoke/smoke_report.json`、`stage1.log`、`stage2.log`。

## Result

Status: DONE（test 区间 2023-01-01 – 2025-12-31）

IC: 0.0400
ICIR: 0.2319
RankIC: 0.0486
RankICIR: 0.2715

Annual Return: 10.56%（基准 6.40%，超额 4.16%）
Sharpe: 0.5223
Sortino: 0.8435
MDD: -29.68%
Calmar: 0.3557
Turnover: 0.3247

## Conclusion

Phase 2 固定执行器完成正式训练、预测与回测（pinned commit bf4736892a94287de656b655016400bd67885618）。Stage 1 best checkpoint：`alphamaster_csi300_s42-epoch=42-val_loss=1.0983.ckpt`（self-trained）；Stage 2 seed 0。

产物：`artifacts/015/run/`（checkpoints/、res/、stage1.log、stage2.log、backtest.log、summary.json）。
