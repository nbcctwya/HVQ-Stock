# 020 — market-conditioned-routing

## Base

`main`（原始 corrected PRISM-VQ baseline）。

Stage 1 不重新训练，复用 corrected PRISM-VQ exact checkpoint：

`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`

该文件为 14,584,929 bytes，MD5 为
`6b9d9dbfd938c7bd2c7dc5ee33cb38af`。SpatialEncoder、Quantizer 与 RevIN
均已 strict 加载验证（missing=0、unexpected=0）；模型保持 single VQ512、
128 维 embedding 与原数据划分。

## Idea / Motivation

原 PRISM-VQ 的 MoE 只根据个股离散 latent state `z_q` 选择 expert，即建模
`P(expert | z_q)`。同一种 stock latent state 在不同市场 regime 下可能对应不同的
最优 expert specialization，因此本实验检验加入当前市场状态后建模
`P(expert | z_q, market)` 是否更合适。

canonical market 输入为 `[B, T, 63]`，严格只取窗口最近时点：

```text
m_t = market_feature[:, -1, :]
delta_logits = W_m * Norm(m_t)
clean_logits = Router(z_q) + delta_logits
```

## 核心修改

- 对 `m_t` 做逐样本、无可训练参数的 63 维 LayerNorm 标准化。
- `W_m` 严格为单层 `Linear(63, n_expert, bias=False)`，weight 显式全零初始化。
- `delta_logits` 只加到原 `clean_logits`；原 noise network、noisy top-k、`W_h`、
  softmax、SparseDispatcher、experts 与 importance/load-balancing loss 流程不变。
- adapter 构造使用隔离的 RNG 上下文，避免消耗后续 baseline 模块的初始化随机数；
  相同 seed 下，除新增 adapter 外全部既有 state tensor 与 `main` 逐位一致。
- 默认 `configs/config.yaml` 设置 `predictor.market_conditioned_routing: true`，
  无需实验特有 CLI override。

## 与 base 的区别

唯一实验变量是上述 market-conditioned additive routing bias。zero-init 时对任意
合法 market input 都有 `delta_logits = 0`，clean logits、expert routing、原
auxiliary loss 与完整 prediction forward 均与 `main` 逐位相等；adapter 学习后，
不同 market state 可以改变 routing logits 与 expert allocation。

market63 不进入 expert input、DLinear、Temporal Transformer、HyperFusion heads、
prior/latent factor heads、ReturnPredictor 或任何其他 prediction 路径。不加入
market temporal encoder、Shared Expert、adaptive fusion、decoupling、quantization
confidence、prior-latent gating 或其他机制。

Stage 1 encoder、single VQ512 quantizer、RevIN、codebook、assignment 与 loss 全部
保持不变并始终冻结。canonical `158 stock + 13 prior + 63 market + 10 returns = 244`
schema、数据划分（train 2009–2020、valid 2021–2022、test 2023–2025）、70 epoch
预算、early stopping、Stage 1 seed 42、Stage 2 seed 0、其他超参数及 Top30/Drop5
回测协议均保持 `main` 不变。

## Smoke 状态

Status: **PASS**。

- `conda run -n prism-vq python -m unittest discover -s tests -v`：91/91 PASS；
  020 新增机制测试 9/9 PASS，既有 Stage 1 freeze 回归测试继续 PASS。
- `scripts/smoke_market_conditioned_routing.py`：PASS。覆盖 external Stage 1
  provenance、single VQ512 与数据划分、strict load、canonical market63 与最近
  时点提取、parameter-free normalization、adapter zero-init、clean logits / routing /
  auxiliary loss / 完整 prediction forward 逐位等价、既有初始化不受扰动、原
  noisy top-k / noise / `W_h` / load 行为、非零 adapter 改变 allocation、真实
  backward 与 optimizer step、Stage 1 梯度隔离、checkpoint strict round-trip、
  标准 prediction 及 backtest normalizer。
- smoke 中 adapter weight gradient L1 为 `0.6355803609`，并在 optimizer step 后
  更新；Stage 1 参数无梯度。
- 产物位于 `artifacts/020/smoke/`：`unit_tests.log`、`stage2.log`、
  `smoke_report.json`、`checkpoints/` 与 `res/`。

本阶段未启动正式长时间训练或正式回测。
