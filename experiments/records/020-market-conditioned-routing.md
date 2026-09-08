# 020 — market-conditioned-routing

## Idea

在原始 corrected PRISM-VQ baseline 的 MoE router 上加入当前市场状态的轻量
additive bias。canonical `market_feature` 为 `[B, T, 63]`，严格只取最近时点：

```text
m_t = market_feature[:, -1, :]
delta_logits = W_m * Norm(m_t)
clean_logits = Router(z_q) + delta_logits
```

其中 `Norm` 无可训练参数，`W_m` 为 zero-initialized
`Linear(63, n_expert, bias=False)`。

## Motivation

原 PRISM-VQ 仅建模 `P(expert | z_q)`。同一种 stock latent state 在不同 market
regime 下可能对应不同的最优 expert specialization；本实验验证让 routing 建模
`P(expert | z_q, market)` 是否能改善预测与 expert 分工。

## Modification

- `GenerateReturn` 从 canonical market63 中精确提取 `market_feature[:, -1, :]`。
- 对该 `[B, 63]` market state 做逐样本、无参数的 LayerNorm 标准化，并通过
  `Linear(63, n_expert, bias=False)` 生成 `delta_logits`；weight 显式全零初始化。
- `delta_logits` 仅加到原 router 的 clean logits，后续 noise、noisy top-k、
  `W_h`、softmax、SparseDispatcher、experts 与 importance/load-balancing 流程不变。
- adapter 的构造隔离 RNG 消耗；相同 seed 下，除新增 adapter 外全部既有 state
  tensor 与 `main` 初始化逐位一致。
- 默认 `configs/config.yaml` 设置
  `predictor.market_conditioned_routing: true`，核心改动无需 CLI override。
- 新增 `tests/test_market_conditioned_routing.py` 与
  `scripts/smoke_market_conditioned_routing.py`，并更新 Stage 1 freeze 回归测试及
  inference 对 canonical market63 的传递。

## Constraints

- 唯一实验变量是新增上述 market-conditioned additive routing bias。
- market63 只影响 MoE routing logits，不直接进入 expert input、DLinear、Temporal
  Transformer、HyperFusion heads、factor heads、ReturnPredictor 或其他 prediction
  路径；不使用历史 market encoder。
- 原 router/noise 参数、`W_h`、expert 数量 2、top-k 1、SparseDispatcher、expert
  combine、load-balancing loss 定义及 `aux_weight=0.01` 均保持不变。
- zero-init 时 clean logits、routing/load、原 auxiliary loss、MoE output 与完整
  prediction forward 均与 `main` 逐位相等；adapter 非零后，不同 market state
  能在固定 `z_q` 下改变 routing logits 与 expert allocation。
- Stage 1 SpatialEncoder、RevIN、single VQ512 quantizer、128 维 codebook、
  assignment、loss 与训练逻辑完全不变且始终冻结。
- 不加入 Shared Expert、adaptive fusion、decoupling、quantization confidence、
  prior-latent gating 或其他机制。
- canonical `158 stock + 13 prior + 63 market + 10 returns = 244` schema、数据划分
  （train 2009–2020、valid 2021–2022、test 2023–2025）、70 epoch 预算、early
  stopping、Stage 1 seed 42、Stage 2 seed 0、Top30/Drop5 回测协议及其他超参数
  均与 `main` 一致。
- Phase 1 未启动正式长时间训练或正式回测。

## Git

Base: main
Branch: exp/020-market-conditioned-routing
Commit: 2118eaa3696c21d27e695082f47f8ced2d0c3758
Stage 1 provenance: external corrected PRISM-VQ exact checkpoint：
`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`
（14,584,929 bytes；MD5 `6b9d9dbfd938c7bd2c7dc5ee33cb38af`）。

## Smoke Test

Status: PASS

Notes:

- `conda run -n prism-vq python -m unittest discover -s tests -v`：完整测试
  91/91 PASS；020 新增机制测试 9/9 PASS，既有 Stage 1 freeze 回归测试继续
  PASS。
- external checkpoint 存在且非空；模型保持唯一 `VectorQuantiser`、codebook
  shape `(512, 128)` 与原数据划分。Encoder、Quantizer、RevIN strict load 均为
  missing=0 / unexpected=0。
- 单元测试与 smoke 均确认 canonical `[B,20,63]` 输入及最近时点 `[B,63]`
  提取、无参数标准化、zero-initialized bias-free adapter；初始 clean logits、
  gates/load、MoE output/auxiliary loss 与完整 prediction forward 均与 `main`
  逐位相等。
- 相同 seed 下除新增 adapter 外全部既有 state tensor 初始化逐位相等；原 gate、
  noise、`W_h` 与 noisy top-k/load 行为逐位不变。非零 adapter 在固定 `z_q` 下
  能使相反 market state 分配到不同 expert。
- synthetic canonical `[N,20,244]` smoke 完成真实 backward 与 optimizer step；
  adapter weight gradient L1 为 `0.6355803609` 并更新，Stage 1 参数无梯度。
- Stage 2 checkpoint strict round-trip 后输出逐位一致；标准 12 行 prediction、
  metric CSV 与 backtest prediction normalizer 均 PASS。
- 产物位于 `artifacts/020/smoke/`：`unit_tests.log`、`stage2.log`、
  `smoke_report.json`、`checkpoints/` 与 `res/`。

## Result

Status: DONE（test 区间 2023-01-01 – 2025-12-31）

IC: 0.0335
ICIR: 0.1809
RankIC: 0.0528
RankICIR: 0.2778

Annual Return: 13.31%（基准 6.40%，超额 6.91%）
Sharpe: 0.8651
Sortino: 1.2749
MDD: -13.94%
Calmar: 0.9550
Turnover: 0.3220

## Conclusion

Phase 2 固定执行器完成正式训练、预测与回测（pinned commit 2118eaa3696c21d27e695082f47f8ced2d0c3758）。Stage 1 复用外部 exact checkpoint：`/home/nbcctwya/baselines/masterVQ/HVQ-Stock/artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`（本实验未重新训练 Stage 1）；Stage 2 seed 0。

产物：`artifacts/020/run/`（checkpoints/、res/、stage1.log、stage2.log、backtest.log、summary.json）。
