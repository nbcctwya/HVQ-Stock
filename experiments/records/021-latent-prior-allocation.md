# 021 — latent-prior-allocation

## Idea

在原始 corrected PRISM-VQ baseline 上加入 Latent-Conditioned Prior–Latent
Adaptive Allocation。使用冻结 Stage 1 的原始离散 latent `z_q` 计算：

```text
g = 0.5 * tanh(Linear(z_q))
prior_scale = 1 + g
latent_scale = 1 - g
y_pred = alpha + prior_scale * prior_term + latent_scale * latent_term
```

其中 gate 严格为 zero-initialized `Linear(128, 1)`。

## Motivation

原始 PRISM-VQ 对所有 stock latent state 固定使用
`alpha + prior_term + latent_term`。expert prior factors 与 learned latent
factors 的相对可靠性可能随离散 stock state 改变；由 `z_q` 自适应地在两类
已计算完成的 factor contribution 之间做互补分配，可能改善收益排序预测，
同时保持总基准尺度与初始化行为不变。

## Modification

- `ReturnPredictor` 新增且仅新增一个读取原始 `z_q` 的
  `Linear(128, 1)` allocation gate；weight 与 bias 显式全零初始化，固定
  `delta=0.5`。
- gate 只作用于已经计算完成的 `prior_term` 与 `latent_term`：
  `prior_scale=1+g`、`latent_scale=1-g`；不修改 `beta_p`、`beta_l` 或 factor
  heads。
- gate 在所有 baseline 模块构造完成后追加；相同 seed 下，排除新增 gate
  后全部既有 state tensor 与 `main` 逐位一致。
- zero-init 时 `g=0`、两种 scale 均精确为 1，完整 prediction forward 与
  `main` 逐位一致。
- 默认 `configs/config.yaml` 设置
  `predictor.latent_conditioned_allocation: true`，无需实验特有 CLI override。
- 新增 `tests/test_latent_prior_allocation.py` 与
  `scripts/smoke_latent_prior_allocation.py`。

## Constraints

- 唯一实验变量是上述 latent-conditioned prior–latent allocation gate。
- allocation gate 不改变 `beta_p`、`beta_l`、prior/latent factor heads、
  HyperFusion、MoE、router、Temporal Transformer、prediction/auxiliary loss
  或 Stage 1。
- 不加入 Shared Expert、adaptive shared fusion、decoupling、quantization
  confidence、market-conditioned/code-aware routing 或其他机制。
- Stage 1 的 SpatialEncoder、RevIN、single VQ512 quantizer、128 维 codebook、
  assignment、loss 与训练逻辑完全不变；`z_q` 已 detach，Encoder、Quantizer、
  RevIN 与 codebook 始终冻结且不接收梯度。
- canonical `158 stock + 13 prior + 63 market + 10 returns = 244` schema、
  market63 unused 行为、数据划分（train 2009–2020、valid 2021–2022、test
  2023–2025）、70 epoch 预算、early stopping、Stage 1 seed 42、Stage 2
  seed 0、Top30/Drop5 回测协议及其他超参数均与 `main` 一致。
- Stage 1 来源为 external corrected PRISM-VQ exact checkpoint；Phase 1 未启动
  正式长时间训练或正式回测。

## Git

Base: main
Branch: exp/021-latent-prior-allocation
Commit: 4d59b167df3b75c720937ba5322e05b2078e20d6
Stage 1 provenance: external corrected PRISM-VQ exact checkpoint：
`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`
（14,584,929 bytes；MD5 `6b9d9dbfd938c7bd2c7dc5ee33cb38af`）。

## Smoke Test

Status: PASS

Notes:

- `conda run -n prism-vq python -m unittest discover -s tests -v`：完整测试
  92/92 PASS；021 新增机制测试 10/10 PASS，既有 Stage 1 freeze、runner、
  protocol/backtest 回归测试继续 PASS。
- external checkpoint 存在且非空；模型保持唯一 `VectorQuantiser`、codebook
  shape `(512, 128)` 与原数据划分。Encoder、Quantizer、RevIN strict load
  均 missing=0 / unexpected=0。
- 单元测试与 smoke 均确认 gate 是 zero-initialized `Linear(128, 1)`、固定
  `delta=0.5`，直接接收 detach 后的原始 `z_q`；初始 scale 均精确为 1，
  完整 prediction forward 与 `main` 逐位一致。
- 相同 seed 下，除新增 gate 外全部既有 state tensor 初始化逐位一致。
  非零 gate 对不同 `z_q` 产生不同 allocation；两种 scale 严格位于
  `(0.5, 1.5)` 且和精确为 2；prior allocation 增强时 latent allocation
  互补减弱，反向同理。
- synthetic canonical `[N,20,244]` smoke 完成真实 backward 与 optimizer
  step；gate weight/bias gradient L1 分别为 `8.4178371429` /
  `0.3462440372` 并成功更新，Stage 1 参数无梯度且始终保持 eval。
- Stage 2 checkpoint strict save/load 后完整输出逐位一致；标准 12 行
  prediction、metric CSV 与 backtest prediction normalizer 均 PASS。
- 产物位于 `artifacts/021/smoke/`：`unit_tests.log`、`stage2.log`、
  `smoke_report.json`、`checkpoints/` 与 `res/`。

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
