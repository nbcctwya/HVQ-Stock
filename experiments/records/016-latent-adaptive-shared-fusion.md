# 016 — latent-adaptive-shared-fusion

## Idea

基于实验 010 `prism-shared-routed-moe`，在现有 Shared Expert + Routed
Experts 架构上加入 latent-conditioned adaptive shared fusion，使不同离散
latent state 能自适应调整 Shared Expert 的贡献：

```text
alpha = 1 + 0.5 * tanh(f(z_q))
moe_out = alpha * shared_out + routed_out
```

`f` 使用单层可学习 affine map 并显式零初始化，使初始化时 `alpha=1`，
forward 严格等价于 010 的固定 `shared_out + routed_out`。

## Motivation

010 已证明显式建模 shared structure 有效，但固定相加隐含所有 latent state
对共享信息依赖程度相同的假设。不同离散 state 可能需要不同强度的 common
structure；用 `z_q` 在有限范围内自适应缩放 Shared Expert，可能进一步改善
shared information 与 routed specialized information 的协同。

## Modification

- `FactorGatedMoE` 在 010 Shared Expert 之后新增且仅新增
  `shared_fusion = Linear(128, 1)`，直接读取原始 `z_q`，生成每个 sample 的
  标量 `alpha = 1 + 0.5 * tanh(shared_fusion(z_q))`。
- `shared_fusion.weight` 与 `shared_fusion.bias` 显式全零初始化；固定
  `delta=0.5`，因此 `alpha` 限制在 `(0.5, 1.5)` 且初始严格为 1。
- 新层构造使用 RNG 隔离，确保相同 seed 下 010 的 routed experts、router、
  noise、`W_h`、Shared Expert 及后续 HyperFusion heads 的既有初始化逐位不变。
- router 仍读取 010 原有的 LayerNorm 后 latent；新增 fusion map 单独读取原始
  `z_q`，不改变 routing 行为。
- `HyperFusion` / `LoadingGenerator` 只透传新开关；默认
  `configs/config.yaml` 设置 `predictor.adaptive_shared_fusion: true`，无需实验
  特有 CLI override。
- 新增 `tests/test_adaptive_shared_fusion.py` 与
  `scripts/smoke_adaptive_shared_fusion.py`。

## Constraints

- 唯一实验变量是将 010 的 `shared_out + routed_out` 改为
  `alpha(z_q) * shared_out + routed_out`。
- 010 的 always-on Shared Expert 结构与零初始化不变；原 PRISM-VQ routed
  experts、router、noisy top-k、2 experts、`k=1`、`W_h`、
  SparseDispatcher、expert combine、importance/load-balancing loss 均不变。
  Shared Expert 仍不参与 routing、不占 top-k quota，且不新增 auxiliary loss。
- HyperFusion 后续 FiLM、alpha/beta heads、prior/latent decomposition、
  LatentValueHead、ReturnPredictor 与其他 Stage 2 结构和 loss 均不变。
- Stage 1 保持 RevIN、SpatialEncoder、single VQ512、128 维 embedding、原量化
  配置与训练逻辑；Stage 1 来源明确复用实验 010 的 exact checkpoint。
- canonical `158 stock + 13 prior + 63 market + 10 returns = 244` schema、
  market63 unused 行为、数据划分（train 2009–2020、valid 2021–2022、test
  2023–2025）、70 epoch 预算、early stopping、Stage 1 seed 42、Stage 2
  seed 0、Top30/Drop5 回测协议及其他超参数均与 010 一致。
- Phase 1 未启动正式长时间训练或正式回测。

## Git

Base: exp/010-prism-shared-routed-moe
Branch: exp/016-latent-adaptive-shared-fusion
Commit: f0157c7c23cbf7c4749ace313c10b21354e307ba
Stage 1 provenance: 复用实验 010（queue `stage1_source: "010"`）。010 的
`artifacts/010/run/.stage1.done` 记录 commit
`9b854f0436f8a7c3283fd375661dd6152cc965f1`，与 010 canonical queue pinned
commit 完全一致；marker 指向 external corrected PRISM-VQ exact checkpoint：
`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`
（14,584,929 bytes；MD5 `6b9d9dbfd938c7bd2c7dc5ee33cb38af`）。

## Smoke Test

Status: PASS

Notes:

- conda `prism-vq` 下仓库完整单元测试 100/100 PASS，其中 016 新增机制测试
  9/9 PASS；010 原 Shared/Routed MoE 测试 9/9 继续 PASS。
- 等价性测试在 Shared Expert 输出被显式设为非零时验证：zero-init map 产生
  `alpha` 逐位为 1，016 MoE output 与 010 fixed-add output 逐位相等，routed
  auxiliary loss 逐位相等；相同 seed 下除新增 map 外全部参数初始化逐位相等。
- `scripts/smoke_adaptive_shared_fusion.py` 从实验 010 marker 解析 exact Stage 1
  checkpoint；single VQ512 配置与数据划分核对 PASS，Encoder、Quantizer、
  RevIN strict load 均 missing=0 / unexpected=0，冻结参数无梯度。
- synthetic canonical `[N,20,244]` Stage 2 smoke 完成两步真实训练：Shared
  Expert 第一步获得非零梯度并更新，fusion map 第二步 weight/bias 梯度 L1 为
  `0.0624448694` / `0.0029215964` 并更新。
- Stage 2 checkpoint strict save/load 后 prediction 逐位一致；valid/test
  inference、标准 `0_best.pkl` / `0_metric.csv` 与 backtest prediction
  normalizer 均 PASS。
- 产物位于 `artifacts/016/smoke/`：`unit_tests.log`、`stage2.log`、
  `smoke_report.json`、`checkpoints/` 与 `res/`。

## Result

Status: DONE（test 区间 2023-01-01 – 2025-12-31）

IC: 0.0361
ICIR: 0.2017
RankIC: 0.0587
RankICIR: 0.3335

Annual Return: 17.93%（基准 6.40%，超额 11.53%）
Sharpe: 1.0779
Sortino: 1.6830
MDD: -20.13%
Calmar: 0.8911
Turnover: 0.3289

## Conclusion

Phase 2 固定执行器完成正式训练、预测与回测（pinned commit f0157c7c23cbf7c4749ace313c10b21354e307ba）。Stage 1 复用实验 010 的正式 checkpoint：`/home/nbcctwya/baselines/masterVQ/HVQ-Stock/artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`（本实验未重新训练 Stage 1）；Stage 2 seed 0。

产物：`artifacts/016/run/`（checkpoints/、res/、stage1.log、stage2.log、backtest.log、summary.json）。
