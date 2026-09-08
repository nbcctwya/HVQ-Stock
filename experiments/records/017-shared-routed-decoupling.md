# 017 — shared-routed-decoupling

## Idea

基于实验 010 `prism-shared-routed-moe`，保持 Shared Expert 与 Routed Experts
的 forward 融合完全不变：

`moe_out = shared_out + routed_out`

仅新增 Shared–Routed Decoupling Regularization。对每个 sample 的两条 expert
输出计算 cosine similarity，并惩罚其平方：

`L_dec = mean(cosine_similarity(shared_out, routed_out)^2)`

MoE loss 改为 `L_moe = L_route + 0.01 * L_dec`。

## Motivation

010 从结构上将 common structure 与 latent-state-specific structure 分配给
Shared / Routed Experts，但没有机制阻止两条路径学习重复信息。显式降低两者
输出相关性，可能促进互补表示与更清晰的 expert specialization，并进一步改善
预测和投资组合表现。

## Modification

- `FactorGatedMoE` 使用实际的 per-sample `shared_out` 与 `routed_out`，沿表示
  维计算 cosine similarity，平方后对 batch 取均值。
- cosine 计算提升到 float32，并使用 `eps=1e-8` 保证零范数输入下数值稳定；
  penalty 为有限、非负 scalar。
- 原 `L_route` 计算完成后仅追加固定 `lambda_dec=0.01` 的 decoupling 项；
  `shared_out + routed_out` prediction forward 未改变。
- `LoadingGenerator` / `HyperFusion` 只透传该固定配置；默认
  `configs/config.yaml` 设置 `predictor.decoupling_lambda: 0.01`，无需实验特有
  CLI override，且不新增任何可训练参数。
- 新增 `tests/test_shared_routed_decoupling.py` 与
  `scripts/smoke_shared_routed_decoupling.py`。

## Constraints

- 唯一实验变量是在 010 上新增 Shared–Routed squared-cosine decoupling loss；
  不改变 prediction forward 融合。
- 010 的 Shared Expert、Routed Experts、router、noisy top-k、2 experts、
  `k=1`、`W_h`、SparseDispatcher、expert combine 及原 routed
  importance/load-balancing loss 定义和权重均不变。
- 不加入 adaptive fusion 或其他新结构；HyperFusion 后续 FiLM、alpha/beta
  heads、LatentValueHead、ReturnPredictor 与其他 Stage 2 loss 保持 010 不变。
- Stage 1 保持 RevIN、SpatialEncoder、single VQ512、128 维 embedding、量化
  配置与训练逻辑；来源明确复用实验 010 的 exact checkpoint。
- canonical `158 stock + 13 prior + 63 market + 10 returns = 244` schema、
  market63 unused 行为、数据划分（train 2009–2020、valid 2021–2022、test
  2023–2025）、70 epoch 预算、early stopping、Stage 1 seed 42、Stage 2
  seed 0、Top30/Drop5 回测协议及其他超参数均与 010 一致。
- Phase 1 未启动正式长时间训练或正式回测。

## Git

Base: exp/010-prism-shared-routed-moe
Branch: exp/017-shared-routed-decoupling
Commit: c82f2f6ec9248835037f8c956772fe58da2d5abe
Stage 1 provenance: 复用实验 010（queue `stage1_source: "010"`）。010 的
`artifacts/010/run/.stage1.done` 记录 commit
`9b854f0436f8a7c3283fd375661dd6152cc965f1`，与 010 canonical queue pinned
commit 完全一致；marker 指向 corrected PRISM-VQ exact checkpoint：
`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`
（14,584,929 bytes；MD5 `6b9d9dbfd938c7bd2c7dc5ee33cb38af`）。

## Smoke Test

Status: PASS

Notes:

- conda `prism-vq` 下仓库完整单元测试 98/98 PASS；017 新增机制测试 7/7
  PASS，010 原 Shared/Routed MoE 测试 9/9 继续 PASS。
- 等价性测试与 smoke 均显式使用非零 Shared Expert output，验证相同参数下
  017 prediction forward 与 010 逐位一致；route loss 逐位不变，且返回 loss
  满足 `L_route + 0.01 * L_dec`。
- 测试覆盖 decoupling 定义、有限性、非负性、零向量稳定性；相同表示 penalty
  为 1，正交表示 penalty 为 0；Shared/Routed 两条表示均获得有限非零梯度。
- `scripts/smoke_shared_routed_decoupling.py` 从实验 010 marker 解析 exact
  Stage 1 checkpoint；single VQ512 配置与数据划分核对 PASS，Encoder、
  Quantizer、RevIN strict load 均 missing=0 / unexpected=0，冻结参数无梯度。
- synthetic canonical `[N,20,244]` Stage 2 smoke 完成真实 backward 与 optimizer
  step；Shared Expert weight gradient L1 为 `64.5884857178` 并成功更新。
- Stage 2 checkpoint strict save/load 后 prediction 逐位一致；valid/test
  inference、标准 `0_best.pkl` / `0_metric.csv` 与 backtest prediction normalizer
  均 PASS。
- 产物位于 `artifacts/017/smoke/`：`unit_tests.log`、`stage2.log`、
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
