# 018 — adaptive-shared-decoupling

## Idea

基于实验 016 `latent-adaptive-shared-fusion`，完整保留其 latent-conditioned
adaptive shared fusion：

```text
alpha = 1 + 0.5 * tanh(f(z_q))
moe_out = alpha * shared_out + routed_out
```

在此基础上，对 adaptive scaling 前的 raw `shared_out` 与 `routed_out` 新增
Shared–Routed Decoupling Regularization：

```text
L_dec = mean(cosine_similarity(shared_out, routed_out)^2)
L_total = L_rank + aux_weight * L_aux + 0.01 * L_dec
```

`L_dec` 是独立的 top-level regularization term，不进入原 MoE auxiliary
loss，不经过原 `softcap_log1p`，也不被 `aux_weight` 再次缩放。

## Motivation

016 让不同 latent state 自适应控制 Shared Expert 的贡献强度，但没有显式
约束 Shared / Routed Experts 避免学习重复表示。独立的 squared-cosine
decoupling regularization 可能促进两条 expert 路径学习互补信息；与 adaptive
fusion 结合后，可能同时改善 shared-specific specialization 与下游预测表现。

## Modification

- `FactorGatedMoE` 保留 adaptive scaling 前的 raw `shared_out`，与原 routed
  experts 聚合产生的 `routed_out` 计算 per-sample cosine similarity，平方后对
  batch 取均值。
- cosine 输入提升到 float32，使用 `eps=1e-8`；零范数表示下 penalty 仍为
  finite、non-negative scalar。
- 原 MoE forward 第二返回值保持 016 routed importance/load-balancing loss；
  HyperFusion 仍只把该值与原 beta regularization 合并，随后仍由
  `softcap_log1p` 处理，原 `aux_weight` 不变。
- 仅训练/validation 的内部 loss-components 路径透传独立 `L_dec`；标准
  prediction forward 与返回接口不变。
- training 与 validation 共用顶层 objective builder，严格计算
  `L_rank + aux_weight * L_aux + 0.01 * L_dec`，并分别记录
  `train_decoupling_loss` / `val_decoupling_loss`。
- 默认 `configs/config.yaml` 固定
  `predictor.decoupling_lambda: 0.01`，无需实验特有 CLI override。
- 新增 `tests/test_independent_shared_routed_decoupling.py` 与
  `scripts/smoke_adaptive_shared_decoupling.py`。

## Constraints

- 唯一实验变量是在 016 上新增独立 top-level Shared–Routed Decoupling
  Regularization；未引入 017 将 decoupling 合入 MoE auxiliary loss 的逻辑。
- 016 的 adaptive shared fusion 公式、`delta=0.5`、affine map zero-init、原始
  `z_q` conditioning 与 prediction forward 完全不变。
- Shared Expert、Routed Experts、router、noisy top-k、2 experts、`k=1`、
  `W_h`、SparseDispatcher、expert combine、原 routed importance/load-balancing
  loss、beta regularization、softcap、`aux_weight` 及其他 loss 定义和权重均
  保持 016 不变。
- HyperFusion 后续 FiLM/alpha/beta heads、LatentValueHead、ReturnPredictor 与
  其他 Stage 2 结构不变；不新增可训练参数。相同 seed 下 018 与 016 全部既有
  state_dict 张量逐位一致。
- Stage 1 保持 RevIN、SpatialEncoder、single VQ512、128 维 embedding 与原量化
  配置；来源明确复用实验 010 的 exact checkpoint。
- canonical `158 stock + 13 prior + 63 market + 10 returns = 244` schema、
  market63 unused 行为、数据划分（train 2009–2020、valid 2021–2022、test
  2023–2025）、70 epoch 预算、early stopping、Stage 1 seed 42、Stage 2
  seed 0、Top30/Drop5 回测协议及其他超参数均与 016 一致。
- Phase 1 未启动正式长时间训练或正式回测。

## Git

Base: exp/016-latent-adaptive-shared-fusion
Branch: exp/018-adaptive-shared-decoupling
Commit: 8756e8b6be4f0ca00b4e151198b1915e10c6bb65
Stage 1 provenance: 复用实验 010（queue `stage1_source: "010"`）。010 的
`artifacts/010/run/.stage1.done` 记录 commit
`9b854f0436f8a7c3283fd375661dd6152cc965f1`，与 010 canonical queue pinned
commit 完全一致；marker 指向 corrected PRISM-VQ exact checkpoint：
`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`
（14,584,929 bytes；MD5 `6b9d9dbfd938c7bd2c7dc5ee33cb38af`）。

## Smoke Test

Status: PASS

Notes:

- conda `prism-vq` 下仓库完整单元测试 110/110 PASS；018 新增机制测试
  10/10 PASS，016 adaptive fusion 与 010 Shared/Routed 回归测试继续 PASS。
- 完整 016 配置（移除 018 唯一新增配置项）与 018 配置在相同 seed 下实例化，
  全部 state_dict 键和值逐位相等；完整 prediction forward 与原 auxiliary loss
  逐位相等。
- 测试覆盖 016 adaptive formula、zero-init 与 raw `z_q` conditioning；`L_dec`
  定义、有限性、非负性、零向量稳定性；相同表示 penalty 接近 1，正交表示
  penalty 为 0；Shared/Routed 两条路径均获得有限非零梯度。
- objective 严格满足
  `L_rank + aux_weight * L_aux + 0.01 * L_dec`；显式反例检查确认 `L_dec` 未被
  `aux_weight` 或 `softcap_log1p` 缩放，且 training/validation 共用同一公式并
  单独记录。
- `scripts/smoke_adaptive_shared_decoupling.py` 从实验 010 marker 解析 exact
  Stage 1 checkpoint；single VQ512 配置与数据划分核对 PASS，Encoder、
  Quantizer、RevIN strict load 均 missing=0 / unexpected=0，冻结参数无梯度。
- synthetic canonical `[N,20,244]` Stage 2 smoke 完成真实 forward/backward 与
  optimizer step；`L_dec` 对 Shared/Routed 路径 gradient L1 分别为
  `1412.470458984375` / `8.448188781738281`。
- Stage 2 checkpoint strict save/load 后标准输出逐位一致；valid/test inference、
  标准 `0_best.pkl` / `0_metric.csv` 与 backtest prediction normalizer 均 PASS。
- 产物位于 `artifacts/018/smoke/`：`unit_tests.log`、`stage2.log`、
  `smoke_report.json`、`checkpoints/` 与 `res/`。

## Result

Status: DONE（test 区间 2023-01-01 – 2025-12-31）

IC: 0.0360
ICIR: 0.2010
RankIC: 0.0586
RankICIR: 0.3326

Annual Return: 19.77%（基准 6.40%，超额 13.37%）
Sharpe: 1.1674
Sortino: 1.8575
MDD: -20.32%
Calmar: 0.9727
Turnover: 0.3285

## Conclusion

Phase 2 固定执行器完成正式训练、预测与回测（pinned commit 8756e8b6be4f0ca00b4e151198b1915e10c6bb65）。Stage 1 复用实验 010 的正式 checkpoint：`/home/nbcctwya/baselines/masterVQ/HVQ-Stock/artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`（本实验未重新训练 Stage 1）；Stage 2 seed 0。

产物：`artifacts/018/run/`（checkpoints/、res/、stage1.log、stage2.log、backtest.log、summary.json）。
