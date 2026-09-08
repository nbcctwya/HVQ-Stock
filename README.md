# 017 — shared-routed-decoupling

## Base

`exp/010-prism-shared-routed-moe`。

本实验只修改 Stage 2，Stage 1 明确复用实验 010 的 exact checkpoint。010 的
`artifacts/010/run/.stage1.done` 所记录 commit 与 canonical queue pinned
commit 均为 `9b854f0436f8a7c3283fd375661dd6152cc965f1`；marker 指向 corrected
PRISM-VQ baseline 的 single VQ512 checkpoint：

`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`

该 checkpoint 为 14,584,929 bytes。RevIN、SpatialEncoder、VectorQuantiser
strict 加载均为 missing=0、unexpected=0，且数据划分与 010 一致。

## Idea / Motivation

010 用 always-on Shared Expert 学习 common structure，并用 `z_q`-conditioned
Routed Experts 学习 latent-state-specific structure，但固定相加本身无法阻止
两条路径学习重复表示。本实验显式惩罚同一 sample 上两条路径输出的相关性：

```text
moe_out = shared_out + routed_out
L_dec = mean(cosine_similarity(shared_out, routed_out)^2)
L_moe = L_route + 0.01 * L_dec
```

假设是轻量的 Shared–Routed Decoupling Regularization 能促进互补表示和更清晰
的 expert specialization，从而改善预测与投资组合表现。

## 核心修改

- `FactorGatedMoE` 对每个 sample 的 `shared_out` 与 `routed_out` 沿表示维计算
  cosine similarity，平方后对 batch 取均值。
- cosine 计算提升到 float32，并使用 `eps=1e-8` 稳定零范数情形；loss 有限且
  非负。
- 固定 `lambda_dec=0.01`，仅将 `0.01 * L_dec` 追加到原 `L_route`。
- 默认 `configs/config.yaml` 设置 `predictor.decoupling_lambda: 0.01`，无需
  实验特有 CLI override。
- 新增机制回归测试和最小 Stage 2 smoke 脚本；没有新增可训练参数。

## 与 base 的区别

唯一实验变量是新增上述 decoupling auxiliary loss。010 的 prediction forward
融合仍逐字保持 `shared_out + routed_out`；Shared Expert、Routed Experts、router、
noisy top-k、2 experts、`k=1`、`W_h`、SparseDispatcher、expert combine 以及
原 importance/load-balancing loss 的定义和权重均不变。

Stage 1 single VQ512、canonical `158 stock + 13 prior + 63 market + 10 returns`
schema、market63 unused 行为、DLinear、Temporal Transformer、FiLM、alpha/beta
heads、LatentValueHead、ReturnPredictor、loss family、数据划分、70 epoch 预算、
early stopping、Stage 1 seed 42、Stage 2 seed 0、Top30/Drop5 回测协议与其他
超参数均保持 010 不变；没有 adaptive fusion 或其他新机制。

## Smoke 状态

Status: **PASS**。

- `conda run -n prism-vq python -m unittest discover -s tests -v`：98/98 PASS；
  其中 017 新增测试 7/7 PASS，010 Shared/Routed 回归测试 9/9 PASS。
- `scripts/smoke_shared_routed_decoupling.py`：PASS。验证了非零 Shared output
  条件下 prediction 与 010 逐位一致、route loss 逐位不变、组合 loss 公式、
  decoupling 双路径有效梯度、真实 Stage 2 backward/update、Stage 1 strict
  兼容、checkpoint strict round-trip、valid/test inference 和标准回测输入格式。
- 产物：`artifacts/017/smoke/`（`unit_tests.log`、`stage2.log`、
  `smoke_report.json`、`checkpoints/`、`res/`）。

本阶段未启动正式长时间训练或正式回测。
