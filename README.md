# 018 — adaptive-shared-decoupling

## Base

`exp/016-latent-adaptive-shared-fusion`。

Stage 1 不重新训练，复用实验 `010` 的正式 Stage 1 marker 所指向的 exact
checkpoint：

`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`

该来源保持原 PRISM-VQ 的 SpatialEncoder、RevIN、single VQ512 与原数据划分。
实验 010 marker commit 与 canonical queue pinned commit 一致；Encoder、
Quantizer、RevIN 均已 strict 加载验证，missing=0、unexpected=0。

## Idea / Motivation

016 使用原始 `z_q` 自适应控制 Shared Expert 的贡献：

```text
alpha = 1 + 0.5 * tanh(f(z_q))
moe_out = alpha * shared_out + routed_out
```

018 保留该 prediction forward，并对 adaptive scaling 前的 raw
`shared_out` 与 `routed_out` 新增互补性约束：

```text
L_dec = mean(cosine_similarity(shared_out, routed_out)^2)
L_total = L_rank + aux_weight * L_aux + 0.01 * L_dec
```

目标是检验 latent-conditioned adaptive shared fusion 与独立 decoupling
regularization 结合后，是否能减少 Shared / Routed Experts 的重复表示，同时
改善 shared-specific specialization 与下游预测表现。

## 核心修改

- `FactorGatedMoE` 保留 adaptive scaling 前的 raw `shared_out`，与原
  `routed_out` 沿表示维计算 per-sample cosine similarity，平方后对 batch
  取均值。
- cosine 输入提升到 float32，并使用 `eps=1e-8`；零范数输入下 penalty 仍为
  finite、non-negative scalar。
- 标准 MoE / HyperFusion / LoadingGenerator / GenerateReturn prediction
  接口和返回值不变；仅训练/validation 的内部 loss-components 路径额外返回
  `L_dec`。
- `L_dec` 不进入原 MoE auxiliary loss；原 routed importance/load-balancing
  loss 与 beta regularization 仍按 016 原定义合并并经 `softcap_log1p`，然后
  只乘一次原 `aux_weight`。
- training 与 validation 共用同一个顶层 objective builder，并分别记录
  `train_decoupling_loss` / `val_decoupling_loss`。
- 默认 `configs/config.yaml` 固定
  `predictor.decoupling_lambda: 0.01`，无需实验特有 CLI override。

## 与 base 的区别

唯一实验变量是在 016 上新增独立的 top-level Shared–Routed Decoupling
Regularization。016 的
`alpha = 1 + 0.5 * tanh(f(z_q))`、`delta=0.5`、`f` 零初始化、原始 `z_q`
conditioning 及 `alpha * shared_out + routed_out` prediction forward 均保持
不变。

Shared Expert、Routed Experts、router、noisy top-k（2 experts、`k=1`）、
`W_h`、SparseDispatcher、expert combine、HyperFusion 后续 FiLM/alpha/beta
heads、LatentValueHead、ReturnPredictor 与所有模型参数均未新增或改变。相同
seed 下 018 与 016 的全部既有 state_dict 张量逐位一致。

Stage 1、canonical dataset（market63 继续 unused）、数据划分、70 epoch
预算、early stopping、Stage 1 seed 42、Stage 2 seed 0、原 loss 权重与
Top30/Drop5 回测协议均保持 016 不变。本实验不包含 017 的 auxiliary-loss
合并逻辑或其他实现逻辑。

## Smoke 状态

Status: **PASS**。

- `conda run -n prism-vq python -m unittest discover -s tests -v`：110/110
  PASS；018 新增机制测试 10/10 PASS，016 adaptive fusion 与 010
  Shared/Routed 回归测试继续 PASS。
- `scripts/smoke_adaptive_shared_decoupling.py`：PASS；验证完整 016/018 模型
  同 seed 初始化与 prediction/原 aux 逐位一致，adaptive formula、zero-init、
  raw `z_q` conditioning、raw-path `L_dec` 定义、独立顶层 objective、两路径
  梯度、训练/validation objective 一致性。
- 实验 010 Stage 1 marker provenance、single VQ512、数据划分与
  Encoder/Quantizer/RevIN strict load 均 PASS；Stage 1 冻结参数无梯度。
- Stage 2 checkpoint strict save/load、valid/test inference、标准
  prediction/metric 输出和 backtest prediction normalizer 均 PASS。
- 产物位于 `artifacts/018/smoke/`：`unit_tests.log`、`stage2.log`、
  `smoke_report.json`、`checkpoints/` 与 `res/`。

本阶段未启动正式长时间训练或正式回测。
