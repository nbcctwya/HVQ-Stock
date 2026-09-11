# 037 — combine-028-010-shared-routed-stage2

## Base

`exp/034-combine-025-028-temporal-attention-stage1`（028 temporal-attention
Stage 1 + 025 Stage 2 = 010 Shared-Routed MoE + 019 Quantization
Confidence Adapter）。

本分支直接从冻结的 034 实验分支创建。Stage 1 不重新训练，复用实验 028
的正式 Stage 1 provenance（`artifacts/028/run/.stage1.done`），其指向
028 的正式 checkpoint：

`artifacts/028/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=5-val_loss=0.5865.ckpt`

该文件为 14,756,745 bytes。SpatialEncoder、Quantizer 与 RevIN 均 strict
加载验证（missing=0、unexpected=0）；模型保持 single VQ512、128 维
embedding 与原数据划分。

## Idea / Motivation

034 的 Stage 2 在 010 Shared-Routed MoE 之上叠加了来自 019 的
Quantization Confidence Adapter（zero-init `Linear(1, 128)`，
`z_conf = z_q.detach() + adapter(q_error)`，
`q_error = mean((h.detach() - z_q.detach())^2, dim=-1, keepdim=True)`），
即 Confidence-aware Latent Adaptation。037 是 034 的 **w/o Confidence
消融**：移除该 adapter，使 Stage 2 的 latent 恢复为原始的
`z_q.detach()`，从而隔离并量化 Confidence-aware Latent Adaptation 在
028 temporal-attention Stage 1 上的边际贡献。结果模型严格等价于
实验 028 的 temporal-attention Stage 1 + 实验 010 的 Shared-Routed MoE
Stage 2。

## 核心修改

- `trainer/train_ypred.py`：删除 zero-init `Linear(1, 128)` adapter
  模块及其构造分支（`use_quantization_confidence_adapter` flag 一并移
  除），删除 `quantization_error` 与 `build_stage2_latent` 辅助方法；
  `forward` 中 Stage 2 latent 恢复为 010 的原始构造方式
  （`z_q = z_q.detach()`，直接供 `loadings` 与 `latent_value_head`
  消费并作为返回值）。除此之外与 034 一字不差。
- `configs/config.yaml`：删除 `predictor.quantization_confidence_adapter`
  配置键；其余配置（`shared_expert: true`、`n_expert: 2`、encoder
  type、`train.seed: 0`、数据划分、训练预算等）一律不动。
- 测试：删除 025 实验的 adapter 专项测试
  `tests/test_combine_010_019_shared_routed_quant_confidence.py`（adapter
  已不存在）；将 034 的组合测试改造为本实验的组合测试
  `tests/test_combine_028_010_shared_routed_stage2.py`，验证默认 config
  无 adapter、模型无 adapter 模块/参数/辅助方法、Stage 2 latent 与
  `z_q` 逐位相等、Stage 2 forward/backward 正常且 Stage 1 frozen、028
  正式 Stage 1 checkpoint strict 加载（missing=0、unexpected=0）。
- Smoke：删除随 adapter 一并失效的
  `scripts/smoke_combine_010_019_shared_routed_quant_confidence.py` 与
  `scripts/smoke_combine_025_028_temporal_attention_stage1.py`，新增
  `scripts/smoke_combine_028_010_shared_routed_stage2.py`。

保持不变的 034 条件：Stage 1 为 028 的 frozen temporal-attention
encoder（`FeatureTransform -> TemporalAttentionEncoder(input_projection
-> positional_encoding -> tattention -> temporal_aggregation) ->
CrossAssetTransformer`，无 GRU、无 SAttention、无 Market Gate）；
Stage 2 为 010 的 Shared-Routed MoE（always-on Shared Expert 不参与
routing、不占 top-k quota，2 个 Routed Experts，top-k = 1，router/
noise network/W_h/SparseDispatcher/auxiliary loss 不变）；Stage 1
freeze/eval 语义与 strict checkpoint 加载；loss 组合、softcap、
aux_weight。

## 与 base（034）的区别

唯一实验变量是 Quantization Confidence Adapter 的移除：034 的 Stage 2
latent 为 `z_conf = z_q.detach() + adapter(q_error)`，037 的 Stage 2
latent 为原始 `z_q.detach()`（无任何修正项，与 010 原始 Shared-Routed
MoE 实验的 latent 构造完全一致）。

保持不变的 034 条件见上节；数据划分（train 2009–2020、valid
2021–2022、test 2023–2025）、70 epoch 预算、early stopping、Stage 2
seed 0、Top30/Drop5 回测协议及其他超参数均与 034 一致。

## Smoke 状态

Status: **PASS**。

- `conda run -n prism-vq python -m unittest discover -s tests -v`：全量
  PASS（122 个测试，含 028 移植的 encoder 测试、既有公共回归测试与新增
  037 组合测试）。
- `scripts/smoke_combine_028_010_shared_routed_stage2.py`：PASS。覆盖
  028 Stage 1 provenance（marker commit 与 028 Final Experiment Commit
  一致、正式 checkpoint 存在且大小不变）、temporal-attention encoder
  结构（数据流 feature_transform->input_projection->positional_encoding
  ->tattention->temporal_aggregation->cross_asset_transformer，无 GRU、
  无 SAttention、无 Market Gate）、strict load（encoder/quantizer/revin
  missing=0、unexpected=0）、adapter 完全缺席、Stage 2 latent 与 `z_q`
  逐位相等（训练步前后均成立）、真实 backward + optimizer step（Stage 2
  shared/routed 参数有梯度、Stage 1 无梯度保持 eval、codebook 与
  quantizer assignment 不变）、Stage 2 checkpoint strict round-trip、
  标准 prediction 及 backtest normalizer。
- 产物位于 `artifacts/037/smoke/`：`smoke_report.json`、`smoke.log`、
  `unit_tests.log`、`checkpoints/` 与 `res/`。

本阶段未启动正式长时间训练或正式回测。
