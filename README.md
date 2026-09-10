# 034 — combine-025-028-temporal-attention-stage1

## Base

`exp/025-combine-010-019-shared-routed-quant-confidence`（PRISM-VQ + Stage 2
Shared-Routed MoE + Quantization Confidence Adapter）。

本分支直接从冻结的 025 实验分支创建；028 的 temporal-attention Stage 1
代码（`module/layers/encoder.py`、`module/autoencoder.py`）原样移植，
三个 attention 组件（PositionalEncoding、TAttention、TemporalAttention）
与 `AlphaMaster/src/alphamaster/model.py` 保持 AST 级一致，不含任何
计算逻辑改动。

Stage 1 不重新训练，复用实验 028 的正式 Stage 1 provenance
（`artifacts/028/run/.stage1.done`），其指向 028 的正式 checkpoint：

`artifacts/028/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=5-val_loss=0.5865.ckpt`

该文件为 14,756,745 bytes。SpatialEncoder、Quantizer 与 RevIN 均 strict
加载验证（missing=0、unexpected=0）；模型保持 single VQ512、128 维
embedding 与原数据划分。

## Idea / Motivation

025 的 Stage 2（Shared-Routed MoE + Quantization Confidence Adapter）建立
在 PRISM-VQ 原始 Stage 1（GRU + CrossAssetTransformer）产出的 latent 之
上；028 证明 temporal-attention Stage 1（`Input Projection ->
PositionalEncoding -> TAttention -> TemporalAttention`，后接原
CrossAssetTransformer，无 GRU、无 SAttention、无 Market Gate）可以作为
同等接口的 Stage 1 latent 来源。两种 Stage 1 的 latent formation 机制
不同（GRU 单隐状态压缩 vs. 保留完整时间维度的 attention 聚合），其
codebook utilization 与 latent geometry 也可能不同。

若 025 的 Stage 2 机制（共享/路由分解 + quantization-confidence latent
correction）与 Stage 1 的具体实现解耦，则它应能直接迁移到 028 的
temporal-attention latent 上并保持收益；反之则说明 025 的收益依赖于
PRISM-VQ 原始 latent 的特定性质。034 检验的正是 025 的 Stage 2 与 028
的 temporal-attention Stage 1 之间的互补性/可迁移性。

## 核心修改

- Stage 1 整体替换：SpatialEncoder 由 PRISM-VQ 原始的 GRU +
  CrossAssetTransformer 换为 028 的 temporal-attention 结构
  （`FeatureTransform -> TemporalAttentionEncoder(input_projection ->
  positional_encoding -> tattention -> temporal_aggregation) ->
  CrossAssetTransformer(含原 out_layer Linear(128,512)->GELU->
  Linear(512,128))`），无 GRU、无 SAttention、无 Market Gate；
  `module/layers/encoder.py` 与 `module/autoencoder.py` 从 028 原样移植
  （仅报错文案中的实验号由 028 改为 034）。
- `trainer/train_ypred.py` 仅合并 028 的两处 config 读取 hunk
  （`encoder_cfg` 局部变量与 `SpatialEncoder` 的
  `encoder_type/temporal_dropout` kwargs）；025 的
  `quantization_error`、`build_stage2_latent`、strict
  `load_pretrained_vqvae`、freeze 逻辑、loss 等全部不变。
- `configs/config.yaml` 仅在 `vqvae.encoder` 下加入
  `type: 'temporal-attention'` 与 `temporal_dropout: 0.1`（与 028 的
  config 一致）。
- Stage 1（encoder/quantizer/revin）全部 frozen 且强制 eval，只训练
  Stage 2；Stage 1 checkpoint 经 strict 加载（missing=0、unexpected=0）。
- Stage 2 与 025 完全一致：always-on Shared Expert（不参与 routing、不占
  top-k quota）、2 个 Routed Experts、top-k = 1、
  zero-init `Linear(1, 128)` quantization-confidence adapter，
  `z_conf = z_q.detach() + adapter(q_error)`，
  `q_error = mean((h.detach() - z_q.detach())^2, dim=-1, keepdim=True)`。

## 与 base（025）的区别

唯一实验变量是 Stage 1：025 复用 PRISM-VQ 原始 Stage 1（经由 010 的
external baseline checkpoint），034 换为 028 的 frozen
temporal-attention Stage 1。

保持不变的 025 条件：Stage 2 全部模块与机制（Shared-Routed MoE、
Quantization Confidence Adapter、Temporal Transformer、HyperFusion、
LatentValueHead、ReturnPredictor、loss 组合、softcap、aux_weight）、
single VQ512 与 128 维 embedding、Stage 1 freeze/eval 语义、strict
checkpoint 加载。数据划分（train 2009–2020、valid 2021–2022、test
2023–2025）、70 epoch 预算、early stopping、Stage 2 seed 0、Top30/Drop5
回测协议及其他超参数均与 025 一致。

## Smoke 状态

Status: **PASS**。

- `conda run -n prism-vq python -m unittest discover -s tests -v`：全量
  PASS（含 028 移植的 encoder 测试、既有 025/010 机制测试与新增 034 组合
  测试）。
- `scripts/smoke_combine_025_028_temporal_attention_stage1.py`：PASS。覆盖
  028 Stage 1 provenance（marker commit 与 028 Final Experiment Commit
  一致、正式 checkpoint 存在且大小不变）、temporal-attention encoder
  结构（数据流 feature_transform->input_projection->positional_encoding
  ->tattention->temporal_aggregation->cross_asset_transformer，无 GRU、
  无 SAttention、无 Market Gate）、strict load（encoder/quantizer/revin
  missing=0、unexpected=0）、zero-init `z_conf` 与 `z_q` 逐位相等、
  adapter 非零梯度与参数更新、Stage 1 梯度隔离与 codebook/assignment
  不变、Stage 2 checkpoint strict round-trip、标准 prediction 及
  backtest normalizer。
- 产物位于 `artifacts/034/smoke/`：`smoke_report.json`、`checkpoints/` 与
  `res/`。

本阶段未启动正式长时间训练或正式回测。
