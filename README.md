# 035 — combine-025-031-market-gated-stage1

## Base

`exp/025-combine-010-019-shared-routed-quant-confidence`（PRISM-VQ + Stage 2
Shared-Routed MoE + Quantization Confidence Adapter）。

本分支直接从冻结的 025 实验分支创建；031 的 market-gated MASTER-style
Stage 1 代码（`module/layers/encoder.py`、`module/autoencoder.py`）原样
移植，Market Gate 与四个 attention 组件（PositionalEncoding、TAttention、
SAttention、TemporalAttention）与 `AlphaMaster/src/alphamaster/model.py`
保持 AST 级一致，不含任何计算逻辑改动。

Stage 1 不重新训练，复用实验 031 的正式 Stage 1 provenance
（`artifacts/031/run/.stage1.done`），其指向 031 的正式 checkpoint：

`artifacts/031/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=15-val_loss=0.5977.ckpt`

该文件为 13,494,709 bytes（MD5 `41158ad69c45a0c62a792342acba6510`）。
SpatialEncoder、Quantizer 与 RevIN 均 strict 加载验证（missing=0、
unexpected=0）；模型保持 single VQ512、128 维 embedding 与原数据划分。

## Idea / Motivation

025 的 Stage 2（Shared-Routed MoE + Quantization Confidence Adapter）建立
在 PRISM-VQ 原始 Stage 1（GRU + CrossAssetTransformer）产出的 latent 之
上；031 证明 market-gated MASTER-style Stage 1（Market Gate ->
TAttention -> SAttention -> TemporalAttention）可以作为同等接口的
Stage 1 latent 来源，且 Market Gate 让 latent formation 显式条件化于当
日市场状态。两种 Stage 1 的 latent formation 机制不同（循环+跨资产
Transformer vs. 市场门控的 MASTER 时间/空间 attention），其 codebook
utilization 与 latent geometry 也可能不同。

若 025 的 Stage 2 机制（共享/路由分解 + quantization-confidence latent
correction）与 Stage 1 的具体实现解耦，则它应能直接迁移到 031 的
market-gated MASTER-style latent 上并保持收益；反之则说明 025 的收益依
赖于 PRISM-VQ 原始 latent 的特定性质。035 检验的正是这种 latent
formation 与 Stage 2 utilization 之间的互补性/可迁移性，与 033（025 +
027）构成"无 Market Gate vs 有 Market Gate"的对照。

## 核心修改

- Stage 1 整体替换：SpatialEncoder 由 PRISM-VQ 原始的 GRU +
  CrossAssetTransformer 换为 031 的 market-gated MASTER-style 结构
  （`Gate(market[:, -1, :]) * x -> FeatureTransform ->
  MASTERStyleEncoder(x2y -> pe -> tatten -> satten -> temporalatten) ->
  out_layer(Linear(128,512)->GELU->Linear(512,128))`），无 GRU；
  `module/layers/encoder.py` 与 `module/autoencoder.py` 从 031 原样移植
  （仅更新模块 docstring 与报错文案的实验号表述）。
- `trainer/train_vqvae.py` 与 `utils/test.py` 从 031 原样移植
  （market_feature 透传）。
- `trainer/train_ypred.py` 仅合并两组 hunk：027 系的 encoder config 透传
  （`encoder_cfg` 局部变量与 `SpatialEncoder` 的
  `encoder_type/temporal_num_heads/spatial_num_heads/temporal_dropout/
  spatial_dropout` kwargs）与 031 的 market 透传
  （`market_gate_cfg`/`market_input_dim`/`market_beta`、`_get_data` 返回
  market_feature、`forward(feature, prior_factor, market_feature)`）；
  025 的 `quantization_error`、`build_stage2_latent`、strict
  `load_pretrained_vqvae`、freeze/eval 逻辑、loss 与其余代码一字不动。
  market feature 只进入 frozen Stage 1 encoder，不进入任何 Stage 2 模块。
- `configs/config.yaml` 仅在 `vqvae.encoder` 下加入 `type: 'master'`、
  `temporal_num_heads: 2`、`spatial_num_heads: 2`、`temporal_dropout: 0.1`、
  `spatial_dropout: 0.1`、`market_gate: {input_dim: 63, beta: {csi300: 10,
  sp500: 5}}`（与 031 的 config 一致）；025 的 `shared_expert: true` 与
  `quantization_confidence_adapter: true` 保持不变。
- Stage 1（encoder/quantizer/revin）全部 frozen 且强制 eval，只训练
  Stage 2；Stage 1 checkpoint 经 strict 加载（missing=0、unexpected=0）。
- Stage 2 与 025 完全一致：always-on Shared Expert（不参与 routing、不占
  top-k quota）、2 个 Routed Experts、top-k = 1、
  zero-init `Linear(1, 128)` quantization-confidence adapter，
  `z_conf = z_q.detach() + adapter(q_error)`，
  `q_error = mean((h.detach() - z_q.detach())^2, dim=-1, keepdim=True)`。
- 测试：移植 031 的 `tests/test_master_stage1_encoder.py`、
  `tests/test_stage2_freeze.py`、`tests/test_dataset_schema.py`；既有 025
  组合测试补充 market_feature 透传后保留；新增
  `tests/test_combine_025_031_market_gated_stage1.py`。脚本：移植 031 的
  `scripts/smoke_master_stage1_encoder.py`；新增
  `scripts/smoke_combine_025_031_market_gated_stage1.py`。

## 与 base（025）的区别

唯一实验变量是 Stage 1：025 复用 PRISM-VQ 原始 Stage 1，035 换为 031 的
frozen market-gated MASTER-style Stage 1（market feature 经 Market Gate
进入 frozen Stage 1 encoder，不进入 Stage 2）。

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
  PASS（含 031 移植的 encoder/freeze/schema 测试、既有 025/010 机制测试
  与新增 035 组合测试）；031 正式 checkpoint 存在，strict-load 测试实际
  执行未跳过。
- `scripts/smoke_combine_025_031_market_gated_stage1.py`：PASS。覆盖 031
  Stage 1 provenance（marker commit 与 031 Final Experiment Commit 一致、
  正式 checkpoint 存在且 13,494,709 bytes、MD5 不变）、market-gated
  MASTER-style encoder 结构（数据流 RevIN -> Market Gate -> Feature
  Transform -> Input Projection -> PositionalEncoding -> TAttention ->
  SAttention -> TemporalAttention -> Projection MLP，无 GRU；Gate 与四个
  attention 组件同 AlphaMaster 源 AST 一致；Gate 只读
  market_feature[:, -1, :]）、strict load（encoder/quantizer/revin
  missing=0、unexpected=0）、zero-init `z_conf` 与 `z_q` 逐位相等、
  adapter 非零梯度与参数更新、Stage 1 梯度隔离与 codebook/assignment 不
  变、Stage 2 checkpoint strict round-trip、标准 prediction 及 backtest
  normalizer。
- 产物位于 `artifacts/035/smoke/`：`smoke_report.json`、`checkpoints/` 与
  `res/`。

本阶段未启动正式长时间训练或正式回测。
