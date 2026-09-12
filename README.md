# 038 — combine-028-019-quant-confidence-stage2

## Base

`exp/034-combine-025-028-temporal-attention-stage1`（028 temporal-attention
Stage 1 + 025 Stage 2：010 Shared-Routed MoE + 019 Quantization Confidence
Adapter）。

本分支直接从冻结的 034 实验分支创建。

Stage 1 不重新训练，复用实验 028 的正式 Stage 1 provenance
（`artifacts/028/run/.stage1.done`），其指向 028 的正式 checkpoint：

`artifacts/028/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=5-val_loss=0.5865.ckpt`

该文件为 14,756,745 bytes。Encoder、Quantizer 与 RevIN 均 strict 加载
验证（missing=0、unexpected=0）；模型保持 single VQ512、128 维
embedding 与原数据划分。

## Idea / Motivation

本实验是 034 的 w/o Shared-Routed 消融。034 在 028 temporal-attention
Stage 1 之上叠加了 025 的完整 Stage 2，其中包含两个机制：010 的
always-on Shared Expert（Shared-Routed Latent Utilization）与 019 的
Quantization Confidence Adapter。二者的边际贡献在 034 中无法区分。

通过仅移除 010 引入的 always-on Shared Expert（Stage 2 恢复为 019 的
原始 routed-only FactorGatedMoE），本实验与 034 构成严格的 controlled
pair（唯一差异 = 有无 Shared Expert），用于量化 Shared-Routed Latent
Utilization 在 temporal-attention latent 上的边际贡献。对称地，037
（w/o Confidence）覆盖了另一个方向。

## 核心修改

- 移除 010 的 always-on Shared Expert：`module/layers/moe.py`、
  `module/layers/fusion.py`、`module/bidirectional.py` 三个文件恢复为
  main（即 019 所基于的原始代码），`FactorGatedMoE` 不再创建
  zero-init 的 shared expert，前向输出恢复为纯 routed 组合
  `y = dispatcher.combine(expert_outputs)`。
- `configs/config.yaml`：删除 `predictor.shared_expert: true`；其余配置
  （`quantization_confidence_adapter: true`、`vqvae.encoder.type:
  'temporal-attention'`、`temporal_dropout: 0.1`、`train.seed: 0`、数据
  划分、训练预算等）一律不动。默认 config 直接代表本实验，核心改动不
  依赖任何 CLI override。
- 删除随 Shared Expert 一并失效的 010 专项测试/脚本与 025/034 组合
  测试/脚本（`tests/test_shared_routed_moe.py`、
  `scripts/smoke_shared_routed_moe.py`、
  `tests/test_combine_010_019_shared_routed_quant_confidence.py`、
  `scripts/smoke_combine_010_019_shared_routed_quant_confidence.py`、
  `tests/test_combine_025_028_temporal_attention_stage1.py`、
  `scripts/smoke_combine_025_028_temporal_attention_stage1.py`）。
- 新增 `tests/test_combine_028_019_quant_confidence_stage2.py`（由 034
  组合测试改造，shared-expert 存在性断言翻转为缺席断言）与
  `scripts/smoke_combine_028_019_quant_confidence_stage2.py`。

完整保留不变的机制：

- 028 temporal-attention Stage 1（`FeatureTransform ->
  TemporalAttentionEncoder(Input Projection -> PositionalEncoding ->
  TAttention -> TemporalAttention) -> CrossAssetTransformer`，无 GRU、
  无 SAttention、无 Market Gate），frozen 且强制 eval，strict 加载
  028 正式 checkpoint。
- 019 Quantization Confidence Adapter：zero-init `Linear(1, 128)`，
  `z_conf = z_q.detach() + adapter(q_error)`，
  `q_error = mean((h.detach() - z_q.detach())^2, dim=-1, keepdim=True)`。
- 原 router/noise network/W_h/SparseDispatcher、2 个 Routed Experts、
  top-k = 1、auxiliary loss（`aux_weight`、softcap）等 routed 机制。

## 与 base（034）的区别

唯一实验变量：移除 034 中来自 010 的 always-on Shared Expert，Stage 2
恢复为 019 的原始 routed-only MoE；除此之外与 034 保持逐行一致，不做
任何顺手重构。

保持不变的 034 条件：028 Stage 1 全部模块与 freeze/eval 语义、strict
checkpoint 加载、019 adapter、single VQ512 与 128 维 embedding、
Temporal Transformer、HyperFusion、LatentValueHead、ReturnPredictor、
loss 组合。数据划分（train 2009–2020、valid 2021–2022、test
2023–2025）、70 epoch 预算、early stopping、Stage 2 seed 0、Stage 1
seed 42、Top30/Drop5 回测协议及其他超参数均与 034 一致。

## Smoke 状态

Status: **PASS**。

- `conda run -n prism-vq python -m unittest discover -s tests -v`：全量
  PASS（028 移植的 encoder 测试、既有公共测试与新增 038 组合测试；028
  正式 checkpoint strict-load 测试真实执行）。日志位于
  `artifacts/038/smoke/unit_tests.log`。
- `scripts/smoke_combine_028_019_quant_confidence_stage2.py`：PASS。覆盖
  028 Stage 1 provenance（marker commit 与 028 Final Experiment Commit
  一致、正式 checkpoint 存在且大小不变）、temporal-attention encoder
  结构（无 GRU、无 SAttention、无 Market Gate）、strict load
  （encoder/quantizer/revin missing=0、unexpected=0）、Shared Expert
  完全缺席（模块、参数、config key 均不存在）且 routed 路径（experts
  与 router）保持梯度健康、zero-init `z_conf` 与 `z_q` 逐位相等、
  adapter 非零梯度与参数更新、Stage 1 梯度隔离与 codebook/assignment
  不变、Stage 2 checkpoint strict round-trip、标准 prediction 及
  backtest normalizer。
- 产物位于 `artifacts/038/smoke/`：`unit_tests.log`、`smoke.log`、
  `smoke_report.json`、`checkpoints/` 与 `res/`。

本阶段未启动正式长时间训练或正式回测。
