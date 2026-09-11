# 037 — combine-028-010-shared-routed-stage2

## Idea

严格以实验 034 `combine-025-028-temporal-attention-stage1` 为 base，
**仅移除其中来自实验 019 的 Quantization Confidence Adapter**，使
Stage 2 latent 恢复为原始 `z_q`（`z_q.detach()`，不再叠加
`adapter(q_error)` 修正项）。其余一切与 034 完全一致。结果模型严格
等价于：

- Stage 1：实验 028 的 temporal-attention Stage 1
  （`FeatureTransform -> TemporalAttentionEncoder(Input Projection ->
  PositionalEncoding -> TAttention -> TemporalAttention) ->
  原 PRISM CrossAssetTransformer -> VQ`，frozen，复用 028 正式
  checkpoint）；
- Stage 2：实验 010 的 Shared-Routed MoE（always-on Shared Expert、
  2 个 Routed Experts、top-k = 1、原 router/noise network/W_h/
  SparseDispatcher/auxiliary loss）。

## Motivation

本实验是 034 的 w/o Confidence 消融。034 在 028 temporal-attention
Stage 1 之上叠加了 025 的完整 Stage 2（Shared-Routed MoE +
Quantization Confidence Adapter），其中 confidence-aware latent
adaptation 的边际贡献与 MoE 结构本身的贡献无法区分。通过移除 adapter
（Stage 2 latent 恢复为原始 `z_q`），本实验与 034 构成严格的
controlled pair（唯一差异 = 有无 Confidence Adapter），用于量化
Confidence-aware Latent Adaptation 在 temporal-attention latent 上的
边际贡献。

## Modification

- 从冻结的 `exp/034-combine-025-028-temporal-attention-stage1` 创建
  实验分支。
- `trainer/train_ypred.py`：删除 `GenerateReturn.__init__` 中的
  `use_quantization_confidence_adapter` flag 与 zero-init
  `Linear(1, 128)` adapter 模块；删除 `quantization_error` 与
  `build_stage2_latent` 方法；`forward` 中 Stage 2 latent 恢复为
  010 的原始构造 `z_q = z_q.detach()`，`loadings`、
  `latent_value_head` 与返回值均直接使用原始 `z_q`。经与 010 Final
  Experiment Commit 对照，Stage 2 latent 构造与 010 完全一致。
- 默认 `configs/config.yaml`：删除
  `predictor.quantization_confidence_adapter: true` 及其注释；其余
  配置（`shared_expert: true`、`vqvae.encoder.type:
  'temporal-attention'`、`temporal_dropout: 0.1`、`train.seed: 0`、
  数据划分、训练预算等）一律不动。默认 config 直接代表本实验，核心
  改动不依赖任何 CLI override。
- 删除随 adapter 一并失效的 025 adapter 专项测试
  `tests/test_combine_010_019_shared_routed_quant_confidence.py` 与
  两个测试已移除 adapter 的 smoke 脚本
  （`scripts/smoke_combine_010_019_shared_routed_quant_confidence.py`、
  `scripts/smoke_combine_025_028_temporal_attention_stage1.py`）。
- 新增 `tests/test_combine_028_010_shared_routed_stage2.py`（由 034
  组合测试改造）与 smoke 脚本
  `scripts/smoke_combine_028_010_shared_routed_stage2.py`；分支根目录
  README 改写为本实验说明。

## Constraints

- 唯一实验变量：移除 034 中的 Quantization Confidence Adapter；除此
  之外与 034 保持逐行一致，不做任何顺手重构。
- 完整保留 028 的 temporal-attention Stage 1（无 GRU、无 SAttention、
  无 Market Gate）与 010 的 Shared-Routed MoE，不加入其他机制。
- Stage 1 来源：复用实验 028 的正式 run Stage 1 checkpoint
  （`stage1_source: "028"`），不重新训练 Stage 1；Stage 1 全部
  frozen 且强制 eval，不进入 optimizer。
- RevIN、VectorQuantiser、single VQ512、128 维 codebook 与 028 一致；
  strict 加载 028 的 RevIN、encoder、quantizer。
- 数据划分（train 2009–2020、valid 2021–2022、test 2023–2025）、70
  epoch 预算、early stopping、optimizer、learning rate、Stage 2
  seed 0、Stage 1 seed 42、Top30/Drop5 回测协议及其他超参数均与
  034 一致。
- 不做超参数搜索；Phase 1 未启动正式长时间训练或正式回测。

## Git

Base: exp/034-combine-025-028-temporal-attention-stage1
Branch: exp/037-combine-028-010-shared-routed-stage2
Commit: 086d98fff392567c5a24751933a1fcd51926b401
Stage 1 provenance: 复用实验 028 的正式 Stage 1（`stage1_source:
"028"`）；`artifacts/028/run/.stage1.done` marker 存在，其记录的
commit `4494d99542f40be7d3136ab42836f306631f0584` 与 canonical queue
中 028 pinned commit 完全一致；marker 指向 028 self-trained 正式
checkpoint
`artifacts/028/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=5-val_loss=0.5865.ckpt`。

## Smoke Test

Status: PASS

Notes:

- `conda run -n prism-vq python -m unittest discover -s tests -v`：
  全量 122/122 PASS（无 skip、无 fail），028 正式 checkpoint
  strict-load 测试真实执行。日志位于
  `artifacts/037/smoke/unit_tests.log`。
- `conda run -n prism-vq python scripts/smoke_combine_028_010_shared_routed_stage2.py`：
  PASS。验证 028 Stage 1 provenance：marker 存在、marker commit 与
  028 Final Experiment Commit 一致、正式 checkpoint 存在且
  14,756,745 bytes；Encoder、Quantizer、RevIN strict load
  （missing=0 / unexpected=0）；single VQ512、128 维 embedding；
  adapter 完全缺席，Stage 2 latent 在训练步前后均逐位等于原始
  `z_q`；真实 backward + optimizer step：Stage 2（Shared Expert /
  Routed Experts）获得 finite non-zero 梯度并更新，Stage 1 参数无
  梯度、保持 eval、codebook 与 quantizer assignment 不变；Stage 2
  checkpoint strict round-trip 后输出逐位一致；valid/test 推理与
  backtest prediction normalizer 兼容。
- 产物位于 `artifacts/037/smoke/`：`unit_tests.log`、`smoke.log`、
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

正式实验完成后填写。
