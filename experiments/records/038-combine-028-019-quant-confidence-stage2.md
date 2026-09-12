# 038 — combine-028-019-quant-confidence-stage2

## Idea

严格以实验 034 `combine-025-028-temporal-attention-stage1` 为 base，
**仅移除其中来自实验 010 的 always-on Shared Expert**，使 Stage 2 恢复
为 019 的原始 routed-only FactorGatedMoE。其余一切与 034 完全一致。
结果模型严格等价于：

- Stage 1：实验 028 的 temporal-attention Stage 1
  （`FeatureTransform -> TemporalAttentionEncoder(Input Projection ->
  PositionalEncoding -> TAttention -> TemporalAttention) ->
  原 PRISM CrossAssetTransformer -> VQ`，frozen，复用 028 正式
  checkpoint）；
- Stage 2：实验 019 的原始 routed-only Stage 2（2 个 Routed Experts、
  top-k = 1、原 router/noise network/W_h/SparseDispatcher/auxiliary
  loss + zero-init `Linear(1, 128)` Quantization Confidence Adapter，
  `z_conf = z_q.detach() + adapter(q_error)`）。

## Motivation

本实验是 034 的 w/o Shared-Routed 消融。034 在 028 temporal-attention
Stage 1 之上叠加了 025 的完整 Stage 2（Shared-Routed MoE +
Quantization Confidence Adapter），其中 Shared-Routed Latent
Utilization（always-on Shared Expert）的边际贡献与 Confidence Adapter
的贡献无法区分。通过移除 Shared Expert（Stage 2 恢复为 019 原始
routed-only 形态），本实验与 034 构成严格的 controlled pair（唯一
差异 = 有无 Shared Expert），用于验证 Shared-Routed Latent
Utilization 在 temporal-attention latent 上的边际贡献；与 037（w/o
Confidence 消融）互为对称的消融方向。

## Modification

- 从冻结的 `exp/034-combine-025-028-temporal-attention-stage1` 创建
  实验分支。
- `module/layers/moe.py`、`module/layers/fusion.py`、
  `module/bidirectional.py` 恢复为 main 版本（即 019 所基于的原始
  routed-only 代码）：删除 `use_shared_expert` 参数与 zero-init
  Shared Expert 模块，前向输出恢复为纯 routed 组合
  `y = dispatcher.combine(expert_outputs)`。
- 默认 `configs/config.yaml`：删除
  `predictor.shared_expert: true` 及其注释；其余配置
  （`quantization_confidence_adapter: true`、`vqvae.encoder.type:
  'temporal-attention'`、`temporal_dropout: 0.1`、`train.seed: 0`、
  数据划分、训练预算等）一律不动。默认 config 直接代表本实验，核心
  改动不依赖任何 CLI override。
- `trainer/train_ypred.py` 逻辑一字不动，仅更新注释中的实验号表述
  （025 -> 038）。
- 删除随 Shared Expert 一并失效的 010 专项测试/脚本与 025/034 组合
  测试/脚本（`tests/test_shared_routed_moe.py`、
  `scripts/smoke_shared_routed_moe.py`、
  `tests/test_combine_010_019_shared_routed_quant_confidence.py`、
  `scripts/smoke_combine_010_019_shared_routed_quant_confidence.py`、
  `tests/test_combine_025_028_temporal_attention_stage1.py`、
  `scripts/smoke_combine_025_028_temporal_attention_stage1.py`）。
- 新增 `tests/test_combine_028_019_quant_confidence_stage2.py`（由 034
  组合测试改造，shared-expert 断言翻转为缺席断言）与 smoke 脚本
  `scripts/smoke_combine_028_019_quant_confidence_stage2.py`；分支根目录
  README 改写为本实验说明。

## Constraints

- 唯一实验变量：移除 034 中来自 010 的 always-on Shared Expert；除此
  之外与 034 保持逐行一致，不做任何顺手重构。
- 完整保留 028 的 temporal-attention Stage 1（无 GRU、无 SAttention、
  无 Market Gate）与 019 的 Quantization Confidence Adapter，以及原
  router、Routed Experts、top-k = 1、auxiliary loss 等机制，不加入
  其他改动。
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
Branch: exp/038-combine-028-019-quant-confidence-stage2
Commit: 0a968f77e717c52cf5c48d997c77c104f7c1563d
Stage 1 provenance: 复用实验 028 的正式 Stage 1（`stage1_source:
"028"`）；`artifacts/028/run/.stage1.done` marker 存在，其记录的
commit `4494d99542f40be7d3136ab42836f306631f0584` 与 canonical queue
中 028 pinned commit 完全一致；marker 指向 028 self-trained 正式
checkpoint
`artifacts/028/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=5-val_loss=0.5865.ckpt`
（14,756,745 bytes）。

## Smoke Test

Status: PASS

Notes:

- `conda run -n prism-vq python -m unittest discover -s tests -v`：
  全量 111/111 PASS（无 skip、无 fail），028 正式 checkpoint
  strict-load 测试真实执行。日志位于
  `artifacts/038/smoke/unit_tests.log`。
- `conda run -n prism-vq python scripts/smoke_combine_028_019_quant_confidence_stage2.py`：
  PASS。验证 028 Stage 1 provenance：marker 存在、marker commit 与
  028 Final Experiment Commit 一致、正式 checkpoint 存在且
  14,756,745 bytes；Encoder、Quantizer、RevIN strict load
  （missing=0 / unexpected=0）；single VQ512、128 维 embedding；
  Shared Expert 完全缺席（模块、参数、config key 均不存在），MoE
  输出为纯 routed 组合；原 router 与 2 个 Routed Experts 获得
  finite non-zero 梯度（router grad L1 1.368，routed experts grad
  L1 380.92）；zero-init 时 `z_conf` 与 `z_q` 逐位相等，adapter
  获得非零梯度并更新；Stage 1 参数无梯度、保持 eval、codebook 与
  quantizer assignment 不变；Stage 2 checkpoint strict round-trip
  后输出逐位一致；valid/test 推理与 backtest prediction
  normalizer 兼容。
- 产物位于 `artifacts/038/smoke/`：`unit_tests.log`、`smoke.log`、
  `smoke_report.json`、`checkpoints/` 与 `res/`。

## Result

Status: DONE（test 区间 2023-01-01 – 2025-12-31）

IC: 0.0381
ICIR: 0.2285
RankIC: 0.0554
RankICIR: 0.3376

Annual Return: 15.83%（基准 6.40%，超额 9.43%）
Sharpe: 0.8932
Sortino: 1.3640
MDD: -22.06%
Calmar: 0.7177
Turnover: 0.3277

## Conclusion

Phase 2 固定执行器完成正式训练、预测与回测（pinned commit 0a968f77e717c52cf5c48d997c77c104f7c1563d）。Stage 1 复用实验 028 的正式 checkpoint：`/home/nbcctwya/baselines/masterVQ/HVQ-Stock/artifacts/028/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=5-val_loss=0.5865.ckpt`（本实验未重新训练 Stage 1）；Stage 2 seed 0。

产物：`artifacts/038/run/`（checkpoints/、res/、stage1.log、stage2.log、backtest.log、summary.json）。
