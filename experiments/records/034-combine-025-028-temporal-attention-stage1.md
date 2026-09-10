# 034 — combine-025-028-temporal-attention-stage1

## Idea

严格以实验 025 `combine-010-019-shared-routed-quant-confidence` 为 base，
完整保留其 Stage 2（010 Shared-Routed MoE + 019 Quantization Confidence
Adapter），仅将 Stage 1 从 025 原本复用的 PRISM-VQ 原始 Stage 1
（GRU + CrossAssetTransformer）替换为实验 028
`prism-temporal-attention-stage1` 已完成正式训练的 frozen
temporal-attention Stage 1：

```text
FeatureTransform -> TemporalAttentionEncoder(Input Projection ->
PositionalEncoding -> TAttention -> TemporalAttention) ->
原 PRISM CrossAssetTransformer -> VQ
```

不重新训练 Stage 1；Stage 1（encoder / quantizer / RevIN）全部 frozen，
只训练 025 的 Stage 2。028 Stage 1 保持其原始结构，不加入 SAttention、
Market Gate 或其他 Stage 1 改动。

## Motivation

025 的 Stage 2 机制（Shared-Routed MoE 的 common/routed 分解 +
quantization-confidence latent correction）建立在 PRISM-VQ 原始 Stage 1
产出的 latent 之上；028 证明 temporal-attention Stage 1 可以作为同等接口
（single VQ512、128 维 embedding、同一数据划分）的 latent 来源。两种
Stage 1 的 latent formation 机制不同（GRU 单隐状态压缩 vs. 保留完整
时间维度的 attention 聚合），本实验验证：

1. 025 的 Stage 2 是否能有效迁移到 028 的 temporal-attention latent
   representation（本实验 vs 025：只看 Stage 1 更换的影响）；
2. 新的 Stage 1 latent formation 与 025 的 latent utilization 是否互补
   （本实验 vs 028：看 025 Stage 2 在 028 Stage 1 上的边际作用）。

## Modification

- 从冻结的 `exp/025-combine-010-019-shared-routed-quant-confidence`
  创建实验分支。
- `module/layers/encoder.py` 与 `module/autoencoder.py` 从 028 的 Final
  Experiment Commit 原样移植（仅更新报错文案中的实验号表述）；三个
  attention 组件（PositionalEncoding、TAttention、TemporalAttention）与
  `AlphaMaster/src/alphamaster/model.py` 保持 AST 级一致，无 GRU、无
  SAttention、无 Market Gate。
- `trainer/train_ypred.py` 仅合并 028 的两处 config 透传 hunk
  （`encoder_cfg` 局部变量与 `SpatialEncoder` 的
  `encoder_type/temporal_dropout` kwargs）；025 的
  `quantization_error`、`build_stage2_latent`、strict
  `load_pretrained_vqvae`、freeze/eval 逻辑、loss 与其余代码一字不动。
- 默认 `configs/config.yaml` 在 `vqvae.encoder` 下加入
  `type: 'temporal-attention'` 与 `temporal_dropout: 0.1`（与 028 的
  config 一致）；025 的 `shared_expert: true` 与
  `quantization_confidence_adapter: true` 保持不变。
- 移植 028 的 `tests/test_temporal_attention_stage1.py`（测试逻辑不变）；
  新增 `tests/test_combine_025_028_temporal_attention_stage1.py` 与
  `scripts/smoke_combine_025_028_temporal_attention_stage1.py`；分支根目录
  README 改写为本实验说明。

## Constraints

- 唯一实验变量：Stage 1 由 025 复用的 PRISM-VQ 原始 Stage 1 更换为 028
  的 frozen temporal-attention Stage 1；除此之外 025 的 Stage 2 结构、
  配置、loss、训练协议全部保持不变。
- Stage 1 来源：复用实验 028 的正式 run Stage 1 checkpoint
  （`stage1_source: "028"`），不使用 smoke checkpoint，不重新训练
  Stage 1。
- 028 Stage 1 保持其原始 `Input Projection -> PositionalEncoding ->
  TAttention -> TemporalAttention -> CrossAssetTransformer` 结构，不加入
  SAttention、Market Gate 或其他 Stage 1 改动。
- 025 的 Shared-Routed MoE（always-on Shared Expert、2 个 Routed
  Experts、top-k = 1、原 router/noise network/W_h/SparseDispatcher/
  auxiliary loss）与 Quantization Confidence Adapter 完整保留：
  `h` 取 028 encoder 进入 VQ 前的 128 维表示，`z_q` 为同一 028
  quantizer 的输出，`q_error = mean((h.detach() - z_q.detach())^2,
  dim=-1, keepdim=True)`，`z_conf = z_q.detach() + adapter(q_error)`
  （adapter 为 zero-init `Linear(1, 128)`）机制不变。
- Stage 1 全部 frozen 且强制 eval，不进入 optimizer；只训练 Stage 2。
- RevIN、VectorQuantiser、single VQ512、128 维 codebook 与 028 一致；
  strict 加载 028 的 RevIN、encoder、quantizer。
- 数据划分（train 2009–2020、valid 2021–2022、test 2023–2025）、70
  epoch 预算、early stopping、optimizer、learning rate、Stage 2 seed 0、
  Stage 1 seed 42、Top30/Drop5 回测协议及其他超参数均与 025/028 一致。
- 不做超参数搜索，不加入其他实验机制；Phase 1 未启动正式长时间训练或
  正式回测。

## Git

Base: exp/025-combine-010-019-shared-routed-quant-confidence
Branch: exp/034-combine-025-028-temporal-attention-stage1
Commit: 2b1f576e4a7d46b098b6278b2fe76672f5881a23
Stage 1 provenance: 复用实验 028 的正式 Stage 1（`stage1_source: "028"`）；
`artifacts/028/run/.stage1.done` marker 存在，其记录的 commit
`4494d99542f40be7d3136ab42836f306631f0584` 与 canonical queue 中 028
pinned commit 完全一致；marker 指向 028 self-trained 正式 checkpoint
`artifacts/028/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=5-val_loss=0.5865.ckpt`
（14,756,745 bytes；MD5 `523677d7794ce822cc8ac89b25c16cfa`）。

## Smoke Test

Status: PASS

Notes:

- `conda run -n prism-vq python -m unittest discover -s tests -v`：全量
  144/144 PASS（既有 116 + 028 移植 encoder 测试 10 + 新增 034 组合测试
  18）；028 正式 checkpoint 存在，strict-load 测试实际执行未跳过。
  日志位于 `artifacts/034/smoke/unit_tests.log`。
- `conda run -n prism-vq python scripts/smoke_combine_025_028_temporal_attention_stage1.py`：
  PASS。验证 028 Stage 1 provenance：marker 存在、marker commit 与 028
  Final Experiment Commit 一致、正式 checkpoint 存在且 14,756,745 bytes、
  MD5 不变；Encoder、Quantizer、RevIN strict load（missing=0 /
  unexpected=0）；encoder 结构为 `Input Projection -> PositionalEncoding
  -> TAttention -> TemporalAttention -> CrossAssetTransformer`，无 GRU、
  无 SAttention、无 Market Gate；single VQ512、128 维 embedding 与原数据
  划分完全兼容。
- zero-init 时 `z_conf` 与 `z_q` 逐位相等；adapter 关闭时完整 forward
  与既有参数初始化不受扰动。synthetic canonical smoke 完成真实
  backward 与 optimizer step：adapter 获得 finite non-zero 梯度并更新，
  Stage 1 参数无梯度、保持 eval、codebook 与 quantizer assignment 不变；
  Shared Expert 与 Routed Experts 仍获得有效训练信号。Stage 2
  checkpoint strict round-trip 后输出逐位一致；标准 prediction、metric
  CSV 与 backtest prediction normalizer 均 PASS。
- smoke 诊断（一步训练后，仅观测不改训练目标）：`q_error` mean
  `0.01546` / std `0.00529`；`||z_conf - z_q||` 均值约 `1.15e-3`；
  relative correction magnitude `3.90e-4`；adapter weight/bias grad L1
  `0.2176` / `17.795`；Shared Expert final weight/bias grad L1
  `107.65` / `19.86`；Routed Experts grad L1 `371.68`。
- 产物位于 `artifacts/034/smoke/`：`unit_tests.log`、`smoke.log`、
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
