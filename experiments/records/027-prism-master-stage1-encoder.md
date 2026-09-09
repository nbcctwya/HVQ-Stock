# 027 — prism-master-stage1-encoder

## Idea

仅替换 PRISM-VQ Stage 1 的时空编码器。将原来的
`GRU temporal summarization + CrossAssetTransformer` 替换为不含 Market
Gate 的 MASTER-style encoder：

```text
Input Projection -> PositionalEncoding -> TAttention -> SAttention -> TemporalAttention
```

保持 Stage 1 的总体数据流：

```text
RevIN -> 原有前置 feature transform -> 新 MASTER-style encoder
      -> 原有 projection MLP -> VQ
```

## Motivation

原 PRISM-VQ 先通过 GRU 将完整时间序列压缩为单个 hidden state，再做横截面
建模，可能过早丢失时序信息。本实验在横截面建模前保留完整时间维，依次执行
temporal attention、cross-sectional attention，并通过 TemporalAttention
聚合，以检验这种编码顺序能否学习到更有效的 VQ latent representation。

## Modification

- 从 `AlphaMaster/src/alphamaster/model.py` 直接复制
  `PositionalEncoding`、`TAttention`、`SAttention`、`TemporalAttention`；
  未改动四者计算逻辑。开发 smoke 对源文件与实验文件中的四个 class 做 AST
  比对，全部精确一致。
- 将原 `FeatureExtractor` 拆分为保留的前置
  `Linear(158,158) -> LayerNorm(158) -> LeakyReLU` 与新的 MASTER-style
  encoder；只移除其中 GRU summarization。
- 新 encoder 使用 `Linear(158,128)` Input Projection，随后为
  `PositionalEncoding -> TAttention -> SAttention -> TemporalAttention`。
  必要接口适配沿用 base 的 encoder 预算：model dimension 128，T/S attention
  均为 2 heads、dropout 0.1，单个 TAttention 与单个 SAttention block。
- 原 `CrossAssetTransformerEncoder.out_layer` 的
  `Linear(128,512) -> GELU -> Linear(512,128)` 原样保留为 encoder 后置
  projection MLP；最终输出仍为 `(N_t, vq_embed_dim=128)`。
- `configs/config.yaml` 默认设置 `vqvae.encoder.type: master` 并显式记录上述
  head/dropout 配置；核心改动不依赖 CLI override。
- `module/autoencoder.py` 与 `trainer/train_ypred.py` 只增加构造和加载同一新
  encoder 所需的配置透传，Stage 2 算法逻辑不变。
- 新增 `tests/test_master_stage1_encoder.py` 与
  `scripts/smoke_master_stage1_encoder.py`。

## Constraints

- 唯一实验变量：Stage 1 的 `GRU + CrossAssetTransformer` 被上述
  MASTER-style encoder 替换；明确不加入 MASTER Market Gate。
- RevIN、原有前置 feature transform、后置 projection MLP、VectorQuantiser、
  VQ512 / 128 维 codebook、commitment/contrastive/dead-code 机制均不变。
- ReconstructionDecoder、prior-factor FiLM、SequencePredictorGRU、prediction
  target、全部 Stage 1 loss 与 loss weight 均不变。
- 整个 Stage 2 的建模机制、MoE、routing、prior factors、loading generation、
  return predictor、loss 与训练协议均不变；仅透传构造新 Stage 1 encoder 所需
  参数，并通过原 strict loader 加载本实验自训练 checkpoint。
- 数据 schema 与划分（train 2009–2020、valid 2021–2022、test 2023–2025）、
  70 epoch 预算、early stopping、Stage 1 seed 42、Stage 2 seed 0、
  Top30/Drop5 回测协议及其他超参数均与 `main` 一致。
- Stage 1 来源为 `self`；新 encoder 改变了参数结构，不能复用原 PRISM-VQ
  Stage 1 checkpoint。Phase 1 未启动正式长时间训练或正式回测。

## Git

Base: main
Branch: exp/027-prism-master-stage1-encoder
Commit: a0962cbe3eac5e21f4d71e23cf18e2e8f298c7dd
Stage 1 provenance: self（queue `stage1_source: self`）

## Smoke Test

Status: PASS

Notes:

- `conda run -n prism-vq python -m unittest discover -s tests -v`：全量单元测试
  `91/91 PASS`，其中新增 encoder 测试 `9/9 PASS`；日志位于
  `artifacts/027/smoke/unit_tests.log`。
- `conda run -n prism-vq python scripts/smoke_master_stage1_encoder.py`：PASS。
  synthetic smoke 完成完整 Stage 1
  `reconstruction + VQ + pred_weight * prediction` loss 的 backward 与 optimizer
  step，encoder gradient L1 为 `25.763037054333836`，输出 latent shape 为
  `(8,128)`，并生成 self Stage 1 checkpoint。
- Stage 2 使用原 `GenerateReturn.load_pretrained_vqvae` strict loader 加载该
  self checkpoint：encoder、quantizer、RevIN 均为
  `missing=0, unexpected=0`；输出 VQ latent `(8,128)`，single codebook shape
  `(512,128)`。Stage 2 backward/optimizer step 通过，冻结 Stage 1 无梯度，
  Stage 2 checkpoint strict round-trip 输出逐位一致。
- smoke 同时确认四个 AlphaMaster class AST 精确一致、encoder 内无 GRU、
  无 Market Gate、默认数据划分与 seed 未变。
- 产物：`artifacts/027/smoke/`；报告
  `artifacts/027/smoke/smoke_report.json`，日志
  `artifacts/027/smoke/smoke.log`，synthetic checkpoints 位于
  `artifacts/027/smoke/checkpoints/`。

## Result

Status: DONE（test 区间 2023-01-01 – 2025-12-31）

IC: 0.0313
ICIR: 0.1567
RankIC: 0.0534
RankICIR: 0.2715

Annual Return: 11.01%（基准 6.40%，超额 4.61%）
Sharpe: 0.6710
Sortino: 1.0606
MDD: -22.26%
Calmar: 0.4946
Turnover: 0.3318

## Conclusion

Phase 2 固定执行器完成正式训练、预测与回测（pinned commit a0962cbe3eac5e21f4d71e23cf18e2e8f298c7dd）。Stage 1 best checkpoint：`infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=5-val_loss=0.5622.ckpt`（self-trained）；Stage 2 seed 0。

产物：`artifacts/027/run/`（checkpoints/、res/、stage1.log、stage2.log、backtest.log、summary.json）。
