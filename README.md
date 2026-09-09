# Experiment 027 — prism-master-stage1-encoder

## Base

`main`

Stage 1 provenance：`self`。本实验必须自行训练新的 Stage 1；原 PRISM-VQ
Stage 1 checkpoint 与新 encoder 结构不兼容，不能复用。

## Idea / Motivation

仅替换 PRISM-VQ Stage 1 的时空编码器：将原来的
`GRU temporal summarization + CrossAssetTransformer` 替换为不含 Market
Gate 的 MASTER-style encoder：

```text
Input Projection -> PositionalEncoding -> TAttention -> SAttention -> TemporalAttention
```

原实现先用 GRU 将完整序列压缩成单个 hidden state，再进行横截面建模，可能
过早丢失时序信息。本实验保持完整时间维，先做 temporal attention，再做
cross-sectional attention，最后由 TemporalAttention 聚合，以检验这种表示
能否产生更有效的 VQ latent。

## 核心修改

- `PositionalEncoding`、`TAttention`、`SAttention`、`TemporalAttention`
  直接复制自 `AlphaMaster/src/alphamaster/model.py`，计算逻辑不变；smoke 中
  对四个 class 做 AST 比对，结果全部精确一致。
- 保留原 Stage 1 的前置 `Linear -> LayerNorm -> LeakyReLU` feature transform。
- 在前置变换后加入 `Linear(158, 128)` Input Projection，以现有
  `hidden_size=128` 作为 MASTER-style encoder 的 model dimension。
- 使用现有 encoder head/dropout 预算做必要接口适配：TAttention 与
  SAttention 均为 2 heads、dropout 0.1；保持单个 TAttention 和单个
  SAttention block。
- 保留原 CrossAssetTransformer 后的
  `Linear(128, 512) -> GELU -> Linear(512, 128)` projection MLP，使输出仍为
  `(N_t, vq_embed_dim=128)`，无缝接入原 VectorQuantiser。
- `configs/config.yaml` 默认设置 `vqvae.encoder.type: master`，直接运行默认
  配置即为本实验，无需实验特有 CLI override。

## 与 base 的区别

唯一实验变量是 Stage 1 中的
`GRU + CrossAssetTransformer -> MASTER-style encoder`。本实验明确不加入
MASTER Market Gate。

以下均与 `main` 保持不变：RevIN、前置 feature transform、后置 projection
MLP、single VQ512、128 维 codebook、commitment/contrastive/dead-code
机制、ReconstructionDecoder、prior-factor FiLM、SequencePredictorGRU、
prediction target、所有 Stage 1 loss 与 weight；整个 Stage 2 的 MoE、routing、
prior factors、loading generation、return predictor、loss 与训练协议也不变。
数据划分、训练预算、Stage 1 seed 42、Stage 2 seed 0 与回测协议不变。

## Tests / Smoke

Status: **PASS**

- 全量单元测试：`91/91 PASS`；本实验新增测试 `9/9 PASS`。
- 最小 synthetic smoke 完成一次完整 Stage 1 loss 的 backward/optimizer step，
  生成 self Stage 1 checkpoint；Stage 2 随后以 strict 模式加载 encoder、
  quantizer、RevIN，三者均为 `missing=0, unexpected=0`。
- smoke 验证 Stage 1 latent 为 `(8, 128)`、single VQ codebook 为
  `(512, 128)`、Market Gate/encoder GRU 均不存在、Stage 2 冻结 Stage 1 无
  梯度，并完成 Stage 2 backward/optimizer step 与 strict checkpoint
  round-trip。
- 产物：`artifacts/027/smoke/`，其中报告为
  `artifacts/027/smoke/smoke_report.json`，日志为
  `artifacts/027/smoke/smoke.log`。

本阶段未启动正式长时间训练或正式回测。
