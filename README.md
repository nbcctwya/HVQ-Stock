# Experiment 028 — prism-temporal-attention-stage1

## Base

`main`

Stage 1 provenance：`self`。本实验改变了 Stage 1 encoder 参数结构，必须自行
训练新的 Stage 1；原 PRISM-VQ checkpoint 不能 strict 加载到新 encoder。

## Idea / Motivation

仅替换 PRISM-VQ Stage 1 的 GRU temporal summarization：

```text
RevIN
  -> 原有 Linear + LayerNorm + LeakyReLU feature transform
  -> Input Projection + PositionalEncoding + TAttention + TemporalAttention
  -> 原有 CrossAssetTransformer
  -> VQ
```

原 GRU 将每只股票的完整历史序列压缩为单一 hidden state，可能过早损失时序
信息。本实验通过 TAttention 在完整时间维上建模，再由 TemporalAttention
自适应聚合历史信息，检验更充分的 temporal representation 能否改善后续
cross-asset modeling 与 VQ latent factor learning。

## 核心修改

- 将原 `FeatureExtractor` 拆成保持不变的前置
  `Linear(158,158) -> LayerNorm(158) -> LeakyReLU` 与新的 temporal encoder。
- temporal encoder 使用 `Linear(158,128)` Input Projection，随后依次执行
  `PositionalEncoding -> TAttention -> TemporalAttention`，把
  `(N_t,20,158)` 聚合为 `(N_t,128)`，与原 GRU 输出接口一致。
- `PositionalEncoding`、`TAttention`、`TemporalAttention` 直接复制自
  `AlphaMaster/src/alphamaster/model.py`，不改计算逻辑；tests/smoke 对三个
  class 做 AST 精确一致性验证。
- TAttention 使用现有 encoder 的 2 heads 和 dropout 0.1；不做超参数搜索。
- 原 `CrossAssetTransformerEncoder` 及其后置
  `Linear(128,512) -> GELU -> Linear(512,128)` projection MLP 原样保留；smoke
  对该 class 与 `main` 做 AST 精确一致性验证。
- `configs/config.yaml` 默认设置
  `vqvae.encoder.type: temporal-attention` 和 `temporal_dropout: 0.1`；默认配置
  即完整代表本实验，无需实验特有 CLI override。

## 与 base 的区别

唯一实验变量是把 Stage 1 的 GRU temporal summarization 替换为
`Input Projection + PositionalEncoding + TAttention + TemporalAttention`。
原 CrossAssetTransformer 保持不变；本实验不加入 `SAttention`，不加入
Market Gate。

以下均与 `main` 保持不变：RevIN、GRU 前已有的 feature transform、
CrossAssetTransformer 内部结构及其后置 projection MLP、single VQ512、128
维 codebook、commitment/contrastive/dead-code 机制、ReconstructionDecoder、
prior-factor FiLM、SequencePredictorGRU、prediction target、Stage 1 loss 与
所有 loss weight。整个 Stage 2 的 MoE、routing、prior factors、loading、
return predictor、loss 和训练协议不变，只透传构造同一 Stage 1 encoder 所需
配置并通过既有 strict loader 加载 self checkpoint。数据划分、训练预算、
Stage 1 seed 42、Stage 2 seed 0、回测协议及其他超参数不变。

## Tests / Smoke

Status: **PASS**

- 全量单元测试：`92/92 PASS`；本实验新增测试 `10/10 PASS`。
- 最小 synthetic smoke 完成完整 Stage 1 loss 的 backward/optimizer step，
  生成 self Stage 1 checkpoint；Stage 2 使用原 loader strict 加载 encoder、
  quantizer、RevIN，三者均为 `missing=0, unexpected=0`。
- smoke 验证 TAttention 输入/输出均为 `(8,20,128)`，TemporalAttention 聚合为
  `(8,128)`，原 CrossAssetTransformer 接收并输出 `(8,128)`；VQ codebook 为
  `(512,128)`。
- smoke 验证无 GRU、无 SAttention、无 Market Gate，Stage 2 冻结 Stage 1
  无梯度，并完成 Stage 2 backward/optimizer step 与 strict checkpoint
  round-trip。
- 产物：`artifacts/028/smoke/`；报告为
  `artifacts/028/smoke/smoke_report.json`，日志为
  `artifacts/028/smoke/smoke.log`。

本阶段未启动正式长时间训练或正式回测。
