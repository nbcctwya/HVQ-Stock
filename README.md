# Experiment 030 — prism-mean-pooling-stage1-encoder

## Base

`exp/027-prism-master-stage1-encoder`

Stage 1 provenance：`self`。本实验改变了 Stage 1 encoder 的时间聚合结构，
必须自行训练新的 Stage 1，不能复用实验 027 的 checkpoint。

## Idea / Motivation

仅将实验 027 MASTER-style Stage 1 encoder 最后的 learned
`TemporalAttention` 替换为不含参数的时间均值聚合：

```text
Input Projection -> PositionalEncoding -> TAttention -> SAttention -> MeanPooling
```

其中 MeanPooling 严格为 `z = h.mean(dim=1)`。本实验以 uniform temporal
aggregation 对照 027 的 learned weighted temporal pooling，用于验证 learned
temporal aggregation 本身是否改善 VQ latent representation。

## 核心修改

- `MASTERStyleEncoder` 不再构造或调用 `TemporalAttention`。
- `SAttention` 输出保持 `(N_t, T, hidden_dim)`，随后直接执行
  `mean(dim=1)`，得到 `(N_t, hidden_dim)`。
- 聚合结果继续接实验 027 原有的
  `Linear -> GELU -> Linear` projection MLP 与 VectorQuantiser。
- MeanPooling 前后未增加 projection、gate、attention 或任何其他可学习模块。

## 与 base 的区别

唯一实验变量是将 027 最后的 learned `TemporalAttention` 替换为严格算术
平均 `mean(dim=1)`。Input Projection、PositionalEncoding、TAttention、
SAttention 的实现、维度、模块数、超参数及 `TAttention -> SAttention` 顺序
均与 027 一致。

RevIN、前置 `Linear + LayerNorm + LeakyReLU` feature transform、后置
projection MLP、single VQ512 / 128 维 codebook、commitment / contrastive /
dead-code 机制、ReconstructionDecoder、prior-factor FiLM、
SequencePredictorGRU、prediction target、Stage 1 loss 与所有 weight 均不变。
Stage 2 仅按本分支结构构造并 strict 加载 self Stage 1 checkpoint，算法逻辑
不变。数据划分、训练预算、early stopping、Stage 1 seed 42、Stage 2 seed 0、
回测协议及其余超参数均与 027 一致。

## Tests / Smoke

Status: **PASS**

- 全量单元测试：`92/92 PASS`；本实验 encoder 测试 `10/10 PASS`。
- 最小 synthetic smoke 完成一次完整 Stage 1 loss 的 backward/optimizer step，
  生成 self Stage 1 checkpoint；实际捕获顺序为
  `Input Projection -> PositionalEncoding -> TAttention -> SAttention -> Projection MLP`。
- smoke 确认 TemporalAttention/Market Gate/encoder GRU 均未参与模型，
  MeanPooling 前后 shape 分别为 `(8,20,128)` 与 `(8,128)`，并严格等于
  `mean(dim=1)`。
- Stage 2 strict 加载 encoder、quantizer、RevIN 均为
  `missing=0, unexpected=0`，VQ latent `(8,128)`、codebook `(512,128)`，
  Stage 2 backward 与 strict checkpoint round-trip 均通过。
- 产物：`artifacts/030/smoke/`，含 `unit_tests.log`、`smoke.log`、
  `smoke_report.json` 与 synthetic checkpoints。

本阶段不启动正式长时间训练或正式回测。
