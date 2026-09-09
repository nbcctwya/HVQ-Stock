# Experiment 029 — prism-spatial-first-stage1-encoder

## Base

`exp/027-prism-master-stage1-encoder`

Stage 1 provenance：`self`。本实验改变了 Stage 1 encoder 的实际计算顺序，
因此自行训练新的 Stage 1；不复用 027 的 Stage 1 checkpoint。

## Idea / Motivation

在实验 027 的 MASTER-style Stage 1 encoder 上，仅交换 temporal attention
与 spatial attention 的执行顺序：

```text
Input Projection -> PositionalEncoding -> SAttention -> TAttention -> TemporalAttention
```

通过与 027 的 `TAttention -> SAttention` 对照，验证时序关系与横截面关系的
建模顺序是否影响最终离散潜在因子的表示质量。

## 核心修改

- `MASTERStyleEncoder.forward` 在 PositionalEncoding 后先调用原有
  `SAttention`，再调用原有 `TAttention`，最后仍由原有
  `TemporalAttention` 聚合。
- `PositionalEncoding`、`TAttention`、`SAttention`、
  `TemporalAttention` 的具体实现、构造参数与模块数量均保持 027 原样。
- 默认 `configs/config.yaml` 继续选择 `vqvae.encoder.type: master`；由于
  spatial-first 顺序是该分支 encoder 的默认实现，直接运行默认配置即为
  本实验，不依赖实验特有 CLI override。
- 测试与 smoke 明确捕获并断言
  `Input Projection -> PositionalEncoding -> SAttention -> TAttention -> TemporalAttention`
  的实际调用顺序。

## 与 base 的区别

唯一实验变量是将 027 的 `TAttention -> SAttention` 改为
`SAttention -> TAttention`。

以下均与 027 保持不变：Input Projection、PositionalEncoding、TAttention、
SAttention、TemporalAttention 本身的实现、维度与超参数；RevIN、前置
feature transform、后置 projection MLP、VectorQuantiser、decoder、predictor、
Stage 1 loss 及所有 loss weight；整个 Stage 2 算法逻辑；数据划分、训练预算、
seed 协议、回测协议及其他超参数。未加入 Market Gate 或其他模块，也未进行
超参数搜索。

## Tests / Smoke

Status: **PASS**

- 全量单元测试：`92/92 PASS`；本实验 encoder 测试 `10/10 PASS`。
- 最小 synthetic smoke 完成一次完整 Stage 1 loss 的 backward/optimizer
  step，encoder gradient L1 为 `26.096960986033082`，并生成 self Stage 1
  checkpoint。
- 实际调用顺序为
  `Input Projection -> PositionalEncoding -> SAttention -> TAttention -> TemporalAttention`；
  四个 AlphaMaster attention 相关 class 的 AST 均与源文件精确一致。
- Stage 2 strict 加载 self checkpoint 的 encoder、quantizer、RevIN，三者均为
  `missing=0, unexpected=0`；Stage 1 保持冻结，Stage 2 backward/optimizer
  step 与 strict checkpoint round-trip 通过。
- Stage 1 与 Stage 2 的 VQ latent 均为 `(8, 128)`，single codebook 为
  `(512, 128)`，Market Gate 与 encoder GRU 均不存在。
- 产物：`artifacts/029/smoke/`，其中报告为
  `artifacts/029/smoke/smoke_report.json`，日志为
  `artifacts/029/smoke/smoke.log`，单元测试日志为
  `artifacts/029/smoke/unit_tests.log`。

本阶段不启动正式长时间训练或正式回测。
