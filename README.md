# Experiment 032 — prism-spatial-first-market-gated-stage1-encoder

## Base

`exp/029-prism-spatial-first-stage1-encoder`

Stage 1 provenance：`self`。Market Gate 增加了新的 Stage 1 参数，本实验必须
自行训练 Stage 1，不复用实验 029 的 checkpoint。

## Idea / Motivation

在实验 029 的 spatial-first MASTER Stage 1 encoder 上，仅加入实验 031 已
实现并验证的 AlphaMaster Market Gate，验证 market-conditioned feature
selection 的有效性是否依赖 temporal / cross-sectional modeling 的先后
顺序。

结合既有实验构成完整的 2×2 controlled experiment：

| attention 顺序 | 无 Gate | 有 Gate |
| --- | --- | --- |
| TAttention -> SAttention | 027 | 031 |
| SAttention -> TAttention | 029 | **本实验（032）** |

用于验证 attention order 主效应、Market Gate 主效应以及二者的
interaction effect。

完整 Stage 1 路径为：

```text
RevIN
  -> Market Gate
  -> Linear + LayerNorm + LeakyReLU feature transform
  -> Input Projection
  -> PositionalEncoding
  -> SAttention
  -> TAttention
  -> TemporalAttention
  -> Linear -> GELU -> Linear projection MLP
  -> VQ
```

## 核心修改

- `Gate` class 与实验 031 完全一致（AST 级比对），直接复制自
  `AlphaMaster/src/alphamaster/model.py`，核心计算为
  `Linear(63,158) -> softmax(output / beta) * 158`。
- 从 canonical batch 取得 `market_feature`，严格只读取当前窗口最后时点：
  `market_current = market_feature[:, -1, :]`，shape 为 `(N_t,63)`。
- `gate` shape 为 `(N_t,158)`，通过
  `feature_normalized * gate.unsqueeze(1)` 作用于同一股票完整 `T=20` 历史
  窗口，并严格位于原 RevIN 后、原 feature transform 前。
- beta 沿用实验 031 / AlphaMaster 的固定规则：CSI300 为 10、SP500 为 5；
  默认 `configs/config.yaml` 已完整固化该实验，无需实验特有 CLI override。
- Stage 1 training/validation 与 Stage 2 frozen-encoder 路径均透传 market
  feature；Stage 2 的 market 输入仅止于冻结的 Stage 1 encoder。
- attention 顺序保持 029 的 `SAttention -> TAttention -> TemporalAttention`，
  未带入 031/027 的 `TAttention -> SAttention`。

## 与 base 的区别

相对于实验 029，唯一实验变量是在 RevIN 与原 feature transform 之间增加与
031 完全一致的 Market Gate（含必要的 market input plumbing）。029 的
Input Projection、PositionalEncoding、单个 SAttention、单个 TAttention、
TemporalAttention、heads、dropout、维度以及 `SAttention -> TAttention`
顺序全部不变；原前置 feature transform 与后置 projection MLP 也不变。

single VQ512 / 128 维 codebook、VectorQuantiser、commitment/contrastive/
dead-code 机制、ReconstructionDecoder、prior-factor FiLM、
SequencePredictorGRU、prediction target、Stage 1 loss 与全部 weight 均保持
不变。Stage 2 的 LoadingGenerator、MoE、routing、LatentValueHead、
ReturnPredictor、loss 和训练逻辑保持不变，market feature 不直接进入任何
Stage 2 downstream 模块。数据 schema/划分、训练预算、early stopping、
Stage 1 seed 42、Stage 2 seed 0、回测协议和其他超参数均与 029 一致。

## Tests / Smoke

Status: **PASS**

- 全量单元测试：`101/101 PASS`；完整日志位于
  `artifacts/032/smoke/unit_tests.log`。
- 最小 synthetic smoke 完成完整 Stage 1 loss 的 backward/optimizer step，
  encoder gradient L1 为 `28.631042590888683`，并生成 self Stage 1
  checkpoint。
- smoke 确认 AlphaMaster `Gate` 与四个 attention class 的 AST 精确一致；
  Gate 输入/输出为 `(8,63) -> (8,158)`、每样本权重和约为 158、CSI300
  beta=10；实际路径为 `RevIN -> Market Gate -> Feature Transform -> Input
  Projection -> PositionalEncoding -> SAttention -> TAttention ->
  TemporalAttention -> Projection MLP`。仅修改历史 market 不影响
  gate/latent，修改末时点 market 会同时改变 gate 与 encoder latent，
  prior 改变不影响 Gate。
- Stage 2 对 self Stage 1 checkpoint strict load：encoder、quantizer、
  RevIN 均 `missing=0, unexpected=0`；冻结 Stage 1 无梯度，VQ latent 为
  `(8,128)`、codebook 为 `(512,128)`，Stage 2 backward 与 strict
  checkpoint round-trip 通过。
- 产物：`artifacts/032/smoke/`，其中报告为
  `artifacts/032/smoke/smoke_report.json`，日志为
  `artifacts/032/smoke/smoke.log`，单元测试日志为
  `artifacts/032/smoke/unit_tests.log`。

本阶段不启动正式长时间训练或正式回测。
