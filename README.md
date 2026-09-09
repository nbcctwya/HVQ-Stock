# Experiment 031 — prism-market-gated-stage1-encoder

## Base

`exp/027-prism-master-stage1-encoder`

Stage 1 provenance：`self`。Market Gate 增加了新的 Stage 1 参数，本实验必须
自行训练 Stage 1，不复用实验 027 的 checkpoint。

## Idea / Motivation

在实验 027 的 MASTER-style Stage 1 encoder 上，仅加入 AlphaMaster Market
Gate，验证 market-conditioned feature selection 是否能改善 VQ latent factor
learning。股票特征的重要性可能随当前市场状态变化；在进入时空 encoder 与 VQ
之前，以当前 regime 对 158 个个股特征进行条件化重加权，可能使 encoder 更
容易提取有效信息。

完整 Stage 1 路径为：

```text
RevIN
  -> Market Gate
  -> Linear + LayerNorm + LeakyReLU feature transform
  -> Input Projection
  -> PositionalEncoding
  -> TAttention
  -> SAttention
  -> TemporalAttention
  -> Linear -> GELU -> Linear projection MLP
  -> VQ
```

## 核心修改

- `Gate` class 直接复制自 `AlphaMaster/src/alphamaster/model.py`，核心计算为
  `Linear(63,158) -> softmax(output / beta) * 158`；tests 与 smoke 使用 AST
  比对确认计算逻辑一致。
- 从 canonical batch 取得 `market_feature`，严格只读取当前窗口最后时点：
  `market_current = market_feature[:, -1, :]`，shape 为 `(N_t,63)`。
- `gate` shape 为 `(N_t,158)`，通过
  `feature_normalized * gate.unsqueeze(1)` 作用于同一股票完整 `T=20` 历史
  窗口，并严格位于原 RevIN 后、原 feature transform 前。
- beta 沿用 AlphaMaster / 实验 007 的固定规则：CSI300 为 10、SP500 为 5；
  默认 `configs/config.yaml` 已完整固化该实验，无需实验特有 CLI override。
- Stage 1 training/validation 与 Stage 2 frozen-encoder 路径均透传 market
  feature；Stage 2 的 market 输入仅止于冻结的 Stage 1 encoder。

## 与 base 的区别

相对于实验 027，唯一实验变量是在 RevIN 与原 feature transform 之间增加上述
Market Gate。027 的 Input Projection、PositionalEncoding、单个 TAttention、
单个 SAttention、TemporalAttention、heads、dropout、维度以及
`TAttention -> SAttention` 顺序全部不变；原前置 feature transform 与后置
projection MLP 也不变。

single VQ512 / 128 维 codebook、VectorQuantiser、commitment/contrastive/
dead-code 机制、ReconstructionDecoder、prior-factor FiLM、
SequencePredictorGRU、prediction target、Stage 1 loss 与全部 weight 均保持
不变。Stage 2 的 LoadingGenerator、MoE、routing、LatentValueHead、
ReturnPredictor、loss 和训练逻辑保持不变，market feature 不直接进入任何
Stage 2 downstream 模块。数据 schema/划分、训练预算、early stopping、
Stage 1 seed 42、Stage 2 seed 0、回测协议和其他超参数均与 027 一致。

## Tests / Smoke

Status: **PASS**

- 全量单元测试：`99/99 PASS`；完整日志位于
  `artifacts/031/smoke/unit_tests.log`。
- 最小 synthetic smoke 完成完整 Stage 1 loss 的 backward/optimizer step，
  encoder gradient L1 为 `25.304254525253782`，并生成 self Stage 1 checkpoint。
- smoke 确认 AlphaMaster `Gate` 与四个 attention class 的 AST 精确一致；Gate
  输入/输出为 `(8,63) -> (8,158)`、每样本权重和约为 158、CSI300 beta=10，
  实际路径为 `RevIN -> Market Gate -> Feature Transform -> Input Projection ->
  PositionalEncoding -> TAttention -> SAttention -> TemporalAttention -> Projection
  MLP`。仅修改历史 market 不影响 gate/latent，修改末时点 market 会同时改变
  gate 与 encoder latent，prior 改变不影响 Gate。
- Stage 2 对 self Stage 1 checkpoint strict load：encoder、quantizer、RevIN 均
  `missing=0, unexpected=0`；冻结 Stage 1 无梯度，VQ latent 为 `(8,128)`、
  codebook 为 `(512,128)`，Stage 2 backward 与 strict checkpoint round-trip
  均通过。
- 产物目录：`artifacts/031/smoke/`；报告为 `smoke_report.json`，日志为
  `smoke.log`，synthetic checkpoints 位于 `checkpoints/`。

本阶段不启动正式长时间训练或正式回测。
