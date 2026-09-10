# 032 — prism-spatial-first-market-gated-stage1-encoder

## Idea

基于实验 029 的 spatial-first MASTER Stage 1 encoder，仅在 RevIN 与原
feature transform 之间加入与实验 031 完全一致的 AlphaMaster Market Gate：

```text
RevIN -> Market Gate -> Linear + LayerNorm + LeakyReLU feature transform
      -> Input Projection -> PositionalEncoding
      -> SAttention -> TAttention -> TemporalAttention
      -> Linear -> GELU -> Linear projection MLP -> VQ
```

Gate 使用 canonical `market_feature[:, -1, :]` 的当前市场状态，并以同一组
158 维 feature-wise 权重重加权该股票完整 T=20 历史窗口的 RevIN normalized
stock features。

## Motivation

Market-conditioned feature selection 的有效性可能依赖 temporal /
cross-sectional modeling 的先后顺序。实验 031 已测试
`TAttention -> SAttention + Market Gate`，本实验测试
`SAttention -> TAttention + Market Gate`。结合：

- 027：T -> S，无 Gate
- 031：T -> S，有 Gate
- 029：S -> T，无 Gate
- 032：S -> T，有 Gate（本实验）

构成完整的 2×2 controlled experiment，用于验证 attention order 主效应、
Market Gate 主效应以及 attention order × Market Gate interaction effect。

## Modification

- `Gate` class 与实验 031 完全一致（单元测试做 AST 级比对），直接复制自
  `AlphaMaster/src/alphamaster/model.py`；计算逻辑保持为
  `Linear(63,158) -> softmax(output / beta) * 158`。
- `SpatialEncoder` 接收 canonical market window，严格执行
  `market_current = market_feature[:, -1, :]`，再计算 `(N_t,158)` gate 并执行
  `feature_normalized * gate.unsqueeze(1)`；之后才进入 029 原有 feature
  transform。
- `configs/config.yaml` 默认显式固定 market input dim 63，以及实验 031 的
  beta 规则：CSI300=10、SP500=5。核心实验改动不依赖 CLI override。
- Stage 1 training/validation 从 `unpack_batch` 取得 market feature 并传入
  VQVAE/encoder，loss 组成不变。
- Stage 2 training/validation/inference 同样将 market feature 传入冻结的
  Stage 1 encoder；其余 Stage 2 主体仍只消费原有 feature、z_q 与 prior 等
  既有输入。
- attention 顺序保持 029 的 `SAttention -> TAttention -> TemporalAttention`；
  与 031 的 diff 确认除该顺序、docstring 与报错信息外 encoder 完全一致。
- 扩展 `tests/test_master_stage1_encoder.py`（新增 S->T 顺序 + Gate 行为
  测试）、`tests/test_stage2_freeze.py`、canonical inference schema 测试以及
  `scripts/smoke_master_stage1_encoder.py`。

## Constraints

- 唯一实验变量：在 029 的 RevIN 与原
  `Linear + LayerNorm + LeakyReLU` feature transform 之间加入与 031 一致的
  Market Gate；不加入额外 MLP、normalization、residual、attention、market
  encoder 或其他 market conditioning。
- Gate 只读取窗口最后一个市场时点，不读取未来市场信息；market feature 不
  进入 VQ、decoder、predictor 或任何 Stage 2 downstream 模块。
- Input Projection、PositionalEncoding、SAttention、TAttention、
  TemporalAttention 的实现、维度、heads、dropout 和模块数量全部与 029
  一致，顺序仍为 `SAttention -> TAttention -> TemporalAttention`，不得换回
  T -> S。
- 原 feature transform、后置 projection MLP、VectorQuantiser、single VQ512 /
  128 维 codebook、commitment/contrastive/dead-code 机制均不变。
- ReconstructionDecoder、prior-factor FiLM、SequencePredictorGRU、prediction
  target、Stage 1 loss 与全部 loss weight 均不变。
- Stage 2 的 LoadingGenerator、MoE、routing、LatentValueHead、
  ReturnPredictor、loss 与训练算法均与 029 一致；仅适配 frozen Stage 1
  encoder 的 market input 和新 checkpoint 结构。不加入 025/010/019/020 的
  任何 Stage 2 改动，不使用 MeanPooling。
- 数据 schema、数据划分、70 epoch 预算、early stopping、Stage 1 seed 42、
  Stage 2 seed 0、Top30/Drop5 回测协议和其他超参数均与 029 一致；不进行
  beta 搜索或其他超参数搜索。
- Stage 1 来源为 `self`。Gate 改变了 Stage 1 参数结构，不能复用 029 Stage 1
  checkpoint。Phase 1 未启动正式长时间训练或正式回测。

## Git

Base: exp/029-prism-spatial-first-stage1-encoder
Branch: exp/032-prism-spatial-first-market-gated-stage1-encoder
Commit: 0e324ec93c272415a92b61d283d586cfe9e06764
Stage 1 provenance: self（queue `stage1_source: self`）

## Smoke Test

Status: PASS

Notes:

- `python -m unittest discover -s tests -v`（conda env prism-vq）：全量单元测试
  `101/101 PASS`；日志位于 `artifacts/032/smoke/unit_tests.log`。
- `python scripts/smoke_master_stage1_encoder.py`：PASS。synthetic smoke 完成
  完整 Stage 1 `reconstruction + VQ + pred_weight * prediction` loss 的
  backward 与 optimizer step，encoder gradient L1 为 `28.631042590888683`，
  输出 latent `(8,128)`，并生成 self Stage 1 checkpoint。
- AlphaMaster `Gate` 与四个 attention class AST 精确一致；`Gate` class 与实验
  031 分支 AST 精确一致。Gate 实际输入/输出 shape 为 `(8,63) -> (8,158)`，
  每个样本权重和约为 158，CSI300 beta=10；config 与构造测试同时确认 SP500
  beta=5。
- 捕获的实际调用顺序为
  `RevIN -> Market Gate -> Feature Transform -> Input Projection ->
  PositionalEncoding -> SAttention -> TAttention -> TemporalAttention ->
  Projection MLP`，明确保持 029 的 spatial-first 顺序。仅改变末时点 market
  会改变 gate 和 encoder latent；仅改变较早 market 时点不会产生变化；改变
  prior 不改变 Market Gate。
- Stage 2 使用 `GenerateReturn.load_pretrained_vqvae` strict load 本实验 self
  checkpoint：encoder、quantizer、RevIN 均为 `missing=0, unexpected=0`；VQ
  latent `(8,128)`，single codebook `(512,128)`。Stage 2 backward/optimizer
  step 通过，冻结 Stage 1 无梯度；Stage 2 checkpoint strict round-trip 输出
  一致，且 market 在 Stage 2 forward 中只被 frozen encoder 消费。
- 产物位于 `artifacts/032/smoke/`：`unit_tests.log`、`smoke.log`、
  `smoke_report.json` 与 `checkpoints/`。

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
