# 029 — prism-spatial-first-stage1-encoder

## Idea

基于实验 027 的 MASTER-style Stage 1 encoder，仅交换 temporal attention 与
spatial attention 的执行顺序：

```text
Input Projection -> PositionalEncoding -> SAttention -> TAttention -> TemporalAttention
```

## Motivation

时序关系与横截面关系的建模顺序可能影响最终离散潜在因子的表示质量。将本
实验的 `SAttention -> TAttention` 与 027 的
`TAttention -> SAttention` 对照，用于验证 spatio-temporal modeling order
对 VQ latent representation 的影响。

## Modification

- 仅在 `MASTERStyleEncoder.forward` 中交换原有 `TAttention` 与
  `SAttention` 的调用顺序；Input Projection 与 PositionalEncoding 后先执行
  SAttention，再执行 TAttention，最后仍由 TemporalAttention 聚合。
- `PositionalEncoding`、`TAttention`、`SAttention`、
  `TemporalAttention` class 的 AST 与 027 完全一致，构造参数、维度与模块
  数量均未改变。
- `configs/config.yaml` 与 027 字节一致，默认仍为
  `vqvae.encoder.type: master`、单个 TAttention/SAttention block、二者均为
  2 heads 与 dropout 0.1；该分支的默认 encoder 实现即为 spatial-first，核心
  改动不依赖 CLI override。
- 将继承自 027 的顺序测试改为断言 spatial-first，并增加相同模块与参数下
  spatial-first 实际输出路径及其区别于 temporal-first 路径的回归测试；smoke
  同时捕获实际模块调用顺序。

## Constraints

- 唯一实验变量：027 中的 `TAttention -> SAttention` 改为
  `SAttention -> TAttention`。
- Input Projection、PositionalEncoding、TAttention、SAttention、
  TemporalAttention 本身的实现、维度与超参数全部与 027 一致。
- RevIN、前置 feature transform、后置 projection MLP、VectorQuantiser、
  decoder、predictor、Stage 1 loss 及所有 loss weight 全部与 027 一致。
- 整个 Stage 2 与 027 一致；未修改 Stage 2 算法或构造逻辑，仅验证它可 strict
  加载本实验 self Stage 1 checkpoint。
- 不加入 Market Gate 或其他模块，不进行超参数搜索，不重构无关代码。
- 数据划分、70 epoch 训练预算、early stopping、Stage 1 seed 42、Stage 2
  seed 0、Top30/Drop5 回测协议及其他超参数不变。
- Stage 1 来源为 `self`。本实验需要在新执行顺序下自行训练 Stage 1，不复用
  027 已训练的 Stage 1 权重。Phase 1 未启动正式长时间训练或正式回测。

## Git

Base: exp/027-prism-master-stage1-encoder
Branch: exp/029-prism-spatial-first-stage1-encoder
Commit: 8b61fc866b36d1776153c02dc418336c824c07c2
Stage 1 provenance: self（queue `stage1_source: self`）

## Smoke Test

Status: PASS

Notes:

- `conda run -n prism-vq python -m unittest discover -s tests -v`：全量单元测试
  `92/92 PASS`，其中本实验 encoder 测试 `10/10 PASS`；日志位于
  `artifacts/029/smoke/unit_tests.log`。
- `conda run -n prism-vq python scripts/smoke_master_stage1_encoder.py`：PASS。
  synthetic smoke 完成完整 Stage 1
  `reconstruction + VQ + pred_weight * prediction` loss 的 backward 与 optimizer
  step，encoder gradient L1 为 `26.096960986033082`，输出 latent shape 为
  `(8,128)`，并生成 self Stage 1 checkpoint。
- smoke 捕获的实际调用顺序为
  `Input Projection -> PositionalEncoding -> SAttention -> TAttention -> TemporalAttention`；
  四个 AlphaMaster class AST 精确一致，Market Gate 与 encoder GRU 均不存在。
- Stage 2 使用原 `GenerateReturn.load_pretrained_vqvae` strict loader 加载该
  self checkpoint：encoder、quantizer、RevIN 均为
  `missing=0, unexpected=0`；输出 VQ latent `(8,128)`，single codebook shape
  `(512,128)`。Stage 2 backward/optimizer step 通过，冻结 Stage 1 无梯度，
  Stage 2 checkpoint strict round-trip 输出逐位一致。
- 产物位于 `artifacts/029/smoke/`：`unit_tests.log`、`smoke.log`、
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
