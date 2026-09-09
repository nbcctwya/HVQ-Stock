# 030 — prism-mean-pooling-stage1-encoder

## Idea

基于实验 027 的 MASTER-style Stage 1 encoder，仅将最后的 learned
`TemporalAttention` 替换为 parameter-free MeanPooling：

```text
Input Projection -> PositionalEncoding -> TAttention -> SAttention -> MeanPooling
```

MeanPooling 严格为 `z = h.mean(dim=1)`，将
`(N_t, T, hidden_dim)` 聚合为 `(N_t, hidden_dim)`，随后继续接 027 原有的
后置 projection MLP 与 VQ。

## Motivation

实验 027 使用 learned TemporalAttention 对时序与横截面注意力建模后的历史
表示进行自适应加权聚合。本实验以 uniform temporal aggregation 为对照，验证
learned weighted temporal pooling 本身是否能够改善 VQ latent
representation，以及它对离散潜在因子学习质量的影响。

## Modification

- `MASTERStyleEncoder` 不再构造或调用 `TemporalAttention`；保留的
  `TemporalAttention` class 定义未参与模型树或 forward。
- `TAttention -> SAttention` 后直接执行唯一聚合表达式
  `x.mean(dim=1)`，未在 MeanPooling 前后增加 projection、gate、attention
  或其他模块。
- Input Projection、PositionalEncoding、TAttention、SAttention、前置 feature
  transform 与后置 `Linear -> GELU -> Linear` projection MLP 均保持 027
  实现和配置不变。
- `configs/config.yaml` 与 027 字节一致；本实验分支的默认实现直接执行 mean
  pooling，核心改动不依赖实验特有 CLI override。
- 在 027 的 encoder 单元测试与 synthetic smoke 上增加实际调用顺序、
  TemporalAttention 缺席、mean 严格等价性及 pooling 前后 shape 检查。

## Constraints

- 唯一实验变量：027 最后的 learned `TemporalAttention` 替换为时间维简单
  算术平均 `mean(dim=1)`。
- Input Projection、PositionalEncoding、单个 TAttention 与单个 SAttention
  的实现、维度、heads/dropout 及 `TAttention -> SAttention` 顺序均与 027
  一致。
- RevIN、前置 `Linear + LayerNorm + LeakyReLU` feature transform、后置
  projection MLP、VectorQuantiser、single VQ512 / 128 维 codebook、
  commitment/contrastive/dead-code 机制均不变。
- ReconstructionDecoder、prior-factor FiLM、SequencePredictorGRU、prediction
  target、Stage 1 loss 与所有 loss weight 均不变。
- Stage 2 算法逻辑与 027 完全一致，只按本实验结构构造并 strict 加载 self
  Stage 1 checkpoint。
- 不加入 Market Gate、Last Token pooling、其他 pooling 方法或额外聚合模块，
  不进行超参数搜索或无关重构。
- 数据划分、70 epoch 预算、early stopping、Stage 1 seed 42、Stage 2 seed 0、
  Top30/Drop5 回测协议及其他超参数均与 027 一致。
- Stage 1 来源为 `self`；聚合结构和参数集合已改变，不能复用 027 的 Stage 1
  checkpoint。Phase 1 未启动正式长时间训练或正式回测。

## Git

Base: exp/027-prism-master-stage1-encoder
Branch: exp/030-prism-mean-pooling-stage1-encoder
Commit: 63104603553afc51304a5695fbf60dd3931b43ec
Stage 1 provenance: self（queue `stage1_source: self`）

## Smoke Test

Status: PASS

Notes:

- `conda run -n prism-vq python -m unittest discover -s tests -v`：全量单元测试
  `92/92 PASS`，其中本实验 encoder 测试 `10/10 PASS`；日志位于
  `artifacts/030/smoke/unit_tests.log`。
- `conda run -n prism-vq python scripts/smoke_master_stage1_encoder.py`：PASS。
  synthetic smoke 完成完整 Stage 1
  `reconstruction + VQ + pred_weight * prediction` loss 的 backward 与 optimizer
  step，encoder gradient L1 为 `15.843749298714101`，输出 latent shape 为
  `(8,128)`，并生成 self Stage 1 checkpoint。
- smoke 捕获实际路径为
  `Input Projection -> PositionalEncoding -> TAttention -> SAttention -> Projection MLP`；
  SAttention 输出 `(8,20,128)`，projection MLP 输入 `(8,128)`，并逐元素确认
  后者严格等于前者 `mean(dim=1)`。模型树中不存在 TemporalAttention、Market
  Gate 或 encoder GRU。
- Stage 2 使用原 `GenerateReturn.load_pretrained_vqvae` strict loader 加载该
  self checkpoint：encoder、quantizer、RevIN 均为
  `missing=0, unexpected=0`；VQ latent `(8,128)`，single codebook
  `(512,128)`。Stage 2 backward/optimizer step 通过，冻结 Stage 1 无梯度，
  Stage 2 checkpoint strict round-trip 输出逐位一致。
- 产物位于 `artifacts/030/smoke/`：`unit_tests.log`、`smoke.log`、
  `smoke_report.json` 与 `checkpoints/`。

## Result

Status: DONE（test 区间 2023-01-01 – 2025-12-31）

IC: 0.0320
ICIR: 0.1992
RankIC: 0.0450
RankICIR: 0.2793

Annual Return: 13.97%（基准 6.40%，超额 7.57%）
Sharpe: 0.8163
Sortino: 1.2765
MDD: -21.94%
Calmar: 0.6367
Turnover: 0.3287

## Conclusion

Phase 2 固定执行器完成正式训练、预测与回测（pinned commit 63104603553afc51304a5695fbf60dd3931b43ec）。Stage 1 best checkpoint：`infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=8-val_loss=0.5845.ckpt`（self-trained）；Stage 2 seed 0。

产物：`artifacts/030/run/`（checkpoints/、res/、stage1.log、stage2.log、backtest.log、summary.json）。
