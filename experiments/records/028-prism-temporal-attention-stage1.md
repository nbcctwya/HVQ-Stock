# 028 — prism-temporal-attention-stage1

## Idea

仅修改 PRISM-VQ Stage 1 的 temporal encoder。保留原有
`CrossAssetTransformer` 不变，只将其前面的 GRU temporal summarization
替换为：

```text
Input Projection -> PositionalEncoding -> TAttention -> TemporalAttention
```

保持 Stage 1 数据流为：

```text
RevIN
  -> 原有 Linear + LayerNorm + LeakyReLU feature transform
  -> Input Projection + PositionalEncoding + TAttention + TemporalAttention
  -> 原有 CrossAssetTransformer
  -> VQ
```

## Motivation

原 PRISM-VQ 使用 GRU 将历史序列压缩为单一 hidden state，可能过早损失时序
信息。使用保留完整时间维度的 TAttention 建模，并通过 TemporalAttention
自适应聚合历史信息，可能得到更有效的 temporal representation，从而改善
后续 cross-asset modeling 与 VQ latent factor learning。

## Modification

- 将原 `FeatureExtractor` 拆分为保持不变的前置
  `Linear(158,158) -> LayerNorm(158) -> LeakyReLU` feature transform 与新的
  temporal encoder；仅移除 GRU summarization。
- 新 temporal encoder 使用 `Linear(158,128)` Input Projection，依次连接
  `PositionalEncoding -> TAttention -> TemporalAttention`，将
  `(N_t,T,158)` 聚合为与原 GRU 接口一致的 `(N_t,128)`。
- `PositionalEncoding`、`TAttention`、`TemporalAttention` 直接复制自
  `AlphaMaster/src/alphamaster/model.py`，未改计算逻辑；tests/smoke 对三个
  class 做 AST 精确比对，全部一致。
- TAttention 使用现有 encoder 的 `num_heads: 2`，dropout 为 0.1；未进行
  超参数搜索。
- 原 `CrossAssetTransformerEncoder` 内部结构、调用位置及后置
  `Linear(128,512) -> GELU -> Linear(512,128)` projection MLP 原样保留；
  smoke 与 `main` 中该 class 的 AST 精确一致。
- `configs/config.yaml` 默认设置
  `vqvae.encoder.type: temporal-attention` 与 `temporal_dropout: 0.1`，核心改动
  不依赖实验特有 CLI override。
- `module/autoencoder.py` 与 `trainer/train_ypred.py` 只增加构造同一新 encoder
  所需的配置透传；Stage 2 算法逻辑不变。
- 新增 `tests/test_temporal_attention_stage1.py` 与
  `scripts/smoke_temporal_attention_stage1.py`；分支根 README 改写为本实验
  说明。

## Constraints

- 唯一实验变量：Stage 1 中的 GRU temporal summarization 被
  `Input Projection + PositionalEncoding + TAttention + TemporalAttention`
  替换；原 CrossAssetTransformer 保持不变。
- 不加入 SAttention，不加入 Market Gate。
- RevIN、原有前置 `Linear + LayerNorm + LeakyReLU` feature transform、
  CrossAssetTransformer 内部结构及其后置 projection MLP 均保持不变。
- VectorQuantiser、single VQ512、128 维 codebook、assignment、commitment、
  contrastive 与 dead-code 机制保持不变。
- ReconstructionDecoder、prior-factor FiLM、SequencePredictorGRU、prediction
  target、Stage 1 loss 及所有 loss weight 保持不变。
- 整个 Stage 2 的模型结构、MoE、routing、prior factors、loading、return
  predictor、loss 与训练协议保持不变；只透传构造同一 Stage 1 encoder 所需
  配置，并由原 strict loader 加载本实验 self checkpoint。
- 数据划分（train 2009–2020、valid 2021–2022、test 2023–2025）、70 epoch
  预算、early stopping、optimizer、learning rate、Stage 1 seed 42、Stage 2
  seed 0、Top30/Drop5 回测协议及其他超参数均与 `main` 一致。
- Stage 1 来源为 `self`；新 encoder 参数结构不同，不能复用原 PRISM-VQ
  Stage 1 checkpoint。Phase 1 未启动正式长时间训练或正式回测。

## Git

Base: main
Branch: exp/028-prism-temporal-attention-stage1
Commit: 4494d99542f40be7d3136ab42836f306631f0584
Stage 1 provenance: self（queue `stage1_source: self`）

## Smoke Test

Status: PASS

Notes:

- `conda run -n prism-vq python -m unittest discover -s tests -v`：全量单元测试
  `92/92 PASS`；本实验新增测试 `10/10 PASS`。日志位于
  `artifacts/028/smoke/unit_tests.log`。
- `conda run -n prism-vq python scripts/smoke_temporal_attention_stage1.py`：
  PASS。synthetic smoke 完成完整 Stage 1
  `reconstruction + VQ + pred_weight * prediction` loss 的 backward 与 optimizer
  step，encoder gradient L1 为 `39.05178627371788`，输出 latent shape 为
  `(8,128)`，并生成 self Stage 1 checkpoint。
- 数据流 shape 验证：TAttention 输入/输出均为 `(8,20,128)`；
  TemporalAttention 聚合为 `(8,128)`；原 CrossAssetTransformer 接收并输出
  `(8,128)`。三段 AlphaMaster class AST 精确一致，CrossAssetTransformer 与
  `main` AST 精确一致；无 GRU、无 SAttention、无 Market Gate。
- Stage 2 使用原 `GenerateReturn.load_pretrained_vqvae` strict loader 加载该
  self checkpoint：encoder、quantizer、RevIN 均为
  `missing=0, unexpected=0`；输出 VQ latent `(8,128)`，single codebook shape
  `(512,128)`。Stage 2 backward/optimizer step 通过，冻结 Stage 1 无梯度，
  Stage 2 checkpoint strict round-trip 输出逐位一致。
- 产物位于 `artifacts/028/smoke/`：`unit_tests.log`、`smoke.log`、
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
