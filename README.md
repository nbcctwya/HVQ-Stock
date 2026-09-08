# 023 — continuous-residual-correction

## Base

`main`（原始 corrected PRISM-VQ baseline）。

Stage 1 不重新训练，复用 corrected PRISM-VQ exact checkpoint：

`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`

该文件为 14,584,929 bytes，MD5 为
`6b9d9dbfd938c7bd2c7dc5ee33cb38af`。SpatialEncoder、Quantizer 与 RevIN
均已 strict 加载验证（missing=0、unexpected=0）；模型保持 single VQ512、
128 维 embedding 与原数据划分。

## Idea / Motivation

hard vector quantization 提供了有利的离散化与正则化，但同时把连续表示 `h`
压缩为最近的 prototype `z_q`，丢弃了样本相对于 prototype 的细粒度连续偏差
信息。

本实验在 Stage 2 保留离散 VQ prototype `z_q` 作为主体表示的同时，重新利用
被 hard quantization 丢弃的 instance-specific continuous residual：

```text
r = h - z_q
delta_z = f(r)
z_stage2 = z_q + delta_z
```

然后让 Stage 2 统一使用 `z_stage2`，检验一个受控的小型 residual adapter
能否在保留 discrete inductive bias 的前提下恢复部分 instance-specific
信息，从而改善收益排序预测。

与 019 quantization-confidence-adapter 不同：本实验直接利用完整的 128 维
residual vector `h - z_q`，而不是仅利用 residual magnitude /
quantization-error scalar。

## 核心修改

- `f` 严格为单层 `Linear(128, 128)`，weight 与 bias 显式全零初始化；
  不加入额外 MLP、gate、attention、normalization 或 residual-loss。
- 初始化时 `f(r)=0`，因此 `z_stage2 == z_q`；完整 prediction forward
  （含原 auxiliary loss）与 `main` 逐位相等。
- `r = h.detach() - z_q.detach()`；该路径不向 Encoder、Quantizer、RevIN
  或 codebook 反向传播。
- 原来使用 `z_q` 的全部 Stage 2 consumer——`LoadingGenerator`（包括 temporal
  structure token 与 MoE/HyperFusion）和 `LatentValueHead`——统一改用
  `z_stage2`。
- adapter 在所有 baseline 模块构造完成后才追加；相同 seed 下，除新增 adapter
  外的全部既有 state tensor 初始化逐位不变。
- 默认 `configs/config.yaml` 设置
  `predictor.residual_correction_adapter: true`，无需实验特有 CLI override。

## 与 base 的区别

唯一实验变量是新增上述 continuous residual correction adapter。

Stage 1 encoder、quantizer、codebook、assignment、loss 与训练逻辑均不变；
Stage 2 的 DLinear、Temporal Transformer、MoE、HyperFusion、prior/latent heads、
prediction/loss 定义也不变。不加入 quantization-confidence scalar、
Shared Expert、adaptive fusion、decoupling、market-conditioned routing、
prior-latent allocation、code-aware routing 或其他机制。

canonical `158 stock + 13 prior + 63 market + 10 returns = 244` schema、数据划分
（train 2009–2020、valid 2021–2022、test 2023–2025）、70 epoch 预算、early
stopping、Stage 1 seed 42、Stage 2 seed 0、其他超参数及 Top30/Drop5 回测协议
均保持 `main` 不变。

## Smoke 状态

Status: **PASS**。

- `conda run -n prism-vq python -m unittest discover -s tests -v`：94/94 PASS；
  新增机制测试 12/12 PASS，既有 Stage 1 freeze 回归测试继续 PASS。
- `scripts/smoke_continuous_residual_correction.py`：PASS。覆盖 external
  Stage 1 provenance、single VQ512 与数据划分、strict load、zero-init、
  residual 精确定义（`h - z_q`）与 detach、初始 latent/完整 prediction
  forward 逐位等价、既有参数初始化不受扰动、非零 adapter 对不同 residual
  方向产生不同 correction、adapter 非零梯度与参数更新、Stage 1 梯度隔离、
  quantizer assignment / codebook 不变、Stage 2 checkpoint strict
  round-trip、标准 prediction 及 backtest normalizer。
- smoke 中 adapter weight/bias gradient L1 分别为 `270.8412780762` /
  `35.4411125183`；Stage 1 参数无梯度。
- 产物位于 `artifacts/023/smoke/`：`unit_tests.log`、`stage2.log`、
  `smoke_report.json`、`checkpoints/` 与 `res/`。

本阶段未启动正式长时间训练或正式回测。
