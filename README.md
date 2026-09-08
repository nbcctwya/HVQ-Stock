# 019 — quantization-confidence-adapter

## Base

`main`（原始 corrected PRISM-VQ baseline）。

Stage 1 不重新训练，复用 corrected PRISM-VQ exact checkpoint：

`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`

该文件为 14,584,929 bytes，MD5 为
`6b9d9dbfd938c7bd2c7dc5ee33cb38af`。SpatialEncoder、Quantizer 与 RevIN
均已 strict 加载验证（missing=0、unexpected=0）；模型保持 single VQ512、
128 维 embedding 与原数据划分。

## Idea / Motivation

hard vector quantization 只把连续表示 `h` 映射到最近的 code `z_q`，没有把
样本与 prototype 的距离提供给 Stage 2。该距离可能反映 latent assignment 的
可靠程度或边界模糊程度。

本实验从冻结的 Stage 1 输出计算 per-sample scalar quantization error：

```text
q_error = mean((h - z_q)^2, dim=-1, keepdim=True)
z_stage2 = z_q + f(q_error)
```

然后让 Stage 2 统一使用 `z_stage2`，检验这项轻量置信度信息能否改善预测。

## 核心修改

- `f` 严格为单层 `Linear(1, 128)`，weight 与 bias 显式全零初始化。
- 初始化时 `f(q_error)=0`，因此 `z_stage2 == z_q`；完整 prediction forward
  与 `main` 逐位相等。
- 原来使用 `z_q` 的全部 Stage 2 consumer——`LoadingGenerator`（包括 temporal
  structure token 与 MoE/HyperFusion）和 `LatentValueHead`——统一改用
  `z_stage2`。
- `h` 与 `z_q` 在 quantization-error 路径中显式 detach；Stage 1 参数继续冻结，
  Encoder、Quantizer 与 RevIN 继续保持 eval mode。
- adapter 在所有 baseline 模块构造完成后才追加；相同 seed 下，除新增 adapter
  外的全部既有 state tensor 初始化逐位不变。
- 默认 `configs/config.yaml` 设置
  `predictor.quantization_confidence_adapter: true`，无需实验特有 CLI override。

## 与 base 的区别

唯一实验变量是新增上述 quantization-error-conditioned residual adapter。

Stage 1 encoder、quantizer、codebook、assignment、loss 与训练逻辑均不变；
Stage 2 的 DLinear、Temporal Transformer、MoE、HyperFusion、prior/latent heads、
prediction/loss 定义也不变。不加入 Shared Expert、adaptive fusion、decoupling、
market information、continuous residual correction 或其他机制。

canonical `158 stock + 13 prior + 63 market + 10 returns = 244` schema、数据划分
（train 2009–2020、valid 2021–2022、test 2023–2025）、70 epoch 预算、early
stopping、Stage 1 seed 42、Stage 2 seed 0、其他超参数及 Top30/Drop5 回测协议
均保持 `main` 不变。

## Smoke 状态

Status: **PASS**。

- `conda run -n prism-vq python -m unittest discover -s tests -v`：91/91 PASS；
  新增机制测试 9/9 PASS，既有 Stage 1 freeze 回归测试继续 PASS。
- `scripts/smoke_quantization_confidence_adapter.py`：PASS。覆盖 external Stage 1
  provenance、single VQ512 与数据划分、strict load、zero-init、q_error 精确定义、
  初始 latent/完整 prediction forward 逐位等价、既有参数初始化不受扰动、
  adapter 非零梯度与参数更新、Stage 1 梯度隔离、Stage 2 checkpoint strict
  round-trip、标准 prediction 及 backtest normalizer。
- smoke 中 adapter weight/bias gradient L1 分别为 `0.3727933168` /
  `30.3328056335`；Stage 1 参数无梯度。
- 产物位于 `artifacts/019/smoke/`：`unit_tests.log`、`stage2.log`、
  `smoke_report.json`、`checkpoints/` 与 `res/`。

本阶段未启动正式长时间训练或正式回测。
