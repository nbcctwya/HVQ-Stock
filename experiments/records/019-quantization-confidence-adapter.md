# 019 — quantization-confidence-adapter

## Idea

在原始 corrected PRISM-VQ baseline 的 Stage 2 前加入 Quantization-Confidence
Adapter。使用冻结 Stage 1 的连续表示 `h` 与 hard-quantized latent `z_q` 计算：

```text
q_error = mean((h - z_q)^2, dim=-1, keepdim=True)
z_stage2 = z_q + f(q_error)
```

其中 `f` 严格为 zero-initialized `Linear(1, 128)`；后续原本使用 `z_q` 的
Stage 2 模块统一使用 `z_stage2`。

## Motivation

hard vector quantization 只保留最近 code 的表示，却丢弃样本与 prototype 之间
的距离信息。quantization error 可能表征 latent assignment 的可靠程度或边界
模糊程度；将该信息通过最轻量的 residual adapter 重新提供给 Stage 2，可能
改善预测表现，同时保留离散 VQ 的 inductive bias。

## Modification

- `GenerateReturn` 在所有 baseline 模块构造完成后新增且仅新增一个
  `Linear(1, 128)` adapter；weight 与 bias 显式全零初始化。
- 用 detach 后的 `h` 与 `z_q` 按 latent 维均方差计算 per-sample scalar
  `q_error`，并构造 `z_stage2 = z_q + adapter(q_error)`。
- `LoadingGenerator`（含 temporal structure token、MoE/HyperFusion）与
  `LatentValueHead` 统一接收 `z_stage2`。
- adapter 在全部既有模块之后创建，确保相同 seed 下新增层不扰动任何 baseline
  参数初始化；zero-init 确保初始 `z_stage2 == z_q`，完整 prediction forward
  与 `main` 逐位相等。
- 默认 `configs/config.yaml` 设置
  `predictor.quantization_confidence_adapter: true`，核心改动无需 CLI override。
- 新增 `tests/test_quantization_confidence_adapter.py` 与
  `scripts/smoke_quantization_confidence_adapter.py`。

## Constraints

- 唯一实验变量是新增上述 quantization-error-conditioned residual adapter。
- Stage 1 encoder、RevIN、single VQ512 quantizer、128 维 codebook、assignment、
  loss 与训练逻辑完全不变；`h`、`z_q` 与 `q_error` 均不允许将梯度传回 Stage 1。
- Stage 2 的 DLinear、Temporal Transformer、原 Routed MoE、HyperFusion、
  prior/latent heads、ReturnPredictor、prediction loss 与 auxiliary loss 均不变。
- 不加入 Shared Expert、adaptive fusion、decoupling、market information、
  continuous residual correction 或其他机制。
- canonical `158 stock + 13 prior + 63 market + 10 returns = 244` schema、market63
  unused 行为、数据划分（train 2009–2020、valid 2021–2022、test 2023–2025）、
  70 epoch 预算、early stopping、Stage 1 seed 42、Stage 2 seed 0、Top30/Drop5
  回测协议及其他超参数均与 `main` 一致。
- Stage 1 来源为 external corrected PRISM-VQ exact checkpoint；Phase 1 未启动
  正式长时间训练或正式回测。

## Git

Base: main
Branch: exp/019-quantization-confidence-adapter
Commit: f2a928889448e3671755b4277fdc969801238a62
Stage 1 provenance: external corrected PRISM-VQ exact checkpoint：
`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`
（14,584,929 bytes；MD5 `6b9d9dbfd938c7bd2c7dc5ee33cb38af`）。

## Smoke Test

Status: PASS

Notes:

- `conda run -n prism-vq python -m unittest discover -s tests -v`：完整测试
  91/91 PASS；019 新增机制测试 9/9 PASS，既有 Stage 1 freeze 回归测试继续
  PASS。
- external checkpoint 存在且非空；模型保持唯一 `VectorQuantiser`、codebook
  shape `(512, 128)` 与原数据划分。Encoder、Quantizer、RevIN strict load 均为
  missing=0 / unexpected=0。
- 单元测试与 smoke 均确认 adapter 是 zero-initialized `Linear(1, 128)`，
  `q_error` 精确等于 `mean((h-z_q)^2, dim=-1, keepdim=True)` 且 detached；初始
  `z_stage2` 与 `z_q`、019 与 `main` 的完整 prediction forward 均逐位相等。
- 相同 seed 下除新增 adapter 外全部既有 state tensor 初始化逐位相等；hook
  验证 `LoadingGenerator` 与 `LatentValueHead` 均接收同一个 `z_stage2`。
- synthetic canonical `[N,20,244]` smoke 完成真实 backward 与 optimizer step；
  adapter weight/bias gradient L1 分别为 `0.3727933168` / `30.3328056335` 并
  更新，Stage 1 参数无梯度。
- Stage 2 checkpoint strict round-trip 后输出逐位一致；标准 12 行 prediction、
  metric CSV 与 backtest prediction normalizer 均 PASS。
- 产物位于 `artifacts/019/smoke/`：`unit_tests.log`、`stage2.log`、
  `smoke_report.json`、`checkpoints/` 与 `res/`。

## Result

Status: DONE（test 区间 2023-01-01 – 2025-12-31）

IC: 0.0377
ICIR: 0.2230
RankIC: 0.0560
RankICIR: 0.3351

Annual Return: 10.29%（基准 6.40%，超额 3.89%）
Sharpe: 0.5868
Sortino: 0.8863
MDD: -25.95%
Calmar: 0.3963
Turnover: 0.3289

## Conclusion

Phase 2 固定执行器完成正式训练、预测与回测（pinned commit f2a928889448e3671755b4277fdc969801238a62）。Stage 1 复用外部 exact checkpoint：`/home/nbcctwya/baselines/masterVQ/HVQ-Stock/artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`（本实验未重新训练 Stage 1）；Stage 2 seed 0。

产物：`artifacts/019/run/`（checkpoints/、res/、stage1.log、stage2.log、backtest.log、summary.json）。
