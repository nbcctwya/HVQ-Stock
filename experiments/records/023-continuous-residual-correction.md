# 023 — continuous-residual-correction

## Idea

在原始 corrected PRISM-VQ baseline 的 Stage 2 前加入 Continuous Residual
Correction。使用冻结 Stage 1 的连续表示 `h` 与 hard-quantized latent `z_q`
计算完整的 instance-specific continuous residual：

```text
r = h - z_q
delta_z = f(r)
z_stage2 = z_q + delta_z
```

其中 `f` 严格为 zero-initialized `Linear(128, 128)`；后续原本使用 `z_q` 的
Stage 2 模块统一使用 `z_stage2`。

## Motivation

hard vector quantization 提供了有利的离散化与正则化，但同时丢弃了样本相对
于 prototype 的细粒度连续偏差信息。使用一个受控的小型 residual adapter，在
保留 discrete prototype 作为主体表示的同时恢复部分 instance-specific
information，可能改善收益排序预测。

与 019 quantization-confidence-adapter 不同：本实验直接利用完整的 128 维
residual vector `h - z_q`，而不是仅利用 residual magnitude /
quantization-error scalar。

## Modification

- `GenerateReturn` 在所有 baseline 模块构造完成后新增且仅新增一个
  `Linear(128, 128)` adapter；weight 与 bias 显式全零初始化。
- 用 detach 后的 `h` 与 `z_q` 计算 residual `r = h.detach() - z_q.detach()`，
  并构造 `z_stage2 = z_q + adapter(r)`。
- `LoadingGenerator`（含 temporal structure token、MoE/HyperFusion）与
  `LatentValueHead` 统一接收 `z_stage2`。
- adapter 在全部既有模块之后创建，确保相同 seed 下新增层不扰动任何 baseline
  参数初始化；zero-init 确保初始 `z_stage2 == z_q`，完整 prediction forward
  （含原 auxiliary loss）与 `main` 逐位相等。
- 默认 `configs/config.yaml` 设置
  `predictor.residual_correction_adapter: true`，核心改动无需 CLI override。
- 新增 `tests/test_continuous_residual_correction.py` 与
  `scripts/smoke_continuous_residual_correction.py`。

## Constraints

- 唯一实验变量是新增上述 continuous residual correction adapter。
- Stage 1 encoder、RevIN、single VQ512 quantizer、128 维 codebook、assignment、
  loss 与训练逻辑完全不变；`h`、`z_q` 与 residual 均不允许将梯度传回 Stage 1。
- Stage 2 的 DLinear、Temporal Transformer、原 Routed MoE、HyperFusion、
  prior/latent heads、ReturnPredictor、prediction loss 与 auxiliary loss 均不变。
- 不加入 quantization-confidence scalar、Shared Expert、adaptive fusion、
  decoupling、market-conditioned routing、prior-latent allocation、
  code-aware routing 或其他机制。
- canonical `158 stock + 13 prior + 63 market + 10 returns = 244` schema、market63
  unused 行为、数据划分（train 2009–2020、valid 2021–2022、test 2023–2025）、
  70 epoch 预算、early stopping、Stage 1 seed 42、Stage 2 seed 0、Top30/Drop5
  回测协议及其他超参数均与 `main` 一致。
- Stage 1 来源为 external corrected PRISM-VQ exact checkpoint；Phase 1 未启动
  正式长时间训练或正式回测。

## Git

Base: main
Branch: exp/023-continuous-residual-correction
Commit: 1ec3f2027e18b2654c71549423bf36eeec18a6ee
Stage 1 provenance: external corrected PRISM-VQ exact checkpoint：
`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`
（14,584,929 bytes；MD5 `6b9d9dbfd938c7bd2c7dc5ee33cb38af`）。

## Smoke Test

Status: PASS

Notes:

- `conda run -n prism-vq python -m unittest discover -s tests -v`：完整测试
  94/94 PASS；023 新增机制测试 12/12 PASS，既有 Stage 1 freeze 回归测试继续
  PASS。
- external checkpoint 存在且非空；模型保持唯一 `VectorQuantiser`、codebook
  shape `(512, 128)` 与原数据划分。Encoder、Quantizer、RevIN strict load 均为
  missing=0 / unexpected=0。
- 单元测试与 smoke 均确认 adapter 是 zero-initialized `Linear(128, 128)`，
  residual 精确等于 `h - z_q` 且 detached；初始 `z_stage2` 与 `z_q`、023 与
  `main` 的完整 prediction forward（含 auxiliary loss）均逐位相等。
- 相同 seed 下除新增 adapter 外全部既有 state tensor 初始化逐位相等；hook
  验证 `LoadingGenerator` 与 `LatentValueHead` 均接收同一个 `z_stage2`。
- 非零 adapter 下不同 residual direction 产生不同 `delta_z` / `z_stage2`；
  quantizer assignment 与 codebook 在 Stage 2 更新后均不变。
- synthetic canonical `[N,20,244]` smoke 完成真实 backward 与 optimizer step；
  adapter weight/bias gradient L1 分别为 `270.8412780762` / `35.4411125183`
  并更新，Stage 1 参数无梯度。
- Stage 2 checkpoint strict round-trip 后输出逐位一致；标准 12 行 prediction、
  metric CSV 与 backtest prediction normalizer 均 PASS。
- 产物位于 `artifacts/023/smoke/`：`unit_tests.log`、`stage2.log`、
  `smoke_report.json`、`checkpoints/` 与 `res/`。

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
