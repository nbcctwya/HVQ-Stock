# 016 — latent-adaptive-shared-fusion

## Base

`exp/010-prism-shared-routed-moe`。

Stage 1 不重新训练，复用实验 `010` 的正式 Stage 1 marker 所指向的 exact
checkpoint：

`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`

该来源保持原 PRISM-VQ 的 SpatialEncoder、RevIN 与 single VQ512 配置。
实验 010 的 marker commit 与其 queue pinned commit 一致；Encoder、Quantizer、
RevIN 均已 strict 加载验证，missing=0、unexpected=0。

## Idea / Motivation

010 已验证 always-on Shared Expert 与 latent-conditioned Routed Experts 的显式
分工有效，但固定相加默认假定所有离散 latent state 对共享信息的依赖程度相同。
016 让 `z_q` 自适应控制 Shared Expert 的贡献：

```text
alpha = 1 + 0.5 * tanh(f(z_q))
moe_out = alpha * shared_out + routed_out
```

目标是检验不同 latent state 是否能通过有限幅度的共享分支缩放，进一步改善
shared structure 与 specialized structure 的协同。

## 核心修改

- `f` 是尽可能简单的 `Linear(128, 1)`，直接接收原始 `z_q`，输出 per-sample
  标量；weight 与 bias 显式全零初始化。
- `delta` 固定为 `0.5`，因此 `alpha` 始终位于 `(0.5, 1.5)`；初始化时
  `f(z_q)=0`、`alpha=1`。
- 初始化等价性测试使用人为设为非零的 Shared Expert，确认 016 与 010 的
  `shared_out + routed_out` 逐位相等，避免 Shared Expert 自身的零初始化掩盖
  fusion 是否真正等价。
- 新映射的构造隔离并恢复 RNG 状态，因此相同 seed 下 010 所有既有参数的
  初始化逐位不变。
- 默认 `configs/config.yaml` 设置
  `predictor.adaptive_shared_fusion: true`，无需实验特有 CLI override。

## 与 base 的区别

唯一实验变量是把 010 固定的 `shared_out + routed_out` 改为
`alpha(z_q) * shared_out + routed_out`。

010 的 Shared Expert 结构及零初始化、原 PRISM-VQ routed experts、router、
noisy top-k（2 experts、`k=1`）、`W_h`、SparseDispatcher、expert combine、
importance/load-balancing loss 均保持不变；Shared Expert 仍不参与 routing、
不占 top-k quota，也没有新增 auxiliary loss。router 继续使用 010 原有的
LayerNorm 后 latent，只有新增 `f` 读取原始 `z_q`。

Stage 1、canonical dataset（market63 继续 unused）、DLinear、Temporal
Transformer、`z_q` structure token、HyperFusion 后续 FiLM/alpha/beta heads、
LatentValueHead、prior13、ReturnPredictor、loss family、aux 权重、数据划分、
70 epoch 预算、early stopping、Stage 1 seed 42、Stage 2 seed 0 与回测协议
均与 010 相同。

## Smoke 状态

Status: **PASS**。

- `conda run -n prism-vq python -m unittest discover -s tests`：100/100 PASS。
- `scripts/smoke_adaptive_shared_fusion.py`：PASS；覆盖实验 010 Stage 1 marker
  provenance、single VQ512 strict load、初始化时 `alpha=1` 与 010 forward
  逐位等价、Shared Expert 与 fusion map 的梯度/更新、Stage 1 冻结、Stage 2
  checkpoint strict save/load、valid/test inference 和 backtest prediction
  normalizer。
- 产物位于 `artifacts/016/smoke/`：`unit_tests.log`、`stage2.log`、
  `smoke_report.json`、`checkpoints/` 与 `res/`。

本阶段未启动正式长时间训练或正式回测。
