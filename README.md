# 010 — prism-shared-routed-moe

## Base

`main`（原始 PRISM-VQ baseline；single VQ512，Stage 2 seed 0）。

Stage 1 不重新训练，复用当前本地 corrected PRISM-VQ baseline 的 exact
checkpoint：

`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`

该文件与 `../PRISM-VQ/checkpoints/` 中的原始文件字节一致。RevIN、
SpatialEncoder、VectorQuantizer 均已 strict 加载验证，missing=0、
unexpected=0。

## Idea / Motivation

PRISM-VQ 用 `z_q` 对 sample 进行 latent-state-conditioned sparse expert
routing，但不同 latent state 之间也可能存在稳定共享的收益预测结构。纯
sparse routing 可能迫使 routed experts 重复学习这些 common patterns。

本实验把 Stage 2 MoE 显式拆成共享与专门化两部分：

`moe_out = E_shared(h) + sum_j g_j(z_q) E_j(h)`

目标是让 always-on Shared Expert 学习跨 latent state 的公共结构，让原有
Routed Experts 更专注于 `z_q` 区分的 specialized structure。

## 核心修改

- `FactorGatedMoE` 新增且仅新增一个 always-on Shared Expert。
- Shared Expert 与单个 routed expert 使用完全相同的
  `SimpleMLP(expert_input_size, expert_input_size, hidden_size)` 结构。
- Shared Expert 接收完整 batch 的 `h`，不经过 `SparseDispatcher`，不参与
  routing，也不占 top-k quota。
- Shared Expert 最终 Linear 的 weight 与 bias 显式 zero-init，初始化时
  `shared_out` 逐位为 0；最终输出仅做 `shared_out + routed_out`。
- `configs/config.yaml` 默认设置 `predictor.shared_expert: true`，直接运行
  默认配置即启用本实验结构。

## 与 base 的区别

唯一实验变量是 Stage 2 `FactorGatedMoE` 的输出从 routed-only 改成一个
zero-initialized shared residual 加原 routed output。原 routed experts 数量、
`k`、router、noisy top-k、`W_h`、SparseDispatcher、combine 逻辑及 routed
importance/load-balancing loss 均不变。

Stage 1、canonical dataset（包括 market63 继续 unused）、DLinear、Temporal
Transformer、`z_q` structure token、HyperFusion 后续 FiLM/alpha/beta heads、
LatentValueHead、prior13、ReturnPredictor、loss family、aux 权重、数据划分、
训练预算、seed 与回测协议均保持 `main` 不变。

## Smoke 状态

Status: **PASS**（仓库完整单元测试 91/91，本实验机制测试 9/9）。

已通过：

- `tests/test_shared_routed_moe.py` 的机制与等价性测试；
- 仓库完整单元测试；
- `scripts/smoke_shared_routed_moe.py` 的最小真实 Stage 2 shared+routed
  forward/backward、参数更新、checkpoint save/load、valid/test inference 与
  标准 prediction/metric 兼容性检查。

Smoke 产物统一写入 `artifacts/010/smoke/`；机器可读结论见
`smoke_report.json`，执行日志见 `stage2.log` 与 `unit_tests.log`。本阶段未
启动正式长时间训练或正式回测。
