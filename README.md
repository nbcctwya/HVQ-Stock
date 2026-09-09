# 026 — combine-010-020-shared-routed-market-routing

## Base

`exp/010-prism-shared-routed-moe`（pinned at the 010 Final Experiment
Commit）。不从 `main` 或 020 分支重新拼装模型；本分支直接从冻结的 010
分支创建，仅移植实验 020 `market-conditioned-routing` 的机制。

Stage 1 不重新训练，复用实验 010 的正式 Stage 1 provenance
（`stage1_source: "010"`）：`artifacts/010/run/.stage1.done` marker 存在，
`reused=true`，其记录的 source commit 与 canonical queue 中 010 pinned
commit 完全一致；marker 指向 corrected PRISM-VQ baseline exact checkpoint：

`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`

该文件与 `../PRISM-VQ/checkpoints/` 中的原始文件字节一致（14,584,929
bytes，MD5 `6b9d9dbfd938c7bd2c7dc5ee33cb38af`）。RevIN、SpatialEncoder、
VectorQuantizer 均 strict 加载验证（missing=0、unexpected=0），single
VQ512、128 维 codebook 与原数据划分完全兼容。

## Idea / Motivation

010 已证明显式分解 common structure（always-on Shared Expert）与
latent-specific routed structure（Routed Experts）能显著改善原
PRISM-VQ；020 虽然单独使用时 IC / RankIC 未提升，但表现出明显不同的
portfolio risk-return behavior（尤其最大回撤与 Sharpe 改善）。

若 market regime information 与 010 的 Shared/Routed decomposition 互补，
则 market-conditioned routing 可能在保留 Shared Expert 所学 common
structure 的同时，让 specialized Routed Experts 根据当前市场状态做更合理
的 expert allocation，从而改善预测表现、组合表现或风险调整后收益。

## 010 的 Shared-Routed 机制（完整保留）

`moe_out = shared_out + routed_out`

- Shared Expert：always-on，结构与单个 routed expert 相同
  （`SimpleMLP`），最终 Linear zero-init；不参与 routing、不占 top-k
  quota，接收完整 batch 的 `x`，不经过 `SparseDispatcher`。
- Routed Experts：2 个，top-k = 1，由原 PRISM-VQ router 根据当前 latent
  `z_q` 选择；原 router、原 noise network、原 `W_h`、原 SparseDispatcher、
  原 routed expert combine、原 importance/load-balancing auxiliary loss
  全部不变。

## 020 的 Market-Conditioned Routing 机制（原样继承）

从 canonical market feature `[B, T, 63]` 仅取当前窗口最后一个 timestep：

```text
market_state = market_feature[:, -1, :]              # [B, 63]
market_norm  = F.layer_norm(market_state, (63,))     # parameter-free
market_bias  = market_routing_adapter(market_norm)   # Linear(63, n_expert, bias=False), zero-init
clean_logits = Router(z_q) + market_bias             # bias 在 noise / W_h / top-k 之前加入
```

之后完整保留 010 的 routing 流程：clean logits → noisy routing → `W_h`
→ top-k → SparseDispatcher → Routed Experts。

## 026 的组合方式与唯一差异

026 = 010 的全部结构 + 仅新增 020 的 market additive bias。与 010 的唯一
差异：Routed Experts 的 clean routing logits 从 `Router(z_q)` 变为
`Router(z_q) + market_bias`。新增 adapter 在 forked RNG 下构造并显式
zero-init，因此初始化时 `market_bias == 0`，026 与 010 逐位等价；相同
seed 下除新增 adapter 外全部既有 state tensor 初始化与 010 逐位一致。

**Market information 的唯一合法作用路径**：

`market_state -> market_routing_adapter -> Routed clean routing logits`

market information 不进入 Shared Expert、expert input `x`、Temporal
Transformer、HyperFusion hidden representation、LatentValueHead、
prior/latent factor heads、alpha/beta heads、ReturnPredictor 或最终
prediction；不存在任何 market -> prediction 的直接路径。

**Shared Expert 为什么保持 market-agnostic**：Shared Expert 的职责是学习
跨 latent state、跨市场状态稳定的 common structure；若让它接收 market
state，就会把 regime-specific 信息泄漏进 always-on 路径，破坏 010 的
common/specialized 分解语义，也违背本实验"market 只影响 expert
selection"的唯一变量约束。因此 Shared Expert 的输入、输出与梯度路径与
010 完全一致。

`configs/config.yaml` 默认 `shared_expert: true` 且新增
`market_conditioned_routing: true`，直接运行默认配置即为本实验，无需任何
实验特有 CLI override。

## 保持不变的条件

Stage 1（RevIN、SpatialEncoder、single VQ512、128 维 codebook、assignment
与训练逻辑）、Stage 2 latent 仍为 010 原始 `z_q`（无 `z_conf` 或其他
corrected latent）、DLinear、Temporal Transformer、HyperFusion FiLM /
alpha/beta heads、LatentValueHead、ReturnPredictor、prediction loss、
softcap、aux_weight、数据划分、训练预算、early stopping、optimizer、
learning rate、Stage 2 seed 0、Stage 1 seed 42、Top30/Drop5 回测协议均与
010 一致。不引入 016–025 的任何其他机制（无 adaptive shared fusion、
decoupling、quantization confidence、prior-latent allocation、code-aware
routing、continuous residual correction、transition-aware routing）。

## Smoke 状态

Status: **PASS**。

- 仓库完整单元测试 PASS；新增
  `tests/test_combine_010_020_shared_routed_market_routing.py` 覆盖 010
  inheritance、020 mechanism fidelity、zero-init 对 010 的逐位等价、
  market-routing semantics、trainability 与 no-leakage；既有 010 机制
  测试与 Stage 1 freeze 回归测试继续 PASS。
- `scripts/smoke_combine_010_020_shared_routed_market_routing.py`：验证
  Stage 1 provenance（marker commit == queue pinned commit、checkpoint
  存在非空且 MD5 不变、strict load）、zero-init 与 010 的 routing/aux/
  shared/full-forward 逐位等价、真实 backward 与 optimizer step（adapter /
  Shared / Routed 均获得有效梯度，Stage 1 冻结且 codebook 不变）、
  checkpoint strict round-trip、标准 inference / metric / backtest
  normalizer 兼容。
- routing 诊断（只读，不参与训练目标）：market_bias mean/std/norm、
  `||Router(z_q)||`、二者比值、zero-init expert-switch rate = 0、
  synthetic 非零 adapter 的 expert-switch capability、per-date Expert 0/1
  使用比例、adapter weight/gradient norm、Shared/Routed 路径有效梯度。

Smoke 产物统一写入 `artifacts/026/smoke/`；机器可读结论见
`smoke_report.json`，执行日志见 `stage2.log` 与 `unit_tests.log`。本阶段
未启动正式长时间训练或正式回测。
