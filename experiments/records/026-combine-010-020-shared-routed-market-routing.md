# 026 — combine-010-020-shared-routed-market-routing

## Idea

严格以实验 010 `prism-shared-routed-moe` 为 base，仅移植实验 020
`market-conditioned-routing` 的机制：在 010 的 Routed Experts 的 clean
routing logits 上加入 current-market additive bias：

```text
market_state = market_feature[:, -1, :]              # canonical [B,T,63] -> [B,63]
market_norm  = F.layer_norm(market_state, (63,))     # parameter-free
market_bias  = market_routing_adapter(market_norm)   # Linear(63, n_expert, bias=False), zero-init
clean_logits = Router(z_q) + market_bias             # bias 在 noise / W_h / top-k 之前
```

其余 routing 流程（noisy routing、`W_h`、top-k、SparseDispatcher、
Routed Experts）与 `moe_out = shared_out + routed_out` 完整保留 010。

## Motivation

010 已证明显式分解 common structure（always-on Shared Expert）与
latent-specific routed structure（Routed Experts）能显著改善原 PRISM-VQ；
020 单独使用时 IC / RankIC 未提升，但表现出不同的 portfolio risk-return
behavior（最大回撤与 Sharpe 改善）。若 market regime information 与 010
的 Shared/Routed decomposition 互补，market-conditioned routing 可能在保留
Shared Expert 所学 common structure 的同时，让 specialized Routed Experts
根据当前市场状态做更合理的 expert allocation，从而改善预测、组合或风险
调整后收益。zero-init 保证初始 `market_bias == 0`，026 与 010 逐位等价。

## Modification

- 从冻结的 `exp/010-prism-shared-routed-moe` 创建实验分支，不从 main 或
  020 重新拼装模型。
- `FactorGatedMoE` 在所有 010 模块之后、forked RNG 下新增且仅新增
  `Linear(63, n_expert, bias=False)` market routing adapter，weight 显式
  zero-init；相同 seed 下除新增 adapter 外全部既有 state tensor 初始化与
  010 逐位一致。
- 新增 `clean_routing_logits(x, market_state)` = 原 `Router(z_q)` +
  market bias，注入位置在 noise / `W_h` / top-k 之前；noisy routing、
  `W_h`、top-k、dispatcher、combine 与 auxiliary loss 逻辑不变。
- data plumbing：`_get_data` 保留 canonical `market_feature`，training /
  validation / standard inference 均传入；model 仅提取
  `market_feature[:, -1, :]`；LoadingGenerator / HyperFusion 只透传
  `market_state`；`market_state` 最终只进入 `FactorGatedMoE` router。
- 默认 `configs/config.yaml` 保持 `shared_expert: true` 并新增
  `market_conditioned_routing: true`，核心改动无需 CLI override。
- 新增 `tests/test_combine_010_020_shared_routed_market_routing.py` 与
  `scripts/smoke_combine_010_020_shared_routed_market_routing.py`（含只读
  routing diagnostic helpers）；分支根目录 README 改写为本实验说明。

## Constraints

- 唯一实验变量：在 010 上新增并仅新增 020 的 Market-Conditioned Routing。
- 020 mechanism 本身不变：只用 `market_feature[:, -1, :]`；market dim 固定
  63；parameter-free `F.layer_norm`；adapter 严格为
  `Linear(63, n_expert, bias=False)` 且 zero-init；bias 仅加到 Routed clean
  logits，且在 noise / `W_h` / top-k 之前；不使用历史 market sequence、
  GRU、EMA、attention、learnable LayerNorm、gate 或 scaling coefficient。
- market information 的唯一合法路径是
  `market_state -> market_routing_adapter -> Routed clean routing logits`；
  不进入 Shared Expert、expert input `x`、Temporal Transformer、
  HyperFusion hidden、LatentValueHead、factor heads、alpha/beta heads、
  ReturnPredictor 或最终 prediction。
- 保留 010 Shared-Routed MoE 完整结构：always-on Shared Expert（不参与
  routing、不占 top-k quota、不接收 market state）、2 个 Routed Experts、
  top-k = 1、原 router、原 noise network、原 `W_h`、原 SparseDispatcher、
  原 combine、原 importance/load-balancing auxiliary loss、
  `moe_out = shared_out + routed_out`。
- 不加入 016–025 的其他机制（无 adaptive shared fusion、decoupling、
  quantization confidence adapter、prior-latent allocation、code-aware
  routing、continuous residual correction、transition-aware routing）；
  不以 025 为 base；Stage 2 latent 严格为 010 原始 `z_q`。
- 不改变 FiLM、alpha/beta heads、prior/latent decomposition、
  LatentValueHead、ReturnPredictor、prediction loss、softcap、aux_weight
  或其他 objective；market feature 只来自 canonical 63 维 market fields，
  不使用 future return、label 或未来信息。
- Stage 1 的 RevIN、SpatialEncoder、single VQ512、128 维 codebook、
  assignment 与训练逻辑全部保持 010 不变；Stage 1 来源为复用实验 010 的
  正式 Stage 1 provenance。
- 数据划分（train 2009–2020、valid 2021–2022、test 2023–2025）、70 epoch
  预算、early stopping、optimizer、learning rate、Stage 2 seed 0、Stage 1
  seed 42、Top30/Drop5 回测协议及其他超参数均与 010 一致。
- Phase 1 未启动正式长时间训练或正式回测。

## Git

Base: exp/010-prism-shared-routed-moe
Branch: exp/026-combine-010-020-shared-routed-market-routing
Commit: e696ef06db0d87aab5064004c911ddf26998f3a9
Stage 1 provenance: 复用实验 010 的正式 Stage 1（`stage1_source: "010"`）；
`artifacts/010/run/.stage1.done` marker 存在，`reused=true`，其记录的
source commit `9b854f0436f8a7c3283fd375661dd6152cc965f1` 与 canonical
queue 中 010 pinned commit 完全一致；marker 指向 corrected PRISM-VQ exact
checkpoint
`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`
（14,584,929 bytes；MD5 `6b9d9dbfd938c7bd2c7dc5ee33cb38af`）。

## Smoke Test

Status: PASS

Notes:

- `conda run -n prism-vq python -m unittest discover -s tests -v`：完整测试
  110/110 PASS；026 新增组合机制测试覆盖 010 inheritance（Shared Expert
  always-on、不参与 routing/top-k、`moe_out = shared_out + routed_out`、
  aux loss 不变、无 016–025 机制泄漏）、020 mechanism fidelity
  （latest-timestep-only、parameter-free layer_norm、adapter 规格与
  zero-init、注入位置）、zero-init 对 010 的逐位等价、market-routing
  semantics（含 loud fail）、trainability 与 no-leakage；既有 010 机制测试
  与 Stage 1 freeze 回归测试继续 PASS。
- smoke 验证 010 Stage 1 provenance：marker 存在、commit 匹配 queue pinned
  commit、checkpoint 存在非空且 MD5 不变；Encoder、Quantizer、RevIN strict
  load（missing=0 / unexpected=0）；single VQ512、128 维 embedding 与原
  数据划分完全兼容。
- zero-init 时 clean logits、gates/load、MoE 输出、original auxiliary
  loss、Shared Expert 输出与完整 prediction forward 均与 010 逐位相等；
  相同 noise seed 下 noisy logits 与 010 逐位一致；相同 seed 下除新增
  adapter 外全部既有 state tensor 初始化与 010 逐位一致。
- synthetic canonical smoke 完成真实 backward 与 optimizer step：market
  adapter 获得 finite non-zero 梯度并更新，Shared Expert 与 Routed Experts
  仍获得有效训练信号；Stage 1 参数无梯度、保持 eval、codebook 与
  quantizer assignment 不变。Stage 2 checkpoint strict round-trip 后输出
  逐位一致；标准 inference、metric CSV 与 backtest prediction normalizer
  均 PASS。
- routing 诊断（只读，不改训练目标）：zero-init expert-switch rate = 0；
  一步训练后 `market_bias` mean `1.86e-9` / std `5.46e-3`，
  `||market_bias||` `0.0181`，`||Router(z_q)||` `0.3464`，比值 `0.0523`；
  post-step expert-switch rate `0.0`；synthetic 非零 adapter expert-switch
  rate `1.0`；adapter weight norm `1.12e-3`、weight grad L1 `0.632`；
  Shared Expert final weight grad L1 `96.51`；Routed Experts grad L1
  `229.83`；per-date Expert 0/1 使用比例已记录。
- 产物位于 `artifacts/026/smoke/`：`unit_tests.log`、`stage2.log`、
  `smoke_report.json`、`checkpoints/` 与 `res/`。

## Result

Status: DONE（test 区间 2023-01-01 – 2025-12-31）

IC: 0.0331
ICIR: 0.1789
RankIC: 0.0511
RankICIR: 0.2679

Annual Return: 11.52%（基准 6.40%，超额 5.12%）
Sharpe: 0.7644
Sortino: 1.1260
MDD: -13.67%
Calmar: 0.8426
Turnover: 0.3244

## Conclusion

Phase 2 固定执行器完成正式训练、预测与回测（pinned commit e696ef06db0d87aab5064004c911ddf26998f3a9）。Stage 1 复用实验 010 的正式 checkpoint：`/home/nbcctwya/baselines/masterVQ/HVQ-Stock/artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`（本实验未重新训练 Stage 1）；Stage 2 seed 0。

产物：`artifacts/026/run/`（checkpoints/、res/、stage1.log、stage2.log、backtest.log、summary.json）。
