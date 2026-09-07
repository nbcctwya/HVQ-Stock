# 010 — prism-shared-routed-moe

## Idea

基于 `main` 的原始 PRISM-VQ baseline，在 Stage 2 `FactorGatedMoE` 中新增
一个所有 sample 始终经过的 Shared Expert，与原 `z_q`-conditioned top-k
Routed Experts 并行，输出按元素残差相加：

`moe_out = E_shared(h) + sum_j g_j(z_q) E_j(h)`。

## Motivation

现有 sparse routing 能按 latent state 学习 specialized structure，但不同
latent states 之间可能还存在稳定共享的收益预测结构。纯 routed-only MoE
可能迫使多个 experts 重复学习这些 common patterns。显式拆分 shared 与
latent-state-specific structure，可能改善专家分工和横截面收益预测。

## Modification

- `FactorGatedMoE` 新增一个 always-on `shared_expert`，结构与单个 routed
  expert 完全相同：
  `SimpleMLP(expert_input_size, expert_input_size, hidden_size)`。
- Shared Expert 直接接收完整 batch 的 `x`，不经过 `SparseDispatcher`，不
  参与 routing、不占 top-k quota；原 dispatcher/combine 后得到
  `routed_out`，最终仅执行 `shared_out + routed_out`。
- Shared Expert 最终 Linear 的 weight 和 bias 显式 zero-init，初始化时
  `shared_out` 逐位为 0。
- Shared Expert 在所有 routed-path 参数之后创建，因此相同 seed 下 routed
  experts、router、noise network 与 `W_h` 的初始化逐位保持一致。
- `HyperFusion` / `LoadingGenerator` 仅透传开关；默认
  `configs/config.yaml` 设置 `predictor.shared_expert: true`，无需实验特有
  CLI override。
- 新增 `tests/test_shared_routed_moe.py` 与
  `scripts/smoke_shared_routed_moe.py`。

## Constraints

- 唯一实验变量为 Stage 2 routed-only MoE 新增一个 zero-initialized、
  always-on Shared Expert residual。
- routed expert 数量保持 2、`k=1`；router、noisy top-k、`W_h`、
  SparseDispatcher、expert combine 和 routed importance/load-balancing loss
  定义完全不变，Shared Expert 不计入负载统计且无新增 auxiliary loss。
- HyperFusion 后续 FiLM、alpha/beta heads、prior/latent decomposition、
  LatentValueHead 与 ReturnPredictor 不变。
- Stage 1 RevIN、SpatialEncoder、single VQ512、codebook 配置与训练逻辑不变。
- Stage 2 DLinear、Temporal Transformer、`z_q` structure token、prior13、
  loss family 与原 aux 权重不变。
- canonical 158 stock + 13 prior + 63 market + 10 return schema 不变；
  market63 继续 unused。
- 数据划分、70 epoch 预算、early stopping、Stage 2 seed 0 与既有回测协议
  不变；本阶段未进行正式训练或正式回测。
- Stage 1 来源为 external；复用 corrected PRISM-VQ baseline 当前正式使用的
  exact single-VQ512 checkpoint，不重训且缺失时不得 fallback。

## Git

Base: main
Branch: exp/010-prism-shared-routed-moe
Commit: 9b854f0436f8a7c3283fd375661dd6152cc965f1
Stage 1 provenance: external —
`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`
（与 `../PRISM-VQ/checkpoints/` 原始文件字节一致，MD5
`6b9d9dbfd938c7bd2c7dc5ee33cb38af`；该副本仅存在于当前 local workspace，
`artifacts/` 被 gitignore，缺失时 Phase 2 必须 loud fail。）

## Smoke Test

Status: PASS

Notes:

- conda `prism-vq` 下仓库完整单元测试 91/91 PASS；本实验机制测试 9/9
  PASS，覆盖 Shared/routed input-output shape、always-on full-batch path、
  routed initialization/forward 的逐位 base 等价、zero-init 最终输出等价、
  routed auxiliary-loss 等价、Shared final Linear 梯度与更新，以及
  HyperFusion 到 ReturnPredictor 的接口兼容。
- exact Stage 1 checkpoint strict 加载：Encoder、Quantizer、RevIN 均
  missing=0 / unexpected=0。
- `scripts/smoke_shared_routed_moe.py` 使用 synthetic canonical `[N,20,244]`
  batches 和 exact baseline Stage 1 checkpoint，真实执行 shared+routed
  forward/backward 与 optimizer step。Shared final Linear weight/bias 梯度
  L1 分别为 63.8391 / 13.3765，step 后二者均更新；Stage 1 frozen 参数无
  梯度。
- Stage 2 checkpoint 严格 save/load 后 prediction 逐位一致；valid/test
  inference 均完成，标准 `0_best.pkl` / `0_metric.csv` 被现有
  `backtest_qlib.py` prediction normalizer 接受。
- 产物：`artifacts/010/smoke/`（`stage2.log`、`unit_tests.log`、
  `smoke_report.json`、`checkpoints/`、`res/`）。Smoke 指标仅用于流程验证，
  未用于模型调整。

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

