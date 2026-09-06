# 008 — alphamaster-prior-factor-head

## Idea

基于 007 AlphaMaster baseline，将最终单一预测层替换为 Prior Factor Head：
由 AlphaMaster 最终 hidden representation 同时生成隐式 alpha 与动态 13 维
prior-factor loading，并以 `y_hat = alpha + beta^T prior` 预测收益。

## Motivation

验证 HVQ canonical dataset 已有的 13 维 JKP prior factors，能否在不进入
AlphaMaster backbone 的条件下，为其 market-aware、temporal-aware、
cross-sectional stock representation 提供互补的显式金融先验信息。

本实验只验证：dynamic prior-factor decomposition 是否能在 AlphaMaster
baseline 上带来增量信息。

## Modification

- 保持 007 的 hidden representation `h` 生成链路完全不变：Market-Guided
  Feature Gate → linear projection → Positional Encoding → TAttention →
  SAttention → TemporalAttention。
- 将 007 的 `decoder = Linear(256, 1)` 替换为两个 unconstrained linear
  heads：`alpha_head = Linear(256, 1)` 与
  `prior_loading_head = Linear(256, 13)`。
- canonical parser 原样提供当日 `prior_factor` `[N,13]`。模型先只用
  `[stock158, market63]` 得到 `h`，再计算 `alpha = alpha_head(h)`、
  `beta = prior_loading_head(h)`、`prior_contribution = sum(beta * prior)`，
  最终 `prediction = alpha + prior_contribution`。
- 默认 `configs/config.yaml` 明确包含 `alphamaster.num_prior_factors: 13`；
  不需要实验特有 CLI override 即代表 008。
- 新增/改写 AlphaMaster 单测与 `scripts/smoke_alphamaster.py` 验证 factor
  formula、shape、动态 loading、prior isolation、determinism、两市场 forward、
  checkpoint 与现有预测格式兼容性；smoke 附带记录 alpha/prior/beta 诊断统计，
  不改变 loss。

## Constraints

- 唯一实验变量：007 的最终 `Linear(h -> prediction)` 改为
  `AlphaHead(h) + PriorLoadingHead(h)^T prior13`。
- prior13 不 concat 到输入或 backbone；不进入 Market Gate、TAttention、
  SAttention、TemporalAttention；不修改 prior 的生成、预处理或定义。
- beta 为 unconstrained dynamic linear loading；无 sigmoid/tanh/softmax、
  sparsity、额外 loss 或其他约束。
- Market-Guided Feature Gate、Positional Encoding、TAttention、SAttention、
  TemporalAttention 及 market63 用法与 007 完全一致。
- canonical schema 保持 158 stock + 13 prior + 63 market + 10 future returns，
  T=20；数据 split、target_day=5、Stage 1 seed 42、Stage 2 seed 0、训练预算、
  optimizer、指标及回测协议均不变。
- 不引入 VQ/HVQ、z0/z1、MoE、market encoder、regime mechanism 或 RevIN。

## Git

Base: exp/007-alphamaster-baseline
Branch: exp/008-alphamaster-prior-factor-head
Commit: 82fa0a49c0c2227eba851829a12fec9aace2037a
Stage 1 provenance: self（008 改变可训练 prediction-head 参数与 checkpoint
keys；007 的 `master.decoder.*` 与 008 的 `master.alpha_head.*` /
`master.prior_loading_head.*` 不 strict compatible，不能复用 007 Stage 1）

## Smoke Test

Status: PASS

Notes: conda `prism-vq` 下完整单元测试 88/88 PASS；覆盖 canonical
`[N,20,244]` → stock `[N,20,158]` / prior `[N,13]` / market
`[N,20,63]`，alpha `[N]`、beta `[N,13]`、prediction `[N]`，并以零容差
验证 `prediction == alpha + sum(beta * prior)`。固定 stock/market 改 prior
会改变 prediction，但 hidden/alpha/beta 逐位不变；重复 forward 逐位一致；
beta 由 hidden 经 trainable linear head 动态生成且跨股票不同。与 standalone
AlphaMaster 同权重的 backbone hidden forward 逐位一致；007-style 单头
checkpoint strict load 失败断言 PASS。

`scripts/smoke_alphamaster.py` 使用 tiny canonical PKL 完成入口级 smoke：
CSI300 与 SP500 `[4,20,244]` forward 均 PASS，market-gate beta 分别保持
10 / 5，factor formula、prior sensitivity/isolation、dynamic beta、
deterministic forward 与 market sensitivity 均 PASS。

Stage 1 限制为 1 epoch、2 train batches、2 validation batches，生成
`artifacts/008/smoke/checkpoints/alphamaster_smoke-epoch=0-val_loss=2.4168.ckpt`，
validation-best runner discovery PASS。Stage 2 strict 加载该 008 Stage 1
checkpoint PASS，并生成
`artifacts/008/smoke/res/alphamaster_csi300/0_best.pkl` 与 `0_metric.csv`。
Prediction 为 40 行标准 `score/label` DataFrame，现有 `backtest_qlib.py`
normalizer 接受。机器可读报告与日志位于
`artifacts/008/smoke/smoke_report.json`、`stage1.log`、`stage2.log`。
未执行正式长时间训练或正式回测。

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
