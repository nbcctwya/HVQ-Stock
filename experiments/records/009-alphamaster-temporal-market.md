# 009 — alphamaster-temporal-market

## Idea

基于 007 AlphaMaster baseline，在原 Market-Guided Feature Gate 前加入轻量
Temporal Market Encoder。将 canonical 20 日 market63 完整轨迹编码成 63 维
market state，再送入 007 原有 Feature Gate。

## Motivation

007 只使用最后一个交易日的 `market[:, -1, :]`，可能忽略市场状态随时间的
演化。本实验只回答：temporal market history 是否比 single-day market
snapshot 更适合作为 AlphaMaster Feature Gate 的市场状态输入。

## Modification

- 007：`market_state = market[:, -1, :]`。
- 009：`market_state = GRU(market).last_hidden`。
- 新增 batch-first 单层、单向 GRU：`input_size=63`、`hidden_size=63`、
  `num_layers=1`；无 attention、residual、projection、auxiliary loss 或新增
  market features。
- GRU 输出保持 `[N,63]`，继续输入 007 原有 `Gate(63 -> 158)`；Feature
  Gate 本体、stock158 reweighting 与其后完整 AlphaMaster backbone 均不变。
- 默认 `configs/config.yaml` 显式启用并完整描述该 GRU，无需实验特有 CLI
  override。

## Constraints

- canonical schema 保持 158 stock + 13 prior + 63 market + 10 future
  returns，`T=20`；prior13 完全 unused。
- Market-Guided Feature Gate、Linear projection、Positional Encoding、
  TAttention、SAttention、TemporalAttention、final decoder 均与 007 一致。
- CSI300 / SP500 Feature Gate beta 分别保持 10 / 5。
- 不含 Prior Factor Head、VQ/HVQ、z0/z1、MoE、regime quantization、RevIN
  或其他创新；market63 生成和 canonical parser 不变。
- 数据划分、训练预算、Stage 1 seed 42、Stage 2 seed 0、target_day=5、Adam
  `lr=8e-6`、指标及回测协议均与 007 一致。
- Stage 1 provenance 为 `self`：新增 GRU 可训练参数且改变 forward graph，
  007 checkpoint 缺少 `master.market_encoder.*`，不能 strict 加载。

## Git

Base: exp/007-alphamaster-baseline
Branch: exp/009-alphamaster-temporal-market
Commit: db5128fedd751b4de03b981760f39138717ec37c
Stage 1 provenance: self

## Smoke Test

Status: PASS

Notes: conda `prism-vq` Python 环境下完整单元测试 89/89 PASS。覆盖
canonical `[N,20,244]` 解析、stock `[N,20,158]`、market `[N,20,63]`、
market state `[N,63]`、Feature Gate 63 维输入、原 stock reweighting 路径、
prior13 不进入参数或 forward、修改 prior 后 prediction 逐位不变，以及只修改
`market[:, :-1, :]` 且保持最后一天逐位不变时 prediction 仍改变。四个 GRU
参数张量均验证具有非零梯度；prediction shape 与 007 一致。

`scripts/smoke_alphamaster.py` 使用 tiny canonical PKL 验证 CSI300 / SP500
forward PASS、beta 分别为 10 / 5、同 trading-day cross-section batching
PASS。Stage 1 限制为 1 epoch、2 train batches、2 validation batches，生成
`artifacts/009/smoke/checkpoints/alphamaster_smoke-epoch=0-val_loss=0.7461.ckpt`，
validation-best runner discovery PASS。Stage 2 strict 加载该 009 checkpoint
PASS，生成 `artifacts/009/smoke/res/alphamaster_csi300/0_best.pkl` 与
`0_metric.csv`；40 行标准 `score/label` prediction 被现有
`backtest_qlib.py` normalizer 接受。真实 007 smoke checkpoint strict load
到 009 按预期失败。日志与机器可读报告位于
`artifacts/009/smoke/stage1.log`、`stage2.log`、`smoke_report.json`。
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

