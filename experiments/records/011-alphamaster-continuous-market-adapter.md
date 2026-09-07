# 011 — alphamaster-continuous-market-adapter

## Idea

基于 007 AlphaMaster baseline，保留原 current-market Feature Gate，并新增一条
独立的 historical market prediction-side conditioning 分支。前 19 个交易日的
market63 经轻量 GRU 编码为连续市场状态，再由 zero-initialized 线性
hypernetwork 生成当天共享的动态 decoder residual weights：

`prediction = decoder(h) + sum(MarketAdapter(GRU(market[:, :-1, :])) * h)`。

## Motivation

验证两类市场信息能否承担互补而不互相替代的作用：当天市场状态继续在
AlphaMaster 输入侧选择和重加权股票特征；历史市场轨迹则在 MASTER 已提取
股票 latent representation 后，动态修正最终 prediction function。

## Modification

- 007 原路径逐项保留：`market[:, -1, :] -> Feature Gate -> stock158 ->`
  `Linear -> PositionalEncoding -> TAttention -> SAttention ->`
  `TemporalAttention -> h[256] -> decoder -> y_base`。
- 新增 `TemporalMarketEncoder`，只读取 `market[:, :-1, :]`，即 canonical
  `T=20` 下的 `[N,19,63]`，使用与 009 相同的单层 GRU：
  `input_size=63`、`hidden_size=63`、`num_layers=1`、`batch_first=True`、
  `bidirectional=False`、`dropout=0`，输出 `m_t [N,63]`。
- 新增 `Market Adapter = Linear(63,256,bias=False)`，weight 显式 zero-init，
  输出 `delta_w_t [N,256]`；`y_market = sum(delta_w_t * h, dim=-1)`，最终
  `prediction = y_base + y_market`。
- 新模块在原 007 backbone 与 decoder 之后构造，因此相同随机 seed 下原有
  参数初始化逐位保持一致；adapter 初始为零时 011 prediction 与相同
  backbone 参数的 007 prediction 严格等价。
- 默认 `configs/config.yaml` 完整声明并启用该分支，不依赖实验特有 CLI
  override。
- 更新 AlphaMaster 单元测试和 smoke，覆盖 shape、切片隔离、zero-init、
  007 等价、横截面共享、梯度/更新、checkpoint 与标准输出接口。

## Constraints

- 唯一实验变量：在 007 prediction side 增加 previous-19-day market → GRU
  → continuous market state → zero-initialized linear Market Adapter → decoder
  residual 分支。
- 原 current-market Feature Gate 完整保留，仍且仅使用
  `market[:, -1, :]`；历史分支仅使用 `market[:, :-1, :]`。
- Feature Gate、stock linear projection、Positional Encoding、TAttention、
  SAttention、TemporalAttention 与原 decoder 的结构和配置均不变。
- 同一 trading day 的完整股票横截面由 canonical sampler 提供相同 market
  window，因此共享相同的 `m_t` 与 `delta_w_t`。
- canonical schema 保持 `158 stock + 13 prior + 63 market + 10 future`
  `returns = 244`，`T=20`；prior13 继续与 007 一样完全不进入模型。
- `d_feat=158`、`d_model=256`、`t_nhead=4`、`s_nhead=2`、temporal/spatial
  dropout 0.5、CSI300 beta 10、SP500 beta 5、`target_day=5`、Adam
  `lr=8e-6` 均与 007 一致。
- 数据划分、70 epoch 预算、early stopping patience 15、Stage 1 seed 42、
  Stage 2 seed 0、指标和 Top30/Drop5 回测协议均不变；Phase 1 未启动正式训练。
- Stage 1 provenance 为 `self`：011 新增可训练 GRU 与 Market Adapter，需按
  AlphaMaster pipeline 重新训练，007 checkpoint 不可 strict 加载到 011。

## Git

Base: exp/007-alphamaster-baseline
Branch: exp/011-alphamaster-continuous-market-adapter
Commit: 0b318c258f446429bab6092950cd6355e0371cf1
Stage 1 provenance: self

## Smoke Test

Status: PASS

Notes: conda `prism-vq` 下完整单元测试 91/91 PASS；其中机制测试验证
canonical `[N,20,244]`、stock `[N,20,158]`、market `[N,20,63]`、history
`[N,19,63]`、`m_t [N,63]`、`h [N,256]`、`delta_w_t [N,256]` 和 prediction
`[N]`。同 seed 构造的 011 与 007/standalone AlphaMaster backbone 参数逐位
一致，zero-init 下 prediction 以 `rtol=0, atol=0` 验证完全等价。

双路径 hook 验证 Feature Gate 输入逐位等于 `market[:, -1, :]`，GRU 输入
逐位等于 `market[:, :-1, :]`。adapter 非零后只改 history 会改变
`delta_w_t`/prediction 而不改 gate 输入；只改 current day 会改变 gate 输入
和 prediction，而 history/`m_t`/`delta_w_t` 逐位不变。same-day canonical
cross-section 的 market state 与 dynamic weight 共享验证 PASS。adapter 在
zero-init 首次 backward 获得非零梯度并更新；更新后 prediction loss 对 GRU
产生非零梯度。

`scripts/smoke_alphamaster.py` 对 CSI300/SP500 tiny canonical PKL 均完成上述
forward/backward 检查，zero-init 007 等价最大绝对误差均为 0.0。Stage 1
限制为 1 epoch、2 train batches、2 validation batches，生成
`artifacts/011/smoke/checkpoints/alphamaster_smoke-epoch=0-val_loss=0.4739.ckpt`；
Stage 2 strict load PASS，生成标准 40 行 prediction
`artifacts/011/smoke/res/alphamaster_csi300/0_best.pkl` 与 `0_metric.csv`，并被
现有 `backtest_qlib.py` normalizer 接受。报告和日志位于
`artifacts/011/smoke/smoke_report.json`、`stage1.log`、`stage2.log`。

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

