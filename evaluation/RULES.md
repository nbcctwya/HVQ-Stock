# Baseline Results Protocol v1.0 — HVQ-Stock 正式评估协议

本文件是 HVQ-Stock 论文级正式评估的唯一权威规则。`evaluation/` 的实现必须与本文件一致；
如需变更，先修改本文件并同步实现与 validation。所有正式结果写入仓库根目录 `results/`。

本协议独立于科研流水线（Phase1 / Phase2 / Phase3 / 根目录 `backtest_qlib.py`）。
科研流水线继续用于实验期间快速反馈；`results/` 中的正式数字必须由 `evaluation/`
按本协议从 prediction signal 重新计算、重新回测，禁止复制任何旧回测的组合指标。

## 1. 输入：正式 artifact

- 只使用已经完成训练与 prediction 的正式 artifact；禁止重新训练。
- Seed 0：Phase2 正式结果 `artifacts/<experiment>/run/res/<run_name>/0_best.pkl`。
- Seed ≥1：仅使用 `artifacts/_phase3/batches/<batch>/receipts/` 中 `accepted == true`
  的 receipt 指向的 archive prediction。
- 禁止使用 smoke / diagnostic / fixture / incoming / staging / failed / 未 acceptance 的 artifact。
- 同一 (experiment, seed) 存在多个 accepted receipt 时视为冲突，必须明确解决，不得静默选择。
- 正式 prediction 缺失时必须明确失败并让 validation 失败；不得伪造、不得静默跳过。

## 2. results schema

```text
results/
├── metrics/
│   ├── seed_metrics.csv          # 每个 (market, model, seed) 恰好一行
│   ├── aggregate_metrics.csv     # 每个 (market, model) 恰好一行
│   └── ensemble_metrics.csv      # 每个 (market, model, ensemble_method) 恰好一行
├── tables/
│   ├── seed_mean_std.csv         # "mean ± std"，统一四位小数
│   └── ensemble.csv              # 四位小数展示版
├── curves/ensemble/<market>_<model>.csv
├── metadata/
│   ├── eval_config.json          # 实际运行口径
│   └── manifest.json             # 只登记实际存在的文件
└── diagnostics/validation.json
```

- `metrics/` 是机器可计算原始数值，禁止 `"mean ± std"` 字符串，禁止百分号。
- `tables/` 只能由数值 metrics 派生；禁止从 table 字符串反向计算。
- 所有实验共用同一个 `results/` 根，通过 `model` 字段区分。

### seed_metrics.csv

列顺序固定：

```text
market,model,seed,IC,ICIR,RankIC,RankICIR,AR,STD,MDD,Sharpe,Sortino,Calmar,num_test_days,pred_path_or_ckpt_path
```

`pred_path_or_ckpt_path` 必须是相对仓库根目录路径或可移植 URI，禁止本机绝对路径。

### aggregate_metrics.csv

```text
market,model,IC_mean,IC_std,ICIR_mean,ICIR_std,RankIC_mean,RankIC_std,
RankICIR_mean,RankICIR_std,AR_mean,AR_std,STD_mean,STD_std,MDD_mean,MDD_std,
Sharpe_mean,Sharpe_std,Sortino_mean,Sortino_std,Calmar_mean,Calmar_std
```

mean/std 跨 seed，std 使用 `ddof=1`。

### ensemble_metrics.csv

```text
market,model,ensemble_method,IC,ICIR,RankIC,RankICIR,AR,STD,MDD,Sharpe,Sortino,Calmar,num_test_days,seeds,pred_paths
```

### curves/ensemble/<market>_<model>.csv

```text
datetime,daily_ret_gross,cost,daily_ret_net,bench_ret,nav,bench_nav
```

- `daily_ret_net = daily_ret_gross - cost`（已确认 Qlib report 的 return 为 gross）。
- `nav = cumprod(1 + daily_ret_net)`，`bench_nav = cumprod(1 + bench_ret)`；
  禁止 `nav /= nav.iloc[0]`（会丢掉第一天收益）。
- 日期升序、唯一、无 NaN、无 Inf。

## 3. 统一 Qlib 回测

策略：`qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy`，以下参数必须显式传入，
不得依赖 Qlib 默认值：

```text
topk=30, n_drop=5, method_sell="bottom", method_buy="top",
hold_thresh=1, only_tradable=False, forbid_all_trade_at_limit=True,
risk_degree=0.95, freq=day
```

持仓权重使用该配置下 Qlib 原生资金分配与整手处理；evaluation 层不得二次改权重。

交易成本（固定）：`open_cost=0.0005, close_cost=0.0015, min_cost=0`。

净收益：`daily_ret_net = report["return"] - report["cost"]`。
已核实 qlib 0.9.7 `qlib/backtest/account.py` 中
`return_rate = (earning + cost) / last_account_value`，即 `report["return"]` 是**未扣费**收益，
`report["cost"]` 是当日成本率；因此净收益恰好扣一次成本。该判断依据必须记录在
`eval_config.json`。若未来 Qlib 版本改变 return 语义，必须先重新核实再改实现。

Account 固定：`account = 100000000`，long-only，no leverage，所有 market/model/seed/ensemble 一致。

回测区间严格等于正式 test split（来自 `configs/config.yaml` 的 `data.test_period`，
当前为 2023-01-01 → 2025-12-31）。不得使用 train/valid，不得扩大 test；
prediction 覆盖不足时 evaluation 失败，不得静默缩短。`num_test_days` 为实际 Qlib 交易日数。

市场事实（从项目配置读取，不得猜）：

```text
provider_uri = ~/.qlib/qlib_data/cn_data   (dataset/2025_csi300.yaml)
region = cn
instruments = csi300  (Qlib 动态历史成分股；不得自行重建或冻结)
benchmark = SH000300
deal_price = close
limit_threshold = None → 解析为 Qlib CN region 默认 C.limit_threshold = 0.095
trade_unit = 100 (Qlib CN region)
executor = SimulatorExecutor(time_per_step=day, generate_portfolio_metrics=True)
```

## 4. Signal 与日期对齐

- 保持 prediction 原本的日期含义与 label horizon（当前 `predictor.target_day=5`，
  label = `Ref($close,-5)/Ref($close,-1)-1`）。
- signal 转换为 `(datetime, instrument)` 索引。
- Qlib TopkDropoutStrategy：trade_date = t 使用 signal_date = t-1（Qlib 内部 shift = 1）。
- adapter 不得再手工 shift prediction（否则重复滞后），不得移动 prediction 日期，
  不得改变 label horizon。
- `eval_config.json` 必须记录 signal_date / trade_date / Qlib shift / label horizon。

## 5. 预测指标（每日横截面）

```text
IC_t     = Pearson(prediction, label)
RankIC_t = Spearman(prediction, label)
IC       = mean(IC_t)
ICIR     = mean(IC_t) / std(IC_t, ddof=1)     # 不乘 sqrt(252)
RankIC   = mean(RankIC_t)
RankICIR = mean(RankIC_t) / std(RankIC_t, ddof=1)  # 不年化
```

无法计算相关性的日期跳过。必须按本规则从正式 prediction/label 重新计算，
不得复制旧 summary 指标。

## 6. 组合指标

输入为恰好扣一次成本后的每日简单收益 `r_net_t = daily_ret_gross_t - cost_t`。
若 `r_net_t <= -1` 直接报错。

```text
g_t = log(1 + r_net_t)
A = 252,  rf = 0,  MAR_daily = 0

AR       = exp(mean(g_t) * 252) - 1
STD      = std(g_t, ddof=1) * sqrt(252)
NAV      = [1.0, exp(cumsum(g_t))]
MDD      = min(NAV / cumulative_max(NAV) - 1)
Sharpe   = sqrt(252) * mean(g_t) / std(g_t, ddof=1)
DownDev  = sqrt(mean(min(g_t - MAR_daily, 0)^2))   # mean 覆盖全部交易日
Sortino  = sqrt(252) * mean(g_t - MAR_daily) / DownDev
Calmar   = AR / abs(MDD)
num_test_days = 有效 g_t 数量
```

零分母、样本不足等数学未定义情况使用 NaN，禁止用 0 冒充；validation 必须显式记录。
所有模型使用完全相同的 252 / log-return / ddof=1 / rf=0 / MAR=0。
不得把 benchmark information ratio 当作 Sharpe。

## 7. Seed 聚合

从 `seed_metrics.csv` 按 `(market, model)` 聚合，mean/std（ddof=1）跨 seed，
写入 `aggregate_metrics.csv`；`tables/seed_mean_std.csv` 为四位小数
`"mean ± std"` 展示版。

## 8. Ensemble

- 默认 `ensemble_method = avg_none`。
- 不同 seed 的 prediction 按 `(datetime, instrument)` inner join 对齐，
  `ensemble_score = mean(raw seed scores)`。
- 预留 `avg_zscore` / `avg_rank` 扩展，但必须先正确实现 `avg_none`。
- Ensemble 必须从 prediction 层完成：对 ensemble score 重新计算
  IC/ICIR/RankIC/RankICIR，并重新执行完整统一 Qlib 回测得到组合指标。
- 禁止平均单 Seed IC / Sharpe / 收益曲线 / 组合指标来冒充 ensemble。

## 9. Validation

`python -m evaluation.validate`（或 evaluation run 末尾自动执行）：

- 全部 PASS → 退出码 0；存在 failure → 非 0。
- 输出 `results/diagnostics/validation.json`（passed / passes / failures / checks）。
- 必检项见 `evaluation/validate.py`：manifest 文件存在性、(market, model, seed) 覆盖与唯一性、
  非法 NaN/Inf、指标值域、aggregate 精确等于 mean/std、tables 与 metrics 一致、
  ensemble 行数、curve 日期与 net/nav/bench_nav 关系、由 curve 独立反算全部组合指标、
  ensemble ranking metrics 来自 ensemble score、eval_config 显式协议参数、
  回测区间等于正式 test split、prediction 覆盖完整性、signal 时序无二次 shift。
