# Z1 Utilization Diagnosis — Experiments 001–006

## Scope

- 分析范围：001 / 003 / 004 / 005 / 006——即 Residual HVQ 两级表示中
  第二级残差量化表示 z1 的利用方式研究线。
- 002（prior-gated-fusion）属于 prior factor fusion 研究线，不属于本条
  z1 research line，因此不作为主体分析（其受控结论见
  `experiments/CORRECTED_COMPARISON.md`）。
- 诊断日期：2026-09-06。
- 所有专项诊断均为只读（read-only diagnosis）：不重新训练、不重新回测、
  不修改任何冻结实验；数值来自正式 records、checkpoint、metric CSV、
  prediction pkl 与 wandb 离线记录，各实验专项诊断的完整数值见对应
  record 的 `## Post-hoc Diagnosis` 章节。
- 统一语境：CSI300，Stage 2 seed 0，test 2023-01-01 – 2025-12-31，
  回测 Top30/Drop5（open 0.0005 / close 0.0015，min_cost 0，close 成交）；
  003/004/005/006 均复用 001 的 exact Stage 1 checkpoint
  （`stage1_source: "001"`），Stage 2 层面的比较是严格受控的。

## Experiment Chain

| ID | 融合 / 使用方式 | 相对关系 |
| -- | ------------- | -------- |
| 001 | `z = z0 + z1`（固定两级融合） | 研究线基准 |
| 003 | `z = z0`（z0-only 消融） | 等价于下游 α=0 对照 |
| 004 | `z = z0 + α·z1`（全局可学习标量，α=sigmoid(a)） | 在 001 与 003 之间学习全局工作点 |
| 005 | `z_i = z0_i + g_i·z1_i`（sample-wise gate，`g_i = sigmoid(Linear([z0_i; z1_i]))`） | 004 的 sample-wise 推广 |
| 006 | `ŷ = ŷ0 + Δŷ(z1)`（prediction-level residual branch，主路径同 003） | prediction-level 职责分离 |

## Unified Results

数值取自各实验 record 的正式 Result（corrected protocol，seed 0）。

| Experiment | IC | ICIR | RankIC | RankICIR | Annual Return | Excess | Sharpe | Sortino | MDD | Calmar | Turnover |
| ---------- | -: | ---: | -----: | -------: | ------------: | -----: | -----: | ------: | --: | -----: | -------: |
| 001 (z0+z1) | 0.0352 | 0.2174 | 0.0506 | 0.3171 | 13.69% | 7.29% | 0.7591 | 1.1405 | -18.49% | 0.7401 | 0.3293 |
| 003 (z0 only) | 0.0333 | 0.1810 | 0.0533 | 0.2830 | 11.23% | 4.83% | 0.7651 | 1.1160 | -12.64% | 0.8886 | 0.3275 |
| 004 (z0+α·z1) | 0.0351 | 0.2163 | 0.0506 | 0.3166 | 12.15% | 5.75% | 0.6778 | 1.0312 | -19.33% | 0.6286 | 0.3297 |
| 005 (z0+g_i·z1) | 0.0329 | 0.1908 | 0.0488 | 0.2794 | 16.11% | 9.71% | 1.0237 | 1.5858 | -13.20% | 1.2208 | 0.3274 |
| 006 (ŷ0+Δŷ(z1)) | 0.0321 | 0.1720 | 0.0517 | 0.2680 | 14.04% | 7.64% | 0.9454 | 1.3863 | -11.64% | 1.2059 | 0.3283 |

benchmark = SH000300（基准 AR 6.40%）；Excess = 组合 AR − 基准 AR。

## Diagnostic Findings

### 004 — global learnable scalar α

α_init = 0.952574；best（epoch 5）α = 0.945776；final α = 0.941589；
wandb 轨迹总体单调下降，全程仅移动约 0.01。global scalar 基本维持接近 1，
004 的 IC / RankIC 与 001 几乎相同，未改善 ranking / portfolio。
当前实验不支持"global learnable scalar 可以有效改善 z1 利用方式"
（注意：训练目标不直接优化 test 指标，sigmoid 初始化也限制大幅移动，
因此不能写成"global scalar 已被证明学不会最优权重"）。

### 005 — sample-wise gate g_i

best checkpoint（epoch 4）在完整 validation 集（145145 样本）：
mean = 0.916905，std = 0.042620，min = 0.721620，max = 0.987289；
训练末轮 std 增至 0.0903。gate 确实产生 sample-wise heterogeneity
（未退化为常数），但相关性结构显示 gate 几乎不响应 quantization
error / reconstruction residual（相关系数 ≈0.02），最强关联是与 `||z0||`
负相关（Pearson -0.345）及与 z1 相对幅度正相关（+0.205）——主要呈现
representation magnitude rebalancing 特征。组合指标非常强（AR 16.11%、
Sharpe 1.0237、Calmar 1.2208，为本研究线最优），但 ranking 无增益
（RankIC 0.0488 为研究线最低）。

### 006 — prediction residual branch

prediction residual branch 在 test 上失败：Δŷ 与真实 residual
`y - ŷ0` 负相关（Pearson -0.0725，Spearman -0.0686），Δŷ 对 label 的
逐日 RankIC 均值约 -0.0009；final ranking 低于 main（IC -0.0018，
RankIC -0.0022，top60/bottom60 local RankIC 均变差）。组合层面虽优于
003（AR 14.04% vs 11.23%，Sharpe 0.9454 vs 0.7651），但不能视为
residual prediction 成功——这是单 seed 下尚未解释的现象。

## Hypothesis Assessment

| 假设 | 判定 | 依据 |
| ---- | ---- | ---- |
| A. z1 对全截面 ranking 基本没有稳定增量预测价值 | **Strong evidence**（限定在当前 Stage 1 / seed 0 / ranking 语境） | 四条独立机制路径方向一致：003 去掉 z1 后 RankIC 反而最高（0.0533）；004 的 α 未离开 1；005 的 gate 未带来 RankIC 增益；006 的 Δŷ 与真实残差负相关 |
| B. z1 有价值，但 fixed fusion 方式错误 | **Weak evidence / 证据不足** | global α、sample gate、residual branch 三种替代融合均未在 ranking 上超过 z0-only |
| C. z1 价值具有 sample heterogeneity | **Weak evidence** | 005 学出真实且增长的 gate 离散度（std 0→0.043→0.090），但 gate 的相关结构指向 magnitude rebalancing 而非"识别 z1 信息量" |
| D. z1 适合作为 prediction residual correction | **Not supported（seed 0）** | 006：Corr_delta_resid = -0.0725，final 全面低于 main |
| E. 当前 portfolio-level 结论受 single-seed 噪声影响明显 | **Strong evidence** | 所有组合层面优势（005/006 的 Sharpe/MDD/AR）均建立在单一 Stage 2 seed 0 上；006 出现"ranking 全面变差而组合变好"的矛盾，直接暴露组合指标对 seed 的敏感性 |

## Current Research Interpretation

1. 当前 strongest finding **不是**"z1 提升了 ranking"；恰恰相反，在
   当前 Stage 1 / seed 0 / 全截面 ranking 语境下，没有任何一种 z1
   利用方式超过 z0-only（003）。
2. ranking 维度中，z1 的固定（001）、动态全局（004）、动态
   sample-wise（005）、prediction-residual（006）使用均未带来增益。
3. 005 的组合层面表现（AR 16.11%、Sharpe 1.0237、MDD -13.20%）是目前
   最值得保留的异常正信号——但它与 ranking 指标背离，且只有单 seed，
   只能视为待复核的 research signal。
4. 005 的 gate mechanism 更像 magnitude rebalancing（gate 与 `||z0||`
   负相关、与 z1 相对幅度正相关），而不是 reconstruction-error-aware
   selection——gate 尚不响应量化误差。
5. 006 是重要负结果：z1 作为 prediction residual correction 在 seed 0
   下方向性失败（负相关），应防止未来重复尝试同类 residual correction
   机制。

## Next Research Decision

（仅记录当前建议，不创建实验；新实验须走 Phase 1 流程。）

- **Priority 1**：对 001 / 003 / 005 做 Stage 2 multi-seed confirmation，
  优先使用 seed 0/1/2/3/4，Stage 1 继续复用 001 exact checkpoint。
  目的：把组合层面优势与 seed 噪声分离（假设 E 的直接检验）。
- **Priority 2**：若 005 的 portfolio advantage 在多 seed 下稳定，再做
  gate mechanism decomposition，例如 norm-only gate（检验 005 的收益
  是否完全由 magnitude rebalancing 解释）。
- **Priority 3**：只有在 005 多 seed 信号稳定后，才考虑 market / regime
  conditioned gating。

明确：当前不建议基于 seed 0 直接扩展出大量 005 派生实验。
