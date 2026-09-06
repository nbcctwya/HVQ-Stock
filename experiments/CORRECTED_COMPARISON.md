# Corrected Protocol 统一比较（seed 0，CSI300，test 2023-01-01 – 2025-12-31）

四个模型使用完全统一的 corrected 协议重新评估：corrected Stage 2 freeze
（encoder/quantizer/RevIN 全程 eval）+ corrected MDD（初始 NAV=1 计入 peak）；
Stage 2 seed 0；回测 Top30/Drop5，open 0.0005 / close 0.0015，min_cost 0，
close 成交，CN limit=None。旧结果（pre-fix）已在各 record 中标记为
Historical，不再作为比较依据。

| Model | IC | ICIR | RankIC | RankICIR | Annual Return | Excess | Sharpe | Sortino | MDD | Calmar | Turnover |
| ----- | -: | ---: | -----: | -------: | ------------: | -----: | -----: | ------: | --: | -----: | -------: |
| PRISM-VQ baseline | 0.0373 | 0.2214 | 0.0552 | 0.3313 | 9.24% | 2.84% | 0.5223 | 0.7690 | -26.90% | 0.3436 | 0.3295 |
| 001 hvq-residual-2level (z0+z1) | 0.0352 | 0.2174 | 0.0506 | 0.3171 | 13.69% | 7.29% | 0.7591 | 1.1405 | -18.49% | 0.7401 | 0.3293 |
| 002 prior-gated-fusion | 0.0370 | 0.2172 | 0.0552 | 0.3309 | 8.03% | 1.63% | 0.4577 | 0.6687 | -26.52% | 0.3028 | 0.3289 |
| 003 hvq-z0-only (z0) | 0.0333 | 0.1810 | 0.0533 | 0.2830 | 11.23% | 4.83% | 0.7651 | 1.1160 | -12.64% | 0.8886 | 0.3275 |

Excess = 组合 AR 与基准 AR（6.40%）的差值。benchmark = SH000300。

## Stage 1 架构与 checkpoint provenance

- baseline：single VQ512 Stage 1
- 002：single VQ512 Stage 1，架构与配置与 baseline 完全相同
- 001：Residual HVQ 256+256，Stage 1 架构本身就是该实验的实验变量
- 003：与 001 相同的 Residual HVQ Stage 1

| Model | Stage 1 checkpoint | 来源 |
| ----- | ------------------ | ---- |
| baseline | `infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt` | PRISM-VQ 原始 Stage 1（epoch=9 ckpt 已缺失，见 baseline_results/CORRECTED_PROTOCOL.md） |
| 001 | `hvq_csi300_full-epoch=5-val_loss=0.4592.ckpt` | self-trained（seed 42），`artifacts/001/run/checkpoints/` |
| 002 | 同 baseline（exact same file） | 共享 corrected baseline 的 Stage 1 checkpoint，构成严格受控对照 |
| 003 | 同 001（exact same file） | `stage1_source: "001"` 显式复用 |

注：baseline 与 001 的 Stage 1 架构不同（single VQ512 vs Residual HVQ
256+256），该差异属于 001 的实验设计本身，不是混淆因素；但两者的
Stage 1 是各自独立训练的 checkpoint，跨架构比较时数值还包含实现与
训练实例差异。baseline vs 002 与 001 vs 003 均为共享 exact Stage 1
checkpoint 的严格受控比较。

## 结论

### baseline vs 002（共享 exact Stage 1，严格受控的 fusion 对照）

gated fusion 与 fixed fusion 排序能力几乎相同（IC 0.0370 vs 0.0373，
RankIC 0.0552 vs 0.0552，RankICIR 0.3309 vs 0.3313），组合层面略弱
（AR 8.03% vs 9.24%，Sharpe 0.4577 vs 0.5223，MDD -26.52% vs -26.90%）。
可学习 per-sample gate 未带来实质收益。此前观察到的 002 在 Sharpe/MDD
上的优势来自不同的 Stage 1 训练实例，受控后消失。

### baseline vs 001

ranking 指标 baseline 略高（IC 0.0373 vs 0.0352，RankIC 0.0552 vs 0.0506），
组合层面 001 明显更优（AR 13.69% vs 9.24%，Sharpe 0.7591 vs 0.5223，
MDD -18.49% vs -26.90%，Calmar 0.7401 vs 0.3436）。
Residual HVQ 在 corrected protocol 下组合表现优于单层 baseline；
两者 Stage 1 架构不同属于 001 的实验设计本身。

### 001 vs 003（共享 exact Stage 1，严格受控的 z0+z1 vs z0-only 消融）

003（z0-only）RankIC 更高（0.0533 vs 0.0506）、Sharpe 持平略高
（0.7651 vs 0.7591）、MDD 明显更好（-12.64% vs -18.49%），
IC 略低（0.0333 vs 0.0352）、AR 略低（11.23% vs 13.69%）。
结论：第二级残差量化 z1 对下游收益预测没有稳定贡献，
"z1 主要携带 reconstruction/detail 信息、对收益预测帮助有限"的信号
在 corrected protocol 下仍然成立。

## 产物位置

- baseline：`artifacts/baseline/run/`
- 001：`artifacts/001/run/`
- 002：`artifacts/002/run/`（controlled result；旧结果备份在
  `uncontrolled_stage1/`（corrected 但 Stage 1 实例不同）与
  `pre_fix/`（修复前））
- 003：`artifacts/003/run/`（pre-fix 备份在 `pre_fix/`）

## Later experiments: 004–006

（2026-09-06 追加；同一 corrected 协议、同一 Stage 2 seed 0、同一回测
协议，003/004/005/006 均复用 001 的 exact Stage 1 checkpoint。）

| Model | IC | ICIR | RankIC | RankICIR | Annual Return | Excess | Sharpe | Sortino | MDD | Calmar | Turnover |
| ----- | -: | ---: | -----: | -------: | ------------: | -----: | -----: | ------: | --: | -----: | -------: |
| 004 hvq-learnable-z1-scale (z0+α·z1) | 0.0351 | 0.2163 | 0.0506 | 0.3166 | 12.15% | 5.75% | 0.6778 | 1.0312 | -19.33% | 0.6286 | 0.3297 |
| 005 hvq-samplewise-z1-gate (z0+g_i·z1) | 0.0329 | 0.1908 | 0.0488 | 0.2794 | 16.11% | 9.71% | 1.0237 | 1.5858 | -13.20% | 1.2208 | 0.3274 |
| 006 hvq-predictive-residual-z1 (ŷ0+Δŷ) | 0.0321 | 0.1720 | 0.0517 | 0.2680 | 14.04% | 7.64% | 0.9454 | 1.3863 | -11.64% | 1.2059 | 0.3283 |

004–006 是 z1 利用方式研究线（z1 research line）的后续实验：004 全局
可学习 α 维持在 ≈0.95（行为≈001）；005 sample-wise gate 组合指标为本
研究线最优但 RankIC 无增益；006 prediction-residual branch 在 test 上
失败（Δŷ 与真实 residual 负相关）。各实验的专项只读诊断见对应 record
的 `## Post-hoc Diagnosis` 章节；跨实验假设评估与后续研究决策见
`experiments/analysis/z1-utilization-001-006.md`。

产物位置：`artifacts/004/run/`、`artifacts/005/run/`、
`artifacts/006/run/`。
