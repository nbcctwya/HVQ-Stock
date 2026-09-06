# 006 — hvq-predictive-residual-z1

## Idea

在实验 003（hvq-z0-only）的 z0-only Stage 2 预测路径之外，不再将第二级
Residual HVQ 表示 `z1` 与 `z0` 直接融合，而是将 `z1` 定义为一个专门用于修正
z0 预测误差的 prediction-residual branch：

- `ŷ0 = F(z0)`（003 主预测路径完全不变）；
- `r = y - stopgrad(ŷ0)`（stop-gradient 的预测残差目标）；
- `Δŷ = G(z1)`（独立、轻量的 residual prediction head）；
- `ŷ = ŷ0 + Δŷ`（最终正式预测）。

## Motivation

Residual HVQ 的第二级 `z1` 在表示空间中学习的是一级量化后的 reconstruction
residual，但这并不天然等价于未来收益预测所需的增量信息：001 直接融合
`z0 + z1`，而 003 去掉 `z1` 后部分预测/组合指标（RankIC、Sharpe、MDD）反而
更好。006 明确赋予两级表示不同职责——`z0` 学习主要、稳定的收益预测结构，
`z1` 仅学习 z0 主预测器尚未解释的 prediction residual——验证这种
"representation residual → prediction residual" 的职责对齐能否使 `z1`
提供稳定的增量预测价值。

受控对照关系：

- 003：仅 z0 → `ŷ0`；
- 001：representation-level `z0 + z1`；
- 006：prediction-level `ŷ0 + Δŷ(z1)`。

## Modification

- `module/quantise_hvq.py::ResidualVectorQuantiser` 新增
  `forward_two_levels(h_batch)`：分别返回两级量化输出 `(z_q0, z_q1)`，
  逐级 STE 语义与 `forward` 完全一致（`z_q0 + z_q1` 数值上等于默认
  `forward` 的 `z0+z1`）；默认 `forward` 行为不变，Stage 1 不受影响。
- `trainer/train_ypred.py::GenerateReturn`：
  - 新增 `predictor.z1_residual_branch` 开关（默认 False 即 003 行为；
    True 时要求 hvq 量化器且 `z0_only=True`）；
  - residual head：`nn.Linear(vq_embed_dim, 1)`（128→1，129 参数），
    输入为 detach 后的 `z_q1`；无 MLP / attention / MoE / market feature /
    gate；
  - `forward(..., return_components=True)` 返回 `y0` / `delta_y` / `z_q1`
    等诊断组件；默认 5 元组返回的第一个元素即最终 `ŷ = ŷ0 + Δŷ`；
  - 损失（固定、无可调权重）：
    `loss = RankLoss(ŷ0, y) + RankLoss(Δŷ, y - stopgrad(ŷ0)) + aux_weight * aux_loss`，
    复用 Stage 2 已有 `RankLoss`（`rank: 0`）；residual target 经 detach，
    residual branch 梯度不回传到 ŷ0；
  - validation 额外记录 `Val_IC_main` / `Val_RIC_main` 与 `train/val_res_loss`。
- `utils/test.py::run_inference`：启用 `z1_residual_branch` 时预测 DataFrame
  增加 `score_main`（ŷ0）、`delta`（Δŷ）两列；指标额外报告
  `IC_main` / `ICIR_main` / `RankIC_main` / `RankICIR_main`、
  `Delta_mean` / `Delta_std` / `Delta_abs_mean`、
  `Corr_delta_resid` / `RankCorr_delta_resid`（Δŷ 与真实残差 `y - ŷ0` 的
  Pearson / Spearman 相关）。正式指标与 `score` 列始终为最终 `ŷ = ŷ0 + Δŷ`。
- `configs/config.yaml`：`predictor.z1_residual_branch: true`（默认配置即
  006 本身，无实验特有 CLI override）；`predictor.z0_only: true` 保持；
  `train.seed: 0` 不变。
- 新增 `tests/test_predictive_residual_z1.py`（13 个用例）。
- 分支根目录 `README.md` 改写为 006 实验说明。

## Constraints

- 唯一实验变量：在 003 的 z0-only 预测路径之外增加由 z1 驱动的
  prediction-residual correction branch（`ŷ = ŷ0` → `ŷ = ŷ0 + Δŷ`）；
  除此之外均与 003 一致。
- 003 原有 z0 主预测路径 `F(z0)`（loadings / MoE / latent head /
  return_predictor）结构、输入、超参数及行为完全不变；未修改主 predictor、
  MoE、attention、factor fusion。
- z0 与 z1 的生成方式完全不变；Stage 1 不修改、不重新训练
  （`forward_two_levels` 为只读接口，量化器默认 `forward` 与 `stage1.py`
  均未改动）。
- 禁止 representation-level `z0+z1` 融合：z0 与 z1 在 Stage 2 中职责分离
  （单测实证主路径输入严格为 z0）。
- residual target 严格为 `y - stopgrad(ŷ0)`，residual branch 梯度不经
  target 回传到 ŷ0（单测实证）。
- residual head 为最小线性 head（129 参数），无复杂结构、无 gate。
- 复用现有 `RankLoss`，未新增 loss family；损失组合为固定权重
  （1 + 1 + 已有 aux_weight），未引入新的可调 loss-weight 超参数，
  无实验特有 CLI override。
- 数据划分（train 2009–2020 / valid 2021–2022 / test 2023–2025，CSI300）、
  训练预算（max 70 epoch、patience 15）、seed 协议（Stage 2 seed 0）、
  回测协议（Top30/Drop5，open 0.0005 / close 0.0015，min_cost 0，
  close 成交，CN limit=None）均不变。

## Git

Base: exp/003-hvq-z0-only
Branch: exp/006-hvq-predictive-residual-z1
Commit: b2af82df73f63d9387681876017d5fd9e7e678f7
Stage 1 provenance: 复用实验 001 的 exact Stage 1（queue `stage1_source: "001"`，
与 003 同源）：`artifacts/001/run/checkpoints/hvq_csi300_full-epoch=5-val_loss=0.4592.ckpt`；
两级量化器配置（num_levels=2、level_num_embed=[256,256]）、encoder 与 RevIN
结构兼容，strict 加载 missing=0 / unexpected=0。

## Smoke Test

Status: PASS
Notes: 单元测试 13/13（`tests.test_predictive_residual_z1`）+ 回归 20/20
（`tests.test_z0_only`、`tests.test_hvq`、`tests.test_protocol_metrics`）通过
（conda `prism-vq`）。覆盖：forward_two_levels 契约（z_q0+z_q1 == forward）、
006 主路径 ŷ0 与 003 z0-only 输出逐位一致、无 representation-level 融合、
residual target 定义与 stop-gradient、Δŷ 由 z1 head 生成、ŷ == ŷ0 + Δŷ、
residual head 梯度与参数更新、Stage 1 冻结、001 Stage 1 strict load、
checkpoint 保存/恢复、默认 config 即 006。
Stage 2 smoke：`stage2.py train.num_epochs=1 train.seed=0
artifact_root=artifacts/006/smoke
predictor.saved_model="hvq_csi300_full-epoch=5-val_loss=0.4592.ckpt"`
（Stage 1 直接复用 001 正式 checkpoint，置于 smoke artifact 目录），
1 epoch train / valid / test 全流程完成；诊断输出齐全（smoke Test RankIC
0.0350、主路径 RankIC_main 0.0374、Delta_std 0.0191，仅供流程验证）。
另以临时脚本（`artifacts/006/smoke/verify_residual_smoke.py`）逐项验证：
ŷ0/Δŷ/ŷ 形状、ŷ == ŷ0 + Δŷ（max diff 2e-8）、stop-gradient、residual head
非零梯度与参数更新、Stage 1 无梯度、smoke checkpoint strict 恢复
（含 residual_head）、prediction pkl 含 score/score_main/delta 且
score == score_main + delta、metric 含主路径与 Δŷ 诊断指标——全部 PASS。
全部产物落在 `artifacts/006/smoke/`，公共 `checkpoints/` 与 `res/` 无写入。

## Result

Status: DONE（test 区间 2023-01-01 – 2025-12-31）

IC: 0.0321
ICIR: 0.1720
RankIC: 0.0517
RankICIR: 0.2680

Annual Return: 14.04%（基准 6.40%，超额 7.64%）
Sharpe: 0.9454
Sortino: 1.3863
MDD: -11.64%
Calmar: 1.2059
Turnover: 0.3283

## Conclusion

Phase 2 固定执行器完成正式训练、预测与回测（pinned commit b2af82df73f63d9387681876017d5fd9e7e678f7）。Stage 1 复用实验 001 的正式 checkpoint：`/home/nbcctwya/baselines/masterVQ/HVQ-Stock/artifacts/001/run/checkpoints/hvq_csi300_full-epoch=5-val_loss=0.4592.ckpt`（本实验未重新训练 Stage 1）；Stage 2 seed 0。

产物：`artifacts/006/run/`（checkpoints/、res/、stage1.log、stage2.log、backtest.log、summary.json）。

## Post-hoc Diagnosis

Diagnosis date: 2026-09-06
Execution: read-only diagnosis; no retraining / no backtest
（自正式 test prediction `0_best.pkl`（含 `score` / `score_main` /
`delta` / `label`）恢复并复算；`score == score_main + delta` 逐位成立，
全部指标与 `0_metric.csv` 逐位一致。）

- test samples = 217909
- test trading days = 727

### Main vs Final

| Metric   | main `ŷ0` | final `ŷ0+Δŷ` |  Change |
| -------- | --------: | ------------: | ------: |
| IC       |    0.0339 |        0.0321 | -0.0018 |
| ICIR     |    0.1786 |        0.1720 | -0.0066 |
| RankIC   |    0.0539 |        0.0517 | -0.0022 |
| RankICIR |    0.2754 |        0.2680 | -0.0074 |

Delta statistics：

- Delta_mean = 0.00057
- Delta_std = 0.0208
- Delta_abs_mean = 0.0163
- final score std ≈ 0.0989

关键相关性（Δŷ vs 真实 residual `r = y - ŷ0`）：

- Corr_delta_resid = -0.072475
- RankCorr_delta_resid = -0.0686
- daily Pearson mean ≈ -0.0507（仅约 32.5% 交易日 daily Pearson > 0）
- daily Spearman mean ≈ -0.0434（约 34.1% 交易日 daily Spearman > 0）
- delta 对 label 的逐日 RankIC 均值约 -0.0009（纯噪声水平）

其他诊断：

- top60 local RankIC：final 0.0264 vs main 0.0291
- bottom60 local RankIC：final 0.0331 vs main 0.0371
- delta 每天约更换 top30 中 16% 的股票
- 被换入样本 `|delta|` ≈ 0.0286（全体平均 `|delta|` ≈ 0.0163，约 1.75 倍）
- corr(delta, score_main) ≈ +0.081（delta 轻微放大而非压缩主路径极端值）
- corr(score_main_006, score_003) ≈ 0.957（006 主路径 ≈ 003，消融受控成立）

### 结论

- 006 的 prediction-residual hypothesis 在 seed 0 下不被支持；
- Δŷ 与真实 residual `y - ŷ0` 为负相关——residual branch 在 test 上
  不仅没有学到残差，方向反而系统性相反；
- final IC / RankIC 均低于 main（IC -0.0018，RankIC -0.0022）；
- z1 residual branch 没有表现出 z0 之外的增量预测价值；
- 006 的组合表现虽然优于 003（AR 14.04% vs 11.23%，Sharpe 0.9454 vs
  0.7651，MDD -11.64% vs -12.64%），但不能解释为 residual correction
  成功；
- 当前证据不支持"普遍改善排序""关键股票修正""压缩极端值降风险"这些
  机制解释（top60/bottom60 local RankIC 均变差，delta 不压缩极端值，
  top30 换手未提升任何可测排序质量）；
- 更合理的态度是把组合改善视为单 seed 下尚未解释、需要多 seed 验证的
  现象，而不是方法成功。

跨实验综合解读见 `experiments/analysis/z1-utilization-001-006.md`。
