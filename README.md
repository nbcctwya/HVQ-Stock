# exp/006 — hvq-predictive-residual-z1

- Base: `exp/003-hvq-z0-only`
- Branch: `exp/006-hvq-predictive-residual-z1`
- Stage 1 provenance: 复用实验 001 的 exact Stage 1 checkpoint（`stage1_source: "001"`）

## Idea / Motivation

Residual HVQ 的第二级表示 `z1` 在表示空间中学习的是一级量化后的
**reconstruction residual**，但这种 residual 并不天然等价于未来收益预测所需的
增量信息：

- 001 在 Stage 2 直接做 representation-level 融合 `z = z0 + z1`；
- 003 去掉 `z1`（z0-only）后，部分预测/组合指标（RankIC、Sharpe、MDD）反而更好。

这说明直接把 `z1` 加回表示可能无法有效利用第二级信息。006 不再将 `z1` 与
`z0` 做表示层融合，而是把 `z1` 定义为一个专门用于**修正 z0 预测误差**的
prediction-residual branch，明确赋予两级表示不同职责：

- `z0`：学习主要、稳定的收益预测结构（与 003 完全一致）；
- `z1`：仅学习 `z0` 主预测器尚未解释的 prediction residual。

核心假设：**representation residual → prediction residual 的职责对齐**，
能让 `z1` 提供稳定的增量预测价值。

## 与 base（003）的关系

003 的 z0 主预测路径完全保留、逐位不变：

```
ŷ0 = F(z0)
```

006 在其上新增一条由 `z1` 驱动的 correction branch：

```
r   = y - stopgrad(ŷ0)      # stop-gradient 的预测残差目标
Δŷ  = G(z1)                 # 独立、轻量的 residual prediction head
ŷ   = ŷ0 + Δŷ               # 最终正式预测
```

三组受控对照：

- 003：仅 z0 → `ŷ0`；
- 001：representation-level `z0 + z1`；
- 006：prediction-level `ŷ0 + Δŷ(z1)`。

## 核心修改（相对 base 003）

- `module/quantise_hvq.py::ResidualVectorQuantiser` 新增
  `forward_two_levels(h_batch)`：分别返回两级量化输出 `(z_q0, z_q1)`，
  逐级 STE 语义与 `forward` 完全相同（`z_q0 + z_q1` 数值上等于默认
  `forward` 的 `z0+z1`）。默认 `forward` 行为不变，Stage 1 不受影响。
- `trainer/train_ypred.py::GenerateReturn`：
  - 新增 `predictor.z1_residual_branch` 开关（默认 False 即 003 行为；
    为 True 时要求 hvq 量化器且 `z0_only=True`，保证主路径与 003 一致）；
  - 新增 `self.residual_head`（见下）；forward 中 `z_q0` 走与 003 完全相同
    的主路径，`z_q1`（detach 后）只进入 residual head，二者绝无
    representation-level 的 `z0+z1` 融合；
  - `forward(..., return_components=True)` 额外返回 `y0` / `delta_y` /
    `z_q1` 等诊断组件；默认 5 元组返回的第一个元素即最终 `ŷ = ŷ0 + Δŷ`，
    与既有调用方（run_inference 等）兼容；
  - 新增 `_residual_branch_losses(y0, delta_y, label)`，固定损失定义（见下）；
  - validation 额外记录主路径 `Val_IC_main` / `Val_RIC_main` 与
    `train/val_res_loss`。
- `utils/test.py::run_inference`：当模型启用 `z1_residual_branch` 时，预测
  DataFrame 增加 `score_main`（ŷ0）与 `delta`（Δŷ）两列，指标额外报告
  `IC_main` / `ICIR_main` / `RankIC_main` / `RankICIR_main`（z0 主路径
  ŷ0 的 test IC / RankIC）、`Delta_mean` / `Delta_std` / `Delta_abs_mean`
  （Δŷ 基本统计）以及 `Corr_delta_resid` / `RankCorr_delta_resid`
  （Δŷ 与真实残差 `y - ŷ0` 的 Pearson / Spearman 相关）。正式指标
  （`IC` / `RankIC` 等）与保存的 `score` 列始终是最终 `ŷ = ŷ0 + Δŷ`。
- `configs/config.yaml`：`predictor.z1_residual_branch: true`——默认配置即
  006 实验本身，无需实验特有 CLI override；`predictor.z0_only: true` 保持。
- 新增 `tests/test_predictive_residual_z1.py`（13 个用例）。
- 分支根目录 `README.md` 改写为本实验说明。

## Residual head 结构（最小化）

```python
self.residual_head = nn.Linear(vq_embed_dim, 1)   # 128 -> 1
Δŷ = self.residual_head(z_q1).squeeze(-1)         # z_q1 已 detach
```

单层线性、无 bias 之外的任何结构：无 MLP、无 attention、无 MoE、无 market
feature、无 sample/regime gate。参数量 **129**（128 权重 + 1 bias），只用于
检验 z1 能否预测 z0 主路径尚未解释的残差。

## Loss 定义（固定、无可调权重）

复用 Stage 2 已有的 `RankLoss`（当前 `rank: 0`，即 MSE + 0×rank 项）：

```
main_loss  = RankLoss(ŷ0, y)                    # 与 003 的主路径损失完全相同
res_target = y - stopgrad(ŷ0)                   # detach，不回传梯度到 ŷ0
res_loss   = RankLoss(Δŷ, res_target)           # 同一 loss family 应用于残差任务
loss       = main_loss + res_loss + aux_weight * aux_loss
```

主路径损失与 003 逐位一致（`RankLoss(ŷ0, y)`），residual loss 以固定权重 1
加入，`aux_weight` 是 003 已有的公共超参。**未引入任何新的可调 loss-weight
超参数**，也无实验特有 CLI override。

## 与 base 的唯一区别

唯一实验变量：在 003 的 z0-only Stage 2 预测路径之外，增加由 `z1` 驱动的
prediction-residual correction branch，使最终预测从 `ŷ = ŷ0` 变为
`ŷ = ŷ0 + Δŷ`（`Δŷ = G(z1)` 专门学习 `y - stopgrad(ŷ0)`）。

保持不变：z0 主预测路径 `F(z0)`（loadings / MoE / latent head /
return_predictor 结构、输入、超参与训练梯度）、z0 与 z1 的生成方式、
Stage 1（不修改、不重新训练）、数据划分（train 2009–2020 / valid 2021–2022 /
test 2023–2025，CSI300）、训练预算（max 70 epoch、patience 15）、
`train.seed: 0`、回测协议（Top30/Drop5，open 0.0005 / close 0.0015，
min_cost 0，close 成交，CN limit=None）、`aux_weight` 等全部超参。

## Stage 1 provenance

复用实验 001 的 exact Residual HVQ Stage 1 checkpoint（`stage1_source: "001"`，
与 003 同源）：`artifacts/001/run/checkpoints/hvq_csi300_full-epoch=5-val_loss=0.4592.ckpt`。
两级量化器配置（`num_levels: 2`、`level_num_embed: [256, 256]`）、encoder 与
RevIN 结构均与 001 一致；strict 加载验证 missing=0 / unexpected=0
（见 `tests/test_predictive_residual_z1.py::test_stage1_strict_checkpoint_load`）。

## Smoke 状态

单元测试：PASS（13/13 新增 + 20/20 回归 `test_z0_only` / `test_hvq` /
`test_protocol_metrics`，conda `prism-vq`）。
Stage 2 smoke（1 epoch，`artifact_root=artifacts/006/smoke`，Stage 1 复用 001
正式 checkpoint）：PASS——train / valid / test 全流程完成；ŷ0、Δŷ、ŷ 形状与
`ŷ == ŷ0 + Δŷ` 数值恒等、stop-gradient、residual head 非零梯度与参数更新、
Stage 1 无梯度、checkpoint 恢复、prediction/metric 诊断列均由
`artifacts/006/smoke/verify_residual_smoke.py` 逐项验证通过
（1-epoch smoke 指标仅供流程验证，不构成实验结论）。

## 受控对照与 Phase 2 判读

- 003：仅 z0 → `ŷ0`；001：representation-level `z0+z1`；006：prediction-level
  `ŷ0 + Δŷ(z1)`。
- Phase 2 正式结果中，`score` 列为最终 `ŷ`，`score_main` 列为 z0 主路径
  `ŷ0`，`delta` 列为 `Δŷ`；指标含 `IC_main` / `RankIC_main`（ŷ0）、
  `Delta_*`（Δŷ 统计）、`Corr_delta_resid` / `RankCorr_delta_resid`
  （Δŷ 与真实残差的相关），用于回答：residual branch 是否改善相对 003 的
  IC / RankIC；最终 ŷ 是否优于单独 ŷ0；Δŷ 幅度是否非零且稳定；Δŷ 与真实
  residual 是否有效相关；z1 是否因此获得可观测的增量预测价值。
