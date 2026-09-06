# 005 — hvq-samplewise-z1-gate

## Idea

在 004 `hvq-learnable-z1-scale` 的基础上，将所有样本共享的全局可学习
z1 缩放系数 `z = z0 + α · z1` 进一步改为 **sample-wise adaptive z1
gate**：`z_i = z0_i + g_i · z1_i`，其中
`g_i = sigmoid(Linear(concat(z0_i, z1_i)))`，`g_i` 为每个样本独立的
标量，范围 `[0, 1]`。

## Motivation

004 只能学习一个所有股票、所有日期共享的全局 α，但 z1 的有效性可能
具有样本异质性：部分样本的 residual representation 有增量预测价值，
而另一些样本的 z1 主要包含 reconstruction detail / noise。005 用最
简单的 sample-wise gate 验证：

1. z1 的最优贡献是否明显因样本而异；
2. sample-wise gate 是否优于 004 的 global scalar α；
3. gate 分布是否具有明显的离散度，而不是退化成近似常数；
4. 如果 005 有效，是否值得进一步研究 market-conditioned /
   regime-conditioned z1 selection。

## Modification

- `trainer/train_ypred.py::GenerateReturn`：
  - 移除 004 的全局 `nn.Parameter z1_scale_raw`（α = sigmoid(a)，
    1 个参数），替换为 `self.z1_gate = nn.Linear(2*vq_embed_dim, 1)`
    （`g_i = sigmoid(Linear(concat(z0_i, z1_i)))`，d=128 时参数量
    257）；开启时要求两级 hvq 量化器否则报错；
  - forward 融合改为
    `z_q = z0.detach() + sigmoid(z1_gate(cat(z0, z1))) * z1.detach()`
    （z0/z1 沿用 Stage 1 惯例 detach，梯度只流向 gate 参数）；
  - 训练全程记录 `z1_gate_mean` / `z1_gate_std`（wandb 逐步指标），
    validation 按 epoch 聚合 `Val_z1_gate_mean/std/min/max`，
    best checkpoint 对应 epoch 记录 `Best_Val_z1_gate_mean/std/min/max`，
    训练结束打印最后一轮 validation 的 gate 统计；gate 参数随
    checkpoint 保存/加载（`state_dict['z1_gate.weight/bias']`）。
- `stage2.py`：加载 best checkpoint 后在整个 validation 集上统计并打印
  gate 分布（mean/std/min/max，仅报告用途）。
- `configs/config.yaml`：移除 `learnable_z1_scale` / `z1_scale_init`，
  替换为 `predictor.samplewise_z1_gate: true` 与
  `predictor.z1_gate_bias_init: 3.0`——默认配置即 005 实验本身，无需
  实验特有 CLI override；Stage 2 `train.seed: 0` 不变。
- `tests/test_learnable_z1_scale.py`（004 全局 α 测试）移除，替换为
  `tests/test_samplewise_z1_gate.py`（17 个用例：保留原
  `forward_per_level` 契约测试 + gate 输出 shape/范围/初始化/融合严格
  等价/样本异质性/优化器参数/梯度与更新/Stage 1 冻结/checkpoint 恢复/
  无残留全局 α/默认 config 直接启用实验）。
- 分支根目录 `README.md` 改写为 005 实验说明。

## Constraints

- 唯一实验变量：004 的全局单标量 α 替换为每个样本一个标量 gate g_i；
  gate 保持极简（单个 Linear 输出维度 1），无 MLP / attention /
  market feature；gate 输入仅使用 Stage 1 已产生的 z0/z1
- 不保留 004 的 global α 作为额外乘子（禁止 `z0 + α·g_i·z1` 双重缩放）；
  005 只使用一个 sample-wise gate
- 初始化受控：gate weight 全零、bias = 3.0，所有样本初始
  g_i = sigmoid(3.0) ≈ 0.952574，与 004 的 α_init 完全一致（训练开始
  时 004 与 005 融合行为一致）
- z0 / z1 仍 detach，Stage 1（encoder/quantizer/revin）完全冻结；
  不新增 auxiliary loss；不改变 Stage 1
- Stage 1 两级 HVQ 结构、量化器配置（hvq, num_levels=2,
  level_num_embed=[256,256]）、数据划分与 001/004 完全一致；strict
  checkpoint 加载已实证（missing=0/unexpected=0）
- 数据划分：train 2009–2020，valid 2021–2022，test 2023–2025（CSI300）
- 训练协议：Stage 2 max 70 epoch，early stop patience 15，seed 0
- 回测协议：Top30/Drop5，open 0.0005 / close 0.0015，min_cost 0，
  close 成交，CN limit=None（本阶段不执行）

## Git

Base: exp/004-hvq-learnable-z1-scale
Branch: exp/005-hvq-samplewise-z1-gate
Commit: a232f93d479fa746a0239252edcdcdfd70df7d8a
Stage 1 provenance: 复用实验 001 的 exact Stage 1（queue `stage1_source: "001"`）

## Smoke Test

Status: PASS
Notes: 单元测试 17/17（`tests.test_samplewise_z1_gate`）+ 14/14
（`tests.test_hvq` 与 `tests.test_protocol_metrics` 回归）通过
（conda `prism-vq`）。
Stage 1 strict 兼容性实证（`artifacts/005/smoke/verify_gate_smoke.py`）：
用 005 默认 config 构建 GenerateReturn，对 001 正式 Stage 1 checkpoint
（`hvq_csi300_full-epoch=5-val_loss=0.4592.ckpt`）strict 加载，
encoder/quantizer/revin 均 missing=0/unexpected=0；初始 gate
mean=0.952574、std=0（所有样本恰为 sigmoid(3.0)）；初始化融合数值验证
`z_q == z0 + sigmoid(3.0)*z1`（max diff 0，与 004 初始行为一致）；
gate 经一次 backward 获得非零梯度并被优化器更新（|dW|=4.54e-01，
|db|=3.41e-02）；Stage 1 参数无任何梯度；smoke best checkpoint 的
`z1_gate.weight/bias` 可 strict 恢复，且 checkpoint 中无
`z1_scale_raw` 残留（无双重缩放）。
Stage 2 smoke：`stage2.py train.num_epochs=1 train.seed=0
artifact_root=artifacts/005/smoke`（WANDB_MODE=offline），1 epoch 训练 +
valid/test 推理全流程完成（Test RankIC 0.0367，仅供流程验证）。
gate 行为统计（smoke，仅验证 gate 未失效，不据此调模型）：
- 初始：mean=0.952574，std=0.000000（构造保证，= sigmoid(3.0)）；
- 训练结束（last val epoch）：mean=0.945872，std=0.005627，
  min=0.928064，max=0.963935（bias 3.000000→2.966587，
  |W| 0→0.139317）；
- best checkpoint（epoch=0，valid 集）：mean=0.945872，std=0.005627，
  min=0.928064，max=0.963935（1 epoch smoke 中 best 即最终状态）；
gate 在 1 epoch 内已呈现非零离散度（std>0），未退化为常数。
全部产物（stage2 checkpoint、wandb 文件、prediction、metric、日志、验证
脚本）均落在 `artifacts/005/smoke/` 下；公共 `checkpoints/` 与 `res/`
经前后确认无任何写入。

## Result

Status: DONE（test 区间 2023-01-01 – 2025-12-31）

IC: 0.0329
ICIR: 0.1908
RankIC: 0.0488
RankICIR: 0.2794

Annual Return: 16.11%（基准 6.40%，超额 9.71%）
Sharpe: 1.0237
Sortino: 1.5858
MDD: -13.20%
Calmar: 1.2208
Turnover: 0.3274

## Conclusion

Phase 2 固定执行器完成正式训练、预测与回测（pinned commit a232f93d479fa746a0239252edcdcdfd70df7d8a）。Stage 1 复用实验 001 的正式 checkpoint：`/home/nbcctwya/baselines/masterVQ/HVQ-Stock/artifacts/001/run/checkpoints/hvq_csi300_full-epoch=5-val_loss=0.4592.ckpt`（本实验未重新训练 Stage 1）；Stage 2 seed 0。

产物：`artifacts/005/run/`（checkpoints/、res/、stage1.log、stage2.log、backtest.log、summary.json）。

## Post-hoc Diagnosis

Diagnosis date: 2026-09-06
Execution: read-only diagnosis; no retraining / no backtest
（在 pinned commit `a232f93d` 的临时 worktree 中对 best checkpoint 做
完整 validation 集只读前向诊断；统计与 `artifacts/005/run/stage2.log`
官方记录逐位一致，交叉验证通过。）

- best checkpoint epoch = 4
- validation samples = 145145

### best checkpoint 在完整 validation 集上的 gate 分布

| mean | std | min | p1 | p5 | p10 | p25 | median | p75 | p90 | p95 | p99 | max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.916905 | 0.042620 | 0.721620 | 0.7958 | 0.8351 | 0.8571 | 0.8920 | 0.9244 | 0.9504 | 0.9648 | 0.9716 | 0.9802 | 0.9873 |

其他诊断：

- 25.4% 样本 gate > 0.95；1.29% 样本 gate < 0.8；没有样本 gate < 0.5
- last validation epoch：std = 0.0903，min = 0.4253（离散度随训练持续扩大）
- gate bias = 2.880537，|W| = 0.547314（weight 从全零初始化学出非零值）

### gate 与表示统计的相关性

| Variable | Pearson | Spearman |
| -------- | ------: | -------: |
| `\|\|z0\|\|` | -0.345 | -0.282 |
| `\|\|z1\|\| / (\|\|z0\|\| + eps)` | +0.205 | +0.185 |
| `\|\|z1\|\|` | -0.020 | -0.034 |
| 一级残差范数 `\|\|h - z0\|\|` | +0.015 | +0.016 |
| 总量化误差 `\|\|h - z0 - z1\|\|` | +0.021 | +0.017 |

### 结论

- gate 没有退化成常数（std 0.0426，训练末 0.0903）；
- 存在真实但温和的 sample-wise heterogeneity（分布左偏压缩在高位，
  中位数 0.924，无样本接近 0）；
- 005 相对 004 确实学到了 sample-wise differentiation（均值 0.917
  低于 004 的全局 α=0.946，且从 std=0 的初始化学出了非零离散度）；
- 但 gate 几乎不响应 quantization error / reconstruction residual
  （相关系数 ≈0.02）；
- gate 最明显的关联是 `||z0||`（负相关）与 z1 相对幅度（正相关），
  因此当前机制更像 representation magnitude rebalancing，而不是可靠
  识别"哪些样本 z1 有预测信息"；
- 005 的 AR / Sharpe / Calmar 很强（16.11% / 1.0237 / 1.2208，为本研究
  线最优），但 RankIC 没有改善（0.0488，低于 001/003/004）；
- 当前只有 Stage 2 seed 0，因此组合层面优势只能视为值得复核的
  research signal，不能写成稳定结论。

跨实验综合解读见 `experiments/analysis/z1-utilization-001-006.md`。
