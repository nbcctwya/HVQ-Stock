# exp/005 — hvq-samplewise-z1-gate

- Base: `exp/004-hvq-learnable-z1-scale`
- Branch: `exp/005-hvq-samplewise-z1-gate`

## Idea / Motivation

004 将所有样本共享的固定融合 `z = z0 + z1` 改为全局可学习缩放
`z = z0 + α · z1`（α 为所有股票、所有日期共享的单个标量）。但 z1 的
有效性可能具有样本异质性：部分样本的 residual representation 有增量
预测价值，而另一些样本的 z1 主要包含 reconstruction detail / noise。

本实验（005）用最简单的 sample-wise gate 验证：

1. z1 的最优贡献是否明显因样本而异；
2. sample-wise gate 是否优于 004 的 global scalar α；
3. gate 分布是否具有明显的离散度，而不是退化成近似常数；
4. 如果 005 有效，是否值得进一步研究 market-conditioned /
   regime-conditioned z1 selection。

## gate 定义与初始化

Stage 2 两级融合：

```
z_i = z0_i + g_i · z1_i
g_i = sigmoid(Linear(concat(z0_i, z1_i)))
```

- `g_i` 为每个样本独立的标量，sigmoid 约束到 `[0, 1]`；
- gate 网络为单个 `nn.Linear(2*d -> 1)`（d = vq_embed_dim = 128），
  参数量 `2*128 + 1 = 257`；不引入 MLP / attention / market feature；
- gate 输入仅使用 Stage 1 已产生的 `z0`、`z1`（均 detach），不引入
  prior factor、market feature、raw feature 或其他信息；
- 初始化：weight 全零、bias = 3.0，因此所有样本初始
  `g_i = sigmoid(3.0) ≈ 0.952574`，与 004 的
  `α_init = sigmoid(3.0)` 完全一致——训练开始时 004 与 005 的融合
  行为一致，区别仅在于 004 后续只能学习一个 global α，而 005 后续
  可以学习 sample-dependent g_i。

## 与 base 004 的唯一核心区别

唯一实验变量：004 的**全局单标量 α**（`α = sigmoid(a)`，1 个参数）
被替换为**每个样本一个标量 gate g_i**（`nn.Linear(256->1)`，257 个
参数）。其余全部保持一致：

- Stage 1 来源、HVQ 结构、z0/z1 生成方式（同一
  `forward_per_level` 残差链）、codebook、数据划分不变；
- Stage 2 seed（0）、训练预算、loss、optimizer、prior/latent fusion、
  MoE、回测协议及其他超参数不变；
- z0 / z1 仍 detach，Stage 1 encoder / quantizer / RevIN 完全冻结；
- 不新增 auxiliary loss；
- **不保留 004 的 global α 作为额外乘子**（无 `z0 + α·g_i·z1` 双重
  缩放）：`learnable_z1_scale` / `z1_scale_raw` 已从本分支移除，
  005 只使用一个 sample-wise gate。

## 核心修改（相对 base 004）

- `trainer/train_ypred.py::GenerateReturn`：
  - 移除 `z1_scale_raw`（004 全局 α），替换为
    `self.z1_gate = nn.Linear(2*vq_embed_dim, 1)`（weight 全零初始化、
    bias 由 `predictor.z1_gate_bias_init` 初始化，默认 3.0）；
  - 新配置 `predictor.samplewise_z1_gate`（代码默认 False 保持 001 行为；
    本分支默认配置为 true）与 `predictor.z1_gate_bias_init`；
    开启时要求两级 hvq 量化器，否则报错；
  - forward 融合改为
    `z_q = z0.detach() + sigmoid(z1_gate(cat(z0, z1))) * z1.detach()`；
  - 训练过程记录 `z1_gate_mean` / `z1_gate_std`（wandb 逐步指标，
    batch 级），validation 按 epoch 聚合 `Val_z1_gate_mean/std/min/max`，
    best checkpoint 对应 epoch 记录 `Best_Val_z1_gate_mean/std/min/max`，
    训练结束打印最后一轮 validation 的 gate 统计；gate 参数随
    checkpoint 保存/加载（`state_dict['z1_gate.weight/bias']`）。
- `stage2.py`：加载 best checkpoint 后在整个 validation 集上统计并打印
  gate 分布（mean/std/min/max，仅报告用途）。
- `module/quantise_hvq.py`：`forward_per_level` docstring 更新
  （机制本身不变，与 004 相同）。
- `configs/config.yaml`：`predictor.samplewise_z1_gate: true`、
  `predictor.z1_gate_bias_init: 3.0`——默认配置即 005 实验本身，无需
  实验特有 CLI override；Stage 2 `train.seed: 0` 不变。
- `tests/test_learnable_z1_scale.py`（004 的全局 α 测试）移除，替换为
  `tests/test_samplewise_z1_gate.py`（保留原有 `forward_per_level`
  契约测试 + 14 个 gate 用例）。

## Stage 1 复用关系

- `stage1_source: "001"`：复用实验 001 的正式 Stage 1 best checkpoint
  （`hvq_csi300_full-epoch=5-val_loss=0.4592.ckpt`），本实验不重新训练
  Stage 1，与 004 完全相同。
- Stage 1 结构、量化器配置、数据划分与 001/004 完全一致；checkpoint 以
  **strict** 方式加载（encoder/quantizer/revin 均 missing=0 /
  unexpected=0，已实证）。
- Stage 1 在 Stage 2 中保持完全冻结（encoder/quantizer/revin
  requires_grad=False 且强制 eval）。

## gate 分布的读取位置（Phase 2 正式运行后）

- best checkpoint 的 `state_dict['z1_gate.weight']` /
  `state_dict['z1_gate.bias']`；
- 控制台日志：`== sample-wise z1 gate enabled: ... g_init=... ==`
  （初始值）、`========== Final z1_gate stats (last val epoch): ...`
  （训练结束）与 `========== Best checkpoint z1_gate stats (valid
  set): ... ==========`（best checkpoint）；
- wandb 指标 `z1_gate_mean` / `z1_gate_std`（逐步轨迹）、
  `Val_z1_gate_mean/std/min/max`（每 epoch）与
  `Best_Val_z1_gate_mean/std/min/max`（best epoch）。

## 运行

默认配置即为本实验，无需额外 override 启用核心改动：

```bash
# Stage 2（Stage 1 复用 001，checkpoint 由执行器放入 artifact_root/checkpoints）
conda run -n prism-vq python stage2.py data.universe=csi300 \
    artifact_root=artifacts/005/run
```

CLI override 仅用于统一执行层参数（`train.seed`、`train.num_epochs`、
`artifact_root` 等）。

## 单元测试

```bash
conda run -n prism-vq python -m unittest tests.test_samplewise_z1_gate -v
# 回归：tests.test_hvq、tests.test_protocol_metrics
```

## Smoke 状态

PASS — 单元测试 17/17（`tests.test_samplewise_z1_gate`）+ 14/14
（`tests.test_hvq`、`tests.test_protocol_metrics` 回归）通过；001 Stage 1
checkpoint strict 加载验证（missing=0/unexpected=0）、初始化融合严格
等价于 `z0 + sigmoid(3.0) * z1`（与 004 初始行为一致，max diff 0）、
gate 梯度与更新、Stage 1 冻结均实证通过；Stage 2 smoke（1 epoch,
seed 0, artifact_root=artifacts/005/smoke）全流程跑通，日志中可读取
初始 / 训练结束 / best checkpoint 的 gate mean/std/min/max。详见 `main`
分支 `experiments/records/005-hvq-samplewise-z1-gate.md`。

---

原始 PRISM-VQ 项目说明见 `main` 分支的 README。
