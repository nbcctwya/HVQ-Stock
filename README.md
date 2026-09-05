# exp/004 — hvq-learnable-z1-scale

- Base: `exp/001-hvq-residual-2level`
- Branch: `exp/004-hvq-learnable-z1-scale`

## Idea / Motivation

001 的 Stage 2 固定使用两级融合 `z = z0 + z1`（等价于 α=1）；003 的
z0-only 消融等价于 α=0。003 在 RankIC、Sharpe、MDD 上并未弱于 001，
说明第二级残差表示 z1 可能并非完全无用，而是以固定权重 1 注入 Stage 2 时
贡献过强、混入了更多 reconstruction/detail 信息或噪声。

本实验为第二级残差量化表示 z1 引入一个**全局可学习缩放系数 α**，让模型
自行学习 z1 的最优贡献强度，以判断：

1. 最优 α 是否趋近于 0（z1 对收益预测整体价值有限）；
2. 最优 α 是否稳定落在 0 与 1 之间（z1 有增量预测价值，但需降低注入强度）；
3. learnable α 能否在 001（α=1）与 003（α=0）之间取得更好的预测或
   投资组合表现。

## 融合定义

Stage 2 中原本固定使用的两级融合：

```
z = z0 + z1
```

修改为：

```
z = z0 + α · z1
```

其中 z0 / z1 的生成方式与 001 完全一致（同一残差链：第 0 级量化 h，
第 1 级量化残差 h − z0），仅融合权重由固定 1 变为可学习 α。

## α 的参数化与初始化

- α 为**单个全局可学习标量**（非 per-stock / per-date / per-sample /
  market-conditioned，无任何 gate 网络），通过 sigmoid 约束到 [0, 1]：

  ```
  α = sigmoid(a)
  ```

  `a`（代码中 `GenerateReturn.z1_scale_raw`）是 `nn.Parameter`，属于
  Stage 2 可训练参数，随 Stage 2 一起由 AdamW 优化（与其他 Stage 2 参数
  同一优化器、同一学习率协议，不引入新的超参数）。
- z0 / z1 沿用 Stage 1 惯例 `detach()`（Stage 1 全冻结），**α 不 detach**，
  保证 `a` 能从 Stage 2 损失获得梯度。
- 初始化：`a_init = 3.0`，即 **α_init = sigmoid(3.0) ≈ 0.9526**。
  设计理由：起点尽可能接近 base 001 的 α=1 行为，同时 sigmoid 在 a=3 处
  未饱和（σ′(3) ≈ 0.045），梯度仍可正常流动；不通过初始化引入额外实验
  变量（单一的、明确的、可复现的标量初值）。

## 核心修改（相对 base 001）

- `module/quantise_hvq.py::ResidualVectorQuantiser` 新增
  `forward_per_level(h_batch)`：按与 `forward` 完全相同的残差链运行各级
  量化，返回按级的量化输出 list（`sum(z_q_levels)` 数值上等于
  `forward` 的 z0+z1）；默认 `forward` 行为不变，Stage 1 不受影响。
- `trainer/train_ypred.py::GenerateReturn`：
  - 新增配置 `predictor.learnable_z1_scale`（代码默认 False 保持 001 行为；
    本分支默认配置为 true）与 `predictor.z1_scale_init`（a 的初值，默认 3.0）；
    开启时要求两级 hvq 量化器，否则报错；
  - 新增 `nn.Parameter z1_scale_raw`（α = sigmoid(z1_scale_raw)）；
  - forward 中融合改为 `z_q = z0.detach() + α · z1.detach()`；
  - 训练过程记录 `z1_alpha`（每个 train step 的 wandb 指标）、
    `Best_Val_z1_alpha`（best checkpoint 对应 epoch 的 α），并在训练结束
    时打印最终 α；α 作为模型参数随 checkpoint 保存/加载。
- `stage2.py`：加载 best checkpoint 后打印其对应的 α（Best checkpoint
  z1_alpha），仅报告用途，不影响训练逻辑。
- `configs/config.yaml`：`predictor.learnable_z1_scale: true`、
  `predictor.z1_scale_init: 3.0`——默认配置即 004 实验本身，无需实验特有
  CLI override；Stage 2 `train.seed: 0` 不变。
- 新增 `tests/test_learnable_z1_scale.py`（12 个用例）。

## 与 base 001 的唯一区别

唯一实验变量：Stage 2 两级融合由固定 `z0 + z1` 改为 `z0 + α·z1`
（α = sigmoid(a)，全局可学习标量）。模型结构、Stage 1、两级 VQ 配置
（hvq, num_levels=2, level_num_embed=[256,256]）、Stage 2 网络、损失函数、
优化器、数据处理及全部训练/回测配置均与 001 一致。

对照关系：base 001 等价于固定 α=1；003（hvq-z0-only）等价于下游使用层面的
α=0；本实验学习 α∈[0,1]。

## Stage 1 复用关系

- `stage1_source: "001"`：复用实验 001 的正式 Stage 1 best checkpoint
  （`hvq_csi300_full-epoch=5-val_loss=0.4592.ckpt`），本实验不重新训练
  Stage 1。
- Stage 1 结构、量化器配置、数据划分与 001 完全一致；`forward_per_level`
  只新增读取接口、不改动任何 Stage 1 参数命名，001 checkpoint 以
  **strict** 方式加载（encoder/quantizer/revin 均 missing=0 /
  unexpected=0，已实证）。
- Stage 1 在 Stage 2 中保持完全冻结（encoder/quantizer/revin
  requires_grad=False 且强制 eval，与 001 相同）。

## α 的读取位置（Phase 2 正式运行后）

- best checkpoint 的 `state_dict['z1_scale_raw']`：α = sigmoid(z1_scale_raw)；
- 控制台日志：`== learnable z1 scale enabled: ... alpha_init=... ==`（初始值）、
  `========== Best checkpoint z1_alpha: ... ==========` 与
  `========== Final learned z1_alpha: ... ==========`（训练结束）；
- wandb 指标 `z1_alpha`（逐步轨迹）与 `Best_Val_z1_alpha`（best epoch 的 α）。

## 运行

默认配置即为本实验，无需额外 override 启用核心改动：

```bash
# Stage 2（Stage 1 复用 001，checkpoint 由执行器放入 artifact_root/checkpoints）
conda run -n prism-vq python stage2.py data.universe=csi300 \
    artifact_root=artifacts/004/run
```

CLI override 仅用于统一执行层参数（`train.seed`、`train.num_epochs`、
`artifact_root` 等）。

## 单元测试

```bash
conda run -n prism-vq python -m unittest tests.test_learnable_z1_scale -v
# 回归：tests.test_hvq、tests.test_protocol_metrics
```

## Smoke 状态

PASS — 单元测试 12/12（`tests.test_learnable_z1_scale`）+ 14/14
（`tests.test_hvq`、`tests.test_protocol_metrics` 回归）通过；001 Stage 1
checkpoint strict 加载验证（missing=0/unexpected=0）、融合数值一致性
（z_q == z0 + α·z1，max diff 0）、α 梯度与更新、Stage 1 冻结均实证通过；
Stage 2 smoke（1 epoch, seed 0, artifact_root=artifacts/004/smoke）全流程
跑通，日志中可读取 α 初始值与训练后值。详见 `main` 分支
`experiments/records/004-hvq-learnable-z1-scale.md`。

---

原始 PRISM-VQ 项目说明见 `main` 分支的 README。
