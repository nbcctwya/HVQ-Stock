# exp/002 — prior-gated-fusion

实验分支：在 PRISM-VQ 的 Stage 2 中，将 prior factor 与 learned latent
factor 的固定融合改为可学习的 gated fusion。

- Base: `main`（原始 PRISM-VQ）
- Branch: `exp/002-prior-gated-fusion`

## Idea / Motivation

原始 PRISM-VQ 的 Stage 2 收益预测为固定相加：

```
y = alpha + prior_term + latent_term
```

prior 信息与 latent 信息的相对贡献无法随样本调整。本实验引入一个
可学习的 per-sample 标量门 `g`，让模型自适应决定当前样本中两类信息
各占多少权重。

## 核心修改

`trainer/train_ypred.py::ReturnPredictor` 新增 `fusion` 参数
（`'fixed' | 'gated'`，默认 `'fixed'` 保持原行为）。`'gated'` 模式下：

```
g = sigmoid(Linear([f_prior, f_latent]))        # (B,)
y = alpha + 2*g*prior_term + 2*(1-g)*latent_term
```

- gate 参数显式零初始化（`gate.weight = 0`，`gate.bias = 0`），
  保证初始化时所有样本 `g = 0.5`；
- 2 倍缩放使 `g = 0.5` 时严格等价于原始固定融合
  `alpha + prior_term + latent_term`，即模型从"与 base 完全等价"的
  起点开始学习如何偏离等权；
- `configs/config.yaml` 设置 `predictor.fusion: 'gated'`，
  默认 `train.seed: 0`；
- `use_prior=False` 消融路径不受影响；
- 新增 `tests/test_gated_fusion.py`（9 个单元测试）。

## 与 base 的区别

唯一实验变量是 Stage 2 `ReturnPredictor` 中 prior/latent 两项的融合
方式：base 为固定相加，本分支为零初始化的可学习 gate 加权（初始与
base 严格等价）。数据划分、训练协议（Stage 1 固定 seed 42，Stage 2
seed 0，70 epoch / patience 15）、回测协议与模型其余部分均不变。

## Smoke 状态

PASS — 单元测试 9/9 通过；Stage 1 smoke（1 epoch，
`gfuse_smoke-epoch=0-val_loss=0.9228.ckpt`）与 Stage 2 smoke
（1 epoch，seed 0，gated fusion）全流程跑通，checkpoint 中确认
`return_predictor.gate` 参数存在且参与训练。
日志见 `logs/stage1_gfuse_smoke.log`、`logs/stage2_gfuse_smoke.log`。

## 运行与 artifact 隔离

默认配置即为本实验（gated fusion + seed 0），无需实验特有 override。
CLI override 仅用于统一执行层参数，例如用 `artifact_root` 把全部产物
隔离到实验专属目录（checkpoints、res、wandb 文件都会落在其下，
Stage 2 也会从该目录的 checkpoints/ 加载 Stage 1 checkpoint）：

```bash
# Stage 1
python stage1.py artifact_root=artifacts/002/run

# Stage 2（saved_model 为相对文件名时，从 <artifact_root>/checkpoints 解析）
python stage2.py artifact_root=artifacts/002/run \
  predictor.saved_model="<stage1_ckpt_name>"

# 回测产物跟随 --pred_path / --output_dir，天然隔离
python backtest_qlib.py --pred_path artifacts/002/run/res/<run>/0_best.pkl \
  --universe csi300
```

不传 `artifact_root` 时行为与上游一致（项目级 `checkpoints/` 与 `res/`）。

---

原始 PRISM-VQ 项目说明见 `main` 分支的 README。
