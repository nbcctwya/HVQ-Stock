# 004 — hvq-learnable-z1-scale

## Idea

在 001 的 Residual HVQ 两级表示基础上，为第二级残差量化表示 z1 引入一个
全局可学习缩放系数 α，将 Stage 2 中原本固定使用的两级融合 `z = z0 + z1`
修改为 `z = z0 + α · z1`。α 为单个可学习标量参数，通过 sigmoid 约束到
[0, 1]：`α = sigmoid(a)`，`a`（代码中 `GenerateReturn.z1_scale_raw`）随
Stage 2 一起训练。

## Motivation

base 001 固定 α=1，而 003（hvq-z0-only）等价于下游使用层面的 α=0 对照。
003 在 RankIC、Sharpe、MDD 上并未弱于 001，说明 z1 可能并非完全无用，
而是以固定权重 1 注入 Stage 2 时贡献过强、混入更多 reconstruction/detail
信息或噪声。本实验让模型自行学习 z1 的全局最优贡献强度，以判断：

1. 最优 α 是否趋近于 0（支持"z1 对收益预测整体价值有限"）；
2. 最优 α 是否稳定落在 0 与 1 之间（支持"z1 有增量预测价值，但需降低
   注入强度"）；
3. learnable α 能否在 001 与 003 之间取得更好的预测或投资组合表现。

## Modification

- `module/quantise_hvq.py::ResidualVectorQuantiser` 新增
  `forward_per_level(h_batch)`：按与 `forward` 完全相同的残差链运行各级
  量化，返回按级量化输出 list（`sum(z_q_levels)` 数值上等于 `forward` 的
  z0+z1）；默认 `forward` 行为不变，Stage 1 不受影响，z0/z1 生成方式未改。
- `trainer/train_ypred.py::GenerateReturn`：
  - 新增 `predictor.learnable_z1_scale` 开关（代码默认 False 保持 001 行为；
    本实验默认配置为 true）与 `predictor.z1_scale_init`（a 的初值 3.0）；
    开启时要求两级 hvq 量化器否则报错；
  - 新增单个全局 `nn.Parameter z1_scale_raw`，α = sigmoid(z1_scale_raw)；
  - forward 融合改为 `z_q = z0.detach() + α · z1.detach()`（z0/z1 沿用
    Stage 1 惯例 detach，α 不 detach 以获得梯度）；
  - 训练全程记录 `z1_alpha`（wandb 逐步指标）、`Best_Val_z1_alpha`
    （best checkpoint 对应 epoch 的 α），训练结束打印最终 α；α 作为模型
    参数随 checkpoint 保存/加载（`state_dict['z1_scale_raw']`）。
- `stage2.py`：加载 best checkpoint 后打印其对应 α（仅报告用途）。
- `configs/config.yaml`：`predictor.learnable_z1_scale: true`、
  `predictor.z1_scale_init: 3.0`——默认配置即 004 实验本身，无需实验特有
  CLI override；Stage 2 `train.seed: 0` 不变。
- 新增 `tests/test_learnable_z1_scale.py`（12 个用例：per-level 输出契约、
  α 为 Stage 2 可训练参数且在优化器中、sigmoid 约束 [0,1]、初始化数值、
  融合严格为 z0+α·z1、α 获梯度并更新、Stage 1 冻结、checkpoint 保存/加载
  恢复 α、默认 config 直接启用实验）。
- 分支根目录 `README.md` 改写为 004 实验说明。

## Constraints

- 唯一实验变量：Stage 2 两级融合由固定 `z0 + z1` 改为 `z0 + α·z1`
  （α = sigmoid(a)，单个全局可学习标量）；无 per-stock / per-date /
  per-sample / market-conditioned / MLP gate
- z0 / z1 生成方式、quantizer 结构未改，无新增 auxiliary loss
- α 属 Stage 2 可训练参数；Stage 1（encoder/quantizer/revin）完全冻结
- α 初始化：a_init = 3.0，**α_init = sigmoid(3.0) ≈ 0.9526**（起点接近
  001 的 α=1，sigmoid 未饱和 σ′(3)≈0.045，梯度可流动；单一明确可复现
  初值，不引入额外实验变量）
- Stage 1 两级 HVQ 结构、量化器配置（hvq, num_levels=2,
  level_num_embed=[256,256]）、数据划分与 001 完全一致；strict checkpoint
  加载已实证（missing=0/unexpected=0）
- 数据划分：train 2009–2020，valid 2021–2022，test 2023–2025（CSI300）
- 训练协议：Stage 2 max 70 epoch，early stop patience 15，seed 0
- 回测协议：Top30/Drop5，open 0.0005 / close 0.0015，min_cost 0，
  close 成交，CN limit=None（本阶段不执行）

## Git

Base: exp/001-hvq-residual-2level
Branch: exp/004-hvq-learnable-z1-scale
Commit: f8b782c9e95c05b27a15e8c433e48efb12df5c59
Stage 1 provenance: 复用实验 001 的 exact Stage 1（queue `stage1_source: "001"`）

## Smoke Test

Status: PASS
Notes: 单元测试 12/12（`tests.test_learnable_z1_scale`）+ 14/14
（`tests.test_hvq` 与 `tests.test_protocol_metrics` 回归）通过
（conda `prism-vq`）。
Stage 1 strict 兼容性实证（`artifacts/004/smoke/verify_alpha_smoke.py`）：
用 004 默认 config 构建 GenerateReturn，对 001 正式 Stage 1 checkpoint
（`hvq_csi300_full-epoch=5-val_loss=0.4592.ckpt`）strict 加载，
encoder/quantizer/revin 均 missing=0/unexpected=0；融合数值验证
`z_q == z0 + α·z1`（max diff 0）；α 经一次 backward 获得梯度并被优化器
更新（a: 3.000000 → 2.998957，α: 0.952574 → 0.952527）；Stage 1 参数
无任何梯度。
Stage 2 smoke：`stage2.py train.num_epochs=1 train.seed=0
artifact_root=artifacts/004/smoke`（WANDB_MODE=offline），1 epoch 训练 +
valid/test 推理全流程完成（Test RankIC 0.0413，仅供流程验证）。日志中可
直接读取 α：初始 `alpha_init=0.952574`，训练结束
`Final learned z1_alpha = 0.951051`（z1_scale_raw: 3.000000 → 2.966799，
证明 α 实际进入优化过程），best checkpoint `z1_alpha = 0.951051`，且
checkpoint `state_dict['z1_scale_raw']` 可正确恢复该值；wandb 记录
`z1_alpha` 轨迹与 `Best_Val_z1_alpha`。
全部产物（stage2 checkpoint、wandb 文件、prediction、metric、日志、验证
脚本）均落在 `artifacts/004/smoke/` 下；公共 `checkpoints/` 与 `res/`
经前后确认无任何写入。

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

Phase 2 正式结果除常规指标外，必须报告最终/best-checkpoint 的 α：
从 best checkpoint `state_dict['z1_scale_raw']`（α = sigmoid(raw)）、
运行日志的 `Best checkpoint z1_alpha` / `Final learned z1_alpha`，以及
wandb 指标 `z1_alpha` / `Best_Val_z1_alpha` 读取。

## Conclusion

正式实验完成后填写。
