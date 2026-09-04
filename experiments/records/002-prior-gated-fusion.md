# 002 — prior-gated-fusion

## Idea

在 PRISM-VQ 中，将 prior factor 与 learned latent factor 的融合方式改成
可学习的 gated fusion：模型不再用固定相加融合两类因子，而是学习一个
per-sample 的 gate，自适应决定当前样本中 prior 信息和 latent 信息各占
多少权重。

## Motivation

原实现中 Stage 2 的收益预测为 `alpha + prior_term + latent_term` 的固定
相加（`trainer/train_ypred.py::ReturnPredictor`），prior 与 latent 两类
信息的相对贡献无法随样本状态调整。引入可学习 gate 后，模型可以按样本
自适应地在先验因子与学习到的潜因子之间分配权重，验证该自适应融合能否
改善预测与回测表现。

## Modification

- `trainer/train_ypred.py::ReturnPredictor` 新增 `fusion` 参数
  （`'fixed' | 'gated'`，默认 `'fixed'` 保持原行为）。`'gated'` 模式下
  新增 `self.gate = nn.Linear(num_prior + num_latent, 1)`，以
  `[f_prior, f_latent]` 为输入、经 sigmoid 得到 per-sample 标量门
  `g`，输出 `alpha + 2*g*prior_term + 2*(1-g)*latent_term`；
  gate 显式零初始化（`gate.weight = 0`、`gate.bias = 0`），
  保证初始化时所有样本 `g = 0.5`，配合 2 倍缩放使初始输出严格等于
  原始固定融合 `alpha + prior_term + latent_term`。
  `use_prior=False` 路径不变。
- `GenerateReturn.__init__` 读取 `config['predictor'].get('fusion',
  'fixed')` 并传入 `ReturnPredictor`。
- `configs/config.yaml`：`predictor.fusion: 'gated'`，默认
  `train.seed: 0`。
- 分支根目录 `README.md` 改写为本实验说明（按 RULES 的 README 规则）。
- 新增 `tests/test_gated_fusion.py`（9 个 unittest 用例：fixed 模式与
  原公式一致、gate 显式零初始化、新建 gated predictor 初始输出严格等于
  fixed fusion（无需手动清零）、gate 两端极限分别退化为
  `alpha + 2*prior_term` / `alpha + 2*latent_term`、gate 随样本变化、
  梯度可达 gate 参数、非法 fusion 报错）。

## Constraints

- 数据划分：train 2009–2020，valid 2021–2022，test 2023–2025（CSI300）
- 训练协议：两 stage 均 max 70 epoch，early stop patience 15；
  Stage 1 固定 seed 42，Stage 2 仅验证 seed 0
- 回测协议：Top30/Drop5，open 0.0005 / close 0.0015，min_cost 0，
  close 成交，CN limit=None（本阶段不执行）
- Stage 1（VQVAE）、HyperFusion、MoE、数据与回测管线均不改动；
  唯一实验变量为 Stage 2 `ReturnPredictor` 的 prior/latent 融合方式

## Git

Base: main
Branch: exp/002-prior-gated-fusion
Commit: 2752a2f（ac5d1f6 初版 gated fusion；2752a2f 修正为 2 倍缩放 +
gate 显式零初始化，初始输出与原固定融合严格等价）

## Smoke Test

Status: PASS
Notes: 单元测试 9/9 通过（`python -m unittest tests.test_gated_fusion`，
环境 conda `prism-vq`）。Stage 1 smoke：`stage1.py train.num_epochs=1
train.run_name=gfuse_smoke`，1 epoch 完成，val_loss=0.9228，产出
`checkpoints/gfuse_smoke-epoch=0-val_loss=0.9228.ckpt`。Stage 2 smoke：
`stage2.py train.num_epochs=1 train.seed=0
predictor.saved_model="gfuse_smoke-epoch=0-val_loss=0.9228.ckpt"`，
训练 + valid/test 推理全流程完成（Test RankIC 0.0448，仅供流程验证），
产出 checkpoint 中确认含 `return_predictor.gate.weight`（1×141）且
weight/bias 已从零初始化更新，证明 gate 端到端参与训练。日志：
`logs/stage1_gfuse_smoke.log`、`logs/stage2_gfuse_smoke.log`。
注意：Stage 2 smoke 的预测输出覆盖了 `res/VQ512_csi300_mo2_k1_.../
0_best.pkl`（该路径按配置参数命名，001 的同名产物被 smoke 结果覆盖）。

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

## Conclusion

正式实验完成后填写。
