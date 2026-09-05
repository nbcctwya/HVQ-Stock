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
Commit: 1d16627（ac5d1f6 初版 gated fusion；2752a2f 修正为 2 倍缩放 +
gate 显式零初始化，初始输出与原固定融合严格等价；632365c 引入
artifact_root 产物隔离；1d16627 分支 README 补充 artifact_root 用法）

## Smoke Test

Status: PASS
Notes: 单元测试 9/9 通过（`python -m unittest tests.test_gated_fusion`，
环境 conda `prism-vq`）。Stage 1 smoke：`stage1.py train.num_epochs=1
train.run_name=gfuse_smoke artifact_root=artifacts/002/smoke`，
1 epoch 完成，val_loss=0.9228，产出
`artifacts/002/smoke/checkpoints/gfuse_smoke-epoch=0-val_loss=0.9228.ckpt`。
Stage 2 smoke：`stage2.py train.num_epochs=1 train.seed=0
artifact_root=artifacts/002/smoke
predictor.saved_model="gfuse_smoke-epoch=0-val_loss=0.9228.ckpt"`，
Stage 2 正确从 artifact root 的 checkpoints/ 加载 Stage 1 checkpoint，
训练 + valid/test 推理全流程完成（Test RankIC 0.0448，与不使用
artifact_root 的上一轮 smoke 完全一致，仅供流程验证）；全部产物
（stage1/stage2 checkpoint、wandb 文件、prediction、metric、日志）
均落在 `artifacts/002/smoke/` 下，项目级 `checkpoints/` 与 `res/`
经前后 diff 确认无任何写入。产出 checkpoint 中确认含
`return_predictor.gate.weight`（1×141）且 weight/bias 已从零初始化
更新，证明 gate 端到端参与训练。
日志：`artifacts/002/smoke/stage1.log`、`artifacts/002/smoke/stage2.log`。
注意：更早一轮 smoke（未使用 artifact_root）曾覆盖
`res/VQ512_csi300_mo2_k1_.../0_best.pkl` 中 001 的同名产物；
引入 artifact_root 后该问题已消除。

## Result

Status: DONE — Corrected Protocol seed0（test 区间 2023-01-01 – 2025-12-31）

协议：corrected Stage 2 freeze + corrected MDD；Stage 2 seed 0；回测
Top30/Drop5，open 0.0005 / close 0.0015，min_cost 0，close 成交，
CN limit=None。

### Controlled result（正式比较依据，与 corrected baseline 共享 exact Stage 1）

Stage 1 checkpoint provenance：`infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`
（PRISM-VQ 原始 single-VQ512 Stage 1，与 corrected baseline 完全相同；
strict 加载验证 missing=0 / unexpected=0）。
唯一核心变量：Stage 2 融合方式 fixed fusion（baseline）vs gated fusion（002）。
Stage 2 seed：0
产物：`artifacts/002/run/`（res/、backtest、summary.json、stage2.log、backtest.log）。

IC: 0.0370
ICIR: 0.2172
RankIC: 0.0552
RankICIR: 0.3309

Annual Return: 8.03%（基准 6.40%，超额 1.63%，AR 差值口径）
Sharpe: 0.4577
Sortino: 0.6687
MDD: -26.52%
Calmar: 0.3028
Turnover: 0.3289

### Previous corrected result（存在 Stage 1 实例混淆，不再作为比较依据）

使用 002 自训练 Stage 1 `infucsi300...epoch=10-val_loss=0.5933.ckpt`
（与 baseline 不同训练实例）：
IC 0.0317 / ICIR 0.1697 / RankIC 0.0508 / RankICIR 0.2664 /
AR 9.72% / Sharpe 0.6638 / Sortino 0.9282 / MDD -16.32% /
Calmar 0.5955 / Turnover 0.3251。
产物备份：`artifacts/002/run/uncontrolled_stage1/`。

### Historical（pre-fix，不再作为比较依据）

旧 Stage 2 freeze 行为 + 旧 MDD 实现时期的 seed0 结果与 previous corrected
逐位一致（freeze bug 在 PL 2.6.4 下未实际触发）。产物备份：
`artifacts/002/run/pre_fix/`。

代码版本：实验代码 1d16627 + 公共 correctness fix 726836e（`train()` 覆写
强制冻结模块 eval；不改变实验 Idea、结构、超参、数据或协议）。

## Conclusion

在与 baseline 共享 exact Stage 1 checkpoint 的严格受控对照下（唯一变量为
Stage 2 融合方式），gated fusion 与 fixed fusion 的排序能力几乎相同
（IC 0.0370 vs 0.0373，RankIC 0.0552 vs 0.0552，RankICIR 0.3309 vs 0.3313），
组合层面略弱（AR 8.03% vs 9.24%，Sharpe 0.4577 vs 0.5223，
MDD -26.52% vs -26.90%，Turnover 0.3289 vs 0.3295）。

结论：per-sample 可学习 gate 没有带来实质收益——gate 从零初始化
（g=0.5 等价 fixed fusion）出发，训练后几乎没有学到优于固定相加的
融合方式。此前基于不同 Stage 1 实例观察到的"002 的 Sharpe/MDD 优于
baseline"由 Stage 1 实例差异造成，在受控条件下消失。
