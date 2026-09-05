# 001 — hvq-residual-2level

## Idea

在 PRISM-VQ 基础上引入残差式层次化向量量化（Residual HVQ），
用 2 级 codebook（256 + 256）替代单层 512 codebook，并端到端验证
该 baseline 在 CSI300 上的预测与回测效果。

## Motivation

这是本仓库的首个完整实验，作为本仓库首个改进实验：
验证残差式 2 级 VQ 能否在保持 IC 的同时改善码本利用与训练稳定性，
为后续基于 HVQ 的改进（如 market gating）提供对照基准。

## Modification

- 新增 `module/quantise_hvq.py::ResidualVectorQuantiser`：残差式 2 级量化
  （256+256 codebook），第 1 级量化第 0 级残差，`z_q = z_q0 + z_q1`，
  整体 STE；每级复用上游 `VectorQuantiser`（含 dead-code 重初始化与对比损失）
- `create_quantizer` 工厂接入 Stage 1/2，
  `vqvae.quantizer.type: single|hvq` 开关，`single` 为上游原行为
- 修复上游 Stage 2 codebook 冻结漏洞：`GenerateReturn.train()` 强制
  quantizer 保持 eval，防止 training mode 下 `.data` 改写 codebook
- 数据 pickle 通过 `data.pickle_dir` 复用 `../PRISM-VQ/dataset/data`，
  未重复生成

## Constraints

- 数据划分：train 2009–2020，valid 2021–2022，test 2023–2025（CSI300）
- 训练协议：两 stage 均 max 70 epoch，early stop patience 15；
  Stage 1 固定 seed 42，Stage 2 seed 0
- 回测协议：Top30/Drop5，open 0.0005 / close 0.0015，min_cost 0，
  close 成交，CN limit=None
- 关键配置：quantizer hvq，num_levels=2，level_num_embed=[256, 256]，
  embed_dim=128，distance=l2，commit_weight=0.25，contras_loss=True；
  Stage 2 MoE n_expert=2（k=1），use_prior=True，target_day=5

## Git

Base: main
Branch: exp/001-hvq-residual-2level
Commit: 0a75fd1（训练代码；其后 ab3e117 仅将 Stage 1 checkpoint 名填入 config，不影响训练行为）

## Smoke Test

Status: PASS（追溯登记；完整 stage1/stage2/backtest 流程已实际跑通）
Notes: Stage 1 训练至 epoch 20 触发 early stop，最佳 epoch 5
（val_loss=0.4592）；Stage 2 同样 epoch 20 early stop，最佳 epoch 5
（val_loss=1.0125）。两个 stage 的 val_loss 均为 total loss
（recon + vq + pred_weight·pred，见 `trainer/train_vqvae.py:70`）。

## Result

Status: DONE — Corrected Protocol seed0（test 区间 2023-01-01 – 2025-12-31）

协议：corrected Stage 2 freeze（encoder/quantizer/RevIN 全程 eval）+ corrected MDD（初始 NAV=1 计入 peak）；Stage 2 seed 0；回测 Top30/Drop5，open 0.0005 / close 0.0015，min_cost 0，close 成交，CN limit=None。

IC: 0.0352
ICIR: 0.2174
RankIC: 0.0506
RankICIR: 0.3171

Annual Return: 13.69%（基准 6.40%，超额 7.29%，AR 差值口径）
Sharpe: 0.7591
Sortino: 1.1405
MDD: -18.49%
Calmar: 0.7401
Turnover: 0.3293

Stage 1 checkpoint provenance：self-trained（seed 42）：`hvq_csi300_full-epoch=5-val_loss=0.4592.ckpt`，由项目级 `checkpoints/` 原样迁入 `artifacts/001/run/checkpoints/`（Stage 1 不受本轮修复影响，未重训）。
Stage 2 seed：0
产物：`artifacts/001/run/`（checkpoints/、res/、backtest、summary.json、stage1/stage2/backtest 日志）。

代码版本：实验代码 0a75fd1/ab3e117 + 公共 correctness fix ea11ab3（`train()` 覆写将冻结保护从 quantizer 扩展到 encoder/RevIN；不改变实验 Idea、结构、超参、数据或协议）。

### Historical（不再作为正式比较依据）

pre-fix seed0（quantizer-only freeze override + 旧 MDD 实现，旧产物在项目级 `res/` 与 `logs/`）：IC 0.0352 / ICIR 0.2174 / RankIC 0.0506 / RankICIR 0.3171 / AR 13.69% / Sharpe 0.7591 / MDD -18.49% / Calmar 0.7401 / Turnover 0.3293。

### Corrected 与 Historical 的关系

corrected 重跑与 historical 逐位一致。PL 2.6.4 的 validation loop 按子模块 capture/restore training mode，`freeze_vqvae()` 的 eval 设置在旧代码路径下也未被破坏，freeze bug 在本环境未实际触发；MDD 修复对本实验数值无影响（净值在最大回撤前已超过初始 NAV 1.0）。corrected 协议确认了旧数值有效。

## Conclusion

Corrected Protocol 下，Residual HVQ（z0+z1）相对 corrected PRISM-VQ baseline：
ranking 指标略低（IC 0.0352 vs 0.0373，RankIC 0.0506 vs 0.0552），
但组合层面明显更优（AR 13.69% vs 9.24%，Sharpe 0.7591 vs 0.5223，
MDD -18.49% vs -26.90%）。两级 HVQ 在 corrected protocol 下整体优于单层
baseline 的组合表现，ranking 指标上基本持平略弱。
注意：001 的 Stage 1 是 Residual HVQ 256+256（本仓库自训，val_loss 0.4592），
baseline 的 Stage 1 是 single VQ512（PRISM-VQ 原始 checkpoint epoch=7，
val_loss 0.5712）——Stage 1 架构不同属于本实验的实验设计本身，并非
"同架构不同训练实例"的混淆；但两者的 Stage 1 为各自独立训练的
checkpoint，跨架构比较的数值同时包含实现与训练实例差异。
