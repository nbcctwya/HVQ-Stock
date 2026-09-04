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

Status: DONE（test 区间 2023-01-03 – 2025-12-31，727 个交易日）

IC: 0.0352
ICIR: 0.2174
RankIC: 0.0506
RankICIR: 0.3171

Annual Return: 13.69%（基准 6.40%，超额 6.34%）
Sharpe: 0.7591
Sortino: 1.1405
MDD: -18.49%
Calmar: 0.7401
Turnover: 0.3293

## Conclusion

残差式 2 级 HVQ 端到端跑通，效果与 PRISM-VQ 单层 512 codebook 基本持平：
IC 0.0352 vs 0.0354 持平，ICIR 0.2174 vs 0.2042 略胜，
RankIC/RankICIR 略低（0.0506/0.3171 vs 0.0567/0.3293，同 seed 0 对照）。
Stage 1 最佳 val_loss（total）0.459 明显低于单层基线 0.568，但因
total loss 中 vq 项规模不同，该对比仅供参考，不代表重构质量提升。

产物：
- Stage 1 checkpoint：`checkpoints/hvq_csi300_full-epoch=5-val_loss=0.4592.ckpt`
- 预测：`res/VQ512_csi300_mo2_k1_mh64_md0.1_dm64_nh2_l1_d0.1_au0.01_1h2_1e128_1d0.1_1l2_p20_ai3_ks3/0_best.pkl`
- 回测：同目录 `backtest/seed0_top30_drop5/`
- 日志：`logs/stage1_hvq_full.log`、`logs/stage2_hvq_seed0.log`、`logs/backtest_hvq_seed0.log`

备注：回测口径为 PRISM 原生（CN 不限涨跌停），与 fusion_analysis 统一
协议（limit=0.095）不直接可比；单 seed 噪声大，后续应补 seed 1–4，
并观察 per-level perplexity 判断各级 codebook 利用率。
