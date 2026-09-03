# 0001-hvq-residual-2level-csi300-seed0

- **日期**：2026-09-03
- **Git 版本**：训练代码对应 `0a75fd1`（Bootstrap HVQ-Stock from PRISM-VQ with residual hierarchical VQ）；训练后仅 `ab3e117` 将 Stage 1 checkpoint 名填入 config，不影响训练行为
- **市场 / Seed**：CSI300 / seed 0（Stage 1 固定 seed 42）
- **实验目的**：HVQ 首次完整实验，验证残差式 2 级 VQ 在 CSI300 上的端到端效果

## 创新点 / 魔改点

- 新增 `module/quantise_hvq.py::ResidualVectorQuantiser`：残差式 2 级量化（256+256 codebook），第 1 级量化第 0 级残差，`z_q = z_q0 + z_q1`，整体 STE；每级复用上游 `VectorQuantiser`（含 dead-code 重初始化与对比损失）
- `create_quantizer` 工厂接入 Stage 1/2，`vqvae.quantizer.type: single|hvq` 开关，`single` 为上游原行为
- 修复上游 Stage 2 codebook 冻结漏洞：`GenerateReturn.train()` 强制 quantizer 保持 eval，防止 training mode 下 `.data` 改写 codebook
- 数据 pickle 通过 `data.pickle_dir` 复用 `../PRISM-VQ/dataset/data`，未重复生成

## 关键配置

- quantizer：hvq，num_levels=2，level_num_embed=[256, 256]，embed_dim=128，distance=l2，commit_weight=0.25，contras_loss=True
- Stage 1：max 70 epoch，early stop patience 15；训练至 epoch 20 触发 early stop，最佳 checkpoint 在 epoch 5（val_loss=0.4592）
- Stage 2：max 70 epoch，early stop patience 15；训练至 epoch 20 触发 early stop，最佳 checkpoint 在 epoch 5（val_loss=1.0125）；MoE n_expert=2（k=1），use_prior=True，target_day=5

注：两个 stage 日志中的 `val_loss` 均为 **total validation loss**（`recon_loss + vq_loss + pred_weight * pred_loss`，见 `trainer/train_vqvae.py:70`），不是单独的 reconstruction loss；分项 val_recon_loss / val_vq_loss / val_pred_loss 只进 logger，未落控制台日志。

## 训练耗时（单 seed）

- Stage 1：约 55 分钟（21 个 epoch × 约 2.4 分钟，含数据加载）
- Stage 2：约 36 分钟（21 个 epoch × 约 1.7 分钟 + 数据加载 + valid/test 推理）
- 回测：约 1 分钟

## 预测指标（test，2023-01-03 – 2025-12-31，727 个交易日）

| IC | ICIR | RankIC | RankICIR |
|---|---|---|---|
| 0.0352 | 0.2174 | 0.0506 | 0.3171 |

对照（PRISM-VQ 单层 512 codebook，同 seed 0）：IC 0.0354 / ICIR 0.2042 / RankIC 0.0567 / RankICIR 0.3293。
即 IC 持平、ICIR 略胜、RankIC/RankICIR 略低；单 seed 噪声大，待补 seed 1–4。

## 回测（Top30/Drop5，open 0.0005 / close 0.0015，min_cost 0，close 成交，CN limit=None）

| 指标 | 组合 | 基准（SH000300） | 超额 |
|---|---|---|---|
| 年化收益 | 13.69% | 6.40% | 6.34% |
| 年化波动 | 16.90% | 17.02% | 9.64% |
| Sharpe | 0.7591 | 0.3644 | 0.6378 |
| Sortino | 1.1405 | 0.5492 | 0.9420 |
| MDD | -18.49% | -24.80% | -12.66% |
| Calmar | 0.7401 | 0.2579 | 0.5008 |
| 累计收益 | 44.78% | 19.59% | 19.40% |
| 换手率 | 0.3293 | — | 0.3293 |

## 产物路径

- Stage 1 checkpoint：`checkpoints/hvq_csi300_full-epoch=5-val_loss=0.4592.ckpt`
- Stage 2 预测：`res/VQ512_csi300_mo2_k1_mh64_md0.1_dm64_nh2_l1_d0.1_au0.01_1h2_1e128_1d0.1_1l2_p20_ai3_ks3/0_best.pkl`
- 回测目录：`res/VQ512_csi300_mo2_k1_mh64_md0.1_dm64_nh2_l1_d0.1_au0.01_1h2_1e128_1d0.1_1l2_p20_ai3_ks3/backtest/seed0_top30_drop5/`
- 日志：`logs/stage1_hvq_full.log`、`logs/stage2_hvq_seed0.log`、`logs/backtest_hvq_seed0.log`

## 备注

- Stage 1 最佳 val_loss（total）明显低于单层基线（0.459 vs 0.568），总损失改善未转化为 RankIC 增益；由于 total_loss 中 vq_loss 占比可观、两个量化器的 vq 项规模不同，该对比只能作为参考，不能解读为重构质量提升
- 回测口径为 PRISM 原生（CN 不限涨跌停），与 fusion_analysis 统一协议（limit=0.095）不直接可比
- 下一步：补跑 seed 1–4；跑 `type: single` 同仓库对照组；观察 per-level perplexity 判断各级 codebook 利用率
