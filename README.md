# 036 — combine-025-017-shared-routed-decoupling

## Base

`exp/025-combine-010-019-shared-routed-quant-confidence`（PRISM-VQ + Shared-Routed
MoE + Quantization Confidence Adapter）。

本分支直接从冻结的 025 实验分支创建，完整保留 025 的 Shared-Routed MoE 与
Quantization Confidence Adapter，仅在其上加入实验 017 的 Shared–Routed
Decoupling Regularization。

Stage 1 不重新训练，沿用 025 的正式 Stage 1 provenance：复用实验 010 的
Stage 1（`artifacts/010/run/.stage1.done`，`reused=true`），其指向 corrected
PRISM-VQ exact checkpoint：

`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`

该文件为 14,584,929 bytes，MD5 `6b9d9dbfd938c7bd2c7dc5ee33cb38af`。marker 中
记录的 source commit 与 canonical queue 中 010 的 pinned commit 完全一致。
SpatialEncoder、Quantizer 与 RevIN 均 strict 加载验证（missing=0、
unexpected=0）；模型保持 single VQ512、128 维 embedding 与原数据划分。

## Idea / Motivation

025 表明：019 的 quantization-confidence latent correction
（`z_conf = z_q + Linear(1, 128)(q_error)`）可以在 010 的 Shared-Routed
Stage 2 之上提供 latent reliability signal。

017 表明：Shared Expert 与 Routed Experts 的表示可能冗余；按原始定义

```text
L_dec = mean(cosine_similarity(shared_out, routed_out)^2)
L_moe = L_route + 0.01 * L_dec
```

对两条路径做 squared-cosine 解耦，可促使 Shared Expert 学习真正的公共结构、
Routed Experts 学习专门化结构。

若两种机制互补（confidence correction 改善 latent 输入质量，decoupling 改善
expert 分工），则在 025 之上加入 017 的 decoupling regularizer 应进一步改善
预测或投资组合表现。

## 核心修改

唯一实验变量：在 025 上加入 017 的 Shared–Routed Decoupling
Regularization，严格按 017 原始定义复用：

- `module/layers/moe.py`：`FactorGatedMoE` 新增 `decoupling_lambda` 参数
  （非负校验；`lambda > 0` 时要求 `use_shared_expert=True`），新增静态方法
  `shared_routed_decoupling_loss(shared_out, routed_out)`（float32
  per-sample cosine，square 后取 mean）；forward 中当
  `decoupling_lambda > 0` 时
  `loss = route_loss + decoupling_lambda * decoupling_loss`，否则保持原
  `route_loss`。
- `module/layers/fusion.py`：`HyperFusion` 透传 `decoupling_lambda` 到 MoE。
- `module/bidirectional.py`：`LoadingGenerator` 从
  `config['predictor'].get('decoupling_lambda', 0.0)` 读取并透传。
- `configs/config.yaml`：默认设置 `predictor.decoupling_lambda: 0.01`，与
  `shared_expert: true`、`quantization_confidence_adapter: true` 同时生效，
  核心改动无需任何 CLI override。
- decoupling 只改变 auxiliary loss，不改变 prediction forward：不引入任何新
  的可训练模块或参数。

## 与 base（025）的区别

唯一区别是上述 017 decoupling regularizer（`decoupling_lambda: 0.01`）。

保持不变的 025/010/019 条件：Quantization Confidence Adapter
（`Linear(1, 128)` 全零初始化、`q_error` detach 定义、`z_conf` 的全部使用
位置）、always-on Shared Expert（不参与 routing、不占 top-k quota）、2 个
Routed Experts、top-k = 1、原 router、原 noise network、原 `W_h`、原
SparseDispatcher、原 expert combine、原 importance/load-balancing
auxiliary loss。不采用 018 的 top-level decoupling 形式，不改变 prediction
forward。Stage 1 的 RevIN、SpatialEncoder、single VQ512、128 维 codebook、
assignment 与训练逻辑不变；不加入 adaptive shared fusion 或其他新机制，
不进行超参数搜索。

数据划分（train 2009–2020、valid 2021–2022、test 2023–2025）、70 epoch
预算、early stopping、Stage 1 seed 42、Stage 2 seed 0、Top30/Drop5 回测
协议及其他超参数均与 025 一致。

## Smoke 状态

Status: **PASS**。

- `conda run -n prism-vq python -m unittest discover -s tests -v`：完整测试
  131/131 PASS；新增组合机制测试 15/15 PASS（017 公式精确性、loss 组合
  `L_moe = L_route + 0.01 * L_dec`、prediction forward 与 025 逐位等价、
  参数初始化与 025 逐位一致、decoupling 不新增模块、adapter 与 shared
  expert 同步可训练、Stage 1 冻结、strict checkpoint round-trip），既有
  025/010 机制测试与公共回归测试继续 PASS。
- `scripts/smoke_combine_025_017_shared_routed_decoupling.py`：PASS。覆盖
  010 Stage 1 provenance（marker commit 与 queue pinned commit 一致、
  checkpoint 存在且 MD5 不变）、single VQ512 与数据划分、strict load、
  adapter zero-init、`q_error` 精确定义、初始 forward 对 010 逐位等价、
  既有参数初始化不受扰动、017 探针（非零 Shared Expert 下 prediction
  逐位不变、route loss 逐位不变、`L_moe = L_route + 0.01 * L_dec` 闭式
  成立）、decoupling 对 shared/routed 两条路径均产生有限非零梯度、
  adapter 非零梯度与参数更新、Stage 1 梯度隔离与 codebook 不变、quantizer
  assignment 不变、Stage 2 checkpoint strict round-trip、标准 prediction
  及 backtest normalizer。
- smoke 诊断（一步训练后）：decoupling penalty `0.03862`，shared/routed
  cosine mean `-0.1959`；`q_error` mean `0.01176`；relative correction
  magnitude `4.59e-4`；adapter weight/bias grad L1 `0.2255` / `17.58`；
  Shared Expert final weight/bias grad L1 `91.10` / `19.12`；Routed Experts
  grad L1 `301.53`。诊断仅用于观测，未修改训练目标。
- 产物位于 `artifacts/036/smoke/`：`unit_tests.log`、`stage2.log`、
  `smoke_report.json`、`checkpoints/` 与 `res/`。

本阶段未启动正式长时间训练或正式回测。
