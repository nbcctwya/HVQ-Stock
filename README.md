# 025 — combine-010-019-shared-routed-quant-confidence

## Base

`exp/010-prism-shared-routed-moe`（PRISM-VQ + Shared-Routed MoE）。

本分支直接从冻结的 010 实验分支创建，不从 main 或 019 重新拼装模型；019 的
mechanism 以最小必要改动移植到 010 之上。

Stage 1 不重新训练，复用实验 010 的正式 Stage 1 provenance
（`artifacts/010/run/.stage1.done`，`reused=true`），其指向 corrected
PRISM-VQ exact checkpoint：

`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`

该文件为 14,584,929 bytes，MD5 `6b9d9dbfd938c7bd2c7dc5ee33cb38af`。marker 中
记录的 source commit 与 canonical queue 中 010 的 pinned commit 完全一致。
SpatialEncoder、Quantizer 与 RevIN 均 strict 加载验证（missing=0、
unexpected=0）；模型保持 single VQ512、128 维 embedding 与原数据划分。

## Idea / Motivation

010 已证明：把 Stage 2 MoE 显式拆成 always-on Shared Expert（学习跨 latent
state 的公共结构）与 Routed Experts（学习 `z_q` 区分的专门化结构），即

```text
moe_out = E_shared(h) + sum_j g_j(z_q) E_j(h)
```

能稳定改善 PRISM-VQ。

019 表明：hard vector quantization 丢弃的 quantization-error information
可能提供弱但正向的 latent reliability signal：

```text
q_error = mean((h - z_q)^2, dim=-1, keepdim=True)
z_conf  = z_q + Linear(1, 128)(q_error)
```

若两种机制互补，则在 010 的 Shared-Routed Stage 2 之前加入 019 的
confidence-aware latent correction，应进一步改善预测或投资组合表现。

## 核心修改

025 保留 010 的全部 Shared-Routed 结构，只在 Stage 1 输出与 010 Stage 2
之间插入 019 的 Quantization Confidence Adapter：

- `GenerateReturn` 在所有 010 模块构造完成后新增且仅新增一个
  `Linear(1, 128)` confidence adapter，weight 与 bias 显式全零初始化；
  相同 seed 下，除新增 adapter 外全部既有 state tensor 初始化与 010 逐位一致。
- 用 detach 后的 `h` 与 `z_q` 计算 per-sample scalar `q_error`（019 公式
  不变，confidence 路径不向 Stage 1 反向传播），构造
  `z_conf = z_q.detach() + adapter(q_error)`。
- 010 中原本进入 Stage 2 的 `z_q` 统一替换为 `z_conf`，作用范围完整沿用
  019：
  1. Temporal Transformer 的 latent token / conditioning；
  2. HyperFusion 的 latent input；
  3. 010 Routed Experts 的 routing latent（经 HyperFusion 内部 `norm_z`）；
  4. 通过 confidence-aware Temporal Transformer representation 间接影响
     Shared Expert 的输入；
  5. LatentValueHead 的输入。
- Shared Expert 不直接读取 `q_error` 或 confidence scalar；025 不新增任何
  专门的 shared-confidence interaction。
- zero-init 保证初始 `z_conf == z_q`，完整 prediction forward、original
  auxiliary loss 与 010 逐位相等。
- 标准 forward 五元组接口保持兼容，第四项为实际进入 Stage 2 的 latent
  （`z_conf`）；raw `z_q` / `q_error` 通过内部 helper 诊断，不影响标准
  inference interface。
- 默认 `configs/config.yaml` 同时设置 `predictor.shared_expert: true` 与
  `predictor.quantization_confidence_adapter: true`，核心改动无需 CLI
  override。

## 与 base（010）的区别

唯一实验变量是新增上述 quantization-confidence latent correction。

保持不变的 010 条件：always-on Shared Expert（不参与 routing、不占 top-k
quota）、2 个 Routed Experts、top-k = 1、原 router、原 noise network、原
`W_h`、原 SparseDispatcher、原 expert combine、原 importance/load-balancing
auxiliary loss。Stage 1 的 RevIN、SpatialEncoder、single VQ512、128 维
codebook、assignment 与训练逻辑不变；HyperFusion 后续 FiLM、alpha/beta
heads、prior/latent decomposition、ReturnPredictor、prediction loss、
softcap、aux_weight 均不变。不加入 020–024 的 market routing、
prior-latent allocation、explicit code bias、continuous residual
correction、transition routing 等机制。

数据划分（train 2009–2020、valid 2021–2022、test 2023–2025）、70 epoch
预算、early stopping、Stage 1 seed 42、Stage 2 seed 0、Top30/Drop5 回测
协议及其他超参数均与 010 一致。

## Smoke 状态

Status: **PASS**。

- `conda run -n prism-vq python -m unittest discover -s tests -v`：完整测试
  116/116 PASS；新增组合机制测试 25/25 PASS，既有 010 机制测试与 Stage 1
  freeze 回归测试继续 PASS。
- `scripts/smoke_combine_010_019_shared_routed_quant_confidence.py`：PASS。
  覆盖 010 Stage 1 provenance（marker commit 与 queue pinned commit 一致、
  checkpoint 存在且 MD5 不变）、single VQ512 与数据划分、strict load、
  zero-init、`q_error` 精确定义、初始 `z_conf` 与完整 prediction forward
  对 010 逐位等价、既有参数初始化不受扰动、adapter 非零梯度与参数更新、
  Stage 1 梯度隔离与 codebook 不变、quantizer assignment 不变、Stage 2
  checkpoint strict round-trip、标准 prediction 及 backtest normalizer。
- smoke 诊断（一步训练后）：`q_error` mean `0.01176` / std `0.00423`
  （p50 `0.01085`，p05–p95 `0.00772`–`0.01834`）；`||z_conf - z_q||` 均值
  `0.001142`；relative correction magnitude `4.59e-4`；adapter weight/bias
  L2 norm 均约 `0.00113`；adapter weight/bias grad L1 `0.2255` / `17.58`；
  Shared Expert final weight/bias grad L1 `91.10` / `19.12`；Routed Experts
  grad L1 `301.53`。诊断仅用于观测，未修改训练目标。
- 产物位于 `artifacts/025/smoke/`：`unit_tests.log`、`stage2.log`、
  `smoke_report.json`、`checkpoints/` 与 `res/`。

本阶段未启动正式长时间训练或正式回测。
