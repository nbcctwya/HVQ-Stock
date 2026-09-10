# 036 — combine-025-017-shared-routed-decoupling

## Idea

严格以实验 025 `combine-010-019-shared-routed-quant-confidence` 为 base，
完整保留其 Shared-Routed MoE + Quantization Confidence Adapter，仅加入
实验 017 `shared-routed-decoupling` 的 Shared–Routed Decoupling
Regularization，保持 017 的原始定义：

```text
L_dec = mean(cosine_similarity(shared_out, routed_out)^2)
L_moe = L_route + 0.01 * L_dec
```

## Motivation

验证 019 的 quantization-confidence latent correction 与 017 的
Shared/Routed expert decoupling 是否具有互补性：confidence correction
改善进入 Stage 2 的 latent 质量，decoupling regularizer 促使 Shared
Expert 学习公共结构、Routed Experts 学习专门化结构；两者作用层面不同，
若互补则组合应优于任一单独机制。

## Modification

- 从冻结的 `exp/025-combine-010-019-shared-routed-quant-confidence`
  创建实验分支。
- `module/layers/moe.py`：`FactorGatedMoE` 新增 `decoupling_lambda`
  参数（非负校验；`lambda > 0` 要求 `use_shared_expert=True`）与静态方法
  `shared_routed_decoupling_loss`（float32 per-sample cosine，square 后
  mean）；forward 中 `lambda > 0` 时
  `loss = route_loss + lambda * decoupling_loss`，否则保持原
  `route_loss`。与 017 的实现逐行一致。
- `module/layers/fusion.py`：`HyperFusion` 透传 `decoupling_lambda`。
- `module/bidirectional.py`：`LoadingGenerator` 从
  `config['predictor'].get('decoupling_lambda', 0.0)` 读取并透传。
- 默认 `configs/config.yaml` 新增 `predictor.decoupling_lambda: 0.01`，
  与 025 的 `shared_expert: true`、
  `quantization_confidence_adapter: true` 同时生效；核心改动不依赖任何
  实验特有 CLI override。
- 新增 `tests/test_combine_025_017_shared_routed_decoupling.py` 与
  `scripts/smoke_combine_025_017_shared_routed_decoupling.py`；分支根目录
  README 改写为本实验说明。

## Constraints

- 唯一实验变量：在 025 上加入 017 的 Shared–Routed Decoupling
  （`decoupling_lambda: 0.01`）；除此之外全部与 025 保持一致。
- 017 的 decoupling 机制严格按原实验复用（MoE 内部
  shared_out/routed_out 的 squared-cosine penalty），不采用 018 的
  top-level decoupling 形式，不改变 prediction forward，不引入任何新的
  可训练模块或参数。
- 025 的 Quantization Confidence Adapter（zero-init `Linear(1, 128)`、
  `q_error = mean((h.detach() - z_q.detach())^2, dim=-1, keepdim=True)`、
  `z_conf = z_q.detach() + adapter(q_error)` 及其全部使用位置）、Shared
  Expert（always-on、不参与 routing、不占 top-k quota）、2 个 Routed
  Experts、top-k = 1、原 router/noise network/W_h/SparseDispatcher/
  importance-load auxiliary loss 全部保持不变。
- Stage 1 来源：复用实验 010 的正式 Stage 1（`stage1_source: "010"`），
  沿用 025 的正式 Stage 1 provenance，不重新训练 Stage 1；结构与
  量化器配置、数据划分与 010/025 完全一致，RevIN/Encoder/Quantizer
  strict 加载（missing=0、unexpected=0）。
- 数据划分（train 2009–2020、valid 2021–2022、test 2023–2025）、70
  epoch 预算、early stopping、Stage 1 seed 42、Stage 2 seed 0、
  Top30/Drop5 回测协议及其他超参数均与 025 一致。
- 不加入 adaptive shared fusion 或其他新机制，不进行超参数搜索；
  Phase 1 未启动正式长时间训练或正式回测。

## Git

Base: exp/025-combine-010-019-shared-routed-quant-confidence
Branch: exp/036-combine-025-017-shared-routed-decoupling
Commit: c6bf8cef0056fec179489a5b7c1e0994e232f261
Stage 1 provenance: 复用实验 010 的正式 Stage 1（`stage1_source: "010"`）；
`artifacts/010/run/.stage1.done` marker 存在（`reused=true`），其记录的
commit `9b854f0436f8a7c3283fd375661dd6152cc965f1` 与 canonical queue 中
010 pinned commit 完全一致；marker 指向 corrected PRISM-VQ external exact
checkpoint
`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`
（14,584,929 bytes；MD5 `6b9d9dbfd938c7bd2c7dc5ee33cb38af`）。

## Smoke Test

Status: PASS

Notes:

- `conda run -n prism-vq python -m unittest discover -s tests -v`：全量
  131/131 PASS（既有 025/010 机制测试、公共回归测试与新增 036 组合测试
  15/15 全部 PASS）。
- `conda run -n prism-vq python scripts/smoke_combine_025_017_shared_routed_decoupling.py`：
  PASS。验证 010 Stage 1 provenance：marker 存在、marker commit 与 010
  Final Experiment Commit 一致、checkpoint 存在且 14,584,929 bytes、MD5
  不变；Encoder、Quantizer、RevIN strict load（missing=0 /
  unexpected=0）；single VQ512、128 维 embedding 与原数据划分。
- 017 机制精确复用验证：非零 Shared Expert 探针下 prediction 与
  `lambda=0` 逐位一致、route loss 逐位不变、
  `L_moe = L_route + 0.01 * L_dec` 闭式成立；decoupling loss 对
  shared/routed 两条路径均产生有限非零梯度；decoupling 不新增任何模块
  或参数，既有参数初始化与 025 逐位一致。
- 025 机制保持验证：adapter zero-init、`q_error` 公式精确、初始
  `z_conf` 与完整 forward 对 010 逐位等价；一步真实训练中 adapter 与
  Shared/Routed Experts 均获得有效梯度并更新，Stage 1 无梯度、保持
  eval、codebook 与 quantizer assignment 不变。Stage 2 checkpoint
  strict round-trip 后输出逐位一致；标准 prediction、metric CSV 与
  backtest prediction normalizer 均 PASS。
- 产物位于 `artifacts/036/smoke/`：`unit_tests.log`、`stage2.log`、
  `smoke_report.json`、`checkpoints/` 与 `res/`。

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
