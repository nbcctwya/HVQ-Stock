# 022 — code-aware-routing

## Idea

在原始 corrected PRISM-VQ baseline 上加入 Explicit Code-Aware Routing：
MoE router 除了读取连续的 quantized embedding `z_q` 外，还显式利用该样本
的离散 VQ code identity `vq_idx`。新增 code-specific expert preference
table `B ∈ R^(K × n_expert)`（`K = vqvae.num_embed = 512`），对样本
code ID `k = vq_idx`：

```text
code_bias = B[k]
clean_logits = Router(z_q) + code_bias
```

随后原 noisy gating、noise network、`W_h`、top-k、softmax、
SparseDispatcher、load-balancing loss 等流程全部保持不变。`B` 显式全零
初始化，因此初始化时完整 routing、auxiliary loss 与 prediction forward
与原 PRISM-VQ 严格等价。

## Motivation

虽然 `z_q` 已编码对应 prototype 的连续表示，但离散 code identity 本身
可能携带稳定的 expert-specialization prior。显式学习
`P(expert | code_id)` 的 additive routing bias，可能帮助不同 discrete
latent states 建立更稳定的 expert preference，而无需每次完全依赖连续
router 从 `z_q` 重新推断。

## Modification

- `FactorGatedMoE` 新增且仅新增一个 code-bias 表
  `loadings.fusion.moe.code_bias = nn.Parameter(torch.zeros(num_codes,
  num_experts))`，维度由 `vqvae.num_embed` 与 `predictor.n_expert` 决定
  （默认配置 `[512, 2]`），显式全零初始化；`torch.zeros` 不消耗 RNG，
  新增参数的构造不扰动任何 base 既有参数初始化。
- `FactorGatedMoE.clean_routing_logits(x, vq_idx)`：`gate(x)` 之后加上
  `code_bias[vq_idx]`；`vq_idx` 校验整数 dtype、shape `[B]` 与范围
  `[0, K-1]`（非法即 raise），并 `detach().long()` 作为离散索引，不存在
  通向 Stage 1 的梯度路径。
- `vq_idx` 直接复用冻结 Stage 1 quantizer 已返回的 `encoding_indices`，
  经 `GenerateReturn.forward -> LoadingGenerator -> HyperFusion ->
  FactorGatedMoE.noisy_top_k_gating` 传递；quantization assignment、
  codebook、Stage 1 loss 与训练逻辑完全不变。
- 默认 `configs/config.yaml` 设置 `predictor.code_aware_routing: true`，
  无需实验特有 CLI override。
- 新增 `tests/test_code_aware_routing.py` 与
  `scripts/smoke_code_aware_routing.py`；`tests/test_stage2_freeze.py`
  的 tiny config 启用该机制以覆盖实验路径。

## Constraints

- 唯一实验变量是上述 code-ID-specific additive routing bias；除此之外
  一律与 `main` 原始 PRISM-VQ baseline 保持一致。
- code bias 仅作用于 MoE clean routing logits，不进入 expert input、
  Temporal Transformer、HyperFusion projections、factor heads、
  LatentValueHead、ReturnPredictor 或其他 prediction 路径。
- 原 router、noise network、`W_h`、top-k、SparseDispatcher 与
  load-balancing loss 的定义和权重完全不变。
- 不加入 Shared Expert、adaptive fusion、decoupling、quantization
  confidence、market-conditioned routing、prior-latent allocation 或
  其他机制。
- Stage 1 的 SpatialEncoder、RevIN、single VQ512 quantizer、128 维
  codebook 完全冻结且不接收梯度；Stage 1 seed 42、Stage 2 seed 0、数据
  划分（train 2009–2020、valid 2021–2022、test 2023–2025）、70 epoch
  预算、early stopping、Top30/Drop5 回测协议及其他超参数均与 `main`
  一致。
- Stage 1 来源为 external corrected PRISM-VQ exact checkpoint；Phase 1
  未启动正式长时间训练或正式回测。

## Git

Base: main
Branch: exp/022-code-aware-routing
Commit: ea379a4981f868bea27733f06ca305126f9040d6
Stage 1 provenance: external corrected PRISM-VQ exact checkpoint：
`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`
（14,584,929 bytes；MD5 `6b9d9dbfd938c7bd2c7dc5ee33cb38af`）。

## Smoke Test

Status: PASS

Notes:

- `conda run -n prism-vq python -m unittest discover -s tests -v`：完整测试
  92/92 PASS；022 新增机制测试 10/10 PASS，既有 Stage 1 freeze、runner、
  protocol/backtest 回归测试继续 PASS。
- external checkpoint 存在且非空（14,584,929 bytes，MD5 匹配）；模型保持
  唯一 `VectorQuantiser`、codebook shape `(512, 128)` 与原数据划分。
  Encoder、Quantizer、RevIN strict load 均 missing=0 / unexpected=0。
- 单元测试与 smoke 均确认 code-bias 表为显式全零初始化的
  `[num_embed, n_expert]` `nn.Parameter`；zero-init 时 clean logits、
  routing/load、MoE 输出与 aux loss、完整 prediction forward 与 base
  逐位一致；相同 seed 下除新增表外全部既有 state tensor 逐位一致。
- `vq_idx` 从 quantizer 到 MoE router 的传递逐位匹配；非零 bias 下相同
  `z_q` 配合不同 code ID 产生不同 routing logits 与 expert 分配；非法
  code ID（越界、错误 shape、float dtype）明确报错；code ID 不进入
  Temporal Transformer、LatentValueHead、ReturnPredictor 等其他路径。
- 原 noisy top-k、noise、`W_h`、load 行为在相同噪声 seed 下逐位一致。
- synthetic canonical `[N,20,244]` smoke 完成真实 backward 与 optimizer
  step；code-bias gradient L1 为 `0.0121509153` 并成功更新，Stage 1 参数
  无梯度且始终保持 eval。
- Stage 2 checkpoint strict save/load 后完整输出逐位一致；标准 12 行
  prediction、metric CSV 与 backtest prediction normalizer 均 PASS。
- 产物位于 `artifacts/022/smoke/`：`unit_tests.log`、`stage2.log`、
  `smoke_report.json`、`checkpoints/` 与 `res/`。

## Result

Status: DONE（test 区间 2023-01-01 – 2025-12-31）

IC: 0.0370
ICIR: 0.2200
RankIC: 0.0550
RankICIR: 0.3309

Annual Return: 12.25%（基准 6.40%，超额 5.85%）
Sharpe: 0.6837
Sortino: 1.0258
MDD: -26.76%
Calmar: 0.4578
Turnover: 0.3292

## Conclusion

Phase 2 固定执行器完成正式训练、预测与回测（pinned commit ea379a4981f868bea27733f06ca305126f9040d6）。Stage 1 复用外部 exact checkpoint：`/home/nbcctwya/baselines/masterVQ/HVQ-Stock/artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`（本实验未重新训练 Stage 1）；Stage 2 seed 0。

产物：`artifacts/022/run/`（checkpoints/、res/、stage1.log、stage2.log、backtest.log、summary.json）。
