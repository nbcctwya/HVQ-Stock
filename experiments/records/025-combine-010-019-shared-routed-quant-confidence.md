# 025 — combine-010-019-shared-routed-quant-confidence

## Idea

严格以实验 010 `prism-shared-routed-moe` 为 base，仅移植实验 019
`quantization-confidence-adapter` 的机制：在冻结 Stage 1 输出与 010 的
Shared-Routed Stage 2 之间插入 Quantization Confidence Adapter：

```text
q_error = mean((h - z_q)^2, dim=-1, keepdim=True)   # h / z_q 均 detach
z_conf  = z_q + Linear(1, 128)(q_error)             # weight / bias 显式 zero-init
```

随后 010 中原本进入 Stage 2 的 `z_q` 统一替换为 `z_conf`。

## Motivation

010 已证明显式分解 common structure（always-on Shared Expert）与
latent-specific routed structure（Routed Experts）能稳定改善 PRISM-VQ；
019 表明 hard quantization 丢弃的 quantization-error information 可能提供
弱但正向的 latent reliability signal。若二者作用机制互补，在 010
Shared-Routed Stage 2 前加入 confidence-aware latent correction 应进一步
改善预测或投资组合表现。zero-init 保证初始 `z_conf == z_q`，完整 forward
与 010 严格等价。

## Modification

- 从冻结的 `exp/010-prism-shared-routed-moe` 创建实验分支，不从 main 或
  019 重新拼装模型。
- `GenerateReturn` 在所有 010 模块构造完成后新增且仅新增一个
  `Linear(1, 128)` confidence adapter，weight / bias 显式全零初始化；
  相同 seed 下除新增 adapter 外全部既有 state tensor 初始化与 010 逐位一致。
- 用 detach 后的 `h` 与 `z_q` 计算 `q_error`（019 公式不变，confidence
  路径不向 Stage 1 反向传播），构造 `z_conf = z_q.detach() + adapter(q_error)`。
- `LoadingGenerator`（含 Temporal Transformer latent token、HyperFusion
  latent input、Routed Experts routing latent，并间接影响 Shared Expert
  输入）与 `LatentValueHead` 统一接收 `z_conf`；标准 forward 五元组第四项
  为 `z_conf`。
- 默认 `configs/config.yaml` 保持 `shared_expert: true` 并新增
  `quantization_confidence_adapter: true`，核心改动无需 CLI override。
- 新增 `tests/test_combine_010_019_shared_routed_quant_confidence.py` 与
  `scripts/smoke_combine_010_019_shared_routed_quant_confidence.py`；
  分支根目录 README 改写为本实验说明。

## Constraints

- 唯一实验变量：在 010 上新增并仅新增 019 的 Quantization Confidence
  Adapter。
- 019 mechanism 本身不变：不改成 `bias=False`，不加 LayerNorm / MLP / gate
  / clipping / normalization，不修改 quantization error 定义；adapter
  weight / bias 显式 zero-init；`q_error` 必须由 detached `h` 与 detached
  `z_q` 计算。
- `z_conf` 统一替换 010 Stage 2 全部 `z_q` consumer，不得只作用于 router；
  Shared Expert 不直接读取 `q_error`，不新增 shared-confidence interaction。
- 保留 010 Shared-Routed MoE 完整结构：always-on Shared Expert（不参与
  routing、不占 top-k quota）、2 个 Routed Experts、top-k = 1、原 router、
  原 noise network、原 `W_h`、原 SparseDispatcher、原 expert combine、原
  importance/load-balancing auxiliary loss；不加入 adaptive shared fusion
  或 decoupling。
- 不加入 020–024 的 market routing、prior-latent allocation、explicit
  code bias、continuous residual correction、transition routing 等机制。
- 不改变 HyperFusion 后续 FiLM、alpha/beta heads、prior/latent
  decomposition、ReturnPredictor、prediction loss、softcap、aux_weight
  或其他 objective。
- Stage 1 的 RevIN、SpatialEncoder、single VQ512、128 维 codebook、
  assignment 与训练逻辑全部保持 010 不变；Stage 1 来源为复用实验 010 的
  正式 Stage 1 provenance（其本身为 external corrected PRISM-VQ exact
  checkpoint）。
- 数据划分（train 2009–2020、valid 2021–2022、test 2023–2025）、70 epoch
  预算、early stopping、optimizer、learning rate、Stage 2 seed 0、Stage 1
  seed 42、Top30/Drop5 回测协议及其他超参数均与 010 一致。
- Phase 1 未启动正式长时间训练或正式回测。

## Git

Base: exp/010-prism-shared-routed-moe
Branch: exp/025-combine-010-019-shared-routed-quant-confidence
Commit: 5650e9fc9f3cf0ba89ac5e08b0965cc041ef4a2f
Stage 1 provenance: 复用实验 010 的正式 Stage 1（`stage1_source: "010"`）；
`artifacts/010/run/.stage1.done` marker 存在，`reused=true`，其记录的
source commit `9b854f0436f8a7c3283fd375661dd6152cc965f1` 与 canonical
queue 中 010 pinned commit 完全一致；marker 指向 corrected PRISM-VQ exact
checkpoint
`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`
（14,584,929 bytes；MD5 `6b9d9dbfd938c7bd2c7dc5ee33cb38af`）。

## Smoke Test

Status: PASS

Notes:

- `conda run -n prism-vq python -m unittest discover -s tests -v`：完整测试
  116/116 PASS；025 新增组合机制测试 25/25 PASS（010 inheritance、019
  mechanism fidelity、zero-init 对 010 的逐位等价、trainability、
  compatibility），既有 010 机制测试与 Stage 1 freeze 回归测试继续 PASS。
- smoke 验证 010 Stage 1 provenance：marker 存在、commit 匹配 queue pinned
  commit、checkpoint 存在非空且 MD5 不变；Encoder、Quantizer、RevIN strict
  load（missing=0 / unexpected=0）；single VQ512、128 维 embedding 与原数据
  划分完全兼容。
- zero-init 时 `z_conf` 与 `z_q`、025 与 010 的完整 prediction forward、
  original auxiliary loss 均逐位相等；相同 seed 下除新增 adapter 外全部
  既有参数初始化与 010 逐位一致。
- synthetic canonical smoke 完成真实 backward 与 optimizer step：adapter
  获得 finite non-zero 梯度并更新，Stage 1 参数无梯度、保持 eval、codebook
  与 quantizer assignment 不变；Shared Expert 与 Routed Experts 仍获得有效
  训练信号。Stage 2 checkpoint strict round-trip 后输出逐位一致；标准
  prediction、metric CSV 与 backtest prediction normalizer 均 PASS。
- smoke 诊断（一步训练后，仅观测不改训练目标）：`q_error` mean
  `0.01176` / std `0.00423`（p05/p25/p50/p75/p95 = `0.00772` / `0.00868` /
  `0.01085` / `0.01325` / `0.01834`）；`||z_conf - z_q||` 均值 `0.001142`；
  relative correction magnitude `4.59e-4`；adapter weight/bias L2 norm
  `0.001131` / `0.001131`；adapter weight/bias grad L1 `0.2255` / `17.58`；
  Shared Expert final weight/bias grad L1 `91.10` / `19.12`；Routed Experts
  grad L1 `301.53`。
- 产物位于 `artifacts/025/smoke/`：`unit_tests.log`、`stage2.log`、
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
