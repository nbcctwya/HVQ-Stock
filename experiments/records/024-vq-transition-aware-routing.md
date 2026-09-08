# 024 — vq-transition-aware-routing

## Idea

在原始 corrected PRISM-VQ baseline 的 MoE clean routing logits 上加入由
最近 VQ latent-state transition trajectory 驱动的 additive bias。固定
code-history length `L = 5`：当前 code `k_t` 加最多 4 个严格早于 `t` 的
同 instrument 历史 code，仅建模相邻 frozen prototype 的差分：

```text
e_k            = frozen_codebook[k].detach()
delta_z_j      = e_{k_j} - e_{k_{j-1}}
transition_state = GRU(delta_z sequence)   # 单层 GRU, input=128, hidden=64
transition_bias  = W_t(transition_state)   # W_t: R^64 -> R^n_expert, zero-init
clean_logits     = Router(z_q,t) + transition_bias
```

## Motivation

原 PRISM-VQ 仅建模 `P(expert | current latent state)`。相同当前离散状态
可能来自完全不同的演化路径（`37->37->37->37->37` 稳定驻留 vs
`12->18->25->31->37` 快速迁移），历史 latent-state transition 可能包含
当前 `z_q` 无法表达的 state dynamics。本实验验证在已知当前 `z_q,t` 的
条件下，历史离散状态迁移轨迹是否仍包含能改善 expert routing 的增量
信息，即 `P(expert | current state, transition history)` 是否优于仅依据
当前 state 的 routing。

## Modification

- 新增 `module/code_history.py`：用 exact frozen Stage 1（RevIN ->
  SpatialEncoder -> VectorQuantiser，eval + no_grad，只读 stock feature
  slice，绝不读 label）对 train/valid/test 各 split 的每个样本按位置序
  计算 `vq_idx`，并基于真实 MultiIndex 构造 identity-keyed 历史表
  `hist_codes [N,4]`（oldest-first）与 `hist_len [N]`。逐 instrument 按
  datetime 升序 + `searchsorted(side="left")` 严格保证
  `history_datetime < current_datetime`；train 只读更早 train，valid 可读
  更早 train+valid，test 可读更早 train+valid+test；duplicate key /
  malformed index / split chronology 破坏 / stale cache 全部 loud fail；
  历史不足的样本不删除（`hist_len=0` 时 transition state 严格为 0）。
- `TransitionHistoryDataset` 将历史绑定到 dataset position（而非 batch
  顺序），`init_data_loader(..., history=...)` 产出
  `(hist_codes, hist_len, batch)` 三元组，训练日期 shuffle 不改变任何
  样本的历史序列（有 shuffle-invariance 测试）。
- 新增 `module/transition.py` 的 `VQTransitionEncoder`：prototype lookup
  直接引用 frozen quantizer codebook 并显式 detach（经 `__dict__` 持有
  引用，避免被注册成新参数）；不新增 `nn.Embedding`；只编码相邻
  prototype difference，含显式的 last-history→current transition；
  padding 经 mask + packed sequence 排除，绝不进入 GRU；GRU 构造在
  `torch.random.fork_rng` 下完成，base 既有参数初始化逐位不变。
- 当前 `k_t` 永远来自当前 forward 的 quantizer 实时 `vq_idx`
  （detach + 范围校验），不使用 cache。
- `FactorGatedMoE.clean_routing_logits` 增加可选 `transition_bias` 加性
  输入，在 noise 注入与 top-k 之前加入 clean logits；noise network、
  `W_h`、top-k、softmax、SparseDispatcher、experts、load-balancing loss
  全部不变。
- `GenerateReturn` 在 frozen Stage 1 加载后创建该分支；默认
  `configs/config.yaml` 设置 `predictor.transition_aware_routing: true`
  （及固定 `transition_history_len: 4`、`transition_gru_hidden: 64`），
  无需实验特有 CLI override。
- `stage2.py` 在模型构建后、训练前构建（或校验并加载 provenance-keyed
  cache）code history 并重建三个 loader；`utils/test.py` 的
  `run_inference` 解包三元组 batch。新增
  `tests/test_vq_transition_routing.py`（40 个测试）与
  `scripts/smoke_vq_transition_routing.py`；Stage 1 freeze 回归测试扩展
  到 transition 分支。

## Constraints

- 唯一实验变量是新增上述 VQ transition-aware additive routing bias；
  研究的是 transition history 的增量价值，不是泛化 temporal feature
  encoder。
- `L = 5` 固定；Transition Encoder 固定为单层单向 GRU（input=128，
  hidden=64）；不使用 attention / Transformer / bidirectional GRU。
- transition branch 只影响 MoE clean routing logits：不进入 expert
  input、Temporal Transformer、HyperFusion hidden representation、
  factor heads、LatentValueHead、ReturnPredictor、alpha/beta 或最终
  预测；不新增 transition-specific auxiliary loss；不重新学习 code
  embedding。
- `W_t` weight/bias zero-init：初始时 `transition_bias == 0`，routing、
  原 auxiliary loss 与完整 prediction forward 与 base 逐位一致；相同
  seed 下除 `transition_encoder.*` 外全部既有参数初始化逐位一致。
- 不使用 label、future return、未来 code 或未来日期信息构造 history；
  不因历史不足删除样本或改变数据划分。
- Stage 1 完全冻结：Encoder、RevIN、Quantizer、codebook、VQ assignment
  与 Stage 1 loss 不变，无梯度、始终 eval。
- 不改变 daily batching、train-date shuffle、训练预算（70 epoch）、
  early stop、optimizer、lr、seed（Stage 2 seed 0）、预测与回测协议；
  不加入 Shared Expert、adaptive fusion、decoupling、quantization
  confidence、market-conditioned routing、prior-latent allocation、
  code-aware bias、continuous residual correction 或其他已有机制。
- Phase 1 未启动正式长时间训练或正式回测。

## Git

Base: main
Branch: exp/024-vq-transition-aware-routing
Commit: da989e74400b2ebed792f79efda285d329432514
Stage 1 provenance: external corrected PRISM-VQ exact checkpoint：
`artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`
（14,584,929 bytes；MD5 `6b9d9dbfd938c7bd2c7dc5ee33cb38af`）。

## Smoke Test

Status: PASS

Notes:

- `conda run -n prism-vq python -m unittest discover -s tests`：完整测试
  121/121 PASS（024 新增 40 个：history/leakage、code provenance、
  transition semantics、baseline equivalence、compatibility），既有
  Stage 1 freeze 回归测试在 transition 分支启用下继续 PASS。
- external checkpoint 存在且非空（bytes/MD5 校验通过）；唯一
  `VectorQuantiser`、codebook `(512, 128)`；Encoder/Quantizer/RevIN
  strict load missing=0 / unexpected=0；数据划分 2009–2020 / 2021–2022 /
  2023–2025 不变。
- 历史因果逐样本校验通过；shuffle 前后 `(data row -> history)` 映射完全
  一致；label poison（NaN）后 code map 逐位相同；当前 code 经 spy 验证
  来自 live quantizer；codebook lookup 无梯度、无额外 learnable
  embedding；padding 不进入 GRU；`hist_len=0` 输出严格零状态。
- zero-init 时 clean logits、gates/load、MoE output/aux loss、完整
  forward 与 base 逐位相等；既有参数初始化逐位相等；noisy top-k /
  noise / `W_h` / load 行为逐位不变。非零 `W_t` 后不同 history 在固定
  `z_q` 下改变 routing 与 expert allocation。
- 真实 backward + optimizer step：`W_t` grad L1 = 0.01361 并更新，首次
  更新后 GRU grad L1 = 2.89e-05；Stage 1 参数无梯度、codebook 不变；
  GPU deterministic backward（cuDNN GRU + 全模型）PASS。
- Stage 2 checkpoint strict round-trip 输出逐位一致；标准 inference、
  12 行 prediction、metric CSV 与 backtest prediction normalizer PASS。
- 真实 canonical 数据（train 811,199 / valid 145,145 / test 217,909
  样本）历史构建成功，诊断：history length=4 占比约 99.7%；相邻 code
  change rate ≈ 0.728–0.738；zero-transition 比例 ≈ 0.261–0.272；有效
  transition sequence 数量 810,493 / 145,070 / 217,876。
- 产物位于 `artifacts/024/smoke/`：`unit_tests.log`、`smoke.log`、
  `smoke_report.json`、`checkpoints/`（smoke ckpt + provenance-keyed
  code-history cache）与 `res/`。

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
