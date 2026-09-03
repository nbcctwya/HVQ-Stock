# H003 — z0-only

- **类型**：消融实验（训练 Stage 2，不训练 Stage 1）
- **假设**：Residual VQ 的 Level 1 可能提升了 reconstruction，但引入的是
  prediction-irrelevant information（对收益预测无帮助的细粒度重构信息）。
  如果 Stage 2 只用 z0（第 0 级量化表示）与 z0+z1 效果相当甚至更好，说明
  Level 1 对预测是冗余甚至有害的。

## 要求

1. **复用** H001 的 HVQ Stage 1 checkpoint（绝对路径只读引用）：
   `checkpoints/hvq_csi300_full-epoch=5-val_loss=0.4592.ckpt`
   **禁止重新训练 Stage 1。**
2. Stage 2 的预测器输入**只使用 z0**（第 0 级量化输出），丢弃 z1。
   - 实现方式建议：给 `ResidualVectorQuantiser` 加一个推理期开关（如
     `vqvae.quantizer.active_levels: [0]` 或 `z0_only: true`），默认关闭，
     不改变现有 `hvq` 默认行为；**不改动** `single` 路径。
   - 注意保持 checkpoint 参数加载兼容（只改 forward 取值，不改参数结构）。
3. `train.seed=0`（**只有 seed 0**，不要多 seed）。
4. 数据划分、Stage 2 超参（MoE n_expert=2, aux_weight=0.01, target_day=5
   等）与 H001 完全一致——除了 z0-only 这一个变量，其他一律不动。
5. 训练完成后用 `backtest_qlib.py` 回测，口径与 H001 完全一致
   （Top30/Drop5，open 0.0005 / close 0.0015，min_cost 0，close 成交）。

## 交付

- 代码改动 + 单元测试（验证 z0-only 开关生效：`z_q == z_q0`，且默认行为
  不变）。
- `runner/jobs/H003/run.sh`：Stage 2（seed 0）→ 回测；产物与日志路径全部
  带 `H003` 标识，不得覆盖 H001 的任何产物。
  - 预测 pkl 与回测目录需与其他实验隔离（用 `--output_dir` 显式指定到
    带 H003 的目录，或确认默认命名不与 H001 冲突）。
  - 小型指标汇总写到 `results/H003/metrics.json`（IC/ICIR/RankIC/RankICIR
    与回测各指标）。
- `runner/jobs/H003/manifest.json`：`requires_stage1: false`，
  `stage1_checkpoint` 填复用的绝对路径。
- 日志：`runner/logs/H003/stage2_seed0.log`、`runner/logs/H003/backtest_seed0.log`。

## 验收标准（Review 用）

- diff 中只有 z0-only 开关及其接线，无其他实验变量改动。
- Stage 2 seed 0 正常完成；加载的 Stage 1 checkpoint 与 H001 同一文件。
- 数据划分与回测协议与 H001 一致。
- Review 必须给出结论：z0-only vs z0+z1（H001 seed 0）的 IC/ICIR/RankIC/
  RankICIR 与回测指标对比，回答「Level 1 对预测是否有帮助」。
