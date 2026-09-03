# H002 — Level-wise Diagnostics

- **类型**：诊断分析（不训练新模型）
- **假设背景**：H001（残差式 2 级 VQ）端到端指标与单层 baseline 基本持平，
  但总 val_loss 明显更低。需要弄清每一级量化器到底学到了什么、第 1 级
  codebook 是否被有效利用，再决定后续方向。
- **seed**：不涉及时序训练的随机性；统一用 seed 0 语境处理数据即可。

## 目标

对已有 HVQ Stage 1 checkpoint 做逐级别（per-level）量化诊断：

1. **per-level perplexity**：每一级 codebook 的 perplexity（exp entropy of
   usage distribution）。
2. **active code ratio**：每一级被实际使用过的 code 数 / codebook 大小。
3. **code utilization**：每一级 code 使用频率分布（top-k 占比、是否高度
   集中），可用直方图/分位数描述。
4. **residual norm before / after**：每一级量化前后残差范数
   （`||r_l||` 与 `||r_{l+1}|| = ||r_l - z_ql||`）的均值/分位数。
5. **quantization error**：整体 `||h - z_q||² / ||h||²`（relative quantization
   error），以及只到第 0 级时的同样指标作为对照。

## 约束

- **必须复用**主仓库现有 Stage 1 checkpoint：
  `checkpoints/hvq_csi300_full-epoch=5-val_loss=0.4592.ckpt`
  （worktree 内用绝对路径只读引用）。
- **不重新训练 Stage 1，不训练 Stage 2，不做回测**。
- 在 validation 集上统计即可（如成本可忽略可附 test 集，需在记录里注明）。
- 数据划分与 H001 完全一致（csi300，window 20，valid 2021–2022）。
- 诊断脚本建议放 `scripts/` 或 `analysis/`，通过 `create_quantizer` 工厂
  构建模型并加载 checkpoint，不改变任何现有训练路径的行为。
- 统计过程中模型处于 eval 模式、无梯度；batch 遍历，显存安全。

## 交付

- 诊断脚本 + 单元测试（至少验证输出结构和数值非负/有限）。
- `runner/jobs/H002/run.sh`：执行诊断，输出到 `results/H002/`：
  - `metrics.json`：上述全部标量指标（per-level）。
  - `usage_level{0,1}.csv`：每级 code 使用计数表（小文件）。
  - 可选 `*.png` 直方图（小文件，可入库）。
- `runner/jobs/H002/manifest.json`：`requires_stage1: false`，
  `stage1_checkpoint` 填复用的绝对路径。
- 日志：`runner/logs/H002/diagnostics.log`。

## 验收标准（Review 用）

- 5 类指标全部产出且数值有限；active code ratio ∈ [0,1]。
- 使用的 checkpoint 与 H001 记录中的 Stage 1 checkpoint 是同一个文件
  （核对文件名与大小/hash）。
- 未触发任何训练（无新 checkpoint 生成）。
- Review 需回答：第 1 级 codebook 是否被有效利用？第 1 级对残差范数的
  削减贡献多大？
