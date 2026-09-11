# Baseline seed0 Phase 3 兼容性审计报告

审计日期：2026-09-11。审计对象：`artifacts/baseline/run/`（2026-09-05 在
main `2c78051bdb9f2cc66e59bbc6a31bd93aa0acd1bb` 上执行的 corrected-protocol
seed0）。结论：**可比（pass）**，结论由逐 bit 复现支撑，非推断。

## 历史实现

- Baseline seed0 执行方式见 `baseline_results/CORRECTED_PROTOCOL.md`：main 上的
  原始 PRISM-VQ 代码 + corrected freeze + artifact_root，`stage2.py train.seed=0`。
- 当时数据管道（`2c78051`）：`dataset/data/CN/csi300_20_h10_dl2_*.pkl`，
  Alpha158（158 特征）+ JKP prior（13）+ 10 维 label（RET_1D..RET_10D，
  `Ref($close,-h)/Ref($close,-1)-1`），TSDatasetH 采样，无 market 通道。

## main 数据改造（2026-09-06，晚于 seed0 一天）

- `619a3c6`/`1052484`/`db85a36`/`5b4ac7f`：引入 schema v2
  （158 feature + 13 prior + 63 market + 10 label = 244 维）、CanonicalSampler、
  canonical 数据 `dataset/processed/CN/csi300_20_h10_*.pkl`。
- 差异分析：
  - **market63 通道是新增的，但 main 的 baseline 模型不消费它**
    （`trainer/train_ypred.py`：`market_feature is deliberately unused by the
    current baseline`）。模型输入仍只有 feature/prior/label。
  - 模型消费的三组切片定义新旧一致：Alpha158 定义继承未改、JKP
    `build_factor_matrix`（window=20）逐行相同、label 公式相同、
    处理链（RobustZScoreNorm/Fillna/DropnaLabel/CSRankNorm）相同。
  - 采样：CanonicalSampler 对 stock/prior/label 保持 Qlib ffill+bfill 语义，
    并以扩展 lookback 保证与 TSDatasetH 相同的完整窗口；market 按日历对齐但
    不被模型使用。
  - `stage2.py` 差异仅数据路径解析；`backtest_qlib.py` 零差异；
    `module/` 零差异；`stage1.py` 差异与 Stage 2-only 协议无关（Stage 1 固定）。
- Stage 1：新旧均固定加载同一 checkpoint
  `infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt`
  （sha256 `a9ac1599d129550d…`），strict 加载 missing=0/unexpected=0。

## 实际模型输入/标签证据（seed0 产物）

- 旧 seed0 预测 vs canonical 010 seed0 预测的 label 列：217909 行 index 完全一致，
  217591 行 label 完全相等，318 行同为 NaN 且位置一致——test 标签逐点相同。

## 决定性证据：新 seed0 逐 bit 复现（2026-09-11）

在当前 main（`0b823c079393ea6752580978d3593e16aeac3cf6`）+ canonical 数据上，
用与 seed1-4 相同的标准（同一 stage2.py 协议、同一 Stage 1 ckpt、train.seed=0）
重跑 baseline seed0，产物在 `artifacts/_phase3/audit/baseline-seed0-repro/`：

- 29 个 epoch 的 Val_RIC 轨迹与 2026-09-05 原 run 逐值相同；
- test 指标完全相同：IC 0.03734755100665368 / ICIR 0.22143133656484254 /
  RankIC 0.05521290926890501 / RankICIR 0.33134680221916957；
- 预测 pkl：217909/217909 score 完全相等（max diff 0.0），label 完全相等；
- `0_metric.csv` 完全一致。

## 结论

旧 seed0 与「冻结 commit `0b823c0` + canonical 数据 + 固定 Stage 1」下新跑的
seed 在代码、数据、协议上等价，可比。Phase 3 统计中 seed0（只读）与
新 seeds 1–4 混合有效。跨 GPU 数值差异（新 seeds 在 2080 Ti 上训练）不属于本
审计范围，按 README 既有声明处理。

配套 JSON：`phase3_seed0_compatibility.json`（preflight 读取）。
