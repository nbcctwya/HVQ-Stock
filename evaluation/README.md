# evaluation/ — HVQ-Stock 正式评估层

依据 `RULES.md`（Baseline Results Protocol v1.0）对**已完成**的正式实验产物重新计算
预测指标、重新执行统一 Qlib 回测、进行多 seed 聚合与 ensemble，并生成论文级统一
`results/`。

本目录独立于科研流水线：Phase1/2/3、根目录 `backtest_qlib.py` 与既有回测结果保持原样，
只被读取，不被修改，也不被当作正式结果来源。

## 用法

```bash
# 在 HVQ-Stock 仓库根目录，使用项目环境（prism-vq）
python -m evaluation.run --experiments baseline 010 019 025 034
# 可选：--market csi300  --seeds 0 1 2 3 4  --ensemble-methods avg_none  --out results

# 独立重新校验既有 results/
python -m evaluation.validate --out results   # 全 PASS 退出码 0，否则非 0
```

被评估的实验必须已经有正式 prediction：seed 0 来自 `artifacts/<exp>/run/`，
seed ≥1 来自 `artifacts/_phase3/batches/*/receipts/` 中 accepted receipt 指向的 archive。
缺失或冲突会明确失败，不会静默跳过。

## 结构

```text
evaluation/
├── README.md       # 本文件
├── RULES.md        # Baseline Results Protocol v1.0（权威规则）
├── protocol.py     # 固定协议常量与项目事实（config.yaml / dataset yaml）
├── discovery.py    # 正式 accepted prediction 发现（Phase2 seed0 / Phase3 receipts）
├── metrics.py      # IC/RankIC 与 AR/STD/MDD/Sharpe/Sortino/Calmar 公式（唯一实现处）
├── backtest.py     # prediction 载入、coverage 检查、统一 Qlib 回测、曲线生成
├── ensemble.py     # prediction 层 ensemble（avg_none，inner join）
├── run.py          # 正式入口：python -m evaluation.run
└── validate.py     # 正式校验：python -m evaluation.validate
```

输出（默认仓库根 `results/`，可用 `--out` 更改）：见 `RULES.md` 第 2 节。

测试：`tests/test_evaluation.py`（指标公式、协议参数、时序等），与项目一致使用 unittest：

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_evaluation -v
```
