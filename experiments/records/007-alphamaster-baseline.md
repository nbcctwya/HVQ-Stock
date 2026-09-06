# 007 — alphamaster-baseline

## Idea

AlphaMaster under HVQ canonical dataset and unified experiment protocol：将纯
AlphaMaster architecture 纳入当前固定的 Stage 1 → Stage 2 → backtest
科研流水线，作为后续 AlphaMaster 创新的基础 baseline。

## Motivation

当前 001–006 围绕 HVQ/VQ 架构展开，缺少在相同 canonical 数据、split、
指标和回测协议下的 AlphaMaster 对照。007 不提出新方法，而是忠实移植已在
同级 `AlphaMaster` 仓库验证过的正式实现，隔离验证“将 HVQ 模型替换为
AlphaMaster architecture”这一唯一实验变量。

This experiment is AlphaMaster under the HVQ canonical dataset and unified
evaluation protocol, not a bitwise reproduction of the standalone
AlphaMaster repository's historical run.

## Modification

- 从只读 sibling AlphaMaster 实现直接 adapted `Gate`、
  `PositionalEncoding`、`TAttention`、`SAttention`、
  `TemporalAttention` 和 `MASTER`；同权重、同输入的 standalone/local
  forward 单测逐位一致。
- 保留核心链路：market-guided feature gate → linear projection →
  positional encoding → TAttention → SAttention → TemporalAttention →
  prediction。
- 正式默认值：`d_feat=158`、`d_model=256`、`t_nhead=4`、`s_nhead=2`、
  temporal/spatial dropout 均为 0.5、Adam `lr=8e-6`；CSI300 beta=10，
  SP500 beta=5。
- 通过现有 canonical parser 将 `[N,20,244]` 解析为 stock158 / prior13 /
  market63 / future-return10；模型只接收 stock158 + market63，target 为
  `target_day=5`，prior13 不进入 forward、参数或 loss。
- Stage 1 改为 AlphaMaster 正式训练（固定 seed 42）与 validation-best
  checkpoint；checkpoint 命名兼容 runner discovery 和 artifact_root。
- Stage 2 改为 strict 加载该完整 AlphaMaster checkpoint，以协议 seed 0
  执行 test inference，计算统一 IC/ICIR/RankIC/RankICIR，并输出
  `0_best.pkl` / `0_metric.csv`。
- 复用 canonical market63：CSI300 为 `sh000300` / `sh000852` /
  `sh000905`，SP500 为 `^gspc` / `^dji` / `^ndx`；未复制 AlphaMaster
  的历史 Qlib dataset pipeline，未新增专用数据格式。
- `backtest_qlib.py` 与 `experiments/runner.py` 均未修改。

## Constraints

- 唯一实验变量：用 pure AlphaMaster architecture 替换 base 的 HVQ 模型；
  不含 VQ/HVQ、z0/z1、MoE、prior fusion、额外 market encoder 或其他创新。
- canonical schema 固定为 158 + 13 + 63 + 10 = 244，窗口固定 T=20；
  数据 split 保持 train 2009–2020 / valid 2021–2022 / test 2023–2025。
- Stage 1 seed 42，Stage 2 protocol seed 0；max epoch 70、validation early
  stopping patience 15；target_day=5。
- 回测继续使用固定 Top30/Drop5、open 0.0005 / close 0.0015、min_cost 0、
  close 成交协议。
- AlphaMaster sibling 仓库保持只读且工作区 clean。
- Stage 1 provenance: self；不复用 001–006 或 HVQ/VQ checkpoint。

## Git

Base: main
Branch: exp/007-alphamaster-baseline
Commit: feed430d9e8a2c13157424583b3f6870ea28cfb9
Stage 1 provenance: self（007 Stage 1 为 AlphaMaster 自身正式训练）

## Smoke Test

Status: PASS

Notes: conda `prism-vq` 下完整单元测试 88/88 PASS；最终相关回归 72/72
PASS。`scripts/smoke_alphamaster.py` 使用 tiny canonical PKL 完成入口级 smoke：
CSI300 / SP500 的 `[4,20,244]` 均解析为 stock `[4,20,158]`、market
`[4,20,63]` 并 forward PASS；beta 分别为 10 / 5；同一 trading day
cross-section batching PASS；prior13 修改后 prediction 逐位不变，market63
修改后 prediction 改变。

Stage 1 限制为 1 epoch、2 train batches、2 validation batches，生成
`alphamaster_smoke-epoch=0-val_loss=0.4735.ckpt`，runner discovery PASS。
Stage 2 strict load PASS，生成
`artifacts/007/smoke/res/alphamaster_csi300/0_best.pkl` 和
`0_metric.csv`；40 行 prediction 经现有 `backtest_qlib.py` normalizer
接受。日志与机器可读结果位于 `artifacts/007/smoke/stage1.log`、
`stage2.log`、`smoke_report.json`。未执行正式长时间训练或回测。

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
