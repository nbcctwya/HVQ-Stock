# 007 — AlphaMaster Baseline

## Base

`main`

## Idea / Motivation

This experiment is **AlphaMaster under HVQ canonical dataset and unified
experiment protocol**. It establishes a pure AlphaMaster baseline inside the
current HVQ research pipeline so later experiments can build on AlphaMaster
without introducing a second data or evaluation convention.

This experiment is AlphaMaster under the HVQ canonical dataset and unified
evaluation protocol, not a bitwise reproduction of the standalone
AlphaMaster repository's historical run.

## Core modification

The HVQ/VQ two-stage model is replaced by the standalone AlphaMaster model's
core sequence:

```text
Market-Guided Feature Gate
→ Linear projection
→ Positional Encoding
→ TAttention
→ SAttention
→ TemporalAttention
→ prediction
```

`Gate`, `PositionalEncoding`, `TAttention`, `SAttention`,
`TemporalAttention`, and `MASTER` are adapted directly from the sibling
AlphaMaster repository. Formal defaults remain faithful to that implementation:
`d_feat=158`, `d_model=256`, temporal heads 4, spatial heads 2, both dropout
rates 0.5, Adam learning rate `8e-6`, and market-gate beta 10 for CSI300 / 5
for SP500.

## Difference from base

- The single experimental variable is replacing the current HVQ architecture
  with pure AlphaMaster.
- The canonical `[N,20,244]` batch is parsed into stock158, prior13, market63,
  and future-return10 using the existing schema. AlphaMaster consumes only
  stock158 and market63; prior13 is completely unused.
- The market63 channel remains the already-generated canonical channel:
  CSI300 uses `sh000300`, `sh000852`, `sh000905`; SP500 uses `^gspc`, `^dji`,
  `^ndx`. No AlphaMaster-specific dataset is created.
- Stage 1 is AlphaMaster training at fixed seed 42 with validation-best
  checkpointing. Stage 2 strictly loads that checkpoint, performs unified
  test inference at protocol seed 0, and writes `0_best.pkl` / `0_metric.csv`.
- `backtest_qlib.py`, data splits, target day 5, metrics, and the fixed
  Top30/Drop5 backtest protocol are unchanged.

## Smoke status

PASS. The smoke run covers canonical parsing, stock/market dimensions,
prior invariance, market sensitivity, same-day cross-sectional batching,
CSI300 and SP500 forwards, one-epoch limited Stage 1 training, runner checkpoint
discovery, strict Stage 2 checkpoint loading, output discovery, and prediction
normalization by the existing backtest adapter.

Artifacts and logs are isolated under `artifacts/007/smoke/`.
