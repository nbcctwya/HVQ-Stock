# 014 — AlphaMaster Day-Level EMA Market Adapter

## Base

`exp/013-alphamaster-ema-market-adapter`

## Idea / Motivation

Experiment 013 treats every stock row in a daily cross-section as a separate
observation when it updates the EMA codebook. Those rows share the same
previous-19-day market window and therefore the same historical market state,
but a day with more stocks contributes a proportionally larger EMA count and
embedding sum.

This experiment tests whether historical market regimes are better modeled as
trading-day states. The model path is unchanged:

```text
market history
    -> GRU
    -> continuous market state
    -> EMA VQ
    -> quantized market state
    -> Market Adapter
    -> prediction residual
```

Only the EMA observation granularity changes:

```text
one trading day
    -> one shared market state
    -> one shared code assignment
    -> EMA count += 1
    -> EMA embedding sum += market_state
```

The hypothesis is that day-weighted statistics will produce regime prototypes
with clearer market-state semantics and smoother, more stable trajectories than
statistics weighted by the number of stocks present in each cross-section.

## Core modification

The canonical daily sampler already emits exactly one complete trading-day
cross-section per model forward. During training, the EMA update now uses the
first row of that shared historical market state as the day's single
observation. The quantizer still returns one index and quantized vector per
stock row, so every stock retains the same shared assignment and adapter input.

The default configuration explicitly sets
`alphamaster.market_quantizer.statistics_level: trading_day`; the trainer
validates this value and passes it through the normal model construction path.
No experiment-specific CLI override is required.

## Difference from base

- The sole experimental variable is EMA statistics granularity: 013 accumulates
  every stock row, while 014 accumulates exactly one shared observation per
  daily batch. Cross-section size no longer weights EMA count or embedding sum.
- EMA decay remains `0.99`. The EMA buffers, initial pseudo-observation,
  prototype normalization, train/eval update isolation, and checkpoint state
  remain unchanged.
- VQ remains an `8 x 63` codebook with squared-L2 nearest-neighbor assignment,
  the same straight-through estimator, and commitment weight `0.25`. The VQ
  loss scalar remains `detached codebook MSE + 0.25 * commitment MSE`.
- The previous-19-day market slice, single-layer unidirectional GRU
  (`63 -> 63`, batch-first, dropout 0), zero-initialized
  `Linear(63,256,bias=False)` Market Adapter, and
  `y_base + sum(delta_w_t * h)` decoder residual are unchanged.
- The current-day Feature Gate, complete MASTER backbone and decoder, canonical
  `158 stock + 13 prior + 63 market + 10 returns = 244` schema, unused prior13,
  dimensions, dropout, beta, target day, Adam learning rate, data splits,
  70-epoch budget, patience, seed protocol, metrics, and Top30/Drop5 backtest
  protocol remain unchanged from 013.
- Stage 1 provenance is `self`: although an 013 checkpoint is structurally
  strict-load compatible because parameter and buffer shapes are unchanged,
  it was trained with sample-level EMA statistics and therefore cannot stand in
  for a model formally trained under the 014 mechanism.

## Smoke status

PASS. The complete unit suite passes 94/94 tests. Mechanism coverage includes
the exact decay-`0.99` update, one-observation day weighting, bitwise-identical
EMA counts/sums/prototypes for a 1-row versus 257-row shared cross-section,
frozen codebook gradients, optimizer/EMA separation, eval isolation, L2
assignment, STE and commitment gradients, same-day regime sharing, path
isolation, strict checkpoint loading, and unchanged backbone behavior.

The isolated CPU smoke ran one epoch with two training and two validation
batches. It produced
`artifacts/014/smoke/checkpoints/alphamaster_smoke-epoch=0-val_loss=0.5833.ckpt`,
strict-loaded it in Stage 2, emitted the standard 40-row prediction and metric
files, and passed the existing backtest input normalizer. Ten test trading days
used 5 of 8 codes with counts `[0, 0, 4, 1, 1, 0, 2, 2]`; every daily
cross-section shared exactly one regime. These tiny-run diagnostics validate
the pipeline only and are not formal experiment results.

All smoke artifacts, logs, and the machine-readable report are isolated under
`artifacts/014/smoke/`.
