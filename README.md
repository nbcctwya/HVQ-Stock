# 013 — AlphaMaster EMA Market Adapter

## Base

`exp/012-alphamaster-discrete-market-adapter`

## Idea / Motivation

Experiment 012 quantizes the historical GRU market state with a standard VQ
whose codebook is learned by ordinary gradients. This experiment changes only
that update mechanism to exponential moving averages with decay `0.99`.

The question is whether EMA produces more stable historical market-regime
prototypes and thereby improves prediction-side market adaptation. The model
path remains:

```text
market history
    -> GRU
    -> continuous market state
    -> EMA VQ
    -> quantized market state
    -> Market Adapter
    -> prediction residual
```

## Core modification

The `8 x 63` codebook is no longer an optimizer-updated parameter. During
training, L2 nearest-neighbor assignments accumulate per-code cluster counts
and embedding sums, and the corresponding buffers and prototypes are updated
with EMA decay `0.99`. One initial pseudo-observation per code preserves unused
prototypes and keeps the first normalization stable. Evaluation and validation
forwards do not update the EMA state.

The selected code remains the exact forward value and the straight-through
estimator remains the identity path to the GRU. The reported VQ loss remains
`codebook_mse + 0.25 * commitment_mse`, preserving 012's training, validation,
and checkpoint-selection scale. The codebook term is detached, so only the
commitment term contributes gradients and codebook learning occurs only through
EMA.

## Difference from base

- The sole experimental variable is the 012 codebook update: ordinary gradient
  updates are replaced by EMA count/sum updates with decay `0.99`.
- Code assignment remains squared L2 nearest-neighbor; codebook size remains 8,
  embedding dimension remains 63, the straight-through estimator is unchanged,
  and commitment weight remains 0.25.
- The historical market slice, single-layer GRU, quantized adapter input,
  zero-initialized `Linear(63,256,bias=False)` Market Adapter, and decoder
  residual equation are unchanged from 012.
- The current-day Feature Gate, complete MASTER backbone, decoder, canonical
  schema, unused prior13 behavior, model dimensions, dropout, beta, target day,
  Adam learning rate, splits, training budget, early stopping, seeds, metrics,
  and Top30/Drop5 backtest protocol are unchanged.
- Stage 1 provenance is `self`: EMA buffers and the formal codebook-training
  mechanism differ from 012, so the complete model must be retrained. A real
  012 smoke checkpoint fails strict loading because it lacks the EMA state;
  the 013 smoke checkpoint strict-loads successfully.

## Smoke status

PASS. The complete unit suite passes 93/93 tests. Coverage includes the exact
EMA update at decay `0.99`, frozen codebook gradients, optimizer/EMA separation,
no EMA mutation in evaluation mode, L2 assignment, STE and commitment gradient
semantics, canonical shapes and path isolation, same-day regime sharing,
GRU/Adapter gradients, strict checkpoint state, and unchanged baseline paths.

The isolated CPU smoke runs one epoch with two training and two validation
batches. It produced
`artifacts/013/smoke/checkpoints/alphamaster_smoke-epoch=0-val_loss=0.5791.ckpt`,
strict-loaded it in Stage 2, emitted the standard 40-row prediction and metric
files, and passed the existing backtest input normalizer. Ten test trading days
used 4 of 8 codes with counts `[0, 0, 4, 1, 0, 0, 1, 4]`; every daily
cross-section shared exactly one code. These tiny-run diagnostics verify the
pipeline only and are not formal experiment results.

All smoke artifacts, logs, and the machine-readable report are isolated under
`artifacts/013/smoke/`.
