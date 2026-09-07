# 012 — AlphaMaster Discrete Market Adapter

## Base

`exp/011-alphamaster-continuous-market-adapter`

## Idea / Motivation

Experiment 011 conditions the prediction-side Market Adapter directly on a
continuous 63-dimensional state produced from the previous 19 market days.
This experiment asks whether compressing that state into one of a small number
of reusable regimes filters market noise and improves prediction-side
adaptation.

The current-day and historical market signals retain separate roles. The
current day controls the original input-side Feature Gate; only the historical
GRU state is quantized before it reaches the existing Market Adapter.

## Core modification

The complete 011 path remains, with one standard VQ inserted between its GRU
and adapter:

```text
market[:, :-1, :] [N,19,63]
-> unchanged single-layer GRU -> m_t [N,63]
-> standard VQ (8 codes, embedding_dim=63) -> z_q [N,63]
-> unchanged Linear(63,256,bias=False), zero-initialized -> delta_w_t [N,256]
-> dot(delta_w_t, h) -> y_market
```

Quantization uses squared L2 nearest-neighbor assignment, a standard
straight-through estimator, and the loss
`codebook_mse + 0.25 * commitment_mse`. The selected embedding is the exact
forward value, while the straight-through path sends prediction gradients to
the GRU. VQ loss is added to the unchanged prediction MSE during Stage 1.

The canonical daily sampler emits one complete trading-day cross-section per
batch. Because all stocks on that day share the same historical market window,
they select the same code and receive the exact same quantized regime vector.

## Difference from base

- Added only a trainable `8 x 63` standard VQ codebook between the existing GRU
  output and existing Market Adapter input, plus its standard VQ loss.
- The 011 GRU is unchanged: `input_size=63`, `hidden_size=63`, `num_layers=1`,
  `batch_first=True`, unidirectional, and dropout 0.
- The Market Adapter remains `Linear(63,256,bias=False)` with explicit zero
  initialization, and the decoder residual equation remains unchanged.
- The original Feature Gate still and only receives `market[:, -1, :]`; the
  GRU/VQ branch still and only receives `market[:, :-1, :]`.
- The MASTER backbone, original decoder, canonical `158 + 13 + 63 + 10 = 244`
  schema, `T=20`, unused prior13 behavior, splits, model dimensions, dropout,
  beta, target day, optimizer, seeds, training budget, early stopping, metrics,
  and Top30/Drop5 backtest protocol are unchanged from 011.
- Stage 1 provenance is `self`, because VQ is newly trainable and changes the
  formal forward graph, so the complete model must be retrained.

## Smoke status

PASS. The full unit suite passes 92/92 tests. Tests and the isolated smoke cover
canonical GRU/VQ/Adapter shapes, the exact configured VQ dimensions and loss,
straight-through gradients, zero initialization, exact initial prediction
equivalence, same-day quantized-regime sharing, current/history path isolation,
VQ/GRU/Adapter gradients and parameter updates, strict checkpoint save/load,
Stage 1 to Stage 2 inference, and compatibility with the existing prediction
and backtest interfaces.

The CPU smoke uses one epoch with two train and two validation batches. Its
trained checkpoint assigned 10 test trading days to 5 of 8 codes, with counts
`[0, 0, 4, 1, 1, 0, 2, 2]`; every daily cross-section shared exactly one code.
This tiny diagnostic shows no immediate single-code collapse but is not a
substitute for the formal run.

Artifacts, report, and logs are isolated under `artifacts/012/smoke/`.
