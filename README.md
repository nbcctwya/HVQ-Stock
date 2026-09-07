# 011 — AlphaMaster Continuous Market Adapter

## Base

`exp/007-alphamaster-baseline`

## Idea / Motivation

007 uses the current market snapshot to gate and reweight stock features before
the AlphaMaster backbone. This experiment keeps that path intact and asks a
separate question: can the preceding 19-day market trajectory condition the
final prediction function after AlphaMaster has extracted its 256-dimensional
stock representation?

The two market signals have deliberately different roles. The current day
controls stock-feature selection; historical market state supplies a dynamic
decoder-weight residual.

## Core modification

The original 007 path remains:

```text
market[:, -1, :] -> Feature Gate -> stock158 -> AlphaMaster backbone
-> h[256] -> original decoder -> y_base
```

The only added path is:

```text
market[:, :-1, :] [N,19,63]
-> single-layer GRU -> m_t [N,63]
-> Linear(63,256,bias=False), zero-initialized -> delta_w_t [N,256]
-> dot(delta_w_t, h) -> y_market
```

The final prediction is `y_base + y_market`. The adapter weight is explicitly
zero-initialized, so a newly initialized 011 model is exactly prediction-
equivalent to 007 when the backbone parameters match. The adapter first learns
a prediction-side correction; once it becomes nonzero, prediction loss also
propagates into the GRU.

The temporal market encoder matches experiment 009's lightweight GRU:
`input_size=63`, `hidden_size=63`, `num_layers=1`, `batch_first=True`,
`bidirectional=False`, and `dropout=0`.

## Difference from base

- Added only the previous-19-day GRU and zero-initialized linear Market Adapter
  residual on the prediction side.
- The original current-market Feature Gate still receives exactly
  `market[:, -1, :]` and is not replaced by the GRU state.
- The historical branch receives exactly `market[:, :-1, :]`; it does not see
  the current day.
- The Feature Gate, projection, positional encoding, temporal/spatial
  attention, TemporalAttention, and original decoder are unchanged.
- The canonical `158 + 13 + 63 + 10 = 244` schema, `T=20`, unused prior13
  behavior, model dimensions, dropout, beta, target day, optimizer, splits,
  seeds, training budget, metrics, and backtest protocol remain those of 007.
- Stage 1 provenance is `self`, because the GRU and Market Adapter are new
  trainable parameters and must be trained through the AlphaMaster pipeline.

## Smoke status

PASS. The full unit suite passes 91/91 tests. The isolated smoke run validates
canonical shapes, exact current/history slicing, GRU/hidden/adapter shapes,
zero initialization, exact zero-init 007 prediction equivalence, same-day
cross-sectional market-state sharing, current/history path decoupling, adapter
update, subsequent GRU gradients, strict Stage 1 to Stage 2 checkpoint loading,
and standard prediction/metric compatibility with the existing backtest
normalizer.

Artifacts and logs are isolated under `artifacts/011/smoke/`.
