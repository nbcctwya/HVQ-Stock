# 009 — AlphaMaster Temporal Market

## Base

`exp/007-alphamaster-baseline`

## Idea / Motivation

AlphaMaster 007 conditions its Market-Guided Feature Gate on only the final
trading day's 63-dimensional market snapshot. This experiment tests whether a
lightweight temporal encoder can extract a more informative and stable market
state from the complete canonical 20-day market trajectory.

This experiment answers only:

> Is temporal market history more suitable than a single-day market snapshot
> as the market-state input to AlphaMaster's Feature Gate?

## Core modification

The sole model change is the market-state calculation immediately before the
existing Feature Gate:

```text
007:
market_state = market[:, -1, :]

009:
market_state = GRU(market[:, :, :]).last_hidden

Both:
market_state
→ original AlphaMaster Feature Gate
→ unchanged AlphaMaster backbone
```

The temporal market encoder is a batch-first, one-layer, unidirectional GRU
with `input_size=63` and `hidden_size=63`. It has no attention, residual,
projection, auxiliary loss, or additional market features. Its final hidden
state is exactly `[N,63]`, preserving the original Feature Gate's input width
and mathematical structure.

The complete path is:

```text
canonical market [N,20,63]
→ GRU(63→63, one layer, unidirectional)
→ final hidden market_state [N,63]
→ original Gate(63→158, beta=10 CSI300 / 5 SP500)
→ stock158 feature reweighting
→ Linear projection
→ Positional Encoding
→ TAttention
→ SAttention
→ TemporalAttention
→ original decoder
→ prediction [N]
```

## Difference from base

- Only `market[:, -1, :]` is replaced by the GRU final hidden state as the
  Feature Gate input.
- The Feature Gate, stock158 path, projection, all attention blocks, decoder,
  optimizer, target, metrics, data splits, seed protocol, training budget, and
  backtest protocol remain those of 007.
- The canonical schema remains stock158 + prior13 + market63 + return10 at
  `T=20`; prior13 remains completely unused.
- Stage 1 provenance is `self`: the new trainable GRU changes parameters and
  the forward graph, so a 007 Stage 1 checkpoint is not strict-compatible.

## Smoke status

PASS. Unit and entry-level smoke tests cover canonical parsing, both universes,
the 63-dimensional encoder/gate contract, unchanged stock path, prior
invariance, sensitivity to only the first 19 market days while the final day is
fixed, encoder gradients, same-day cross-sectional batching, validation-best
checkpoint discovery, strict Stage 2 loading, standard prediction artifacts,
and compatibility with the existing backtest normalizer.

Artifacts and logs are isolated under `artifacts/009/smoke/`.
