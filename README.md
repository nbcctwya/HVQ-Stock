# 008 — AlphaMaster Prior Factor Head

## Base

`exp/007-alphamaster-baseline`

## Idea / Motivation

This experiment tests whether the 13 JKP prior factors already present in the
HVQ canonical dataset provide incremental explicit financial information on
top of AlphaMaster's market-aware, temporal-aware, and cross-sectional hidden
representation.

The only question tested is whether **dynamic prior-factor decomposition can
add incremental information to the AlphaMaster baseline**.

## Core modification

The AlphaMaster backbone and its `[stock158, market63]` input are unchanged.
Only the final prediction head is replaced:

```text
007:
y_hat = Linear(h)

008:
alpha = AlphaHead(h)
beta = PriorLoadingHead(h)
y_hat = alpha + beta^T prior
```

`AlphaHead` is an unconstrained `Linear(256, 1)`. `PriorLoadingHead` is an
unconstrained `Linear(256, 13)`. For each stock, the latter dynamically maps
the final AlphaMaster hidden representation `h` to 13 loadings. The explicit
prior contribution is `sum(beta * prior13)`, and the final prediction is its
sum with the implicit AlphaMaster alpha.

## Difference from base

- The single `Linear(h -> prediction)` in 007 becomes an alpha head plus a
  dynamic 13-dimensional prior-loading head.
- Prior13 is read unchanged from the canonical `[N,20,244]` batch and enters
  only the final factor decomposition. It is not concatenated into the model
  input and cannot enter the Market-Guided Feature Gate, TAttention,
  SAttention, or TemporalAttention.
- Market63 usage, all AlphaMaster backbone layers, dimensions, attention
  heads, dropout, market-gate beta, optimizer, training budget, splits,
  target day, metrics, seeds, and backtest protocol remain identical to 007.
- Stage 1 provenance is `self`: the trainable prediction-head structure has
  changed, so a 007 Stage 1 checkpoint is not strict-compatible.

No VQ/HVQ, z0/z1, MoE, market encoder, regime mechanism, RevIN, prior
preprocessing, constraints on beta, sparsity, or auxiliary loss is introduced.

## Smoke status

PASS. Unit and end-to-end smoke coverage verifies canonical parsing, unchanged
backbone inputs and hidden computation, prior isolation, exact
`alpha + sum(beta * prior)` prediction, dynamic sample-dependent loadings,
deterministic forward, CSI300/SP500 forward, minimal Stage 1 training,
validation-best checkpoint discovery, Stage 2 strict loading and output
generation, and compatibility with the existing backtest prediction adapter.

Artifacts and logs are isolated under `artifacts/008/smoke/`.
