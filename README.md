# Experiment 021 — Latent-Conditioned Prior–Latent Allocation

## Base

This experiment is based directly on `main`, the corrected original PRISM-VQ
baseline. It changes only the final Stage 2 return composition and reuses the
external corrected PRISM-VQ Stage 1 checkpoint.

## Idea / Motivation

The baseline always combines expert prior factors and learned latent factors
with the same relative scale:

```text
y_pred = alpha + prior_term + latent_term
```

Their relative reliability may instead depend on the discrete stock state.
Experiment 021 therefore lets the frozen Stage 1 raw quantized latent `z_q`
select a small complementary reallocation between the two already-computed
factor contributions.

## Core Modification

`ReturnPredictor` gains one zero-initialized `Linear(128, 1)` allocation gate:

```text
g = 0.5 * tanh(Linear(z_q))
prior_scale = 1 + g
latent_scale = 1 - g
y_pred = alpha + prior_scale * prior_term + latent_scale * latent_term
```

The scales lie in `(0.5, 1.5)` and sum to 2. At initialization, `g = 0` and
both scales equal 1, so the complete prediction forward is bitwise identical
to the baseline. The gate is appended only after every baseline module has
been constructed, preserving all existing parameter initialization for the
same seed.

The default `configs/config.yaml` enables the experiment with
`predictor.latent_conditioned_allocation: true`; no experiment-specific CLI
override is required.

## Difference from Base

The sole experimental variable is the latent-conditioned complementary
scaling of the completed `prior_term` and `latent_term`. The gate does not
change `beta_p`, `beta_l`, factor heads, HyperFusion, MoE/router, the Temporal
Transformer, losses, data, training budget, metrics, or backtest protocol.
Encoder, Quantizer, RevIN, and codebook remain frozen and receive no gradient.

Stage 1 provenance:

```text
external
artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt
```

This is the corrected PRISM-VQ exact single-VQ512, 128-dimensional Stage 1
checkpoint trained with the fixed Stage 1 seed 42.

## Smoke Status

PASS. The full unit suite and the minimal synthetic Stage 2 smoke validate
zero-init baseline equivalence, initialization isolation, complementary
allocation behavior, gate gradients, strict external Stage 1 loading and
freezing, strict Stage 2 checkpoint round-trip, standard inference/metrics,
and backtest input normalization. Smoke artifacts are isolated under
`artifacts/021/smoke/`.

No formal long-running training or portfolio backtest is performed in Phase 1.
