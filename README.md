# Experiment 022 — Explicit Code-Aware Routing

## Base

This experiment is based directly on `main`, the corrected original PRISM-VQ
baseline. It changes only the Stage 2 MoE clean routing logits and reuses the
external corrected PRISM-VQ Stage 1 checkpoint.

## Idea / Motivation

In the baseline, the MoE router infers expert preference purely from the
continuous quantized embedding:

```text
clean_logits = Router(z_q)
```

Although `z_q` already encodes the continuous representation of the matched
prototype, the discrete VQ code identity itself may carry a stable
expert-specialization prior. Experiment 022 explicitly learns an additive
routing bias `P(expert | code_id)`, so that distinct discrete latent states
can build stable expert preferences instead of relying on the continuous
router to re-infer them from `z_q` every time.

## Core Modification

A code-specific expert preference table is added to `FactorGatedMoE`:

```text
B ∈ R^(K x n_expert),  K = vqvae.num_embed = 512
code_bias = B[vq_idx]
clean_logits = Router(z_q) + code_bias
```

- `B` (`loadings.fusion.moe.code_bias`) is an `nn.Parameter` of shape
  `[vqvae.num_embed, predictor.n_expert]` — `[512, 2]` under the default
  config — explicitly initialized to all zeros. `torch.zeros` consumes no RNG
  state, so for the same seed every pre-existing baseline parameter is
  bitwise identical to `main`.
- The frozen Stage 1 quantizer already returns `vq_idx`; it is validated to
  lie in `[0, K-1]`, detached, cast to a long discrete index, and passed
  `GenerateReturn.forward -> LoadingGenerator -> HyperFusion ->
  FactorGatedMoE.noisy_top_k_gating`, where it only indexes `B` and the
  selected row is added to the clean routing logits.
- Everything downstream is untouched: the original router, noise network,
  `W_h`, top-k, softmax, `SparseDispatcher`, and the load-balancing loss keep
  their baseline definitions and weights. With zero-init, clean logits,
  routing, the original auxiliary loss, and the complete prediction forward
  are bitwise identical to the baseline.

The default `configs/config.yaml` enables the experiment with
`predictor.code_aware_routing: true`; no experiment-specific CLI override is
required.

## Difference from Base

The sole experimental variable is the additive code-ID-specific routing bias
on the MoE clean logits. The bias does not enter expert inputs, the Temporal
Transformer, HyperFusion projections, factor heads, `LatentValueHead`,
`ReturnPredictor`, or any other prediction path, and it cannot backpropagate
into Stage 1. Quantizer assignment, codebook, Stage 1 loss and training
logic, data splits, training budget, seed protocol, metrics, and backtest
protocol are all unchanged. Encoder, Quantizer, RevIN, and codebook remain
frozen and receive no gradient. No shared expert, adaptive fusion,
decoupling, quantization confidence, market-conditioned routing, or
prior-latent allocation mechanism is introduced.

Stage 1 provenance:

```text
external
artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt
```

This is the corrected PRISM-VQ exact single-VQ512, 128-dimensional Stage 1
checkpoint trained with the fixed Stage 1 seed 42.

## Smoke Status

PASS. The full unit suite and the minimal synthetic Stage 2 smoke validate
`vq_idx` pass-through from the quantizer to the MoE router, `[512, n_expert]`
zero-initialized table shape, zero-init bitwise equivalence of clean
logits/routing/auxiliary loss/full forward with the baseline, bitwise
initialization isolation of all pre-existing parameters, unchanged noisy
top-k/noise/`W_h`/dispatcher/load-balancing behavior, distinct routing
preferences for distinct code ids under a non-zero bias, finite non-zero
code-bias gradients and updates, loud errors on invalid code ids, isolation
of the code id to the routing path, strict external Stage 1 loading and
freezing, strict Stage 2 checkpoint round-trip, standard inference/metrics,
and backtest input normalization. Smoke artifacts are isolated under
`artifacts/022/smoke/`.

No formal long-running training or portfolio backtest is performed in Phase 1.
