# Experiment 024 — VQ Transition-Aware Routing

## Base

This experiment is based directly on `main`, the corrected original PRISM-VQ
baseline. It changes only the Stage 2 MoE clean routing logits and reuses the
external corrected PRISM-VQ Stage 1 checkpoint.

## Idea / Motivation

In the baseline, the MoE router selects experts purely from the current
quantized latent state:

```text
clean_logits = Router(z_q,t)          # P(expert | current latent state)
```

The same current discrete state can be reached by very different state
evolution paths, e.g. `37 -> 37 -> 37 -> 37 -> 37` (stable dwelling) versus
`12 -> 18 -> 25 -> 31 -> 37` (rapid recent migration). The historical
latent-state transition trajectory may carry state dynamics that the current
`z_q` alone cannot express, so this experiment tests whether

```text
P(expert | current state, transition history)
```

improves expert routing over conditioning on the current state only.

## Historical Code Sequence Construction

`module/code_history.py` builds a deterministic, identity-keyed historical
code context before training:

1. Each split sampler's real MultiIndex provides every sample's
   `(instrument, datetime)`; indices must be 2-level with `datetime` /
   `instrument` levels and unique keys, otherwise the builder fails loudly.
2. The exact frozen corrected Stage 1 (RevIN -> SpatialEncoder ->
   VectorQuantiser, eval mode, `no_grad`) encodes every sample's stock
   feature window into its VQ code `vq_idx`, in fixed positional order.
   Only the stock-feature slice is read; future-return labels are never
   touched.
3. Split chronology is verified (all train datetimes < all valid datetimes <
   all test datetimes; any violation fails loudly).
4. For sample `(i, t)` only codes of the same instrument with
   `datetime < t` are eligible: train samples read earlier train
   observations, valid samples read earlier train + valid observations, test
   samples read earlier train + valid + test observations. No sample can
   ever read its own date or the future.
5. Each sample keeps up to `L - 1 = 4` most recent history codes
   (`hist_codes`, oldest first) plus the valid count (`hist_len`, 0..4).
   Samples with insufficient history are never dropped and the train/valid/
   test splits are unchanged.

The per-sample history is bound to the dataset position by
`TransitionHistoryDataset` (`init_data_loader(..., history=...)`), which
emits `(hist_codes, hist_len, batch)` triples. Sample identity — not batch
position or iteration order — determines the history, so training-date
shuffling cannot change any sample's historical sequence. The code map
depends only on the frozen Stage 1 checkpoint and the canonical data; a
cache under the artifact root is accepted only when the checkpoint content
hash and every split's index digest match exactly, and stale caches fail
loudly.

## Leakage Prevention

- History is selected by `np.searchsorted(..., side="left")` on per-instrument
  sorted datetimes, which enforces `datetime < t` strictly; an observation
  can never read itself or any future code.
- Duplicate `(instrument, datetime)` keys, malformed index schemas, and
  non-increasing split chronology all raise immediately.
- The builder never reads the label slice; unit tests poison future returns
  with NaN and verify identical codes/histories.
- Shuffle invariance is tested: two training epochs with different shuffle
  seeds must yield identical `(data row -> history)` mappings.

## Transition Representation

No code embedding is learned. Prototypes are read directly from the frozen
Stage 1 codebook and explicitly detached:

```text
e_k       = frozen_codebook[k].detach()
delta_z_j = e_{k_j} - e_{k_{j-1}}        # consecutive prototype difference
```

For a sample with `m` valid history codes, the code chain
`[k_{t-m}, ..., k_{t-1}, k_t]` yields exactly `m` transitions
`[delta_z_{t-m+1}, ..., delta_z_t]` (including the explicit
last-history -> current transition when `m < 4`). Padding codes are masked
out and excluded from the GRU via packed sequences. The current code `k_t`
always comes from the live quantizer output of the current forward — never
from the cache. A single-layer unidirectional GRU (input 128 = codebook dim,
hidden 64, no attention/Transformer) encodes the transition sequence; when
`hist_len == 0` the transition state is exactly zero.

## Routing Injection

```text
transition_state = GRU(delta_z sequence)          # R^64
transition_bias  = W_t(transition_state)          # W_t: R^64 -> R^n_expert
clean_logits     = Router(z_q,t) + transition_bias
```

`W_t` (`transition_encoder.proj`) is explicitly zero-initialized (weight and
bias), so at initialization `transition_bias == 0` and the complete model —
clean logits, noisy gating, auxiliary loss, prediction forward — is bitwise
identical to the baseline. The bias is added to the clean logits before
noise injection and top-k, so the original noise network, `W_h`, top-k,
softmax, `SparseDispatcher`, experts, and load-balancing loss all keep their
baseline definitions and behavior.

The branch is constructed under `torch.random.fork_rng`, so under the same
seed every pre-existing baseline parameter is bitwise identical to `main`;
only `transition_encoder.*` parameters are added.

The default `configs/config.yaml` enables the experiment via
`predictor.transition_aware_routing: true` (with the fixed
`transition_history_len: 4` and `transition_gru_hidden: 64`); no
experiment-specific CLI override is required.

## Difference from Base

The sole experimental variable is the additive VQ transition-aware routing
bias on the MoE clean logits. The transition representation does not enter
expert inputs, the Temporal Transformer, HyperFusion hidden representations,
factor heads, `LatentValueHead`, `ReturnPredictor`, alpha/beta, the final
prediction, or any other path, and it cannot backpropagate into Stage 1 (the
codebook lookup is detached and the frozen modules stay in eval with no
gradients). There is no transition-specific auxiliary loss. The router
structure, expert count, top-k, dispatcher, load-balancing loss and
`aux_weight`, data splits, daily batching, train-date shuffle, training
budget, early stopping, optimizer, learning rate, seed protocol, and
prediction/backtest protocols are all unchanged. No shared expert, adaptive
fusion, decoupling, quantization confidence, market-conditioned routing,
prior-latent allocation, code-aware bias, or continuous residual correction
mechanism is introduced.

Stage 1 provenance:

```text
external
artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt
```

This is the corrected PRISM-VQ exact single-VQ512, 128-dimensional Stage 1
checkpoint trained with the fixed Stage 1 seed 42; it loads into this
experiment's Stage 1 modules with `strict=True`.

## Smoke Status

PASS. The full unit suite (121 tests) and the minimal smoke
(`scripts/smoke_vq_transition_routing.py`) validate causal identity-keyed
history construction, loud failure on duplicate keys / malformed indices /
impossible chronology, shuffle-invariant history binding, label-free
code-map construction, live-quantizer current codes, detached
frozen-codebook lookups with no learnable code embedding,
consecutive-difference transition semantics with padding excluded, zero
state for empty history, zero-init bitwise equivalence with the baseline
(clean logits, routing/load, MoE output/aux loss, full forward, noisy
top-k/noise/`W_h`), bitwise initialization isolation of all pre-existing
parameters, finite non-zero gradients and updates for `W_t` and the GRU,
re-routing at fixed `z_q` once `W_t` is non-zero, GPU deterministic
backward, strict checkpoint round-trip, standard inference/metrics, and
backtest input normalization. Smoke artifacts — including history-length
distributions, adjacent code change rates, zero-transition ratios, and valid
transition sequence counts for both synthetic and real canonical splits —
are isolated under `artifacts/024/smoke/`.

No formal long-running training or portfolio backtest is performed in
Phase 1.
