# 015 — AlphaMaster VQ Continuous Warm-up

## Base

`exp/012-alphamaster-discrete-market-adapter`

## Idea / Motivation

Experiment 012 quantizes the GRU historical-market state from the first
training step. This experiment tests whether first learning a prediction-useful
continuous market representation makes the later discrete-regime phase more
stable and improves prediction-side market adaptation.

The model is structurally identical to 012 from initialization: the same GRU,
Standard VQ, zero-initialized Market Adapter, Feature Gate, MASTER backbone,
and decoder residual are all present at epoch 0. Only the training route and
the eligibility boundary for formal validation selection change.

## Core modification

The default configuration declares `train.warmup_epochs: 10` and retains the
70-epoch budget.

During epochs 0–9, Standard VQ is bypassed and receives no gradient:

```text
market[:, :-1, :] -> GRU -> continuous m_t
                                  -> Market Adapter -> prediction residual

loss = prediction loss
adapter_input = m_t
```

Starting at epoch 10, the forward path and objective are exactly those of 012:

```text
market[:, :-1, :] -> GRU -> m_t -> Standard VQ -> z_q
                                                   -> Market Adapter
                                                   -> prediction residual

loss = prediction loss + VQ loss
adapter_input = z_q
```

The implementation uses the single boundary
`use_vq = current_epoch >= warmup_epochs`. Normal inference defaults to the VQ
path, so Stage 2 evaluates the formal discrete model.

Formal `val_loss` early stopping and best-checkpoint selection use warm-up-aware
callbacks. Epochs 0–9 still run validation and log metrics, but neither callback
updates its state or saves a candidate. At epoch 10 both callbacks begin with
fresh state; epochs 10–69 use the unchanged 012 monitor (`val_loss`), mode
(`min`), `min_delta=1e-5`, patience 15, and top-1 checkpoint rule.

At the epoch-10 validation boundary, the model records mean quantization
distortion, trading-day code counts, active-code count, perplexity, and
continuous-versus-quantized prediction MAE/RMSE/max absolute difference. The
continuous counterfactual reuses the same backbone hidden state, GRU state, and
Market Adapter parameters as the quantized prediction.

## Difference from base

- Added exactly 10 epochs of continuous GRU-to-adapter warm-up before enabling
  the existing Standard VQ and VQ loss.
- Delayed formal early-stopping state and best-checkpoint eligibility to epoch
  10 so the two different validation objectives are never compared.
- Added switch-point diagnostics; these are observational and do not affect
  optimization.
- Kept the 012 Standard VQ unchanged: `K=8`, `D=63`, squared-L2 nearest
  neighbor, straight-through estimator, and commitment weight 0.25.
- Kept the single-layer `GRU(63,63)`, `Linear(63,256,bias=False)` Market Adapter
  with zero initialization, current-market Feature Gate, MASTER backbone,
  decoder residual equation, input schema, data splits, optimizer, seed
  protocol, all other hyperparameters, metrics, and Top30/Drop5 backtest
  protocol unchanged.
- Stage 1 provenance is `self`: the formal training flow changes, so the whole
  model must be retrained. No formal long-running training was started in
  Phase 1.

## Smoke status

PASS. The complete `unittest` suite passes 96/96 tests. New coverage verifies
the exact epoch boundary, true VQ bypass, continuous adapter input,
prediction-only warm-up loss, GRU/Adapter gradients with an untouched codebook,
restoration of the exact 012 VQ objective at epoch 10, and delayed callback
activation.

The isolated CPU smoke ran epochs 0–10 with two train and two validation batches
per epoch. It produced only the formally eligible checkpoint
`alphamaster_smoke-epoch=10-val_loss=0.5284.ckpt`. At the switch, validation
reported distortion `0.088955`, code usage `[0,0,0,0,1,0,0,1]`, 2/8 active
codes, perplexity `2.0`, and continuous/quantized prediction difference MAE
`0.040977`, RMSE `0.045349`, max absolute `0.065437`.

The checkpoint strict-loaded in Stage 2, produced the standard 40-row
prediction and metric files, and was accepted by the existing backtest
normalizer. Post-smoke test inference used 5/8 codes across 10 trading days,
with counts `[0,0,4,1,1,0,2,2]`; every cross-section shared one regime.

All smoke data, logs, checkpoint, predictions, metrics, and the machine-readable
report are isolated under `artifacts/015/smoke/`.
