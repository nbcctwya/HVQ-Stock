"""Minimal end-to-end Stage 2 smoke for experiment 038.

Experiment 038 = 028 temporal-attention Stage 1 + 019 routed-only Stage 2
(Quantization Confidence Adapter on the original FactorGatedMoE; 010's
always-on Shared Expert removed).  This is the w/o Shared-Routed ablation of
experiment 034.  This smoke reuses the exact Stage 1 provenance recorded by
experiment 028 (artifacts/028/run/.stage1.done), verifies the
temporal-attention encoder structure (Input Projection -> PositionalEncoding
-> TAttention -> TemporalAttention -> CrossAssetTransformer; no GRU, no
SAttention, no Market Gate), verifies strict Stage 1 loading, verifies the
Shared Expert is entirely absent while the routed path (2 Routed Experts,
top-k = 1, router/noise network/W_h/SparseDispatcher/auxiliary loss) is
intact, zero-init bitwise equivalence with the adapter-off model, exercises
routed forward/backward, adapter optimization, strict Stage 2 checkpoint
round-trip, validation/test inference, and the standard prediction format
consumed by the Phase 2/backtest pipeline.  It also records the
quantization-confidence diagnostics required for Phase 2 analysis (q_error
statistics, correction magnitude, adapter norms/gradients, routed gradient
health).  Diagnostics are observational only and do not modify the training
objective.
"""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest_qlib import _normalize_prediction_frame
from dataset.schema import TOTAL_DIM
from module.layers.encoder import (
    PositionalEncoding,
    TAttention,
    TemporalAttention,
    TemporalAttentionEncoder,
)
from module.quantise import VectorQuantiser
from trainer.train_ypred import GenerateReturn
from utils import run_inference, seed_everything


ARTIFACT_ROOT = ROOT / "artifacts" / "038" / "smoke"
STAGE1_MARKER = ROOT / "artifacts" / "028" / "run" / ".stage1.done"
# Final Experiment Commit of experiment 028 (exp/028-prism-temporal-attention-stage1).
EXPECTED_028_COMMIT = "4494d99542f40be7d3136ab42836f306631f0584"
EXPECTED_CHECKPOINT_BYTES = 14756745
EXPECTED_SPLITS = {
    "train_period": ["2009-01-01", "2020-12-31"],
    "valid_period": ["2021-01-01", "2022-12-31"],
    "test_period": ["2023-01-01", "2025-12-31"],
}
EPS = 1e-12


class SyntheticCanonicalDataset(Dataset):
    def __init__(self, seed, start_date, days=2, stocks_per_day=8):
        generator = torch.Generator().manual_seed(seed)
        count = days * stocks_per_day
        self.values = torch.randn(count, 20, TOTAL_DIM, generator=generator)
        dates = pd.bdate_range(start_date, periods=days)
        self.index = pd.MultiIndex.from_product(
            [dates, [f"SMOKE{i:03d}" for i in range(stocks_per_day)]],
            names=["datetime", "instrument"],
        )

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def get_index(self):
        return self.index


def file_md5(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_stage1_checkpoint():
    """Resolve and validate the exact Stage 1 provenance of experiment 028."""
    if not STAGE1_MARKER.is_file():
        raise FileNotFoundError(f"Stage 1 marker missing: {STAGE1_MARKER}")
    marker = {}
    for line in STAGE1_MARKER.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            marker[key.strip()] = value.strip()
    if marker.get("commit") != EXPECTED_028_COMMIT:
        raise AssertionError(
            f"028 Stage 1 marker commit {marker.get('commit')} != expected "
            f"028 commit {EXPECTED_028_COMMIT}"
        )
    checkpoint = Path(marker["best"])
    if not checkpoint.is_file():
        # The 028 marker records only the checkpoint filename; resolve it
        # against the marker's own run checkpoints directory.
        checkpoint = STAGE1_MARKER.parent / "checkpoints" / checkpoint.name
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"028 Stage 1 checkpoint missing or empty: {checkpoint}")
    return checkpoint, marker


def build_config(stage1_checkpoint, adapter=True):
    if not OmegaConf.has_resolver("half"):
        OmegaConf.register_new_resolver("half", lambda value: int(value) // 2)
    config = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs" / "config.yaml"), resolve=True
    )
    config["predictor"]["saved_model"] = str(stage1_checkpoint)
    config["predictor"]["quantization_confidence_adapter"] = adapter
    return config


def build_model(config):
    seed_everything(0)
    pl.seed_everything(0, workers=True)
    return GenerateReturn(copy.deepcopy(config), T_max=2)


def capture_strict_load(model, checkpoint):
    """Re-run load_pretrained_vqvae while recording strict load statistics."""
    captured = {}
    for name, module in (
        ("encoder", model.encoder),
        ("quantizer", model.quantizer),
        ("revin", model.revin),
    ):
        original = module.load_state_dict

        def wrapper(state_dict, strict=True, _name=name, _original=original):
            result = _original(state_dict, strict)
            captured[_name] = {
                "missing": len(result.missing_keys),
                "unexpected": len(result.unexpected_keys),
            }
            return result

        module.load_state_dict = wrapper
    model.load_pretrained_vqvae(str(checkpoint))
    for module in (model.encoder, model.quantizer, model.revin):
        module.eval()
    return captured


def verify_temporal_attention_encoder_structure(model, feature):
    encoder = model.encoder
    temporal = getattr(encoder, "temporal_encoder", None)
    if not isinstance(temporal, TemporalAttentionEncoder):
        raise AssertionError("Stage 1 encoder is not temporal-attention (temporal_encoder missing)")
    if not isinstance(temporal.positional_encoding, PositionalEncoding):
        raise AssertionError("Temporal-attention encoder positional encoding missing")
    if not isinstance(temporal.temporal_attention, TAttention):
        raise AssertionError("Temporal-attention encoder TAttention block missing")
    if not isinstance(temporal.temporal_aggregation, TemporalAttention):
        raise AssertionError("Temporal-attention encoder aggregation block missing")
    if any(isinstance(module, nn.GRU) for module in encoder.modules()):
        raise AssertionError("Stage 1 encoder still contains a GRU")
    if any(type(module).__name__ == "SAttention" for module in encoder.modules()):
        raise AssertionError("Stage 1 encoder must not contain SAttention")
    if any("gate" in name.lower() for name, _ in encoder.named_modules()):
        raise AssertionError("Stage 1 encoder still contains a Market Gate")

    seen = []
    modules = [
        ("feature_transform", encoder.feature_transform),
        ("input_projection", temporal.input_projection),
        ("positional_encoding", temporal.positional_encoding),
        ("tattention", temporal.temporal_attention),
        ("temporal_aggregation", temporal.temporal_aggregation),
        ("cross_asset_transformer", encoder.cross_asset_transformer),
    ]
    handles = [
        module.register_forward_hook(
            lambda _module, _inputs, _output, name=name: seen.append(name)
        )
        for name, module in modules
    ]
    try:
        with torch.no_grad():
            encoder(model.revin(feature, mode="norm"))
    finally:
        for handle in handles:
            handle.remove()
    if seen != [name for name, _ in modules]:
        raise AssertionError(f"Temporal-attention encoder data flow order changed: {seen}")
    return True


def main():
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = ARTIFACT_ROOT / "checkpoints"
    result_dir = ARTIFACT_ROOT / "res" / "combine_028_019_smoke"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    stage1_checkpoint, marker = resolve_stage1_checkpoint()
    checkpoint_bytes = stage1_checkpoint.stat().st_size
    checkpoint_md5 = file_md5(stage1_checkpoint)
    if checkpoint_bytes != EXPECTED_CHECKPOINT_BYTES:
        raise AssertionError("028 Stage 1 checkpoint size changed")

    config = build_config(stage1_checkpoint, adapter=True)
    predictor_cfg = config["predictor"]
    if "shared_expert" in predictor_cfg:
        raise AssertionError("configs/config.yaml must not contain 010 predictor.shared_expert")
    if predictor_cfg["quantization_confidence_adapter"] is not True:
        raise AssertionError("configs/config.yaml must keep 019 quantization_confidence_adapter")
    encoder_cfg = config["vqvae"]["encoder"]
    if encoder_cfg.get("type") != "temporal-attention":
        raise AssertionError("Default config must select the 028 temporal-attention encoder")
    if encoder_cfg.get("num_heads") != 2 or encoder_cfg.get("num_layers") != 1:
        raise AssertionError("028 temporal-attention encoder heads/layers changed")
    if encoder_cfg.get("temporal_dropout") != 0.1:
        raise AssertionError("028 temporal-attention encoder dropout changed")
    if config["vqvae"]["vq_embed_dim"] != 128:
        raise AssertionError("Stage 1 latent dimension must remain 128")
    if config["vqvae"]["num_embed"] != 512:
        raise AssertionError("Stage 1 must retain VQ512")
    if config["train"]["seed"] != 0:
        raise AssertionError("Stage 2 seed must remain 0")
    for key, expected in EXPECTED_SPLITS.items():
        if config["data"][key] != expected:
            raise AssertionError(f"Data split changed for {key}")

    # The adapter-off model is the same code with the adapter flag off; the
    # adapter is the only Stage 2 code difference inside experiment 038.
    base = build_model(build_config(stage1_checkpoint, adapter=False)).eval()
    model = build_model(config).eval()

    strict_load = capture_strict_load(model, stage1_checkpoint)
    for name, counts in strict_load.items():
        if counts["missing"] != 0 or counts["unexpected"] != 0:
            raise AssertionError(f"028 Stage 1 strict load failed for {name}: {counts}")

    if not isinstance(model.quantizer, VectorQuantiser):
        raise AssertionError("Stage 1 quantizer is not VectorQuantiser")
    quantizers = [m for m in model.modules() if isinstance(m, VectorQuantiser)]
    if len(quantizers) != 1:
        raise AssertionError("Stage 1 must contain exactly one quantizer")
    if tuple(model.quantizer.embedding.weight.shape) != (512, 128):
        raise AssertionError("Stage 1 codebook is not VQ512 x 128")

    moe = model.loadings.fusion.moe
    if hasattr(moe, "shared_expert"):
        raise AssertionError("010 always-on Shared Expert must be entirely absent")
    if any("shared" in name.lower() for name, _ in moe.named_parameters()):
        raise AssertionError("MoE still contains shared-expert parameters")
    if moe.num_experts != 2 or moe.k != 1:
        raise AssertionError("Routed configuration changed (n_expert/top-k)")
    for attribute in ("gate", "noise", "W_h", "softplus", "mean", "std"):
        if not hasattr(moe, attribute):
            raise AssertionError(f"Original router component missing: {attribute}")

    adapted_base_state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("quantization_confidence_adapter.")
    }
    if base.state_dict().keys() != adapted_base_state.keys():
        raise AssertionError("Adapter changed Stage 2 state_dict keys")
    existing_init_exact = all(
        torch.equal(value, adapted_base_state[key])
        for key, value in base.state_dict().items()
    )
    if not existing_init_exact:
        raise AssertionError("Adapter perturbed a Stage 2 parameter initialization")

    adapter = model.quantization_confidence_adapter
    adapter_zero_init = bool(
        torch.count_nonzero(adapter.weight) == 0
        and torch.count_nonzero(adapter.bias) == 0
    )
    if not adapter_zero_init or adapter.in_features != 1 or adapter.out_features != 128:
        raise AssertionError("Adapter must be zero-initialized Linear(1, 128)")

    train_data = SyntheticCanonicalDataset(10, "2020-01-02", days=1)
    train_batch = train_data.values
    feature = train_batch[:, :, :158]
    prior = train_batch[:, -1, 158:171]

    temporal_structure_ok = verify_temporal_attention_encoder_structure(model, feature)

    with torch.no_grad():
        feature_normalized = model.revin(feature, mode="norm")
        h_batch = model.encoder(feature_normalized)
        z_q, _, (_, _, vq_idx_before) = model.quantizer(h_batch)
        z_q = z_q.detach()
        expected_q_error = torch.mean((h_batch - z_q) ** 2, dim=-1, keepdim=True)
        q_error = model.quantization_error(h_batch, z_q)
        z_stage2 = model.build_stage2_latent(h_batch, z_q)
        base_output = base(feature, prior)
        adapted_output = model(feature, prior)

    if h_batch.shape[-1] != 128:
        raise AssertionError("028 Stage 1 pre-VQ latent h is not 128-dim")
    q_error_exact = torch.equal(q_error, expected_q_error)
    initial_latent_exact = torch.equal(z_stage2, z_q)
    initial_forward_exact = all(
        torch.equal(base_value, adapted_value)
        for base_value, adapted_value in zip(base_output, adapted_output)
    )
    if not q_error_exact:
        raise AssertionError("Quantization error does not match the required formula")
    if not initial_latent_exact:
        raise AssertionError("Zero-init z_conf is not bitwise equal to z_q")
    if not initial_forward_exact:
        raise AssertionError("Initial prediction forward is not bitwise equal to adapter-off")

    # One real training step: adapter must learn, Stage 1 must stay frozen,
    # and the routed path (experts + router) must keep receiving gradients.
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["train"]["learning_rate"],
        weight_decay=1e-5,
    )
    adapter_weight_before = adapter.weight.detach().clone()
    adapter_bias_before = adapter.bias.detach().clone()
    codebook_before = model.quantizer.embedding.weight.detach().clone()
    optimizer.zero_grad()
    y_pred, _, _, _, aux_loss = model(feature, prior)
    loss = model.rank_loss(y_pred, train_batch[:, -1, 238]) + model.aux_weight * aux_loss
    loss.backward()

    adapter_weight_grad_l1 = float(adapter.weight.grad.abs().sum())
    adapter_bias_grad_l1 = float(adapter.bias.grad.abs().sum())
    if adapter_weight_grad_l1 <= 0 or adapter_bias_grad_l1 <= 0:
        raise AssertionError("Quantization-confidence adapter did not receive gradient")
    if not torch.isfinite(adapter.weight.grad).all():
        raise AssertionError("Adapter weight gradient is not finite")

    routed_grads = [
        float(p.grad.abs().sum())
        for p in moe.experts.parameters()
        if p.grad is not None
    ]
    routed_grad_l1 = sum(routed_grads)
    if not routed_grads or routed_grad_l1 <= 0:
        raise AssertionError("Routed Experts lost their training signal")
    router_grads = [
        float(p.grad.abs().sum())
        for p in moe.gate.parameters()
        if p.grad is not None
    ]
    router_grad_l1 = sum(router_grads)
    if not router_grads or router_grad_l1 <= 0:
        raise AssertionError("Original router lost its training signal")
    frozen_stage1_has_grad = any(
        p.grad is not None
        for module in (model.encoder, model.quantizer, model.revin)
        for p in module.parameters()
    )
    if frozen_stage1_has_grad:
        raise AssertionError("Frozen Stage 1 parameters received gradients")

    optimizer.step()
    adapter_updated = bool(
        not torch.equal(adapter_weight_before, adapter.weight.detach())
        and not torch.equal(adapter_bias_before, adapter.bias.detach())
    )
    if not adapter_updated:
        raise AssertionError("Quantization-confidence adapter did not update")
    if not torch.equal(codebook_before, model.quantizer.embedding.weight.detach()):
        raise AssertionError("Codebook changed during the Stage 2 step")
    for module in (model.encoder, model.quantizer, model.revin):
        if module.training:
            raise AssertionError("Frozen Stage 1 module left eval mode")

    # Post-update diagnostics (observational only).
    model.eval()
    with torch.no_grad():
        h_after = model.encoder(model.revin(feature, mode="norm"))
        z_q_after, _, (_, _, vq_idx_after) = model.quantizer(h_after)
        z_q_after = z_q_after.detach()
        q_error_after = model.quantization_error(h_after, z_q_after)
        z_conf_after = model.build_stage2_latent(h_after, z_q_after)
        correction = z_conf_after - z_q_after
        correction_norm = float(correction.norm(dim=-1).mean())
        z_q_norm = float(z_q_after.norm(dim=-1).mean())
        relative_correction = correction_norm / (z_q_norm + EPS)
    if not torch.equal(vq_idx_before, vq_idx_after):
        raise AssertionError("Quantizer assignment changed through the adapter path")

    diagnostics = {
        "q_error_mean": float(q_error_after.mean()),
        "q_error_std": float(q_error_after.std()),
        "q_error_quantiles": {
            "p05": float(torch.quantile(q_error_after.flatten(), 0.05)),
            "p25": float(torch.quantile(q_error_after.flatten(), 0.25)),
            "p50": float(torch.quantile(q_error_after.flatten(), 0.50)),
            "p75": float(torch.quantile(q_error_after.flatten(), 0.75)),
            "p95": float(torch.quantile(q_error_after.flatten(), 0.95)),
        },
        "correction_l2_norm_mean": correction_norm,
        "z_q_l2_norm_mean": z_q_norm,
        "relative_correction_magnitude": relative_correction,
        "adapter_weight_l2_norm": float(adapter.weight.norm()),
        "adapter_bias_l2_norm": float(adapter.bias.norm()),
        "adapter_weight_grad_l1": adapter_weight_grad_l1,
        "adapter_bias_grad_l1": adapter_bias_grad_l1,
        "routed_experts_grad_l1": routed_grad_l1,
        "router_grad_l1": router_grad_l1,
    }

    checkpoint_path = checkpoint_dir / "combine-028-019-smoke.ckpt"
    torch.save(
        {
            "epoch": 0,
            "global_step": 1,
            "pytorch-lightning_version": pl.__version__,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    restored = GenerateReturn.load_from_checkpoint(
        checkpoint_path,
        config=copy.deepcopy(config),
        T_max=2,
        strict=True,
    )
    restored.freeze_vqvae()
    restored.eval()
    with torch.no_grad():
        reference = model(feature, prior)
        reloaded = restored(feature, prior)
    checkpoint_exact = all(
        torch.equal(reference_value, reloaded_value)
        for reference_value, reloaded_value in zip(reference, reloaded)
    )
    if not checkpoint_exact:
        raise AssertionError("Strict checkpoint round-trip changed model output")

    inference_results = {}
    for split, split_seed, start_date in (
        ("valid", 20, "2021-01-04"),
        ("test", 30, "2023-01-03"),
    ):
        dataset = SyntheticCanonicalDataset(split_seed, start_date)
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        prediction, _, metrics = run_inference(restored, loader, config, device="cpu")
        prediction_path = result_dir / f"0_{split}.pkl"
        metric_path = result_dir / f"0_{split}_metric.csv"
        prediction.to_pickle(prediction_path)
        pd.DataFrame([metrics], index=["values"]).transpose().to_csv(metric_path)
        normalized, signal = _normalize_prediction_frame(prediction_path)
        if len(normalized) != len(dataset) or len(signal) != len(dataset):
            raise AssertionError(
                f"{split} prediction is incompatible with backtest normalizer"
            )
        inference_results[split] = {
            "rows": len(prediction),
            "prediction": str(prediction_path.relative_to(ROOT)),
            "metric": str(metric_path.relative_to(ROOT)),
            "metrics": {key: float(value) for key, value in metrics.items()},
        }

    # Match the standard Phase 2 filename in addition to retaining explicit
    # valid/test files for smoke auditability.
    test_prediction = pd.read_pickle(result_dir / "0_test.pkl")
    standard_prediction_path = result_dir / "0_best.pkl"
    test_prediction.to_pickle(standard_prediction_path)
    test_metrics = pd.read_csv(result_dir / "0_test_metric.csv", index_col=0)
    standard_metric_path = result_dir / "0_metric.csv"
    test_metrics.to_csv(standard_metric_path)
    _normalize_prediction_frame(standard_prediction_path)

    report = {
        "status": "PASS",
        "stage1": {
            "source": "028",
            "marker": str(STAGE1_MARKER.relative_to(ROOT)),
            "marker_commit": marker.get("commit"),
            "expected_028_commit": EXPECTED_028_COMMIT,
            "checkpoint": str(stage1_checkpoint.relative_to(ROOT)),
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_md5": checkpoint_md5,
            "strict_load": strict_load,
            "encoder_structure": "InputProjection->PositionalEncoding->TAttention->TemporalAttention->CrossAssetTransformer",
            "temporal_attention_structure_verified": temporal_structure_ok,
            "no_gru_no_sattention_no_market_gate": True,
            "single_vq512": True,
            "embedding_dimension": 128,
            "data_splits": EXPECTED_SPLITS,
            "frozen_parameters_have_grad": frozen_stage1_has_grad,
        },
        "stage2_routed_only": {
            "shared_expert_present": False,
            "shared_expert_config_key_present": False,
            "num_routed_experts": moe.num_experts,
            "top_k": moe.k,
            "router_components_intact": True,
        },
        "quantization_confidence_adapter": {
            "module": "Linear(1, 128)",
            "zero_initialized": adapter_zero_init,
            "q_error_formula_exact": q_error_exact,
            "q_error_requires_grad": bool(q_error.requires_grad),
            "initial_z_conf_bitwise_equal_z_q": initial_latent_exact,
            "initial_prediction_forward_bitwise_equal_adapter_off": initial_forward_exact,
            "existing_parameter_initialization_bitwise_equal_adapter_off": existing_init_exact,
            "updated": adapter_updated,
            "quantizer_assignment_unchanged": True,
        },
        "diagnostics": diagnostics,
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "strict_round_trip": True,
            "output_bitwise_equal": checkpoint_exact,
        },
        "inference": inference_results,
        "standard_outputs": {
            "prediction": str(standard_prediction_path.relative_to(ROOT)),
            "metric": str(standard_metric_path.relative_to(ROOT)),
            "backtest_normalizer": "PASS",
        },
    }
    report_path = ARTIFACT_ROOT / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
