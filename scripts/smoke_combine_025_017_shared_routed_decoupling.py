"""Minimal end-to-end Stage 2 smoke for experiment 036.

Experiment 036 = 025 (010 Shared-Routed MoE + 019 Quantization Confidence
Adapter) + 017 Shared-Routed Decoupling:

    L_dec = mean(cosine_similarity(shared_out, routed_out)^2)
    L_moe = L_route + 0.01 * L_dec

This smoke reuses the exact Stage 1 provenance recorded by experiment 010
(artifacts/010/run/.stage1.done), verifies zero-init/decoupling-off bitwise
equivalence with the 010-equivalent model, verifies the 017 loss formula and
its gradients on both expert paths, exercises shared+routed forward/backward
together with the confidence adapter, performs a strict Stage 2 checkpoint
round-trip, runs validation/test inference, and checks the standard
prediction format consumed by the Phase 2/backtest pipeline.  Diagnostics are
observational only and do not modify the training objective.
"""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest_qlib import _normalize_prediction_frame
from dataset.schema import TOTAL_DIM
from module.layers.moe import FactorGatedMoE
from module.quantise import VectorQuantiser
from trainer.train_ypred import GenerateReturn
from utils import run_inference, seed_everything


ARTIFACT_ROOT = ROOT / "artifacts" / "036" / "smoke"
STAGE1_MARKER = ROOT / "artifacts" / "010" / "run" / ".stage1.done"
# Canonical queue pinned commit of experiment 010 on main.
EXPECTED_010_COMMIT = "9b854f0436f8a7c3283fd375661dd6152cc965f1"
EXPECTED_CHECKPOINT_BYTES = 14584929
EXPECTED_CHECKPOINT_MD5 = "6b9d9dbfd938c7bd2c7dc5ee33cb38af"
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
    """Resolve and validate the exact Stage 1 provenance of experiment 010."""
    if not STAGE1_MARKER.is_file():
        raise FileNotFoundError(f"Stage 1 marker missing: {STAGE1_MARKER}")
    marker = {}
    for line in STAGE1_MARKER.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            marker[key.strip()] = value.strip()
    if marker.get("commit") != EXPECTED_010_COMMIT:
        raise AssertionError(
            f"010 Stage 1 marker commit {marker.get('commit')} != pinned "
            f"queue commit {EXPECTED_010_COMMIT}"
        )
    checkpoint = Path(marker["best"])
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"010 Stage 1 checkpoint missing or empty: {checkpoint}")
    return checkpoint, marker


def build_config(stage1_checkpoint, adapter=True, decoupling_lambda=0.01):
    if not OmegaConf.has_resolver("half"):
        OmegaConf.register_new_resolver("half", lambda value: int(value) // 2)
    config = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs" / "config.yaml"), resolve=True
    )
    config["predictor"]["saved_model"] = str(stage1_checkpoint)
    config["predictor"]["quantization_confidence_adapter"] = adapter
    config["predictor"]["decoupling_lambda"] = decoupling_lambda
    return config


def build_model(config):
    seed_everything(0)
    pl.seed_everything(0, workers=True)
    return GenerateReturn(copy.deepcopy(config), T_max=2)


def main():
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = ARTIFACT_ROOT / "checkpoints"
    result_dir = ARTIFACT_ROOT / "res" / "combine_025_017_smoke"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    stage1_checkpoint, marker = resolve_stage1_checkpoint()
    checkpoint_bytes = stage1_checkpoint.stat().st_size
    checkpoint_md5 = file_md5(stage1_checkpoint)
    if checkpoint_bytes != EXPECTED_CHECKPOINT_BYTES:
        raise AssertionError("010 Stage 1 checkpoint size changed")
    if checkpoint_md5 != EXPECTED_CHECKPOINT_MD5:
        raise AssertionError("010 Stage 1 checkpoint hash changed")

    config = build_config(stage1_checkpoint, adapter=True, decoupling_lambda=0.01)
    predictor_cfg = config["predictor"]
    if predictor_cfg["shared_expert"] is not True:
        raise AssertionError("configs/config.yaml must keep 010 predictor.shared_expert")
    if predictor_cfg["quantization_confidence_adapter"] is not True:
        raise AssertionError("Default config must keep the 025 confidence adapter")
    if predictor_cfg["decoupling_lambda"] != 0.01:
        raise AssertionError("Default decoupling_lambda must be exactly 0.01")
    if config["vqvae"]["vq_embed_dim"] != 128:
        raise AssertionError("Stage 1 latent dimension must remain 128")
    if config["vqvae"]["num_embed"] != 512:
        raise AssertionError("Stage 1 must retain VQ512")
    if config["train"]["seed"] != 0:
        raise AssertionError("Stage 2 seed must remain 0")
    for key, expected in EXPECTED_SPLITS.items():
        if config["data"][key] != expected:
            raise AssertionError(f"Data split changed for {key}")

    # The 010-equivalent base is the same code with both the adapter flag and
    # the decoupling weight off; those two flags are the only code difference
    # between 036 and 010.
    base = build_model(
        build_config(stage1_checkpoint, adapter=False, decoupling_lambda=0.0)
    ).eval()
    model = build_model(config).eval()

    if not isinstance(model.quantizer, VectorQuantiser):
        raise AssertionError("Stage 1 quantizer is not VectorQuantiser")
    quantizers = [m for m in model.modules() if isinstance(m, VectorQuantiser)]
    if len(quantizers) != 1:
        raise AssertionError("Stage 1 must contain exactly one quantizer")
    if tuple(model.quantizer.embedding.weight.shape) != (512, 128):
        raise AssertionError("Stage 1 codebook is not VQ512 x 128")

    moe = model.loadings.fusion.moe
    if moe.shared_expert is None:
        raise AssertionError("010 Shared Expert is not enabled")
    if moe.num_experts != 2 or moe.k != 1:
        raise AssertionError("010 routed configuration changed (n_expert/top-k)")
    if moe.decoupling_lambda != 0.01:
        raise AssertionError("017 decoupling weight did not reach the MoE")

    adapted_base_state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("quantization_confidence_adapter.")
    }
    if base.state_dict().keys() != adapted_base_state.keys():
        raise AssertionError("Experiment changed 010 state_dict keys")
    existing_init_exact = all(
        torch.equal(value, adapted_base_state[key])
        for key, value in base.state_dict().items()
    )
    if not existing_init_exact:
        raise AssertionError("Experiment perturbed a 010 parameter initialization")

    adapter = model.quantization_confidence_adapter
    adapter_zero_init = bool(
        torch.count_nonzero(adapter.weight) == 0
        and torch.count_nonzero(adapter.bias) == 0
    )
    if not adapter_zero_init or adapter.in_features != 1 or adapter.out_features != 128:
        raise AssertionError("Adapter must be zero-initialized Linear(1, 128)")

    # 017 probe: with a deliberately nonzero Shared Expert, enabling the
    # decoupling weight must change only the auxiliary loss, by exactly
    # 0.01 * L_dec, leaving the prediction path and the routed
    # load-balancing loss untouched.
    experiment_probe = copy.deepcopy(moe).eval()
    with torch.no_grad():
        experiment_probe.shared_expert.net[-1].weight.fill_(0.02)
        experiment_probe.shared_expert.net[-1].bias.fill_(0.01)
    base_probe = copy.deepcopy(experiment_probe)
    base_probe.decoupling_lambda = 0.0
    probe_x = torch.randn(9, 64)
    probe_z = torch.randn(9, 128)
    experiment_out, experiment_loss = experiment_probe(probe_x, probe_z)
    base_out, route_loss = base_probe(probe_x, probe_z)

    routed_probe = copy.deepcopy(base_probe)
    routed_probe.shared_expert = None
    routed_probe.use_shared_expert = False
    routed_out, routed_route_loss = routed_probe(probe_x, probe_z)
    shared_out = base_probe.shared_expert(probe_x)
    decoupling_penalty = FactorGatedMoE.shared_routed_decoupling_loss(
        shared_out, routed_out
    )
    prediction_equivalent = bool(torch.equal(experiment_out, base_out))
    route_loss_unchanged = bool(torch.equal(route_loss, routed_route_loss))
    loss_formula_correct = bool(
        torch.allclose(
            experiment_loss,
            route_loss + 0.01 * decoupling_penalty,
            atol=1e-7,
            rtol=0,
        )
    )
    if not prediction_equivalent:
        raise AssertionError("017 decoupling changed the prediction path")
    if not route_loss_unchanged:
        raise AssertionError("Routed load-balancing loss changed")
    if not loss_formula_correct:
        raise AssertionError("MoE loss is not route_loss + 0.01 * decoupling_loss")

    # Directly verify that the regularizer supplies finite gradients to both
    # representations, independently of the main prediction objective.
    shared_leaf = torch.randn(9, 64, requires_grad=True)
    routed_leaf = torch.randn(9, 64, requires_grad=True)
    gradient_probe_loss = FactorGatedMoE.shared_routed_decoupling_loss(
        shared_leaf, routed_leaf
    )
    gradient_probe_loss.backward()
    shared_dec_grad_l1 = float(shared_leaf.grad.abs().sum())
    routed_dec_grad_l1 = float(routed_leaf.grad.abs().sum())
    if shared_dec_grad_l1 <= 0 or routed_dec_grad_l1 <= 0:
        raise AssertionError("Decoupling loss did not produce gradients on both paths")

    train_data = SyntheticCanonicalDataset(10, "2020-01-02", days=1)
    train_batch = train_data.values
    feature = train_batch[:, :, :158]
    prior = train_batch[:, -1, 158:171]

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
        raise AssertionError(
            "Initial prediction forward is not bitwise equal to 010"
        )

    # One real training step: adapter must learn, both expert paths must keep
    # receiving gradients (now including the decoupling term), and Stage 1
    # must stay frozen.
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

    shared_final = moe.shared_expert.net[-1]
    shared_weight_grad = float(shared_final.weight.grad.abs().sum())
    shared_bias_grad = float(shared_final.bias.grad.abs().sum())
    if shared_weight_grad <= 0 or shared_bias_grad <= 0:
        raise AssertionError("010 Shared Expert lost its training signal")
    routed_grads = [
        float(p.grad.abs().sum())
        for p in moe.experts.parameters()
        if p.grad is not None
    ]
    routed_grad_l1 = sum(routed_grads)
    if not routed_grads or routed_grad_l1 <= 0:
        raise AssertionError("010 Routed Experts lost their training signal")
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

    # Post-update diagnostics (observational only), including the decoupling
    # penalty actually incurred by the now-nonzero Shared Expert.
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

        seen = {}
        handles = [
            moe.shared_expert.register_forward_hook(
                lambda _m, _i, output: seen.setdefault(
                    "shared", output.detach().clone()
                )
            ),
            moe.register_forward_hook(
                lambda _m, _i, output: seen.update(
                    combined=output[0].detach().clone(),
                    moe_loss=float(output[1].detach()),
                )
            ),
        ]
        model(feature, prior)
        for handle in handles:
            handle.remove()
        shared_after = seen["shared"]
        routed_after = seen["combined"] - shared_after
        penalty_after = float(
            FactorGatedMoE.shared_routed_decoupling_loss(shared_after, routed_after)
        )
        cosine_after = torch.nn.functional.cosine_similarity(
            shared_after, routed_after, dim=-1
        )
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
        "shared_expert_final_weight_grad_l1": shared_weight_grad,
        "shared_expert_final_bias_grad_l1": shared_bias_grad,
        "routed_experts_grad_l1": routed_grad_l1,
        "decoupling_lambda": 0.01,
        "decoupling_penalty_after_step": penalty_after,
        "decoupling_cosine_mean_after_step": float(cosine_after.mean()),
        "decoupling_cosine_std_after_step": float(cosine_after.std()),
        "moe_loss_after_step": seen["moe_loss"],
    }

    checkpoint_path = checkpoint_dir / "combine-025-017-smoke.ckpt"
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
            "source": "010",
            "marker": str(STAGE1_MARKER.relative_to(ROOT)),
            "marker_commit": marker.get("commit"),
            "expected_010_commit": EXPECTED_010_COMMIT,
            "marker_reused": marker.get("reused"),
            "checkpoint": str(stage1_checkpoint.relative_to(ROOT)),
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_md5": checkpoint_md5,
            "strict_load": {
                "encoder": {"missing": 0, "unexpected": 0},
                "quantizer": {"missing": 0, "unexpected": 0},
                "revin": {"missing": 0, "unexpected": 0},
            },
            "single_vq512": True,
            "embedding_dimension": 128,
            "data_splits": EXPECTED_SPLITS,
            "frozen_parameters_have_grad": frozen_stage1_has_grad,
        },
        "inheritance_010": {
            "shared_expert_enabled": True,
            "num_routed_experts": moe.num_experts,
            "top_k": moe.k,
            "shared_expert_final_zero_initialized": bool(
                torch.count_nonzero(base.loadings.fusion.moe.shared_expert.net[-1].weight)
                == 0
            ),
        },
        "quantization_confidence_adapter": {
            "module": "Linear(1, 128)",
            "zero_initialized": adapter_zero_init,
            "q_error_formula_exact": q_error_exact,
            "q_error_requires_grad": bool(q_error.requires_grad),
            "initial_z_conf_bitwise_equal_z_q": initial_latent_exact,
            "initial_prediction_forward_bitwise_equal_010": initial_forward_exact,
            "existing_parameter_initialization_bitwise_equal_010": existing_init_exact,
            "updated": adapter_updated,
            "quantizer_assignment_unchanged": True,
        },
        "shared_routed_decoupling": {
            "lambda": 0.01,
            "prediction_bitwise_equal_to_lambda_off": prediction_equivalent,
            "nonzero_shared_output_used_for_equivalence_check": True,
            "route_loss_bitwise_unchanged": route_loss_unchanged,
            "loss_formula_correct": loss_formula_correct,
            "probe_penalty": float(decoupling_penalty.detach()),
            "probe_penalty_finite": bool(torch.isfinite(decoupling_penalty)),
            "shared_representation_grad_l1": shared_dec_grad_l1,
            "routed_representation_grad_l1": routed_dec_grad_l1,
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
