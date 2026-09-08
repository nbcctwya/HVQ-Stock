"""Minimal Stage 2 smoke for experiment 018.

Reuses experiment 010's exact Stage 1 artifact and validates the independent
raw Shared/Routed penalty without changing experiment 016 prediction outputs.
"""

import copy
import json
import sys
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest_qlib import _normalize_prediction_frame
from module.layers.moe import FactorGatedMoE
from module.quantise import VectorQuantiser
from scripts.smoke_adaptive_shared_fusion import (
    EXPECTED_SPLITS,
    SyntheticCanonicalDataset,
    build_config,
    resolve_stage1_checkpoint,
)
from trainer.train_ypred import GenerateReturn, softcap_log1p
from utils import run_inference, seed_everything


ARTIFACT_ROOT = ROOT / "artifacts" / "018" / "smoke"


def loss_components(model, batch):
    feature = batch[:, :, :158]
    prior = batch[:, -1, 158:171]
    label = batch[:, -1, 238]
    outputs = model.forward_with_decoupling_loss(feature, prior)
    prediction, _, _, _, aux_loss, decoupling_loss = outputs
    rank_loss = model.rank_loss(prediction, label)
    total_loss = model._total_objective(rank_loss, aux_loss, decoupling_loss)
    return total_loss, rank_loss, aux_loss, decoupling_loss, outputs


def main():
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = ARTIFACT_ROOT / "checkpoints"
    result_dir = ARTIFACT_ROOT / "res" / "adaptive_shared_decoupling_smoke"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    marker, stage1_checkpoint = resolve_stage1_checkpoint()
    seed_everything(0)
    pl.seed_everything(0, workers=True)
    config = build_config(stage1_checkpoint)

    expected_predictor = {
        "shared_expert": True,
        "adaptive_shared_fusion": True,
        "decoupling_lambda": 0.01,
        "aux_weight": 0.01,
        "aux_imp": 3,
    }
    for key, expected in expected_predictor.items():
        if config["predictor"][key] != expected:
            raise AssertionError(f"Unexpected predictor.{key}")
    if config["vqvae"]["num_embed"] != 512:
        raise AssertionError("Stage 1 must retain single VQ512")
    for key, expected in EXPECTED_SPLITS.items():
        if config["data"][key] != expected:
            raise AssertionError(f"Data split changed for {key}")

    # A 016 config has no decoupling key. The new scalar objective weight adds
    # no parameters and must not perturb any existing seeded initialization.
    config_016 = copy.deepcopy(config)
    config_016["predictor"].pop("decoupling_lambda")
    seed_everything(0)
    pl.seed_everything(0, workers=True)
    model_016 = GenerateReturn(config_016, T_max=2)
    seed_everything(0)
    pl.seed_everything(0, workers=True)

    # Construction performs strict Encoder/Quantizer/RevIN Stage 1 loading.
    model = GenerateReturn(config, T_max=2)
    if not isinstance(model.quantizer, VectorQuantiser):
        raise AssertionError("Stage 1 quantizer is not the single VectorQuantiser")
    if tuple(model.quantizer.embedding.weight.shape) != (512, 128):
        raise AssertionError("Stage 1 codebook is not VQ512 with dimension 128")
    state_016 = model_016.state_dict()
    state_018 = model.state_dict()
    parameter_initialization_equal = bool(
        state_016.keys() == state_018.keys()
        and all(torch.equal(state_016[key], state_018[key]) for key in state_016)
    )
    if not parameter_initialization_equal:
        raise AssertionError("018 perturbed an existing 016 parameter initialization")

    moe = model.loadings.fusion.moe
    if moe.shared_expert is None or moe.shared_fusion is None:
        raise AssertionError("Experiment 016 adaptive shared path is not enabled")
    fusion_zero_init = bool(
        torch.count_nonzero(moe.shared_fusion.weight) == 0
        and torch.count_nonzero(moe.shared_fusion.bias) == 0
    )
    if not fusion_zero_init:
        raise AssertionError("Latent-to-scale map is not zero-initialized")

    # Use nonzero Shared output and nonconstant alpha. The standard prediction
    # path is the exact 016 interface; the detailed 018 path may only append L_dec.
    probe = copy.deepcopy(moe).eval()
    with torch.no_grad():
        probe.shared_expert.net[-1].weight.normal_(0.0, 0.1)
        probe.shared_expert.net[-1].bias.normal_(0.0, 0.1)
        probe.shared_fusion.weight.normal_(0.0, 0.2)
        probe.shared_fusion.bias.fill_(0.1)
    probe_x = torch.randn(9, 64)
    probe_z = torch.randn(9, 128)
    standard_output, standard_aux = probe(
        probe_x, probe_z, shared_condition=probe_z
    )
    detailed_output, detailed_aux, probe_dec = probe(
        probe_x,
        probe_z,
        shared_condition=probe_z,
        return_decoupling_loss=True,
    )
    moe_prediction_equivalence = bool(
        torch.equal(standard_output, detailed_output)
        and torch.equal(standard_aux, detailed_aux)
    )
    if not moe_prediction_equivalence:
        raise AssertionError("018 prediction or original auxiliary loss changed")

    raw_shared = probe.shared_expert(probe_x)
    alpha = 1.0 + 0.5 * torch.tanh(probe.shared_fusion(probe_z))
    routed = detailed_output - alpha * raw_shared
    expected_dec = FactorGatedMoE.shared_routed_decoupling_loss(
        raw_shared, routed
    )
    if not torch.allclose(probe_dec, expected_dec, atol=1e-7, rtol=0):
        raise AssertionError("L_dec was not computed from raw Shared/Routed outputs")
    if not torch.all((alpha > 0.5) & (alpha < 1.5)):
        raise AssertionError("Adaptive shared alpha left its 016 bounds")

    identical = FactorGatedMoE.shared_routed_decoupling_loss(
        raw_shared, raw_shared
    )
    orthogonal_shared = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    orthogonal_routed = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    orthogonal = FactorGatedMoE.shared_routed_decoupling_loss(
        orthogonal_shared, orthogonal_routed
    )
    if not torch.isfinite(probe_dec) or probe_dec < 0:
        raise AssertionError("L_dec must be finite and non-negative")
    if not torch.allclose(identical, torch.tensor(1.0)) or orthogonal != 0:
        raise AssertionError("L_dec similarity diagnostics failed")

    train_data = SyntheticCanonicalDataset(10, "2020-01-02", days=1)
    train_batch = train_data.values
    model_016.eval()
    model.eval()
    with torch.no_grad():
        output_016 = model_016(
            train_batch[:, :, :158], train_batch[:, -1, 158:171]
        )
        output_018 = model.forward_with_decoupling_loss(
            train_batch[:, :, :158], train_batch[:, -1, 158:171]
        )
    prediction_equivalence = all(
        torch.equal(expected, actual)
        for expected, actual in zip(output_016, output_018[:5])
    )
    if not prediction_equivalence:
        raise AssertionError("Full 018 prediction/aux is not bitwise equal to 016")
    del model_016

    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["train"]["learning_rate"],
        weight_decay=1e-5,
    )

    # First prediction step moves the zero-initialized Shared Expert so the
    # cosine penalty has a meaningful nonzero representation on the next step.
    shared_final = moe.shared_expert.net[-1]
    shared_before = shared_final.weight.detach().clone()
    optimizer.zero_grad()
    first_total, _, _, _, _ = loss_components(model, train_batch)
    first_total.backward()
    optimizer.step()
    if torch.equal(shared_before, shared_final.weight.detach()):
        raise AssertionError("Shared Expert did not update")

    optimizer.zero_grad()
    total_loss, rank_loss, aux_loss, decoupling_loss, _ = loss_components(
        model, train_batch
    )
    expected_total = (
        rank_loss
        + model.aux_weight * aux_loss
        + 0.01 * decoupling_loss
    )
    if not torch.equal(total_loss, expected_total):
        raise AssertionError("Top-level objective formula is incorrect")
    wrong_aux_scaled = rank_loss + model.aux_weight * (
        aux_loss + 0.01 * decoupling_loss
    )
    wrong_softcapped = (
        rank_loss
        + model.aux_weight * aux_loss
        + 0.01 * softcap_log1p(decoupling_loss, model.aux_imp)
    )
    if torch.equal(total_loss, wrong_aux_scaled) or torch.equal(
        total_loss, wrong_softcapped
    ):
        raise AssertionError("L_dec entered the original auxiliary pipeline")

    path_parameters = [shared_final.weight] + [
        expert.net[-1].weight for expert in moe.experts
    ]
    path_gradients = torch.autograd.grad(
        decoupling_loss,
        path_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    shared_dec_grad_l1 = float(path_gradients[0].abs().sum())
    routed_dec_grad_l1 = float(
        sum(
            gradient.abs().sum()
            for gradient in path_gradients[1:]
            if gradient is not None
        )
    )
    if shared_dec_grad_l1 <= 0 or routed_dec_grad_l1 <= 0:
        raise AssertionError("L_dec did not reach both Shared/Routed paths")
    total_loss.backward()
    optimizer.step()

    frozen_stage1_has_grad = any(
        parameter.grad is not None
        for module in (model.encoder, model.quantizer, model.revin)
        for parameter in module.parameters()
    )
    if frozen_stage1_has_grad:
        raise AssertionError("Frozen Stage 1 parameters received gradients")

    checkpoint_path = checkpoint_dir / "adaptive-shared-decoupling-smoke.ckpt"
    torch.save(
        {
            "epoch": 0,
            "global_step": 2,
            "pytorch-lightning_version": pl.__version__,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    restored = GenerateReturn.load_from_checkpoint(
        checkpoint_path, config=config, T_max=2, strict=True
    )
    restored.freeze_vqvae()
    restored.eval()
    model.eval()
    with torch.no_grad():
        reference = model(train_batch[:, :, :158], train_batch[:, -1, 158:171])
        reloaded = restored(train_batch[:, :, :158], train_batch[:, -1, 158:171])
    checkpoint_exact = all(
        torch.equal(expected, actual)
        for expected, actual in zip(reference, reloaded)
    )
    if not checkpoint_exact:
        raise AssertionError("Strict checkpoint round-trip changed standard output")

    inference_results = {}
    for split, split_seed, start_date in (
        ("valid", 20, "2021-01-04"),
        ("test", 30, "2023-01-03"),
    ):
        dataset = SyntheticCanonicalDataset(split_seed, start_date)
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        prediction, _, metrics = run_inference(
            restored, loader, config, device="cpu"
        )
        prediction_path = result_dir / f"0_{split}.pkl"
        metric_path = result_dir / f"0_{split}_metric.csv"
        prediction.to_pickle(prediction_path)
        pd.DataFrame([metrics], index=["values"]).transpose().to_csv(metric_path)
        normalized, signal = _normalize_prediction_frame(prediction_path)
        if len(normalized) != len(dataset) or len(signal) != len(dataset):
            raise AssertionError(f"{split} output is incompatible with backtest")
        inference_results[split] = {
            "rows": len(prediction),
            "prediction": str(prediction_path.relative_to(ROOT)),
            "metric": str(metric_path.relative_to(ROOT)),
            "metrics": {key: float(value) for key, value in metrics.items()},
        }

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
            "source_commit": marker["commit"],
            "checkpoint": str(stage1_checkpoint.relative_to(ROOT)),
            "checkpoint_bytes": stage1_checkpoint.stat().st_size,
            "strict_load": {
                "encoder": {"missing": 0, "unexpected": 0},
                "quantizer": {"missing": 0, "unexpected": 0},
                "revin": {"missing": 0, "unexpected": 0},
            },
            "single_vq512": True,
            "data_splits": EXPECTED_SPLITS,
            "frozen_parameters_have_grad": frozen_stage1_has_grad,
        },
        "adaptive_shared_fusion": {
            "formula": "1 + 0.5 * tanh(Linear(z_q))",
            "zero_initialized": fusion_zero_init,
            "raw_zq_conditioning": True,
            "prediction_bitwise_equal_to_016": prediction_equivalence,
            "original_aux_bitwise_equal_to_016": prediction_equivalence,
            "existing_parameter_initialization_equal_to_016": parameter_initialization_equal,
        },
        "decoupling": {
            "formula": "mean(cosine_similarity(raw_shared, routed)^2)",
            "lambda": model.decoupling_lambda,
            "value": float(decoupling_loss.detach()),
            "finite_nonnegative": True,
            "identical_penalty": float(identical.detach()),
            "orthogonal_penalty": float(orthogonal.detach()),
            "shared_gradient_l1": shared_dec_grad_l1,
            "routed_gradient_l1": routed_dec_grad_l1,
            "independent_top_level_term": True,
            "training_validation_objective_shared": True,
        },
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "strict_round_trip": True,
            "standard_output_bitwise_equal": checkpoint_exact,
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
