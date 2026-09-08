"""Minimal synthetic smoke for experiment 021.

Uses the exact external corrected PRISM-VQ Stage 1 checkpoint.  It performs
one synthetic Stage 2 optimizer step and standard inference/output checks;
it does not launch formal training or a portfolio backtest.
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
from dataset.schema import TOTAL_DIM, unpack_batch
from module.quantise import VectorQuantiser
from trainer.train_ypred import GenerateReturn
from utils import run_inference, seed_everything


ARTIFACT_ROOT = ROOT / "artifacts" / "021" / "smoke"
STAGE1_CHECKPOINT = (
    ROOT
    / "artifacts"
    / "baseline"
    / "run"
    / "checkpoints"
    / "infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt"
)
EXPECTED_CHECKPOINT_BYTES = 14584929
EXPECTED_CHECKPOINT_MD5 = "6b9d9dbfd938c7bd2c7dc5ee33cb38af"
EXPECTED_SPLITS = {
    "train_period": ["2009-01-01", "2020-12-31"],
    "valid_period": ["2021-01-01", "2022-12-31"],
    "test_period": ["2023-01-01", "2025-12-31"],
}


class SyntheticCanonicalDataset(Dataset):
    def __init__(self, seed, start_date, days=2, stocks_per_day=6):
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


def build_config(allocation=True):
    if not OmegaConf.has_resolver("half"):
        OmegaConf.register_new_resolver("half", lambda value: int(value) // 2)
    config = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs" / "config.yaml"), resolve=True
    )
    config["predictor"]["saved_model"] = str(STAGE1_CHECKPOINT)
    config["predictor"]["latent_conditioned_allocation"] = allocation
    return config


def build_model(allocation):
    seed_everything(0)
    pl.seed_everything(0, workers=True)
    return GenerateReturn(copy.deepcopy(build_config(allocation)), T_max=2)


def main():
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = ARTIFACT_ROOT / "checkpoints"
    result_dir = ARTIFACT_ROOT / "res" / "latent_prior_allocation_smoke"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    if not STAGE1_CHECKPOINT.is_file() or STAGE1_CHECKPOINT.stat().st_size == 0:
        raise FileNotFoundError(
            f"External Stage 1 checkpoint missing: {STAGE1_CHECKPOINT}"
        )
    checkpoint_bytes = STAGE1_CHECKPOINT.stat().st_size
    checkpoint_md5 = file_md5(STAGE1_CHECKPOINT)
    if checkpoint_bytes != EXPECTED_CHECKPOINT_BYTES:
        raise AssertionError("External Stage 1 checkpoint size changed")
    if checkpoint_md5 != EXPECTED_CHECKPOINT_MD5:
        raise AssertionError("External Stage 1 checkpoint hash changed")

    config = build_config(True)
    if config["predictor"]["latent_conditioned_allocation"] is not True:
        raise AssertionError("Default config must enable experiment 021")
    if config["vqvae"]["num_embed"] != 512:
        raise AssertionError("Stage 1 must remain single VQ512")
    if config["vqvae"]["vq_embed_dim"] != 128:
        raise AssertionError("Stage 1 latent dimension must remain 128")
    for key, expected in EXPECTED_SPLITS.items():
        if config["data"][key] != expected:
            raise AssertionError(f"Data split changed for {key}")

    # Both constructors strict-load Encoder, Quantizer, and RevIN from the
    # external checkpoint.  Re-seeding isolates the sole experiment variable.
    base = build_model(False).eval()
    model = build_model(True).eval()
    quantizers = [
        module for module in model.modules()
        if isinstance(module, VectorQuantiser)
    ]
    if len(quantizers) != 1:
        raise AssertionError("Stage 1 must contain exactly one quantizer")
    if tuple(model.quantizer.embedding.weight.shape) != (512, 128):
        raise AssertionError("Stage 1 codebook is not VQ512 x 128")

    gate_prefix = "return_predictor.allocation_gate."
    allocated_base_state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith(gate_prefix)
    }
    if base.state_dict().keys() != allocated_base_state.keys():
        raise AssertionError("Allocation gate changed baseline state keys")
    existing_init_exact = all(
        torch.equal(value, allocated_base_state[key])
        for key, value in base.state_dict().items()
    )
    if not existing_init_exact:
        raise AssertionError("Allocation gate perturbed baseline initialization")

    gate = model.return_predictor.allocation_gate
    gate_zero_init = bool(
        gate.in_features == 128
        and gate.out_features == 1
        and torch.count_nonzero(gate.weight) == 0
        and torch.count_nonzero(gate.bias) == 0
    )
    if not gate_zero_init:
        raise AssertionError("Gate must be zero-initialized Linear(128, 1)")

    train_data = SyntheticCanonicalDataset(10, "2020-01-02", days=1)
    train_batch = train_data.values
    parts = unpack_batch(train_batch)
    feature, prior = parts.stock_feature, parts.prior_factor

    seen_gate_inputs = []
    handle = gate.register_forward_pre_hook(
        lambda _module, inputs: seen_gate_inputs.append(inputs[0].detach().clone())
    )
    with torch.no_grad():
        base_output = base(feature, prior)
        allocated_output = model(feature, prior)
    handle.remove()
    initial_forward_exact = all(
        torch.equal(base_value, allocated_value)
        for base_value, allocated_value in zip(base_output, allocated_output)
    )
    if not initial_forward_exact:
        raise AssertionError("Zero-init prediction forward differs from main")
    if len(seen_gate_inputs) != 1 or not torch.equal(
        seen_gate_inputs[0], allocated_output[3]
    ):
        raise AssertionError("Allocation gate did not receive raw detached z_q")

    prior_scale, latent_scale = model.return_predictor.allocation_scales(
        allocated_output[3]
    )
    zero_scales_exact = bool(
        torch.equal(prior_scale, torch.ones_like(prior_scale))
        and torch.equal(latent_scale, torch.ones_like(latent_scale))
    )
    if not zero_scales_exact:
        raise AssertionError("Zero-init scales are not exact ones")

    probe = copy.deepcopy(model.return_predictor).eval()
    direction = torch.linspace(-0.1, 0.1, 128)
    with torch.no_grad():
        probe.allocation_gate.weight.copy_(direction.unsqueeze(0))
        probe.allocation_gate.bias.zero_()
    z_positive = direction.unsqueeze(0)
    z_negative = -z_positive
    probe_z = torch.cat([z_positive, z_negative], dim=0)
    probe_prior_scale, probe_latent_scale = probe.allocation_scales(probe_z)
    bounded = bool(
        torch.all(probe_prior_scale > 0.5)
        and torch.all(probe_prior_scale < 1.5)
        and torch.all(probe_latent_scale > 0.5)
        and torch.all(probe_latent_scale < 1.5)
    )
    complementary = torch.equal(
        probe_prior_scale + probe_latent_scale,
        torch.full_like(probe_prior_scale, 2.0),
    )
    different_allocation = bool(
        probe_prior_scale[0] > probe_prior_scale[1]
        and probe_latent_scale[0] < probe_latent_scale[1]
    )
    if not bounded or not complementary or not different_allocation:
        raise AssertionError("Learned allocation constraints were not satisfied")

    beta_p = torch.full((2, 13), 2.0 / 13)
    f_prior = torch.full((2, 13), 2.0)
    beta_l = torch.full((2, 128), 1.0 / 128)
    f_latent = torch.full((2, 128), 3.0)
    prior_term = (beta_p * f_prior).sum(dim=1)
    latent_term = (beta_l * f_latent).sum(dim=1)
    probe_output = probe(
        torch.zeros(2), beta_p, beta_l, f_prior, f_latent, z_q=probe_z
    )
    contribution_direction = bool(
        (probe_prior_scale * prior_term)[0]
        > (probe_prior_scale * prior_term)[1]
        and (probe_latent_scale * latent_term)[0]
        < (probe_latent_scale * latent_term)[1]
        and torch.equal(
            probe_output,
            probe_prior_scale * prior_term + probe_latent_scale * latent_term,
        )
    )
    if not contribution_direction:
        raise AssertionError("Complementary factor contribution direction failed")

    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["train"]["learning_rate"],
        weight_decay=1e-5,
    )
    gate_weight_before = gate.weight.detach().clone()
    gate_bias_before = gate.bias.detach().clone()
    optimizer.zero_grad()
    y_pred, _, _, _, aux_loss = model(feature, prior)
    loss = (
        model.rank_loss(y_pred, parts.target(5))
        + model.aux_weight * aux_loss
    )
    loss.backward()
    gate_weight_grad_l1 = float(gate.weight.grad.abs().sum())
    gate_bias_grad_l1 = float(gate.bias.grad.abs().sum())
    if gate_weight_grad_l1 <= 0 or gate_bias_grad_l1 <= 0:
        raise AssertionError("Allocation gate did not receive nonzero gradient")
    if not torch.isfinite(gate.weight.grad).all() or not torch.isfinite(
        gate.bias.grad
    ).all():
        raise AssertionError("Allocation gate gradient is not finite")
    frozen_stage1_has_grad = any(
        parameter.grad is not None
        for module in (model.encoder, model.quantizer, model.revin)
        for parameter in module.parameters()
    )
    if frozen_stage1_has_grad:
        raise AssertionError("Frozen Stage 1 received gradients")
    if any(
        module.training for module in (model.encoder, model.quantizer, model.revin)
    ):
        raise AssertionError("Frozen Stage 1 left eval mode")
    optimizer.step()
    gate_updated = bool(
        not torch.equal(gate_weight_before, gate.weight.detach())
        and not torch.equal(gate_bias_before, gate.bias.detach())
    )
    if not gate_updated:
        raise AssertionError("Allocation gate did not update")

    checkpoint_path = checkpoint_dir / "latent-prior-allocation-smoke.ckpt"
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
    ).eval()
    model.eval()
    with torch.no_grad():
        reference = model(feature, prior)
        reloaded = restored(feature, prior)
    checkpoint_exact = all(
        torch.equal(reference_value, reloaded_value)
        for reference_value, reloaded_value in zip(reference, reloaded)
    )
    if not checkpoint_exact:
        raise AssertionError("Strict checkpoint round-trip changed output")

    test_data = SyntheticCanonicalDataset(30, "2023-01-03")
    test_loader = DataLoader(test_data, batch_size=6, shuffle=False)
    prediction, _, metrics = run_inference(
        restored, test_loader, config, device="cpu"
    )
    prediction_path = result_dir / "0_best.pkl"
    metric_path = result_dir / "0_metric.csv"
    prediction.to_pickle(prediction_path)
    pd.DataFrame([metrics], index=["values"]).transpose().to_csv(metric_path)
    normalized, signal = _normalize_prediction_frame(prediction_path)
    if len(normalized) != len(test_data) or len(signal) != len(test_data):
        raise AssertionError("Prediction is incompatible with backtest normalizer")

    report = {
        "status": "PASS",
        "stage1": {
            "source": "external",
            "checkpoint": str(STAGE1_CHECKPOINT.relative_to(ROOT)),
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
        "latent_prior_allocation": {
            "module": "Linear(128, 1)",
            "delta": 0.5,
            "zero_initialized": gate_zero_init,
            "zero_scales_exact_ones": zero_scales_exact,
            "full_prediction_forward_bitwise_equal_main": initial_forward_exact,
            "existing_parameter_initialization_bitwise_equal_main": existing_init_exact,
            "gate_received_raw_z_q": True,
            "scales_strictly_bounded": bounded,
            "scales_sum_exactly_two": complementary,
            "nonzero_gate_changes_allocation_by_z_q": different_allocation,
            "complementary_contribution_direction": contribution_direction,
            "weight_grad_l1": gate_weight_grad_l1,
            "bias_grad_l1": gate_bias_grad_l1,
            "updated": gate_updated,
        },
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "strict_round_trip": True,
            "output_bitwise_equal": checkpoint_exact,
        },
        "standard_outputs": {
            "prediction": str(prediction_path.relative_to(ROOT)),
            "metric": str(metric_path.relative_to(ROOT)),
            "rows": len(prediction),
            "metrics": {key: float(value) for key, value in metrics.items()},
            "backtest_normalizer": "PASS",
        },
    }
    report_path = ARTIFACT_ROOT / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
