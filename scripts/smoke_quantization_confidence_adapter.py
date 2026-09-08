"""Minimal synthetic smoke for experiment 019.

Validates the external corrected PRISM-VQ Stage 1 checkpoint, exact initial
equivalence to the baseline prediction forward, detached quantization error,
adapter optimization, strict Stage 2 checkpoint round-trip, and standard
prediction output compatibility.
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
from module.quantise import VectorQuantiser
from trainer.train_ypred import GenerateReturn
from utils import run_inference, seed_everything


ARTIFACT_ROOT = ROOT / "artifacts" / "019" / "smoke"
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


def build_config(adapter=True):
    if not OmegaConf.has_resolver("half"):
        OmegaConf.register_new_resolver("half", lambda value: int(value) // 2)
    config = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs" / "config.yaml"), resolve=True
    )
    config["predictor"]["saved_model"] = str(STAGE1_CHECKPOINT)
    config["predictor"]["quantization_confidence_adapter"] = adapter
    return config


def build_model(adapter):
    seed_everything(0)
    pl.seed_everything(0, workers=True)
    return GenerateReturn(copy.deepcopy(build_config(adapter)), T_max=2)


def train_loss(model, batch):
    feature = batch[:, :, :158]
    prior = batch[:, -1, 158:171]
    label = batch[:, -1, 238]
    y_pred, _, _, _, aux_loss = model(feature, prior)
    return model.rank_loss(y_pred, label) + model.aux_weight * aux_loss


def main():
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = ARTIFACT_ROOT / "checkpoints"
    result_dir = ARTIFACT_ROOT / "res" / "quantization_confidence_smoke"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    if not STAGE1_CHECKPOINT.is_file() or STAGE1_CHECKPOINT.stat().st_size == 0:
        raise FileNotFoundError(f"External Stage 1 checkpoint missing: {STAGE1_CHECKPOINT}")
    checkpoint_bytes = STAGE1_CHECKPOINT.stat().st_size
    checkpoint_md5 = file_md5(STAGE1_CHECKPOINT)
    if checkpoint_bytes != EXPECTED_CHECKPOINT_BYTES:
        raise AssertionError("External Stage 1 checkpoint size changed")
    if checkpoint_md5 != EXPECTED_CHECKPOINT_MD5:
        raise AssertionError("External Stage 1 checkpoint hash changed")

    config = build_config(adapter=True)
    if config["predictor"]["quantization_confidence_adapter"] is not True:
        raise AssertionError("Default config must enable experiment 019")
    if config["vqvae"]["vq_embed_dim"] != 128:
        raise AssertionError("Stage 1 latent dimension must remain 128")
    if config["vqvae"]["num_embed"] != 512:
        raise AssertionError("Stage 1 must retain VQ512")
    for key, expected in EXPECTED_SPLITS.items():
        if config["data"][key] != expected:
            raise AssertionError(f"Data split changed for {key}")

    # Constructors strict-load Encoder, Quantizer, and RevIN from the exact
    # external checkpoint.  Re-seeding proves the late-appended adapter does
    # not perturb any baseline parameter initialization.
    base = build_model(adapter=False).eval()
    model = build_model(adapter=True).eval()
    if not isinstance(model.quantizer, VectorQuantiser):
        raise AssertionError("Stage 1 quantizer is not VectorQuantiser")
    quantizers = [module for module in model.modules() if isinstance(module, VectorQuantiser)]
    if len(quantizers) != 1:
        raise AssertionError("Stage 1 must contain exactly one quantizer")
    if tuple(model.quantizer.embedding.weight.shape) != (512, 128):
        raise AssertionError("Stage 1 codebook is not VQ512 x 128")

    adapted_base_state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("quantization_confidence_adapter.")
    }
    if base.state_dict().keys() != adapted_base_state.keys():
        raise AssertionError("Adapter changed baseline state_dict keys")
    existing_init_exact = all(
        torch.equal(value, adapted_base_state[key])
        for key, value in base.state_dict().items()
    )
    if not existing_init_exact:
        raise AssertionError("Adapter perturbed a baseline parameter initialization")

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

    with torch.no_grad():
        feature_normalized = model.revin(feature, mode="norm")
        h_batch = model.encoder(feature_normalized)
        z_q = model.quantizer(h_batch)[0].detach()
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
        raise AssertionError("Zero-init z_stage2 is not bitwise equal to z_q")
    if not initial_forward_exact:
        raise AssertionError("Initial prediction forward is not bitwise equal to main")

    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["train"]["learning_rate"],
        weight_decay=1e-5,
    )
    adapter_weight_before = adapter.weight.detach().clone()
    adapter_bias_before = adapter.bias.detach().clone()
    optimizer.zero_grad()
    loss = train_loss(model, train_batch)
    loss.backward()
    adapter_weight_grad_l1 = float(adapter.weight.grad.abs().sum())
    adapter_bias_grad_l1 = float(adapter.bias.grad.abs().sum())
    if adapter_weight_grad_l1 <= 0 or adapter_bias_grad_l1 <= 0:
        raise AssertionError("Quantization-confidence adapter did not receive gradient")
    frozen_stage1_has_grad = any(
        parameter.grad is not None
        for module in (model.encoder, model.quantizer, model.revin)
        for parameter in module.parameters()
    )
    if frozen_stage1_has_grad:
        raise AssertionError("Frozen Stage 1 received gradients")
    optimizer.step()
    adapter_updated = bool(
        not torch.equal(adapter_weight_before, adapter.weight.detach())
        and not torch.equal(adapter_bias_before, adapter.bias.detach())
    )
    if not adapter_updated:
        raise AssertionError("Quantization-confidence adapter did not update")

    checkpoint_path = checkpoint_dir / "quantization-confidence-smoke.ckpt"
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
    model.eval()
    with torch.no_grad():
        reference = model(feature, prior)
        reloaded = restored(feature, prior)
    checkpoint_exact = all(
        torch.equal(reference_value, reloaded_value)
        for reference_value, reloaded_value in zip(reference, reloaded)
    )
    if not checkpoint_exact:
        raise AssertionError("Strict checkpoint round-trip changed model output")

    test_data = SyntheticCanonicalDataset(30, "2023-01-03")
    test_loader = DataLoader(test_data, batch_size=6, shuffle=False)
    prediction, _, metrics = run_inference(restored, test_loader, config, device="cpu")
    prediction_path = result_dir / "0_best.pkl"
    metric_path = result_dir / "0_metric.csv"
    prediction.to_pickle(prediction_path)
    pd.DataFrame([metrics], index=["values"]).transpose().to_csv(metric_path)
    normalized, signal = _normalize_prediction_frame(prediction_path)
    if len(normalized) != len(test_data) or len(signal) != len(test_data):
        raise AssertionError("Prediction output is incompatible with backtest normalizer")

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
        "quantization_confidence_adapter": {
            "module": "Linear(1, 128)",
            "zero_initialized": adapter_zero_init,
            "q_error_formula_exact": q_error_exact,
            "q_error_mean": float(q_error.mean()),
            "q_error_min": float(q_error.min()),
            "q_error_max": float(q_error.max()),
            "q_error_requires_grad": q_error.requires_grad,
            "initial_z_stage2_bitwise_equal_z_q": initial_latent_exact,
            "initial_prediction_forward_bitwise_equal_main": initial_forward_exact,
            "existing_parameter_initialization_bitwise_equal_main": existing_init_exact,
            "weight_grad_l1": adapter_weight_grad_l1,
            "bias_grad_l1": adapter_bias_grad_l1,
            "updated": adapter_updated,
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
